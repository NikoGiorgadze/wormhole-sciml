"""Preprocessing, training, and validation utilities for the finite-time flow model.

This module is intentionally separate from recursive Model-A rollout logic.  It
reuses the established MLP, initialization, hashing, and batched-inference
utilities, but models one direct map (x0, xi0, E0, s) -> (Delta_x, Delta_xi).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dynamics import conserved_energy, timelike_margin
from .model_a import (
    ModelA,
    Normalization,
    load_trained_model,
    parameter_count,
    predict_standardized,
    state_dict_sha256,
)
from .physics_gate import experiment_parameters, state_from_xi


INPUT_COLUMNS = ("x0", "xi0", "E0", "s")
TARGET_COLUMNS = ("Delta_x", "Delta_xi")
TRAINING_SEEDS = (101, 202, 303)
HIDDEN_DIMENSIONS = (64, 64)
LEARNING_RATE = 1.0e-3
BATCH_SIZE = 512
MAXIMUM_EPOCHS = 1500
PATIENCE = 40
EXPECTED_PARAMETER_COUNT = 4610
PREPROCESSING_SCHEMA = "wormhole_finite_time_componentwise_standardization/v1"
PREPROCESSING_IMPLEMENTATION = "wormhole_sciml.finite_time/v1"


@dataclass(frozen=True, slots=True)
class FiniteTimePreprocessing:
    """Frozen ordered training-only componentwise mean/std constants."""

    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    source_training_sha256: str

    @classmethod
    def fit(
        cls,
        training: dict[str, np.ndarray],
        source_training_sha256: str,
    ) -> "FiniteTimePreprocessing":
        inputs = stack_columns(training, INPUT_COLUMNS)
        targets = stack_columns(training, TARGET_COLUMNS)
        input_mean = np.mean(inputs, axis=0, dtype=np.float64)
        input_std = np.std(inputs, axis=0, ddof=0, dtype=np.float64)
        target_mean = np.mean(targets, axis=0, dtype=np.float64)
        target_std = np.std(targets, axis=0, ddof=0, dtype=np.float64)
        all_std = np.concatenate((input_std, target_std))
        if not np.all(np.isfinite(all_std)) or np.any(all_std <= 0.0):
            raise ValueError("all six training-only standard deviations must be finite and positive")
        return cls(input_mean, input_std, target_mean, target_std, source_training_sha256)

    @classmethod
    def from_json(cls, path: Path) -> "FiniteTimePreprocessing":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema"] != PREPROCESSING_SCHEMA:
            raise ValueError("unsupported finite-time preprocessing schema")
        if tuple(payload["input_order"]) != INPUT_COLUMNS:
            raise ValueError("finite-time input ordering mismatch")
        if tuple(payload["target_order"]) != TARGET_COLUMNS:
            raise ValueError("finite-time target ordering mismatch")
        columns = payload["columns"]
        return cls(
            np.asarray([columns[name]["mean"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["mean"] for name in TARGET_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in TARGET_COLUMNS], dtype=np.float64),
            str(payload["source_training_dataset_sha256"]),
        )

    def as_normalization(self) -> Normalization:
        """Return the established reusable normalization representation."""

        return Normalization(
            self.input_mean.copy(), self.input_std.copy(),
            self.target_mean.copy(), self.target_std.copy(),
        )

    def standardize_inputs(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().standardize_inputs(values)

    def standardize_targets(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().standardize_targets(values)

    def unstandardize_targets(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().unstandardize_targets(values)

    def payload(self, source_path: Path, source_rows: int) -> dict[str, Any]:
        names = INPUT_COLUMNS + TARGET_COLUMNS
        means = np.concatenate((self.input_mean, self.target_mean))
        stds = np.concatenate((self.input_std, self.target_std))
        return {
            "schema": PREPROCESSING_SCHEMA,
            "implementation": PREPROCESSING_IMPLEMENTATION,
            "source_split": "train only",
            "source_training_dataset": str(source_path.resolve()),
            "source_training_dataset_sha256": self.source_training_sha256,
            "source_training_row_count": int(source_rows),
            "input_order": list(INPUT_COLUMNS),
            "target_order": list(TARGET_COLUMNS),
            "physical_time_feature": "s",
            "horizon_fraction_used_as_feature": False,
            "statistics_dtype": "float64",
            "model_tensor_dtype": "float32",
            "standard_deviation_definition": "population (ddof=0)",
            "transformation": "componentwise (value - mean) / standard_deviation only",
            "columns": {
                name: {"mean": float(mean), "standard_deviation": float(std)}
                for name, mean, std in zip(names, means, stds)
            },
        }


def stack_columns(data: dict[str, np.ndarray], columns: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in columns if name not in data]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    return np.column_stack(tuple(np.asarray(data[name], dtype=np.float64) for name in columns))


def standardized_range_audit(
    training: dict[str, np.ndarray], preprocessing: FiniteTimePreprocessing
) -> dict[str, Any]:
    standardized = np.column_stack((
        preprocessing.standardize_inputs(stack_columns(training, INPUT_COLUMNS)),
        preprocessing.standardize_targets(stack_columns(training, TARGET_COLUMNS)),
    )).astype(np.float64)
    quantiles = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
    names = INPUT_COLUMNS + TARGET_COLUMNS
    columns: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        values = standardized[:, index]
        q = np.quantile(values, quantiles)
        columns[name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "p0.1": float(q[0]),
            "p1": float(q[1]),
            "p5": float(q[2]),
            "median": float(q[3]),
            "p95": float(q[4]),
            "p99": float(q[5]),
            "p99.9": float(q[6]),
        }
    maximum_absolute = float(np.max(np.abs(standardized)))
    finite = bool(np.all(np.isfinite(standardized)))
    # A deliberately broad, predeclared stop gate: values beyond 25 sigma are
    # an obvious numerical warning for this baseline two-tanh experiment.
    pathology = (not finite) or maximum_absolute > 25.0
    return {
        "source_split": "train only",
        "row_count": int(standardized.shape[0]),
        "columns": columns,
        "all_finite": finite,
        "maximum_absolute_standardized_value": maximum_absolute,
        "predeclared_pathology_threshold_absolute_z": 25.0,
        "obvious_numerical_pathology": pathology,
        "decision": "FAIL_STOP_BEFORE_TRAINING" if pathology else "PASS_FREEZE_PREPROCESSING",
    }


def tensor_dataset(
    data: dict[str, np.ndarray], preprocessing: FiniteTimePreprocessing
) -> TensorDataset:
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, INPUT_COLUMNS)))
    targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(data, TARGET_COLUMNS)))
    return TensorDataset(inputs, targets)


def set_deterministic_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed,
        "numpy_legacy_seed": seed,
        "torch_manual_seed": seed,
        "torch_deterministic_algorithms": True,
        "torch_num_threads": 1,
        "dataloader_num_workers": 0,
    }


def orbit_averaged_standardized_mse(
    prediction: np.ndarray,
    target: np.ndarray,
    orbit_ids: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    orbit_ids = np.asarray(orbit_ids)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("prediction and target must have matching shape (rows, 2)")
    if orbit_ids.shape != (prediction.shape[0],):
        raise ValueError("orbit_ids must contain one value per row")
    unique, inverse, counts = np.unique(orbit_ids, return_inverse=True, return_counts=True)
    row_loss = np.mean((prediction - target) ** 2, axis=1)
    orbit_sums = np.bincount(inverse, weights=row_loss, minlength=unique.size)
    orbit_loss = orbit_sums / counts
    orbit_average = float(np.mean(orbit_loss))
    row_average = float(np.mean(row_loss))
    return {
        "orbit_averaged_standardized_mse": orbit_average,
        "ordinary_row_averaged_standardized_mse": row_average,
        "absolute_difference": abs(orbit_average - row_average),
        "orbit_count": int(unique.size),
        "rows_per_orbit_minimum": int(np.min(counts)),
        "rows_per_orbit_maximum": int(np.max(counts)),
    }


@torch.no_grad()
def validation_metric(
    model: ModelA,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    orbit_ids: np.ndarray,
) -> dict[str, float | int]:
    prediction = predict_standardized(model, inputs, batch_size=8192).cpu().numpy()
    return orbit_averaged_standardized_mse(prediction, targets.cpu().numpy(), orbit_ids)


def train_seed(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    preprocessing: FiniteTimePreprocessing,
    seed: int,
    output_dir: Path,
    *,
    progress: bool = True,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported finite-time seed {seed}")
    if maximum_epochs < 1 or patience < 1:
        raise ValueError("maximum_epochs and patience must be positive")
    reproducibility = set_deterministic_seed(seed)
    train_data = tensor_dataset(training, preprocessing)
    validation_data = tensor_dataset(validation, preprocessing)
    validation_inputs, validation_targets = validation_data.tensors
    model = ModelA(4, HIDDEN_DIMENSIONS).to(dtype=torch.float32)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("4->64->64->2 parameter count mismatch")
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, float | int]] = []
    steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))

    for epoch in range(1, maximum_epochs + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 10_000_000 + epoch * 10 + 1)
        loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            drop_last=False,
        )
        model.train()
        squared_sum = 0.0
        element_count = 0
        for batch_inputs, batch_targets in loader:
            prediction = model(batch_inputs)
            loss = torch.mean((prediction - batch_targets) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            errors = prediction.detach() - batch_targets
            squared_sum += float(torch.sum(errors.double() ** 2).item())
            element_count += int(errors.numel())
        training_loss = squared_sum / element_count
        measured = validation_metric(
            model, validation_inputs, validation_targets, validation["orbit_id"]
        )
        current = float(measured["orbit_averaged_standardized_mse"])
        history.append({
            "epoch": epoch,
            "training_standardized_mse": training_loss,
            "validation_orbit_averaged_standardized_mse": current,
            "validation_row_averaged_standardized_mse": float(
                measured["ordinary_row_averaged_standardized_mse"]
            ),
        })
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            print(
                f"finite-time seed={seed} epoch={epoch} val={current:.8g} best={best_metric:.8g}",
                flush=True,
            )
        if without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("finite-time training produced no checkpoint")
    stopping_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)
    restored = validation_metric(
        model, validation_inputs, validation_targets, validation["orbit_id"]
    )
    restored_value = float(restored["orbit_averaged_standardized_mse"])
    if restored_value != best_metric:
        raise RuntimeError("restored finite-time checkpoint does not reproduce best metric")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint_path)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "architecture": "4->64->64->2 with two tanh hidden layers and no output activation",
        "parameter_count": parameter_count(model),
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "dtype": "float32",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": 0.0,
        "scheduler": None,
        "loss": "equal two-component standardized MSE",
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "checkpoint_selection_metric": "validation orbit-averaged standardized MSE only",
        "best_epoch": best_epoch,
        "stopping_epoch": stopping_epoch,
        "best_validation_orbit_averaged_standardized_mse": best_metric,
        "final_training_standardized_mse": float(history[-1]["training_standardized_mse"]),
        "final_validation_orbit_averaged_standardized_mse": float(
            history[-1]["validation_orbit_averaged_standardized_mse"]
        ),
        "early_stopping_triggered": stopping_epoch < maximum_epochs,
        "ceiling_reached": stopping_epoch == maximum_epochs,
        "best_epoch_at_ceiling": best_epoch == maximum_epochs,
        "batches_per_epoch": steps_per_epoch,
        "total_optimizer_updates": steps_per_epoch * stopping_epoch,
        "restored_validation": restored,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": str(history_path.resolve()),
        "reproducibility": reproducibility,
        "sealed_test_predictions_computed": False,
    }


def absolute_error_summary(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    q = np.quantile(absolute, (0.5, 0.9, 0.95, 0.99))
    return {
        "median": float(q[0]), "p90": float(q[1]), "p95": float(q[2]),
        "p99": float(q[3]), "maximum": float(np.max(absolute)),
    }


def component_metrics(error: np.ndarray) -> dict[str, float | dict[str, float]]:
    error = np.asarray(error, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "absolute_error": absolute_error_summary(error),
    }


def two_component_metrics(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        TARGET_COLUMNS[index]: component_metrics(errors[:, index])
        for index in range(2)
    }


def breakdown_metrics(
    errors: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        result[name] = {
            "row_count": int(np.sum(mask)),
            "metrics": None if not np.any(mask) else two_component_metrics(errors[mask]),
        }
    return result


@torch.no_grad()
def evaluate_validation(
    checkpoint: Path,
    validation: dict[str, np.ndarray],
    preprocessing: FiniteTimePreprocessing,
) -> dict[str, Any]:
    model = load_trained_model(checkpoint)
    standardized_inputs = torch.from_numpy(
        preprocessing.standardize_inputs(stack_columns(validation, INPUT_COLUMNS))
    )
    standardized_targets = preprocessing.standardize_targets(
        stack_columns(validation, TARGET_COLUMNS)
    )
    standardized_prediction = predict_standardized(
        model, standardized_inputs, batch_size=8192
    ).cpu().numpy()
    prediction = preprocessing.unstandardize_targets(standardized_prediction)
    exact = stack_columns(validation, TARGET_COLUMNS)
    residual_errors = prediction - exact
    reconstructed = np.column_stack((validation["x0"], validation["xi0"])) + prediction
    exact_state = np.column_stack((validation["x1"], validation["xi1"]))
    state_errors = reconstructed - exact_state
    equality_difference = float(np.max(np.abs(residual_errors - state_errors)))
    standardized = orbit_averaged_standardized_mse(
        standardized_prediction, standardized_targets, validation["orbit_id"]
    )

    time_masks = {
        "short_s_le_5": validation["s"] <= 5.0,
        "intermediate_5_lt_s_le_20": (validation["s"] > 5.0) & (validation["s"] <= 20.0),
        "long_s_gt_20": validation["s"] > 20.0,
    }
    family_masks = {
        "hard_u_th_le_0p30": validation["u_th"] <= 0.30,
        "ordinary_u_th_gt_0p30": validation["u_th"] > 0.30,
    }
    group_masks = {
        "global": validation["sample_group"] == "global64",
        "hard_targeted": validation["sample_group"] == "hard_targeted32",
        "ordinary_additional_global": validation["sample_group"] == "ordinary_global32",
    }
    subtype_masks = {
        name: validation["sample_subtype"] == name
        for name in (
            "within_sensitive", "sensitive_exit", "sensitive_to_central",
            "sensitive_to_long_outgoing",
        )
    }
    identity = np.asarray(validation["s"] == 0.0)
    identity_errors = residual_errors[identity]
    identity_diagnostics = {
        "row_count": int(np.sum(identity)),
        "max_abs_Delta_x_prediction": float(np.max(np.abs(prediction[identity, 0]))),
        "max_abs_Delta_xi_prediction": float(np.max(np.abs(prediction[identity, 1]))),
        "rmse_Delta_x": float(np.sqrt(np.mean(identity_errors[:, 0] ** 2))),
        "rmse_Delta_xi": float(np.sqrt(np.mean(identity_errors[:, 1] ** 2))),
    }

    wormhole, spiral = experiment_parameters()
    x_hat, xi_hat = reconstructed[:, 0], reconstructed[:, 1]
    _, u_hat = state_from_xi(x_hat, xi_hat, wormhole, spiral)
    margin = timelike_margin(x_hat, u_hat, wormhole, spiral)
    xi_invalid = np.abs(xi_hat) >= 1.0
    margin_invalid = margin <= 0.0
    valid_energy = np.isfinite(margin) & (margin > 0.0)
    energy_error = np.full(x_hat.shape, np.nan, dtype=np.float64)
    if np.any(valid_energy):
        energy_error[valid_energy] = (
            conserved_energy(x_hat[valid_energy], u_hat[valid_energy], wormhole, spiral)
            - validation["E0"][valid_energy]
        )
    finite_energy_error = energy_error[np.isfinite(energy_error)]
    energy_stats = {
        "finite_count": int(finite_energy_error.size),
        "rmse": float(np.sqrt(np.mean(finite_energy_error**2))),
        "mae": float(np.mean(np.abs(finite_energy_error))),
        "median_absolute": float(np.median(np.abs(finite_energy_error))),
        "p90_absolute": float(np.quantile(np.abs(finite_energy_error), 0.90)),
        "p95_absolute": float(np.quantile(np.abs(finite_energy_error), 0.95)),
        "p99_absolute": float(np.quantile(np.abs(finite_energy_error), 0.99)),
        "maximum_absolute": float(np.max(np.abs(finite_energy_error))),
    }
    physical_diagnostics = {
        "row_count": int(x_hat.size),
        "absolute_xi_ge_1": {
            "count": int(np.sum(xi_invalid)),
            "fraction": float(np.mean(xi_invalid)),
        },
        "C_le_0": {
            "count": int(np.sum(margin_invalid)),
            "fraction": float(np.mean(margin_invalid)),
        },
        "energy_error_E_hat_minus_E0": energy_stats,
        "used_in_loss": False,
    }

    return {
        "standardized_validation": standardized,
        "physical_residual_metrics": two_component_metrics(residual_errors),
        "physical_reconstructed_state_metrics": {
            "x1": component_metrics(state_errors[:, 0]),
            "xi1": component_metrics(state_errors[:, 1]),
        },
        "maximum_absolute_residual_vs_state_error_difference": equality_difference,
        "residual_error_equals_reconstructed_state_error": equality_difference <= 2.0e-14,
        "time_regime_breakdown": breakdown_metrics(residual_errors, time_masks),
        "hard_family_breakdown": breakdown_metrics(residual_errors, family_masks),
        "sample_group_breakdown": breakdown_metrics(residual_errors, group_masks),
        "hard_targeted_subtype_breakdown": breakdown_metrics(residual_errors, subtype_masks),
        "identity_diagnostics": identity_diagnostics,
        "physical_diagnostics": physical_diagnostics,
        "validation_rows": int(residual_errors.shape[0]),
        "sealed_test_predictions_computed": False,
    }
