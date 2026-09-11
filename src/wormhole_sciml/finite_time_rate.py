"""Identity-preserving average-rate finite-time model utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .finite_time import (
    BATCH_SIZE,
    EXPECTED_PARAMETER_COUNT,
    HIDDEN_DIMENSIONS,
    INPUT_COLUMNS,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PATIENCE,
    TRAINING_SEEDS,
    FiniteTimePreprocessing,
    orbit_averaged_standardized_mse,
    stack_columns,
)
from .model_a import ModelA, Normalization, load_trained_model, parameter_count, predict_standardized, state_dict_sha256
from .physics_gate import experiment_parameters, xi_time_derivative


RATE_TARGET_COLUMNS = ("V_x", "V_xi")
RATE_PREPROCESSING_SCHEMA = "wormhole_finite_time_average_rate_standardization/v1"
RATE_PREPROCESSING_IMPLEMENTATION = "wormhole_sciml.finite_time_rate/v1"


@dataclass(frozen=True, slots=True)
class RatePreprocessing:
    """Old frozen input constants plus new training-only rate-target constants."""

    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    source_training_sha256: str
    source_input_preprocessing_sha256: str

    @classmethod
    def fit(
        cls,
        training: dict[str, np.ndarray],
        old_preprocessing: FiniteTimePreprocessing,
        source_training_sha256: str,
        source_input_preprocessing_sha256: str,
    ) -> "RatePreprocessing":
        targets = stack_columns(training, RATE_TARGET_COLUMNS)
        target_mean = np.mean(targets, axis=0, dtype=np.float64)
        target_std = np.std(targets, axis=0, ddof=0, dtype=np.float64)
        if not np.all(np.isfinite(target_std)) or np.any(target_std <= 0.0):
            raise ValueError("rate-target standard deviations must be finite and positive")
        return cls(
            old_preprocessing.input_mean.copy(),
            old_preprocessing.input_std.copy(),
            target_mean,
            target_std,
            source_training_sha256,
            source_input_preprocessing_sha256,
        )

    @classmethod
    def from_json(cls, path: Path) -> "RatePreprocessing":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema"] != RATE_PREPROCESSING_SCHEMA:
            raise ValueError("unsupported average-rate preprocessing schema")
        if tuple(payload["input_order"]) != INPUT_COLUMNS:
            raise ValueError("average-rate input ordering mismatch")
        if tuple(payload["target_order"]) != RATE_TARGET_COLUMNS:
            raise ValueError("average-rate target ordering mismatch")
        columns = payload["columns"]
        return cls(
            np.asarray([columns[name]["mean"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["mean"] for name in RATE_TARGET_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in RATE_TARGET_COLUMNS], dtype=np.float64),
            str(payload["source_training_dataset_sha256"]),
            str(payload["source_input_preprocessing_sha256"]),
        )

    def as_normalization(self) -> Normalization:
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
        names = INPUT_COLUMNS + RATE_TARGET_COLUMNS
        means = np.concatenate((self.input_mean, self.target_mean))
        stds = np.concatenate((self.input_std, self.target_std))
        return {
            "schema": RATE_PREPROCESSING_SCHEMA,
            "implementation": RATE_PREPROCESSING_IMPLEMENTATION,
            "source_split": "train only",
            "source_training_dataset": str(source_path.resolve()),
            "source_training_dataset_sha256": self.source_training_sha256,
            "source_training_row_count": int(source_rows),
            "source_input_preprocessing_sha256": self.source_input_preprocessing_sha256,
            "input_constants_policy": "copied exactly from frozen accumulated-residual preprocessing",
            "input_order": list(INPUT_COLUMNS),
            "target_order": list(RATE_TARGET_COLUMNS),
            "statistics_dtype": "float64",
            "model_tensor_dtype": "float32",
            "standard_deviation_definition": "population (ddof=0)",
            "output_gate": "physical Delta_hat = physical s * unstandardized V_hat",
            "columns": {
                name: {"mean": float(mean), "standard_deviation": float(std)}
                for name, mean, std in zip(names, means, stds)
            },
        }


def construct_rate_targets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add exact average rates without ever dividing identity rows by zero."""

    elapsed = np.asarray(data["s"], dtype=np.float64)
    delta_x = np.asarray(data["Delta_x"], dtype=np.float64)
    delta_xi = np.asarray(data["Delta_xi"], dtype=np.float64)
    positive = elapsed > 0.0
    if np.any(elapsed < 0.0):
        raise ValueError("physical elapsed time cannot be negative")
    rate_x = np.empty_like(elapsed)
    rate_xi = np.empty_like(elapsed)
    np.divide(delta_x, elapsed, out=rate_x, where=positive)
    np.divide(delta_xi, elapsed, out=rate_xi, where=positive)
    identity = ~positive
    if np.any(identity):
        wormhole, spiral = experiment_parameters()
        rate_x[identity] = np.asarray(data["u0"], dtype=np.float64)[identity]
        rate_xi[identity] = xi_time_derivative(
            np.asarray(data["x0"], dtype=np.float64)[identity],
            np.asarray(data["u0"], dtype=np.float64)[identity],
            wormhole,
            spiral,
        )
    if not np.all(np.isfinite(rate_x)) or not np.all(np.isfinite(rate_xi)):
        raise FloatingPointError("average-rate targets contain nonfinite values")
    return {**data, "V_x": rate_x, "V_xi": rate_xi}


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999))
    return {
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=0)),
        "median": float(q[3]), "p0.1": float(q[0]), "p1": float(q[1]),
        "p5": float(q[2]), "p95": float(q[4]), "p99": float(q[5]), "p99.9": float(q[6]),
    }


def rate_target_audit(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Audit raw rate targets globally, by time, and against the local generator."""

    elapsed = np.asarray(data["s"], dtype=np.float64)
    wormhole, spiral = experiment_parameters()
    generator_x = np.asarray(data["u0"], dtype=np.float64)
    generator_xi = xi_time_derivative(data["x0"], data["u0"], wormhole, spiral)
    time_masks = {
        "identity_s_eq_0": elapsed == 0.0,
        "very_small_0_lt_s_le_0p01": (elapsed > 0.0) & (elapsed <= 0.01),
        "small_0p01_lt_s_le_0p1": (elapsed > 0.01) & (elapsed <= 0.1),
        "0p1_lt_s_le_1": (elapsed > 0.1) & (elapsed <= 1.0),
        "1_lt_s_le_5": (elapsed > 1.0) & (elapsed <= 5.0),
        "5_lt_s_le_20": (elapsed > 5.0) & (elapsed <= 20.0),
        "long_s_gt_20": elapsed > 20.0,
    }
    family_masks = {
        "hard_u_th_le_0p30": np.asarray(data["u_th"]) <= 0.30,
        "ordinary_u_th_gt_0p30": np.asarray(data["u_th"]) > 0.30,
        "high_edge_u_th_ge_0p85": np.asarray(data["u_th"]) >= 0.85,
    }
    by_time: dict[str, Any] = {}
    for name, mask in time_masks.items():
        by_time[name] = {
            "row_count": int(np.sum(mask)),
            "V_x": distribution(data["V_x"][mask]),
            "V_xi": distribution(data["V_xi"][mask]),
            "V_x_minus_u0": distribution(data["V_x"][mask] - generator_x[mask]),
            "V_xi_minus_dot_xi0": distribution(data["V_xi"][mask] - generator_xi[mask]),
        }
    by_family = {
        name: {
            "row_count": int(np.sum(mask)),
            "V_x": distribution(data["V_x"][mask]),
            "V_xi": distribution(data["V_xi"][mask]),
        }
        for name, mask in family_masks.items()
    }
    very_small = time_masks["very_small_0_lt_s_le_0p01"]
    all_finite = bool(np.all(np.isfinite(data["V_x"])) and np.all(np.isfinite(data["V_xi"])))
    maximum_absolute = float(max(np.max(np.abs(data["V_x"])), np.max(np.abs(data["V_xi"]))))
    small_p99_generator_difference = float(max(
        np.quantile(np.abs(data["V_x"][very_small] - generator_x[very_small]), 0.99),
        np.quantile(np.abs(data["V_xi"][very_small] - generator_xi[very_small]), 0.99),
    ))
    pathology = (not all_finite) or maximum_absolute > 1.0e4 or small_p99_generator_difference > 1.0
    return {
        "row_count": int(elapsed.size),
        "identity_row_count": int(np.sum(elapsed == 0.0)),
        "minimum_positive_s": float(np.min(elapsed[elapsed > 0.0])),
        "global": {name: distribution(data[name]) for name in RATE_TARGET_COLUMNS},
        "by_physical_s": by_time,
        "by_family": by_family,
        "all_finite": all_finite,
        "maximum_absolute_raw_rate": maximum_absolute,
        "small_positive_s_p99_absolute_generator_difference": small_p99_generator_difference,
        "predeclared_stop_checks": {
            "maximum_absolute_raw_rate_lt_1e4": maximum_absolute < 1.0e4,
            "small_positive_s_p99_generator_difference_lt_1": small_p99_generator_difference < 1.0,
        },
        "obvious_numerical_pathology": pathology,
        "decision": "FAIL_STOP_BEFORE_TRAINING" if pathology else "PASS_TRAINING_ALLOWED",
    }


def tensor_dataset(data: dict[str, np.ndarray], preprocessing: RatePreprocessing) -> TensorDataset:
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, INPUT_COLUMNS)))
    targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(data, RATE_TARGET_COLUMNS)))
    return TensorDataset(inputs, targets)


def set_deterministic_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed, "numpy_legacy_seed": seed,
        "torch_manual_seed": seed, "torch_deterministic_algorithms": True,
        "torch_num_threads": 1, "dataloader_num_workers": 0,
    }


@torch.no_grad()
def predict_rates(
    model: ModelA, preprocessing: RatePreprocessing, data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, INPUT_COLUMNS)))
    standardized = predict_standardized(model, inputs, batch_size=8192).cpu().numpy()
    rates = preprocessing.unstandardize_targets(standardized)
    elapsed = np.asarray(data["s"], dtype=np.float64)
    residual = elapsed[:, None] * rates
    state = np.column_stack((data["x0"], data["xi0"])) + residual
    return {
        "standardized_V_x": standardized[:, 0], "standardized_V_xi": standardized[:, 1],
        "predicted_V_x": rates[:, 0], "predicted_V_xi": rates[:, 1],
        "predicted_Delta_x": residual[:, 0], "predicted_Delta_xi": residual[:, 1],
        "predicted_x1": state[:, 0], "predicted_xi1": state[:, 1],
    }


def _component_metrics(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    q = np.quantile(absolute, (0.5, 0.9, 0.95, 0.99))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(absolute)),
        "median_absolute": float(q[0]), "p90_absolute": float(q[1]),
        "p95_absolute": float(q[2]), "p99_absolute": float(q[3]),
        "maximum_absolute": float(np.max(absolute)),
    }


def two_component_metrics(x_error: np.ndarray, xi_error: np.ndarray) -> dict[str, Any]:
    return {"x": _component_metrics(x_error), "xi": _component_metrics(xi_error)}


def subset_metrics(prediction: dict[str, np.ndarray], data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "row_count": int(np.sum(mask)),
        **two_component_metrics(
            prediction["predicted_x1"][mask] - np.asarray(data["x1"])[mask],
            prediction["predicted_xi1"][mask] - np.asarray(data["xi1"])[mask],
        ),
    }


def train_rate_seed(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    preprocessing: RatePreprocessing,
    seed: int,
    output_dir: Path,
    *,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience: int = PATIENCE,
    progress: bool = True,
) -> dict[str, Any]:
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported finite-time seed {seed}")
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
            train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=0, drop_last=False,
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
        model.eval()
        with torch.no_grad():
            validation_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
        measured = orbit_averaged_standardized_mse(
            validation_prediction, validation_targets.cpu().numpy(), validation["orbit_id"]
        )
        current = float(measured["orbit_averaged_standardized_mse"])
        history.append({
            "epoch": epoch, "training_standardized_rate_mse": training_loss,
            "validation_orbit_averaged_standardized_rate_mse": current,
            "validation_row_averaged_standardized_rate_mse": float(measured["ordinary_row_averaged_standardized_mse"]),
        })
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            without_improvement = 0
        else:
            without_improvement += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            print(f"average-rate seed={seed} epoch={epoch} val={current:.8g} best={best_metric:.8g}", flush=True)
        if without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("average-rate training produced no checkpoint")
    stopping_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
    restored = orbit_averaged_standardized_mse(
        restored_prediction, validation_targets.cpu().numpy(), validation["orbit_id"]
    )
    if float(restored["orbit_averaged_standardized_mse"]) != best_metric:
        raise RuntimeError("restored average-rate checkpoint does not reproduce best metric")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "architecture": "4->64->64->2 with two tanh hidden layers and no output activation",
        "parameter_count": parameter_count(model),
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "dtype": "float32", "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "weight_decay": 0.0, "scheduler": None,
        "loss": "equal two-component standardized MSE on V_x and V_xi",
        "maximum_epochs": maximum_epochs, "early_stopping_patience": patience,
        "checkpoint_selection_metric": "validation orbit-averaged standardized rate MSE only",
        "best_epoch": best_epoch, "stopping_epoch": stopping_epoch,
        "best_validation_orbit_averaged_standardized_rate_mse": best_metric,
        "final_training_standardized_rate_mse": float(history[-1]["training_standardized_rate_mse"]),
        "final_validation_orbit_averaged_standardized_rate_mse": float(history[-1]["validation_orbit_averaged_standardized_rate_mse"]),
        "early_stopping_triggered": stopping_epoch < maximum_epochs,
        "batches_per_epoch": steps_per_epoch,
        "total_optimizer_updates": steps_per_epoch * stopping_epoch,
        "restored_validation": restored,
        "checkpoint": str(checkpoint.resolve()), "history": str(history_path.resolve()),
        "reproducibility": reproducibility, "sealed_test_predictions_computed": False,
    }


def load_rate_model(path: Path) -> ModelA:
    return load_trained_model(path)

