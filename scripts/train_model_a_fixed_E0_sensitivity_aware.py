#!/usr/bin/env python3
"""Train exactly three fixed-E0 models with the controlled sensitivity-aware loss."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wormhole_sciml.energy_gradient import standardized_energy_normal
from wormhole_sciml.model_a import (
    LEARNING_RATE,
    PATIENCE,
    PHYSICAL_BATCH_SIZE,
    ModelA,
    Normalization,
    TRAINING_SEEDS,
    _batch_slices,
    _permutation,
    parameter_count,
    predict_standardized,
    state_dict_sha256,
)
from wormhole_sciml.stage1_data import file_sha256, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling"
TRAIN_DATA = DATA_DIR / "outer_microcore40k_train_x_xi.npz"
VALIDATION_DATA = DATA_DIR / "outer_microcore8k_validation_x_xi.npz"
BASELINE = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
BASELINE_MANIFEST = BASELINE / "energy_xi_training_manifest.json"
NORMALIZATION_PATH = BASELINE / "training" / "energy_input_normalization.json"
BASELINE_SUMMARY = BASELINE / "energy_xi_rollout_summary.json"
BASELINE_ARRAYS = BASELINE / "energy_xi_rollout_arrays.npz"
GATE = ROOT / "output" / "model_a_sensitivity_map_physics_gate"
GATE_ARRAYS = GATE / "sensitivity_map_arrays.npz"
GATE_SUMMARY = GATE / "sensitivity_map_summary.json"
GATE_CLASSES = GATE / "microcore40k_orbit_classes.csv"
ALIGNMENT_ARRAYS = (
    ROOT / "output" / "model_a_energy_gradient_alignment" / "energy_gradient_alignment_arrays.npz"
)
SENSITIVITY_ARRAYS = (
    ROOT / "output" / "model_a_fixed_E0_throat_sensitivity" / "throat_sensitivity_arrays.npz"
)

OUTPUT = ROOT / "output" / "model_a_fixed_E0_sensitivity_aware_loss"
TRAINING = OUTPUT / "training"
DERIVED = OUTPUT / "derived_sensitivity_loss_arrays.npz"
MANIFEST = OUTPUT / "sensitivity_aware_training_manifest.json"

INPUT_COLUMNS = ("x", "xi", "E0")
TARGET_COLUMNS = ("delta_x", "delta_xi")
NORMALIZATION_SOURCE = "outer_microcore40k_train_x_xi_energy_input_only"
HIDDEN = (32, 32)
MAXIMUM_EPOCHS = 1500
LAMBDA_S = 1.0
MEAN_ABS_S = 0.3526914868127516
BASELINE_FACTOR_C = 0.5
DEGENERACY_THRESHOLD = 1.0e-14
TOP_FRACTIONS = (0.001, 0.01, 0.05)


def hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required protected artifact(s) missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def baseline_runs() -> dict[int, dict[str, Any]]:
    payload = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    runs = {int(row["seed"]): row for row in payload["runs"]}
    if set(runs) != set(TRAINING_SEEDS):
        raise RuntimeError("the three frozen fixed-E0 baseline runs are incomplete")
    return runs


def protected_paths() -> tuple[Path, ...]:
    paths = [
        TRAIN_DATA,
        VALIDATION_DATA,
        NORMALIZATION_PATH,
        BASELINE_MANIFEST,
        BASELINE_SUMMARY,
        BASELINE_ARRAYS,
        GATE_ARRAYS,
        GATE_SUMMARY,
        GATE_CLASSES,
        ALIGNMENT_ARRAYS,
        SENSITIVITY_ARRAYS,
        ROOT / "src" / "wormhole_sciml" / "energy_gradient.py",
        ROOT / "src" / "wormhole_sciml" / "dynamics.py",
        ROOT / "src" / "wormhole_sciml" / "physics_gate.py",
    ]
    for row in baseline_runs().values():
        run_dir = Path(row["checkpoint"]).parent
        paths.extend((Path(row["checkpoint"]), Path(row["history"]), run_dir / "metadata.json"))
    return tuple(paths)


def model_arrays(
    data: dict[str, np.ndarray], normalization: Normalization
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = np.column_stack(tuple(data[name] for name in INPUT_COLUMNS))
    targets = np.column_stack(tuple(data[name] for name in TARGET_COLUMNS))
    return (
        torch.from_numpy(normalization.standardize_inputs(inputs)),
        torch.from_numpy(normalization.standardize_targets(targets)),
    )


def derive_normals(
    data: dict[str, np.ndarray], normalization: Normalization
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.allclose(data["x"] + data["delta_x"], data["x_next"], rtol=0.0, atol=2e-16):
        raise RuntimeError("stored x_next does not match the exact target increment")
    if not np.allclose(data["xi"] + data["delta_xi"], data["xi_next"], rtol=0.0, atol=2e-16):
        raise RuntimeError("stored xi_next does not match the exact target increment")
    return standardized_energy_normal(
        data["x_next"],
        data["xi_next"],
        float(normalization.target_std[0]),
        float(normalization.target_std[1]),
        degeneracy_threshold=DEGENERACY_THRESHOLD,
    )


def preflight(
    normalization: Normalization,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_normal, train_g_norm, train_degenerate = derive_normals(train, normalization)
    val_normal, val_g_norm, val_degenerate = derive_normals(validation, normalization)
    if np.any(train_degenerate) or np.any(val_degenerate):
        raise RuntimeError("a derived energy gradient is numerically degenerate")

    gate_summary = json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))
    with np.load(GATE_ARRAYS, allow_pickle=False) as gate:
        if not (
            np.array_equal(gate["train_x"], train["x"])
            and np.array_equal(gate["train_xi"], train["xi"])
            and np.array_equal(gate["train_E0"], train["E0"])
        ):
            raise RuntimeError("saved sensitivity classification is not row-aligned with microcore40k")
        train_eligible = np.asarray(gate["train_eligible"], dtype=bool)
        eligible_indices = np.asarray(gate["eligible_indices"], dtype=np.int64)
        if not np.array_equal(eligible_indices, np.flatnonzero(train_eligible)):
            raise RuntimeError("saved eligible index map is inconsistent")
        train_weights = np.zeros(train_eligible.size, dtype=np.float64)
        train_weights[eligible_indices] = np.asarray(
            gate["eligible_normalized_abs_S_weight"], dtype=np.float64
        )
        train_branch = np.asarray(gate["train_orbit_branch"], dtype=np.int8)
        val_eligible = np.asarray(gate["validation_eligible"], dtype=bool)
        val_weights = np.zeros(val_eligible.size, dtype=np.float64)
        val_weights[val_eligible] = np.asarray(
            gate["validation_normalized_abs_S_weight"], dtype=np.float64
        )

    if int(train_eligible.sum()) != 10_352 or int(val_eligible.sum()) != 2_122:
        raise RuntimeError("saved eligible counts do not reproduce the completed gate")
    left_to_right = int(np.sum(train_eligible & (train_branch == 1)))
    right_to_left = int(np.sum(train_eligible & (train_branch == -1)))
    if (left_to_right, right_to_left) != (10_081, 271):
        raise RuntimeError("saved directed eligible-branch counts changed")
    if not np.isclose(np.mean(train_weights[train_eligible]), 1.0, rtol=0.0, atol=2e-15):
        raise RuntimeError("eligible mean sensitivity weight is not one")
    if not np.isclose(
        gate_summary["normalized_abs_S"]["normalization_mean_raw"],
        MEAN_ABS_S,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("saved |S| normalization differs from the frozen value")

    maximum_gate_difference = 0.0
    with np.load(ALIGNMENT_ARRAYS, allow_pickle=False) as alignment:
        for u_th in (0.05, 0.15, 0.30):
            key = f"u_th_{u_th:.2f}".replace(".", "p")
            measured, _, degenerate = standardized_energy_normal(
                alignment[f"{key}__post_x"],
                alignment[f"{key}__post_xi"],
                float(normalization.target_std[0]),
                float(normalization.target_std[1]),
                degeneracy_threshold=DEGENERACY_THRESHOLD,
            )
            if np.any(degenerate):
                raise RuntimeError("an alignment-gate state became degenerate")
            maximum_gate_difference = max(
                maximum_gate_difference,
                float(np.max(np.abs(measured - alignment[f"{key}__n_E"]))),
            )
    if maximum_gate_difference > 2e-15:
        raise RuntimeError("training-side n_E does not reproduce the alignment gate")

    expected_hard = {
        "0.05": (8.101074758633867, 22.96929486969679),
        "0.15": (3.0117565557361163, 8.539351439846623),
        "0.30": (1.5456799573690159, 4.3825269822564135),
    }
    hard_checks: dict[str, Any] = {}
    for row in gate_summary["hard_families"]:
        family = f"{float(row['family_u_th']):.2f}"
        expected_s, expected_weight = expected_hard[family]
        s_difference = abs(float(row["abs_S"]) - expected_s)
        weight_difference = abs(float(row["normalized_abs_S_weight"]) - expected_weight)
        if s_difference > 1e-13 or weight_difference > 1e-13:
            raise RuntimeError("hard-family sensitivity preflight failed")
        hard_checks[family] = {
            "abs_S": float(row["abs_S"]),
            "normalized_weight": float(row["normalized_abs_S_weight"]),
            "abs_S_difference": s_difference,
            "weight_difference": weight_difference,
        }

    probe_weight = expected_hard["0.05"][1]
    loss_geometry = {
        "unit_tangent_noneligible": {"base": 0.5, "sensitivity": 0.0, "total": 0.5},
        "unit_normal_noneligible": {"base": 0.5, "sensitivity": 0.0, "total": 0.5},
        "unit_tangent_eligible_hard_0p05": {
            "base": 0.5,
            "sensitivity": 0.0,
            "total": 0.5,
        },
        "unit_normal_eligible_hard_0p05": {
            "base": 0.5,
            "sensitivity": 0.5 * probe_weight,
            "total": 0.5 + 0.5 * probe_weight,
        },
    }
    arrays = {
        "train_n_E": train_normal,
        "train_g_E_norm": train_g_norm,
        "train_eligible": train_eligible,
        "train_sensitivity_weight": train_weights,
        "train_source_row_index": np.asarray(train["source_row_index"]),
        "train_x_next": np.asarray(train["x_next"]),
        "train_xi_next": np.asarray(train["xi_next"]),
        "validation_n_E": val_normal,
        "validation_g_E_norm": val_g_norm,
        "validation_eligible": val_eligible,
        "validation_sensitivity_weight": val_weights,
        "validation_source_row_index": np.asarray(validation["source_row_index"]),
        "validation_x_next": np.asarray(validation["x_next"]),
        "validation_xi_next": np.asarray(validation["xi_next"]),
    }
    summary = {
        "completed_alignment_gate_maximum_absolute_n_E_difference": maximum_gate_difference,
        "hard_family_checks": hard_checks,
        "loss_geometry": loss_geometry,
        "mean_abs_S_normalization": MEAN_ABS_S,
        "train": {
            "row_count": int(train_eligible.size),
            "eligible_count": int(train_eligible.sum()),
            "noneligible_count": int((~train_eligible).sum()),
            "eligible_fraction": float(train_eligible.mean()),
            "left_to_right_eligible_count": left_to_right,
            "right_to_left_eligible_count": right_to_left,
            "degenerate_count": int(train_degenerate.sum()),
            "eligible_weight_mean": float(np.mean(train_weights[train_eligible])),
            "eligible_weight_maximum": float(np.max(train_weights)),
        },
        "validation": {
            "row_count": int(val_eligible.size),
            "eligible_count": int(val_eligible.sum()),
            "noneligible_count": int((~val_eligible).sum()),
            "eligible_fraction": float(val_eligible.mean()),
            "degenerate_count": int(val_degenerate.sum()),
            "eligible_weight_mean": float(np.mean(val_weights[val_eligible])),
            "eligible_weight_maximum": float(np.max(val_weights)),
        },
    }
    return arrays, summary


def subset_metrics(
    error: torch.Tensor,
    projection: torch.Tensor,
    weights: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for name, mask in (("eligible", eligible), ("noneligible", ~eligible)):
        count = int(torch.sum(mask).item())
        result[f"{name}_count"] = count
        result[f"{name}_baseline_mse"] = float(0.5 * torch.mean(torch.sum(error[mask] ** 2, dim=1)).item())
        result[f"{name}_energy_normal_rms"] = float(torch.sqrt(torch.mean(projection[mask] ** 2)).item())
        result[f"{name}_energy_normal_mae"] = float(torch.mean(torch.abs(projection[mask])).item())
    result["sensitivity_loss"] = float(0.5 * torch.mean(weights * projection**2).item())
    result["total_loss"] = float(torch.mean(error**2).item() + LAMBDA_S * result["sensitivity_loss"])
    return result


@torch.no_grad()
def validation_metrics(
    model: ModelA,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, float | int]:
    prediction = predict_standardized(model, inputs)
    error = prediction - targets
    projection = torch.sum(error * normals, dim=1)
    tangent = torch.stack((-normals[:, 1], normals[:, 0]), dim=1)
    tangent_projection = torch.sum(error * tangent, dim=1)
    result: dict[str, float | int] = {
        "standardized_mse": float(torch.mean(error**2).item()),
        "energy_normal_rms": float(torch.sqrt(torch.mean(projection**2)).item()),
        "energy_normal_mae": float(torch.mean(torch.abs(projection)).item()),
        "energy_tangent_rms": float(torch.sqrt(torch.mean(tangent_projection**2)).item()),
        "energy_tangent_mae": float(torch.mean(torch.abs(tangent_projection)).item()),
    }
    result.update(subset_metrics(error, projection, weights, eligible))
    return result


def top_weight_masks(weights: np.ndarray, eligible: np.ndarray) -> dict[str, torch.Tensor]:
    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[np.argsort(weights[eligible_indices])]
    result: dict[str, torch.Tensor] = {}
    for fraction in TOP_FRACTIONS:
        count = max(1, int(np.ceil(order.size * fraction)))
        mask = np.zeros(weights.size, dtype=bool)
        mask[order[-count:]] = True
        result[f"top_{100 * fraction:g}_percent"] = torch.from_numpy(mask)
    return result


def train_seed(
    seed: int,
    normalization: Normalization,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    derived: dict[str, np.ndarray],
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    train_inputs, train_targets = model_arrays(train_data, normalization)
    val_inputs, val_targets = model_arrays(validation_data, normalization)
    train_normals = torch.from_numpy(np.asarray(derived["train_n_E"], dtype=np.float32))
    val_normals = torch.from_numpy(np.asarray(derived["validation_n_E"], dtype=np.float32))
    train_weights = torch.from_numpy(np.asarray(derived["train_sensitivity_weight"], dtype=np.float32))
    val_weights = torch.from_numpy(np.asarray(derived["validation_sensitivity_weight"], dtype=np.float32))
    train_eligible = torch.from_numpy(np.asarray(derived["train_eligible"], dtype=bool))
    val_eligible = torch.from_numpy(np.asarray(derived["validation_eligible"], dtype=bool))
    top_masks = top_weight_masks(np.asarray(derived["train_sensitivity_weight"]), np.asarray(derived["train_eligible"]))

    model = ModelA(3, HIDDEN).to(dtype=torch.float32)
    if parameter_count(model) != 1250:
        raise RuntimeError("sensitivity-aware model is not 3->32->32->2")
    initial_hash = state_dict_sha256(model.state_dict())
    baseline = baseline_runs()[seed]
    if initial_hash != baseline["initial_state_sha256"]:
        raise RuntimeError("same-seed initialization differs from frozen fixed-E0 baseline")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    slices = _batch_slices(train_inputs.shape[0], PHYSICAL_BATCH_SIZE)
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        order = _permutation(train_inputs.shape[0], seed, epoch, 1)
        sum_squared = sum_weighted_projection_squared = 0.0
        sum_projection_squared = sum_abs_projection = 0.0
        eligible_squared = noneligible_squared = 0.0
        eligible_projection_squared = noneligible_projection_squared = 0.0
        eligible_count = noneligible_count = count = 0
        top_sensitivity_sums = {name: 0.0 for name in top_masks}
        gradient_norm_sum = maximum_gradient_norm = 0.0
        maximum_batch_sensitivity = maximum_batch_total = 0.0
        maximum_weight_batch_sensitivity = 0.0
        maximum_weight_batch_index = -1
        maximum_sensitivity_batch_index = -1
        for batch_index, batch_slice in enumerate(slices):
            indices = order[batch_slice]
            prediction = model(train_inputs[indices])
            error = prediction - train_targets[indices]
            normal = train_normals[indices]
            projection = torch.sum(error * normal, dim=1)
            weight = train_weights[indices]
            base_loss = torch.mean(error**2)
            sensitivity_loss = 0.5 * torch.mean(weight * projection**2)
            objective = base_loss + LAMBDA_S * sensitivity_loss
            if not torch.isfinite(objective):
                raise FloatingPointError(f"seed {seed} epoch {epoch}: nonfinite objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradient_norm = math.sqrt(
                sum(float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in model.parameters() if parameter.grad is not None)
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"seed {seed} epoch {epoch}: nonfinite gradient norm")
            optimizer.step()

            detached_error = error.detach().to(dtype=torch.float64)
            detached_projection = projection.detach().to(dtype=torch.float64)
            detached_weighted = (weight * projection**2).detach().to(dtype=torch.float64)
            batch_eligible = train_eligible[indices]
            batch_noneligible = ~batch_eligible
            sum_squared += float(torch.sum(detached_error**2).item())
            sum_weighted_projection_squared += float(torch.sum(detached_weighted).item())
            sum_projection_squared += float(torch.sum(detached_projection**2).item())
            sum_abs_projection += float(torch.sum(torch.abs(detached_projection)).item())
            eligible_squared += float(torch.sum(detached_error[batch_eligible] ** 2).item())
            noneligible_squared += float(torch.sum(detached_error[batch_noneligible] ** 2).item())
            eligible_projection_squared += float(torch.sum(detached_projection[batch_eligible] ** 2).item())
            noneligible_projection_squared += float(torch.sum(detached_projection[batch_noneligible] ** 2).item())
            eligible_count += int(torch.sum(batch_eligible).item())
            noneligible_count += int(torch.sum(batch_noneligible).item())
            count += int(error.shape[0])
            for name, mask in top_masks.items():
                top_sensitivity_sums[name] += float(torch.sum(detached_weighted[mask[indices]]).item())
            batch_sens = float(sensitivity_loss.detach().item())
            batch_total = float(objective.detach().item())
            if batch_sens > maximum_batch_sensitivity:
                maximum_batch_sensitivity = batch_sens
                maximum_sensitivity_batch_index = batch_index
            maximum_batch_total = max(maximum_batch_total, batch_total)
            if torch.any(weight == torch.max(train_weights)):
                maximum_weight_batch_sensitivity = batch_sens
                maximum_weight_batch_index = batch_index
            gradient_norm_sum += gradient_norm
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)

        training_base = BASELINE_FACTOR_C * sum_squared / count
        training_sensitivity = BASELINE_FACTOR_C * sum_weighted_projection_squared / count
        validation = validation_metrics(
            model, val_inputs, val_targets, val_normals, val_weights, val_eligible
        )
        concentration = {
            name: (value / sum_weighted_projection_squared if sum_weighted_projection_squared > 0 else 0.0)
            for name, value in top_sensitivity_sums.items()
        }
        history.append(
            {
                "epoch": epoch,
                "training_base_standardized_mse": training_base,
                "training_sensitivity_loss": training_sensitivity,
                "training_total_loss": training_base + LAMBDA_S * training_sensitivity,
                "training_energy_normal_rms": float(np.sqrt(sum_projection_squared / count)),
                "training_energy_normal_mae": sum_abs_projection / count,
                "training_eligible_baseline_mse": BASELINE_FACTOR_C * eligible_squared / eligible_count,
                "training_noneligible_baseline_mse": BASELINE_FACTOR_C * noneligible_squared / noneligible_count,
                "training_eligible_energy_normal_rms": float(np.sqrt(eligible_projection_squared / eligible_count)),
                "training_noneligible_energy_normal_rms": float(np.sqrt(noneligible_projection_squared / noneligible_count)),
                "training_sensitivity_loss_top_weight_contribution": concentration,
                "mean_gradient_norm_before_step": gradient_norm_sum / len(slices),
                "maximum_gradient_norm_before_step": maximum_gradient_norm,
                "maximum_batch_sensitivity_loss": maximum_batch_sensitivity,
                "maximum_batch_total_loss": maximum_batch_total,
                "maximum_weight_batch_sensitivity_loss": maximum_weight_batch_sensitivity,
                "maximum_weight_batch_index": maximum_weight_batch_index,
                "maximum_sensitivity_batch_index": maximum_sensitivity_batch_index,
                "maximum_weight_batch_is_maximum_sensitivity_batch": maximum_weight_batch_index == maximum_sensitivity_batch_index,
                "maximum_eligible_weight": float(torch.max(train_weights).item()),
                **{f"validation_{name}": value for name, value in validation.items()},
            }
        )
        if float(validation["standardized_mse"]) < best_validation:
            best_validation = float(validation["standardized_mse"])
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"sensitivity_aware seed={seed} epoch={epoch} val_base={validation['standardized_mse']:.8g} "
                f"val_sens={validation['sensitivity_loss']:.8g} best={best_validation:.8g} "
                f"max_grad={maximum_gradient_norm:.5g}",
                flush=True,
            )
        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    restored = validation_metrics(model, val_inputs, val_targets, val_normals, val_weights, val_eligible)
    if float(restored["standardized_mse"]) != best_validation:
        raise RuntimeError("restored checkpoint does not reproduce the selection metric")
    run_dir = TRAINING / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    history_path = run_dir / "training_history.json"
    torch.save(best_state, checkpoint)
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_history = history[best_epoch - 1]
    metadata: dict[str, Any] = {
        "stage": "controlled fixed-E0 sensitivity-aware local-loss training",
        "seed": seed,
        "architecture": "3->32->32->2 with two tanh hidden layers",
        "hidden_dimensions": list(HIDDEN),
        "parameter_count": 1250,
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "dtype": "float32",
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "same_seed_initialization_matches_fixed_E0_baseline": True,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "scheduler": None,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "physical_epoch_steps": len(slices),
        "maximum_epochs": MAXIMUM_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "checkpoint_selection_metric": "ordinary physical validation standardized increment MSE only",
        "lambda_S": LAMBDA_S,
        "sensitivity_weight": "eligible_indicator * abs(S) / 0.3526914868127516",
        "clipping_capping_or_transform": None,
        "objective": "mean(e**2) + 0.5*mean(I*w_absS*(n_E dot e)**2)",
        "eligible_training_count": int(train_eligible.sum().item()),
        "eligible_validation_count": int(val_eligible.sum().item()),
        "best_epoch": best_epoch,
        "stopping_epoch": int(history[-1]["epoch"]),
        "early_stopping_triggered": int(history[-1]["epoch"]) < MAXIMUM_EPOCHS,
        "best_validation_standardized_mse": best_validation,
        "selected_checkpoint_validation": restored,
        "selected_epoch_training_diagnostics": selected_history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "history": str(history_path),
        "normalization": str(NORMALIZATION_PATH),
        "physical_train_dataset": str(TRAIN_DATA),
        "physical_validation_dataset": str(VALIDATION_DATA),
        "derived_sensitivity_loss_arrays": str(DERIVED),
        "sealed_data_used": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.preflight_only:
        raise FileExistsError(f"refusing to overwrite existing output {OUTPUT}")
    protected = protected_paths()
    before = hashes(protected)
    normalization = Normalization.from_stage1(
        NORMALIZATION_PATH, INPUT_COLUMNS, TARGET_COLUMNS, NORMALIZATION_SOURCE
    )
    train = load_dataset(TRAIN_DATA)
    validation = load_dataset(VALIDATION_DATA)
    derived, preflight_summary = preflight(normalization, train, validation)
    if args.preflight_only:
        print(json.dumps(preflight_summary, indent=2, sort_keys=True))
        return
    OUTPUT.mkdir()
    TRAINING.mkdir()
    np.savez_compressed(DERIVED, **derived)
    runs = [train_seed(seed, normalization, train, validation, derived) for seed in TRAINING_SEEDS]
    after = hashes(protected)
    if before != after:
        raise RuntimeError("a protected baseline, dataset, physics, or gate artifact changed")
    manifest: dict[str, Any] = {
        "stage": "controlled fixed-E0 sensitivity-aware local-loss training",
        "status": "exactly_three_sensitivity_aware_runs_completed",
        "runs": runs,
        "training_seeds": list(TRAINING_SEEDS),
        "architecture": "3->32->32->2 with two tanh hidden layers",
        "parameter_count": 1250,
        "lambda_S": LAMBDA_S,
        "loss": "mean(e**2) + 0.5*mean(I*w_absS*(n_E dot e)**2)",
        "checkpoint_selection_metric": "ordinary validation standardized MSE, unchanged from fixed-E0 baseline",
        "preflight": preflight_summary,
        "derived_sensitivity_loss_arrays": {
            "path": str(DERIVED),
            "sha256": file_sha256(DERIVED),
            "source_gate_arrays": str(GATE_ARRAYS),
            "source_gate_arrays_sha256": before[str(GATE_ARRAYS)],
            "correspondence_fields": ["source_row_index", "x_next", "xi_next"],
        },
        "normalization": {
            "path": str(NORMALIZATION_PATH),
            "sha256": before[str(NORMALIZATION_PATH)],
            "target_std": normalization.target_std.tolist(),
        },
        "frozen_fixed_E0_baseline_manifest": str(BASELINE_MANIFEST),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protocol": {
            "baseline_retrained": False,
            "new_run_count": 3,
            "only_loss_changed": True,
            "raw_abs_S_weighting_used": True,
            "S_squared_weighting_used": False,
            "weight_clipped_capped_floored_or_transformed": False,
            "raw_energy_gradient_magnitude_used": False,
            "exact_K_used_in_training": False,
            "multistep_or_recursive_training_used": False,
            "architecture_dataset_split_optimizer_or_normalization_changed": False,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
