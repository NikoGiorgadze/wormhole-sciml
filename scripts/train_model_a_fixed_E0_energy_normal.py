#!/usr/bin/env python3
"""Train the three controlled fixed-E0 models with lambda_E=1 EN loss."""

from __future__ import annotations

import argparse
import json
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
BASELINE_ROLLOUT_SUMMARY = BASELINE / "energy_xi_rollout_summary.json"
BASELINE_ROLLOUT_ARRAYS = BASELINE / "energy_xi_rollout_arrays.npz"
SENSITIVITY_ARRAYS = (
    ROOT / "output" / "model_a_fixed_E0_throat_sensitivity" / "throat_sensitivity_arrays.npz"
)
ALIGNMENT_ARRAYS = (
    ROOT / "output" / "model_a_energy_gradient_alignment" / "energy_gradient_alignment_arrays.npz"
)
OUTPUT = ROOT / "output" / "model_a_fixed_E0_energy_normal_loss"
TRAINING = OUTPUT / "training"
DERIVED = OUTPUT / "derived_energy_normals.npz"
MANIFEST = OUTPUT / "energy_normal_training_manifest.json"

INPUT_COLUMNS = ("x", "xi", "E0")
TARGET_COLUMNS = ("delta_x", "delta_xi")
NORMALIZATION_SOURCE = "outer_microcore40k_train_x_xi_energy_input_only"
HIDDEN = (32, 32)
MAXIMUM_EPOCHS = 1500
LAMBDA_E = 1.0
BASELINE_FACTOR_C = 0.5
DEGENERACY_THRESHOLD = 1.0e-14


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
        BASELINE_ROLLOUT_SUMMARY,
        BASELINE_ROLLOUT_ARRAYS,
        SENSITIVITY_ARRAYS,
        ALIGNMENT_ARRAYS,
    ]
    for row in baseline_runs().values():
        paths.extend((Path(row["checkpoint"]), Path(row["history"]), Path(row["checkpoint"]).parent / "metadata.json"))
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
    if not np.allclose(
        data["x"] + data["delta_x"], data["x_next"], rtol=0.0, atol=2.0e-16
    ):
        raise RuntimeError("stored x_next does not match exact target increment")
    if not np.allclose(
        data["xi"] + data["delta_xi"], data["xi_next"], rtol=0.0, atol=2.0e-16
    ):
        raise RuntimeError("stored xi_next does not match exact target increment")
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
    train_normal, train_norm, train_degenerate = derive_normals(train, normalization)
    val_normal, val_norm, val_degenerate = derive_normals(validation, normalization)
    if np.any(train_degenerate) or np.any(val_degenerate):
        raise RuntimeError("a derived energy gradient is numerically degenerate")
    if not (np.all(np.isfinite(train_normal)) and np.all(np.isfinite(val_normal))):
        raise RuntimeError("a derived energy-normal direction is nonfinite")

    maximum_gate_difference = 0.0
    with np.load(ALIGNMENT_ARRAYS, allow_pickle=False) as gate:
        for u_th in (0.05, 0.15, 0.30):
            key = f"u_th_{u_th:.2f}".replace(".", "p")
            expected = np.asarray(gate[f"{key}__n_E"], dtype=np.float64)
            measured, _, degenerate = standardized_energy_normal(
                gate[f"{key}__post_x"],
                gate[f"{key}__post_xi"],
                float(normalization.target_std[0]),
                float(normalization.target_std[1]),
                degeneracy_threshold=DEGENERACY_THRESHOLD,
            )
            if np.any(degenerate):
                raise RuntimeError("alignment-gate state became degenerate")
            maximum_gate_difference = max(
                maximum_gate_difference, float(np.max(np.abs(measured - expected)))
            )
    if maximum_gate_difference > 2.0e-15:
        raise RuntimeError("training-side n_E does not reproduce the completed alignment gate")

    probes = train_normal[[0, train_normal.shape[0] // 2, -1]]
    geometry_rows: list[dict[str, float]] = []
    for normal in probes:
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
        tangent_dot = float(np.dot(tangent, normal))
        tangent_base = BASELINE_FACTOR_C * float(np.dot(tangent, tangent))
        tangent_perp = BASELINE_FACTOR_C * float(np.dot(normal, tangent) ** 2)
        normal_base = BASELINE_FACTOR_C * float(np.dot(normal, normal))
        normal_perp = BASELINE_FACTOR_C * float(np.dot(normal, normal) ** 2)
        geometry_rows.append(
            {
                "tangent_dot_normal": tangent_dot,
                "unit_tangent_base_loss": tangent_base,
                "unit_tangent_perpendicular_loss": tangent_perp,
                "unit_tangent_total_loss": tangent_base + LAMBDA_E * tangent_perp,
                "unit_normal_base_loss": normal_base,
                "unit_normal_perpendicular_loss": normal_perp,
                "unit_normal_total_loss": normal_base + LAMBDA_E * normal_perp,
            }
        )
    if any(
        abs(row["tangent_dot_normal"]) > 2.0e-15
        or abs(row["unit_tangent_total_loss"] - 0.5) > 2.0e-15
        or abs(row["unit_normal_total_loss"] - 1.0) > 2.0e-15
        for row in geometry_rows
    ):
        raise RuntimeError("energy-normal loss geometry preflight failed")
    arrays = {
        "train_n_E": train_normal,
        "train_g_E_norm": train_norm,
        "train_source_row_index": np.asarray(train["source_row_index"]),
        "train_x_next": np.asarray(train["x_next"]),
        "train_xi_next": np.asarray(train["xi_next"]),
        "validation_n_E": val_normal,
        "validation_g_E_norm": val_norm,
        "validation_source_row_index": np.asarray(validation["source_row_index"]),
        "validation_x_next": np.asarray(validation["x_next"]),
        "validation_xi_next": np.asarray(validation["xi_next"]),
    }
    summary = {
        "completed_alignment_gate_maximum_absolute_n_E_difference": maximum_gate_difference,
        "loss_geometry_probes": geometry_rows,
        "train": {
            "row_count": int(train_normal.shape[0]),
            "degenerate_count": int(np.sum(train_degenerate)),
            "minimum_g_E_norm": float(np.min(train_norm)),
            "maximum_g_E_norm": float(np.max(train_norm)),
        },
        "validation": {
            "row_count": int(val_normal.shape[0]),
            "degenerate_count": int(np.sum(val_degenerate)),
            "minimum_g_E_norm": float(np.min(val_norm)),
            "maximum_g_E_norm": float(np.max(val_norm)),
        },
    }
    return arrays, summary


@torch.no_grad()
def validation_metrics(
    model: ModelA, inputs: torch.Tensor, targets: torch.Tensor, normals: torch.Tensor
) -> dict[str, float]:
    prediction = predict_standardized(model, inputs)
    error = prediction - targets
    normal_error = torch.sum(error * normals, dim=1)
    tangent = torch.stack((-normals[:, 1], normals[:, 0]), dim=1)
    tangent_error = torch.sum(error * tangent, dim=1)
    base = torch.mean(error**2)
    perpendicular = 0.5 * torch.mean(normal_error**2)
    return {
        "standardized_mse": float(base.item()),
        "energy_normal_loss": float(perpendicular.item()),
        "energy_normal_total_loss": float((base + LAMBDA_E * perpendicular).item()),
        "rms_e_perp": float(torch.sqrt(torch.mean(normal_error**2)).item()),
        "mae_e_perp": float(torch.mean(torch.abs(normal_error)).item()),
        "rms_e_parallel": float(torch.sqrt(torch.mean(tangent_error**2)).item()),
        "mae_e_parallel": float(torch.mean(torch.abs(tangent_error)).item()),
    }


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
    validation_inputs, validation_targets = model_arrays(validation_data, normalization)
    train_normals = torch.from_numpy(np.asarray(derived["train_n_E"], dtype=np.float32))
    validation_normals = torch.from_numpy(
        np.asarray(derived["validation_n_E"], dtype=np.float32)
    )
    model = ModelA(3, HIDDEN).to(dtype=torch.float32)
    if parameter_count(model) != 1250:
        raise RuntimeError("energy-normal model is not 3->32->32->2")
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
        sum_squared = sum_normal_squared = sum_tangent_squared = 0.0
        sum_abs_normal = sum_abs_tangent = 0.0
        count = 0
        for batch_slice in slices:
            indices = order[batch_slice]
            prediction = model(train_inputs[indices])
            error = prediction - train_targets[indices]
            normal = train_normals[indices]
            projection = torch.sum(error * normal, dim=1)
            tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
            tangent_projection = torch.sum(error * tangent, dim=1)
            base_loss = torch.mean(error**2)
            perpendicular_loss = 0.5 * torch.mean(projection**2)
            objective = base_loss + LAMBDA_E * perpendicular_loss
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            detached = error.detach().to(dtype=torch.float64)
            detached_projection = projection.detach().to(dtype=torch.float64)
            detached_tangent = tangent_projection.detach().to(dtype=torch.float64)
            sum_squared += float(torch.sum(detached**2).item())
            sum_normal_squared += float(torch.sum(detached_projection**2).item())
            sum_tangent_squared += float(torch.sum(detached_tangent**2).item())
            sum_abs_normal += float(torch.sum(torch.abs(detached_projection)).item())
            sum_abs_tangent += float(torch.sum(torch.abs(detached_tangent)).item())
            count += int(error.shape[0])
        training_base = BASELINE_FACTOR_C * sum_squared / count
        training_perp = BASELINE_FACTOR_C * sum_normal_squared / count
        validation = validation_metrics(
            model, validation_inputs, validation_targets, validation_normals
        )
        history.append(
            {
                "epoch": epoch,
                "training_base_standardized_mse": training_base,
                "training_energy_normal_loss": training_perp,
                "training_energy_normal_total_loss": training_base + LAMBDA_E * training_perp,
                "training_rms_e_perp": float(np.sqrt(sum_normal_squared / count)),
                "training_mae_e_perp": sum_abs_normal / count,
                "training_rms_e_parallel": float(np.sqrt(sum_tangent_squared / count)),
                "training_mae_e_parallel": sum_abs_tangent / count,
                **{f"validation_{name}": value for name, value in validation.items()},
            }
        )
        if validation["standardized_mse"] < best_validation:
            best_validation = validation["standardized_mse"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"energy_normal seed={seed} epoch={epoch} val_base={validation['standardized_mse']:.8g} "
                f"val_perp={validation['energy_normal_loss']:.8g} best={best_validation:.8g}",
                flush=True,
            )
        if epochs_without_improvement >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    restored = validation_metrics(
        model, validation_inputs, validation_targets, validation_normals
    )
    if restored["standardized_mse"] != best_validation:
        raise RuntimeError("restored energy-normal checkpoint does not reproduce selection metric")
    run_dir = TRAINING / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    history_path = run_dir / "training_history.json"
    torch.save(best_state, checkpoint)
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "stage": "controlled fixed-E0 energy-normal loss training",
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
        "checkpoint_selection_metric": "ordinary physical_validation_standardized_increment_mse_only",
        "baseline_loss_reduction": "torch.mean(error**2) over batch and two outputs",
        "baseline_per_sample_factor_c": BASELINE_FACTOR_C,
        "lambda_E": LAMBDA_E,
        "objective": "c * (e^T e + lambda_E * (n_E^T e)^2)",
        "best_epoch": best_epoch,
        "stopping_epoch": int(history[-1]["epoch"]),
        "early_stopping_triggered": int(history[-1]["epoch"]) < MAXIMUM_EPOCHS,
        "best_validation_standardized_mse": best_validation,
        "selected_checkpoint_validation": restored,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "history": str(history_path),
        "normalization": str(NORMALIZATION_PATH),
        "physical_train_dataset": str(TRAIN_DATA),
        "physical_validation_dataset": str(VALIDATION_DATA),
        "derived_energy_normals": str(DERIVED),
        "sealed_data_used": False,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        NORMALIZATION_PATH,
        INPUT_COLUMNS,
        TARGET_COLUMNS,
        NORMALIZATION_SOURCE,
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
    derived_hash = file_sha256(DERIVED)
    runs = [
        train_seed(seed, normalization, train, validation, derived)
        for seed in TRAINING_SEEDS
    ]
    after = hashes(protected)
    if before != after:
        raise RuntimeError("a protected baseline, data, sensitivity, or gate artifact changed")
    manifest: dict[str, Any] = {
        "stage": "controlled fixed-E0 energy-normal loss training",
        "status": "exactly_three_energy_normal_runs_completed",
        "runs": runs,
        "training_seeds": list(TRAINING_SEEDS),
        "architecture": "3->32->32->2 with two tanh hidden layers",
        "parameter_count": 1250,
        "lambda_E": LAMBDA_E,
        "baseline_per_sample_factor_c": BASELINE_FACTOR_C,
        "loss": "c * (e^T e + lambda_E * (n_E^T e)^2)",
        "checkpoint_selection_metric": "ordinary validation standardized MSE, unchanged from fixed-E0 baseline",
        "preflight": preflight_summary,
        "derived_energy_normals": {
            "path": str(DERIVED),
            "sha256": derived_hash,
            "source_train_path": str(TRAIN_DATA),
            "source_train_sha256": before[str(TRAIN_DATA)],
            "source_validation_path": str(VALIDATION_DATA),
            "source_validation_sha256": before[str(VALIDATION_DATA)],
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
            "sensitivity_weighting_used": False,
            "raw_energy_residual_loss_used": False,
            "multistep_or_recursive_training_used": False,
            "architecture_dataset_split_optimizer_or_normalization_changed": False,
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
