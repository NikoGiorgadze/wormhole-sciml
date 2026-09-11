#!/usr/bin/env python3
"""Train the three physical-only Model-A runs in ``(x, xi)`` coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a import TRAINING_SEEDS, train_round1_run
from wormhole_sciml.stage1_data import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TREE = "stage1_model_a_x_xi"
OUTPUT_TREE = "model_a_x_xi_physical_1000"
OUTPUT_DIR = PROJECT_ROOT / "output" / OUTPUT_TREE
BASELINE_MANIFEST = (
    PROJECT_ROOT / "output" / "round1_model_a_1000" / "training_manifest.json"
)
MAXIMUM_EPOCHS = 1000
INPUT_COLUMNS = ("x", "xi")
TARGET_COLUMNS = ("delta_x", "delta_xi")


def artifact_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths}


def baseline_runs() -> dict[int, dict[str, Any]]:
    payload = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    rows = {
        int(row["seed"]): row
        for row in payload["runs"]
        if row["treatment"] == "physical_only"
    }
    if set(rows) != set(TRAINING_SEEDS):
        raise RuntimeError("the physical-only 1000-epoch baseline is incomplete")
    return rows


def plot_histories(runs: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {101: "#277da1", 202: "#7b2cbf", 303: "#2a9d8f"}
    for run in runs:
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = np.asarray([row["epoch"] for row in history])
        train = np.asarray(
            [row["training_physical_standardized_mse"] for row in history]
        )
        validation = np.asarray(
            [row["physical_validation_standardized_mse"] for row in history]
        )
        axes[0].plot(epoch, train, color=colors[run["seed"]], alpha=0.55)
        axes[0].plot(
            epoch,
            validation,
            color=colors[run["seed"]],
            label=f"seed {run['seed']}",
        )
        axes[1].plot(
            epoch,
            [row["physical_validation_rmse_delta_x"] for row in history],
            color=colors[run["seed"]],
        )
        axes[1].plot(
            epoch,
            [row["physical_validation_rmse_delta_xi"] for row in history],
            color=colors[run["seed"]],
            linestyle="--",
        )
    axes[0].set(
        xlabel="epoch", ylabel="standardized MSE", yscale="log", title="Train (faint) and validation"
    )
    axes[0].legend()
    axes[1].set(
        xlabel="epoch", ylabel="unstandardized RMSE", yscale="log",
        title=r"Validation RMSE: $\Delta x$ solid, $\Delta\xi$ dashed",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    path = OUTPUT_DIR / "training_validation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    data_dir = PROJECT_ROOT / "output" / DATA_TREE
    protected = [
        data_dir / "physical_train_x_xi.npz",
        data_dir / "physical_validation_x_xi.npz",
        data_dir / "normalization.json",
        data_dir / "manifest.json",
        BASELINE_MANIFEST,
    ]
    baselines = baseline_runs()
    protected.extend(Path(baselines[seed]["checkpoint"]) for seed in TRAINING_SEEDS)
    hashes_before = artifact_hashes(protected)

    completed = []
    protocol_fields = (
        "architecture", "parameter_count", "initialization", "dtype", "optimizer",
        "learning_rate", "weight_decay", "maximum_epochs", "early_stopping_patience",
        "checkpoint_selection_metric", "physical_batch_size", "physical_epoch_steps",
        "physical_order_scheme",
    )
    for seed in TRAINING_SEEDS:
        run = train_round1_run(
            PROJECT_ROOT,
            "physical_only",
            seed,
            maximum_epochs=MAXIMUM_EPOCHS,
            output_tree=OUTPUT_TREE,
            stage_label="Model-A physical-only (x,xi) 1000-epoch experiment",
            data_tree=DATA_TREE,
            physical_train_filename="physical_train_x_xi.npz",
            physical_validation_filename="physical_validation_x_xi.npz",
            input_columns=INPUT_COLUMNS,
            target_columns=TARGET_COLUMNS,
            normalization_source_dataset="physical_train_x_xi",
        )
        baseline = baselines[seed]
        for field in protocol_fields:
            if run[field] != baseline[field]:
                raise RuntimeError(f"protocol differs from baseline for {field}")
        if run["initial_state_sha256"] != baseline["initial_state_sha256"]:
            raise RuntimeError("same-seed initialization differs from baseline")
        if run["collar_dataset"] is not None:
            raise RuntimeError("collar data entered a physical-only run")
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        completed.append(run)

    hashes_after = artifact_hashes(protected)
    if hashes_before != hashes_after:
        raise RuntimeError("a transformed-data or baseline artifact changed")
    curve_path = plot_histories(completed)
    metric_keys = (
        "standardized_mse", "physical_rmse_delta_x", "physical_rmse_delta_xi"
    )
    aggregate = {}
    for key in metric_keys:
        values = np.asarray([run["restored_validation"][key] for run in completed])
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {
                str(run["seed"]): float(run["restored_validation"][key])
                for run in completed
            },
        }
    manifest = {
        "stage": "Model-A physical-only (x,xi) 1000-epoch training and basic validation",
        "status": "three_runs_completed",
        "scientific_change_from_baseline": {
            "only_change": "coordinate representation",
            "old": "(x,u)->(delta_x,delta_u)",
            "new": "(x,xi)->(delta_x,delta_xi)",
        },
        "seeds": list(TRAINING_SEEDS),
        "run_count": len(completed),
        "runs": completed,
        "baseline_protocol_audit_passed": True,
        "same_seed_initialization_audit_passed": True,
        "validation_aggregate": aggregate,
        "training_curve": str(curve_path),
        "protected_artifact_hashes_before": hashes_before,
        "protected_artifact_hashes_after": hashes_after,
        "protocol": {
            "model_unconstrained": True,
            "collar_data_used": False,
            "rollout_evaluation_performed": False,
            "dense_grid_evaluation_performed": False,
            "restricted_data_accessed": False,
            "scheduler_used": False,
        },
    }
    path = OUTPUT_DIR / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
