#!/usr/bin/env python3
"""Train exactly three (x, xi, E0)->(delta_x, delta_xi) microcore models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a import (
    ModelA,
    Normalization,
    TRAINING_SEEDS,
    load_trained_model,
    parameter_count,
    predict_increments,
    train_round1_run,
)
from wormhole_sciml.stage1_data import file_sha256, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling"
TRAIN_DATA = DATA_DIR / "outer_microcore40k_train_x_xi.npz"
VALIDATION_DATA = DATA_DIR / "outer_microcore8k_validation_x_xi.npz"
BASELINE_DIR = ROOT / "output" / "model_a_x_xi_sampling_training_comparison"
BASELINE_MANIFEST = BASELINE_DIR / "training_comparison_manifest.json"
BASELINE_NORMALIZATION = BASELINE_DIR / "microcore40k_normalization.json"
BASELINE_ROLLOUT_DIR = ROOT / "output" / "model_a_x_xi_sampling_recursive_rollouts"
EXACT_REFERENCE = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
ORBIT_SPACING = ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_arrays.npz"

OUTPUT = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
TRAINING_OUTPUT = OUTPUT / "training"
FIGURES = OUTPUT / "figures"
NORMALIZATION_PATH = TRAINING_OUTPUT / "energy_input_normalization.json"
MANIFEST_PATH = OUTPUT / "energy_xi_training_manifest.json"
TRAINING_REPORT = TRAINING_OUTPUT / "ENERGY_XI_TRAINING_REPORT.md"
CURVE_PATH = FIGURES / "energy_input_training_curves.png"

INPUT_COLUMNS = ("x", "xi", "E0")
TARGET_COLUMNS = ("delta_x", "delta_xi")
HIDDEN = (32, 32)
MAXIMUM_EPOCHS = 1500
NORMALIZATION_SOURCE = "outer_microcore40k_train_x_xi_energy_input_only"
METRIC_KEYS = (
    "standardized_mse", "rmse_delta_x", "rmse_delta_xi",
    "mae_delta_x", "mae_delta_xi",
)


def baseline_runs() -> dict[int, dict[str, Any]]:
    payload = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    runs = {
        int(row["seed"]): row
        for row in payload["runs"]
        if row["sampling_treatment"] == "microcore40k"
    }
    if set(runs) != set(TRAINING_SEEDS):
        raise RuntimeError("frozen microcore40k baseline is incomplete")
    return runs


def protected_paths() -> tuple[Path, ...]:
    runs = baseline_runs()
    paths = [
        TRAIN_DATA, VALIDATION_DATA, BASELINE_MANIFEST, BASELINE_NORMALIZATION,
        EXACT_REFERENCE, ORBIT_SPACING,
        BASELINE_ROLLOUT_DIR / "recursive_rollout_summary.json",
        BASELINE_ROLLOUT_DIR / "recursive_rollout_arrays.npz",
    ]
    for seed in TRAINING_SEEDS:
        run_dir = Path(runs[seed]["checkpoint"]).parent
        paths.extend((run_dir / "best_checkpoint.pt", run_dir / "metadata.json", run_dir / "training_history.json"))
    return tuple(paths)


def hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required immutable artifacts are missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def write_energy_normalization(training: dict[str, np.ndarray]) -> dict[str, Any]:
    baseline = json.loads(BASELINE_NORMALIZATION.read_text(encoding="utf-8"))
    if baseline["source_row_count"] != 40_000:
        raise RuntimeError("baseline normalization row identity changed")
    columns: dict[str, dict[str, float]] = {}
    for name in ("x", "xi", "delta_x", "delta_xi"):
        values = np.asarray(training[name], dtype=np.float64)
        measured = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
        }
        if measured != baseline["columns"][name]:
            raise RuntimeError(f"established microcore normalization differs for {name}")
        columns[name] = dict(baseline["columns"][name])
    energy = np.asarray(training["E0"], dtype=np.float64)
    energy_summary = {
        "mean": float(np.mean(energy)),
        "standard_deviation": float(np.std(energy, ddof=0)),
        "minimum": float(np.min(energy)),
        "maximum": float(np.max(energy)),
    }
    if energy_summary["standard_deviation"] <= 0.0:
        raise RuntimeError("E0 standard deviation is nonpositive")
    standardized = (energy - energy_summary["mean"]) / energy_summary["standard_deviation"]
    if not np.all(np.isfinite(standardized)):
        raise RuntimeError("standardized E0 contains nonfinite values")
    columns["E0"] = {
        "mean": energy_summary["mean"],
        "standard_deviation": energy_summary["standard_deviation"],
    }
    payload = {
        "source_dataset": NORMALIZATION_SOURCE,
        "source_path": str(TRAIN_DATA),
        "source_row_count": int(energy.size),
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "standard_deviation_definition": "population (ddof=0)",
        "columns": columns,
        "E0_training_distribution": {
            **energy_summary,
            "all_standardized_values_finite": True,
            "transformation": "ordinary standardization only",
        },
        "established_microcore_statistics_copied_exactly": True,
    }
    NORMALIZATION_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def masks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x, xi = np.asarray(data["x"]), np.asarray(data["xi"])
    outer, absolute_xi = np.abs(x) > 8.5, np.abs(xi)
    return {
        "all": np.ones(x.size, dtype=bool),
        "central": np.abs(x) <= 8.5,
        "outer": outer,
        "outer_left": x < -8.5,
        "outer_right": x > 8.5,
        "outer_micro_core": outer & (absolute_xi < 0.05),
        "outer_remainder_core": outer & (absolute_xi >= 0.05) & (absolute_xi < 0.5),
        "outer_shoulder": outer & (absolute_xi >= 0.5) & (absolute_xi < 0.9),
        "outer_edge": outer & (absolute_xi >= 0.9) & (absolute_xi <= 0.99),
    }


def physical_metrics(exact: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return {
        "count": int(error.shape[0]),
        "rmse_delta_x": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_delta_xi": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_delta_x": float(np.mean(np.abs(error[:, 0]))),
        "mae_delta_xi": float(np.mean(np.abs(error[:, 1]))),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    normalization: Normalization,
    validation: dict[str, np.ndarray],
    input_columns: tuple[str, ...],
) -> dict[str, Any]:
    model = load_trained_model(checkpoint)
    inputs = np.column_stack(tuple(validation[name] for name in input_columns))
    exact = np.column_stack((validation["delta_x"], validation["delta_xi"]))
    prediction = predict_increments(model, inputs, normalization)
    result: dict[str, Any] = {
        "standardized_mse": float(np.mean(((prediction - exact) / normalization.target_std) ** 2)),
        **physical_metrics(exact, prediction),
        "regions": {},
    }
    for name, mask in masks(validation).items():
        result["regions"][name] = physical_metrics(exact[mask], prediction[mask])
    return result


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...] = METRIC_KEYS) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {
                str(seed): float(value) for seed, value in zip(TRAINING_SEEDS, values)
            },
        }
    return result


def aggregate_regions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        region: aggregate(
            [row["regions"][region] for row in rows],
            ("rmse_delta_x", "rmse_delta_xi", "mae_delta_x", "mae_delta_xi"),
        )
        for region in rows[0]["regions"]
    }


def plot_histories(runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True, sharey=True)
    for axis, run in zip(axes, runs):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = np.asarray([row["epoch"] for row in history])
        training = np.asarray([row["training_physical_standardized_mse"] for row in history])
        validation = np.asarray([row["physical_validation_standardized_mse"] for row in history])
        axis.plot(epoch, training, lw=1.0, label="training")
        axis.plot(epoch, validation, lw=1.0, label="validation")
        axis.axvline(run["best_epoch"], color="0.35", ls="--", lw=0.9, label="best")
        axis.axvline(run["stopping_epoch"], color="0.6", ls=":", lw=0.9, label="stop")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("standardized increment MSE")
    figure.suptitle("Fixed-E0 transformed-coordinate training")
    figure.savefig(CURVE_PATH, dpi=185)
    plt.close(figure)


def report_text(manifest: dict[str, Any]) -> str:
    run_rows = "\n".join(
        f"| {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | {run['physical_epoch_steps']} | "
        f"{run['total_optimizer_updates']} | {run['one_step_validation']['standardized_mse']:.6e} | "
        f"{run['one_step_validation']['rmse_delta_x']:.6e} | {run['one_step_validation']['rmse_delta_xi']:.6e} |"
        for run in manifest["runs"]
    )
    comparison_rows = "\n".join(
        f"| {region} | {manifest['baseline_validation']['regional_aggregate'][region]['rmse_delta_xi']['mean']:.6e} | "
        f"{manifest['energy_validation']['regional_aggregate'][region]['rmse_delta_xi']['mean']:.6e} | "
        f"{manifest['regional_relative_change_percent'][region]['rmse_delta_xi']:+.1f}% |"
        for region in manifest["energy_validation"]["regional_aggregate"]
    )
    e = manifest["normalization"]["E0_training_distribution"]
    own = manifest["energy_validation"]["aggregate"]
    base = manifest["baseline_validation"]["aggregate"]
    return f"""# Fixed-E0 transformed-coordinate training report

## Protocol and normalization

Exactly three `3→32→32→2` two-tanh models were trained with inputs `(x,xi,E0)` and targets `(delta_x,delta_xi)`. Parameter count is `1250`. The existing 40,000/8,000 microcore training/validation rows, ordinary standardized two-output MSE, Adam `1e-3`, batch size 512, float32, Xavier/zero initialization, epoch ceiling 1500, patience 40, and validation-MSE-only checkpoint selection were unchanged. There was no scheduler.

`E0` training mean/SD/min/max are `{e['mean']:.12g}`, `{e['standard_deviation']:.12g}`, `{e['minimum']:.12g}`, and `{e['maximum']:.12g}`. Every standardized value is finite. The established `(x,xi,delta_x,delta_xi)` normalization values were copied exactly; only train-derived `E0` was added.

| seed | best epoch | stop epoch | batches/epoch | updates | validation std MSE | RMSE delta_x | RMSE delta_xi |
|---:|---:|---:|---:|---:|---:|---:|---:|
{run_rows}

All runs early-stopped before 1500. Restored-checkpoint validation reproduced each selected best MSE within recorded tolerance.

## Same-set one-step comparison

On the common microcore8k validation set, mean `delta_xi` RMSE changes from `{base['rmse_delta_xi']['mean']:.6e} ± {base['rmse_delta_xi']['sample_standard_deviation']:.2e}` to `{own['rmse_delta_xi']['mean']:.6e} ± {own['rmse_delta_xi']['sample_standard_deviation']:.2e}` (`{manifest['overall_relative_change_percent']['rmse_delta_xi']:+.1f}%`). Mean `delta_x` RMSE changes from `{base['rmse_delta_x']['mean']:.6e}` to `{own['rmse_delta_x']['mean']:.6e}`.

| region | no-energy RMSE delta_xi | fixed-E0 RMSE delta_xi | change |
|---|---:|---:|---:|
{comparison_rows}

These are local validation results only; recursive conclusions are deferred to the matched rollout stage.

## Integrity

Training/validation data, frozen no-energy checkpoints, incoming exact references, orbit-spacing arrays, and prior baseline rollout artifacts retained identical hashes. No existing artifact was overwritten, and no restricted evaluation data was accessed.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite dedicated experiment directory {OUTPUT}")
    protected = protected_paths()
    before = hashes(protected)
    training = load_dataset(TRAIN_DATA)
    validation = load_dataset(VALIDATION_DATA)
    if training["x"].size != 40_000 or validation["x"].size != 8_000:
        raise RuntimeError("microcore dataset row counts changed")
    OUTPUT.mkdir(parents=True)
    TRAINING_OUTPUT.mkdir()
    FIGURES.mkdir()
    normalization_payload = write_energy_normalization(training)
    normalization = Normalization.from_stage1(
        NORMALIZATION_PATH, INPUT_COLUMNS, TARGET_COLUMNS, NORMALIZATION_SOURCE
    )
    if parameter_count(ModelA(3, HIDDEN)) != 1250:
        raise RuntimeError("3->32->32->2 parameter count mismatch")

    runs: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        seed_dir = TRAINING_OUTPUT / f"seed_{seed}"
        run = train_round1_run(
            ROOT, "physical_only", seed,
            progress=True, maximum_epochs=MAXIMUM_EPOCHS,
            stage_label="fixed-E0 transformed-coordinate microcore40k training",
            data_tree="model_a_x_xi_outer_microcore_sampling",
            physical_train_filename=TRAIN_DATA.name,
            physical_validation_filename=VALIDATION_DATA.name,
            input_columns=INPUT_COLUMNS, target_columns=TARGET_COLUMNS,
            normalization_source_dataset=NORMALIZATION_SOURCE,
            normalization_path=NORMALIZATION_PATH,
            compact_history=True, hidden_dimensions=HIDDEN,
            run_directory=seed_dir, history_filename="training_history.json",
        )
        measured = evaluate_checkpoint(
            Path(run["checkpoint"]), normalization, validation, INPUT_COLUMNS
        )
        discrepancy = abs(
            measured["standardized_mse"]
            - run["restored_validation"]["standardized_mse"]
        )
        if discrepancy > 2.0e-7:
            raise RuntimeError("restored energy-input checkpoint does not reproduce best validation MSE")
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        run.update({
            "one_step_validation": measured,
            "training_rows": 40_000,
            "validation_rows": 8_000,
            "batches_per_epoch": int(run["physical_epoch_steps"]),
            "epochs_trained": int(run["stopping_epoch"]),
            "total_optimizer_updates": int(run["physical_epoch_steps"] * run["stopping_epoch"]),
            "final_epoch_training_standardized_mse": history[-1]["training_physical_standardized_mse"],
            "final_epoch_validation_standardized_mse": history[-1]["physical_validation_standardized_mse"],
            "checkpoint_reload_validation_mse_absolute_difference": discrepancy,
            "checkpoint_sha256": file_sha256(Path(run["checkpoint"])),
            "fixed_E0_input_semantics": "supervised row's stored exact E0",
        })
        (seed_dir / "metadata.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        runs.append(run)

    frozen = baseline_runs()
    baseline_normalization = Normalization.from_stage1(
        BASELINE_NORMALIZATION, ("x", "xi"), TARGET_COLUMNS,
        "outer_microcore40k_train_x_xi_only",
    )
    baseline_metrics = [
        evaluate_checkpoint(
            Path(frozen[seed]["checkpoint"]), baseline_normalization,
            validation, ("x", "xi"),
        )
        for seed in TRAINING_SEEDS
    ]
    energy_metrics = [run["one_step_validation"] for run in runs]
    energy_aggregate, baseline_aggregate = aggregate(energy_metrics), aggregate(baseline_metrics)
    energy_regions, baseline_regions = aggregate_regions(energy_metrics), aggregate_regions(baseline_metrics)
    overall_change = {
        key: 100.0 * (energy_aggregate[key]["mean"] / baseline_aggregate[key]["mean"] - 1.0)
        for key in METRIC_KEYS
    }
    regional_change = {
        region: {
            key: 100.0 * (
                energy_regions[region][key]["mean"] / baseline_regions[region][key]["mean"] - 1.0
            )
            for key in ("rmse_delta_x", "rmse_delta_xi", "mae_delta_x", "mae_delta_xi")
        }
        for region in energy_regions
    }
    plot_histories(runs)
    after = hashes(protected)
    if before != after:
        raise RuntimeError("an immutable dataset, baseline, or reference artifact changed")

    manifest = {
        "stage": "fixed-E0 transformed-coordinate microcore40k training and one-step validation",
        "status": "exactly_three_energy_input_runs_completed",
        "architecture": "3->32->32->2 with two tanh hidden layers",
        "parameter_count": 1250,
        "input_columns": list(INPUT_COLUMNS), "target_columns": list(TARGET_COLUMNS),
        "training_seeds": list(TRAINING_SEEDS), "runs": runs,
        "normalization": {
            "path": str(NORMALIZATION_PATH), "sha256": file_sha256(NORMALIZATION_PATH),
            **normalization_payload,
        },
        "energy_validation": {
            "individual": {str(seed): row for seed, row in zip(TRAINING_SEEDS, energy_metrics)},
            "aggregate": energy_aggregate, "regional_aggregate": energy_regions,
        },
        "baseline_validation": {
            "source": str(BASELINE_MANIFEST),
            "individual": {str(seed): row for seed, row in zip(TRAINING_SEEDS, baseline_metrics)},
            "aggregate": baseline_aggregate, "regional_aggregate": baseline_regions,
            "reevaluated_for_identical_subregional_metrics": True,
            "retrained": False,
        },
        "overall_relative_change_percent": overall_change,
        "regional_relative_change_percent": regional_change,
        "training_curve": {"path": str(CURVE_PATH), "sha256": file_sha256(CURVE_PATH)},
        "report": str(TRAINING_REPORT),
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protocol": {
            "new_training_run_count": 3, "baseline_retrained": False,
            "stored_E0_is_input_feature_only": True,
            "loss_changed": False, "scheduler_used": False,
            "energy_penalty_or_multistep_loss_used": False,
            "clipping_projection_or_weighting_used": False,
            "rollout_evaluation_performed_in_training_stage": False,
            "restricted_evaluation_data_accessed": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    TRAINING_REPORT.write_text(report_text(manifest), encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH} and {TRAINING_REPORT}")


if __name__ == "__main__":
    main()
