#!/usr/bin/env python3
"""Freeze Phase-D preprocessing and train the three-seed Phase-E baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wormhole_sciml.finite_time import (
    BATCH_SIZE,
    EXPECTED_PARAMETER_COUNT,
    HIDDEN_DIMENSIONS,
    INPUT_COLUMNS,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PATIENCE,
    PREPROCESSING_IMPLEMENTATION,
    PREPROCESSING_SCHEMA,
    TARGET_COLUMNS,
    TRAINING_SEEDS,
    FiniteTimePreprocessing,
    evaluate_validation,
    standardized_range_audit,
    train_seed,
)
from wormhole_sciml.model_a import ModelA, parameter_count
from wormhole_sciml.phase_c_finite_time import array_content_sha256, load_dataset
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
PHASE_C = ROOT / "output" / "phase_c_finite_time_dataset"
PHASE_C_MANIFEST = PHASE_C / "phase_c_manifest.json"
DATASETS = {
    "train": PHASE_C / "datasets" / "phase_c_train_raw.npz",
    "validation": PHASE_C / "datasets" / "phase_c_validation_raw.npz",
    "test": PHASE_C / "datasets" / "phase_c_test_sealed_raw.npz",
}
EXPECTED = {
    "train": (393_216, 4_096),
    "validation": (98_304, 1_024),
    "test": (98_304, 1_024),
}
OUTPUT = ROOT / "output" / "finite_time_baseline"
PREPROCESSING_DIR = OUTPUT / "preprocessing"
TRAINING_DIR = OUTPUT / "training"
VALIDATION_DIR = OUTPUT / "validation"
FIGURES_DIR = OUTPUT / "figures"
TESTS_DIR = OUTPUT / "tests"
CONSTANTS = PREPROCESSING_DIR / "preprocessing_constants.json"
AUDIT = PREPROCESSING_DIR / "standardized_range_audit.json"
PREPROCESSING_MANIFEST = PREPROCESSING_DIR / "preprocessing_manifest.json"
PREPROCESSING_MANIFEST_HASH = PREPROCESSING_DIR / "preprocessing_manifest.sha256"
GATE = OUTPUT / "phase_c_integrity_gate.json"
CONFIG = TRAINING_DIR / "training_config.json"
ENVIRONMENT = TRAINING_DIR / "environment.json"
METRICS = VALIDATION_DIR / "per_seed_validation_metrics.json"
METRICS_CSV = VALIDATION_DIR / "per_seed_summary.csv"
SEED_COMPARISON = VALIDATION_DIR / "seed_comparison.json"
CURVES = FIGURES_DIR / "training_histories.png"
MANIFEST = OUTPUT / "finite_time_baseline_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_baseline_manifest.sha256"
REPORT = OUTPUT / "PHASE_D_E_FINITE_TIME_BASELINE_REPORT.md"
PHASE_D_REPORT = OUTPUT / "PHASE_D_PREPROCESSING_REPORT.md"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_phase_c_gate() -> dict[str, Any]:
    manifest = json.loads(PHASE_C_MANIFEST.read_text(encoding="utf-8"))
    loaded: dict[str, dict[str, np.ndarray]] = {}
    result: dict[str, Any] = {"datasets": {}, "split_intersections": {}}
    failures: list[str] = []
    for name, path in DATASETS.items():
        expected_rows, expected_orbits = EXPECTED[name]
        expected_manifest = manifest["datasets"][name]
        measured_file_hash = file_sha256(path)
        data = load_dataset(path)
        loaded[name] = data
        ids, counts = np.unique(data["orbit_id"], return_counts=True)
        content_hash = array_content_sha256(data)
        checks = {
            "file_sha256_matches_manifest": measured_file_hash == expected_manifest["file_sha256"],
            "content_sha256_matches_manifest": content_hash == expected_manifest["content_sha256"],
            "row_count_matches": len(data["orbit_id"]) == expected_rows,
            "orbit_count_matches": len(ids) == expected_orbits,
            "exactly_96_rows_per_orbit": bool(np.all(counts == 96)),
        }
        if not all(checks.values()):
            failures.append(name)
        result["datasets"][name] = {
            "path": str(path.resolve()),
            "file_sha256": measured_file_hash,
            "content_sha256": content_hash,
            "row_count": int(len(data["orbit_id"])),
            "orbit_count": int(len(ids)),
            "rows_per_orbit_minimum": int(np.min(counts)),
            "rows_per_orbit_maximum": int(np.max(counts)),
            "checks": checks,
        }
    sets = {name: set(data["orbit_id"].tolist()) for name, data in loaded.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        count = len(sets[left] & sets[right])
        result["split_intersections"][f"{left}_{right}"] = count
        if count:
            failures.append(f"{left}_{right}_orbit_id_overlap")
    result.update({
        "phase_c_manifest": str(PHASE_C_MANIFEST.resolve()),
        "phase_c_manifest_sha256": file_sha256(PHASE_C_MANIFEST),
        "failures": failures,
        "passed": not failures,
        "sealed_test_use": "hash/count/orbit-ID integrity only; no preprocessing fit or predictions",
    })
    return result


def write_phase_d(training: dict[str, np.ndarray], gate: dict[str, Any]) -> tuple[FiniteTimePreprocessing, dict[str, Any]]:
    training_hash = gate["datasets"]["train"]["file_sha256"]
    fitted = FiniteTimePreprocessing.fit(training, training_hash)
    payload = fitted.payload(DATASETS["train"], len(training["orbit_id"]))
    audit = standardized_range_audit(training, fitted)
    if CONSTANTS.exists():
        existing = json.loads(CONSTANTS.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing frozen preprocessing differs from a training-only refit")
    else:
        write_json(CONSTANTS, payload)
    if AUDIT.exists():
        existing_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if existing_audit != audit:
            raise RuntimeError("existing standardized audit differs from reproducible refit")
    else:
        write_json(AUDIT, audit)
    preprocessing_manifest = {
        "phase": "D_finite_time_preprocessing",
        "decision": "FAIL" if audit["obvious_numerical_pathology"] else "PASS",
        "schema": PREPROCESSING_SCHEMA,
        "implementation": PREPROCESSING_IMPLEMENTATION,
        "input_order": list(INPUT_COLUMNS),
        "target_order": list(TARGET_COLUMNS),
        "source_split": "train only",
        "source_training_dataset_sha256": training_hash,
        "validation_constants_policy": "reuse frozen training constants identically",
        "sealed_test_used_for_fit_or_distribution_inspection": False,
        "constants": {"path": str(CONSTANTS.resolve()), "sha256": file_sha256(CONSTANTS)},
        "standardized_audit": {"path": str(AUDIT.resolve()), "sha256": file_sha256(AUDIT)},
        "phase_c_integrity_gate": {"path": str(GATE.resolve()), "sha256": file_sha256(GATE)},
    }
    write_json(PREPROCESSING_MANIFEST, preprocessing_manifest)
    PREPROCESSING_MANIFEST_HASH.write_text(
        f"{file_sha256(PREPROCESSING_MANIFEST)}  {PREPROCESSING_MANIFEST.name}\n",
        encoding="utf-8",
    )
    return fitted, audit


def phase_d_report(preprocessing: FiniteTimePreprocessing, audit: dict[str, Any]) -> str:
    names = INPUT_COLUMNS + TARGET_COLUMNS
    means = np.concatenate((preprocessing.input_mean, preprocessing.target_mean))
    stds = np.concatenate((preprocessing.input_std, preprocessing.target_std))
    constant_rows = "\n".join(
        f"| {name} | {mean:.12g} | {std:.12g} |"
        for name, mean, std in zip(names, means, stds)
    )
    audit_rows = "\n".join(
        f"| {name} | {row['mean']:.3e} | {row['standard_deviation']:.6g} | "
        f"{row['minimum']:.6g} | {row['p0.1']:.6g} | {row['p1']:.6g} | "
        f"{row['p5']:.6g} | {row['median']:.6g} | {row['p95']:.6g} | "
        f"{row['p99']:.6g} | {row['p99.9']:.6g} | {row['maximum']:.6g} |"
        for name, row in audit["columns"].items()
    )
    decision = "FAIL" if audit["obvious_numerical_pathology"] else "PASS"
    return f"""# Phase D finite-time preprocessing report

## Decision

**PHASE D: {decision}.** Constants were fitted from the frozen training split only using float64 population mean/std. Physical `s` is the fourth input; `horizon_fraction` and all metadata are excluded. The largest absolute standardized training value is `{audit['maximum_absolute_standardized_value']:.6g}`, below the predeclared stop threshold of `25`.

## Frozen constants

| variable | mean | population standard deviation |
|:---|---:|---:|
{constant_rows}

Input order is exactly `[x0, xi0, E0, s]`; target order is exactly `[Delta_x, Delta_xi]`.

## Standardized training-range audit

| variable | mean | std | min | p0.1 | p1 | p5 | median | p95 | p99 | p99.9 | max |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{audit_rows}

The sealed test values were not used for fitting or distribution inspection. Its artifact was accessed only for the required integrity gate.
"""


def environment_payload() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "cpu_count": os.cpu_count(),
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
        "device": "cpu",
    }


def plot_histories(runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True, sharey=True)
    for axis, run in zip(axes, runs):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = np.asarray([row["epoch"] for row in history])
        training = np.asarray([row["training_standardized_mse"] for row in history])
        validation = np.asarray([row["validation_orbit_averaged_standardized_mse"] for row in history])
        axis.plot(epoch, training, lw=1.0, label="training")
        axis.plot(epoch, validation, lw=1.0, label="validation")
        axis.axvline(run["best_epoch"], color="0.25", ls="--", lw=0.9, label="best")
        axis.axvline(run["stopping_epoch"], color="0.55", ls=":", lw=0.9, label="stop")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("standardized two-target MSE")
    figure.suptitle("Direct finite-time baseline training histories")
    figure.savefig(CURVES, dpi=190)
    plt.close(figure)


def trend_analysis(run: dict[str, Any]) -> dict[str, Any]:
    history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
    validation = np.asarray([row["validation_orbit_averaged_standardized_mse"] for row in history])
    width = min(20, validation.size)
    relative_change = float((validation[-width] - validation[-1]) / validation[-width]) if width > 1 else 0.0
    return {
        "epochs_compared": width,
        "relative_validation_improvement_over_final_window": relative_change,
        "ceiling_reached": run["ceiling_reached"],
        "materially_improving_at_ceiling": bool(run["ceiling_reached"] and relative_change > 0.01),
        "classification": (
            "ceiling_while_materially_improving" if run["ceiling_reached"] and relative_change > 0.01
            else "ceiling_not_materially_improving" if run["ceiling_reached"]
            else "early_stopped_converged_or_overfit"
        ),
    }


def metric_value(metrics: dict[str, Any], component: str, key: str) -> float:
    return float(metrics["physical_residual_metrics"][component][key])


def make_seed_comparison(runs: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for run in runs:
        seed_metrics = metrics[str(run["seed"])]
        hard = seed_metrics["hard_family_breakdown"]["hard_u_th_le_0p30"]["metrics"]
        long_time = seed_metrics["time_regime_breakdown"]["long_s_gt_20"]["metrics"]
        rows.append({
            "seed": run["seed"],
            "best_validation_standardized_mse": run["best_validation_orbit_averaged_standardized_mse"],
            "rmse_Delta_x": metric_value(seed_metrics, "Delta_x", "rmse"),
            "mae_Delta_x": metric_value(seed_metrics, "Delta_x", "mae"),
            "p90_Delta_x": seed_metrics["physical_residual_metrics"]["Delta_x"]["absolute_error"]["p90"],
            "p99_Delta_x": seed_metrics["physical_residual_metrics"]["Delta_x"]["absolute_error"]["p99"],
            "rmse_Delta_xi": metric_value(seed_metrics, "Delta_xi", "rmse"),
            "mae_Delta_xi": metric_value(seed_metrics, "Delta_xi", "mae"),
            "p90_Delta_xi": seed_metrics["physical_residual_metrics"]["Delta_xi"]["absolute_error"]["p90"],
            "p99_Delta_xi": seed_metrics["physical_residual_metrics"]["Delta_xi"]["absolute_error"]["p99"],
            "hard_rmse_Delta_x": hard["Delta_x"]["rmse"],
            "hard_rmse_Delta_xi": hard["Delta_xi"]["rmse"],
            "long_rmse_Delta_x": long_time["Delta_x"]["rmse"],
            "long_rmse_Delta_xi": long_time["Delta_xi"]["rmse"],
            "identity": seed_metrics["identity_diagnostics"],
        })
    validation_values = np.asarray([row["best_validation_standardized_mse"] for row in rows])
    best_index = int(np.argmin(validation_values))
    median = float(np.median(validation_values))
    outliers = [row["seed"] for row in rows if row["best_validation_standardized_mse"] > 1.5 * median]
    return {
        "rows": rows,
        "best_validation_seed": int(rows[best_index]["seed"]),
        "best_validation_metric": float(validation_values[best_index]),
        "validation_metric_mean": float(np.mean(validation_values)),
        "validation_metric_sample_standard_deviation": float(np.std(validation_values, ddof=1)),
        "validation_metric_range": [float(np.min(validation_values)), float(np.max(validation_values))],
        "obvious_outlier_rule": "best validation loss greater than 1.5 times the three-seed median",
        "obvious_outlier_seeds": outliers,
        "seed_discarded": False,
    }


def write_metrics_csv(comparison: dict[str, Any]) -> None:
    rows = comparison["rows"]
    fields = [name for name in rows[0] if name != "identity"]
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def hashes_for_artifacts(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(OUTPUT)): {
            "path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
        for path in paths
    }


def report_text(
    preprocessing: FiniteTimePreprocessing,
    audit: dict[str, Any],
    runs: list[dict[str, Any]],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
) -> str:
    constants = phase_d_report(preprocessing, audit).split("## Frozen constants\n\n", 1)[1].split("\n\n## Standardized", 1)[0]
    audit_table = phase_d_report(preprocessing, audit).split("## Standardized training-range audit\n\n", 1)[1].split("\n\nThe sealed", 1)[0]
    training_rows = "\n".join(
        f"| {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | "
        f"{run['best_validation_orbit_averaged_standardized_mse']:.7g} | "
        f"{run['final_training_standardized_mse']:.7g} | {run['checkpoint_sha256']} |"
        for run in runs
    )
    physical_rows = "\n".join(
        f"| {run['seed']} | {metric_value(metrics[str(run['seed'])], 'Delta_x', 'rmse'):.6g} | "
        f"{metric_value(metrics[str(run['seed'])], 'Delta_x', 'mae'):.6g} | "
        f"{metric_value(metrics[str(run['seed'])], 'Delta_xi', 'rmse'):.6g} | "
        f"{metric_value(metrics[str(run['seed'])], 'Delta_xi', 'mae'):.6g} |"
        for run in runs
    )
    percentile_rows = "\n".join(
        f"| {run['seed']} | {component} | " + " | ".join(
            f"{metrics[str(run['seed'])]['physical_residual_metrics'][component]['absolute_error'][key]:.6g}"
            for key in ("median", "p90", "p95", "p99", "maximum")
        ) + " |"
        for run in runs for component in TARGET_COLUMNS
    )
    breakdown_rows = []
    for run in runs:
        seed = str(run["seed"])
        for category, groups in (
            ("time", metrics[seed]["time_regime_breakdown"]),
            ("family", metrics[seed]["hard_family_breakdown"]),
        ):
            for group, entry in groups.items():
                value = entry["metrics"]
                breakdown_rows.append(
                    f"| {seed} | {category} | {group} | {entry['row_count']} | "
                    f"{value['Delta_x']['rmse']:.6g} | {value['Delta_xi']['rmse']:.6g} |"
                )
    identity_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['identity_diagnostics']['row_count']} | "
        f"{metrics[str(run['seed'])]['identity_diagnostics']['max_abs_Delta_x_prediction']:.6g} | "
        f"{metrics[str(run['seed'])]['identity_diagnostics']['max_abs_Delta_xi_prediction']:.6g} | "
        f"{metrics[str(run['seed'])]['identity_diagnostics']['rmse_Delta_x']:.6g} | "
        f"{metrics[str(run['seed'])]['identity_diagnostics']['rmse_Delta_xi']:.6g} |"
        for run in runs
    )
    diagnostic_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['physical_diagnostics']['absolute_xi_ge_1']['count']} "
        f"({metrics[str(run['seed'])]['physical_diagnostics']['absolute_xi_ge_1']['fraction']:.3%}) | "
        f"{metrics[str(run['seed'])]['physical_diagnostics']['C_le_0']['count']} "
        f"({metrics[str(run['seed'])]['physical_diagnostics']['C_le_0']['fraction']:.3%}) | "
        f"{metrics[str(run['seed'])]['physical_diagnostics']['energy_error_E_hat_minus_E0']['rmse']:.6g} | "
        f"{metrics[str(run['seed'])]['physical_diagnostics']['energy_error_E_hat_minus_E0']['p99_absolute']:.6g} | "
        f"{metrics[str(run['seed'])]['physical_diagnostics']['energy_error_E_hat_minus_E0']['maximum_absolute']:.6g} |"
        for run in runs
    )
    outliers = comparison["obvious_outlier_seeds"] or "none"
    return f"""# Phase D/E direct finite-time baseline report

## Decisions

**PHASE D: PASS preprocessing.** Training-only componentwise mean/std constants are finite, frozen, reproducible, and numerically reasonable.

**PHASE E: PASS baseline training.** All three prescribed seeds trained and restored successfully with finite validation metrics. This is an implementation/reproducibility decision, not a claim of trajectory-level scientific sufficiency.

## Phase C integrity and scope

All frozen raw file/content hashes matched the Phase C manifest; row/orbit counts were `393216/4096`, `98304/1024`, and `98304/1024`; every orbit had exactly 96 rows; all split orbit-ID intersections were empty. Sealed test access was limited to these integrity checks. No test prediction, loss, error, plot, or preprocessing statistic was computed.

## Preprocessing constants

{constants}

## Standardized training audit

{audit_table}

## Fixed training protocol

Exact architecture: `4→64→64→2`, two tanh hidden layers, no output activation, `{EXPECTED_PARAMETER_COUNT}` parameters. Xavier-uniform weights and zero biases; equal standardized two-output MSE; Adam `lr={LEARNING_RATE:g}`, batch `{BATCH_SIZE}`, weight decay `0`; no scheduler; float32 model tensors; shuffled row batches without resampling; maximum `{MAXIMUM_EPOCHS}` epochs; patience `{PATIENCE}`; seeds `{list(TRAINING_SEEDS)}`. Checkpoint selection used validation-only orbit-averaged standardized MSE. The established MLP, initialization, state hashing, batched inference, deterministic settings, Adam/early-stop/checkpoint conventions, JSON manifests, and plotting style were reused; recursive/collar/physics-loss code was not.

| seed | best epoch | stop epoch | best validation std MSE | final training std MSE | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|:---|
{training_rows}

Best validation seed: `{comparison['best_validation_seed']}`. Three-seed validation mean ± sample SD: `{comparison['validation_metric_mean']:.7g} ± {comparison['validation_metric_sample_standard_deviation']:.3g}`; range `{comparison['validation_metric_range']}`. Obvious outliers under the fixed 1.5×-median rule: `{outliers}`. No seed was discarded.

## Physical-unit validation metrics

Residual and reconstructed-state errors agreed componentwise for every seed (maximum discrepancies are recorded in the metrics JSON). Thus `Delta_x` metrics equal `x1` metrics and `Delta_xi` metrics equal `xi1` metrics.

| seed | RMSE Delta_x / x1 | MAE Delta_x / x1 | RMSE Delta_xi / xi1 | MAE Delta_xi / xi1 |
|---:|---:|---:|---:|---:|
{physical_rows}

| seed | component | median abs | p90 abs | p95 abs | p99 abs | max abs |
|---:|:---|---:|---:|---:|---:|---:|
{percentile_rows}

## Validation breakdowns

The Phase C physical-time convention is fixed in advance: short `s≤5`, intermediate `5<s≤20`, long `s>20`. Hard family is the physical label `u_th≤0.30`.

| seed | category | group | rows | RMSE Delta_x | RMSE Delta_xi |
|---:|:---|:---|---:|---:|---:|
{chr(10).join(breakdown_rows)}

Sample-group and all four hard-targeted subtype breakdowns are in `validation/per_seed_validation_metrics.json`.

## Identity diagnostics

| seed | rows | max abs predicted Delta_x | max abs predicted Delta_xi | RMSE Delta_x | RMSE Delta_xi |
|---:|---:|---:|---:|---:|---:|
{identity_rows}

These are diagnostics only; no identity-specific loss was used.

## Limited physical diagnostics

| seed | abs(xi_hat)≥1 | C≤0 | energy-error RMSE | energy abs p99 | energy abs max |
|---:|:---|:---|---:|---:|---:|
{diagnostic_rows}

These quantities were computed after validation prediction and were not used in training or model selection.

## Artifacts and next step

The full machine-readable record is `finite_time_baseline_manifest.json`; checkpoints/histories are under `training/seed_*`; validation tables under `validation/`; the history plot is `figures/training_histories.png`; test evidence is under `tests/`. Artifact SHA-256 values are recorded in the manifest and companion `.sha256` files.

Recommendation: proceed to the separately scoped full trajectory-level validation only after reviewing the validation breakdowns and physical-violation counts above. Do not unseal the test set within Phase D/E.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_DIR.mkdir(exist_ok=True)
    TESTS_DIR.mkdir(exist_ok=True)
    gate = verify_phase_c_gate()
    write_json(GATE, gate)
    if not gate["passed"]:
        raise RuntimeError(f"Phase C integrity gate failed: {gate['failures']}")
    raw_hashes_before = {name: file_sha256(path) for name, path in DATASETS.items()}
    training = load_dataset(DATASETS["train"])
    preprocessing, audit = write_phase_d(training, gate)
    PHASE_D_REPORT.write_text(phase_d_report(preprocessing, audit), encoding="utf-8")
    if audit["obvious_numerical_pathology"]:
        raise RuntimeError("Phase D standardized-range audit failed; stopping before training")
    if args.preprocess_only:
        print(f"PHASE D PASS: wrote {PHASE_D_REPORT}")
        return

    if TRAINING_DIR.exists() and any(TRAINING_DIR.glob("seed_*")):
        raise FileExistsError("finite-time seed outputs already exist; refusing to overwrite")
    TRAINING_DIR.mkdir(exist_ok=True)
    VALIDATION_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    validation = load_dataset(DATASETS["validation"])
    training_config = {
        "architecture": [4, 64, 64, 2], "activation": "tanh", "output_activation": None,
        "parameter_count": EXPECTED_PARAMETER_COUNT, "input_order": list(INPUT_COLUMNS),
        "target_order": list(TARGET_COLUMNS), "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "weight_decay": 0.0, "scheduler": None,
        "maximum_epochs": MAXIMUM_EPOCHS, "early_stopping_patience": PATIENCE,
        "seeds": list(TRAINING_SEEDS), "model_dtype": "float32",
        "loss": "mean of squared standardized error over rows and the two output components",
        "checkpoint_metric": "validation orbit-averaged standardized MSE",
        "sampling": "ordinary shuffled row minibatches; no oversampling or weighting",
        "sealed_test_prediction_evaluation": False,
    }
    write_json(CONFIG, training_config)
    write_json(ENVIRONMENT, environment_payload())
    if parameter_count(ModelA(4, HIDDEN_DIMENSIONS)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("programmatic parameter count gate failed")

    runs: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        run = train_seed(training, validation, preprocessing, seed, TRAINING_DIR / f"seed_{seed}")
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        run["history_sha256"] = file_sha256(Path(run["history"]))
        run["training_trend"] = trend_analysis(run)
        measured = evaluate_validation(Path(run["checkpoint"]), validation, preprocessing)
        discrepancy = abs(
            measured["standardized_validation"]["orbit_averaged_standardized_mse"]
            - run["best_validation_orbit_averaged_standardized_mse"]
        )
        if discrepancy > 2.0e-7:
            raise RuntimeError("checkpoint restore validation metric discrepancy is too large")
        if not measured["residual_error_equals_reconstructed_state_error"]:
            raise RuntimeError("residual and reconstructed-state validation errors differ")
        run["checkpoint_reload_metric_absolute_difference"] = discrepancy
        metrics_path = TRAINING_DIR / f"seed_{seed}" / "validation_metrics.json"
        metadata_path = TRAINING_DIR / f"seed_{seed}" / "metadata.json"
        write_json(metrics_path, measured)
        run["validation_metrics"] = str(metrics_path.resolve())
        write_json(metadata_path, run)
        per_seed[str(seed)] = measured
        runs.append(run)

    write_json(METRICS, per_seed)
    comparison = make_seed_comparison(runs, per_seed)
    write_json(SEED_COMPARISON, comparison)
    write_metrics_csv(comparison)
    plot_histories(runs)
    raw_hashes_after = {name: file_sha256(path) for name, path in DATASETS.items()}
    if raw_hashes_before != raw_hashes_after:
        raise RuntimeError("a frozen Phase C raw artifact changed during Phase D/E")

    REPORT.write_text(report_text(preprocessing, audit, runs, per_seed, comparison), encoding="utf-8")
    artifact_paths = [
        CONSTANTS, AUDIT, PREPROCESSING_MANIFEST, PREPROCESSING_MANIFEST_HASH, GATE,
        CONFIG, ENVIRONMENT, METRICS, METRICS_CSV, SEED_COMPARISON, CURVES, PHASE_D_REPORT, REPORT,
    ]
    for run in runs:
        seed_dir = TRAINING_DIR / f"seed_{run['seed']}"
        artifact_paths.extend([
            seed_dir / "best_checkpoint.pt", seed_dir / "training_history.json",
            seed_dir / "validation_metrics.json", seed_dir / "metadata.json",
        ])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "D_preprocessing_and_E_direct_finite_time_baseline",
        "phase_d_decision": "PASS",
        "phase_e_decision": "PASS",
        "ready_for_full_trajectory_validation": True,
        "scientific_quality_decision_deferred": True,
        "phase_c_gate": gate,
        "raw_phase_c_hashes_before": raw_hashes_before,
        "raw_phase_c_hashes_after": raw_hashes_after,
        "preprocessing": json.loads(CONSTANTS.read_text(encoding="utf-8")),
        "standardized_range_audit": audit,
        "training_configuration": training_config,
        "environment": json.loads(ENVIRONMENT.read_text(encoding="utf-8")),
        "runs": runs,
        "seed_comparison": comparison,
        "sealed_test_policy": {
            "integrity_only": True, "predictions": False, "loss": False,
            "errors": False, "plots": False, "seed_selection": False,
        },
        "artifacts": hashes_for_artifacts(artifact_paths),
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"PHASE D PASS; PHASE E PASS; wrote {REPORT}")


if __name__ == "__main__":
    main()
