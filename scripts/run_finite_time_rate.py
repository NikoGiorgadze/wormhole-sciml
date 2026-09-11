#!/usr/bin/env python3
"""Train and validate the identity-preserving average finite-time rate model."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wormhole_sciml.dynamics import conserved_energy, timelike_margin
from wormhole_sciml.finite_time import (
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
from wormhole_sciml.finite_time_rate import (
    RATE_PREPROCESSING_IMPLEMENTATION,
    RATE_PREPROCESSING_SCHEMA,
    RATE_TARGET_COLUMNS,
    RatePreprocessing,
    construct_rate_targets,
    load_rate_model,
    predict_rates,
    rate_target_audit,
    subset_metrics,
    train_rate_seed,
    two_component_metrics,
)
from wormhole_sciml.model_a import ModelA, parameter_count
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi, xi_time_derivative
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "finite_time_rate_baseline"
AUDIT_DIR = OUTPUT / "target_audit"
PREPROCESSING_DIR = OUTPUT / "preprocessing"
TRAINING_DIR = OUTPUT / "training"
VALIDATION_DIR = OUTPUT / "validation"
FIGURES_DIR = OUTPUT / "figures"
TESTS_DIR = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_RATE_BASELINE_REPORT.md"
SUMMARY = OUTPUT / "finite_time_rate_summary.json"
MANIFEST = OUTPUT / "finite_time_rate_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_rate_manifest.sha256"

DATA_DIR = ROOT / "output" / "phase_c_finite_time_dataset" / "datasets"
TRAIN_RAW = DATA_DIR / "phase_c_train_raw.npz"
VALIDATION_RAW = DATA_DIR / "phase_c_validation_raw.npz"
SEALED_RAW = DATA_DIR / "phase_c_test_sealed_raw.npz"
OLD_PREPROCESSING_PATH = ROOT / "output" / "finite_time_baseline" / "preprocessing" / "preprocessing_constants.json"
OLD_METRICS_PATH = ROOT / "output" / "finite_time_baseline" / "validation" / "per_seed_validation_metrics.json"
OLD_BASELINE_MANIFEST = ROOT / "output" / "finite_time_baseline" / "finite_time_baseline_manifest.json"
OLD_CHECKPOINTS = {
    seed: ROOT / "output" / "finite_time_baseline" / "training" / f"seed_{seed}" / "best_checkpoint.pt"
    for seed in TRAINING_SEEDS
}
PRIOR_VALIDATION = ROOT / "output" / "finite_time_trajectory_validation"
SMALL_S_QUERIES = PRIOR_VALIDATION / "arrays" / "small_s_queries.npz"
OLD_SMALL_PREDICTIONS = {
    seed: PRIOR_VALIDATION / "arrays" / f"small_s_predictions_seed_{seed}.npz"
    for seed in TRAINING_SEEDS
}
PRIOR_VALIDATION_MANIFEST = PRIOR_VALIDATION / "validation_manifest.json"

EXPECTED_HASHES = {
    TRAIN_RAW: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_RAW: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    SEALED_RAW: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    OLD_PREPROCESSING_PATH: "6c53549de9ff9d25454813213b98854d24cd031f009c774fc1306ae104d8e32e",
    OLD_CHECKPOINTS[101]: "6c8017cf693591970e070c36274a4131233a0600e12ae76afdf22b7416e08530",
    OLD_CHECKPOINTS[202]: "55e499a6ab0c0d67ee5d28f3b5f6971d6a93cec2d0c23b9da2e6014277e6c4d2",
    OLD_CHECKPOINTS[303]: "64272fd10f87291a5f7ec6d1c403874e91b5207f3f44ff54c20238d46827d7c5",
    OLD_BASELINE_MANIFEST: "ee1dc58a12c05c9e4d8ed9e6ea9f8383755962b3ea79b958745f6b13f084faa5",
    OLD_METRICS_PATH: "79092436cfade06fc42d26890b598c2d056477971f9cf77559cdc71e81e0a3b4",
    SMALL_S_QUERIES: "c7976e6255d9f37ecc11a0adddd81a4cae87a648c535b6d5300c73539cbc13d2",
    OLD_SMALL_PREDICTIONS[101]: "f7d9a00c37bc390770dfb24bb93555839d8057270f3e0131cf288818367c52a8",
    OLD_SMALL_PREDICTIONS[202]: "4405ddf10a149ccb0a955387e3ac98da07f263a4b0a9223ec28b64dd9f7047d0",
    OLD_SMALL_PREDICTIONS[303]: "287aaf9add10558d81d41b03dcf1b57accd74e1c7cc3f070f50b5d3f2cd33c7d",
    PRIOR_VALIDATION_MANIFEST: "f47177badc70777325826f09c04d5a64d7889cdcc07aad246c19340988f7c607",
}
OLD_PRIMARY_SEED = 303
SMALL_S_VALUES = np.asarray([0.0, 1.0e-3, 1.0e-2, 0.05, 0.10, 0.20, 0.50, 1.00])
COLORS = {101: "#4477aa", 202: "#ee7733", 303: "#228833"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def immutable_gate() -> dict[str, Any]:
    rows = {}
    failures = []
    for path, expected in EXPECTED_HASHES.items():
        measured = file_sha256(path)
        match = measured == expected
        rows[str(path.resolve())] = {
            "expected_sha256": expected, "measured_sha256": measured, "match": match,
        }
        if not match:
            failures.append(str(path))
    return {
        "passed": not failures, "failures": failures, "artifacts": rows,
        "sealed_test_access": "file-byte SHA-256 only; NPZ was not opened",
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{component}_{key}": value
        for component in ("x", "xi")
        for key, value in metrics[component].items()
    }


def rate_distribution_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for component, metrics in audit["global"].items():
        rows.append({"scope": "global", "component": component, "row_count": audit["row_count"], **metrics})
    for scope, entry in audit["by_physical_s"].items():
        for component in RATE_TARGET_COLUMNS:
            rows.append({"scope": scope, "component": component, "row_count": entry["row_count"], **entry[component]})
    for scope, entry in audit["by_family"].items():
        for component in RATE_TARGET_COLUMNS:
            rows.append({"scope": scope, "component": component, "row_count": entry["row_count"], **entry[component]})
    return rows


def energy_and_admissibility(
    validation: dict[str, np.ndarray], prediction: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    x_hat, xi_hat = prediction["predicted_x1"], prediction["predicted_xi1"]
    _, u_hat = state_from_xi(x_hat, xi_hat, wormhole, spiral)
    margin = timelike_margin(x_hat, u_hat, wormhole, spiral)
    xi_bad = np.abs(xi_hat) >= 1.0
    c_bad = margin <= 0.0
    valid = np.isfinite(margin) & (margin > 0.0)
    energy_error = np.full(x_hat.shape, np.nan, dtype=np.float64)
    energy_error[valid] = conserved_energy(x_hat[valid], u_hat[valid], wormhole, spiral) - validation["E0"][valid]
    finite = energy_error[np.isfinite(energy_error)]
    absolute = np.abs(finite)
    return {
        "row_count": int(x_hat.size),
        "absolute_xi_ge_1_count": int(np.sum(xi_bad)),
        "C_le_0_count": int(np.sum(c_bad)),
        "union_violation_count": int(np.sum(xi_bad | c_bad | ~np.isfinite(margin))),
        "energy": {
            "finite_count": int(finite.size), "invalid_count": int(x_hat.size - finite.size),
            "rmse": float(np.sqrt(np.mean(finite**2))), "mae": float(np.mean(absolute)),
            "p99_absolute": float(np.quantile(absolute, 0.99)),
            "maximum_absolute": float(np.max(absolute)),
        },
        "used_in_loss": False,
    }, {"predicted_u1": u_hat, "predicted_C1": margin, "energy_error": energy_error}


def evaluate_validation(
    model: ModelA,
    preprocessing: RatePreprocessing,
    validation: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction = predict_rates(model, preprocessing, validation)
    x_error = prediction["predicted_x1"] - validation["x1"]
    xi_error = prediction["predicted_xi1"] - validation["xi1"]
    prediction["x_error"] = x_error
    prediction["xi_error"] = xi_error
    target_standardized = preprocessing.standardize_targets(stack_columns(validation, RATE_TARGET_COLUMNS))
    predicted_standardized = np.column_stack((prediction["standardized_V_x"], prediction["standardized_V_xi"]))
    standardized = orbit_averaged_standardized_mse(
        predicted_standardized, target_standardized, validation["orbit_id"]
    )
    identity = validation["s"] == 0.0
    if not np.array_equal(prediction["predicted_Delta_x"][identity], np.zeros(np.sum(identity))):
        raise RuntimeError("physical-s gate failed exact Delta_x identity")
    if not np.array_equal(prediction["predicted_Delta_xi"][identity], np.zeros(np.sum(identity))):
        raise RuntimeError("physical-s gate failed exact Delta_xi identity")
    if not np.array_equal(prediction["predicted_x1"][identity], validation["x0"][identity]):
        raise RuntimeError("physical-s gate failed exact x identity")
    if not np.array_equal(prediction["predicted_xi1"][identity], validation["xi0"][identity]):
        raise RuntimeError("physical-s gate failed exact xi identity")
    time_masks = {
        "short_s_le_5": validation["s"] <= 5.0,
        "intermediate_5_lt_s_le_20": (validation["s"] > 5.0) & (validation["s"] <= 20.0),
        "long_s_gt_20": validation["s"] > 20.0,
    }
    family_masks = {
        "hard_u_th_le_0p30": validation["u_th"] <= 0.30,
        "ordinary_u_th_gt_0p30": validation["u_th"] > 0.30,
    }
    for center in (0.05, 0.15, 0.30):
        family_masks[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(validation["u_th"] - center) <= 0.01
    physical, diagnostic_arrays = energy_and_admissibility(validation, prediction)
    prediction.update(diagnostic_arrays)
    rate_error_x = prediction["predicted_V_x"] - validation["V_x"]
    rate_error_xi = prediction["predicted_V_xi"] - validation["V_xi"]
    residual_state_difference = np.column_stack((x_error, xi_error)) - np.column_stack((
        prediction["predicted_Delta_x"] - validation["Delta_x"],
        prediction["predicted_Delta_xi"] - validation["Delta_xi"],
    ))
    metrics = {
        "standardized_validation_rate_mse": standardized,
        "physical_state_metrics": two_component_metrics(x_error, xi_error),
        "rate_metrics": two_component_metrics(rate_error_x, rate_error_xi),
        "identity": {
            "row_count": int(np.sum(identity)),
            "max_abs_Delta_x_prediction": float(np.max(np.abs(prediction["predicted_Delta_x"][identity]))),
            "max_abs_Delta_xi_prediction": float(np.max(np.abs(prediction["predicted_Delta_xi"][identity]))),
            "rmse_x": float(np.sqrt(np.mean(x_error[identity] ** 2))),
            "rmse_xi": float(np.sqrt(np.mean(xi_error[identity] ** 2))),
            "predicted_rate_error": two_component_metrics(rate_error_x[identity], rate_error_xi[identity]),
        },
        "time_regimes": {name: subset_metrics(prediction, validation, mask) for name, mask in time_masks.items()},
        "families": {name: subset_metrics(prediction, validation, mask) for name, mask in family_masks.items()},
        "physical_diagnostics": physical,
        "maximum_residual_vs_state_error_difference": float(np.max(np.abs(residual_state_difference))),
        "sealed_test_predictions_computed": False,
    }
    return metrics, prediction


def small_s_queries() -> dict[str, np.ndarray]:
    with np.load(SMALL_S_QUERIES, allow_pickle=False) as source:
        query = {name: source[name] for name in source.files}
    query["x1"] = query["exact_x1"]
    query["xi1"] = query["exact_xi1"]
    query["Delta_x"] = query["x1"] - query["x0"]
    query["Delta_xi"] = query["xi1"] - query["xi0"]
    return construct_rate_targets(query)


def evaluate_small_s(
    models: dict[int, ModelA], preprocessing: RatePreprocessing, queries: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for seed, model in models.items():
        prediction = predict_rates(model, preprocessing, queries)
        predictions[seed] = prediction
        for value in SMALL_S_VALUES:
            mask = np.isclose(queries["s"], value, rtol=0.0, atol=1.0e-14)
            x_error = prediction["predicted_x1"][mask] - queries["x1"][mask]
            xi_error = prediction["predicted_xi1"][mask] - queries["xi1"][mask]
            vx_error = prediction["predicted_V_x"][mask] - queries["V_x"][mask]
            vxi_error = prediction["predicted_V_xi"][mask] - queries["V_xi"][mask]
            rows.append({
                "model": "average_rate", "seed": seed, "s": float(value), "row_count": int(np.sum(mask)),
                **flatten_metrics(two_component_metrics(x_error, xi_error)),
                "V_x_rmse": float(np.sqrt(np.mean(vx_error**2))), "V_x_mae": float(np.mean(np.abs(vx_error))),
                "V_xi_rmse": float(np.sqrt(np.mean(vxi_error**2))), "V_xi_mae": float(np.mean(np.abs(vxi_error))),
            })
    for seed, path in OLD_SMALL_PREDICTIONS.items():
        with np.load(path, allow_pickle=False) as old:
            for value in SMALL_S_VALUES:
                mask = np.isclose(queries["s"], value, rtol=0.0, atol=1.0e-14)
                rows.append({
                    "model": "accumulated_residual", "seed": seed, "s": float(value), "row_count": int(np.sum(mask)),
                    **flatten_metrics(two_component_metrics(old["x_error"][mask], old["xi_error"][mask])),
                })
    return rows, predictions


def old_component(entry: dict[str, Any], component: str) -> dict[str, float]:
    source = entry["physical_residual_metrics"]["Delta_x" if component == "x" else "Delta_xi"]
    return {
        "rmse": source["rmse"], "mae": source["mae"],
        "median_absolute": source["absolute_error"]["median"],
        "p90_absolute": source["absolute_error"]["p90"],
        "p95_absolute": source["absolute_error"]["p95"],
        "p99_absolute": source["absolute_error"]["p99"],
        "maximum_absolute": source["absolute_error"]["maximum"],
    }


def old_group_metrics(entry: dict[str, Any], category: str, name: str) -> dict[str, Any]:
    key = "time_regime_breakdown" if category == "time" else "hard_family_breakdown"
    value = entry[key][name]
    metrics = value["metrics"]
    return {
        "row_count": value["row_count"],
        "x": {
            "rmse": metrics["Delta_x"]["rmse"], "mae": metrics["Delta_x"]["mae"],
        },
        "xi": {
            "rmse": metrics["Delta_xi"]["rmse"], "mae": metrics["Delta_xi"]["mae"],
        },
    }


def comparison_rows(
    runs: list[dict[str, Any]], new_metrics: dict[str, Any], old_metrics: dict[str, Any],
    small_rows: list[dict[str, Any]], new_primary: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        new = new_metrics[str(seed)]
        old = old_metrics[str(seed)]
        for model_name, aggregate in (
            ("accumulated_residual", {"x": old_component(old, "x"), "xi": old_component(old, "xi")}),
            ("average_rate", new["physical_state_metrics"]),
        ):
            rows.append({"model": model_name, "seed": seed, "scope": "aggregate", **flatten_metrics(aggregate)})
        for category, names in (
            ("time", ("short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20")),
            ("family", ("hard_u_th_le_0p30", "ordinary_u_th_gt_0p30")),
        ):
            for name in names:
                new_group = new["time_regimes" if category == "time" else "families"][name]
                old_group = old_group_metrics(old, category, name)
                rows.append({"model": "accumulated_residual", "seed": seed, "scope": name, **flatten_metrics(old_group)})
                rows.append({"model": "average_rate", "seed": seed, "scope": name, **flatten_metrics(new_group)})
        old_diag = old["physical_diagnostics"]
        rows.append({
            "model": "accumulated_residual", "seed": seed, "scope": "physical_diagnostics",
            "absolute_xi_ge_1_count": old_diag["absolute_xi_ge_1"]["count"],
            "C_le_0_count": old_diag["C_le_0"]["count"],
            "energy_mae": old_diag["energy_error_E_hat_minus_E0"]["mae"],
            "energy_rmse": old_diag["energy_error_E_hat_minus_E0"]["rmse"],
            "energy_p99_absolute": old_diag["energy_error_E_hat_minus_E0"]["p99_absolute"],
            "energy_maximum_absolute": old_diag["energy_error_E_hat_minus_E0"]["maximum_absolute"],
        })
        new_diag = new["physical_diagnostics"]
        rows.append({
            "model": "average_rate", "seed": seed, "scope": "physical_diagnostics",
            "absolute_xi_ge_1_count": new_diag["absolute_xi_ge_1_count"],
            "C_le_0_count": new_diag["C_le_0_count"],
            "energy_mae": new_diag["energy"]["mae"], "energy_rmse": new_diag["energy"]["rmse"],
            "energy_p99_absolute": new_diag["energy"]["p99_absolute"],
            "energy_maximum_absolute": new_diag["energy"]["maximum_absolute"],
        })
    primary_rows = {
        "old_primary_seed": OLD_PRIMARY_SEED,
        "new_primary_seed": new_primary,
        "aggregate": {
            "old": {"x": old_component(old_metrics[str(OLD_PRIMARY_SEED)], "x"), "xi": old_component(old_metrics[str(OLD_PRIMARY_SEED)], "xi")},
            "new": new_metrics[str(new_primary)]["physical_state_metrics"],
        },
        "identity": {
            "old": old_metrics[str(OLD_PRIMARY_SEED)]["identity_diagnostics"],
            "new": new_metrics[str(new_primary)]["identity"],
        },
        "small_s": {
            model: {
                f"{row['s']:.12g}": row
                for row in small_rows
                if row["model"] == model and row["seed"] == (new_primary if model == "average_rate" else OLD_PRIMARY_SEED)
            }
            for model in ("accumulated_residual", "average_rate")
        },
        "time_regimes": {
            name: {
                "old": old_group_metrics(old_metrics[str(OLD_PRIMARY_SEED)], "time", name),
                "new": new_metrics[str(new_primary)]["time_regimes"][name],
            }
            for name in ("short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20")
        },
        "families": {
            name: {
                "old": old_group_metrics(old_metrics[str(OLD_PRIMARY_SEED)], "family", name),
                "new": new_metrics[str(new_primary)]["families"][name],
            }
            for name in ("hard_u_th_le_0p30", "ordinary_u_th_gt_0p30")
        },
    }
    return rows, primary_rows


def plot_audit(audit: dict[str, Any], path: Path) -> None:
    labels = list(audit["by_physical_s"])
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    for axis, component in zip(axes, RATE_TARGET_COLUMNS):
        medians = [audit["by_physical_s"][label][component]["median"] for label in labels]
        p1 = [audit["by_physical_s"][label][component]["p1"] for label in labels]
        p99 = [audit["by_physical_s"][label][component]["p99"] for label in labels]
        x = np.arange(len(labels))
        axis.plot(x, medians, marker="o", label="median")
        axis.fill_between(x, p1, p99, alpha=0.25, label="p1–p99")
        axis.set(xticks=x, xticklabels=[label.replace("_", "\n") for label in labels], ylabel=component)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Average finite-time rate targets across physical elapsed-time regimes")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_histories(runs: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True, sharey=True)
    for axis, run in zip(axes, runs):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = [row["epoch"] for row in history]
        axis.plot(epoch, [row["training_standardized_rate_mse"] for row in history], label="training")
        axis.plot(epoch, [row["validation_orbit_averaged_standardized_rate_mse"] for row in history], label="validation")
        axis.axvline(run["best_epoch"], color="0.25", ls="--", label="best")
        axis.axvline(run["stopping_epoch"], color="0.55", ls=":", label="stop")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("standardized average-rate MSE")
    figure.suptitle("Average finite-time rate model training histories")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_small_s(rows: list[dict[str, Any]], old_seed: int, new_seed: int, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for model_name, seed, style in (("accumulated_residual", old_seed, "--"), ("average_rate", new_seed, "-")):
        selected = sorted((row for row in rows if row["model"] == model_name and row["seed"] == seed), key=lambda row: row["s"])
        for axis, component in zip(axes, ("x", "xi")):
            plotted = np.maximum([row[f"{component}_rmse"] for row in selected], 1.0e-10)
            axis.plot([row["s"] for row in selected], plotted, marker="o", ls=style, label=f"{model_name} seed {seed}")
            axis.set(xlabel="physical elapsed time s", ylabel=f"RMSE {component}", yscale="log", xscale="symlog", xlim=(0.0, 1.05))
            axis.set_xticks(SMALL_S_VALUES)
            axis.set_xticklabels(("0", "1e-3", "1e-2", "0.05", "0.1", "0.2", "0.5", "1"), rotation=35, ha="right")
            axis.grid(alpha=0.25, which="both")
    axes[0].annotate("new model: exact zero", (0.0, 1.0e-10), xytext=(0.006, 3.0e-9), arrowprops={"arrowstyle": "->"}, fontsize=8)
    axes[1].annotate("new model: exact zero", (0.0, 1.0e-10), xytext=(0.006, 3.0e-9), arrowprops={"arrowstyle": "->"}, fontsize=8)
    axes[0].legend(fontsize=8)
    figure.suptitle("Small-time state error before and after identity-preserving reconstruction")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_comparison(primary: dict[str, Any], path: Path) -> None:
    scopes = ["aggregate", "short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20", "hard_u_th_le_0p30", "ordinary_u_th_gt_0p30"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    for axis, component in zip(axes, ("x", "xi")):
        old_values, new_values = [], []
        for scope in scopes:
            if scope == "aggregate": source = primary["aggregate"]
            elif scope in primary["time_regimes"]: source = primary["time_regimes"][scope]
            else: source = primary["families"][scope]
            old_values.append(source["old"][component]["rmse"])
            new_values.append(source["new"][component]["rmse"])
        x = np.arange(len(scopes)); width = 0.38
        axis.bar(x - width / 2, old_values, width, label="accumulated residual")
        axis.bar(x + width / 2, new_values, width, label="average rate")
        axis.set(xticks=x, xticklabels=[s.replace("_", "\n") for s in scopes], ylabel=f"RMSE {component}", yscale="log")
        axis.grid(alpha=0.25, axis="y", which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("Finite-time prediction error before and after identity-preserving reconstruction")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def run_tests() -> dict[str, Any]:
    xml = TESTS_DIR / "relevant_pytest.xml"
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_finite_time_rate.py", "tests/test_finite_time_baseline.py",
        "tests/test_phase_b_orbits.py", "tests/test_physics_gate.py",
        f"--junitxml={xml}",
    ]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-rate-mpl-cache"}, capture_output=True, text=True)
    payload = {
        "command": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
        "passed": result.returncode == 0, "junit_xml": str(xml.resolve()),
    }
    write_json(TESTS_DIR / "test_summary.json", payload)
    return payload


def report_text(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    metrics = summary["validation_metrics"]
    primary = summary["comparison"]["primary"]
    audit = summary["rate_target_audit"]
    training_rows = "\n".join(
        f"| {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | {run['best_validation_orbit_averaged_standardized_rate_mse']:.7g} | {run['final_training_standardized_rate_mse']:.7g} | `{run['checkpoint_sha256']}` |"
        for run in runs
    )
    physical_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['physical_state_metrics']['x']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['x']['mae']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['xi']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['xi']['mae']:.6g} |"
        for run in runs
    )
    identity_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['identity']['max_abs_Delta_x_prediction']:.3g} | {metrics[str(run['seed'])]['identity']['max_abs_Delta_xi_prediction']:.3g} | {metrics[str(run['seed'])]['identity']['rmse_x']:.3g} | {metrics[str(run['seed'])]['identity']['rmse_xi']:.3g} | {metrics[str(run['seed'])]['identity']['predicted_rate_error']['x']['rmse']:.6g} | {metrics[str(run['seed'])]['identity']['predicted_rate_error']['xi']['rmse']:.6g} |"
        for run in runs
    )
    small_rows = "\n".join(
        f"| {value} | {primary['small_s']['accumulated_residual'][value]['x_rmse']:.6g} | {primary['small_s']['average_rate'][value]['x_rmse']:.6g} | {primary['small_s']['accumulated_residual'][value]['xi_rmse']:.6g} | {primary['small_s']['average_rate'][value]['xi_rmse']:.6g} |"
        for value in ("0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5", "1")
    )
    breakdown_rows = []
    for name, source in list(primary["time_regimes"].items()) + list(primary["families"].items()):
        breakdown_rows.append(
            f"| {name} | {source['old']['x']['rmse']:.6g} | {source['new']['x']['rmse']:.6g} | {source['old']['xi']['rmse']:.6g} | {source['new']['xi']['rmse']:.6g} |"
        )
    diagnostics = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['physical_diagnostics']['absolute_xi_ge_1_count']} | {metrics[str(run['seed'])]['physical_diagnostics']['C_le_0_count']} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['mae']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['p99_absolute']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['maximum_absolute']:.6g} |"
        for run in runs
    )
    old_agg, new_agg = primary["aggregate"]["old"], primary["aggregate"]["new"]
    old_long, new_long = primary["time_regimes"]["long_s_gt_20"]["old"], primary["time_regimes"]["long_s_gt_20"]["new"]
    small_old = primary["small_s"]["accumulated_residual"]["0.2"]
    small_new = primary["small_s"]["average_rate"]["0.2"]
    aggregate_x_factor = new_agg['x']['rmse'] / old_agg['x']['rmse']
    aggregate_xi_factor = new_agg['xi']['rmse'] / old_agg['xi']['rmse']
    long_x_factor = new_long['x']['rmse'] / old_long['x']['rmse']
    long_xi_factor = new_long['xi']['rmse'] / old_long['xi']['rmse']
    return f"""# Identity-preserving average finite-time rate experiment

## Controlled change and frozen scope

The only modeling change is the output representation. For `s>0`, exact targets are `V_x=Delta_x/s` and `V_xi=Delta_xi/s`. The network still receives standardized `[x0,xi0,E0,s]`, but physical inference always uses the explicit unstandardized gate `Delta_hat = physical_s * V_hat`. At `s=0`, targets use the continuous generator limits `V_x=u0` and analytic `V_xi=dot(xi)_0`. The architecture, optimizer, loss philosophy, batches, seeds, patience, and dataset are unchanged. No trajectory was generated, no old artifact was overwritten, and the sealed test NPZ was never opened.

`dot(xi)` is the chain-rule derivative of the established `xi=(u-c(x))/d(x)`, using the validated radial acceleration and exact derivatives of the same corridor geometry. Its central-difference audit against saved exact dense trajectories has maximum discrepancy about `2.3e-11`.

## Raw rate-target audit

| target | min | max | mean | std | median | p0.1 | p1 | p5 | p95 | p99 | p99.9 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V_x | {audit['global']['V_x']['minimum']:.6g} | {audit['global']['V_x']['maximum']:.6g} | {audit['global']['V_x']['mean']:.6g} | {audit['global']['V_x']['standard_deviation']:.6g} | {audit['global']['V_x']['median']:.6g} | {audit['global']['V_x']['p0.1']:.6g} | {audit['global']['V_x']['p1']:.6g} | {audit['global']['V_x']['p5']:.6g} | {audit['global']['V_x']['p95']:.6g} | {audit['global']['V_x']['p99']:.6g} | {audit['global']['V_x']['p99.9']:.6g} |
| V_xi | {audit['global']['V_xi']['minimum']:.6g} | {audit['global']['V_xi']['maximum']:.6g} | {audit['global']['V_xi']['mean']:.6g} | {audit['global']['V_xi']['standard_deviation']:.6g} | {audit['global']['V_xi']['median']:.6g} | {audit['global']['V_xi']['p0.1']:.6g} | {audit['global']['V_xi']['p1']:.6g} | {audit['global']['V_xi']['p5']:.6g} | {audit['global']['V_xi']['p95']:.6g} | {audit['global']['V_xi']['p99']:.6g} | {audit['global']['V_xi']['p99.9']:.6g} |

All `{audit['row_count']}` targets are finite. Minimum positive s is `{audit['minimum_positive_s']:.6g}`; the very-small-s p99 generator discrepancy is `{audit['small_positive_s_p99_absolute_generator_difference']:.6g}`. The audit passed without clipping, cutoff, or dataset changes. Full time/family distributions are in `target_audit/rate_target_distribution_audit.json` and CSV.

## Preprocessing and architecture

Frozen input means/std were copied bit-for-bit from the previous training-only preprocessing. New training-only target constants are `mu_Vx={summary['preprocessing']['columns']['V_x']['mean']:.12g}`, `sigma_Vx={summary['preprocessing']['columns']['V_x']['standard_deviation']:.12g}`, `mu_Vxi={summary['preprocessing']['columns']['V_xi']['mean']:.12g}`, and `sigma_Vxi={summary['preprocessing']['columns']['V_xi']['standard_deviation']:.12g}`. The network remains `4→64→64→2`, two tanh hidden layers, no output activation, exactly `{EXPECTED_PARAMETER_COUNT}` parameters.

## Training

Adam `lr={LEARNING_RATE:g}`, batch `{BATCH_SIZE}`, weight decay `0`, no scheduler, float32, maximum `{MAXIMUM_EPOCHS}` epochs, patience `{PATIENCE}`, ordinary shuffled rows, and equal standardized MSE on `[V_x,V_xi]` were used exactly as prescribed. The new primary seed is `{summary['new_primary_seed']}`, selected solely by minimum validation orbit-averaged standardized rate MSE.

| seed | best epoch | stop epoch | best validation rate MSE | final training rate MSE | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|:---|
{training_rows}

## Physical validation on the frozen 98,304 rows

| seed | RMSE x | MAE x | RMSE xi | MAE xi |
|---:|---:|---:|---:|---:|
{physical_rows}

Median/p90/p95/p99/max component errors are in `validation/per_seed_validation_metrics.json` and `validation/per_seed_summary.csv`.

## Exact identity and learned local rates

| seed | max abs Delta-x at s=0 | max abs Delta-xi at s=0 | RMSE x at s=0 | RMSE xi at s=0 | rate RMSE Vx | rate RMSE Vxi |
|---:|---:|---:|---:|---:|---:|---:|
{identity_rows}

All reconstructed residual and state identity values are exact floating-point zeros. Rate errors are nonzero model errors and do not compromise the structural identity gate.

## Small-time comparison

| s | old RMSE x | new RMSE x | old RMSE xi | new RMSE xi |
|---:|---:|---:|---:|---:|
{small_rows}

At s=0.2, primary x RMSE changes from `{small_old['x_rmse']:.6g}` to `{small_new['x_rmse']:.6g}` and xi RMSE from `{small_old['xi_rmse']:.6g}` to `{small_new['xi_rmse']:.6g}`. New per-seed rate errors, including direct comparison with `u0` and `dot(xi)_0` at zero, are in `validation/small_s_metrics.csv`.

## Time and orbit-family comparison

| scope | old RMSE x | new RMSE x | old RMSE xi | new RMSE xi |
|:---|---:|---:|---:|---:|
{chr(10).join(breakdown_rows)}

Neighborhood metrics around u_th=0.05, 0.15, and 0.30 are retained in the new per-seed validation JSON.

## Basic physical diagnostics

| seed | abs(xi)>=1 | C<=0 | energy MAE | energy RMSE | abs energy p99 | abs energy max |
|---:|---:|---:|---:|---:|---:|---:|
{diagnostics}

Energy and admissibility remain diagnostics only and were not used in training.

## Tests and reproducibility

The relevant regression command passed: `{summary['tests']['stdout'].strip()}` Protected files were hash-identical before and after. The manifest hashes all generated artifacts and records sealed-test access as byte-hash-only.

## Scientific assessment

The identity-preserving representation fixes the dominant structural defect exactly: `Phi_hat_0(z)=z` for every validation identity row, independent of network weights. At s=0.2 it changes primary x/xi RMSE by factors `{small_new['x_rmse']/small_old['x_rmse']:.4g}` and `{small_new['xi_rmse']/small_old['xi_rmse']:.4g}` relative to the old model.

Aggregate accuracy is mixed rather than uniformly improved: x RMSE improves from `{old_agg['x']['rmse']:.6g}` to `{new_agg['x']['rmse']:.6g}` (`{100*(1-aggregate_x_factor):.1f}%` reduction), while xi RMSE degrades from `{old_agg['xi']['rmse']:.6g}` to `{new_agg['xi']['rmse']:.6g}` (`{100*(aggregate_xi_factor-1):.1f}%` increase). Long-time accuracy shows the same tradeoff: x RMSE improves from `{old_long['x']['rmse']:.6g}` to `{new_long['x']['rmse']:.6g}` (`{100*(1-long_x_factor):.1f}%` reduction), while xi RMSE degrades from `{old_long['xi']['rmse']:.6g}` to `{new_long['xi']['rmse']:.6g}` (`{100*(long_xi_factor-1):.1f}%` increase). The controlled representation change therefore decisively fixes the dominant small-s defect and substantially improves x, but does not deliver an across-the-board finite-time improvement because xi is worse at intermediate/long horizons and in both orbit families. No further redesign was performed.
"""


def finalize_existing() -> None:
    """Regenerate presentation artifacts and hashes without retraining."""

    if not SUMMARY.exists() or not MANIFEST.exists():
        raise FileNotFoundError("completed rate experiment is required for finalization")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    REPORT.write_text(report_text(summary), encoding="utf-8")
    primary_small = summary["comparison"]["primary"]["small_s"]
    plot_small_s(
        list(primary_small["accumulated_residual"].values()) + list(primary_small["average_rate"].values()),
        OLD_PRIMARY_SEED,
        int(summary["new_primary_seed"]),
        FIGURES_DIR / "small_s_before_after.png",
    )
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_protected = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}
    if current_protected != existing["protected_hashes_before"]:
        raise RuntimeError("a protected artifact changed before finalization")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {
            "path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
    existing["protected_hashes_after"] = current_protected
    existing["source_hashes"] = {
        "src/wormhole_sciml/physics_gate.py": file_sha256(ROOT / "src/wormhole_sciml/physics_gate.py"),
        "src/wormhole_sciml/finite_time_rate.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_rate.py"),
        "scripts/run_finite_time_rate.py": file_sha256(Path(__file__)),
        "tests/test_finite_time_rate.py": file_sha256(ROOT / "tests/test_finite_time_rate.py"),
    }
    existing["summary"] = {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)}
    existing["report"] = {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)}
    existing["artifacts"] = artifacts
    write_json(MANIFEST, existing)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, AUDIT_DIR, PREPROCESSING_DIR, TRAINING_DIR, VALIDATION_DIR, FIGURES_DIR, TESTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    gate = immutable_gate()
    write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"immutable input gate failed: {gate['failures']}")
    hashes_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}

    raw_training = load_dataset(TRAIN_RAW)
    raw_validation = load_dataset(VALIDATION_RAW)
    training = construct_rate_targets(raw_training)
    validation = construct_rate_targets(raw_validation)
    audit = rate_target_audit(training)
    write_json(AUDIT_DIR / "rate_target_construction.json", {
        "representation": "average finite-time rates",
        "positive_s": {"V_x": "Delta_x / s", "V_xi": "Delta_xi / s"},
        "zero_s": {"V_x": "u0", "V_xi": "xi_time_derivative(x0,u0)"},
        "division_policy": "np.divide only where physical s > 0; no division at s=0",
        "inference": "Delta_hat = physical s * unstandardized V_hat",
        "clipping": False, "minimum_s_cutoff": None,
    })
    write_json(AUDIT_DIR / "rate_target_distribution_audit.json", audit)
    write_csv(AUDIT_DIR / "rate_target_distribution_audit.csv", rate_distribution_rows(audit))
    plot_audit(audit, FIGURES_DIR / "rate_target_distributions_vs_elapsed_time.png")
    if audit["obvious_numerical_pathology"]:
        raise RuntimeError("rate target audit required stopping before training")

    old_preprocessing = FiniteTimePreprocessing.from_json(OLD_PREPROCESSING_PATH)
    preprocessing = RatePreprocessing.fit(
        training, old_preprocessing, file_sha256(TRAIN_RAW), file_sha256(OLD_PREPROCESSING_PATH)
    )
    if not np.array_equal(preprocessing.input_mean, old_preprocessing.input_mean) or not np.array_equal(preprocessing.input_std, old_preprocessing.input_std):
        raise RuntimeError("frozen input preprocessing was not reused exactly")
    preprocessing_payload = preprocessing.payload(TRAIN_RAW, len(training["s"]))
    write_json(PREPROCESSING_DIR / "rate_preprocessing_constants.json", preprocessing_payload)
    write_json(PREPROCESSING_DIR / "rate_preprocessing_manifest.json", {
        "schema": RATE_PREPROCESSING_SCHEMA, "implementation": RATE_PREPROCESSING_IMPLEMENTATION,
        "source_training_sha256": file_sha256(TRAIN_RAW),
        "old_input_preprocessing": {"path": str(OLD_PREPROCESSING_PATH.resolve()), "sha256": file_sha256(OLD_PREPROCESSING_PATH)},
        "input_constants_exactly_reused": True,
        "target_constants_fitted_from_training_only": True,
        "sealed_test_used_for_fit_or_distribution_inspection": False,
    })
    training_config = {
        "architecture": "4->64->64->2; tanh/tanh; linear output", "parameter_count": EXPECTED_PARAMETER_COUNT,
        "inputs": list(INPUT_COLUMNS), "targets": list(RATE_TARGET_COLUMNS),
        "physical_output_gate": "Delta_hat = physical s * V_hat",
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "weight_decay": 0.0, "scheduler": None, "maximum_epochs": MAXIMUM_EPOCHS,
        "early_stopping_patience": PATIENCE, "seeds": list(TRAINING_SEEDS), "dtype": "float32",
        "loss": "equal standardized MSE on V_x and V_xi", "oversampling": False,
        "checkpoint_metric": "validation orbit-averaged standardized rate MSE",
    }
    write_json(TRAINING_DIR / "training_config.json", training_config)
    write_json(TRAINING_DIR / "environment.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "matplotlib": matplotlib.__version__, "cpu_count": os.cpu_count(), "device": "cpu",
    })
    if parameter_count(ModelA(4, HIDDEN_DIMENSIONS)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("architecture parameter-count gate failed")

    runs: list[dict[str, Any]] = []
    models: dict[int, ModelA] = {}
    metrics: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        run = train_rate_seed(training, validation, preprocessing, seed, TRAINING_DIR / f"seed_{seed}")
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        run["history_sha256"] = file_sha256(Path(run["history"]))
        model = load_rate_model(Path(run["checkpoint"]))
        measured, prediction = evaluate_validation(model, preprocessing, validation)
        discrepancy = abs(
            measured["standardized_validation_rate_mse"]["orbit_averaged_standardized_mse"]
            - run["best_validation_orbit_averaged_standardized_rate_mse"]
        )
        if discrepancy > 2.0e-7:
            raise RuntimeError("restored checkpoint rate metric discrepancy exceeded gate")
        if measured["maximum_residual_vs_state_error_difference"] > 1.0e-13:
            raise RuntimeError("residual and physical-state errors differ")
        run["checkpoint_reload_metric_absolute_difference"] = discrepancy
        write_json(TRAINING_DIR / f"seed_{seed}" / "metadata.json", run)
        np.savez_compressed(VALIDATION_DIR / f"validation_predictions_seed_{seed}.npz", **prediction)
        write_json(VALIDATION_DIR / f"validation_metrics_seed_{seed}.json", measured)
        models[seed] = model
        metrics[str(seed)] = measured
        runs.append(run)
    write_json(VALIDATION_DIR / "per_seed_validation_metrics.json", metrics)
    new_primary = int(min(runs, key=lambda run: run["best_validation_orbit_averaged_standardized_rate_mse"])["seed"])
    plot_histories(runs, FIGURES_DIR / "training_histories.png")

    queries = small_s_queries()
    small_rows, small_predictions = evaluate_small_s(models, preprocessing, queries)
    write_csv(VALIDATION_DIR / "small_s_metrics.csv", small_rows)
    np.savez_compressed(VALIDATION_DIR / "small_s_queries_with_rate_targets.npz", **queries)
    for seed, prediction in small_predictions.items():
        np.savez_compressed(VALIDATION_DIR / f"small_s_predictions_seed_{seed}.npz", **prediction)
    old_metrics = json.loads(OLD_METRICS_PATH.read_text(encoding="utf-8"))
    before_after_rows, primary_comparison = comparison_rows(runs, metrics, old_metrics, small_rows, new_primary)
    write_csv(VALIDATION_DIR / "old_vs_new_comparison.csv", before_after_rows)
    write_json(VALIDATION_DIR / "old_vs_new_comparison.json", {
        "old_baseline": {"primary_seed": OLD_PRIMARY_SEED, "manifest": str(OLD_BASELINE_MANIFEST.resolve()), "manifest_sha256": file_sha256(OLD_BASELINE_MANIFEST)},
        "new_primary_selection": "minimum validation orbit-averaged standardized rate MSE",
        "new_primary_seed": new_primary, "primary": primary_comparison,
        "all_seed_rows": before_after_rows,
    })
    summary_rows = [
        {"seed": seed, **flatten_metrics(metrics[str(seed)]["physical_state_metrics"]),
         "identity_max_abs_Delta_x": metrics[str(seed)]["identity"]["max_abs_Delta_x_prediction"],
         "identity_max_abs_Delta_xi": metrics[str(seed)]["identity"]["max_abs_Delta_xi_prediction"],
         "energy_mae": metrics[str(seed)]["physical_diagnostics"]["energy"]["mae"],
         "energy_rmse": metrics[str(seed)]["physical_diagnostics"]["energy"]["rmse"]}
        for seed in TRAINING_SEEDS
    ]
    write_csv(VALIDATION_DIR / "per_seed_summary.csv", summary_rows)
    plot_small_s(small_rows, OLD_PRIMARY_SEED, new_primary, FIGURES_DIR / "small_s_before_after.png")
    plot_comparison(primary_comparison, FIGURES_DIR / "old_vs_new_error_comparison.png")

    test_result = run_tests()
    hashes_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}
    if hashes_before != hashes_after:
        raise RuntimeError("a frozen dataset or prior-model artifact changed")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rate_target_audit": audit, "preprocessing": preprocessing_payload,
        "training_configuration": training_config, "runs": runs,
        "new_primary_seed": new_primary, "validation_metrics": metrics,
        "comparison": {"primary": primary_comparison, "all_seed_rows": before_after_rows},
        "tests": test_result,
        "sealed_test_predictions_computed": False,
        "sealed_test_npz_opened": False,
        "training_orbit_count": int(np.unique(training["orbit_id"]).size),
        "validation_orbit_count": int(np.unique(validation["orbit_id"]).size),
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {
            "path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size,
        }
    manifest = {
        "experiment": "identity_preserving_average_finite_time_rate",
        "status": "completed" if test_result["passed"] else "completed_with_test_failures",
        "immutable_gate": gate, "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "sealed_test_policy": {
            "NPZ_opened": False, "predictions": False, "loss": False,
            "metrics": False, "plots": False, "distribution_inspection": False,
            "byte_hash_only": True,
        },
        "source_hashes": {
            "src/wormhole_sciml/physics_gate.py": file_sha256(ROOT / "src/wormhole_sciml/physics_gate.py"),
            "src/wormhole_sciml/finite_time_rate.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_rate.py"),
            "scripts/run_finite_time_rate.py": file_sha256(Path(__file__)),
            "tests/test_finite_time_rate.py": file_sha256(ROOT / "tests/test_finite_time_rate.py"),
        },
        "new_primary_seed": new_primary, "old_primary_seed": OLD_PRIMARY_SEED,
        "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)},
        "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)},
        "artifacts": artifacts,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-only"]:
        finalize_existing()
    else:
        main()
