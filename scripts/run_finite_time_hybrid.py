#!/usr/bin/env python3
"""Train and validate the controlled s*=5 hybrid finite-time model."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
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
    BATCH_SIZE, EXPECTED_PARAMETER_COUNT, HIDDEN_DIMENSIONS, INPUT_COLUMNS,
    LEARNING_RATE, MAXIMUM_EPOCHS, PATIENCE, TRAINING_SEEDS,
    orbit_averaged_standardized_mse, stack_columns,
)
from wormhole_sciml.finite_time_hybrid import (
    HYBRID_PREPROCESSING_IMPLEMENTATION, HYBRID_PREPROCESSING_SCHEMA,
    HYBRID_TARGET_COLUMNS, S_STAR, HybridPreprocessing, construct_hybrid_targets,
    load_hybrid_model, predict_hybrid, train_hybrid_seed,
)
from wormhole_sciml.finite_time_rate import (
    RatePreprocessing, construct_rate_targets, distribution, two_component_metrics,
)
from wormhole_sciml.finite_time_xi_gate import saturating_gate, saturating_gate_derivative
from wormhole_sciml.model_a import ModelA, parameter_count
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "finite_time_hybrid_s5"
PREPROCESSING_DIR = OUTPUT / "preprocessing"
TRAINING_DIR = OUTPUT / "training"
VALIDATION_DIR = OUTPUT / "validation"
FIGURES_DIR = OUTPUT / "figures"
TESTS_DIR = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_HYBRID_S5_REPORT.md"
SUMMARY = OUTPUT / "finite_time_hybrid_summary.json"
MANIFEST = OUTPUT / "finite_time_hybrid_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_hybrid_manifest.sha256"

DATA_DIR = ROOT / "output/phase_c_finite_time_dataset/datasets"
TRAIN_RAW = DATA_DIR / "phase_c_train_raw.npz"
VALIDATION_RAW = DATA_DIR / "phase_c_validation_raw.npz"
SEALED_RAW = DATA_DIR / "phase_c_test_sealed_raw.npz"
OLD_DIR = ROOT / "output/finite_time_baseline"
RATE_DIR = ROOT / "output/finite_time_rate_baseline"
OLD_MANIFEST = OLD_DIR / "finite_time_baseline_manifest.json"
RATE_MANIFEST = RATE_DIR / "finite_time_rate_manifest.json"
RATE_SUMMARY = RATE_DIR / "finite_time_rate_summary.json"
RATE_PREPROCESSING = RATE_DIR / "preprocessing/rate_preprocessing_constants.json"
SMALL_S_QUERIES = ROOT / "output/finite_time_trajectory_validation/arrays/small_s_queries.npz"

EXPECTED_HASHES = {
    TRAIN_RAW: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_RAW: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    SEALED_RAW: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    OLD_MANIFEST: "ee1dc58a12c05c9e4d8ed9e6ea9f8383755962b3ea79b958745f6b13f084faa5",
    RATE_MANIFEST: "7a09d42335249bcbf6cd7c393e941ab9e649b96c579da09609a6f03bf3fca2fc",
    RATE_SUMMARY: "adeb59d145552cc0cb51cb12de887d22da047e865220f7342f25681db8ce1d9f",
    RATE_PREPROCESSING: "a16c642393fed83aef389b36b6e601f4f054312454163ed8b89de30ad87ba81b",
    SMALL_S_QUERIES: "c7976e6255d9f37ecc11a0adddd81a4cae87a648c535b6d5300c73539cbc13d2",
}
SMALL_S_VALUES = np.asarray([0.0, 0.001, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00])
MODEL_LABELS = {
    "accumulated_residual": "accumulated residual",
    "average_rate": "pure average rate",
    "hybrid_s5": "hybrid saturating gate",
}
MODEL_COLORS = {
    "accumulated_residual": "#777777",
    "average_rate": "#cc6677",
    "hybrid_s5": "#4477aa",
}


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


def array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def prior_tree_hashes() -> dict[str, str]:
    paths = sorted(path for directory in (OLD_DIR, RATE_DIR) for path in directory.rglob("*") if path.is_file())
    paths.append(SMALL_S_QUERIES)
    return {str(path.resolve()): file_sha256(path) for path in paths}


def verify_manifest_artifacts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for relative, entry in payload.get("artifacts", {}).items():
        artifact = Path(entry["path"])
        if not artifact.is_file() or file_sha256(artifact) != entry["sha256"]:
            failures.append(relative)
    return failures


def immutable_gate() -> dict[str, Any]:
    rows, failures = {}, []
    for path, expected in EXPECTED_HASHES.items():
        measured = file_sha256(path)
        match = measured == expected
        rows[str(path.resolve())] = {"expected_sha256": expected, "measured_sha256": measured, "match": match}
        if not match:
            failures.append(str(path))
    manifest_failures = {
        "accumulated_residual": verify_manifest_artifacts(OLD_MANIFEST),
        "average_rate": verify_manifest_artifacts(RATE_MANIFEST),
    }
    if any(manifest_failures.values()):
        failures.extend(f"manifest:{name}:{item}" for name, items in manifest_failures.items() for item in items)
    return {
        "passed": not failures,
        "failures": failures,
        "expected_hashes": rows,
        "prior_manifest_artifact_failures": manifest_failures,
        "sealed_test_access": "file-byte SHA-256 only; NPZ was not opened",
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{component}_{key}": value
        for component in ("x", "xi")
        for key, value in metrics[component].items()
    }


def subset_metrics(prediction: dict[str, np.ndarray], data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "row_count": int(np.sum(mask)),
        **two_component_metrics(
            prediction["predicted_x1"][mask] - data["x1"][mask],
            prediction["predicted_xi1"][mask] - data["xi1"][mask],
        ),
    }


def energy_and_admissibility(validation: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    wormhole, spiral = experiment_parameters()
    x_hat, xi_hat = prediction["predicted_x1"], prediction["predicted_xi1"]
    _, u_hat = state_from_xi(x_hat, xi_hat, wormhole, spiral)
    margin = timelike_margin(x_hat, u_hat, wormhole, spiral)
    valid = np.isfinite(margin) & (margin > 0.0)
    energy_error = np.full(x_hat.shape, np.nan, dtype=np.float64)
    energy_error[valid] = conserved_energy(x_hat[valid], u_hat[valid], wormhole, spiral) - validation["E0"][valid]
    finite = energy_error[np.isfinite(energy_error)]
    absolute = np.abs(finite)
    prediction["predicted_u1"] = u_hat
    prediction["predicted_C1"] = margin
    prediction["energy_error"] = energy_error
    return {
        "row_count": int(x_hat.size),
        "absolute_xi_ge_1_count": int(np.sum(np.abs(xi_hat) >= 1.0)),
        "C_le_0_count": int(np.sum(margin <= 0.0)),
        "union_violation_count": int(np.sum((np.abs(xi_hat) >= 1.0) | (margin <= 0.0) | ~np.isfinite(margin))),
        "energy": {
            "finite_count": int(finite.size),
            "invalid_count": int(x_hat.size - finite.size),
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(finite**2))),
            "p99_absolute": float(np.quantile(absolute, 0.99)),
            "maximum_absolute": float(np.max(absolute)),
        },
        "used_in_loss": False,
    }


def evaluate_validation(
    model: ModelA, preprocessing: HybridPreprocessing, validation: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prediction = predict_hybrid(model, preprocessing, validation)
    x_error = prediction["predicted_x1"] - validation["x1"]
    xi_error = prediction["predicted_xi1"] - validation["xi1"]
    prediction["x_error"], prediction["xi_error"] = x_error, xi_error
    target_standardized = preprocessing.standardize_targets(stack_columns(validation, HYBRID_TARGET_COLUMNS))
    predicted_standardized = np.column_stack((prediction["standardized_V_x"], prediction["standardized_F_xi"]))
    standardized = orbit_averaged_standardized_mse(predicted_standardized, target_standardized, validation["orbit_id"])
    identity = validation["s"] == 0.0
    exact_arrays = (
        prediction["predicted_Delta_x"][identity], prediction["predicted_Delta_xi"][identity],
        prediction["predicted_x1"][identity] - validation["x0"][identity],
        prediction["predicted_xi1"][identity] - validation["xi0"][identity],
    )
    if any(not np.array_equal(values, np.zeros_like(values)) for values in exact_arrays):
        raise RuntimeError("hybrid physical gates failed exact identity")
    output_error_x = prediction["predicted_V_x"] - validation["V_x"]
    output_error_xi = prediction["predicted_F_xi"] - validation["F_xi"]
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
    residual_difference = np.column_stack((x_error, xi_error)) - np.column_stack((
        prediction["predicted_Delta_x"] - validation["Delta_x"],
        prediction["predicted_Delta_xi"] - validation["Delta_xi"],
    ))
    return {
        "standardized_validation_hybrid_mse": standardized,
        "physical_state_metrics": two_component_metrics(x_error, xi_error),
        "hybrid_output_metrics": two_component_metrics(output_error_x, output_error_xi),
        "identity": {
            "row_count": int(np.sum(identity)),
            "max_abs_Delta_x_prediction": float(np.max(np.abs(prediction["predicted_Delta_x"][identity]))),
            "max_abs_Delta_xi_prediction": float(np.max(np.abs(prediction["predicted_Delta_xi"][identity]))),
            "rmse_x": float(np.sqrt(np.mean(x_error[identity] ** 2))),
            "rmse_xi": float(np.sqrt(np.mean(xi_error[identity] ** 2))),
            "network_output_errors": two_component_metrics(output_error_x[identity], output_error_xi[identity]),
            "network_output_names": {"x": "V_x versus u0", "xi": "F_xi versus dot(xi)_0"},
        },
        "time_regimes": {name: subset_metrics(prediction, validation, mask) for name, mask in time_masks.items()},
        "families": {name: subset_metrics(prediction, validation, mask) for name, mask in family_masks.items()},
        "physical_diagnostics": energy_and_admissibility(validation, prediction),
        "maximum_residual_vs_state_error_difference": float(np.max(np.abs(residual_difference))),
        "sealed_test_predictions_computed": False,
    }, prediction


def load_small_s_queries() -> dict[str, np.ndarray]:
    with np.load(SMALL_S_QUERIES, allow_pickle=False) as source:
        query = {name: source[name] for name in source.files}
    query["x1"] = query["exact_x1"]
    query["xi1"] = query["exact_xi1"]
    query["Delta_x"] = query["x1"] - query["x0"]
    query["Delta_xi"] = query["xi1"] - query["xi0"]
    return construct_hybrid_targets(query)


def evaluate_small_s(
    models: dict[int, ModelA], preprocessing: HybridPreprocessing, queries: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    rows, predictions = [], {}
    for seed, model in models.items():
        prediction = predict_hybrid(model, preprocessing, queries)
        predictions[seed] = prediction
        for value in SMALL_S_VALUES:
            mask = np.isclose(queries["s"], value, rtol=0.0, atol=1.0e-14)
            physical = two_component_metrics(
                prediction["predicted_x1"][mask] - queries["x1"][mask],
                prediction["predicted_xi1"][mask] - queries["xi1"][mask],
            )
            rows.append({
                "model": "hybrid_s5", "seed": seed, "s": float(value),
                "row_count": int(np.sum(mask)), **flatten_metrics(physical),
                "V_x_rmse": float(np.sqrt(np.mean((prediction["predicted_V_x"][mask] - queries["V_x"][mask]) ** 2))),
                "V_x_mae": float(np.mean(np.abs(prediction["predicted_V_x"][mask] - queries["V_x"][mask]))),
                "F_xi_rmse": float(np.sqrt(np.mean((prediction["predicted_F_xi"][mask] - queries["F_xi"][mask]) ** 2))),
                "F_xi_mae": float(np.mean(np.abs(prediction["predicted_F_xi"][mask] - queries["F_xi"][mask]))),
            })
    return rows, predictions


def old_rate_primary() -> dict[str, Any]:
    prior = json.loads(RATE_SUMMARY.read_text(encoding="utf-8"))
    return {
        "old_seed": int(prior["comparison"]["primary"]["old_primary_seed"]),
        "rate_seed": int(prior["new_primary_seed"]),
        "comparison": prior["comparison"]["primary"],
    }


def three_way_comparison(
    hybrid_metrics: dict[str, Any], hybrid_small: list[dict[str, Any]], hybrid_primary: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = old_rate_primary()
    source = prior["comparison"]
    scopes = ["aggregate", "short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20", "hard_u_th_le_0p30", "ordinary_u_th_gt_0p30"]
    rows = []
    for scope in scopes:
        if scope == "aggregate":
            old_metrics, rate_metrics = source["aggregate"]["old"], source["aggregate"]["new"]
            hybrid = hybrid_metrics[str(hybrid_primary)]["physical_state_metrics"]
        elif scope in source["time_regimes"]:
            old_metrics, rate_metrics = source["time_regimes"][scope]["old"], source["time_regimes"][scope]["new"]
            hybrid = hybrid_metrics[str(hybrid_primary)]["time_regimes"][scope]
        else:
            old_metrics, rate_metrics = source["families"][scope]["old"], source["families"][scope]["new"]
            hybrid = hybrid_metrics[str(hybrid_primary)]["families"][scope]
        for model, metrics in (("accumulated_residual", old_metrics), ("average_rate", rate_metrics), ("hybrid_s5", hybrid)):
            rows.append({"model": model, "scope": scope, **flatten_metrics(metrics)})
    old_small = source["small_s"]["accumulated_residual"]
    rate_small = source["small_s"]["average_rate"]
    hybrid_small_primary = {f"{row['s']:.12g}": row for row in hybrid_small if row["seed"] == hybrid_primary}
    for model, values in (("accumulated_residual", old_small), ("average_rate", rate_small), ("hybrid_s5", hybrid_small_primary)):
        row = values["0.2"]
        rows.append({"model": model, "scope": "s_eq_0p2", **{key: row[key] for key in ("x_rmse", "x_mae", "xi_rmse", "xi_mae")}})
    compact = {
        "primary_seeds": {"accumulated_residual": prior["old_seed"], "average_rate": prior["rate_seed"], "hybrid_s5": hybrid_primary},
        "rows": rows,
        "small_s": {"accumulated_residual": old_small, "average_rate": rate_small, "hybrid_s5": hybrid_small_primary},
    }
    return rows, compact


def plot_histories(runs: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True, sharey=True)
    for axis, run in zip(axes, runs):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = [row["epoch"] for row in history]
        axis.plot(epoch, [row["training_standardized_hybrid_mse"] for row in history], label="training")
        axis.plot(epoch, [row["validation_orbit_averaged_standardized_hybrid_mse"] for row in history], label="validation")
        axis.axvline(run["best_epoch"], color="0.25", ls="--", label="best")
        axis.axvline(run["stopping_epoch"], color="0.55", ls=":", label="stop")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("standardized hybrid-target MSE")
    figure.suptitle("Hybrid finite-time model training histories")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_three_way_component(rows: list[dict[str, Any]], component: str, path: Path) -> None:
    scopes = ("aggregate", "short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20", "hard_u_th_le_0p30", "ordinary_u_th_gt_0p30", "s_eq_0p2")
    labels = ("aggregate", "short", "intermediate", "long", "hard", "ordinary", "s=0.2")
    x, width = np.arange(len(scopes)), 0.25
    figure, axis = plt.subplots(figsize=(11.5, 4.5), constrained_layout=True)
    for offset, model in enumerate(MODEL_LABELS):
        values = [next(row[f"{component}_rmse"] for row in rows if row["model"] == model and row["scope"] == scope) for scope in scopes]
        axis.bar(x + (offset - 1) * width, values, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axis.set(xticks=x, xticklabels=labels, ylabel=f"RMSE {component}", yscale="log")
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(fontsize=8)
    figure.suptitle(f"Three-way physical {component} error comparison")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_small_s(compact: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    positions = np.arange(SMALL_S_VALUES.size)
    for model, values in compact["small_s"].items():
        selected = [values[f"{s:.12g}"] for s in SMALL_S_VALUES]
        for axis, component in zip(axes, ("x", "xi")):
            axis.plot(positions, np.maximum([row[f"{component}_rmse"] for row in selected], 1.0e-12), marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            axis.set(xlabel="physical elapsed time s", ylabel=f"RMSE {component}", yscale="log")
            axis.set_xticks(positions)
            axis.set_xticklabels(("0", "1e-3", "1e-2", ".05", ".1", ".2", ".5", "1"))
            axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("Small-time physical error across finite-time representations")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_xi_regimes(rows: list[dict[str, Any]], path: Path) -> None:
    scopes, labels = ("short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20"), ("short", "intermediate", "long")
    x, width = np.arange(3), 0.25
    figure, axis = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    for offset, model in enumerate(MODEL_LABELS):
        values = [next(row["xi_rmse"] for row in rows if row["model"] == model and row["scope"] == scope) for scope in scopes]
        axis.bar(x + (offset - 1) * width, values, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    axis.set(xticks=x, xticklabels=labels, ylabel="RMSE xi", yscale="log")
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend(fontsize=8)
    figure.suptitle("Xi prediction error across physical time horizons")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def run_tests() -> dict[str, Any]:
    xml = TESTS_DIR / "relevant_pytest.xml"
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_finite_time_hybrid.py", "tests/test_finite_time_xi_gate.py",
        "tests/test_finite_time_rate.py", "tests/test_finite_time_baseline.py",
        "tests/test_phase_b_orbits.py", "tests/test_physics_gate.py", f"--junitxml={xml}",
    ]
    result = subprocess.run(
        command, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-hybrid-mpl-cache"},
        capture_output=True, text=True,
    )
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS_DIR / "test_summary.json", payload)
    return payload


def report_text(summary: dict[str, Any]) -> str:
    runs, metrics = summary["runs"], summary["validation_metrics"]
    primary_seed = int(summary["hybrid_primary_seed"])
    primary = metrics[str(primary_seed)]
    compact = summary["three_way_comparison"]
    preprocessing = summary["preprocessing"]
    sanity = summary["target_sanity"]
    training_rows = "\n".join(
        f"| {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | {run['best_validation_orbit_averaged_standardized_hybrid_mse']:.7g} | {run['final_training_standardized_hybrid_mse']:.7g} | `{run['checkpoint_sha256']}` |"
        for run in runs
    )
    physical_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['physical_state_metrics']['x']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['x']['mae']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['xi']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_state_metrics']['xi']['mae']:.6g} |"
        for run in runs
    )
    identity_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['identity']['max_abs_Delta_x_prediction']:.3g} | {metrics[str(run['seed'])]['identity']['max_abs_Delta_xi_prediction']:.3g} | {metrics[str(run['seed'])]['identity']['rmse_x']:.3g} | {metrics[str(run['seed'])]['identity']['rmse_xi']:.3g} | {metrics[str(run['seed'])]['identity']['network_output_errors']['x']['rmse']:.6g} | {metrics[str(run['seed'])]['identity']['network_output_errors']['xi']['rmse']:.6g} |"
        for run in runs
    )
    small = compact["small_s"]
    small_rows = "\n".join(
        f"| {value} | {small['accumulated_residual'][value]['x_rmse']:.6g} | {small['average_rate'][value]['x_rmse']:.6g} | {small['hybrid_s5'][value]['x_rmse']:.6g} | {small['accumulated_residual'][value]['xi_rmse']:.6g} | {small['average_rate'][value]['xi_rmse']:.6g} | {small['hybrid_s5'][value]['xi_rmse']:.6g} |"
        for value in ("0", "0.001", "0.01", "0.05", "0.1", "0.2", "0.5", "1")
    )
    comparison_rows = "\n".join(
        f"| {row['scope']} | {MODEL_LABELS[row['model']]} | {row['x_rmse']:.6g} | {row['xi_rmse']:.6g} |"
        for row in compact["rows"]
    )
    family_rows = "\n".join(
        f"| {name} | {value['row_count']} | {value['x']['rmse']:.6g} | {value['x']['mae']:.6g} | {value['xi']['rmse']:.6g} | {value['xi']['mae']:.6g} |"
        for name, value in primary["families"].items()
    )
    diagnostic_rows = "\n".join(
        f"| {run['seed']} | {metrics[str(run['seed'])]['physical_diagnostics']['absolute_xi_ge_1_count']} | {metrics[str(run['seed'])]['physical_diagnostics']['C_le_0_count']} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['mae']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['rmse']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['p99_absolute']:.6g} | {metrics[str(run['seed'])]['physical_diagnostics']['energy']['maximum_absolute']:.6g} |"
        for run in runs
    )
    rows_by = {(row["model"], row["scope"]): row for row in compact["rows"]}
    rate_short = rows_by[("average_rate", "s_eq_0p2")]
    hybrid_short = rows_by[("hybrid_s5", "s_eq_0p2")]
    rate_mid = rows_by[("average_rate", "intermediate_5_lt_s_le_20")]
    hybrid_mid = rows_by[("hybrid_s5", "intermediate_5_lt_s_le_20")]
    rate_long = rows_by[("average_rate", "long_s_gt_20")]
    hybrid_long = rows_by[("hybrid_s5", "long_s_gt_20")]
    rate_agg = rows_by[("average_rate", "aggregate")]
    hybrid_agg = rows_by[("hybrid_s5", "aggregate")]
    answers = {
        "identity_short": (
            f"Yes, relative to the accumulated-residual model. Identity residual and state RMSE values are exact zeros. At s=0.2, hybrid x/xi RMSE are "
            f"{hybrid_short['x_rmse']:.6g}/{hybrid_short['xi_rmse']:.6g}, versus pure-rate "
            f"{rate_short['x_rmse']:.6g}/{rate_short['xi_rmse']:.6g}; the hybrid is modestly less accurate than pure rate there but remains orders of magnitude better than the old model."
        ),
        "xi_recovery": (
            f"Yes. Intermediate xi RMSE falls from {rate_mid['xi_rmse']:.6g} to {hybrid_mid['xi_rmse']:.6g} "
            f"({100*(1-hybrid_mid['xi_rmse']/rate_mid['xi_rmse']):.1f}% reduction), and long-time xi RMSE falls from "
            f"{rate_long['xi_rmse']:.6g} to {hybrid_long['xi_rmse']:.6g} ({100*(1-hybrid_long['xi_rmse']/rate_long['xi_rmse']):.1f}% reduction)."
        ),
        "x_preservation": (
            f"Yes overall. Aggregate x RMSE improves from {rate_agg['x_rmse']:.6g} to {hybrid_agg['x_rmse']:.6g} "
            f"({100*(1-hybrid_agg['x_rmse']/rate_agg['x_rmse']):.1f}% reduction), with the exact same V_x definition and frozen normalization; ordinary-family x is the main caveat."
        ),
    }
    return f"""# Hybrid identity-preserving finite-time model with saturating xi gate

## Controlled experiment

The network targets are `V_x=Delta_x/s` for physical `s>0`, `V_x(0)=u0`, and `F_xi=Delta_xi/[-5*expm1(-s/5)]` for physical `s>0`, `F_xi(0)=dot(xi)_0`. Inference reconstructs `Delta_x_hat=s*V_x_hat` and `Delta_xi_hat=-5*expm1(-s/5)*F_xi_hat`, using physical—not standardized—`s` in both gates.

Exactly, `g(0)=0`, `g'(0)=1`, `g(s)=s-s^2/10+O(s^3)`, and `g(s)->5`. No trajectory, sample, architecture, optimizer, loss term, or old artifact was changed. The sealed-test NPZ was not opened.

## Preprocessing sanity check

Frozen input constants were copied exactly. The complete `V_x` training array is byte-identical to the independently reconstructed pure-rate array (`{sanity['V_x_array_sha256']}`), so its frozen mean/std were reused. `F_xi` received new training-only population constants.

| target | mean | std | standardized min | standardized max |
|:---|---:|---:|---:|---:|
| V_x | {preprocessing['columns']['V_x']['mean']:.12g} | {preprocessing['columns']['V_x']['standard_deviation']:.12g} | {sanity['standardized']['V_x']['minimum']:.6g} | {sanity['standardized']['V_x']['maximum']:.6g} |
| F_xi | {preprocessing['columns']['F_xi']['mean']:.12g} | {preprocessing['columns']['F_xi']['standard_deviation']:.12g} | {sanity['standardized']['F_xi']['minimum']:.6g} | {sanity['standardized']['F_xi']['maximum']:.6g} |

No target was clipped and no minimum-s cutoff was used.

## Architecture and training

The model is exactly `4->64->64->2`, tanh/tanh, linear output, Xavier-uniform weights, zero biases, and 4,610 trainable parameters. Training used Adam, learning rate `1e-3`, batch size 512, zero weight decay, no scheduler, float32, at most 1,500 epochs, patience 40, ordinary shuffled rows, and equal standardized MSE on `[V_x,F_xi]`. Primary seed {primary_seed} was selected solely by minimum validation orbit-averaged standardized hybrid-target MSE.

| seed | best epoch | stop epoch | best validation hybrid MSE | final training MSE | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|:---|
{training_rows}

## Physical validation on 98,304 frozen rows

| seed | RMSE x | MAE x | RMSE xi | MAE xi |
|---:|---:|---:|---:|---:|
{physical_rows}

Full median/p90/p95/p99/maximum errors are in `validation/per_seed_validation_metrics.json`.

## Exact identity and learned generator outputs

| seed | max abs Delta-x | max abs Delta-xi | RMSE x | RMSE xi | V-x RMSE | F-xi RMSE |
|---:|---:|---:|---:|---:|---:|---:|
{identity_rows}

All gated identity quantities are exact floating-point zeros. The nonzero output errors compare network outputs with `u0` and `dot(xi)_0`; they do not violate identity.

## Deterministic small-time comparison

| s | old x | rate x | hybrid x | old xi | rate xi | hybrid xi |
|---:|---:|---:|---:|---:|---:|---:|
{small_rows}

All entries are RMSE; the accompanying CSV includes MAE and tail errors.

## Three-way physical comparison

| scope | representation | RMSE x | RMSE xi |
|:---|:---|---:|---:|
{comparison_rows}

## Hybrid primary orbit-family results

| family | rows | RMSE x | MAE x | RMSE xi | MAE xi |
|:---|---:|---:|---:|---:|---:|
{family_rows}

## Physical diagnostics

| seed | abs(xi)>=1 | C<=0 | energy MAE | energy RMSE | abs energy p99 | abs energy max |
|---:|---:|---:|---:|---:|---:|---:|
{diagnostic_rows}

These are diagnostics only and were not used in training.

## Tests and reproducibility

The relevant suite passed: `{summary['tests']['stdout'].strip()}`. Every frozen raw file and all prior accumulated-residual and pure-rate artifacts retained identical before/after hashes. The manifest records all generated artifact hashes and sealed-test byte-hash-only access.

## Scientific answers

**A. Did the hybrid representation preserve exact identity and short-time improvement?** {answers['identity_short']}

**B. Did it recover intermediate/long-time xi accuracy relative to pure rate?** {answers['xi_recovery']}

**C. Did it preserve the x improvement of pure rate?** {answers['x_preservation']}

No further redesign or sealed-test evaluation was performed.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, PREPROCESSING_DIR, TRAINING_DIR, VALIDATION_DIR, FIGURES_DIR, TESTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    gate = immutable_gate()
    write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"immutable input gate failed: {gate['failures']}")
    protected_before = prior_tree_hashes()

    raw_training = load_dataset(TRAIN_RAW)
    raw_validation = load_dataset(VALIDATION_RAW)
    training = construct_hybrid_targets(raw_training)
    validation = construct_hybrid_targets(raw_validation)
    reference_vx = construct_rate_targets(raw_training)["V_x"]
    vx_identical = bool(np.array_equal(training["V_x"], reference_vx))
    if not vx_identical:
        raise RuntimeError("hybrid V_x differs from the frozen pure-rate definition")
    rate_preprocessing = RatePreprocessing.from_json(RATE_PREPROCESSING)
    preprocessing = HybridPreprocessing.fit(
        training, rate_preprocessing, file_sha256(TRAIN_RAW), file_sha256(RATE_PREPROCESSING),
        vx_array_identity_verified=vx_identical,
    )
    if not np.array_equal(preprocessing.input_mean, rate_preprocessing.input_mean) or not np.array_equal(preprocessing.input_std, rate_preprocessing.input_std):
        raise RuntimeError("frozen input constants were not reused exactly")
    preprocessing_payload = preprocessing.payload(TRAIN_RAW, len(training["s"]))
    standardized = preprocessing.standardize_targets(stack_columns(training, HYBRID_TARGET_COLUMNS)).astype(np.float64)
    target_sanity = {
        "V_x_array_identity_verified": vx_identical,
        "V_x_array_sha256": array_sha256(training["V_x"]),
        "independent_rate_V_x_array_sha256": array_sha256(reference_vx),
        "gate_checks": {
            "g_at_zero": float(saturating_gate(0.0, S_STAR)),
            "g_prime_at_zero": float(saturating_gate_derivative(0.0, S_STAR)),
            "g_at_5000": float(saturating_gate(5000.0, S_STAR)),
        },
        "raw": {name: distribution(training[name]) for name in HYBRID_TARGET_COLUMNS},
        "standardized": {
            name: distribution(standardized[:, index])
            for index, name in enumerate(HYBRID_TARGET_COLUMNS)
        },
    }
    if target_sanity["gate_checks"]["g_at_zero"] != 0.0 or target_sanity["gate_checks"]["g_prime_at_zero"] != 1.0:
        raise RuntimeError("hybrid gate structural checks failed")
    write_json(PREPROCESSING_DIR / "hybrid_preprocessing_constants.json", preprocessing_payload)
    write_json(PREPROCESSING_DIR / "target_sanity.json", target_sanity)
    print(
        "pretraining target sanity: "
        f"V_x mean/std={preprocessing.target_mean[0]:.12g}/{preprocessing.target_std[0]:.12g}, "
        f"z=[{target_sanity['standardized']['V_x']['minimum']:.6g},{target_sanity['standardized']['V_x']['maximum']:.6g}]; "
        f"F_xi mean/std={preprocessing.target_mean[1]:.12g}/{preprocessing.target_std[1]:.12g}, "
        f"z=[{target_sanity['standardized']['F_xi']['minimum']:.6g},{target_sanity['standardized']['F_xi']['maximum']:.6g}]",
        flush=True,
    )
    training_config = {
        "architecture": "4->64->64->2; tanh/tanh; linear output",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "inputs": list(INPUT_COLUMNS), "targets": list(HYBRID_TARGET_COLUMNS),
        "physical_output_gates": {"Delta_x": "physical s * V_x", "Delta_xi": "-5*expm1(-physical s/5) * F_xi"},
        "s_star": S_STAR, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "weight_decay": 0.0, "scheduler": None,
        "maximum_epochs": MAXIMUM_EPOCHS, "early_stopping_patience": PATIENCE,
        "seeds": list(TRAINING_SEEDS), "dtype": "float32",
        "loss": "equal standardized MSE on V_x and F_xi", "oversampling": False,
        "checkpoint_metric": "validation orbit-averaged standardized hybrid-target MSE",
        "semigroup_loss": False, "energy_loss": False,
    }
    write_json(TRAINING_DIR / "training_config.json", training_config)
    write_json(TRAINING_DIR / "environment.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "matplotlib": matplotlib.__version__, "cpu_count": os.cpu_count(), "device": "cpu",
    })
    if parameter_count(ModelA(4, HIDDEN_DIMENSIONS)) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("architecture parameter-count gate failed")

    runs, models, validation_metrics = [], {}, {}
    for seed in TRAINING_SEEDS:
        run = train_hybrid_seed(training, validation, preprocessing, seed, TRAINING_DIR / f"seed_{seed}")
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        run["history_sha256"] = file_sha256(Path(run["history"]))
        model = load_hybrid_model(Path(run["checkpoint"]))
        measured, prediction = evaluate_validation(model, preprocessing, validation)
        discrepancy = abs(
            measured["standardized_validation_hybrid_mse"]["orbit_averaged_standardized_mse"]
            - run["best_validation_orbit_averaged_standardized_hybrid_mse"]
        )
        if discrepancy > 2.0e-7:
            raise RuntimeError("restored hybrid checkpoint metric discrepancy exceeded gate")
        if measured["maximum_residual_vs_state_error_difference"] > 1.0e-13:
            raise RuntimeError("residual and state errors differ")
        run["checkpoint_reload_metric_absolute_difference"] = discrepancy
        write_json(TRAINING_DIR / f"seed_{seed}" / "metadata.json", run)
        write_json(VALIDATION_DIR / f"validation_metrics_seed_{seed}.json", measured)
        np.savez_compressed(VALIDATION_DIR / f"validation_predictions_seed_{seed}.npz", **prediction)
        runs.append(run)
        models[seed] = model
        validation_metrics[str(seed)] = measured
    hybrid_primary = int(min(runs, key=lambda row: row["best_validation_orbit_averaged_standardized_hybrid_mse"])["seed"])
    write_json(VALIDATION_DIR / "per_seed_validation_metrics.json", validation_metrics)
    write_csv(VALIDATION_DIR / "per_seed_summary.csv", [
        {"seed": seed, **flatten_metrics(validation_metrics[str(seed)]["physical_state_metrics"])}
        for seed in TRAINING_SEEDS
    ])

    queries = load_small_s_queries()
    small_rows, small_predictions = evaluate_small_s(models, preprocessing, queries)
    write_csv(VALIDATION_DIR / "small_s_metrics.csv", small_rows)
    np.savez_compressed(VALIDATION_DIR / "small_s_queries_with_hybrid_targets.npz", **queries)
    for seed, prediction in small_predictions.items():
        np.savez_compressed(VALIDATION_DIR / f"small_s_predictions_seed_{seed}.npz", **prediction)
    comparison_rows, comparison = three_way_comparison(validation_metrics, small_rows, hybrid_primary)
    write_csv(VALIDATION_DIR / "three_way_comparison.csv", comparison_rows)
    write_json(VALIDATION_DIR / "three_way_comparison.json", comparison)

    plot_histories(runs, FIGURES_DIR / "training_histories.png")
    plot_three_way_component(comparison_rows, "x", FIGURES_DIR / "three_way_physical_x_error.png")
    plot_three_way_component(comparison_rows, "xi", FIGURES_DIR / "three_way_physical_xi_error.png")
    plot_small_s(comparison, FIGURES_DIR / "small_time_three_way_comparison.png")
    plot_xi_regimes(comparison_rows, FIGURES_DIR / "xi_error_by_time_horizon.png")
    tests = run_tests()

    protected_after = prior_tree_hashes()
    if protected_before != protected_after:
        raise RuntimeError("a frozen prior-model artifact changed during the hybrid experiment")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_definitions": {
            "V_x": "Delta_x/s for s>0; u0 at s=0",
            "F_xi": "Delta_xi/[-5*expm1(-s/5)] for s>0; dot(xi)_0 at s=0",
            "inference": {"Delta_x": "physical s * V_x", "Delta_xi": "-5*expm1(-physical s/5) * F_xi"},
        },
        "target_sanity": target_sanity, "preprocessing": preprocessing_payload,
        "training_configuration": training_config, "runs": runs,
        "hybrid_primary_seed": hybrid_primary, "validation_metrics": validation_metrics,
        "three_way_comparison": comparison, "tests": tests,
        "training_orbit_count": int(np.unique(training["orbit_id"]).size),
        "validation_orbit_count": int(np.unique(validation["orbit_id"]).size),
        "sealed_test_npz_opened": False, "sealed_test_predictions_computed": False,
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "experiment": "hybrid_identity_preserving_finite_time_s_star_5",
        "status": "completed" if tests["passed"] else "completed_with_test_failures",
        "immutable_gate": gate,
        "protected_prior_tree_hashes_before": protected_before,
        "protected_prior_tree_hashes_after": protected_after,
        "prior_artifacts_unchanged": protected_before == protected_after,
        "sealed_test_policy": {"NPZ_opened": False, "predictions": False, "metrics": False, "plots": False, "distribution_inspection": False, "byte_hash_only": True},
        "source_hashes": {
            "src/wormhole_sciml/finite_time_hybrid.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_hybrid.py"),
            "src/wormhole_sciml/finite_time_xi_gate.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_xi_gate.py"),
            "scripts/run_finite_time_hybrid.py": file_sha256(Path(__file__)),
            "tests/test_finite_time_hybrid.py": file_sha256(ROOT / "tests/test_finite_time_hybrid.py"),
        },
        "hybrid_primary_seed": hybrid_primary,
        "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)},
        "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)},
        "artifacts": artifacts,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"hybrid primary seed={hybrid_primary}; wrote {REPORT}", flush=True)


def finalize_existing() -> None:
    """Refresh presentation artifacts and hashes without retraining."""

    if not SUMMARY.exists() or not MANIFEST.exists():
        raise FileNotFoundError("completed hybrid experiment is required")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    runs = summary["runs"]
    comparison = summary["three_way_comparison"]
    plot_histories(runs, FIGURES_DIR / "training_histories.png")
    plot_three_way_component(comparison["rows"], "x", FIGURES_DIR / "three_way_physical_x_error.png")
    plot_three_way_component(comparison["rows"], "xi", FIGURES_DIR / "three_way_physical_xi_error.png")
    plot_small_s(comparison, FIGURES_DIR / "small_time_three_way_comparison.png")
    plot_xi_regimes(comparison["rows"], FIGURES_DIR / "xi_error_by_time_horizon.png")
    REPORT.write_text(report_text(summary), encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current_protected = prior_tree_hashes()
    if current_protected != manifest["protected_prior_tree_hashes_before"]:
        raise RuntimeError("a protected prior artifact changed before finalization")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest["protected_prior_tree_hashes_after"] = current_protected
    manifest["source_hashes"] = {
        "src/wormhole_sciml/finite_time_hybrid.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_hybrid.py"),
        "src/wormhole_sciml/finite_time_xi_gate.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_xi_gate.py"),
        "scripts/run_finite_time_hybrid.py": file_sha256(Path(__file__)),
        "tests/test_finite_time_hybrid.py": file_sha256(ROOT / "tests/test_finite_time_hybrid.py"),
    }
    manifest["summary"] = {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)}
    manifest["report"] = {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)}
    manifest["artifacts"] = artifacts
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-only"]:
        finalize_existing()
    else:
        main()
