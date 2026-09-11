#!/usr/bin/env python3
"""Validation-only shared-vs-split finite-time architecture comparison.

Consumes the six guarded split-head runs, the six retained shared controls,
the frozen validation rows/regions, and the established dense reference
trajectories.  It never trains a model, changes checkpoint selection, or
touches the sealed test set.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_finite_time_phase_space import (
    BIN_DEFINITIONS,
    bin_mask,
    decompose_phase_error,
    load_frozen_regions,
    metric_record,
)
from wormhole_sciml.dynamics import conserved_energy, radial_acceleration
from wormhole_sciml.finite_time_derivative_audit import predict_xi_and_physical_s_derivative
from wormhole_sciml.finite_time_derivative_experiment import treatment_validation_diagnostics
from wormhole_sciml.finite_time_hybrid import (
    HybridPreprocessing,
    construct_hybrid_targets,
    load_hybrid_model,
    predict_hybrid,
)
from wormhole_sciml.phase_b_orbits import file_sha256
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi, xi_time_derivative


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "output/finite_time_split_head_derivative_experiment"
SHARED = ROOT / "output/finite_time_hybrid_derivative_loss_experiment"
PHASE_AUDIT = ROOT / "output/finite_time_hybrid_phase_space_audit"
OUTPUT = ROOT / "output/finite_time_split_head_architecture_comparison"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
ARRAYS = OUTPUT / "arrays"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_SPLIT_HEAD_ARCHITECTURE_COMPARISON.md"
SUMMARY = OUTPUT / "finite_time_split_head_architecture_comparison_summary.json"
MANIFEST = OUTPUT / "finite_time_split_head_architecture_comparison_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_split_head_architecture_comparison_manifest.sha256"

TRAIN_ROWS = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_train_raw.npz"
VALIDATION_ROWS = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_validation_raw.npz"
PREPROCESSING = ROOT / "output/finite_time_hybrid_s5/preprocessing/hybrid_preprocessing_constants.json"
SHARED_CHECKPOINTS = SHARED / "checkpoint_manifest.json"
SPLIT_MANIFEST = SPLIT / "split_head_training_manifest.json"
SHARED_MANIFEST = SHARED / "finite_time_hybrid_derivative_loss_manifest.json"
PHASE_MANIFEST = PHASE_AUDIT / "finite_time_hybrid_phase_space_audit_manifest.json"
FEATURE_WINDOWS = SHARED / "protocol/frozen_validation_feature_windows.csv"
REGION_REALIZATION = SHARED / "protocol/frozen_regional_rule_realization.json"
REFERENCE_TIMING = SHARED / "tables/dense_reference_feature_timing.csv"

ARCHITECTURES = ("shared", "split_head")
LAMBDAS = (0.0, 0.034)
SEEDS = (101, 202, 303)
TARGETS = (0.05, 0.15, 0.30)
CELL_ORDER = (("shared", 0.0), ("split_head", 0.0), ("shared", 0.034), ("split_head", 0.034))
CELL_LABELS = {
    ("shared", 0.0): "shared, lambda=0",
    ("split_head", 0.0): "split, lambda=0",
    ("shared", 0.034): "shared, lambda=0.034",
    ("split_head", 0.034): "split, lambda=0.034",
}
COLORS = {
    ("shared", 0.0): "#666666",
    ("split_head", 0.0): "#cc6677",
    ("shared", 0.034): "#4477aa",
    ("split_head", 0.034): "#228833",
}
IMPORTANT_METRICS = (
    "validation_selection_mse",
    "global_x_rmse",
    "low_u_x_rmse",
    "low_u_post_x_rmse",
    "global_xi_rmse",
    "hard_xi_rmse",
    "hard_long_xi_rmse",
    "during_dot_xi_rmse",
    "post_xi_rmse",
    "low_u_post_e_perp_rmse",
    "low_u_post_e_parallel_rmse",
    "dense_0p05_e_perp_rmse",
    "dense_0p05_e_parallel_rmse",
    "dense_0p05_x_rmse",
    "energy_rmse",
)
METRIC_LABELS = {
    "validation_selection_mse": "validation selection MSE",
    "global_x_rmse": "global x RMSE",
    "low_u_x_rmse": "lowest-bin global x RMSE",
    "low_u_post_x_rmse": "lowest-bin post-change x RMSE",
    "global_xi_rmse": "global xi RMSE",
    "hard_xi_rmse": "hard xi RMSE",
    "hard_long_xi_rmse": "hard long-horizon xi RMSE",
    "during_dot_xi_rmse": "rapid-change dot-xi RMSE",
    "post_xi_rmse": "post-change xi RMSE",
    "low_u_post_e_perp_rmse": "lowest-bin post-change e_perp RMSE",
    "low_u_post_e_parallel_rmse": "lowest-bin post-change e_parallel RMSE",
    "dense_0p05_e_perp_rmse": "dense u_th=0.05 e_perp RMSE",
    "dense_0p05_e_parallel_rmse": "dense u_th=0.05 e_parallel RMSE",
    "dense_0p05_x_rmse": "dense u_th=0.05 x RMSE",
    "energy_rmse": "energy RMSE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def slug(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.3f}".replace(".", "p")


def protected_inputs(split_manifest: dict[str, Any]) -> dict[Path, str]:
    expected = {
        TRAIN_ROWS: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
        VALIDATION_ROWS: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
        PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
        SPLIT_MANIFEST: file_sha256(SPLIT_MANIFEST),
    }
    shared_manifest = json.loads(SHARED_MANIFEST.read_text(encoding="utf-8"))
    phase_manifest = json.loads(PHASE_MANIFEST.read_text(encoding="utf-8"))
    expected[SHARED_MANIFEST] = file_sha256(SHARED_MANIFEST)
    expected[PHASE_MANIFEST] = file_sha256(PHASE_MANIFEST)
    for name in (
        "checkpoint_manifest.json",
        "protocol/frozen_validation_feature_windows.csv",
        "protocol/frozen_regional_rule_realization.json",
        "tables/dense_reference_feature_timing.csv",
    ):
        expected[SHARED / name] = shared_manifest["artifacts"][name]["sha256"]
    for value in LAMBDAS:
        for seed in SEEDS:
            name = f"validation/predictions_lambda_{slug(value)}_seed_{seed}.npz"
            expected[SHARED / name] = shared_manifest["artifacts"][name]["sha256"]
    expected[PHASE_AUDIT / "arrays/reference_phase_space_predictions.npz"] = phase_manifest["artifacts"]["arrays/reference_phase_space_predictions.npz"]["sha256"]
    for run in split_manifest["runs"]:
        expected[Path(run["checkpoint"])] = run["checkpoint_sha256"]
        expected[Path(run["history"])] = run["history_sha256"]
    shared_rows = json.loads(SHARED_CHECKPOINTS.read_text(encoding="utf-8"))["checkpoints"]
    for row in shared_rows:
        if float(row["lambda_dot_xi"]) in LAMBDAS and int(row["seed"]) in SEEDS:
            expected[Path(row["checkpoint"])] = row["checkpoint_sha256"]
    return expected


def verify_hashes(expected: dict[Path, str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    failures = []
    for path, target in expected.items():
        measured = file_sha256(path)
        match = measured == target
        rows[str(path.resolve())] = {"expected": target, "measured": measured, "match": match}
        if not match:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"protected artifact mismatch: {failures}")
    return rows


def checkpoint_rows(split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shared = json.loads(SHARED_CHECKPOINTS.read_text(encoding="utf-8"))["checkpoints"]
    for row in shared:
        value, seed = float(row["lambda_dot_xi"]), int(row["seed"])
        if value in LAMBDAS and seed in SEEDS:
            rows.append({"architecture": "shared", **row})
    for run in split_manifest["runs"]:
        rows.append({
            "architecture": "split_head",
            "lambda_dot_xi": float(run["lambda_dot_xi"]),
            "seed": int(run["seed"]),
            "checkpoint": run["checkpoint"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "best_epoch": int(run["best_epoch"]),
            "stopping_epoch": int(run["stopping_epoch"]),
            "selection_metric": float(run["best_validation_orbit_averaged_standardized_hybrid_mse"]),
            "training_wall_seconds": float(run["training_wall_seconds"]),
        })
    rows.sort(key=lambda row: (row["architecture"], float(row["lambda_dot_xi"]), int(row["seed"])))
    expected = {(architecture, value, seed) for architecture in ARCHITECTURES for value in LAMBDAS for seed in SEEDS}
    measured = {(row["architecture"], float(row["lambda_dot_xi"]), int(row["seed"])) for row in rows}
    if measured != expected or len(rows) != 12:
        raise RuntimeError(f"incomplete 2x2x3 checkpoint matrix: {measured}")
    return rows


def dominant_peak_times() -> dict[float, list[float]]:
    with REFERENCE_TIMING.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        target: sorted({
            float(row["exact_peak_s"])
            for row in rows
            if float(row["target_u_th"]) == target
            and int(row["seed"]) == 101
            and float(row["lambda_dot_xi"]) == 0.0
            and float(row["exact_relative_height"]) >= 0.99
        })
        for target in TARGETS
    }


def cell_aggregate(per_run: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for architecture, value in CELL_ORDER:
        group = [row for row in per_run if row["architecture"] == architecture and row["lambda_dot_xi"] == value]
        record: dict[str, Any] = {
            "architecture": architecture,
            "lambda_dot_xi": value,
            "seed_count": len(group),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values, ddof=0))
            record[f"{metric}_min"] = float(np.min(values))
            record[f"{metric}_max"] = float(np.max(values))
        rows.append(record)
    return rows


def sign_consistency(values: list[float]) -> str:
    if all(value < 0.0 for value in values):
        return "all_negative_improvement"
    if all(value > 0.0 for value in values):
        return "all_positive_degradation"
    if all(value == 0.0 for value in values):
        return "all_zero"
    return "mixed"


def paired_architecture_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["architecture"], row["lambda_dot_xi"], row["seed"]): row for row in per_run}
    output = []
    for metric in IMPORTANT_METRICS:
        for value in LAMBDAS:
            differences = []
            for seed in SEEDS:
                shared = float(lookup[("shared", value, seed)][metric])
                split = float(lookup[("split_head", value, seed)][metric])
                delta = split - shared
                differences.append(delta)
                output.append({
                    "metric": metric,
                    "lambda_dot_xi": value,
                    "seed": seed,
                    "shared": shared,
                    "split": split,
                    "delta_split_minus_shared": delta,
                    "relative_delta": delta / shared if shared != 0.0 else None,
                    "row_type": "seed",
                })
            output.append({
                "metric": metric,
                "lambda_dot_xi": value,
                "seed": "mean",
                "shared": float(np.mean([lookup[("shared", value, seed)][metric] for seed in SEEDS])),
                "split": float(np.mean([lookup[("split_head", value, seed)][metric] for seed in SEEDS])),
                "delta_split_minus_shared": float(np.mean(differences)),
                "relative_delta": float(np.mean([
                    (lookup[("split_head", value, seed)][metric] - lookup[("shared", value, seed)][metric])
                    / lookup[("shared", value, seed)][metric]
                    for seed in SEEDS
                ])),
                "row_type": "aggregate",
                "sign_consistency": sign_consistency(differences),
            })
    return output


def split_derivative_rows(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["architecture"], row["lambda_dot_xi"], row["seed"]): row for row in per_run}
    output = []
    for metric in IMPORTANT_METRICS:
        differences = []
        for seed in SEEDS:
            baseline = float(lookup[("split_head", 0.0, seed)][metric])
            derivative = float(lookup[("split_head", 0.034, seed)][metric])
            delta = derivative - baseline
            differences.append(delta)
            output.append({
                "metric": metric,
                "seed": seed,
                "split_lambda_0": baseline,
                "split_lambda_0p034": derivative,
                "delta_0p034_minus_0": delta,
                "relative_delta": delta / baseline if baseline != 0.0 else None,
                "row_type": "seed",
            })
        output.append({
            "metric": metric,
            "seed": "mean",
            "split_lambda_0": float(np.mean([lookup[("split_head", 0.0, seed)][metric] for seed in SEEDS])),
            "split_lambda_0p034": float(np.mean([lookup[("split_head", 0.034, seed)][metric] for seed in SEEDS])),
            "delta_0p034_minus_0": float(np.mean(differences)),
            "relative_delta": float(np.mean([
                (lookup[("split_head", 0.034, seed)][metric] - lookup[("split_head", 0.0, seed)][metric])
                / lookup[("split_head", 0.0, seed)][metric]
                for seed in SEEDS
            ])),
            "row_type": "aggregate",
            "sign_consistency": sign_consistency(differences),
        })
    return output


def interaction_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate = {(row["metric"], row["lambda_dot_xi"]): row for row in paired if row["row_type"] == "aggregate"}
    rows = []
    for metric in IMPORTANT_METRICS:
        zero = aggregate[(metric, 0.0)]
        derivative = aggregate[(metric, 0.034)]
        rows.append({
            "metric": metric,
            "architecture_delta_lambda_0": zero["delta_split_minus_shared"],
            "architecture_delta_lambda_0p034": derivative["delta_split_minus_shared"],
            "interaction": derivative["delta_split_minus_shared"] - zero["delta_split_minus_shared"],
            "interaction_relative_to_shared_lambda_0_mean": (
                (derivative["delta_split_minus_shared"] - zero["delta_split_minus_shared"]) / zero["shared"]
                if zero["shared"] != 0.0 else None
            ),
        })
    return rows


def plot_binned_phase(phase_aggregate: list[dict[str, Any]]) -> None:
    bins = [definition[0] for definition in BIN_DEFINITIONS]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("e_perp_rmse", "e_parallel_rmse"),
        ("post-change orbit-normal error", "post-change tangential/progression error"),
    ):
        for cell in CELL_ORDER:
            rows = [next(row for row in phase_aggregate if (row["architecture"], row["lambda_dot_xi"], row["u_th_bin"]) == (*cell, name)) for name in bins]
            mean = np.asarray([row[f"{metric}_mean"] for row in rows])
            low = np.asarray([row[f"{metric}_min"] for row in rows])
            high = np.asarray([row[f"{metric}_max"] for row in rows])
            ax.plot(x, mean, marker="o", lw=1.5, color=COLORS[cell], label=CELL_LABELS[cell])
            ax.fill_between(x, low, high, color=COLORS[cell], alpha=0.09)
        ax.set_title(title)
        ax.set_ylabel("scaled phase-space RMSE")
        ax.set_xticks(x, bins, rotation=25, ha="right")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen validation bins; lines are seed means, bands are seed ranges")
    fig.savefig(FIGURES / "binned_post_change_phase_errors.png", dpi=180)
    plt.close(fig)


def plot_dense_references(references: dict[float, dict[str, np.ndarray]], predictions: dict[tuple[str, float, int, float], dict[str, np.ndarray]]) -> None:
    peaks = dominant_peak_times()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for column, target in enumerate(TARGETS):
        exact = references[target]
        for row in range(2):
            ax = axes[row, column]
            mask = np.ones(exact["elapsed_s"].size, dtype=bool)
            if row == 1 and peaks[target]:
                mask = exact["elapsed_s"] >= peaks[target][0]
            ax.plot(exact["exact_x"][mask], exact["exact_xi"][mask], color="black", lw=2.2, label="exact")
            for cell in CELL_ORDER:
                curves_x = np.vstack([predictions[(*cell, seed, target)]["predicted_x"] for seed in SEEDS])[:, mask]
                curves_xi = np.vstack([predictions[(*cell, seed, target)]["predicted_xi"] for seed in SEEDS])[:, mask]
                ax.plot(np.mean(curves_x, axis=0), np.mean(curves_xi, axis=0), color=COLORS[cell], lw=1.45, label=CELL_LABELS[cell])
            ax.set(xlabel="x", ylabel="$\\xi$")
            ax.grid(alpha=0.23)
            scope = "full reference" if row == 0 else "post dominant rapid-change peak"
            ax.set_title(f"$u_{{th}}={target:.2f}$: {scope}")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Dense reference phase-space trajectories (colored curves are three-seed means)")
    fig.savefig(FIGURES / "dense_reference_phase_space_comparison.png", dpi=180)
    plt.close(fig)


def plot_safeguards(aggregates: list[dict[str, Any]]) -> None:
    metrics = ("global_x_rmse", "low_u_x_rmse", "global_xi_rmse", "hard_long_xi_rmse")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        means = [next(row for row in aggregates if (row["architecture"], row["lambda_dot_xi"]) == cell)[f"{metric}_mean"] for cell in CELL_ORDER]
        lows = [next(row for row in aggregates if (row["architecture"], row["lambda_dot_xi"]) == cell)[f"{metric}_min"] for cell in CELL_ORDER]
        highs = [next(row for row in aggregates if (row["architecture"], row["lambda_dot_xi"]) == cell)[f"{metric}_max"] for cell in CELL_ORDER]
        positions = np.arange(4)
        ax.bar(positions, means, color=[COLORS[cell] for cell in CELL_ORDER], alpha=0.82)
        ax.errorbar(positions, means, yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(positions, ["shared\n0", "split\n0", "shared\n.034", "split\n.034"])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Global and difficult-region safeguards; bars are seed means, whiskers are seed ranges")
    fig.savefig(FIGURES / "global_and_hard_safeguards.png", dpi=180)
    plt.close(fig)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.7g}"


def report_text(summary: dict[str, Any]) -> str:
    runs = summary["checkpoints"]
    per_run = summary["per_run_metrics"]
    aggregate = summary["aggregate_metrics"]
    paired = summary["paired_architecture_differences"]
    split_derivative = summary["split_derivative_differences"]
    interactions = summary["interactions"]
    dense = summary["dense_reference_metrics"]

    training_lines = []
    for row in runs:
        if row["architecture"] != "split_head":
            continue
        training_lines.append(
            f"| {row['lambda_dot_xi']:.3g} | {row['seed']} | {row['best_epoch']} | {row['stopping_epoch']} | {row['selection_metric']:.7g} | `{Path(row['checkpoint']).relative_to(ROOT)}` |"
        )
    standard_lines = []
    for cell in CELL_ORDER:
        row = next(item for item in aggregate if (item["architecture"], item["lambda_dot_xi"]) == cell)
        standard_lines.append(
            f"| {cell[0]} | {cell[1]:.3g} | {row['validation_selection_mse_mean']:.7g} | {row['global_x_rmse_mean']:.7g} | {row['global_xi_rmse_mean']:.7g} | {row['hard_xi_rmse_mean']:.7g} | {row['hard_long_xi_rmse_mean']:.7g} | {row['during_dot_xi_rmse_mean']:.7g} | {row['post_xi_rmse_mean']:.7g} |"
        )

    architecture_lines = []
    for metric in IMPORTANT_METRICS:
        for value in LAMBDAS:
            row = next(item for item in paired if item["metric"] == metric and item["lambda_dot_xi"] == value and item["row_type"] == "aggregate")
            architecture_lines.append(
                f"| {METRIC_LABELS[metric]} | {value:.3g} | {num(row['shared'])} | {num(row['split'])} | {percent(row['relative_delta'])} | {row['sign_consistency']} |"
            )

    seed_lines = []
    for metric in IMPORTANT_METRICS:
        for value in LAMBDAS:
            group = [item for item in paired if item["metric"] == metric and item["lambda_dot_xi"] == value and item["row_type"] == "seed"]
            mean = next(item for item in paired if item["metric"] == metric and item["lambda_dot_xi"] == value and item["row_type"] == "aggregate")
            seed_lines.append(
                f"| {METRIC_LABELS[metric]} | {value:.3g} | "
                + " | ".join(num(item["delta_split_minus_shared"]) for item in group)
                + f" | {num(mean['delta_split_minus_shared'])} | {mean['sign_consistency']} |"
            )

    derivative_lines = []
    for metric in IMPORTANT_METRICS:
        row = next(item for item in split_derivative if item["metric"] == metric and item["row_type"] == "aggregate")
        derivative_lines.append(
            f"| {METRIC_LABELS[metric]} | {num(row['split_lambda_0'])} | {num(row['split_lambda_0p034'])} | {percent(row['relative_delta'])} | {row['sign_consistency']} |"
        )

    interaction_lines = []
    for row in interactions:
        interaction_lines.append(
            f"| {METRIC_LABELS[row['metric']]} | {num(row['architecture_delta_lambda_0'])} | {num(row['architecture_delta_lambda_0p034'])} | {num(row['interaction'])} | {percent(row['interaction_relative_to_shared_lambda_0_mean'])} |"
        )

    low_lines = []
    for cell in CELL_ORDER:
        group = [row for row in per_run if (row["architecture"], row["lambda_dot_xi"]) == cell]
        low_lines.append(
            f"| {cell[0]} | {cell[1]:.3g} | {np.mean([row['low_u_post_e_perp_rmse'] for row in group]):.7g} | {np.mean([row['low_u_post_e_parallel_rmse'] for row in group]):.7g} | {np.mean([row['low_u_post_Q'] for row in group]):.4f} | {np.mean([row['low_u_post_x_rmse'] for row in group]):.7g} |"
        )

    dense_lines = []
    for target in TARGETS:
        for cell in CELL_ORDER:
            group = [row for row in dense if row["target_u_th"] == target and (row["architecture"], row["lambda_dot_xi"]) == cell]
            dense_lines.append(
                f"| {target:.2f} | {cell[0]} | {cell[1]:.3g} | {np.mean([row['x_xi_e_perp_rmse'] for row in group]):.7g} | {np.mean([row['x_xi_e_parallel_rmse'] for row in group]):.7g} | {np.mean([row['x_rmse'] for row in group]):.7g} |"
            )

    physical_lines = []
    for cell in CELL_ORDER:
        group = [row for row in per_run if (row["architecture"], row["lambda_dot_xi"]) == cell]
        physical_lines.append(
            f"| {cell[0]} | {cell[1]:.3g} | {np.mean([row['energy_rmse'] for row in group]):.7g} | "
            f"{sum(int(row['union_violation_count']) for row in group)} | {sum(int(row['absolute_xi_ge_1_count']) for row in group)} | "
            f"{sum(int(row['C_le_0_count']) for row in group)} | {np.mean([row['low_u_post_x_u_e_perp_rmse'] for row in group]):.7g} | "
            f"{np.mean([row['low_u_post_x_u_e_parallel_rmse'] for row in group]):.7g} |"
        )

    verdict = summary["verdict"]
    return f"""# Parameter-matched split-head architecture experiment

## Integrity and completion

All six authorized split-head runs completed. The comparison uses the six retained shared checkpoints; no shared model was retrained. Dataset, split, preprocessing, targets, gates, derivative normalization, loss normalization, Adam settings, epoch budget, early stopping, deterministic seeds, and the validation-selection criterion remained frozen. Checkpoint selection remained the orbit-averaged standardized hybrid-target MSE. The sealed test set was not opened, hashed, or evaluated.

| lambda | seed | selected epoch | stop epoch | selection MSE | checkpoint |
|---:|---:|---:|---:|---:|:---|
{chr(10).join(training_lines)}

## Standard validation metrics

Values are three-seed means. Full seed-level values and spreads are in `tables/per_run_metrics.csv` and `tables/aggregate_metrics.csv`.

| architecture | lambda | selection MSE | global x RMSE | global xi RMSE | hard xi RMSE | hard-long xi RMSE | rapid dot-xi RMSE | post-change xi RMSE |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(standard_lines)}

## Shared versus split

The percent column is the mean of matched-seed relative changes, `(split-shared)/shared`; negative is improvement for all listed error metrics.

| metric | lambda | shared mean | split mean | paired relative change | seed-sign consistency |
|:---|---:|---:|---:|---:|:---|
{chr(10).join(architecture_lines)}

### All paired seed differences

Absolute differences are `split-shared` in seed order 101, 202, 303.

| metric | lambda | seed 101 | seed 202 | seed 303 | mean | sign |
|:---|---:|---:|---:|---:|---:|:---|
{chr(10).join(seed_lines)}

## Derivative effect within the split architecture

| metric | split lambda=0 | split lambda=.034 | paired relative change | seed-sign consistency |
|:---|---:|---:|---:|---:|:---|
{chr(10).join(derivative_lines)}

## Architecture by derivative interaction

`I = (split-shared)_0.034 - (split-shared)_0`; negative means splitting becomes more beneficial when derivative supervision is present.

| metric | architecture delta at 0 | architecture delta at .034 | interaction I | I / shared-0 mean |
|:---|---:|---:|---:|---:|
{chr(10).join(interaction_lines)}

## Lowest-u_th phase-space result

Frozen lowest bin `[0.01,0.05]`, post-change rows; scaled `(x,xi)` decomposition. `Q=RMSE(e_perp)/RMSE(e_parallel)`.

| architecture | lambda | e_perp RMSE | e_parallel RMSE | Q | x RMSE |
|:---|---:|---:|---:|---:|---:|
{chr(10).join(low_lines)}

## Dense references

Three-seed means over the complete dense reference trajectory.

| u_th | architecture | lambda | scaled e_perp RMSE | scaled e_parallel RMSE | x RMSE |
|---:|:---|---:|---:|---:|---:|
{chr(10).join(dense_lines)}

## Physical safeguards

The physical `(x,u)` decomposition is a coordinate-consistency check; the scaled `(x,xi)` result above remains primary.

| architecture | lambda | energy RMSE | union violations | abs(xi)>=1 | C<=0 | low-bin physical e_perp | low-bin physical e_parallel |
|:---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(physical_lines)}

## Verdict

**{verdict['classification']} — {verdict['headline']}**

{verdict['explanation']}

The later split-head `lambda=0.068` experiment is **{verdict['lambda_0p068_recommendation']}**. It was not launched.

## Compact figures

- `figures/dense_reference_phase_space_comparison.png`
- `figures/binned_post_change_phase_errors.png`
- `figures/global_and_hard_safeguards.png`

Focused numerical tests: `{summary['tests']['stdout'].strip()}`. All protected artifacts were unchanged across the validation-only analysis.
"""


def run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_finite_time_phase_space_audit.py",
        "tests/test_finite_time_derivative_experiment.py",
        "tests/test_finite_time_derivative_audit.py",
        "tests/test_finite_time_split_head.py",
        f"--junitxml={TESTS / 'focused_pytest.xml'}",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-split-comparison-mpl"},
        capture_output=True,
        text=True,
    )
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload)
    return payload


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if split_manifest["status"] != "SIX_SPLIT_HEAD_RUNS_COMPLETED" or split_manifest.get("held_out_test_accessed") is not False:
        raise RuntimeError("split training manifest is incomplete or violates scope")
    if split_manifest["protected_before"] != split_manifest["protected_after"]:
        raise RuntimeError("split training did not preserve protected inputs")
    protocol = split_manifest["protocol"]
    if (
        protocol["architecture"] != "split_head"
        or tuple(protocol["lambda_dot_xi"]) != LAMBDAS
        or tuple(protocol["seeds"]) != SEEDS
        or int(protocol["run_count"]) != 6
    ):
        raise RuntimeError(f"split training protocol is not the frozen six-run matrix: {protocol}")
    for directory in (OUTPUT, TABLES, FIGURES, ARRAYS, TESTS):
        directory.mkdir(parents=True, exist_ok=False)
    expected = protected_inputs(split_manifest)
    protected_before = verify_hashes(expected)
    checkpoints = checkpoint_rows(split_manifest)

    validation = construct_hybrid_targets(load_dataset(VALIDATION_ROWS))
    training = load_dataset(TRAIN_ROWS)
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    scales = {
        "x": float(preprocessing.input_std[0]),
        "xi": float(preprocessing.input_std[1]),
        "u": float(np.std(training["u0"], ddof=0, dtype=np.float64)),
    }
    del training
    regions, _ = load_frozen_regions(validation)
    wormhole, spiral = experiment_parameters()
    exact_dot = xi_time_derivative(validation["x1"], validation["u1"], wormhole, spiral)
    exact_acceleration = radial_acceleration(validation["x1"], validation["u1"], wormhole, spiral)

    per_run: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    phase_physical_rows: list[dict[str, Any]] = []
    validation_arrays: dict[str, np.ndarray] = {}
    maximum_shared_cache_discrepancy = 0.0
    maximum_pythagorean_residual = 0.0
    maximum_physical_pythagorean_residual = 0.0
    for checkpoint in checkpoints:
        architecture = checkpoint["architecture"]
        value = float(checkpoint["lambda_dot_xi"])
        seed = int(checkpoint["seed"])
        model = load_hybrid_model(Path(checkpoint["checkpoint"]))
        diagnostics, prediction = treatment_validation_diagnostics(model, preprocessing, validation, exact_dot, regions)
        if architecture == "shared":
            cached = load_npz(SHARED / f"validation/predictions_lambda_{slug(value)}_seed_{seed}.npz")
            maximum_shared_cache_discrepancy = max(
                maximum_shared_cache_discrepancy,
                float(np.max(np.abs(prediction["predicted_x1"] - cached["predicted_x"]))),
                float(np.max(np.abs(prediction["predicted_xi1"] - cached["predicted_xi"]))),
                float(np.max(np.abs(prediction["predicted_dot_xi1"] - cached["predicted_dot_xi"]))),
            )
        x_error = prediction["predicted_x1"] - validation["x1"]
        xi_error = prediction["predicted_xi1"] - validation["xi1"]
        _, predicted_u = state_from_xi(prediction["predicted_x1"], prediction["predicted_xi1"], wormhole, spiral)
        u_error = predicted_u - validation["u1"]
        decomposition = decompose_phase_error(validation["u1"], exact_dot, x_error, xi_error, scales["x"], scales["xi"])
        physical = decompose_phase_error(validation["u1"], exact_acceleration, x_error, u_error, scales["x"], scales["u"])
        maximum_pythagorean_residual = max(maximum_pythagorean_residual, decomposition["maximum_pythagorean_residual"])
        maximum_physical_pythagorean_residual = max(maximum_physical_pythagorean_residual, physical["maximum_pythagorean_residual"])
        for definition in BIN_DEFINITIONS:
            name = definition[0]
            selection = bin_mask(validation["u_th"], definition) & regions["post"]
            phase_rows.append({
                "architecture": architecture,
                "lambda_dot_xi": value,
                "seed": seed,
                "u_th_bin": name,
                **metric_record(decomposition, x_error, selection),
            })
            phase_physical_rows.append({
                "architecture": architecture,
                "lambda_dot_xi": value,
                "seed": seed,
                "u_th_bin": name,
                **metric_record(physical, x_error, selection),
            })
        lowest = next(row for row in phase_rows if row["architecture"] == architecture and row["lambda_dot_xi"] == value and row["seed"] == seed and row["u_th_bin"] == "[0.01,0.05]")
        lowest_physical = next(row for row in phase_physical_rows if row["architecture"] == architecture and row["lambda_dot_xi"] == value and row["seed"] == seed and row["u_th_bin"] == "[0.01,0.05]")
        lowest_global = metric_record(decomposition, x_error, bin_mask(validation["u_th"], BIN_DEFINITIONS[0]))
        per_run.append({
            "architecture": architecture,
            "lambda_dot_xi": value,
            "seed": seed,
            "best_epoch": int(checkpoint["best_epoch"]),
            "stopping_epoch": checkpoint.get("stopping_epoch"),
            "validation_selection_mse": float(checkpoint["selection_metric"]),
            "global_x_rmse": diagnostics["global"]["x"]["rmse"],
            "global_xi_rmse": diagnostics["global"]["xi"]["rmse"],
            "global_dot_xi_rmse": diagnostics["global"]["dot_xi"]["rmse"],
            "hard_xi_rmse": diagnostics["families"]["hard_u_th_le_0p30"]["xi"]["rmse"],
            "ordinary_xi_rmse": diagnostics["families"]["ordinary_u_th_gt_0p30"]["xi"]["rmse"],
            "hard_long_xi_rmse": diagnostics["horizons"]["hard_long_s_gt_20"]["xi"]["rmse"],
            "during_dot_xi_rmse": diagnostics["regions"]["during"]["dot_xi"]["rmse"],
            "post_xi_rmse": diagnostics["regions"]["post"]["xi"]["rmse"],
            "energy_rmse": diagnostics["physical"]["energy_rmse"],
            "energy_mae": diagnostics["physical"]["energy_mae"],
            "energy_p99_absolute": diagnostics["physical"]["energy_p99_absolute"],
            "energy_maximum_absolute": diagnostics["physical"]["energy_maximum_absolute"],
            "absolute_xi_ge_1_count": diagnostics["physical"]["absolute_xi_ge_1_count"],
            "C_le_0_count": diagnostics["physical"]["C_le_0_count"],
            "union_violation_count": diagnostics["physical"]["union_violation_count"],
            "low_u_x_rmse": lowest_global["x_rmse"],
            "low_u_post_x_rmse": lowest["x_rmse"],
            "low_u_post_e_perp_rmse": lowest["e_perp_rmse"],
            "low_u_post_e_parallel_rmse": lowest["e_parallel_rmse"],
            "low_u_post_Q": lowest["Q_perp_over_parallel"],
            "low_u_post_x_u_e_perp_rmse": lowest_physical["e_perp_rmse"],
            "low_u_post_x_u_e_parallel_rmse": lowest_physical["e_parallel_rmse"],
            "checkpoint": checkpoint["checkpoint"],
        })
        tag = f"{architecture}_lambda_{slug(value)}_seed_{seed}"
        validation_arrays[f"{tag}_predicted_x"] = prediction["predicted_x1"]
        validation_arrays[f"{tag}_predicted_xi"] = prediction["predicted_xi1"]
        validation_arrays[f"{tag}_predicted_dot_xi"] = prediction["predicted_dot_xi1"]

    if maximum_shared_cache_discrepancy > 2.0e-15:
        raise RuntimeError(f"retained shared inference disagrees with cache: {maximum_shared_cache_discrepancy}")
    np.savez_compressed(ARRAYS / "validation_predictions.npz", transition_id=validation["transition_id"], **validation_arrays)

    phase_aggregate = []
    for architecture, value in CELL_ORDER:
        for definition in BIN_DEFINITIONS:
            name = definition[0]
            group = [row for row in phase_rows if row["architecture"] == architecture and row["lambda_dot_xi"] == value and row["u_th_bin"] == name]
            record: dict[str, Any] = {"architecture": architecture, "lambda_dot_xi": value, "u_th_bin": name, "seed_count": len(group)}
            for metric in ("e_perp_rmse", "e_parallel_rmse", "Q_perp_over_parallel", "x_rmse"):
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                record[f"{metric}_mean"] = float(np.mean(values))
                record[f"{metric}_std"] = float(np.std(values, ddof=0))
                record[f"{metric}_min"] = float(np.min(values))
                record[f"{metric}_max"] = float(np.max(values))
            phase_aggregate.append(record)

    # Reuse the exact dense references from the established phase-space audit,
    # while restoring every selected checkpoint for a uniform four-cell pass.
    prior_reference = load_npz(PHASE_AUDIT / "arrays/reference_phase_space_predictions.npz")
    references: dict[float, dict[str, np.ndarray]] = {}
    query_parts = {name: [] for name in ("x0", "xi0", "E0", "s")}
    slices: dict[float, slice] = {}
    offset = 0
    for target in TARGETS:
        tag = f"u_th_{target:.2f}".replace(".", "p")
        exact = {name: prior_reference[f"{tag}_{name}"] for name in ("elapsed_s", "exact_x", "exact_xi", "exact_u", "exact_dot_xi")}
        references[target] = exact
        count = exact["elapsed_s"].size
        slices[target] = slice(offset, offset + count)
        offset += count
        energy = float(conserved_energy(exact["exact_x"][0], exact["exact_u"][0], wormhole, spiral))
        query_parts["x0"].append(np.full(count, exact["exact_x"][0]))
        query_parts["xi0"].append(np.full(count, exact["exact_xi"][0]))
        query_parts["E0"].append(np.full(count, energy))
        query_parts["s"].append(exact["elapsed_s"])
    dense_query = {name: np.concatenate(parts) for name, parts in query_parts.items()}
    dense_predictions: dict[tuple[str, float, int, float], dict[str, np.ndarray]] = {}
    dense_rows: list[dict[str, Any]] = []
    dense_arrays: dict[str, np.ndarray] = {}
    for checkpoint in checkpoints:
        architecture, value, seed = checkpoint["architecture"], float(checkpoint["lambda_dot_xi"]), int(checkpoint["seed"])
        model = load_hybrid_model(Path(checkpoint["checkpoint"]))
        state = predict_hybrid(model, preprocessing, dense_query)
        derivative = predict_xi_and_physical_s_derivative(model, preprocessing, dense_query)
        for target in TARGETS:
            part = slices[target]
            exact = references[target]
            item = {
                "predicted_x": state["predicted_x1"][part],
                "predicted_xi": state["predicted_xi1"][part],
                "predicted_dot_xi": derivative["predicted_dot_xi1"][part],
            }
            _, item["predicted_u"] = state_from_xi(item["predicted_x"], item["predicted_xi"], wormhole, spiral)
            dense_predictions[(architecture, value, seed, target)] = item
            x_error = item["predicted_x"] - exact["exact_x"]
            xi_error = item["predicted_xi"] - exact["exact_xi"]
            u_error = item["predicted_u"] - exact["exact_u"]
            acceleration = radial_acceleration(exact["exact_x"], exact["exact_u"], wormhole, spiral)
            ml = decompose_phase_error(exact["exact_u"], exact["exact_dot_xi"], x_error, xi_error, scales["x"], scales["xi"])
            physical = decompose_phase_error(exact["exact_u"], acceleration, x_error, u_error, scales["x"], scales["u"])
            selection = np.ones(x_error.size, dtype=bool)
            ml_metrics = metric_record(ml, x_error, selection)
            physical_metrics = metric_record(physical, x_error, selection)
            dense_rows.append({
                "target_u_th": target,
                "architecture": architecture,
                "lambda_dot_xi": value,
                "seed": seed,
                "x_xi_e_perp_rmse": ml_metrics["e_perp_rmse"],
                "x_xi_e_parallel_rmse": ml_metrics["e_parallel_rmse"],
                "x_xi_Q": ml_metrics["Q_perp_over_parallel"],
                "x_u_e_perp_rmse": physical_metrics["e_perp_rmse"],
                "x_u_e_parallel_rmse": physical_metrics["e_parallel_rmse"],
                "x_u_Q": physical_metrics["Q_perp_over_parallel"],
                "x_rmse": ml_metrics["x_rmse"],
                "xi_rmse": float(np.sqrt(np.mean(xi_error**2))),
                "dot_xi_rmse": float(np.sqrt(np.mean((item["predicted_dot_xi"] - exact["exact_dot_xi"])**2))),
            })
            dense_tag = f"u_th_{target:.2f}_{architecture}_lambda_{slug(value)}_seed_{seed}".replace(".", "p")
            for name, array in item.items():
                dense_arrays[f"{dense_tag}_{name}"] = array
    for target, exact in references.items():
        tag = f"u_th_{target:.2f}".replace(".", "p")
        for name, array in exact.items():
            dense_arrays[f"{tag}_{name}"] = array
    np.savez_compressed(ARRAYS / "dense_reference_predictions.npz", **dense_arrays)

    for row in per_run:
        dense_0p05 = next(
            item for item in dense_rows
            if item["target_u_th"] == 0.05
            and item["architecture"] == row["architecture"]
            and item["lambda_dot_xi"] == row["lambda_dot_xi"]
            and item["seed"] == row["seed"]
        )
        row["dense_0p05_e_perp_rmse"] = dense_0p05["x_xi_e_perp_rmse"]
        row["dense_0p05_e_parallel_rmse"] = dense_0p05["x_xi_e_parallel_rmse"]
        row["dense_0p05_x_rmse"] = dense_0p05["x_rmse"]

    aggregate = cell_aggregate(per_run, IMPORTANT_METRICS)
    paired = paired_architecture_rows(per_run)
    derivative_rows = split_derivative_rows(per_run)
    interactions = interaction_rows(paired)
    write_csv(TABLES / "per_run_metrics.csv", per_run)
    write_csv(TABLES / "aggregate_metrics.csv", aggregate)
    write_csv(TABLES / "paired_architecture_differences.csv", paired)
    write_csv(TABLES / "split_derivative_differences.csv", derivative_rows)
    write_csv(TABLES / "architecture_derivative_interactions.csv", interactions)
    write_csv(TABLES / "post_change_phase_space_metrics.csv", phase_rows)
    write_csv(TABLES / "post_change_physical_x_u_metrics.csv", phase_physical_rows)
    write_csv(TABLES / "aggregate_post_change_phase_space_metrics.csv", phase_aggregate)
    write_csv(TABLES / "dense_reference_metrics.csv", dense_rows)

    plot_binned_phase(phase_aggregate)
    plot_dense_references(references, dense_predictions)
    plot_safeguards(aggregate)

    pair_lookup = {(row["metric"], row["lambda_dot_xi"]): row for row in paired if row["row_type"] == "aggregate"}
    split_effect = {row["metric"]: row for row in derivative_rows if row["row_type"] == "aggregate"}
    split0_phase = pair_lookup[("low_u_post_e_perp_rmse", 0.0)]
    split0_parallel = pair_lookup[("low_u_post_e_parallel_rmse", 0.0)]
    split34_phase = pair_lookup[("low_u_post_e_perp_rmse", 0.034)]
    split34_parallel = pair_lookup[("low_u_post_e_parallel_rmse", 0.034)]
    split34_global_x = pair_lookup[("global_x_rmse", 0.034)]
    split34_low_x = pair_lookup[("low_u_x_rmse", 0.034)]
    split34_dense_perp = pair_lookup[("dense_0p05_e_perp_rmse", 0.034)]
    split34_dense_parallel = pair_lookup[("dense_0p05_e_parallel_rmse", 0.034)]
    split34_dense_x = pair_lookup[("dense_0p05_x_rmse", 0.034)]
    split34_global_xi = pair_lookup[("global_xi_rmse", 0.034)]
    split34_post_xi = pair_lookup[("post_xi_rmse", 0.034)]
    intrinsic_clear = (
        (split0_phase["relative_delta"] <= -0.05 or split0_parallel["relative_delta"] <= -0.05)
        and (split0_phase["sign_consistency"] == "all_negative_improvement" or split0_parallel["sign_consistency"] == "all_negative_improvement")
    )
    derivative_arch_clear = (
        (split34_phase["relative_delta"] <= -0.05 or split34_parallel["relative_delta"] <= -0.05)
        and (split34_phase["sign_consistency"] == "all_negative_improvement" or split34_parallel["sign_consistency"] == "all_negative_improvement")
    )
    x_preserved = split34_global_x["relative_delta"] <= 0.05 and split34_low_x["relative_delta"] <= 0.05
    preferred = derivative_arch_clear and x_preserved
    if intrinsic_clear:
        classification = "Outcome A" + (" + Outcome D" if preferred else "")
        headline = "task specialization is intrinsically useful" + (" and the split derivative model is preferred" if preferred else "")
    elif derivative_arch_clear:
        classification = "Outcome B" + (" + Outcome D" if preferred else "")
        headline = "the benefit is specific to architecture under derivative supervision" + ("; the split derivative model is preferred" if preferred else "")
    else:
        classification = "Outcome C"
        headline = "the split head does not meaningfully improve the defining low-u_th residual"
    explanation = (
        f"At lambda=0, splitting changes lowest-bin post-change e_perp by {percent(split0_phase['relative_delta'])} "
        f"with seed pattern {split0_phase['sign_consistency']}. At lambda=0.034 it changes e_perp by "
        f"{percent(split34_phase['relative_delta'])} and e_parallel by {percent(split34_parallel['relative_delta'])}; "
        f"global and lowest-bin x change by {percent(split34_global_x['relative_delta'])} and "
        f"{percent(split34_low_x['relative_delta'])}. The single dense u_th=0.05 reference is more favorable—e_perp, "
        f"e_parallel, and x change by {percent(split34_dense_perp['relative_delta'])}, "
        f"{percent(split34_dense_parallel['relative_delta'])}, and {percent(split34_dense_x['relative_delta'])}—but that "
        f"localized result does not generalize to the frozen lowest-bin validation population. Global and post-change xi "
        f"change by {percent(split34_global_xi['relative_delta'])} and {percent(split34_post_xi['relative_delta'])}. "
        f"Within the split architecture, derivative supervision changes "
        f"rapid-change dot-xi RMSE by {percent(split_effect['during_dot_xi_rmse']['relative_delta'])} and lowest-bin "
        f"e_perp by {percent(split_effect['low_u_post_e_perp_rmse']['relative_delta'])}. This classification is a "
        "validation-only architecture inference; it does not prove the absence of interference elsewhere in the shared trunk."
    )
    verdict = {
        "classification": classification,
        "headline": headline,
        "explanation": explanation,
        "preferred_architecture": "split_head_lambda_0p034" if preferred else "retain_shared_lambda_0p034",
        "lambda_0p068_recommendation": "scientifically warranted as a separate follow-up" if preferred else "not warranted by this result",
        "criteria": {
            "clear_phase_improvement_threshold": "at least 5% paired-mean reduction with all three seeds improving in e_perp or e_parallel",
            "x_preservation_threshold": "no more than 5% paired-mean degradation in global and lowest-bin global x RMSE",
            "intrinsic_clear": intrinsic_clear,
            "derivative_architecture_clear": derivative_arch_clear,
            "x_preserved": x_preserved,
        },
    }

    tests = run_tests()
    if not tests["passed"]:
        raise RuntimeError(f"focused tests failed: {tests}")
    protected_after = verify_hashes(expected)
    if protected_before != protected_after:
        raise RuntimeError("protected artifacts changed during comparison")
    summary = {
        "created_utc": utc_now(),
        "status": "SPLIT_HEAD_ARCHITECTURE_COMPARISON_COMPLETED",
        "scope": {
            "split_runs_trained": 6,
            "shared_runs_retrained": 0,
            "validation_only_analysis": True,
            "sealed_test_accessed": False,
            "lambda_0p068_trained": False,
        },
        "checkpoints": checkpoints,
        "scales": scales,
        "per_run_metrics": per_run,
        "aggregate_metrics": aggregate,
        "paired_architecture_differences": paired,
        "split_derivative_differences": derivative_rows,
        "interactions": interactions,
        "phase_space_aggregate": phase_aggregate,
        "dense_reference_metrics": dense_rows,
        "numerical_verification": {
            "maximum_shared_cache_discrepancy": maximum_shared_cache_discrepancy,
            "maximum_scaled_pythagorean_residual": maximum_pythagorean_residual,
            "maximum_physical_pythagorean_residual": maximum_physical_pythagorean_residual,
        },
        "verdict": verdict,
        "tests": tests,
        "protected_hashes_unchanged": True,
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {
        str(path.relative_to(OUTPUT)): {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)
    }
    manifest = {
        "status": summary["status"],
        "scope": summary["scope"],
        "protected_before": protected_before,
        "protected_after": protected_after,
        "source_hashes": {str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve())},
        "summary_sha256": file_sha256(SUMMARY),
        "report_sha256": file_sha256(REPORT),
        "artifacts": artifacts,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
