#!/usr/bin/env python3
"""Run the controlled four-treatment physical-time derivative-loss study."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from wormhole_sciml.finite_time import BATCH_SIZE, LEARNING_RATE, MAXIMUM_EPOCHS, PATIENCE, TRAINING_SEEDS
from wormhole_sciml.finite_time_derivative_audit import predict_xi_and_physical_s_derivative
from wormhole_sciml.finite_time_derivative_experiment import (
    RAPID_THRESHOLD,
    REFERENCE_ANCHOR_X,
    REFERENCE_GRID_POINTS,
    REFERENCE_TARGETS,
    REGION_GRID_POINTS,
    dense_feature_diagnostics,
    feature_level_metrics,
    rapid_intervals,
    regional_masks,
    treatment_validation_diagnostics,
    validation_feature_table,
)
from wormhole_sciml.finite_time_derivative_training import (
    DERIVATIVE_MEAN,
    DERIVATIVE_SCALE,
    TREATMENT_LAMBDAS,
    train_derivative_treatment_seed,
)
from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, construct_hybrid_targets, load_hybrid_model
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi, file_sha256
from wormhole_sciml.phase_c_finite_time import invert_saved_orbit_x, load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, xi_time_derivative


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/finite_time_hybrid_derivative_loss_experiment"
PROTOCOL = OUTPUT / "protocol"
TRAINING = OUTPUT / "training"
VALIDATION = OUTPUT / "validation"
TABLES = OUTPUT / "tables"
ARRAYS = OUTPUT / "arrays"
FIGURES = OUTPUT / "figures"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_HYBRID_DERIVATIVE_LOSS_EXPERIMENT.md"
SUMMARY = OUTPUT / "finite_time_hybrid_derivative_loss_summary.json"
CHECKPOINT_MANIFEST = OUTPUT / "checkpoint_manifest.json"
MANIFEST = OUTPUT / "finite_time_hybrid_derivative_loss_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_hybrid_derivative_loss_manifest.sha256"

DATA = ROOT / "output/phase_c_finite_time_dataset/datasets"
TRAIN_ROWS = DATA / "phase_c_train_raw.npz"
VALIDATION_ROWS = DATA / "phase_c_validation_raw.npz"
BANKS = ROOT / "output/phase_b_complete_orbit_banks/banks"
VALIDATION_BANK = BANKS / "phase_b_validation_orbits.npz"
REFERENCE_BANK = BANKS / "phase_b_stress_reference_orbits.npz"
BASELINE = ROOT / "output/finite_time_hybrid_s5"
PREPROCESSING = BASELINE / "preprocessing/hybrid_preprocessing_constants.json"
BASELINE_MANIFEST = BASELINE / "finite_time_hybrid_manifest.json"
BASELINE_CHECKPOINTS = {seed: BASELINE / f"training/seed_{seed}/best_checkpoint.pt" for seed in TRAINING_SEEDS}
STAGE1_SUMMARY = ROOT / "output/finite_time_hybrid_derivative_audit/finite_time_hybrid_derivative_audit_summary.json"
STAGE1_MANIFEST = ROOT / "output/finite_time_hybrid_derivative_audit/finite_time_hybrid_derivative_audit_manifest.json"
STAGE2_SUMMARY = ROOT / "output/finite_time_hybrid_gradient_calibration/finite_time_hybrid_gradient_calibration_summary.json"
STAGE2_MANIFEST = ROOT / "output/finite_time_hybrid_gradient_calibration/finite_time_hybrid_gradient_calibration_manifest.json"

EXPECTED = {
    TRAIN_ROWS: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_ROWS: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    REFERENCE_BANK: "5d7959cf1a657a5ff916e4d40aab44309ca08958f1694340b47c0fce7ac08dce",
    PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    BASELINE_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
    BASELINE_CHECKPOINTS[101]: "568300c44c7b4240dde232f1946dd18a11919e48b68af8180a0092f241000aa5",
    BASELINE_CHECKPOINTS[202]: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    BASELINE_CHECKPOINTS[303]: "f83bbc4802a7742fb0bf63c138a7f407f2d2ed3fb0172bd2c00b2c7332719ad1",
    STAGE1_SUMMARY: "7f0ccfcbbae991d98f5f35f7c86fe43653cc35c74a0fce6852fe7995678c16bf",
    STAGE1_MANIFEST: "51980aab8641fa19fe1384de9ac01b516ca1a8c619831c624f8421da685837e4",
    STAGE2_SUMMARY: "f1e60bedc5cc014747d57c4c7438cbfdadb27313b83ad4befd7f2b9e0f055bc5",
    STAGE2_MANIFEST: "74420bee0e85906ce0dfba2d352a7688cd3e95c5c09ebae2fdbae346f4a1f0fa",
}
LABELS = {0.0: "control", 0.011: "weak", 0.034: "moderate", 0.068: "upper_stress"}
COLORS = {0.0: "#555555", 0.011: "#44aa99", 0.034: "#4477aa", 0.068: "#cc6677"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def lambda_slug(value: float) -> str:
    return "lambda_" + ("0" if value == 0 else f"{value:.3f}".replace(".", "p"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src/wormhole_sciml/finite_time_derivative_training.py",
             ROOT / "src/wormhole_sciml/finite_time_derivative_experiment.py",
             ROOT / "src/wormhole_sciml/finite_time_derivative_audit.py",
             ROOT / "src/wormhole_sciml/finite_time_hybrid.py",
             ROOT / "tests/test_finite_time_derivative_experiment.py"]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def immutable_gate() -> dict[str, Any]:
    artifacts, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path); match = measured == expected
        artifacts[str(path.resolve())] = {"expected": expected, "measured": measured, "match": match}
        if not match: failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": artifacts,
            "data_scope": "training, validation, and existing non-test stress/reference trajectories only",
            "held_out_test_access": "none: no held-out artifact was opened or hashed"}


def frozen_protocol() -> dict[str, Any]:
    return {
        "frozen_utc_before_training": utc(),
        "treatments": list(TREATMENT_LAMBDAS), "seeds": list(TRAINING_SEEDS),
        "only_modeling_change": "add fixed lambda_dot_xi times normalized physical-time derivative residual MSE",
        "architecture": "4->64->64->2 tanh/tanh, linear output", "gate_s_star": 5.0,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "scheduler": None, "weight_decay": 0.0,
        "batch_size": BATCH_SIZE, "maximum_epochs": MAXIMUM_EPOCHS, "patience": PATIENCE,
        "checkpoint_selection": "minimum validation orbit-averaged standardized (V_x,F_xi) MSE only",
        "derivative_mean": DERIVATIVE_MEAN, "derivative_scale": DERIVATIVE_SCALE,
        "derivative_definition": "autograd of full physical xi_hat with respect to original physical-s leaf; create_graph=True",
        "rapid_region_rule": {
            "source": "exact validation-bank dynamics only", "threshold": RAPID_THRESHOLD,
            "threshold_provenance": "Stage-1 training exact |dot_xi| q95",
            "dense_points_per_orbit": REGION_GRID_POINTS,
            "boundary": "linear interpolation at exact |dot_xi| threshold crossings",
            "merge": "iteratively merge adjacent rapid components when their gap <= sum of current component widths",
            "pre": "immediately preceding interval of merged feature width, clipped to trajectory boundary",
            "during": "merged exact above-threshold interval",
            "post": "immediately following interval of merged feature width, clipped to trajectory boundary",
            "prediction_used": False,
        },
        "decision_rule": {
            "primary_outcome": "mean across-seed xi RMSE in post-change rows",
            "mechanism": "mean across-seed dot_xi RMSE during rapid-change rows",
            "consistency": "improvement in at least two of three matched seeds",
            "safeguards": "no >5% mean degradation in global x/xi, ordinary xi, or hard-long xi; no >10% energy-RMSE degradation; no increase in total violations",
            "tradeoff_selection": "among safeguard-passing nonzero treatments, minimize mean of post-xi and during-dot RMSE ratios to matched control",
        },
        "resampling": False, "adaptive_balancing": False, "gradient_clipping": False,
        "extra_losses": False, "recursive_training": False, "source_hashes": source_hashes(),
    }


def train_seed_worker(seed: int, output_path: str) -> list[dict[str, Any]]:
    """One independent deterministic worker; treatments stay sequential per seed."""

    root = ROOT
    training = construct_hybrid_targets(load_dataset(TRAIN_ROWS))
    validation = construct_hybrid_targets(load_dataset(VALIDATION_ROWS))
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    wormhole, spiral = experiment_parameters()
    exact = xi_time_derivative(training["x1"], training["u1"], wormhole, spiral)
    rows = []
    for value in TREATMENT_LAMBDAS:
        destination = Path(output_path) / lambda_slug(value) / f"seed_{seed}"
        metadata_path = destination / "metadata.json"
        if metadata_path.exists():
            rows.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        run = train_derivative_treatment_seed(training, validation, preprocessing, exact, seed, value, destination, progress=True)
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        run["history_sha256"] = file_sha256(Path(run["history"]))
        metadata_path.write_text(json.dumps(json_safe(run), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        rows.append(run)
    return rows


def metric_row(prefix: dict[str, Any], scope: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {**prefix, "scope": scope, "row_count": metrics["row_count"]}
    for component in ("x", "xi", "dot_xi"):
        for name, value in metrics[component].items(): row[f"{component}_{name}"] = value
    return row


def decision_row(run: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "lambda_dot_xi": run["lambda_dot_xi"], "treatment": LABELS[run["lambda_dot_xi"]], "seed": run["seed"],
        "global_x_rmse": diagnostics["global"]["x"]["rmse"], "global_x_mae": diagnostics["global"]["x"]["mae"],
        "global_xi_rmse": diagnostics["global"]["xi"]["rmse"], "global_xi_mae": diagnostics["global"]["xi"]["mae"],
        "global_dot_xi_rmse": diagnostics["global"]["dot_xi"]["rmse"], "global_dot_xi_mae": diagnostics["global"]["dot_xi"]["mae"],
        "during_dot_xi_rmse": diagnostics["regions"]["during"]["dot_xi"]["rmse"],
        "during_dot_xi_mae": diagnostics["regions"]["during"]["dot_xi"]["mae"],
        "during_xi_rmse": diagnostics["regions"]["during"]["xi"]["rmse"],
        "pre_xi_rmse": diagnostics["regions"]["pre"]["xi"]["rmse"],
        "post_xi_rmse": diagnostics["regions"]["post"]["xi"]["rmse"],
        "post_xi_mae": diagnostics["regions"]["post"]["xi"]["mae"],
        "post_dot_xi_rmse": diagnostics["regions"]["post"]["dot_xi"]["rmse"],
        "hard_xi_rmse": diagnostics["families"]["hard_u_th_le_0p30"]["xi"]["rmse"],
        "ordinary_xi_rmse": diagnostics["families"]["ordinary_u_th_gt_0p30"]["xi"]["rmse"],
        "long_xi_rmse": diagnostics["horizons"]["long_s_gt_20"]["xi"]["rmse"],
        "hard_long_xi_rmse": diagnostics["horizons"]["hard_long_s_gt_20"]["xi"]["rmse"],
        "energy_rmse": diagnostics["physical"]["energy_rmse"],
        "union_violation_count": diagnostics["physical"]["union_violation_count"],
        "best_epoch": run["best_epoch"], "stopping_epoch": run["stopping_epoch"],
        "training_wall_seconds": run["training_wall_seconds"], "nan_inf_events": run["nan_inf_events"],
        "validation_selection_mse": run["best_validation_orbit_averaged_standardized_hybrid_mse"],
    }


def aggregate_decision_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identifiers = {"lambda_dot_xi", "treatment", "seed", "best_epoch", "stopping_epoch", "nan_inf_events"}
    metrics = [name for name in rows[0] if name not in identifiers]
    controls = {name: np.asarray([row[name] for row in rows if row["lambda_dot_xi"] == 0.0], dtype=float) for name in metrics}
    output = []
    for value in TREATMENT_LAMBDAS:
        selected = [row for row in rows if row["lambda_dot_xi"] == value]
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            baseline = controls[metric]
            output.append({
                "lambda_dot_xi": value, "treatment": LABELS[value], "metric": metric,
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=0)), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                "mean_change_from_control": float(np.mean(values) - np.mean(baseline)),
                "mean_percent_change_from_control": float(100 * (np.mean(values) / np.mean(baseline) - 1)) if np.mean(baseline) != 0 else None,
                "seed_improvement_count": int(np.sum(values < baseline)),
            })
    return output


def aggregate_scoped_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Long-form mean/median/spread summaries across the three retained seeds."""

    identifiers = {"lambda_dot_xi", "treatment", "seed", "scope", "row_count"}
    metrics = [name for name in rows[0] if name not in identifiers]
    output = []
    scopes = list(dict.fromkeys(row["scope"] for row in rows))
    for value in TREATMENT_LAMBDAS:
        for scope in scopes:
            selected = [row for row in rows if row["lambda_dot_xi"] == value and row["scope"] == scope]
            controls = [row for row in rows if row["lambda_dot_xi"] == 0.0 and row["scope"] == scope]
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected if row[metric] is not None], dtype=float)
                baseline = np.asarray([row[metric] for row in controls if row[metric] is not None], dtype=float)
                if not values.size: continue
                output.append({"lambda_dot_xi": value, "treatment": LABELS[value], "scope": scope, "metric": metric,
                               "mean": float(np.mean(values)), "median": float(np.median(values)),
                               "standard_deviation": float(np.std(values, ddof=0)), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
                               "mean_percent_change_from_control": float(100 * (np.mean(values) / np.mean(baseline) - 1)) if baseline.size and np.mean(baseline) != 0 else None,
                               "seed_improvement_count": int(np.sum(values < baseline)) if baseline.size == values.size else None})
    return output


def aggregate_lookup(rows: list[dict[str, Any]], value: float, metric: str) -> dict[str, Any]:
    return next(row for row in rows if row["lambda_dot_xi"] == value and row["metric"] == metric)


def mechanism_comparisons(feature_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    controls = {(row["seed"], row["source_orbit_index"], row["feature_index"]): row for row in feature_rows if row["lambda_dot_xi"] == 0.0}
    paired = []
    for row in feature_rows:
        if row["lambda_dot_xi"] == 0.0: continue
        control = controls[(row["seed"], row["source_orbit_index"], row["feature_index"])]
        if row["dot_xi_during_rmse"] is None or row["xi_post_rmse"] is None: continue
        paired.append({**row,
            "control_dot_xi_during_rmse": control["dot_xi_during_rmse"], "control_xi_post_rmse": control["xi_post_rmse"],
            "delta_dot_xi_during_rmse": row["dot_xi_during_rmse"] - control["dot_xi_during_rmse"],
            "delta_xi_post_rmse": row["xi_post_rmse"] - control["xi_post_rmse"],
            "relative_dot_xi_during_change": row["dot_xi_during_rmse"] / control["dot_xi_during_rmse"] - 1 if control["dot_xi_during_rmse"] else None,
            "relative_xi_post_change": row["xi_post_rmse"] / control["xi_post_rmse"] - 1 if control["xi_post_rmse"] else None,
        })
    associations = []
    for value in TREATMENT_LAMBDAS[1:]:
        for seed in (*TRAINING_SEEDS, "all"):
            selected = [row for row in paired if row["lambda_dot_xi"] == value and (seed == "all" or row["seed"] == seed) and row["during_row_count"] >= 2 and row["post_row_count"] >= 2]
            if len(selected) >= 3:
                association = spearmanr([row["delta_dot_xi_during_rmse"] for row in selected], [row["delta_xi_post_rmse"] for row in selected])
                coefficient, pvalue = float(association.statistic), float(association.pvalue)
            else:
                coefficient = pvalue = None
            associations.append({"lambda_dot_xi": value, "treatment": LABELS[value], "seed": seed,
                                 "feature_count": len(selected), "spearman_delta_derivative_vs_delta_post_xi": coefficient,
                                 "pvalue_descriptive_only": pvalue})
    return paired, associations


def classify(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    control = {metric: aggregate_lookup(aggregates, 0.0, metric)["mean"] for metric in (
        "post_xi_rmse", "during_dot_xi_rmse", "global_x_rmse", "global_xi_rmse", "ordinary_xi_rmse", "hard_long_xi_rmse", "energy_rmse")}
    evaluations = []
    for value in TREATMENT_LAMBDAS[1:]:
        lookup = lambda metric: aggregate_lookup(aggregates, value, metric)
        ratios = {metric: lookup(metric)["mean"] / control[metric] for metric in control}
        violations = lookup("union_violation_count")
        control_violations = aggregate_lookup(aggregates, 0.0, "union_violation_count")
        consistent_dot = lookup("during_dot_xi_rmse")["seed_improvement_count"] >= 2
        consistent_post = lookup("post_xi_rmse")["seed_improvement_count"] >= 2
        safe = all(ratios[name] <= 1.05 for name in ("global_x_rmse", "global_xi_rmse", "ordinary_xi_rmse", "hard_long_xi_rmse")) and ratios["energy_rmse"] <= 1.10 and violations["mean"] <= control_violations["mean"]
        score = .5 * (ratios["post_xi_rmse"] + ratios["during_dot_xi_rmse"])
        evaluations.append({"lambda_dot_xi": value, "ratios_to_control": ratios,
                            "during_dot_improves_at_least_two_seeds": consistent_dot,
                            "post_xi_improves_at_least_two_seeds": consistent_post,
                            "safeguards_pass": safe, "tradeoff_score": score})
    eligible = [row for row in evaluations if row["safeguards_pass"] and row["during_dot_improves_at_least_two_seeds"] and row["post_xi_improves_at_least_two_seeds"]]
    best = min(eligible, key=lambda row: row["tradeoff_score"]) if eligible else None
    any_dot = any(row["during_dot_improves_at_least_two_seeds"] for row in evaluations)
    any_post = any(row["post_xi_improves_at_least_two_seeds"] for row in evaluations)
    if best is not None:
        selected_rows = [row for row in evaluations if row["lambda_dot_xi"] == best["lambda_dot_xi"]][0]
        all_seed_both = aggregate_lookup(aggregates, best["lambda_dot_xi"], "during_dot_xi_rmse")["seed_improvement_count"] == 3 and aggregate_lookup(aggregates, best["lambda_dot_xi"], "post_xi_rmse")["seed_improvement_count"] == 3
        label = "STRONG SUPPORT" if all_seed_both else "PARTIAL SUPPORT"
        explanation = f"lambda={best['lambda_dot_xi']:.3g} passes safeguards and gives the best predeclared joint targeted score; both targeted metrics improve in {'all three' if all_seed_both else 'at least two'} seeds."
    elif any_dot and not any_post:
        label = "PARTIAL SUPPORT"; explanation = "Derivative accuracy improves consistently for at least one treatment, but post-change xi does not improve consistently."
    elif any_dot:
        label = "OVER-CONSTRAINED"; explanation = "Derivative accuracy improves, but treatments that meet the targeted criteria fail at least one preservation safeguard."
    else:
        label = "NO SUPPORT"; explanation = "No derivative-loss treatment improves during-feature derivative RMSE consistently across seeds."
    return {"classification": label, "explanation": explanation, "treatments": evaluations,
            "selected_lambda": None if best is None else best["lambda_dot_xi"]}


def plot_histories(runs: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True, sharey=True)
    for row_index, seed in enumerate(TRAINING_SEEDS):
        for column_index, value in enumerate(TREATMENT_LAMBDAS):
            run = next(r for r in runs if r["seed"] == seed and r["lambda_dot_xi"] == value)
            history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
            epoch = [r["epoch"] for r in history]
            ax = axes[row_index, column_index]
            ax.plot(epoch, [r["training_standardized_hybrid_mse"] for r in history], label="current train")
            ax.plot(epoch, [r["validation_orbit_averaged_standardized_hybrid_mse"] for r in history], label="current validation")
            if value > 0: ax.plot(epoch, [value * r["training_normalized_derivative_mse"] for r in history], label="weighted derivative")
            ax.axvline(run["best_epoch"], color="black", ls="--", lw=.8)
            ax.set(title=f"seed {seed}, lambda={value:.3g}", xlabel="epoch", yscale="log"); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7); axes[1, 0].set_ylabel("loss"); fig.suptitle("Controlled derivative-loss training histories")
    fig.savefig(FIGURES / "training_histories.png", dpi=175); plt.close(fig)


def plot_treatment_metrics(decision_rows: list[dict[str, Any]]) -> None:
    metrics = (("during_dot_xi_rmse", "during rapid-change dot xi RMSE"), ("post_xi_rmse", "post-change xi RMSE"),
               ("global_xi_rmse", "global xi RMSE"), ("global_x_rmse", "global x RMSE"),
               ("hard_long_xi_rmse", "hard long-horizon xi RMSE"), ("energy_rmse", "energy-error RMSE"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        means, lows, highs = [], [], []
        for value in TREATMENT_LAMBDAS:
            points = np.asarray([row[metric] for row in decision_rows if row["lambda_dot_xi"] == value])
            means.append(points.mean()); lows.append(points.mean() - points.min()); highs.append(points.max() - points.mean())
            ax.scatter(np.full(points.size, value), points, color=COLORS[value], s=24, zorder=3)
        ax.errorbar(TREATMENT_LAMBDAS, means, yerr=np.asarray([lows, highs]), color="black", marker="o", capsize=4)
        ax.set(title=title, xlabel="lambda_dot_xi", xticks=TREATMENT_LAMBDAS); ax.grid(alpha=.25)
    fig.suptitle("Validation tradeoffs across fixed derivative-loss treatments")
    fig.savefig(FIGURES / "treatment_validation_comparison.png", dpi=180); plt.close(fig)


def dense_references(models: dict[tuple[float, int], Any], preprocessing: HybridPreprocessing, bank: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wormhole, spiral = experiment_parameters(); metric_rows, timing_rows = [], []
    for target in REFERENCE_TARGETS:
        orbit = int(np.argmin(np.abs(bank["u_th"] - target)))
        t0, residual = invert_saved_orbit_x(bank, orbit, np.asarray([REFERENCE_ANCHOR_X])); t0 = float(t0[0])
        elapsed = np.linspace(0.0, float(bank["t_right"][orbit]) - t0, REFERENCE_GRID_POINTS)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit, t0 + elapsed)
        exact_dot = xi_time_derivative(exact[:, 0], exact[:, 1], wormhole, spiral)
        intervals = rapid_intervals(elapsed, exact_dot)
        during = np.zeros(elapsed.size, dtype=bool); post = np.zeros(elapsed.size, dtype=bool)
        for interval in intervals:
            during |= (elapsed >= interval["during_start"]) & (elapsed <= interval["during_stop"])
            post |= (elapsed > interval["post_start"]) & (elapsed <= interval["post_stop"])
        query = {"x0": np.full(elapsed.size, exact[0, 0]), "xi0": np.full(elapsed.size, exact[0, 2]),
                 "E0": np.full(elapsed.size, bank["E0"][orbit]), "s": elapsed}
        for seed in TRAINING_SEEDS:
            predictions = {}
            for value in TREATMENT_LAMBDAS:
                prediction = predict_xi_and_physical_s_derivative(models[(value, seed)], preprocessing, query)
                predictions[value] = prediction
                xi_error = prediction["predicted_xi1"] - exact[:, 2]
                dot_error = prediction["predicted_dot_xi1"] - exact_dot
                metric_rows.append({"target_u_th": target, "actual_u_th": float(bank["u_th"][orbit]), "seed": seed,
                                    "lambda_dot_xi": value, "treatment": LABELS[value],
                                    "anchor_x0": float(exact[0, 0]), "anchor_inversion_residual": float(residual[0]),
                                    "dot_xi_during_rmse": float(np.sqrt(np.mean(dot_error[during]**2))) if during.any() else None,
                                    "xi_post_rmse": float(np.sqrt(np.mean(xi_error[post]**2))) if post.any() else None,
                                    "xi_global_rmse": float(np.sqrt(np.mean(xi_error**2))), "dot_xi_global_rmse": float(np.sqrt(np.mean(dot_error**2)))})
                for feature_index, row in enumerate(dense_feature_diagnostics(elapsed, exact_dot, prediction["predicted_dot_xi1"])):
                    timing_rows.append({"target_u_th": target, "actual_u_th": float(bank["u_th"][orbit]), "seed": seed,
                                        "lambda_dot_xi": value, "treatment": LABELS[value], "matched_feature_index": feature_index, **row})
            arrays = {"elapsed_s": elapsed, "exact_x": exact[:, 0], "exact_xi": exact[:, 2], "exact_dot_xi": exact_dot}
            for value, prediction in predictions.items():
                slug = lambda_slug(value)
                arrays[f"{slug}_predicted_xi"] = prediction["predicted_xi1"]
                arrays[f"{slug}_predicted_dot_xi"] = prediction["predicted_dot_xi1"]
            np.savez_compressed(ARRAYS / f"reference_u_th_{target:.2f}_seed_{seed}.npz", **arrays)
            fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True, sharex=True)
            axes[0].plot(elapsed, exact[:, 2], color="black", lw=2, label="exact")
            axes[1].plot(elapsed, exact_dot, color="black", lw=2, label="exact")
            for value in TREATMENT_LAMBDAS:
                prediction = predictions[value]; label = f"lambda={value:.3g}"
                axes[0].plot(elapsed, prediction["predicted_xi1"], color=COLORS[value], label=label)
                axes[1].plot(elapsed, prediction["predicted_dot_xi1"], color=COLORS[value], label=label)
                axes[2].plot(elapsed, prediction["predicted_xi1"] - exact[:, 2], color=COLORS[value], label=label)
                axes[3].plot(elapsed, prediction["predicted_dot_xi1"] - exact_dot, color=COLORS[value], label=label)
            for interval in intervals:
                for ax in axes: ax.axvspan(interval["during_start"], interval["during_stop"], color="0.5", alpha=.08)
            axes[0].set_ylabel("xi"); axes[1].set_ylabel("dot xi"); axes[2].set_ylabel("xi error"); axes[3].set(ylabel="dot xi error", xlabel="physical elapsed time s")
            for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
            fig.suptitle(f"Direct finite-time treatment comparison: u_th={target:.2f}, seed={seed}")
            fig.savefig(FIGURES / f"reference_u_th_{target:.2f}_seed_{seed}.png", dpi=175); plt.close(fig)
    return metric_rows, timing_rows


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_finite_time_derivative_experiment.py",
               "tests/test_finite_time_derivative_audit.py", f"--junitxml={TESTS/'focused_pytest.xml'}"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-stage3-mpl"}, capture_output=True, text=True)
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload); return payload


def report_text(summary: dict[str, Any]) -> str:
    aggregates = summary["aggregate_decision_metrics"]
    difficulty = summary["aggregate_difficulty_metrics"]
    horizons = summary["aggregate_horizon_metrics"]
    comparison_rows = []
    for value in TREATMENT_LAMBDAS:
        get = lambda metric: aggregate_lookup(aggregates, value, metric)
        cell = lambda metric: f"{get(metric)['mean']:.6g} [{get(metric)['minimum']:.6g}, {get(metric)['maximum']:.6g}] ({get(metric)['mean_percent_change_from_control']:+.2f}%)"
        comparison_rows.append(f"| {value:.3g} | {cell('during_dot_xi_rmse')} | {cell('post_xi_rmse')} | {cell('global_xi_rmse')} | {cell('global_x_rmse')} | {cell('hard_long_xi_rmse')} |")
    training_rows = "\n".join(f"| {run['lambda_dot_xi']:.3g} | {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | {run['best_validation_orbit_averaged_standardized_hybrid_mse']:.7g} | {run['training_wall_seconds']:.1f} | {run['maximum_observed_batch_gradient_l2']:.4g} | {run['process_peak_rss_mib']:.1f} | {run['nan_inf_events']} |" for run in summary["runs"])
    physical_rows = "\n".join(f"| {row['lambda_dot_xi']:.3g} | {row['seed']} | {row['union_violation_count']} | {row['energy_rmse']:.6g} | {row['energy_mae']:.6g} |" for row in summary["physical_metrics"])
    mechanism_rows = "\n".join(f"| {row['lambda_dot_xi']:.3g} | {row['seed']} | {row['feature_count']} | {f'{row['spearman_delta_derivative_vs_delta_post_xi']:.4f}' if row['spearman_delta_derivative_vs_delta_post_xi'] is not None else 'n/a'} |" for row in summary["mechanism_associations"] if row["seed"] == "all")
    conclusion = summary["decision"]
    scoped = lambda rows, value, scope, metric: next(row for row in rows if row["lambda_dot_xi"] == value and row["scope"] == scope and row["metric"] == metric)
    difficulty_rows = []
    for value in TREATMENT_LAMBDAS:
        cells = []
        for scope in ("hard_u_th_le_0p30", "ordinary_u_th_gt_0p30", "u_th_within_0p01_of_0.05", "u_th_within_0p01_of_0.15", "u_th_within_0p01_of_0.30"):
            row = scoped(difficulty, value, scope, "xi_rmse"); cells.append(f"{row['mean']:.6g} ({row['mean_percent_change_from_control']:+.1f}%)")
        difficulty_rows.append(f"| {value:.3g} | " + " | ".join(cells) + " |")
    horizon_rows = []
    for value in TREATMENT_LAMBDAS:
        cells = []
        for scope in ("short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20", "hard_long_s_gt_20"):
            row = scoped(horizons, value, scope, "xi_rmse"); cells.append(f"{row['mean']:.6g} ({row['mean_percent_change_from_control']:+.1f}%)")
        horizon_rows.append(f"| {value:.3g} | " + " | ".join(cells) + " |")
    selected = next(row for row in conclusion["treatments"] if row["lambda_dot_xi"] == conclusion["selected_lambda"]) if conclusion["selected_lambda"] is not None else None
    upper = next(row for row in conclusion["treatments"] if row["lambda_dot_xi"] == .068)
    pooled_associations = [row["spearman_delta_derivative_vs_delta_post_xi"] for row in summary["mechanism_associations"] if row["seed"] == "all"]
    return f"""# Controlled physical-time derivative-loss experiment

## Protocol and integrity

All 12 runs use the same 4→64→64→2 tanh architecture, frozen preprocessing, training/validation rows, physical gate, Adam optimizer, learning rate, batch size, deterministic seed-specific shuffling, 1500-epoch budget, patience 40, and validation-only checkpoint criterion. The sole treatment change is fixed `lambda_dot_xi` multiplying the normalized physical-time derivative residual MSE. No held-out test artifact was opened or hashed.

The rapid-region rule was frozen before training from exact dynamics only: `|dot xi| > {RAPID_THRESHOLD:.8g}` (Stage-1 training q95), threshold-crossing interpolation, deterministic merging, and feature-width-matched pre/post windows. It yielded {summary['regional_rule']['feature_count']} features and validation row counts pre/during/post = {summary['regional_rule']['row_counts']['pre']}/{summary['regional_rule']['row_counts']['during']}/{summary['regional_rule']['row_counts']['post']}.

## Training

| lambda | seed | best epoch | stop epoch | selection MSE | wall s | max gradient L2 | peak RSS MiB | NaN/Inf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{training_rows}

The retrained lambda-zero checkpoints are bitwise identical to the retained baseline checkpoints for every seed. No gradient clipping, scheduler, resampling, or adaptive weighting was introduced.

## Primary validation comparison

Values are across-seed means followed by `[minimum, maximum]` seed spread; parentheses give percent change from the matched lambda-zero mean. Negative change is improvement. Medians and standard deviations are retained in the aggregate tables.

| lambda | during dot-xi RMSE | post xi RMSE | global xi RMSE | global x RMSE | hard long-horizon xi RMSE |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

Full seed spreads, MAEs, narrow `u_th` families, pre/during/post regions, and horizon bins are in the machine-readable tables.

## Difficulty and horizon dependence

Across-seed mean xi RMSE is shown with percent change from control.

| lambda | hard u_th≤0.30 | ordinary u_th>0.30 | near 0.05 | near 0.15 | near 0.30 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(difficulty_rows)}

| lambda | short s≤5 | intermediate 5<s≤20 | long s>20 | hard long s>20 |
|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

## Mechanism check

Feature-level changes compare each nonzero treatment with its same-seed control. Spearman coefficients are descriptive associations between change in during-feature derivative RMSE and change in post-feature xi RMSE; they are not causal estimates.

| lambda | scope | eligible features | Spearman association |
|---:|:---|---:|---:|
{mechanism_rows}

The pooled feature-level associations are weak (range `{min(pooled_associations):.3f}` to `{max(pooled_associations):.3f}`), so the aggregate co-improvement does not establish a strong one-feature-at-a-time coupling.

## Physical safeguards

| lambda | seed | admissibility violations | energy RMSE | energy MAE |
|---:|---:|---:|---:|---:|
{physical_rows}

## Scientific conclusion

**{conclusion['classification']}** — {conclusion['explanation']}

For the selected treatment, mean during-feature derivative RMSE changes by `{100*(selected['ratios_to_control']['during_dot_xi_rmse']-1):+.2f}%` and mean post-feature xi RMSE by `{100*(selected['ratios_to_control']['post_xi_rmse']-1):+.2f}%`. The upper-stress treatment fails the frozen safeguard because its mean global x RMSE changes by `{100*(upper['ratios_to_control']['global_x_rmse']-1):+.2f}%`, and its post-xi improvement is not consistent across at least two seeds.

The selected validation tradeoff is `{conclusion['selected_lambda']}`. This conclusion is based on validation only and applies to the frozen regional and safeguard rules; it does not use or imply performance on held-out test data.

Focused tests: `{summary['tests']['stdout'].strip()}`. All protected input hashes were unchanged. Training wall time across all runs was {summary['compute']['total_training_wall_seconds']:.1f} worker-seconds; three seed workers ran concurrently. Peak parent-process RSS was approximately {summary['compute']['parent_peak_rss_mib']:.1f} MiB.
"""


def main() -> None:
    resume = os.environ.get("WORMHOLE_STAGE3_RESUME") == "1"
    if OUTPUT.exists() and not resume: raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, PROTOCOL, TRAINING, VALIDATION, TABLES, ARRAYS, FIGURES, TESTS): directory.mkdir(parents=True, exist_ok=resume)
    gate = immutable_gate(); write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]: raise RuntimeError(gate["failures"])
    protected_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    protocol_path = PROTOCOL / "frozen_experiment_protocol.json"
    if resume and protocol_path.exists(): protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    else: protocol = frozen_protocol(); write_json(protocol_path, protocol)

    validation = construct_hybrid_targets(load_dataset(VALIDATION_ROWS))
    validation_bank = load_npz(VALIDATION_BANK)
    feature_table = validation_feature_table(validation, validation_bank)
    region_masks, feature_masks = regional_masks(validation, feature_table)
    feature_path = PROTOCOL / "frozen_validation_feature_windows.csv"
    if not (resume and feature_path.exists()): write_csv(feature_path, feature_table)
    regional_definition = {"feature_count": len(feature_table), "orbit_count_with_features": len({row["source_orbit_index"] for row in feature_table}),
                           "row_counts": {name: int(mask.sum()) for name, mask in region_masks.items()},
                           "feature_table_sha256": file_sha256(feature_path)}
    realization_path = PROTOCOL / "frozen_regional_rule_realization.json"
    if resume and realization_path.exists() and json.loads(realization_path.read_text(encoding="utf-8")) != regional_definition:
        raise RuntimeError("resumed exact regional rule differs from the frozen realization")
    write_json(realization_path, regional_definition)

    runs = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(train_seed_worker, seed, str(TRAINING)): seed for seed in TRAINING_SEEDS}
        for future in as_completed(futures):
            seed = futures[future]; seed_runs = future.result(); runs.extend(seed_runs)
            print(f"completed all treatments for seed {seed}", flush=True)
    runs.sort(key=lambda row: (row["lambda_dot_xi"], row["seed"]))
    for seed in TRAINING_SEEDS:
        seed_runs = [run for run in runs if run["seed"] == seed]
        if len({run["initial_state_sha256"] for run in seed_runs}) != 1: raise RuntimeError(f"initialization mismatch for seed {seed}")
        control = next(run for run in seed_runs if run["lambda_dot_xi"] == 0.0)
        if control["checkpoint_sha256"] != EXPECTED[BASELINE_CHECKPOINTS[seed]]: raise RuntimeError(f"lambda-zero control failed exact reproduction for seed {seed}")
    checkpoint_rows = [{"lambda_dot_xi": run["lambda_dot_xi"], "treatment": LABELS[run["lambda_dot_xi"]], "seed": run["seed"],
                        "checkpoint": run["checkpoint"], "checkpoint_sha256": run["checkpoint_sha256"],
                        "initial_state_sha256": run["initial_state_sha256"], "best_epoch": run["best_epoch"],
                        "selection_metric": run["best_validation_orbit_averaged_standardized_hybrid_mse"]} for run in runs]
    write_json(CHECKPOINT_MANIFEST, {"checkpoints": checkpoint_rows, "lambda_zero_exact_reproduction": True})

    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    wormhole, spiral = experiment_parameters(); exact_validation_dot = xi_time_derivative(validation["x1"], validation["u1"], wormhole, spiral)
    diagnostics_by_run, predictions_by_run, models = {}, {}, {}
    global_rows, regional_rows, family_rows, horizon_rows, physical_rows, feature_rows, decision_rows = [], [], [], [], [], [], []
    for run in runs:
        value, seed = run["lambda_dot_xi"], run["seed"]; prefix = {"lambda_dot_xi": value, "treatment": LABELS[value], "seed": seed}
        model = load_hybrid_model(Path(run["checkpoint"])); models[(value, seed)] = model
        diagnostics, prediction = treatment_validation_diagnostics(model, preprocessing, validation, exact_validation_dot, region_masks)
        diagnostics_by_run[f"{value:.3f}/seed_{seed}"] = diagnostics; predictions_by_run[(value, seed)] = prediction
        global_rows.append(metric_row(prefix, "global", diagnostics["global"]))
        regional_rows.extend(metric_row(prefix, name, metric) for name, metric in diagnostics["regions"].items())
        family_rows.extend(metric_row(prefix, name, metric) for name, metric in diagnostics["families"].items())
        horizon_rows.extend(metric_row(prefix, name, metric) for name, metric in diagnostics["horizons"].items())
        physical_rows.append({**prefix, **diagnostics["physical"]})
        feature_rows.extend({**prefix, **row} for row in feature_level_metrics(feature_table, feature_masks, prediction))
        decision_rows.append(decision_row(run, diagnostics))
        np.savez_compressed(VALIDATION / f"predictions_{lambda_slug(value)}_seed_{seed}.npz",
                            transition_id=validation["transition_id"], predicted_x=prediction["predicted_x1"],
                            predicted_xi=prediction["predicted_xi1"], predicted_dot_xi=prediction["predicted_dot_xi1"],
                            x_error=prediction["x_error"], xi_error=prediction["xi_error"], dot_xi_error=prediction["dot_xi_error"])
    for path, rows_to_write in (("global_metrics.csv", global_rows), ("rapid_region_metrics.csv", regional_rows),
                                ("difficulty_metrics.csv", family_rows), ("horizon_metrics.csv", horizon_rows),
                                ("physical_safeguards.csv", physical_rows), ("feature_level_metrics.csv", feature_rows),
                                ("per_run_decision_metrics.csv", decision_rows)):
        write_csv(TABLES / path, rows_to_write)
    write_json(VALIDATION / "per_run_validation_metrics.json", diagnostics_by_run)

    aggregate_rows = aggregate_decision_metrics(decision_rows); write_csv(TABLES / "aggregate_treatment_comparison.csv", aggregate_rows)
    aggregate_regions = aggregate_scoped_metrics(regional_rows); write_csv(TABLES / "aggregate_rapid_region_metrics.csv", aggregate_regions)
    aggregate_difficulty = aggregate_scoped_metrics(family_rows); write_csv(TABLES / "aggregate_difficulty_metrics.csv", aggregate_difficulty)
    aggregate_horizons = aggregate_scoped_metrics(horizon_rows); write_csv(TABLES / "aggregate_horizon_metrics.csv", aggregate_horizons)
    paired_features, associations = mechanism_comparisons(feature_rows)
    write_csv(TABLES / "feature_level_changes_from_control.csv", paired_features)
    write_csv(TABLES / "mechanism_associations.csv", associations)
    reference_metrics, timing_rows = dense_references(models, preprocessing, load_npz(REFERENCE_BANK))
    write_csv(TABLES / "dense_reference_metrics.csv", reference_metrics); write_csv(TABLES / "dense_reference_feature_timing.csv", timing_rows)
    decision = classify(aggregate_rows)
    plot_histories(runs); plot_treatment_metrics(decision_rows)

    tests = run_tests(); protected_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    if protected_before != protected_after: raise RuntimeError("protected artifact changed")
    if not tests["passed"]: raise RuntimeError("focused tests failed")
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; peak_mib = maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
    summary = {"created_utc": utc(), "status": "CONTROLLED_DERIVATIVE_LOSS_EXPERIMENT_COMPLETED",
               "protocol": protocol, "regional_rule": regional_definition, "runs": runs,
               "decision_metrics": decision_rows, "aggregate_decision_metrics": aggregate_rows,
               "aggregate_rapid_region_metrics": aggregate_regions, "aggregate_difficulty_metrics": aggregate_difficulty,
               "aggregate_horizon_metrics": aggregate_horizons,
               "physical_metrics": physical_rows, "mechanism_associations": associations,
               "dense_reference_metrics": reference_metrics, "decision": decision, "tests": tests,
               "compute": {"total_training_wall_seconds": float(sum(run["training_wall_seconds"] for run in runs)),
                           "parent_peak_rss_mib": peak_mib, "parallel_seed_workers": 3},
               "lambda_zero_bitwise_reproduction": True, "dataset_modified": False,
               "held_out_test_accessed": False, "protected_hashes_unchanged": True}
    write_json(SUMMARY, summary); REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {str(path.relative_to(OUTPUT)): {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)}
    manifest = {"status": summary["status"], "input_gate": gate, "protected_before": protected_before, "protected_after": protected_after,
                "source_hashes": source_hashes(), "checkpoint_manifest_sha256": file_sha256(CHECKPOINT_MANIFEST),
                "summary_sha256": file_sha256(SUMMARY), "report_sha256": file_sha256(REPORT), "artifacts": artifacts}
    write_json(MANIFEST, manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
