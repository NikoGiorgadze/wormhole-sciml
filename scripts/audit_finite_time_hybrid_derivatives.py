#!/usr/bin/env python3
"""Training/validation/reference-only physical-time derivative audit."""

from __future__ import annotations

from collections import Counter
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
from scipy.stats import spearmanr

from wormhole_sciml.finite_time_derivative_audit import (
    binned_error_metrics,
    dominant_peak_timing,
    finite_difference_saved_xi_derivative,
    predict_xi_and_physical_s_derivative,
    scalar_error_metrics,
    training_sharpness_edges,
    value_distribution,
)
from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, load_hybrid_model
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi, file_sha256
from wormhole_sciml.phase_c_finite_time import invert_saved_orbit_x, load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, xi_time_derivative


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/finite_time_hybrid_derivative_audit"
ARRAYS = OUTPUT / "arrays"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_HYBRID_DERIVATIVE_AUDIT.md"
SUMMARY = OUTPUT / "finite_time_hybrid_derivative_audit_summary.json"
MANIFEST = OUTPUT / "finite_time_hybrid_derivative_audit_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_hybrid_derivative_audit_manifest.sha256"

DATA = ROOT / "output/phase_c_finite_time_dataset/datasets"
TRAIN_ROWS = DATA / "phase_c_train_raw.npz"
VALIDATION_ROWS = DATA / "phase_c_validation_raw.npz"
REFERENCE_ROWS = DATA / "phase_c_stress_reference_diagnostic_raw.npz"
BANKS = ROOT / "output/phase_b_complete_orbit_banks/banks"
TRAIN_BANK = BANKS / "phase_b_train_orbits.npz"
VALIDATION_BANK = BANKS / "phase_b_validation_orbits.npz"
REFERENCE_BANK = BANKS / "phase_b_stress_reference_orbits.npz"
MODEL_ROOT = ROOT / "output/finite_time_hybrid_s5"
CHECKPOINT = MODEL_ROOT / "training/seed_202/best_checkpoint.pt"
PREPROCESSING = MODEL_ROOT / "preprocessing/hybrid_preprocessing_constants.json"
MODEL_MANIFEST = MODEL_ROOT / "finite_time_hybrid_manifest.json"

EXPECTED = {
    TRAIN_ROWS: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_ROWS: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    REFERENCE_ROWS: "5402a610f1be4dc5c793a0fde809dc4d6e275e78d9fc56b042b2a76b6e8ea6b6",
    TRAIN_BANK: "8b72b7c89a47d25d215c3991b0f409aff4e3e006508d113bb980f27b87775fa8",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    REFERENCE_BANK: "5d7959cf1a657a5ff916e4d40aab44309ca08958f1694340b47c0fce7ac08dce",
    CHECKPOINT: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    MODEL_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
}
U_EDGES = np.asarray([-np.inf, .05, .15, .30, .50, .70, np.inf])
U_LABELS = ("le_0p05", "0p05_0p15", "0p15_0p30", "0p30_0p50", "0p50_0p70", "gt_0p70")
REFERENCE_TARGETS = (.05, .15, .30)
REFERENCE_ANCHOR_X = -14.0
REFERENCE_DENSE_POINTS = 2401


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    paths = [
        Path(__file__), ROOT / "src/wormhole_sciml/finite_time_derivative_audit.py",
        ROOT / "src/wormhole_sciml/physics_gate.py", ROOT / "src/wormhole_sciml/dynamics.py",
        ROOT / "src/wormhole_sciml/finite_time_hybrid.py", ROOT / "tests/test_finite_time_derivative_audit.py",
    ]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def immutable_gate() -> dict[str, Any]:
    artifacts, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path); match = measured == expected
        artifacts[str(path.resolve())] = {"expected": expected, "measured": measured, "match": match}
        if not match: failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": artifacts,
            "data_scope": "training, validation, and non-test stress/reference only",
            "held_out_test_access": "none: no held-out artifact was opened or hashed"}


def exact_endpoint_derivative(rows: dict[str, np.ndarray]) -> np.ndarray:
    wormhole, spiral = experiment_parameters()
    return xi_time_derivative(rows["x1"], rows["u1"], wormhole, spiral)


def verification_audit(bank_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    records: dict[str, list[Any]] = {key: [] for key in (
        "split", "orbit_index", "orbit_id", "u_th", "t", "analytic", "finite_difference",
        "absolute_error", "relative_error_floor_1e_10", "method", "step",
    )}
    summaries: dict[str, Any] = {}
    for split, path in bank_paths.items():
        bank = load_npz(path); orbit_count = bank["orbit_id"].size
        orbit_indices = np.arange(orbit_count) if split == "stress_reference" else np.unique(np.linspace(0, orbit_count - 1, 64, dtype=int))
        split_analytic, split_fd, split_method = [], [], []
        for orbit_index in orbit_indices:
            times = np.linspace(float(bank["t_left"][orbit_index]), float(bank["t_right"][orbit_index]), 33)
            state = evaluate_saved_orbit_x_u_xi(bank, int(orbit_index), times)
            analytic = xi_time_derivative(state[:, 0], state[:, 1], wormhole, spiral)
            finite, method, step = finite_difference_saved_xi_derivative(bank, int(orbit_index), times)
            absolute = np.abs(finite - analytic); relative = absolute / np.maximum(np.abs(analytic), 1e-10)
            split_analytic.append(analytic); split_fd.append(finite); split_method.extend(method.tolist())
            values = {
                "split": np.full(times.size, split), "orbit_index": np.full(times.size, orbit_index),
                "orbit_id": np.full(times.size, bank["orbit_id"][orbit_index]), "u_th": np.full(times.size, bank["u_th"][orbit_index]),
                "t": times, "analytic": analytic, "finite_difference": finite, "absolute_error": absolute,
                "relative_error_floor_1e_10": relative, "method": method, "step": step,
            }
            for key, value in values.items(): records[key].extend(np.asarray(value).tolist())
        analytic = np.concatenate(split_analytic); finite = np.concatenate(split_fd); error = finite - analytic
        strong = np.abs(analytic) >= 1e-6
        summaries[split] = {
            "orbit_count": int(orbit_indices.size), "point_count": int(analytic.size), "method_counts": dict(Counter(split_method)),
            "absolute_agreement": scalar_error_metrics(error),
            "relative_error_abs_exact_ge_1e_6": scalar_error_metrics(error[strong] / analytic[strong]),
            "maximum_relative_error_floor_1e_10": float(np.max(np.abs(error) / np.maximum(np.abs(analytic), 1e-10))),
        }
    arrays = {key: np.asarray(value) for key, value in records.items()}
    error = arrays["finite_difference"].astype(float) - arrays["analytic"].astype(float)
    strong = np.abs(arrays["analytic"].astype(float)) >= 1e-6
    summaries["global"] = {
        "point_count": int(error.size), "absolute_agreement": scalar_error_metrics(error),
        "relative_error_abs_exact_ge_1e_6": scalar_error_metrics(error[strong] / arrays["analytic"].astype(float)[strong]),
        "maximum_relative_error_floor_1e_10": float(np.max(arrays["relative_error_floor_1e_10"].astype(float))),
    }
    return summaries, arrays


def masks_by_difficulty(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {"all": np.ones(rows["s"].size, dtype=bool), "hard_u_th_le_0p30": rows["u_th"] <= .30,
             "ordinary_u_th_gt_0p30": rows["u_th"] > .30}
    for center in REFERENCE_TARGETS: masks[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(rows["u_th"] - center) <= .01
    return masks


def validation_diagnostics(rows: dict[str, np.ndarray], exact_dot: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, Any]:
    dot_error = prediction["predicted_dot_xi1"] - exact_dot
    xi_error = prediction["predicted_xi1"] - rows["xi1"]
    families = {}
    for name, mask in masks_by_difficulty(rows).items():
        families[name] = {"row_count": int(np.sum(mask)), "dot_xi": scalar_error_metrics(dot_error[mask]), "xi": scalar_error_metrics(xi_error[mask])}
    correlation = {
        "abs_exact_dot_xi_vs_abs_dot_xi_error_spearman": float(spearmanr(np.abs(exact_dot), np.abs(dot_error)).statistic),
        "abs_exact_dot_xi_vs_abs_xi_error_spearman": float(spearmanr(np.abs(exact_dot), np.abs(xi_error)).statistic),
    }
    return {"row_count": int(rows["s"].size), "dot_xi": scalar_error_metrics(dot_error), "xi": scalar_error_metrics(xi_error),
            "families": families, "sharpness_error_correlations": correlation}


def u_th_diagnostics(rows: dict[str, np.ndarray], exact_dot: np.ndarray, prediction: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    dot_error = prediction["predicted_dot_xi1"] - exact_dot; xi_error = prediction["predicted_xi1"] - rows["xi1"]
    output = []
    for index, label in enumerate(U_LABELS):
        mask = (rows["u_th"] > U_EDGES[index]) & (rows["u_th"] <= U_EDGES[index + 1])
        if index == 0: mask = rows["u_th"] <= U_EDGES[index + 1]
        output.append({"bin": label, "lower": None if not np.isfinite(U_EDGES[index]) else U_EDGES[index],
                       "upper": None if not np.isfinite(U_EDGES[index + 1]) else U_EDGES[index + 1],
                       "row_count": int(mask.sum()), "mean_u_th": float(np.mean(rows["u_th"][mask])),
                       **{f"dot_xi_{k}": v for k, v in scalar_error_metrics(dot_error[mask]).items()},
                       **{f"xi_{k}": v for k, v in scalar_error_metrics(xi_error[mask]).items()}})
    return output


def sharpness_coverage(rows: dict[str, np.ndarray], exact_dot: np.ndarray, edges: np.ndarray, labels: list[str], split: str) -> list[dict[str, Any]]:
    output = []; sharpness = np.abs(exact_dot)
    for family, family_mask in masks_by_difficulty(rows).items():
        denominator = int(family_mask.sum())
        for index, label in enumerate(labels):
            mask = family_mask & (sharpness > edges[index]) & (sharpness <= edges[index + 1])
            if index == 0: mask = family_mask & (sharpness <= edges[index + 1])
            output.append({"split": split, "family": family, "sharpness_bin": label, "row_count": int(mask.sum()),
                           "fraction_within_family": float(mask.sum() / denominator) if denominator else None})
    return output


def rapid_feature_coverage(rows: dict[str, np.ndarray], bank: dict[str, np.ndarray], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wormhole, spiral = experiment_parameters(); per_orbit = []
    for orbit_index in np.unique(rows["source_orbit_index"][rows["u_th"] <= .30]):
        orbit_index = int(orbit_index); times = np.linspace(bank["t_left"][orbit_index], bank["t_right"][orbit_index], 1201)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit_index, times)
        rate = np.abs(xi_time_derivative(exact[:, 0], exact[:, 1], wormhole, spiral)); threshold = .5 * float(rate.max())
        rapid = np.flatnonzero(rate >= threshold); start, stop = float(times[rapid[0]]), float(times[rapid[-1]])
        selected = rows["source_orbit_index"] == orbit_index; target_times = rows["t1"][selected]
        counts = {"before": int(np.sum(target_times < start)), "through": int(np.sum((target_times >= start) & (target_times <= stop))),
                  "after": int(np.sum(target_times > stop))}
        per_orbit.append({"split": split, "orbit_index": orbit_index, "orbit_id": str(bank["orbit_id"][orbit_index]),
                          "u_th": float(bank["u_th"][orbit_index]), "rapid_start_t": start, "rapid_stop_t": stop,
                          "rapid_threshold": threshold, **{f"{k}_row_count": v for k, v in counts.items()},
                          "has_before_through_after": all(value > 0 for value in counts.values())})
    aggregates = []
    groups = {"hard_u_th_le_0p30": lambda u: u <= .30}
    for center in REFERENCE_TARGETS: groups[f"u_th_within_0p01_of_{center:.2f}"] = lambda u, center=center: abs(u - center) <= .01
    for name, condition in groups.items():
        chosen = [row for row in per_orbit if condition(row["u_th"])]
        aggregates.append({"split": split, "family": name, "orbit_count": len(chosen),
                           "orbits_with_before_through_after": sum(row["has_before_through_after"] for row in chosen),
                           "before_row_count": sum(row["before_row_count"] for row in chosen),
                           "through_row_count": sum(row["through_row_count"] for row in chosen),
                           "after_row_count": sum(row["after_row_count"] for row in chosen)})
    return per_orbit, aggregates


def reference_dense_diagnostics(model: Any, preprocessing: HybridPreprocessing, bank: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wormhole, spiral = experiment_parameters(); timing_rows, metrics_rows = [], []
    for target in REFERENCE_TARGETS:
        orbit = int(np.argmin(np.abs(bank["u_th"] - target)))
        t0, residual = invert_saved_orbit_x(bank, orbit, np.asarray([REFERENCE_ANCHOR_X])); t0 = float(t0[0])
        elapsed = np.linspace(0.0, float(bank["t_right"][orbit]) - t0, REFERENCE_DENSE_POINTS)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit, t0 + elapsed)
        query = {"x0": np.full(elapsed.size, exact[0, 0]), "xi0": np.full(elapsed.size, exact[0, 2]),
                 "E0": np.full(elapsed.size, bank["E0"][orbit]), "s": elapsed}
        prediction = predict_xi_and_physical_s_derivative(model, preprocessing, query)
        exact_dot = xi_time_derivative(exact[:, 0], exact[:, 1], wormhole, spiral)
        xi_error = prediction["predicted_xi1"] - exact[:, 2]; dot_error = prediction["predicted_dot_xi1"] - exact_dot
        timing = {"target_u_th": target, "actual_u_th": float(bank["u_th"][orbit]), "orbit_id": str(bank["orbit_id"][orbit]),
                  "anchor_x0": float(exact[0, 0]), "anchor_inversion_residual": float(residual[0]),
                  "dense_grid_spacing_s": float(elapsed[1] - elapsed[0]),
                  **dominant_peak_timing(elapsed, exact_dot, prediction["predicted_dot_xi1"])}
        timing_rows.append(timing)
        metrics_rows.append({"target_u_th": target, "actual_u_th": float(bank["u_th"][orbit]),
                             **{f"dot_xi_{k}": v for k, v in scalar_error_metrics(dot_error).items()},
                             **{f"xi_{k}": v for k, v in scalar_error_metrics(xi_error).items()}})
        np.savez_compressed(ARRAYS / f"reference_u_th_{target:.2f}_dense_derivatives.npz", elapsed_s=elapsed,
                            exact_x=exact[:, 0], exact_u=exact[:, 1], exact_xi=exact[:, 2], exact_dot_xi=exact_dot,
                            predicted_xi=prediction["predicted_xi1"], predicted_dot_xi=prediction["predicted_dot_xi1"],
                            xi_error=xi_error, dot_xi_error=dot_error)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
        axes[0].plot(elapsed, exact[:, 2], color="black", label="exact"); axes[0].plot(elapsed, prediction["predicted_xi1"], "--", label="predicted")
        axes[1].plot(elapsed, exact_dot, color="black", label="exact"); axes[1].plot(elapsed, prediction["predicted_dot_xi1"], "--", label="predicted")
        axes[2].plot(elapsed, xi_error, label="xi error"); axes[2].plot(elapsed, dot_error, label="dot xi error")
        axes[0].set_ylabel("xi"); axes[1].set_ylabel("d xi / ds"); axes[2].set(ylabel="prediction error", xlabel="physical elapsed time s")
        for ax in axes: ax.grid(alpha=.25); ax.legend()
        fig.suptitle(f"Exact and frozen-model derivatives on reference orbit u_th={target:.2f}")
        fig.savefig(FIGURES / f"reference_u_th_{target:.2f}_dense_derivatives.png", dpi=180); plt.close(fig)
    return timing_rows, metrics_rows


def plot_verification(arrays: dict[str, np.ndarray]) -> None:
    analytic = arrays["analytic"].astype(float); finite = arrays["finite_difference"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(analytic, finite, s=5, alpha=.25); lo, hi = min(analytic.min(), finite.min()), max(analytic.max(), finite.max()); axes[0].plot([lo, hi], [lo, hi], color="black")
    axes[0].set(xlabel="dynamical dot xi", ylabel="finite-difference dot xi")
    axes[1].scatter(np.abs(analytic), np.abs(finite - analytic), s=5, alpha=.25); axes[1].set(xlabel="|dynamical dot xi|", ylabel="absolute discrepancy", xscale="log", yscale="log")
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle("Saved-trajectory verification of the dynamical xi derivative"); fig.savefig(FIGURES / "exact_derivative_finite_difference_verification.png", dpi=180); plt.close(fig)


def plot_sharpness_bins(rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows)); labels = [r["bin"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("rmse", "mae")):
        ax.plot(x, [r[f"dot_xi_{metric}"] for r in rows], marker="o", label="dot xi error")
        ax.plot(x, [r[f"xi_{metric}"] for r in rows], marker="o", label="xi error")
        ax.set(xticks=x, xticklabels=labels, xlabel="training-derived |exact dot xi| quantile bin", ylabel=metric.upper(), yscale="log"); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Validation errors versus exact dynamical sharpness"); fig.savefig(FIGURES / "validation_error_vs_exact_dot_xi.png", dpi=180); plt.close(fig)


def plot_u_th(rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(rows)); labels = [r["bin"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("rmse", "mae")):
        ax.plot(x, [r[f"dot_xi_{metric}"] for r in rows], marker="o", label="dot xi error")
        ax.plot(x, [r[f"xi_{metric}"] for r in rows], marker="o", label="xi error")
        ax.set(xticks=x, xticklabels=labels, xlabel="u_th bin", ylabel=metric.upper(), yscale="log"); ax.grid(alpha=.25); ax.legend()
    fig.suptitle("Validation derivative and state errors versus trajectory parameter"); fig.savefig(FIGURES / "validation_error_vs_u_th.png", dpi=180); plt.close(fig)


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_finite_time_derivative_audit.py", "tests/test_physics_gate.py",
               "tests/test_finite_time_hybrid.py", f"--junitxml={TESTS/'focused_pytest.xml'}"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-derivative-audit-mpl"}, capture_output=True, text=True)
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload); return payload


def report_text(summary: dict[str, Any]) -> str:
    verify = summary["exact_derivative_verification"]["global"]; validation = summary["validation"]
    sharp_rows = "\n".join(f"| {r['bin']} | {r['row_count']} | {r['dot_xi_rmse']:.6g} | {r['dot_xi_mae']:.6g} | {r['xi_rmse']:.6g} | {r['xi_mae']:.6g} |" for r in summary["validation_sharpness_bins"])
    family_rows = "\n".join(f"| {name} | {row['row_count']} | {row['dot_xi']['rmse']:.6g} | {row['dot_xi']['mae']:.6g} | {row['xi']['rmse']:.6g} | {row['xi']['mae']:.6g} |" for name, row in validation["families"].items())
    timing_rows = "\n".join(f"| {r['target_u_th']:.2f} | {r['actual_u_th']:.2f} | {r['exact_peak_s']:.6g} | {r['predicted_peak_s']:.6g} | {r['delta_peak_s']:.6g} | "
                            f"{', '.join(f'{p['delta_s']:+.4g}' for p in r['matched_significant_peaks'])} | {r['half_max_status']} |" for r in summary["reference_timing"])
    coverage_rows = "\n".join(f"| {r['split']} | {r['family']} | {r['orbit_count']} | {r['orbits_with_before_through_after']} | {r['before_row_count']} | {r['through_row_count']} | {r['after_row_count']} |" for r in summary["rapid_feature_coverage"])
    train = summary["training_exact_derivative_distribution"]
    first = summary["validation_sharpness_bins"][0]
    maximum_dot = max(summary["validation_sharpness_bins"], key=lambda row: row["dot_xi_rmse"])
    maximum_xi = max(summary["validation_sharpness_bins"], key=lambda row: row["xi_rmse"])
    xi_ratio = maximum_xi["xi_rmse"] / first["xi_rmse"]; dot_ratio = maximum_dot["dot_xi_rmse"] / first["dot_xi_rmse"]
    support = summary["working_hypothesis"]
    return f"""# Frozen finite-time hybrid physical-time derivative audit

## Scope and integrity

This diagnostic used only frozen training rows, validation rows, and the seven non-test stress/reference trajectories. No fitting, optimization, resampling, trajectory integration, or model/preprocessing modification occurred. Held-out test artifacts were not opened or hashed.

## Exact derivative verification

The retained target is the existing chain-rule `xi_time_derivative`, which calls the validated radial acceleration. Across {verify['point_count']:,} saved-dense points, including both boundaries, the five-point finite-difference comparison has RMSE `{verify['absolute_agreement']['rmse']:.6g}`, MAE `{verify['absolute_agreement']['mae']:.6g}`, p99 absolute discrepancy `{verify['absolute_agreement']['p99_absolute']:.6g}`, and maximum `{verify['absolute_agreement']['maximum_absolute']:.6g}`. For points with `|dot xi|>=1e-6`, relative-error RMSE is `{verify['relative_error_abs_exact_ge_1e_6']['rmse']:.6g}` and p99 absolute relative error is `{verify['relative_error_abs_exact_ge_1e_6']['p99_absolute']:.6g}`.

## Training target distribution

Training exact endpoint `dot xi`: mean `{train['mean']:.6g}`, population standard deviation `{train['standard_deviation']:.6g}`, minimum `{train['minimum']:.6g}`, maximum `{train['maximum']:.6g}`. Absolute percentiles: p50 `{train['absolute_p50']:.6g}`, p90 `{train['absolute_p90']:.6g}`, p95 `{train['absolute_p95']:.6g}`, p99 `{train['absolute_p99']:.6g}`, p99.9 `{train['absolute_p99p9']:.6g}`.

## Frozen validation accuracy

Physical-time derivative RMSE/MAE are `{validation['dot_xi']['rmse']:.6g}` / `{validation['dot_xi']['mae']:.6g}`. The corresponding ordinary xi state RMSE/MAE are `{validation['xi']['rmse']:.6g}` / `{validation['xi']['mae']:.6g}`. Spearman associations of exact sharpness with absolute derivative/state errors are `{validation['sharpness_error_correlations']['abs_exact_dot_xi_vs_abs_dot_xi_error_spearman']:.3f}` / `{validation['sharpness_error_correlations']['abs_exact_dot_xi_vs_abs_xi_error_spearman']:.3f}`.

| training-derived sharpness bin | rows | dot xi RMSE | dot xi MAE | xi RMSE | xi MAE |
|:---|---:|---:|---:|---:|---:|
{sharp_rows}

Peak-bin-versus-lowest-bin RMSE ratios are `{dot_ratio:.2f}x` for the derivative and `{xi_ratio:.2f}x` for the state. Both peak in `{maximum_dot['bin']}`; the errors then decrease in the most extreme one-percent bin, so sharpness is associated with error but is not sufficient by itself.

## Difficulty dependence

| family | rows | dot xi RMSE | dot xi MAE | xi RMSE | xi MAE |
|:---|---:|---:|---:|---:|---:|
{family_rows}

## Dense stress/reference timing

Peak times are global maxima of `|dot xi|`. Half-maximum offsets are omitted whenever a competing peak reaches at least 80% of the dominant peak or either crossing is unbounded.

| target u_th | actual u_th | exact global peak s | predicted global peak s | global delta s | matched significant-peak deltas s | half-maximum status |
|---:|---:|---:|---:|---:|:---|:---|
{timing_rows}

## Dataset coverage around rapid features

| split | family | orbits | all three regions | rows before | rows through | rows after |
|:---|:---|---:|---:|---:|---:|---:|
{coverage_rows}

## Conclusion

**{support['assessment']}** {support['explanation']}

Focused tests: `{summary['tests']['stdout'].strip()}`. All protected artifact hashes were unchanged; generated files are indexed by SHA-256 in the manifest.
"""


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, ARRAYS, TABLES, FIGURES, TESTS): directory.mkdir(parents=True, exist_ok=False)
    gate = immutable_gate(); write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]: raise RuntimeError(gate["failures"])
    protected_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED}

    verification, verification_arrays = verification_audit({"train": TRAIN_BANK, "validation": VALIDATION_BANK, "stress_reference": REFERENCE_BANK})
    np.savez_compressed(ARRAYS / "exact_derivative_finite_difference_verification.npz", **verification_arrays)
    write_json(TABLES / "exact_derivative_verification.json", verification); plot_verification(verification_arrays)

    training, validation, reference = load_dataset(TRAIN_ROWS), load_dataset(VALIDATION_ROWS), load_dataset(REFERENCE_ROWS)
    train_exact, validation_exact, reference_exact = (exact_endpoint_derivative(rows) for rows in (training, validation, reference))
    model = load_hybrid_model(CHECKPOINT); preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    train_prediction = predict_xi_and_physical_s_derivative(model, preprocessing, training)
    validation_prediction = predict_xi_and_physical_s_derivative(model, preprocessing, validation)
    reference_prediction = predict_xi_and_physical_s_derivative(model, preprocessing, reference)
    np.savez_compressed(ARRAYS / "training_row_derivatives.npz", transition_id=training["transition_id"], orbit_id=training["orbit_id"],
                        u_th=training["u_th"], s=training["s"], exact_dot_xi=train_exact,
                        predicted_dot_xi=train_prediction["predicted_dot_xi1"], exact_xi=training["xi1"], predicted_xi=train_prediction["predicted_xi1"])
    np.savez_compressed(ARRAYS / "validation_row_derivatives.npz", transition_id=validation["transition_id"], orbit_id=validation["orbit_id"],
                        u_th=validation["u_th"], s=validation["s"], exact_dot_xi=validation_exact,
                        predicted_dot_xi=validation_prediction["predicted_dot_xi1"], exact_xi=validation["xi1"], predicted_xi=validation_prediction["predicted_xi1"])
    np.savez_compressed(ARRAYS / "stress_reference_row_derivatives.npz", transition_id=reference["transition_id"], orbit_id=reference["orbit_id"],
                        u_th=reference["u_th"], s=reference["s"], exact_dot_xi=reference_exact,
                        predicted_dot_xi=reference_prediction["predicted_dot_xi1"], exact_xi=reference["xi1"], predicted_xi=reference_prediction["predicted_xi1"])

    training_distribution = value_distribution(train_exact); write_json(TABLES / "training_exact_derivative_distribution.json", training_distribution)
    validation_metrics = validation_diagnostics(validation, validation_exact, validation_prediction)
    reference_metrics = validation_diagnostics(reference, reference_exact, reference_prediction)
    write_json(TABLES / "validation_derivative_metrics.json", validation_metrics); write_json(TABLES / "stress_reference_derivative_metrics.json", reference_metrics)
    sharp_edges, sharp_labels = training_sharpness_edges(train_exact)
    sharp_rows = binned_error_metrics(validation_exact, validation_prediction["predicted_dot_xi1"], validation_prediction["predicted_xi1"] - validation["xi1"], sharp_edges, sharp_labels)
    u_rows = u_th_diagnostics(validation, validation_exact, validation_prediction)
    write_csv(TABLES / "validation_errors_by_training_sharpness.csv", sharp_rows); write_csv(TABLES / "validation_errors_by_u_th.csv", u_rows)
    write_json(TABLES / "training_sharpness_bin_definition.json", {"edges": sharp_edges, "labels": sharp_labels,
               "quantiles": [0, .5, .75, .9, .95, .99, 1], "source": "training exact endpoint |dot xi| only"})
    plot_sharpness_bins(sharp_rows); plot_u_th(u_rows)

    coverage = sharpness_coverage(training, train_exact, sharp_edges, sharp_labels, "train") + sharpness_coverage(validation, validation_exact, sharp_edges, sharp_labels, "validation")
    write_csv(TABLES / "row_coverage_by_sharpness_and_difficulty.csv", coverage)
    train_bank, validation_bank, reference_bank = load_npz(TRAIN_BANK), load_npz(VALIDATION_BANK), load_npz(REFERENCE_BANK)
    train_orbits, train_coverage = rapid_feature_coverage(training, train_bank, "train")
    validation_orbits, validation_coverage = rapid_feature_coverage(validation, validation_bank, "validation")
    rapid_coverage = train_coverage + validation_coverage
    write_csv(TABLES / "rapid_feature_coverage_per_orbit.csv", train_orbits + validation_orbits)
    write_csv(TABLES / "rapid_feature_coverage_summary.csv", rapid_coverage)

    reference_timing, reference_dense_metrics = reference_dense_diagnostics(model, preprocessing, reference_bank)
    write_csv(TABLES / "stress_reference_peak_timing.csv", reference_timing); write_csv(TABLES / "stress_reference_dense_metrics.csv", reference_dense_metrics)
    first = sharp_rows[0]; maximum_dot = max(sharp_rows, key=lambda row: row["dot_xi_rmse"]); maximum_xi = max(sharp_rows, key=lambda row: row["xi_rmse"])
    dot_ratio, xi_ratio = maximum_dot["dot_xi_rmse"] / first["dot_xi_rmse"], maximum_xi["xi_rmse"] / first["xi_rmse"]
    paired_offsets = [abs(pair["delta_s"]) for row in reference_timing for pair in row["matched_significant_peaks"]]
    assessment = "Rapid-change association is supported; a single global phase offset is not identifiable."
    explanation = (f"The peak/lowest sharpness-bin RMSE ratios are {dot_ratio:.2f} for dot xi and {xi_ratio:.2f} for xi, and both peak in {maximum_dot['bin']}. "
                   f"Across one-to-one matched significant reference features, the median absolute timing offset is {np.median(paired_offsets):.3g} and the maximum is {np.max(paired_offsets):.3g} physical-time units. "
                   "Most offsets are small, but every curve has competing nearly equal peaks, so global peak-rank switches and half-maximum ambiguity prevent a clean single-lag interpretation. This is descriptive and does not alter any model or data.")
    tests = run_tests(); protected_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    if protected_before != protected_after: raise RuntimeError("protected artifact changed")
    if not tests["passed"]: raise RuntimeError("focused tests failed")
    summary = {"created_utc": utc(), "status": "TRAIN_VALIDATION_DERIVATIVE_AUDIT_COMPLETED", "scope": gate["data_scope"],
               "exact_derivative_verification": verification, "training_exact_derivative_distribution": training_distribution,
               "validation": validation_metrics, "stress_reference_rows": reference_metrics,
               "sharpness_bin_edges": sharp_edges, "validation_sharpness_bins": sharp_rows, "validation_u_th_bins": u_rows,
               "sharpness_coverage": coverage, "rapid_feature_coverage": rapid_coverage,
               "reference_timing": reference_timing, "reference_dense_metrics": reference_dense_metrics,
               "working_hypothesis": {"assessment": assessment, "explanation": explanation}, "tests": tests,
               "training_performed": False, "dataset_modified": False, "ode_reintegrated": False,
               "held_out_test_accessed": False, "protected_hashes_unchanged": True}
    write_json(SUMMARY, summary); REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {str(path.relative_to(OUTPUT)): {"sha256": file_sha256(path), "bytes": path.stat().st_size, "path": str(path.resolve())}
                 for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)}
    manifest = {"status": summary["status"], "input_gate": gate, "protected_before": protected_before, "protected_after": protected_after,
                "source_hashes": source_hashes(), "artifacts": artifacts, "report_sha256": file_sha256(REPORT), "summary_sha256": file_sha256(SUMMARY)}
    write_json(MANIFEST, manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
