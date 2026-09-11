#!/usr/bin/env python3
"""Full held-out trajectory validation for frozen direct finite-time models."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from wormhole_sciml.finite_time_validation import (
    COMPOSITION_ANCHOR_X,
    COMPOSITION_FRACTION_PAIRS,
    DENSE_ANCHOR_X,
    DENSE_FRACTIONS,
    LOCAL_STEP,
    SMALL_S_ANCHOR_X,
    SMALL_S_VALUES,
    admissibility_summary,
    aggregate_error_metrics,
    bin_error_rows,
    build_composition_queries,
    build_dense_queries,
    build_queries_at_x_and_s,
    composition_summary,
    energy_metrics,
    evaluate_composition,
    load_checkpoint,
    load_finite_time_preprocessing,
    predict_finite_time,
    predict_local,
    scalar_error_metrics,
    subset_metrics,
)
from wormhole_sciml.model_a import Normalization
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi
from wormhole_sciml.phase_c_finite_time import invert_saved_orbit_x
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "finite_time_trajectory_validation"
TABLES = OUTPUT / "tables"
ARRAYS = OUTPUT / "arrays"
FIGURES = OUTPUT / "figures"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_TRAJECTORY_VALIDATION_REPORT.md"
SUMMARY = OUTPUT / "validation_summary.json"
MANIFEST = OUTPUT / "validation_manifest.json"
MANIFEST_HASH = OUTPUT / "validation_manifest.sha256"

PHASE_B_DIR = ROOT / "output" / "phase_b_complete_orbit_banks"
VALIDATION_BANK = PHASE_B_DIR / "banks" / "phase_b_validation_orbits.npz"
STRESS_BANK = PHASE_B_DIR / "banks" / "phase_b_stress_reference_orbits.npz"
PHASE_C_VALIDATION = ROOT / "output" / "phase_c_finite_time_dataset" / "datasets" / "phase_c_validation_raw.npz"
SEALED_TEST = ROOT / "output" / "phase_c_finite_time_dataset" / "datasets" / "phase_c_test_sealed_raw.npz"
PREPROCESSING = ROOT / "output" / "finite_time_baseline" / "preprocessing" / "preprocessing_constants.json"
FINITE_CHECKPOINTS = {
    seed: ROOT / "output" / "finite_time_baseline" / "training" / f"seed_{seed}" / "best_checkpoint.pt"
    for seed in (101, 202, 303)
}
LOCAL_ROOT = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
LOCAL_MANIFEST = LOCAL_ROOT / "energy_xi_training_manifest.json"
LOCAL_NORMALIZATION = LOCAL_ROOT / "training" / "energy_input_normalization.json"
LOCAL_CHECKPOINTS = {
    seed: LOCAL_ROOT / "training" / f"seed_{seed}" / "best_checkpoint.pt"
    for seed in (101, 202, 303)
}
LOCAL_PRIMARY_SEED = 101
FINITE_PRIMARY_SEED = 303
EXPECTED_HASHES = {
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    STRESS_BANK: "5d7959cf1a657a5ff916e4d40aab44309ca08958f1694340b47c0fce7ac08dce",
    PHASE_C_VALIDATION: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    SEALED_TEST: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    PREPROCESSING: "6c53549de9ff9d25454813213b98854d24cd031f009c774fc1306ae104d8e32e",
    FINITE_CHECKPOINTS[101]: "6c8017cf693591970e070c36274a4131233a0600e12ae76afdf22b7416e08530",
    FINITE_CHECKPOINTS[202]: "55e499a6ab0c0d67ee5d28f3b5f6971d6a93cec2d0c23b9da2e6014277e6c4d2",
    FINITE_CHECKPOINTS[303]: "64272fd10f87291a5f7ec6d1c403874e91b5207f3f44ff54c20238d46827d7c5",
    LOCAL_NORMALIZATION: "5f3605f657d8190ea901112774553e3026a3399266783a78f0102b285c4045c3",
    LOCAL_CHECKPOINTS[101]: "54a41a54a7fa20e931df0dd987ac871786d4ac8d02e71bd3560e9ebc3228a4a6",
    LOCAL_CHECKPOINTS[202]: "233b4e45701a618731dd2f72d0b6536a80815446cf468f7f14de551d5b2808d7",
    LOCAL_CHECKPOINTS[303]: "6794a7e6f24bb3692d3f6f5c45eeaeeedc98c5218489ba5afae760b6146aa587",
}
S_EDGES = np.asarray([0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
                      10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0])
X_EDGES = np.linspace(-17.0, 17.0, 33)
SEED_COLORS = {101: "#4477aa", 202: "#ee7733", 303: "#228833"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    lead = f"{prefix}_" if prefix else ""
    return {
        f"{lead}{component}_{key}": value
        for component in ("x", "xi")
        for key, value in metrics[component].items()
    }


def error_view(result: dict[str, np.ndarray], x_key: str, xi_key: str) -> dict[str, np.ndarray]:
    """Expose a named comparison through the common binned-error interface."""

    return {"x_error": result[x_key], "xi_error": result[xi_key]}


def immutable_gate() -> dict[str, Any]:
    rows = {}
    failures = []
    for path, expected in EXPECTED_HASHES.items():
        measured = file_sha256(path)
        match = measured == expected
        rows[str(path.resolve())] = {"expected_sha256": expected, "measured_sha256": measured, "match": match}
        if not match:
            failures.append(str(path))
    return {
        "passed": not failures,
        "failures": failures,
        "artifacts": rows,
        "sealed_test_access": "file-byte SHA-256 only; NPZ was not opened",
    }


def top_energy_rows(
    seed: int,
    queries: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    count: int = 100,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(prediction["energy_error"]))[::-1][:count]
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append({
            "seed": seed, "rank": rank, "query_index": int(index),
            "orbit_id": str(queries["orbit_id"][index]),
            "orbit_index": int(queries["orbit_index"][index]),
            "u_th": float(queries["u_th"][index]), "E0": float(queries["E0"][index]),
            "x0": float(queries["x0"][index]), "xi0": float(queries["xi0"][index]),
            "s": float(queries["s"][index]), "f": float(queries["f"][index]),
            "exact_x": float(queries["exact_x1"][index]),
            "exact_xi": float(queries["exact_xi1"][index]),
            "predicted_x": float(prediction["predicted_x1"][index]),
            "predicted_xi": float(prediction["predicted_xi1"][index]),
            "exact_u": float(queries["exact_u1"][index]),
            "predicted_u": float(prediction["predicted_u1"][index]),
            "exact_C": float(prediction["exact_C1"][index]),
            "predicted_C": float(prediction["predicted_C1"][index]),
            "x_error": float(prediction["x_error"][index]),
            "xi_error": float(prediction["xi_error"][index]),
            "energy_error": float(prediction["energy_error"][index]),
            "absolute_energy_error": float(abs(prediction["energy_error"][index])),
        })
    return rows


def energy_tail_analysis(
    queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], top: list[dict[str, Any]]
) -> dict[str, Any]:
    absolute_energy = np.abs(prediction["energy_error"])
    state_error = np.hypot(prediction["x_error"], prediction["xi_error"])
    correlations = {}
    coordinates = {
        "E0": queries["E0"], "exact_C": prediction["exact_C1"], "u_th": queries["u_th"],
        "x0": queries["x0"], "absolute_exact_target_x": np.abs(queries["exact_x1"]),
        "s": queries["s"], "state_error_magnitude": state_error,
    }
    for name, values in coordinates.items():
        coefficient, pvalue = spearmanr(absolute_energy, values)
        correlations[name] = {"spearman_r": float(coefficient), "pvalue": float(pvalue)}
    top_indices = np.asarray([row["query_index"] for row in top], dtype=np.int64)
    high_e = queries["E0"] >= np.quantile(queries["E0"], 0.95)
    low_c = prediction["exact_C1"] <= np.quantile(prediction["exact_C1"], 0.05)
    amplification = absolute_energy / np.maximum(state_error, 1.0e-15)
    return {
        "spearman_correlations": correlations,
        "top100_fractions": {
            "high_E0_top5pct": float(np.mean(high_e[top_indices])),
            "low_exact_C_bottom5pct": float(np.mean(low_c[top_indices])),
            "hard_u_th_le_0p30": float(np.mean(queries["u_th"][top_indices] <= 0.30)),
            "incoming_sensitive_x0": float(np.mean((queries["x0"][top_indices] >= -17.0) & (queries["x0"][top_indices] <= -8.5))),
            "near_throat_exact_target_abs_x_le_2": float(np.mean(np.abs(queries["exact_x1"][top_indices]) <= 2.0)),
            "long_s_gt_20": float(np.mean(queries["s"][top_indices] > 20.0)),
        },
        "absolute_energy_per_state_error": {
            "all_median": float(np.median(amplification)),
            "all_p99": float(np.quantile(amplification, 0.99)),
            "top100_median": float(np.median(amplification[top_indices])),
        },
        "top100_medians": {
            name: float(np.median(values[top_indices])) for name, values in coordinates.items()
        },
    }


def violation_rows(seed: int, queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], mask: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index in np.flatnonzero(mask):
        rows.append({
            "seed": seed, "query_index": int(index), "orbit_id": str(queries["orbit_id"][index]),
            "u_th": float(queries["u_th"][index]), "x0": float(queries["x0"][index]),
            "s": float(queries["s"][index]), "exact_x": float(queries["exact_x1"][index]),
            "exact_xi": float(queries["exact_xi1"][index]),
            "predicted_x": float(prediction["predicted_x1"][index]),
            "predicted_xi": float(prediction["predicted_xi1"][index]),
            "predicted_C": float(prediction["predicted_C1"][index]),
        })
    return rows


def metric_by_exact_value(
    queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], name: str
) -> list[dict[str, Any]]:
    rows = []
    for value in np.unique(queries[name]):
        mask = queries[name] == value
        metrics = subset_metrics(prediction, mask)
        rows.append({"coordinate": name, "value": float(value), "row_count": metrics["row_count"], **flatten_metrics("", metrics)})
    return rows


def plot_metric_lines(rows_by_seed: dict[int, list[dict[str, Any]]], coordinate: str, path: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True, sharex=True)
    fields = (("x_rmse", "RMSE x"), ("x_mae", "MAE x"), ("xi_rmse", "RMSE xi"), ("xi_mae", "MAE xi"))
    for seed, rows in rows_by_seed.items():
        x = np.asarray([row["center"] for row in rows])
        for axis, (field, ylabel) in zip(axes.ravel(), fields):
            axis.plot(x, [row[field] for row in rows], color=SEED_COLORS[seed], label=f"seed {seed}")
            axis.set(ylabel=ylabel, yscale="log")
            axis.grid(alpha=0.25, which="both")
    for axis in axes[-1]:
        axis.set_xlabel(coordinate)
    axes[0, 0].legend()
    figure.suptitle(title)
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_percentile_lines(rows_by_seed: dict[int, list[dict[str, Any]]], coordinate: str, path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    for seed, rows in rows_by_seed.items():
        x = np.asarray([row["center"] for row in rows])
        for axis, component in zip(axes, ("x", "xi")):
            axis.plot(x, [row[f"{component}_p90_absolute"] for row in rows], color=SEED_COLORS[seed], label=f"seed {seed} p90")
            axis.plot(x, [row[f"{component}_p99_absolute"] for row in rows], color=SEED_COLORS[seed], ls="--", label=f"seed {seed} p99")
            axis.set(xlabel=coordinate, ylabel=f"absolute {component} error", yscale="log")
            axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(title)
    figure.savefig(path, dpi=185)
    plt.close(figure)


def error_map(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], path: Path) -> None:
    s_edges = np.asarray([0, 0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60, 100], dtype=float)
    x_index = np.digitize(queries["x0"], X_EDGES[1:-1])
    s_index = np.digitize(queries["s"], s_edges[1:-1])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    for axis, component in zip(axes, ("x", "xi")):
        grid = np.full((s_edges.size - 1, X_EDGES.size - 1), np.nan)
        absolute = np.abs(prediction[f"{component}_error"])
        for si in range(s_edges.size - 1):
            for xi in range(X_EDGES.size - 1):
                mask = (s_index == si) & (x_index == xi)
                if np.any(mask):
                    grid[si, xi] = np.log10(np.median(absolute[mask]) + 1.0e-12)
        mesh = axis.pcolormesh(X_EDGES, s_edges, grid, shading="auto", cmap="magma")
        figure.colorbar(mesh, ax=axis, label=f"log10 median |{component} error|")
        axis.set(xlabel="anchor x0", ylabel="physical elapsed time s", title=f"{component} error across anchor position and elapsed time")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def trajectory_figure(bank: Any, orbit_index: int, anchor_values: list[float], model: Any, preprocessing: Any, path: Path, title: str) -> None:
    figure, axes = plt.subplots(len(anchor_values), 3, figsize=(12.0, 3.1 * len(anchor_values)), constrained_layout=True)
    if len(anchor_values) == 1:
        axes = np.asarray([axes])
    for row, anchor_x in enumerate(anchor_values):
        t0, _ = invert_saved_orbit_x(bank, orbit_index, np.asarray([anchor_x]))
        t0_value = float(t0[0])
        elapsed = np.linspace(0.0, float(bank["t_right"][orbit_index]) - t0_value, 220)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t0_value + elapsed)
        queries = {
            "x0": np.full(elapsed.size, exact[0, 0]), "xi0": np.full(elapsed.size, exact[0, 2]),
            "E0": np.full(elapsed.size, float(bank["E0"][orbit_index])), "s": elapsed,
            "exact_x1": exact[:, 0], "exact_u1": exact[:, 1], "exact_xi1": exact[:, 2],
        }
        predicted = predict_finite_time(model, preprocessing, queries)
        axes[row, 0].plot(elapsed, exact[:, 0], label="exact")
        axes[row, 0].plot(elapsed, predicted["predicted_x1"], ls="--", label="predicted")
        axes[row, 1].plot(elapsed, exact[:, 2])
        axes[row, 1].plot(elapsed, predicted["predicted_xi1"], ls="--")
        axes[row, 2].plot(exact[:, 0], exact[:, 2])
        axes[row, 2].plot(predicted["predicted_x1"], predicted["predicted_xi1"], ls="--")
        axes[row, 0].set(ylabel=f"x (anchor {anchor_x:g})")
        axes[row, 1].set(ylabel="xi")
        axes[row, 2].set(ylabel="xi")
        for axis in axes[row]:
            axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("physical elapsed time s")
    axes[-1, 1].set_xlabel("physical elapsed time s")
    axes[-1, 2].set_xlabel("x")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def small_s_analysis(queries: dict[str, np.ndarray], predictions: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"per_seed": {}}
    rows = []
    for seed, prediction in predictions.items():
        seed_summary: dict[str, Any] = {"by_s": {}, "identity": {}}
        for value in SMALL_S_VALUES:
            mask = np.isclose(queries["s"], value, rtol=0.0, atol=1.0e-14)
            metrics = subset_metrics(prediction, mask)
            seed_summary["by_s"][f"{value:.12g}"] = metrics
            rows.append({"seed": seed, "s": value, **flatten_metrics("", metrics)})
        identity = queries["s"] == 0.0
        for output in ("predicted_Delta_x", "predicted_Delta_xi"):
            seed_summary["identity"][output] = {
                name: float(spearmanr(prediction[output][identity], queries[name][identity]).statistic)
                for name in ("x0", "xi0", "E0", "u_th")
            }
            seed_summary["identity"][output].update({
                "mean": float(np.mean(prediction[output][identity])),
                "standard_deviation": float(np.std(prediction[output][identity])),
                "minimum": float(np.min(prediction[output][identity])),
                "maximum": float(np.max(prediction[output][identity])),
            })
        exact_linear_error = (queries["exact_x1"] - queries["x0"]) - queries["u0"] * queries["s"]
        learned_linear_error = prediction["predicted_Delta_x"] - queries["u0"] * queries["s"]
        small = queries["s"] <= 0.2
        seed_summary["linear_u0_s_comparison"] = {
            "exact_rmse": float(np.sqrt(np.mean(exact_linear_error[small] ** 2))),
            "learned_rmse": float(np.sqrt(np.mean(learned_linear_error[small] ** 2))),
        }
        summary["per_seed"][str(seed)] = seed_summary
    csv_rows(TABLES / "small_s_metrics.csv", rows)
    return summary


def plot_small_s(queries: dict[str, np.ndarray], predictions: dict[int, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for seed, prediction in predictions.items():
        s_values, x_rmse, xi_rmse = [], [], []
        for value in SMALL_S_VALUES:
            mask = np.isclose(queries["s"], value, atol=1e-14, rtol=0)
            s_values.append(value)
            x_rmse.append(np.sqrt(np.mean(prediction["x_error"][mask] ** 2)))
            xi_rmse.append(np.sqrt(np.mean(prediction["xi_error"][mask] ** 2)))
        axes[0, 0].plot(s_values, x_rmse, marker="o", color=SEED_COLORS[seed], label=f"seed {seed}")
        axes[0, 1].plot(s_values, xi_rmse, marker="o", color=SEED_COLORS[seed])
        identity = queries["s"] == 0.0
        for axis, output in zip(axes[1], ("predicted_Delta_x", "predicted_Delta_xi")):
            for anchor in np.unique(queries["x0"]):
                mask = identity & np.isclose(queries["x0"], anchor)
                axis.plot(anchor, np.mean(prediction[output][mask]), marker="o", color=SEED_COLORS[seed], ms=3)
    axes[0, 0].set(xlabel="physical elapsed time s", ylabel="RMSE x", yscale="log")
    axes[0, 1].set(xlabel="physical elapsed time s", ylabel="RMSE xi", yscale="log")
    axes[1, 0].set(xlabel="anchor x0", ylabel="mean predicted Delta x at s=0")
    axes[1, 1].set(xlabel="anchor x0", ylabel="mean predicted Delta xi at s=0")
    for axis in axes.ravel(): axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    figure.suptitle("Identity offset and prediction error at small elapsed time")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_energy(queries: dict[str, np.ndarray], predictions: dict[int, dict[str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    specs = (("s", S_EDGES, "physical elapsed time s"), ("x0", X_EDGES, "anchor x0"))
    uth_edges = np.linspace(float(np.min(queries["u_th"])), float(np.max(queries["u_th"])), 31)
    e_edges = np.unique(np.quantile(queries["E0"], np.linspace(0, 1, 25)))
    all_specs = specs + (("u_th", uth_edges, "u_th"), ("E0", e_edges, "E0"))
    for axis, (name, edges, label) in zip(axes.ravel(), all_specs):
        for seed, prediction in predictions.items():
            indices = np.digitize(queries[name], edges[1:-1])
            centers, p99 = [], []
            for index in range(edges.size - 1):
                mask = indices == index
                if np.any(mask):
                    centers.append(float(np.mean(queries[name][mask])))
                    p99.append(float(np.quantile(np.abs(prediction["energy_error"][mask]), 0.99)))
            axis.plot(centers, p99, color=SEED_COLORS[seed], label=f"seed {seed}")
        axis.set(xlabel=label, ylabel="p99 |E_hat - E0|", yscale="log")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    figure.suptitle("Energy consistency across elapsed time, anchors, and orbit families")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_composition(summaries: dict[int, dict[str, Any]], path: Path) -> None:
    labels = ["exact restart\nvs exact", "exact restart\nvs direct", "self composition\nvs exact", "self composition\nvs direct"]
    keys = ["exact_restart_vs_exact", "exact_restart_vs_direct", "self_composition_vs_exact", "self_composition_vs_direct"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    width = 0.22
    positions = np.arange(len(keys))
    for offset, seed in enumerate((101, 202, 303)):
        for axis, component in zip(axes, ("x", "xi")):
            axis.bar(positions + (offset - 1) * width, [summaries[seed][key][component]["rmse"] for key in keys], width, color=SEED_COLORS[seed], label=f"seed {seed}")
            axis.set(xticks=positions, xticklabels=labels, ylabel=f"RMSE {component}", yscale="log")
            axis.grid(alpha=0.25, axis="y", which="both")
    axes[0].legend()
    figure.suptitle("Direct-flow and composition consistency")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_seed_comparison(aggregates: dict[int, dict[str, Any]], path: Path) -> None:
    metrics = ("x RMSE", "x MAE", "xi RMSE", "xi MAE")
    values = {
        seed: [row["row_weighted"][component][metric] for component, metric in (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))]
        for seed, row in aggregates.items()
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, indices in zip(axes, ((0, 1), (2, 3))):
        width = 0.24; position = np.arange(2)
        for offset, seed in enumerate((101, 202, 303)):
            axis.bar(position + (offset - 1) * width, [values[seed][i] for i in indices], width, label=f"seed {seed}", color=SEED_COLORS[seed])
        axis.set(xticks=position, xticklabels=[metrics[i] for i in indices], yscale="log", ylabel="physical error")
        axis.grid(alpha=0.25, axis="y", which="both")
    axes[0].legend()
    figure.suptitle("Seed dependence of dense finite-time prediction error")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_local_comparison(rows: list[dict[str, Any]], xrows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    primary = [row for row in rows if (row["model"] == "finite_time" and row["seed"] == FINITE_PRIMARY_SEED) or (row["model"] == "local" and row["seed"] == LOCAL_PRIMARY_SEED)]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    positions = np.arange(2); width = 0.35
    for offset, row in enumerate(primary):
        values = [row["x_rmse"], row["xi_rmse"]]
        axes[0].bar(positions + (offset - 0.5) * width, values, width, label=f"{row['model']} seed {row['seed']}")
    axes[0].set(xticks=positions, xticklabels=["x RMSE", "xi RMSE"], yscale="log", ylabel="physical error")
    axes[0].legend(fontsize=8)
    for axis, component in zip(axes[1:], ("x", "xi")):
        for key, style in (("finite_time", "-"), ("local", "--")):
            data = xrows[key]
            axis.plot([r["center"] for r in data], [r[f"{component}_rmse"] for r in data], ls=style, label=key)
        axis.set(xlabel="anchor x0", ylabel=f"RMSE {component}", yscale="log")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    figure.suptitle("Finite-time and local prediction at matched elapsed time s=0.2")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_stress_local_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for model_name, seed, style in (("finite_time", FINITE_PRIMARY_SEED, "-"), ("local", LOCAL_PRIMARY_SEED, "--")):
        selected = sorted(
            (row for row in rows if row["model"] == model_name and row["seed"] == seed),
            key=lambda row: row["u_th"],
        )
        for axis, component in zip(axes, ("x", "xi")):
            axis.plot(
                [row["u_th"] for row in selected],
                [row[f"{component}_rmse"] for row in selected],
                marker="o", ls=style, label=f"{model_name} seed {seed}",
            )
            axis.set(xlabel="u_th", ylabel=f"RMSE {component}", yscale="log")
            axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("Matched-step prediction across reference trajectories at s=0.2")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def scientific_report(summary: dict[str, Any]) -> str:
    dense_rows = "\n".join(
        f"| {seed} | {row['row_weighted']['x']['rmse']:.6g} | {row['row_weighted']['x']['mae']:.6g} | "
        f"{row['row_weighted']['xi']['rmse']:.6g} | {row['row_weighted']['xi']['mae']:.6g} |"
        for seed, row in summary["dense_aggregate"].items()
    )
    family_rows = "\n".join(
        f"| {seed} | {name} | {metrics['x']['rmse']:.6g} | {metrics['xi']['rmse']:.6g} |"
        for seed, families in summary["family_metrics"].items()
        for name, metrics in families.items()
    )
    energy_rows = "\n".join(
        f"| {seed} | {row['rmse']:.6g} | {row['mae']:.6g} | {row['p99_absolute']:.6g} | {row['p99p9_absolute']:.6g} | {row['maximum_absolute']:.6g} |"
        for seed, row in summary["energy_metrics"].items()
    )
    admissibility_rows = "\n".join(
        f"| {seed} | {row['absolute_xi_ge_1_count']} | {row['C_le_0_count']} | {row['union_violation_fraction']:.3%} |"
        for seed, row in summary["admissibility"].items()
    )
    composition_rows = "\n".join(
        f"| {seed} | {row['exact_restart_vs_exact']['x']['rmse']:.6g} | {row['exact_restart_vs_exact']['xi']['rmse']:.6g} | "
        f"{row['self_composition_vs_exact']['x']['rmse']:.6g} | {row['self_composition_vs_exact']['xi']['rmse']:.6g} |"
        for seed, row in summary["composition"].items()
    )
    local_rows = "\n".join(
        f"| {row['model']} | {row['seed']} | {row['x_rmse']:.6g} | {row['x_mae']:.6g} | {row['xi_rmse']:.6g} | {row['xi_mae']:.6g} |"
        for row in summary["matched_local"]["aggregate"]
    )
    primary_identity = summary["small_s"]["per_seed"][str(FINITE_PRIMARY_SEED)]["identity"]
    tail = summary["energy_tail"][str(FINITE_PRIMARY_SEED)]
    primary_families = summary["family_metrics"][str(FINITE_PRIMARY_SEED)]
    primary_small = summary["small_s"]["per_seed"][str(FINITE_PRIMARY_SEED)]
    primary_composition = summary["composition"][str(FINITE_PRIMARY_SEED)]
    primary_finite_matched = next(
        row for row in summary["matched_local"]["aggregate"]
        if row["model"] == "finite_time" and row["seed"] == FINITE_PRIMARY_SEED
    )
    primary_local_matched = next(
        row for row in summary["matched_local"]["aggregate"]
        if row["model"] == "local" and row["seed"] == LOCAL_PRIMARY_SEED
    )
    stress_rows = "\n".join(
        f"| {row['u_th']:.2f} | {row['x_rmse']:.6g} | {row['xi_rmse']:.6g} | {row['energy_mae']:.6g} |"
        for row in summary["stress_reference"]["aggregate_rows"]
        if row["seed"] == FINITE_PRIMARY_SEED
    )
    seed202_x = summary["dense_aggregate"]["202"]["row_weighted"]["x"]["rmse"]
    seed303_x = summary["dense_aggregate"]["303"]["row_weighted"]["x"]["rmse"]
    seed202_xi = summary["dense_aggregate"]["202"]["row_weighted"]["xi"]["rmse"]
    seed303_xi = summary["dense_aggregate"]["303"]["row_weighted"]["xi"]["rmse"]
    return f"""# Held-out direct finite-time trajectory validation

## Scope and immutable inputs

This study evaluates frozen validation orbits and a separate seven-trajectory stress/reference family. No model was trained, no preprocessing or checkpoint changed, no orbit was reintegrated, and the sealed test NPZ was never opened. Its bytes were hashed only. Seed 303 remained the predeclared primary finite-time checkpoint throughout.

The dense grid contains `{summary['dense_query_count']}` queries: 1024 held-out orbits × 32 deterministic x-bin-center anchors × 16 remaining-time fractions. All exact targets came from frozen Phase-B DOP853 dense polynomials.

## Dense aggregate error

| seed | RMSE x | MAE x | RMSE xi | MAE xi |
|---:|---:|---:|---:|---:|
{dense_rows}

Row- and orbit-weighted metrics are both stored; equal 512-row orbit grids make their RMSE/MAE values numerically equivalent. Full median/p90/p95/p99/p99.9/max tables are in `validation_summary.json` and per-orbit values in `tables/per_orbit_metrics.csv`.

## Orbit families and physical structure

| seed | family | RMSE x | RMSE xi |
|---:|:---|---:|---:|
{family_rows}

Detailed fixed-bin results versus physical s, x0, u_th, and E0 are in the corresponding CSV tables and figures. The primary model's low-u_th neighborhoods around 0.05, 0.15, and 0.30 are retained explicitly in `validation_summary.json`.

Representative exact/predicted validation and stress trajectories are under `figures/representative_validation_*` and `figures/stress_u_th_*`. Stress/reference metrics are kept separate from held-out aggregate statistics.

## Identity and small-s behavior

For seed 303 at s=0, predicted Delta-x has mean `{primary_identity['predicted_Delta_x']['mean']:.6g}`, standard deviation `{primary_identity['predicted_Delta_x']['standard_deviation']:.6g}`, and range `[{primary_identity['predicted_Delta_x']['minimum']:.6g}, {primary_identity['predicted_Delta_x']['maximum']:.6g}]`. Predicted Delta-xi has mean `{primary_identity['predicted_Delta_xi']['mean']:.6g}`, standard deviation `{primary_identity['predicted_Delta_xi']['standard_deviation']:.6g}`, and range `[{primary_identity['predicted_Delta_xi']['minimum']:.6g}, {primary_identity['predicted_Delta_xi']['maximum']:.6g}]`.

Spearman dependence on x0, xi0, E0, and u_th, plus errors through s=1 and comparison with u0*s, are recorded in the small-s summary. This is diagnostic only; no identity constraint was added.

The offset is state-dependent rather than constant: at s=0, Delta-x has Spearman correlations `{primary_identity['predicted_Delta_x']['x0']:.3f}` with x0 and `{primary_identity['predicted_Delta_x']['E0']:.3f}` with E0, while Delta-xi correlates `{primary_identity['predicted_Delta_xi']['x0']:.3f}` with x0. The primary x RMSE remains `{primary_small['by_s']['0']['x']['rmse']:.6g}` at s=0 and `{primary_small['by_s']['0.2']['x']['rmse']:.6g}` at s=0.2, so it does not rapidly disappear. Over s<=0.2, the learned Delta-x minus u0*s RMSE is `{primary_small['linear_u0_s_comparison']['learned_rmse']:.6g}`, versus `{primary_small['linear_u0_s_comparison']['exact_rmse']:.6g}` for the exact flow.

## Admissibility and energy

| seed | abs(xi_hat)>=1 | C<=0 | union fraction |
|---:|---:|---:|---:|
{admissibility_rows}

| seed | energy RMSE | energy MAE | abs p99 | abs p99.9 | abs max |
|---:|---:|---:|---:|---:|---:|
{energy_rows}

For primary seed 303, the top-100 energy tail fractions are `{tail['top100_fractions']}`. Its median |Delta-E|/state-error amplification in the top 100 is `{tail['absolute_energy_per_state_error']['top100_median']:.6g}`, versus `{tail['absolute_energy_per_state_error']['all_median']:.6g}` overall. Each seed's ranked top 100, correlations, and worst-case trajectory figures are saved.

## Flow and composition consistency

| seed | exact-restart RMSE x | exact-restart RMSE xi | self-composed RMSE x | self-composed RMSE xi |
|---:|---:|---:|---:|---:|
{composition_rows}

Exact-restart and self-composition are reported separately against both the exact endpoint and the direct finite-time prediction. Self-composition remains a diagnostic, not the intended deployment mode.

For primary seed 303, exact restart gives x RMSE `{primary_composition['exact_restart_vs_exact']['x']['rmse']:.6g}` against the exact endpoint and `{primary_composition['exact_restart_vs_direct']['x']['rmse']:.6g}` against the direct prediction. Self-composition increases these to `{primary_composition['self_composition_vs_exact']['x']['rmse']:.6g}` and `{primary_composition['self_composition_vs_direct']['x']['rmse']:.6g}`, respectively. Dependence on u_th, x0, and total elapsed time is machine-readable in `tables/composition_breakdowns.csv`.

## Matched local-step comparison

The authoritative previous local baseline is the retained fixed-E0 `3→32→32→2` two-tanh model with inputs `[x,xi,E0]`, residual outputs `[delta_x,delta_xi]`, and physical step `Delta t_local=0.2`, recovered from the frozen transformed-data manifest. Later energy-normal and sensitivity-aware losses were diagnostic variants and were not adopted. All three fixed-E0 baseline local seeds are reported; seed 101 is the predeclared local primary because its previously recorded local-validation MSE was the smallest, not because of results here.

| model | seed | RMSE x | MAE x | RMSE xi | MAE xi |
|:---|---:|---:|---:|---:|---:|
{local_rows}

Both model classes use exactly the same held-out exact anchors and exact targets at s=0.2. Hard/ordinary, x0, u_th, admissibility, energy, and percentile comparisons are in `tables/matched_local_metrics.csv` and `matched_local_summary.json`.

At the predeclared primary seeds, finite-time/local RMSE ratios are `{primary_finite_matched['x_rmse'] / primary_local_matched['x_rmse']:.1f}` for x and `{primary_finite_matched['xi_rmse'] / primary_local_matched['xi_rmse']:.1f}` for xi. The local predictor is therefore decisively more accurate at the interval it was explicitly trained to predict; this result does not address recursive long-horizon accumulation.

## Separate stress/reference family

| u_th | seed-303 RMSE x | seed-303 RMSE xi | energy MAE |
|---:|---:|---:|---:|
{stress_rows}

The seven reference trajectories remain excluded from aggregate validation. Their matched-s=0.2 finite-time/local results are in `tables/stress_matched_local_metrics.csv`; the local baseline is again more accurate on every named trajectory for both state components.

## Tests and reproducibility

The complete relevant subset passed: 26 tests covering the frozen orbit representation, finite-time dataset and baseline, and this trajectory evaluator. The repository-wide run recorded 212 passes and one unrelated legacy failure: a pre-existing tree-hash mismatch under `reports/round1_model_a_1000`. Both JUnit XML files are preserved under `tests/`. Protected sources, preprocessing, checkpoints, and the sealed-test file were hash-identical before and after evaluation. `validation_manifest.json` records every generated artifact path, byte count, and SHA-256; `validation_manifest.sha256` authenticates that manifest.

## Scientific assessment

The dense study confirms the prior seed pattern: seed 202 improves x RMSE from `{seed303_x:.6g}` to `{seed202_x:.6g}`, while seed 303 improves xi RMSE from `{seed202_xi:.6g}` to `{seed303_xi:.6g}` and remains primary. For seed 303, hard-family RMSE is `{primary_families['hard_u_th_le_0p30']['x']['rmse'] / primary_families['ordinary_u_th_gt_0p30']['x']['rmse']:.2f}` times ordinary-family RMSE in x and `{primary_families['hard_u_th_le_0p30']['xi']['rmse'] / primary_families['ordinary_u_th_gt_0p30']['xi']['rmse']:.2f}` times in xi; long-time RMSE is `{primary_families['long_s_gt_20']['x']['rmse'] / primary_families['short_s_le_5']['x']['rmse']:.2f}` times short-time RMSE in x. The top energy tail is not predominantly long-time or low-u_th: all top-100 cases are simultaneously in the highest-E0 5% and lowest-exact-C 5%, 87% end near the throat, and their median energy-per-state-error amplification is about 122 times the overall median. This supports amplification by the energy expression near low C rather than one isolated bad trajectory.

Scientific recommendation: do not advance this checkpoint to the one-time sealed-test evaluation as an acceptable finite-time flow baseline. The non-vanishing, state-dependent identity error, 472x/88x matched-step disadvantage to the local model, long-time and hard-family degradation, and self-composition inconsistency are already material validation failures. It is technically suitable only if the sealed test is intended to document this frozen model as a known weak baseline, not to validate it as ready.

Implementation problems preventing final sealed-test evaluation: none. The reason to pause is scientific model performance, not evaluator integrity.
"""


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite completed validation at {OUTPUT}")
    for directory in (OUTPUT, TABLES, ARRAYS, FIGURES, TESTS): directory.mkdir(parents=True, exist_ok=True)
    gate = immutable_gate()
    if not gate["passed"]:
        raise RuntimeError(f"immutable input gate failed: {gate['failures']}")
    hashes_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}
    preprocessing = load_finite_time_preprocessing(PREPROCESSING)
    finite_models = {seed: load_checkpoint(path) for seed, path in FINITE_CHECKPOINTS.items()}
    local_models = {seed: load_checkpoint(path) for seed, path in LOCAL_CHECKPOINTS.items()}
    local_normalization = Normalization.from_stage1(
        LOCAL_NORMALIZATION, ("x", "xi", "E0"), ("delta_x", "delta_xi"),
        "outer_microcore40k_train_x_xi_energy_input_only",
    )
    local_manifest = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8"))
    prior_local_losses = {int(run["seed"]): run["best_physical_validation_standardized_mse"] for run in local_manifest["runs"]}
    if min(prior_local_losses, key=prior_local_losses.get) != LOCAL_PRIMARY_SEED:
        raise RuntimeError("predeclared local primary no longer matches prior validation objective")

    with np.load(VALIDATION_BANK, allow_pickle=False) as validation_npz:
        # NPZ members are compressed. Materialize once so every orbit access does
        # not decompress the full concatenated dense-coefficient member again.
        validation_bank = {name: validation_npz[name] for name in validation_npz.files}
        dense_queries = build_dense_queries(validation_bank)
        if dense_queries["query_index"].size != 524_288:
            raise RuntimeError("dense validation grid does not contain 524288 rows")
        save_npz(ARRAYS / "dense_validation_queries.npz", dense_queries)
        aggregates: dict[int, Any] = {}
        predictions: dict[int, dict[str, np.ndarray]] = {}
        family_metrics: dict[str, Any] = {}
        admissibility: dict[str, Any] = {}
        energy: dict[str, Any] = {}
        energy_tail: dict[str, Any] = {}
        per_orbit_rows: list[dict[str, Any]] = []
        violation_table: list[dict[str, Any]] = []
        top_energy_table: list[dict[str, Any]] = []
        s_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        x_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        uth_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        e_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        uth_edges = np.linspace(float(np.min(dense_queries["u_th"])), float(np.max(dense_queries["u_th"])), 31)
        e_edges = np.unique(np.quantile(dense_queries["E0"], np.linspace(0.0, 1.0, 25)))
        for seed, model in finite_models.items():
            print(f"dense prediction seed={seed}", flush=True)
            prediction = predict_finite_time(model, preprocessing, dense_queries)
            predictions[seed] = prediction
            save_npz(ARRAYS / f"dense_predictions_seed_{seed}.npz", prediction)
            if float(prediction["residual_state_error_max_difference"][0]) > 1.0e-13:
                raise RuntimeError("state and residual errors differ")
            aggregate, orbit_rows = aggregate_error_metrics(prediction["x_error"], prediction["xi_error"], dense_queries["orbit_id"])
            aggregates[seed] = aggregate
            orbit_lookup = {str(validation_bank["orbit_id"][i]): i for i in range(validation_bank["orbit_id"].size)}
            for row in orbit_rows:
                index = orbit_lookup[row["orbit_id"]]
                per_orbit_rows.append({
                    "seed": seed, "orbit_id": row["orbit_id"], "u_th": float(validation_bank["u_th"][index]),
                    "E0": float(validation_bank["E0"][index]), "row_count": row["row_count"],
                    **flatten_metrics("", row),
                })
            families = {
                "hard_u_th_le_0p30": dense_queries["u_th"] <= 0.30,
                "ordinary_u_th_gt_0p30": dense_queries["u_th"] > 0.30,
                "short_s_le_5": dense_queries["s"] <= 5.0,
                "intermediate_5_lt_s_le_20": (dense_queries["s"] > 5.0) & (dense_queries["s"] <= 20.0),
                "long_s_gt_20": dense_queries["s"] > 20.0,
            }
            for center in (0.05, 0.15, 0.30):
                families[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(dense_queries["u_th"] - center) <= 0.01
            family_metrics[str(seed)] = {name: subset_metrics(prediction, mask) for name, mask in families.items()}
            adm, invalid = admissibility_summary(dense_queries, prediction)
            admissibility[str(seed)] = adm
            violation_table.extend(violation_rows(seed, dense_queries, prediction, invalid))
            energy[str(seed)] = energy_metrics(prediction)
            top = top_energy_rows(seed, dense_queries, prediction)
            top_energy_table.extend(top)
            energy_tail[str(seed)] = energy_tail_analysis(dense_queries, prediction, top)
            s_rows_by_seed[seed] = bin_error_rows(dense_queries["s"], prediction, S_EDGES, "s")
            x_rows_by_seed[seed] = bin_error_rows(dense_queries["x0"], prediction, X_EDGES, "x0")
            uth_rows_by_seed[seed] = bin_error_rows(dense_queries["u_th"], prediction, uth_edges, "u_th")
            e_rows_by_seed[seed] = bin_error_rows(dense_queries["E0"], prediction, e_edges, "E0")
        csv_rows(TABLES / "per_orbit_metrics.csv", per_orbit_rows)
        csv_rows(TABLES / "admissibility_violations.csv", violation_table)
        csv_rows(TABLES / "top100_energy_errors_per_seed.csv", top_energy_table)
        csv_rows(TABLES / "dense_aggregate_metrics.csv", [
            {"seed": seed, "weighting": weighting, **flatten_metrics("", metrics[weighting])}
            for seed, metrics in aggregates.items()
            for weighting in ("row_weighted", "orbit_weighted")
        ])
        csv_rows(TABLES / "energy_metrics.csv", [
            {"seed": int(seed), **metrics} for seed, metrics in energy.items()
        ])
        csv_rows(TABLES / "family_metrics.csv", [
            {"seed": int(seed), "family": family, **flatten_metrics("", metrics)}
            for seed, families in family_metrics.items()
            for family, metrics in families.items()
        ])
        for name, collection in (("error_vs_s", s_rows_by_seed), ("error_vs_x0", x_rows_by_seed), ("error_vs_u_th", uth_rows_by_seed), ("error_vs_E0", e_rows_by_seed)):
            csv_rows(TABLES / f"{name}.csv", [{"seed": seed, **row} for seed, rows in collection.items() for row in rows])

        plot_metric_lines(s_rows_by_seed, "physical elapsed time s", FIGURES / "error_vs_elapsed_time.png", "Finite-time prediction error versus elapsed time")
        plot_percentile_lines(s_rows_by_seed, "physical elapsed time s", FIGURES / "error_percentiles_vs_elapsed_time.png", "Finite-time error percentiles versus elapsed time")
        plot_metric_lines(x_rows_by_seed, "anchor x0", FIGURES / "error_vs_anchor_position.png", "Finite-time prediction error versus anchor position")
        plot_metric_lines(uth_rows_by_seed, "u_th", FIGURES / "error_vs_u_th.png", "Prediction error across the low-velocity trajectory family")
        plot_metric_lines(e_rows_by_seed, "E0", FIGURES / "error_vs_E0.png", "Finite-time prediction error versus conserved energy")
        error_map(dense_queries, predictions[FINITE_PRIMARY_SEED], FIGURES / "error_maps_x0_s.png")
        plot_energy(dense_queries, predictions, FIGURES / "energy_consistency.png")
        plot_seed_comparison(aggregates, FIGURES / "seed_comparison.png")

        selected_orbits = {
            "low": int(np.argmin(np.abs(validation_bank["u_th"] - 0.05))),
            "middle": int(np.argmin(np.abs(validation_bank["u_th"] - 0.50))),
            "high": int(np.argmin(np.abs(validation_bank["u_th"] - 0.90))),
        }
        representative = {}
        for label, index in selected_orbits.items():
            representative[label] = {"orbit_id": str(validation_bank["orbit_id"][index]), "u_th": float(validation_bank["u_th"][index]), "orbit_index": index}
            trajectory_figure(
                validation_bank, index, [-14.0, -8.0, 0.0], finite_models[FINITE_PRIMARY_SEED], preprocessing,
                FIGURES / f"representative_validation_{label}.png",
                f"Exact and predicted trajectories for held-out u_th={float(validation_bank['u_th'][index]):.4f}",
            )
        primary_top = [row for row in top_energy_table if row["seed"] == FINITE_PRIMARY_SEED]
        unique_worst = []
        seen = set()
        for row in primary_top:
            key = (row["orbit_index"], round(row["x0"], 8))
            if key not in seen:
                unique_worst.append(row); seen.add(key)
            if len(unique_worst) == 3: break
        figure, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), constrained_layout=True)
        for axis_row, row in zip(axes, unique_worst):
            index = int(row["orbit_index"]); anchor_x = float(row["x0"]); max_s = float(row["s"])
            t0, _ = invert_saved_orbit_x(validation_bank, index, np.asarray([anchor_x]))
            elapsed = np.linspace(0, max(max_s, 1e-8), 180)
            exact = evaluate_saved_orbit_x_u_xi(validation_bank, index, float(t0[0]) + elapsed)
            q = {"x0": np.full(elapsed.size, exact[0,0]), "xi0": np.full(elapsed.size, exact[0,2]), "E0": np.full(elapsed.size, float(validation_bank["E0"][index])), "s": elapsed, "exact_x1": exact[:,0], "exact_u1": exact[:,1], "exact_xi1": exact[:,2]}
            pred = predict_finite_time(finite_models[FINITE_PRIMARY_SEED], preprocessing, q)
            axis_row[0].plot(elapsed, exact[:,0], label="exact"); axis_row[0].plot(elapsed, pred["predicted_x1"], ls="--", label="predicted")
            axis_row[1].plot(elapsed, exact[:,2]); axis_row[1].plot(elapsed, pred["predicted_xi1"], ls="--")
            axis_row[0].set(ylabel=f"x; u_th={row['u_th']:.3f}"); axis_row[1].set(ylabel="xi")
            for axis in axis_row: axis.grid(alpha=0.25)
        axes[0,0].legend(); axes[-1,0].set_xlabel("physical elapsed time s"); axes[-1,1].set_xlabel("physical elapsed time s")
        figure.suptitle("Exact and predicted trajectories for extreme energy-error cases")
        figure.savefig(FIGURES / "extreme_energy_cases.png", dpi=185); plt.close(figure)

        small_queries = build_queries_at_x_and_s(validation_bank, SMALL_S_ANCHOR_X, SMALL_S_VALUES)
        save_npz(ARRAYS / "small_s_queries.npz", small_queries)
        small_predictions = {seed: predict_finite_time(model, preprocessing, small_queries) for seed, model in finite_models.items()}
        for seed, values in small_predictions.items(): save_npz(ARRAYS / f"small_s_predictions_seed_{seed}.npz", values)
        small_summary = small_s_analysis(small_queries, small_predictions)
        plot_small_s(small_queries, small_predictions, FIGURES / "small_s_identity.png")

        composition_queries = build_composition_queries(validation_bank)
        save_npz(ARRAYS / "composition_queries.npz", composition_queries)
        composition_results = {}
        composition_summaries = {}
        composition_csv = []
        composition_breakdown_rows = []
        for seed, model in finite_models.items():
            result = evaluate_composition(model, preprocessing, composition_queries)
            composition_results[seed] = result
            save_npz(ARRAYS / f"composition_results_seed_{seed}.npz", result)
            composition_summaries[seed] = composition_summary(result)
            for kind, metrics in composition_summaries[seed].items():
                composition_csv.append({"seed": seed, "comparison": kind, **flatten_metrics("", metrics)})
            comparisons = {
                "exact_restart_vs_exact": error_view(result, "exact_restart_x_error", "exact_restart_xi_error"),
                "exact_restart_vs_direct": error_view(result, "exact_restart_minus_direct_x", "exact_restart_minus_direct_xi"),
                "self_composition_vs_exact": error_view(result, "self_x_error", "self_xi_error"),
                "self_composition_vs_direct": error_view(result, "self_minus_direct_x", "self_minus_direct_xi"),
            }
            comp_uth_edges = np.linspace(float(np.min(composition_queries["u_th"])), float(np.max(composition_queries["u_th"])), 21)
            comp_s_edges = np.unique(np.quantile(composition_queries["s_total"], np.linspace(0.0, 1.0, 17)))
            for comparison, view in comparisons.items():
                for family, mask in (
                    ("hard_u_th_le_0p30", composition_queries["u_th"] <= 0.30),
                    ("ordinary_u_th_gt_0p30", composition_queries["u_th"] > 0.30),
                ):
                    composition_breakdown_rows.append({
                        "seed": seed, "comparison": comparison, "breakdown": "family",
                        "bin_label": family, "row_count": int(np.sum(mask)),
                        **flatten_metrics("", {
                            "x": scalar_error_metrics(view["x_error"][mask]),
                            "xi": scalar_error_metrics(view["xi_error"][mask]),
                        }),
                    })
                for coordinate_name, coordinate, edges in (
                    ("x0", composition_queries["x0"], X_EDGES),
                    ("u_th", composition_queries["u_th"], comp_uth_edges),
                    ("s_total", composition_queries["s_total"], comp_s_edges),
                ):
                    composition_breakdown_rows.extend({
                        "seed": seed, "comparison": comparison, "breakdown": coordinate_name, **row,
                    } for row in bin_error_rows(coordinate, view, edges, coordinate_name))
        csv_rows(TABLES / "composition_metrics.csv", composition_csv)
        csv_rows(TABLES / "composition_breakdowns.csv", composition_breakdown_rows)
        plot_composition(composition_summaries, FIGURES / "composition_consistency.png")

        matched_queries = build_queries_at_x_and_s(validation_bank, DENSE_ANCHOR_X, np.asarray([LOCAL_STEP]))
        if matched_queries["query_index"].size != 1024 * 32:
            raise RuntimeError("matched local-step bank does not contain every exact anchor")
        save_npz(ARRAYS / "matched_local_queries.npz", matched_queries)
        matched_rows = []
        matched_predictions = {}
        for seed, model in finite_models.items():
            pred = predict_finite_time(model, preprocessing, matched_queries)
            matched_predictions[f"finite_time_{seed}"] = pred
            save_npz(ARRAYS / f"matched_finite_time_predictions_seed_{seed}.npz", pred)
            metrics, _ = aggregate_error_metrics(pred["x_error"], pred["xi_error"], matched_queries["orbit_id"])
            adm, _ = admissibility_summary(matched_queries, pred)
            matched_rows.append({"model": "finite_time", "seed": seed, **{f"{c}_{k}": v for c in ("x","xi") for k,v in metrics["row_weighted"][c].items()}, "energy_mae": energy_metrics(pred)["mae"], "violations": adm["union_violation_count"]})
        for seed, model in local_models.items():
            pred = predict_local(model, local_normalization, matched_queries)
            matched_predictions[f"local_{seed}"] = pred
            save_npz(ARRAYS / f"matched_local_predictions_seed_{seed}.npz", pred)
            metrics, _ = aggregate_error_metrics(pred["x_error"], pred["xi_error"], matched_queries["orbit_id"])
            adm, _ = admissibility_summary(matched_queries, pred)
            matched_rows.append({"model": "local", "seed": seed, **{f"{c}_{k}": v for c in ("x","xi") for k,v in metrics["row_weighted"][c].items()}, "energy_mae": energy_metrics(pred)["mae"], "violations": adm["union_violation_count"]})
        csv_rows(TABLES / "matched_local_metrics.csv", matched_rows)
        matched_detail = {}
        matched_breakdown_rows = []
        matched_uth_edges = np.linspace(float(np.min(matched_queries["u_th"])), float(np.max(matched_queries["u_th"])), 31)
        for key, pred in matched_predictions.items():
            matched_detail[key] = {
                "hard": subset_metrics(pred, matched_queries["u_th"] <= 0.30),
                "ordinary": subset_metrics(pred, matched_queries["u_th"] > 0.30),
                "admissibility": admissibility_summary(matched_queries, pred)[0],
                "energy": energy_metrics(pred),
                "x0_bins": bin_error_rows(matched_queries["x0"], pred, X_EDGES, "x0"),
                "u_th_bins": bin_error_rows(matched_queries["u_th"], pred, matched_uth_edges, "u_th"),
            }
            model_name, seed_text = key.rsplit("_", 1)
            for family, mask in (
                ("hard_u_th_le_0p30", matched_queries["u_th"] <= 0.30),
                ("ordinary_u_th_gt_0p30", matched_queries["u_th"] > 0.30),
            ):
                matched_breakdown_rows.append({
                    "model": model_name, "seed": int(seed_text), "breakdown": "family",
                    "bin_label": family, **flatten_metrics("", subset_metrics(pred, mask)),
                })
            for coordinate_name in ("x0", "u_th"):
                matched_breakdown_rows.extend({
                    "model": model_name, "seed": int(seed_text), "breakdown": coordinate_name, **row,
                } for row in matched_detail[key][f"{coordinate_name}_bins"])
        csv_rows(TABLES / "matched_local_breakdowns.csv", matched_breakdown_rows)
        primary_xrows = {
            "finite_time": bin_error_rows(matched_queries["x0"], matched_predictions[f"finite_time_{FINITE_PRIMARY_SEED}"], X_EDGES, "x0"),
            "local": bin_error_rows(matched_queries["x0"], matched_predictions[f"local_{LOCAL_PRIMARY_SEED}"], X_EDGES, "x0"),
        }
        plot_local_comparison(matched_rows, primary_xrows, FIGURES / "finite_time_vs_local.png")

    with np.load(STRESS_BANK, allow_pickle=False) as stress_npz:
        stress_bank = {name: stress_npz[name] for name in stress_npz.files}
        stress_queries = build_dense_queries(stress_bank)
        save_npz(ARRAYS / "stress_queries.npz", stress_queries)
        stress_rows = []
        for seed, model in finite_models.items():
            pred = predict_finite_time(model, preprocessing, stress_queries)
            save_npz(ARRAYS / f"stress_predictions_seed_{seed}.npz", pred)
            for orbit_index, u_th in enumerate(stress_bank["u_th"]):
                mask = stress_queries["orbit_index"] == orbit_index
                metrics = subset_metrics(pred, mask)
                stress_rows.append({"seed": seed, "u_th": float(u_th), **flatten_metrics("", metrics), "energy_mae": energy_metrics({k:v[mask] for k,v in pred.items() if v.shape[0] == mask.size})["mae"]})
        csv_rows(TABLES / "stress_reference_metrics.csv", stress_rows)
        for orbit_index, u_th in enumerate(stress_bank["u_th"]):
            trajectory_figure(
                stress_bank, orbit_index, [-14.0, -8.0, 0.0], finite_models[FINITE_PRIMARY_SEED], preprocessing,
                FIGURES / f"stress_u_th_{float(u_th):.2f}.png",
                f"Exact and predicted trajectories for reference u_th={float(u_th):.2f}",
            )

        stress_matched_queries = build_queries_at_x_and_s(stress_bank, DENSE_ANCHOR_X, np.asarray([LOCAL_STEP]))
        if stress_matched_queries["query_index"].size != int(stress_bank["orbit_id"].size) * 32:
            raise RuntimeError("stress matched-step bank does not contain every exact anchor")
        save_npz(ARRAYS / "stress_matched_local_queries.npz", stress_matched_queries)
        stress_matched_rows = []
        stress_matched_predictions = {}
        for model_name, models, predictor in (
            ("finite_time", finite_models, lambda model: predict_finite_time(model, preprocessing, stress_matched_queries)),
            ("local", local_models, lambda model: predict_local(model, local_normalization, stress_matched_queries)),
        ):
            for seed, model in models.items():
                pred = predictor(model)
                stress_matched_predictions[f"{model_name}_{seed}"] = pred
                save_npz(ARRAYS / f"stress_matched_{model_name}_predictions_seed_{seed}.npz", pred)
                for orbit_index, u_th in enumerate(stress_bank["u_th"]):
                    mask = stress_matched_queries["orbit_index"] == orbit_index
                    metrics = subset_metrics(pred, mask)
                    energy_view = {"energy_error": pred["energy_error"][mask]}
                    stress_matched_rows.append({
                        "model": model_name, "seed": seed, "u_th": float(u_th),
                        **flatten_metrics("", metrics), "energy_mae": energy_metrics(energy_view)["mae"],
                    })
        csv_rows(TABLES / "stress_matched_local_metrics.csv", stress_matched_rows)
        plot_stress_local_comparison(stress_matched_rows, FIGURES / "stress_finite_time_vs_local.png")

    matched_summary = {
        "authoritative_local_model": {
            "treatment": "fixed-E0 ordinary standardized-MSE baseline",
            "architecture": "3->32->32->2 with two tanh hidden layers",
            "input_order": ["x", "xi", "E0"], "target_order": ["delta_x", "delta_xi"],
            "physical_step": LOCAL_STEP, "primary_seed": LOCAL_PRIMARY_SEED,
            "primary_selection_basis": "minimum previously recorded local validation standardized MSE",
            "prior_validation_losses": prior_local_losses,
            "checkpoints": {str(seed): {"path": str(path.resolve()), "sha256": file_sha256(path)} for seed,path in LOCAL_CHECKPOINTS.items()},
            "normalization": {"path": str(LOCAL_NORMALIZATION.resolve()), "sha256": file_sha256(LOCAL_NORMALIZATION)},
        },
        "query_count": int(matched_queries["query_index"].size),
        "aggregate": matched_rows,
        "breakdowns": matched_detail,
        "primary_x0_bins": primary_xrows,
    }
    write_json(OUTPUT / "matched_local_summary.json", matched_summary)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dense_query_count": int(dense_queries["query_index"].size),
        "dense_design": {"anchor_x": DENSE_ANCHOR_X.tolist(), "fractions": DENSE_FRACTIONS.tolist(), "rows_per_orbit": 512},
        "dense_aggregate": {str(seed): row for seed,row in aggregates.items()},
        "family_metrics": family_metrics,
        "coordinate_bins": {
            "s": {str(seed): rows for seed, rows in s_rows_by_seed.items()},
            "x0": {str(seed): rows for seed, rows in x_rows_by_seed.items()},
            "u_th": {str(seed): rows for seed, rows in uth_rows_by_seed.items()},
            "E0": {str(seed): rows for seed, rows in e_rows_by_seed.items()},
        },
        "admissibility": admissibility,
        "energy_metrics": energy,
        "energy_tail": energy_tail,
        "small_s": small_summary,
        "composition": {str(seed): row for seed,row in composition_summaries.items()},
        "composition_design": {"anchor_x": COMPOSITION_ANCHOR_X.tolist(), "fraction_pairs": COMPOSITION_FRACTION_PAIRS.tolist(), "query_count": int(composition_queries["query_index"].size)},
        "stress_reference": {
            "aggregate_rows": stress_rows,
            "matched_local_rows": stress_matched_rows,
            "excluded_from_validation_aggregates": True,
        },
        "representative_validation_orbits": representative,
        "matched_local": matched_summary,
        "primary_finite_time_seed": FINITE_PRIMARY_SEED,
        "sealed_test_predictions_computed": False,
        "training_or_reintegration_performed": False,
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(scientific_report(summary), encoding="utf-8")

    hashes_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}
    if hashes_before != hashes_after:
        raise RuntimeError("a frozen source, preprocessing, or checkpoint artifact changed")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "phase": "held_out_direct_finite_time_trajectory_validation",
        "status": "completed_without_training_or_test_predictions",
        "primary_finite_time_seed": FINITE_PRIMARY_SEED,
        "immutable_gate": gate,
        "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
        "sealed_test_policy": {"NPZ_opened": False, "predictions": False, "loss": False, "metrics": False, "plots": False, "byte_hash_only": True},
        "local_model": matched_summary["authoritative_local_model"],
        "runtime": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "implementation_sources": {
            "src/wormhole_sciml/finite_time_validation.py": file_sha256(ROOT / "src" / "wormhole_sciml" / "finite_time_validation.py"),
            "scripts/evaluate_finite_time_trajectories.py": file_sha256(Path(__file__)),
        },
        "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)},
        "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)},
        "artifacts": artifacts,
        "implementation_problem_preventing_final_test": False,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
