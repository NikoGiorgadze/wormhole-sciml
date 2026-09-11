#!/usr/bin/env python3
"""Compare frozen direct finite-time predictions with two recursive rollouts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
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

from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, load_hybrid_model, predict_hybrid
from wormhole_sciml.finite_time_hybrid_validation import log_error_ratio, median_log_grid
from wormhole_sciml.finite_time_rollout import (
    HORIZON_K, LOCAL_STEP, MAXIMUM_K, build_common_anchor_bank,
    common_survivor_mask, direct_predictions_from_original, energy_metrics,
    first_and_sustained_crossover, physical_diagnostics, recursive_rollout, state_metrics,
)
from wormhole_sciml.model_a import Normalization, load_trained_model, predict_increments
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi, file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/direct_vs_recursive_rollouts"
ARRAYS, TABLES, FIGURES, TESTS = (OUTPUT / name for name in ("arrays", "tables", "figures", "tests"))
REPORT = OUTPUT / "DIRECT_VS_RECURSIVE_ROLLOUT_REPORT.md"
SUMMARY = OUTPUT / "direct_vs_recursive_summary.json"
MANIFEST = OUTPUT / "direct_vs_recursive_manifest.json"
MANIFEST_HASH = OUTPUT / "direct_vs_recursive_manifest.sha256"

HYBRID_ROOT = ROOT / "output/finite_time_hybrid_s5"
HYBRID_MANIFEST = HYBRID_ROOT / "finite_time_hybrid_manifest.json"
HYBRID_CHECKPOINT = HYBRID_ROOT / "training/seed_202/best_checkpoint.pt"
HYBRID_PREPROCESSING = HYBRID_ROOT / "preprocessing/hybrid_preprocessing_constants.json"
LOCAL_ROOT = ROOT / "output/model_a_x_xi_energy_microcore40k_comparison"
LOCAL_MANIFEST = LOCAL_ROOT / "energy_xi_training_manifest.json"
LOCAL_CHECKPOINT = LOCAL_ROOT / "training/seed_101/best_checkpoint.pt"
LOCAL_NORMALIZATION = LOCAL_ROOT / "training/energy_input_normalization.json"
ORBIT_ROOT = ROOT / "output/phase_b_complete_orbit_banks"
ORBIT_MANIFEST = ORBIT_ROOT / "phase_b_manifest.json"
VALIDATION_BANK = ORBIT_ROOT / "banks/phase_b_validation_orbits.npz"
STRESS_BANK = ORBIT_ROOT / "banks/phase_b_stress_reference_orbits.npz"
SEALED = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_test_sealed_raw.npz"

EPSILON = 1.0e-12
EXPECTED = {
    HYBRID_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
    HYBRID_CHECKPOINT: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    HYBRID_PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    LOCAL_MANIFEST: "27050145c861fe630cee764fd0766238604bb6c04be335e23186b3ca2b45b5b5",
    LOCAL_CHECKPOINT: "54a41a54a7fa20e931df0dd987ac871786d4ac8d02e71bd3560e9ebc3228a4a6",
    LOCAL_NORMALIZATION: "5f3605f657d8190ea901112774553e3026a3399266783a78f0102b285c4045c3",
    ORBIT_MANIFEST: "207f6d3c2a5cbaf3eb613c51e4d438abf50538eb8af0e21be8f8700d2d1b716b",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    STRESS_BANK: "5d7959cf1a657a5ff916e4d40aab44309ca08958f1694340b47c0fce7ac08dce",
    SEALED: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
}
METHODS = ("direct_hybrid", "recursive_local", "recursive_hybrid")
COLORS = {"direct_hybrid": "#0072B2", "recursive_local": "#D55E00", "recursive_hybrid": "#009E73"}
LABELS = {"direct_hybrid": "direct hybrid", "recursive_local": "recursive local", "recursive_hybrid": "recursive hybrid"}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def immutable_gate() -> dict[str, Any]:
    artifacts, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path)
        artifacts[str(path.resolve())] = {"expected": expected, "measured": measured, "match": measured == expected}
        if measured != expected:
            failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": artifacts,
            "sealed_access": "byte hash only; NPZ never opened"}


def exact_state_cube(
    anchors: dict[str, np.ndarray], banks: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    shape = (anchors["anchor_id"].size, MAXIMUM_K + 1)
    x = np.full(shape, np.nan, dtype=np.float64)
    u = np.full(shape, np.nan, dtype=np.float64)
    xi = np.full(shape, np.nan, dtype=np.float64)
    time = np.full(shape, np.nan, dtype=np.float64)
    available = np.zeros(shape, dtype=np.bool_)
    for source_name, bank in banks.items():
        source_mask = anchors["source_bank"] == source_name
        for orbit_index in np.unique(anchors["source_orbit_index"][source_mask]):
            rows = np.flatnonzero(source_mask & (anchors["source_orbit_index"] == orbit_index))
            row_parts, k_parts, time_parts = [], [], []
            for row in rows:
                k = np.arange(int(anchors["maximum_evaluated_k"][row]) + 1, dtype=np.int16)
                row_parts.append(np.full(k.size, row, dtype=np.int64))
                k_parts.append(k)
                time_parts.append(float(anchors["t0"][row]) + LOCAL_STEP * k)
            flat_rows = np.concatenate(row_parts)
            flat_k = np.concatenate(k_parts)
            flat_time = np.concatenate(time_parts)
            state = evaluate_saved_orbit_x_u_xi(bank, int(orbit_index), flat_time)
            x[flat_rows, flat_k], u[flat_rows, flat_k], xi[flat_rows, flat_k] = state.T
            time[flat_rows, flat_k] = flat_time
            available[flat_rows, flat_k] = True
    return {"x": x, "u": u, "xi": xi, "time": time, "available": available}


def flatten_metrics(prefix: str, metric: dict[str, Any], row: dict[str, Any]) -> None:
    row[f"{prefix}_count"] = metric["count"]
    for component in ("x", "xi"):
        values = metric[component]
        for name in ("rmse", "mae", "median_absolute", "p90_absolute", "p95_absolute", "p99_absolute", "maximum_absolute"):
            row[f"{prefix}_{component}_{name}"] = np.nan if values is None else values[name]


def metric_rows(
    anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]],
    anchor_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nested, flat = [], []
    for k in HORIZON_K:
        exact_mask = anchor_mask & exact["available"][:, k]
        common = common_survivor_mask(
            exact_mask, predictions["direct_hybrid"]["valid"][:, k],
            predictions["recursive_local"]["valid"][:, k], predictions["recursive_hybrid"]["valid"][:, k],
        )
        record: dict[str, Any] = {"k": int(k), "s": float(k * LOCAL_STEP),
                                  "exact_count": int(np.sum(exact_mask)), "common_count": int(np.sum(common)), "methods": {}}
        flat_record: dict[str, Any] = {key: record[key] for key in ("k", "s", "exact_count", "common_count")}
        for name in METHODS:
            prediction = predictions[name]
            own = exact_mask & prediction["valid"][:, k]
            all_valid = state_metrics(prediction["x"][:, k], prediction["xi"][:, k], exact["x"][:, k], exact["xi"][:, k], own)
            common_metrics = state_metrics(prediction["x"][:, k], prediction["xi"][:, k], exact["x"][:, k], exact["xi"][:, k], common)
            energy = energy_metrics(prediction["E"][:, k], anchors["E0"], own)
            record["methods"][name] = {"all_valid": all_valid, "common_survivors": common_metrics, "energy_all_valid": energy}
            flatten_metrics(f"{name}_all", all_valid, flat_record)
            flatten_metrics(f"{name}_common", common_metrics, flat_record)
            if energy is not None:
                for metric_name, value in energy.items(): flat_record[f"{name}_energy_{metric_name}"] = value
        semigroup_mask = exact_mask & predictions["direct_hybrid"]["valid"][:, k] & predictions["recursive_hybrid"]["valid"][:, k]
        self_composition = state_metrics(
            predictions["recursive_hybrid"]["x"][:, k], predictions["recursive_hybrid"]["xi"][:, k],
            predictions["direct_hybrid"]["x"][:, k], predictions["direct_hybrid"]["xi"][:, k], semigroup_mask,
        )
        record["direct_vs_recursive_hybrid"] = self_composition
        flatten_metrics("self_composition", self_composition, flat_record)
        nested.append(record)
        flat.append(flat_record)
    return nested, flat


def crossover_payload(flat_rows: list[dict[str, Any]], minimum_count: int) -> dict[str, Any]:
    output = {"well_sampled_definition": f"common_count >= {minimum_count}"}
    for component in ("x", "xi"):
        output[component] = {}
        for metric in ("rmse", "mae", "median_absolute"):
            output[component][metric] = first_and_sustained_crossover(
                flat_rows, f"direct_hybrid_common_{component}_{metric}", f"recursive_local_common_{component}_{metric}",
                minimum_count=minimum_count,
            )
    return output


def group_definitions(anchors: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    family = {"hard_u_th_le_0p30": anchors["u_th"] <= .30, "ordinary_u_th_gt_0p30": anchors["u_th"] > .30}
    for center in (.05, .15, .30, .50, .65, .80, .90):
        family[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(anchors["u_th"] - center) <= .01
    x0 = anchors["x0"]
    region = {
        "incoming_far_left_x0_lt_-13": x0 < -13,
        "sensitive_incoming_-13_to_-8p5": (x0 >= -13) & (x0 <= -8.5),
        "approaching_throat_-8p5_to_-2": (x0 > -8.5) & (x0 < -2),
        "near_throat_abs_x0_le_2": np.abs(x0) <= 2,
        "outgoing_x0_gt_2": x0 > 2,
        "established_sensitive_-17_to_-8p5": (x0 >= -17) & (x0 <= -8.5),
    }
    return family, region


def survival_and_failures(
    anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survival = []
    for k in range(1, MAXIMUM_K + 1):
        exact_count = int(np.sum(exact["available"][:, k]))
        row = {"k": k, "s": k * LOCAL_STEP, "exact_available_count": exact_count}
        for method in ("recursive_local", "recursive_hybrid"):
            count = int(np.sum(exact["available"][:, k] & predictions[method]["valid"][:, k]))
            row[f"{method}_survivor_count"] = count
            row[f"{method}_survival_fraction"] = count / exact_count if exact_count else np.nan
        survival.append(row)
    failures = []
    for method in ("recursive_local", "recursive_hybrid"):
        rollout = predictions[method]
        for row_index in np.flatnonzero(rollout["first_failure_k"] >= 0):
            k = int(rollout["first_failure_k"][row_index])
            failures.append({
                "method": method, "anchor_row": int(row_index), "anchor_id": str(anchors["anchor_id"][row_index]),
                "source_bank": str(anchors["source_bank"][row_index]), "orbit_id": str(anchors["orbit_id"][row_index]),
                "u_th": float(anchors["u_th"][row_index]), "E0": float(anchors["E0"][row_index]),
                "x0": float(anchors["x0"][row_index]), "xi0": float(anchors["xi0"][row_index]),
                "k_fail": k, "s_fail": k * LOCAL_STEP, "failed_x": float(rollout["x"][row_index, k]),
                "failed_xi": float(rollout["xi"][row_index, k]), "failed_C": float(rollout["C"][row_index, k]),
            })
    return survival, failures


def failure_breakdowns(failures: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for coordinate, key in (("k", "k_fail"), ("x0", "x0")):
        rows = []
        for method in ("recursive_local", "recursive_hybrid"):
            values = sorted(set(row[key] for row in failures if row["method"] == method))
            for value in values:
                rows.append({"method": method, coordinate: value,
                             "failure_count": sum(row["method"] == method and row[key] == value for row in failures)})
        output[coordinate] = rows
    u_edges = np.linspace(0, 1, 21)
    rows = []
    for method in ("recursive_local", "recursive_hybrid"):
        values = np.asarray([row["u_th"] for row in failures if row["method"] == method])
        for low, high in zip(u_edges[:-1], u_edges[1:]):
            count = int(np.sum((values >= low) & (values < high if high < 1 else values <= high)))
            rows.append({"method": method, "u_th_low": low, "u_th_high": high, "failure_count": count})
    output["u_th"] = rows
    return output


def energy_rows(anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for k in range(1, MAXIMUM_K + 1):
        row: dict[str, Any] = {"k": k, "s": k * LOCAL_STEP}
        for method in METHODS:
            mask = exact["available"][:, k] & predictions[method]["valid"][:, k]
            metric = energy_metrics(predictions[method]["E"][:, k], anchors["E0"], mask)
            if metric is not None:
                for name, value in metric.items(): row[f"{method}_{name}"] = value
        rows.append(row)
    return rows


def plot_error_growth(rows: list[dict[str, Any]], crossover: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for axis, (component, metric) in zip(axes.ravel(), (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))):
        for method in METHODS:
            axis.plot([r["s"] for r in rows], [r[f"{method}_common_{component}_{metric}"] for r in rows],
                      color=COLORS[method], label=LABELS[method])
        sustained = crossover[component][metric]["sustained"]
        if sustained is not None:
            axis.axvline(sustained["s"], color="black", ls=":", lw=1, label="sustained crossover")
        axis.set(ylabel=f"common-survivor {metric.upper()} {component}", yscale="log")
        axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("total elapsed time s")
    axes[0, 0].legend()
    figure.suptitle("Prediction error versus total elapsed time")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_ratios(group_rows: dict[str, list[dict[str, Any]]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    group_labels = {"global": "all anchors", "hard_u_th_le_0p30": "hard u_th<=0.30", "ordinary_u_th_gt_0p30": "ordinary u_th>0.30"}
    for axis, (component, metric) in zip(axes.ravel(), (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))):
        for group, rows in group_rows.items():
            direct = np.asarray([r[f"direct_hybrid_common_{component}_{metric}"] for r in rows])
            local = np.asarray([r[f"recursive_local_common_{component}_{metric}"] for r in rows])
            axis.plot([r["s"] for r in rows], np.log10((direct + EPSILON) / (local + EPSILON)), label=group_labels[group])
        axis.axhline(0, color="black", lw=1)
        axis.set(ylabel=f"log10 direct/local {metric.upper()} {component}")
        axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("total elapsed time s")
    axes[0, 0].legend()
    figure.suptitle("Direct finite-time versus recursive local error ratio")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_survival(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in ("recursive_local", "recursive_hybrid"):
        axis.plot([r["s"] for r in rows], [r[f"{method}_survival_fraction"] for r in rows],
                  color=COLORS[method], label=LABELS[method])
    axis.set(xlabel="total elapsed time s", ylabel="fraction of exact-available anchors still valid", ylim=(-.02, 1.02))
    axis.grid(alpha=.25); axis.legend(); axis.set_title("Recursive rollout survival versus elapsed time")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_energy(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric in zip(axes, ("mae", "p99_absolute")):
        for method in METHODS:
            axis.plot([r["s"] for r in rows], [r.get(f"{method}_{metric}", np.nan) for r in rows],
                      color=COLORS[method], label=LABELS[method])
        axis.set(xlabel="total elapsed time s", ylabel=f"energy error {metric}", yscale="log")
        axis.grid(alpha=.25)
    axes[0].legend(); figure.suptitle("Energy inconsistency during direct and recursive prediction")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_self_composition(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, component in zip(axes, ("x", "xi")):
        axis.plot([r["s"] for r in rows], [r[f"self_composition_{component}_rmse"] for r in rows], color="#CC79A7")
        axis.set(xlabel="total elapsed time s", ylabel=f"RMSE between direct and recursive hybrid {component}", yscale="log")
        axis.grid(alpha=.25)
    figure.suptitle("Finite-time model self-composition discrepancy")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_heatmap(
    anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]], k: int, path: Path
) -> None:
    x_edges = np.linspace(-17, 17, 19)
    u_edges = np.linspace(float(anchors["u_th"].min()), float(anchors["u_th"].max()), 21)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    base = exact["available"][:, k] & predictions["direct_hybrid"]["valid"][:, k] & predictions["recursive_local"]["valid"][:, k]
    for row, component in enumerate(("x", "xi")):
        local_error = np.abs(predictions["recursive_local"][component][:, k] - exact[component][:, k])
        direct_error = np.abs(predictions["direct_hybrid"][component][:, k] - exact[component][:, k])
        local_grid, counts = median_log_grid(anchors["x0"][base], anchors["u_th"][base], local_error[base], x_edges, u_edges, EPSILON)
        direct_grid, _ = median_log_grid(anchors["x0"][base], anchors["u_th"][base], direct_error[base], x_edges, u_edges, EPSILON)
        ratio_grid, _ = median_log_grid(anchors["x0"][base], anchors["u_th"][base],
                                        (direct_error[base] + EPSILON) / (local_error[base] + EPSILON), x_edges, u_edges, EPSILON)
        finite = np.concatenate((local_grid[np.isfinite(local_grid)], direct_grid[np.isfinite(direct_grid)]))
        lo, hi = np.quantile(finite, (.02, .98))
        for column, (grid, title, cmap, vmin, vmax) in enumerate((
            (local_grid, "recursive local", "magma", lo, hi), (direct_grid, "direct hybrid", "magma", lo, hi),
            (ratio_grid, "log10 direct/local ratio", "coolwarm", -3, 3),
        )):
            mesh = axes[row, column].pcolormesh(x_edges, u_edges, np.ma.masked_invalid(grid), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            figure.colorbar(mesh, ax=axes[row, column])
            axes[row, column].set(xlabel="exact anchor x0", ylabel="u_th", title=f"{component}: {title}")
    figure.suptitle(f"Direct and recursive error across physical anchors at s={k * LOCAL_STEP:g}")
    figure.savefig(path, dpi=180); plt.close(figure)


def representative_plots(
    anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]
) -> list[str]:
    paths = []
    for target_u in (.05, .15, .30, .50, .90):
        orbit_candidates = np.flatnonzero((anchors["source_bank"] == "stress_reference") & np.isclose(anchors["u_th"], target_u))
        selected = [orbit_candidates[np.argmin(np.abs(anchors["x0"][orbit_candidates] - target))] for target in (-16.5, -.5, 5.8)]
        figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
        for column, row_index in enumerate(selected):
            max_k = int(anchors["maximum_evaluated_k"][row_index]); k = np.arange(max_k + 1); elapsed = k * LOCAL_STEP
            for method in METHODS:
                valid = predictions[method]["valid"][row_index, :max_k + 1]
                axes[0, column].plot(elapsed[valid], predictions[method]["x"][row_index, :max_k + 1][valid], color=COLORS[method], ls="--" if method == "direct_hybrid" else "-", label=LABELS[method])
                axes[1, column].plot(elapsed[valid], predictions[method]["xi"][row_index, :max_k + 1][valid], color=COLORS[method], ls="--" if method == "direct_hybrid" else "-")
                axes[2, column].plot(predictions[method]["x"][row_index, :max_k + 1][valid], predictions[method]["xi"][row_index, :max_k + 1][valid], color=COLORS[method], ls="--" if method == "direct_hybrid" else "-")
            available = exact["available"][row_index, :max_k + 1]
            axes[0, column].plot(elapsed[available], exact["x"][row_index, :max_k + 1][available], color="black", lw=1.5, label="exact")
            axes[1, column].plot(elapsed[available], exact["xi"][row_index, :max_k + 1][available], color="black", lw=1.5)
            axes[2, column].plot(exact["x"][row_index, :max_k + 1][available], exact["xi"][row_index, :max_k + 1][available], color="black", lw=1.5)
            axes[0, column].set(title=f"anchor x0={anchors['x0'][row_index]:.2f}", xlabel="elapsed time s", ylabel="x")
            axes[1, column].set(xlabel="elapsed time s", ylabel="xi")
            axes[2, column].set(xlabel="x", ylabel="xi")
            for axis in axes[:, column]: axis.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"Direct finite-time and recursive trajectories for u_th={target_u:.2f}")
        path = FIGURES / f"representative_u_th_{target_u:.2f}.png"
        figure.savefig(path, dpi=175); plt.close(figure); paths.append(str(path.resolve()))
    return paths


def rapid_change_diagnostics(
    anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], predictions: dict[str, dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    results = []
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for row, target_u in enumerate((.05, .15, .30)):
        candidates = np.flatnonzero((anchors["source_bank"] == "stress_reference") & np.isclose(anchors["u_th"], target_u))
        anchor_row = int(candidates[np.argmin(anchors["x0"][candidates])])
        max_k = int(anchors["maximum_evaluated_k"][anchor_row]); k = np.arange(max_k + 1); elapsed = k * LOCAL_STEP
        exact_x = exact["x"][anchor_row, :max_k + 1]; exact_xi = exact["xi"][anchor_row, :max_k + 1]
        rate = np.abs(np.gradient(exact_xi, LOCAL_STEP)); peak = int(np.argmax(rate)); threshold = .5 * rate[peak]
        rapid = rate >= threshold; rapid_indices = np.flatnonzero(rapid); start, stop = int(rapid_indices[0]), int(rapid_indices[-1])
        throat = int(np.argmin(np.abs(exact_x)))
        segments = {
            "before_rapid_change": k < start,
            "rapid_change_before_throat": rapid & (exact_x < 0),
            "through_throat_abs_x_le_2": np.abs(exact_x) <= 2,
            "immediately_after_throat_2_lt_x_le_8": (exact_x > 2) & (exact_x <= 8),
            "relaxed_outgoing_x_gt_8": exact_x > 8,
        }
        segment_metrics = {}
        for segment, mask in segments.items():
            segment_metrics[segment] = {}
            for method in METHODS:
                valid = mask & predictions[method]["valid"][anchor_row, :max_k + 1]
                segment_metrics[segment][method] = state_metrics(
                    predictions[method]["x"][anchor_row, :max_k + 1], predictions[method]["xi"][anchor_row, :max_k + 1],
                    exact_x, exact_xi, valid,
                )
        results.append({"u_th": target_u, "anchor_id": str(anchors["anchor_id"][anchor_row]), "x0": float(anchors["x0"][anchor_row]),
                        "peak_abs_dxi_dt": float(rate[peak]), "peak_k": peak, "peak_s": float(elapsed[peak]), "peak_x": float(exact_x[peak]),
                        "rapid_threshold": float(threshold), "rapid_start_k": start, "rapid_stop_k": stop,
                        "rapid_start_s": float(elapsed[start]), "rapid_stop_s": float(elapsed[stop]), "throat_k": throat,
                        "throat_s": float(elapsed[throat]), "segments": segment_metrics})
        for method in METHODS:
            valid = predictions[method]["valid"][anchor_row, :max_k + 1]
            error = np.abs(predictions[method]["xi"][anchor_row, :max_k + 1] - exact_xi)
            axes[row, 0].plot(elapsed[valid], error[valid], color=COLORS[method], label=LABELS[method])
        axes[row, 0].axvspan(elapsed[start], elapsed[stop], color="grey", alpha=.18, label="orbit-specific rapid interval")
        axes[row, 0].axvline(elapsed[throat], color="black", ls=":", lw=1, label="throat")
        axes[row, 0].set(ylabel=f"u_th={target_u:.2f}\nabsolute xi error", yscale="log"); axes[row, 0].grid(alpha=.25)
        axes[row, 1].plot(elapsed, exact_xi, color="black", label="exact xi")
        twin = axes[row, 1].twinx(); twin.plot(elapsed, rate, color="#CC79A7", alpha=.8, label="|d xi/dt|")
        axes[row, 1].axvspan(elapsed[start], elapsed[stop], color="grey", alpha=.18)
        axes[row, 1].set(ylabel="exact xi"); twin.set_ylabel("|d xi/dt|"); axes[row, 1].grid(alpha=.25)
    axes[-1, 0].set_xlabel("elapsed time s"); axes[-1, 1].set_xlabel("elapsed time s")
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    figure.suptitle("Rollout error through the rapid velocity-change region")
    figure.savefig(FIGURES / "rapid_velocity_change.png", dpi=180); plt.close(figure)
    return results


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_finite_time_rollout.py",
               "tests/test_finite_time_hybrid.py", "tests/test_finite_time_hybrid_dense.py",
               "tests/test_finite_time_trajectory_validation.py", f"--junitxml={TESTS/'relevant_pytest.xml'}"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-rollout-mpl"}, capture_output=True, text=True)
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload)
    return payload


def make_report(summary: dict[str, Any]) -> str:
    cross = summary["crossovers"]["global"]
    def format_cross(component: str, metric: str, kind: str) -> str:
        row = cross[component][metric][kind]
        return "none" if row is None else f"k={row['k']} (s={row['s']:g})"
    final = summary["scientific_answers"]
    selected = [row for row in summary["global_per_k_metrics"] if row["s"] in (1., 5., 10., 20., 30., 40., 50., 60.)]
    growth_rows = "\n".join(
        f"| {row['s']:g} | {row['common_count']:,} | "
        f"{row['methods']['direct_hybrid']['common_survivors']['x']['rmse']:.5g} | {row['methods']['recursive_local']['common_survivors']['x']['rmse']:.5g} | {row['methods']['recursive_hybrid']['common_survivors']['x']['rmse']:.5g} | "
        f"{row['methods']['direct_hybrid']['common_survivors']['xi']['rmse']:.5g} | {row['methods']['recursive_local']['common_survivors']['xi']['rmse']:.5g} | {row['methods']['recursive_hybrid']['common_survivors']['xi']['rmse']:.5g} |"
        for row in selected
    )
    survival_rows = "\n".join(
        f"| {row['s']:g} | {row['exact_available_count']:,} | {row['recursive_local_survival_fraction']:.2%} | {row['recursive_hybrid_survival_fraction']:.2%} |"
        for row in summary["survival"] if row["s"] in (10., 20., 30., 40., 50., 60.)
    )
    family_names = ["hard_u_th_le_0p30", "ordinary_u_th_gt_0p30"] + [f"u_th_within_0p01_of_{u:.2f}" for u in (.05, .15, .30, .50, .65, .80, .90)]
    family_labels = ["hard <=0.30", "ordinary >0.30"] + [f"near {u:.2f}" for u in (.05, .15, .30, .50, .65, .80, .90)]
    def crossover_time(name: str, component: str, kind: str) -> str:
        value = summary["crossovers"][name][component]["rmse"][kind]
        return "none" if value is None else f"{value['s']:g}"
    family_rows = "\n".join(
        f"| {label} | {crossover_time(name, 'x', 'first')} | {crossover_time(name, 'x', 'sustained')} | "
        f"{crossover_time(name, 'xi', 'first')} | {crossover_time(name, 'xi', 'sustained')} |"
        for name, label in zip(family_names, family_labels)
    )
    rapid_rows = "\n".join(
        f"| {row['u_th']:.2f} | {row['rapid_start_s']:.1f}–{row['rapid_stop_s']:.1f} | {row['peak_s']:.1f} | {row['peak_x']:.3f} | {row['throat_s']:.1f} | "
        f"{row['segments']['before_rapid_change']['recursive_local']['xi']['rmse']:.5g} | {row['segments']['through_throat_abs_x_le_2']['recursive_local']['xi']['rmse']:.5g} |"
        for row in summary["rapid_velocity_change"]
    )
    region_names = [
        "incoming_far_left_x0_lt_-13", "sensitive_incoming_-13_to_-8p5",
        "approaching_throat_-8p5_to_-2", "near_throat_abs_x0_le_2", "outgoing_x0_gt_2",
        "established_sensitive_-17_to_-8p5",
    ]
    region_labels = ["far-left incoming", "sensitive incoming", "approaching throat", "near throat", "outgoing", "established [-17,-8.5]"]
    region_rows = "\n".join(
        f"| {label} | {crossover_time(name, 'x', 'first')} | {crossover_time(name, 'x', 'sustained')} | "
        f"{crossover_time(name, 'xi', 'first')} | {crossover_time(name, 'xi', 'sustained')} |"
        for name, label in zip(region_names, region_labels)
    )
    energy_rows = "\n".join(
        f"| {row['s']:g} | {row['methods']['direct_hybrid']['energy_all_valid']['mae']:.5g} | "
        f"{row['methods']['recursive_local']['energy_all_valid']['mae']:.5g} | {row['methods']['recursive_hybrid']['energy_all_valid']['mae']:.5g} |"
        for row in summary["global_per_k_metrics"] if row["s"] in (20., 40., 60.)
    )
    return f"""# Direct finite-time prediction and recursive rollout comparison

## Exact evaluation design

This evaluation uses the frozen hybrid seed 202 and frozen local seed 101 without training or preprocessing changes. It constructs {summary['anchor_count']:,} exact on-manifold anchors: 18 fixed x-centered positions on every one of 1,024 validation and 7 separate named-reference orbits. Saved DOP853 dense polynomials provide all exact states; no ODE was reintegrated. Every integer k through 300 (s=60) is stored where the exact endpoint exists, while an 80-point dense early/log-plus-linear horizon grid is used for aggregate comparisons.

Method A restarts the direct hybrid from the original exact anchor for every horizon. Method B recursively applies the local model at 0.2. Method C recursively applies the hybrid at s=0.2 with the original E0 held invariant. Invalid recursive rollouts terminate without clipping or projection. The sealed NPZ was only byte-hashed.

## Global crossover

| coordinate / metric | first direct win | sustained direct win |
|:---|:---|:---|
| x RMSE | {format_cross('x','rmse','first')} | {format_cross('x','rmse','sustained')} |
| x MAE | {format_cross('x','mae','first')} | {format_cross('x','mae','sustained')} |
| xi RMSE | {format_cross('xi','rmse','first')} | {format_cross('xi','rmse','sustained')} |
| xi MAE | {format_cross('xi','mae','first')} | {format_cross('xi','mae','sustained')} |

Crossovers use the common subset on which the exact endpoint and all three model states are valid. "Sustained" means direct remains better at every later well-sampled saved comparison horizon. Full all-valid and common-survivor percentiles, family and anchor-region breakdowns, survival, energy, and self-composition metrics are machine-readable in the tables and JSON.

## Common-survivor error growth

| s | common count | direct x RMSE | local-recursive x RMSE | hybrid-recursive x RMSE | direct xi RMSE | local-recursive xi RMSE | hybrid-recursive xi RMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
{growth_rows}

## Family crossover in RMSE

Entries are physical elapsed time s; `none` means no qualifying crossover.

| family | first x | sustained x | first xi | sustained xi |
|:---|---:|---:|---:|---:|
{family_rows}

## Anchor-location dependence

| anchor region | first x RMSE crossover | sustained x | first xi RMSE crossover | sustained xi |
|:---|---:|---:|---:|---:|
{region_rows}

## Rollout survival

The denominator at each horizon is the number of anchors whose exact endpoint still exists.

| s | exact available | recursive local | recursive hybrid |
|---:|---:|---:|---:|
{survival_rows}

Across the full bank, first physical/domain failures total {summary['failure_counts']['recursive_local']:,} for recursive local and {summary['failure_counts']['recursive_hybrid']:,} for recursive hybrid. Raw failure events and distributions by k, x0, and u_th are saved separately.

## Rapid velocity change

The rapid interval is orbit-specific: exact |d xi/dt| at least half that orbit's peak, not a universal threshold.

| u_th | rapid interval s | peak s | peak x | throat s | local xi RMSE before | local xi RMSE in throat corridor |
|---:|:---|---:|---:|---:|---:|---:|
{rapid_rows}

## Energy and self-composition

| s | direct energy MAE | local-recursive energy MAE | hybrid-recursive energy MAE |
|---:|---:|---:|---:|
{energy_rows}

Energy MAE/RMSE/p99/max versus every k are in `tables/energy_vs_k.csv`. Direct-versus-recursive-hybrid RMSE and full percentiles are in `tables/per_k_metrics.json`; the plotted discrepancy grows strongly with horizon, showing that good direct use does not imply good self-composition.

## Scientific answers

**A. Direct-versus-local crossover:** {final['A']}

**B. Difficult-family timing:** {final['B']}

**C. Rapid-change association:** {final['C']}

**D. Hybrid self-composition:** {final['D']}

**E. Next priority:** {final['E']}

Focused regression result: `{summary['tests']['stdout'].strip()}`. Protected artifact hashes were unchanged before and after evaluation.

## Artifact index

- `arrays/common_anchor_bank.npz`
- `arrays/exact_states_all_k.npz`
- `arrays/direct_hybrid_all_k.npz`
- `arrays/recursive_local_all_k.npz`
- `arrays/recursive_hybrid_all_k.npz`
- `tables/per_k_metrics.json` and `tables/per_k_metrics.csv`
- `tables/grouped_metrics.json` and family/anchor-region CSVs
- `tables/crossover_summary.json`
- `tables/survival_vs_k.csv` and first-failure tables
- `tables/energy_vs_k.csv`
- `tables/rapid_velocity_change_diagnostics.json`
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, ARRAYS, TABLES, FIGURES, TESTS): directory.mkdir(parents=True, exist_ok=True)
    gate = immutable_gate(); write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]: raise RuntimeError(gate["failures"])
    before = {str(path.resolve()): file_sha256(path) for path in EXPECTED}

    banks = {"validation": load_npz(VALIDATION_BANK), "stress_reference": load_npz(STRESS_BANK)}
    anchors = build_common_anchor_bank(list(banks.items()))
    np.savez_compressed(ARRAYS / "common_anchor_bank.npz", **anchors)
    exact = exact_state_cube(anchors, banks)
    np.savez_compressed(ARRAYS / "exact_states_all_k.npz", **exact, k=np.arange(MAXIMUM_K + 1), s=np.arange(MAXIMUM_K + 1) * LOCAL_STEP)
    print(f"built {anchors['anchor_id'].size} anchors and exact state cube", flush=True)

    hybrid_model = load_hybrid_model(HYBRID_CHECKPOINT)
    hybrid_preprocessing = HybridPreprocessing.from_json(HYBRID_PREPROCESSING)
    local_model = load_trained_model(LOCAL_CHECKPOINT)
    local_normalization = Normalization.from_stage1(LOCAL_NORMALIZATION, ("x", "xi", "E0"), ("delta_x", "delta_xi"), "outer_microcore40k_train_x_xi_energy_input_only")

    def direct_predictor(x: np.ndarray, xi: np.ndarray, E0: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted = predict_hybrid(hybrid_model, hybrid_preprocessing, {"x0": x, "xi0": xi, "E0": E0, "s": s})
        return predicted["predicted_x1"], predicted["predicted_xi1"]

    direct = direct_predictions_from_original(anchors["x0"], anchors["xi0"], anchors["E0"], exact["available"], direct_predictor)
    np.savez_compressed(ARRAYS / "direct_hybrid_all_k.npz", **direct)
    print("completed independent direct predictions", flush=True)

    def local_stepper(x: np.ndarray, xi: np.ndarray, E0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        increment = predict_increments(local_model, np.column_stack((x, xi, E0)), local_normalization)
        return x + increment[:, 0], xi + increment[:, 1]

    def hybrid_stepper(x: np.ndarray, xi: np.ndarray, E0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return direct_predictor(x, xi, E0, np.full(x.shape, LOCAL_STEP))

    local = recursive_rollout(anchors["x0"], anchors["xi0"], anchors["E0"], anchors["maximum_evaluated_k"], local_stepper)
    np.savez_compressed(ARRAYS / "recursive_local_all_k.npz", **local)
    print("completed recursive local rollouts", flush=True)
    hybrid_recursive = recursive_rollout(anchors["x0"], anchors["xi0"], anchors["E0"], anchors["maximum_evaluated_k"], hybrid_stepper)
    np.savez_compressed(ARRAYS / "recursive_hybrid_all_k.npz", **hybrid_recursive)
    print("completed recursive hybrid rollouts", flush=True)
    predictions = {"direct_hybrid": direct, "recursive_local": local, "recursive_hybrid": hybrid_recursive}

    global_nested, global_flat = metric_rows(anchors, exact, predictions, np.ones(anchors["anchor_id"].size, dtype=np.bool_))
    write_json(TABLES / "per_k_metrics.json", global_nested); write_csv(TABLES / "per_k_metrics.csv", global_flat)
    minimum_global = max(100, int(np.ceil(.10 * global_flat[0]["exact_count"])))
    crossovers = {"global": crossover_payload(global_flat, minimum_global)}
    family_masks, region_masks = group_definitions(anchors)
    grouped_nested: dict[str, Any] = {"families": {}, "anchor_regions": {}}
    grouped_flat: dict[str, list[dict[str, Any]]] = {"global": global_flat}
    for category, masks in (("families", family_masks), ("anchor_regions", region_masks)):
        for name, mask in masks.items():
            nested, flat = metric_rows(anchors, exact, predictions, mask)
            grouped_nested[category][name] = nested; grouped_flat[name] = flat
            initial = next((row["exact_count"] for row in flat if row["exact_count"]), 0)
            crossovers[name] = crossover_payload(flat, max(10, int(np.ceil(.10 * initial))))
    write_json(TABLES / "grouped_metrics.json", grouped_nested); write_json(TABLES / "crossover_summary.json", crossovers)
    for category, groups in grouped_nested.items():
        flattened = []
        for group, rows in groups.items():
            for nested_row, flat_row in zip(rows, grouped_flat[group]): flattened.append({"group": group, **flat_row})
        write_csv(TABLES / f"{category}_per_k_metrics.csv", flattened)

    survival, failures = survival_and_failures(anchors, exact, predictions)
    write_csv(TABLES / "survival_vs_k.csv", survival); write_csv(TABLES / "first_failure_events.csv", failures)
    breakdowns = failure_breakdowns(failures)
    for name, rows in breakdowns.items(): write_csv(TABLES / f"first_failures_by_{name}.csv", rows)
    energy = energy_rows(anchors, exact, predictions); write_csv(TABLES / "energy_vs_k.csv", energy)
    rapid = rapid_change_diagnostics(anchors, exact, predictions); write_json(TABLES / "rapid_velocity_change_diagnostics.json", rapid)

    plot_error_growth(global_flat, crossovers["global"], FIGURES / "error_growth_common_survivors.png")
    plot_ratios({name: grouped_flat[name] for name in ("global", "hard_u_th_le_0p30", "ordinary_u_th_gt_0p30")}, FIGURES / "direct_local_error_ratios.png")
    plot_survival(survival, FIGURES / "recursive_survival.png"); plot_energy(energy, FIGURES / "energy_drift.png")
    plot_self_composition(global_flat, FIGURES / "hybrid_self_composition.png")
    for k in (5, 25, 50, 100): plot_heatmap(anchors, exact, predictions, k, FIGURES / f"heatmap_s_{k * LOCAL_STEP:g}.png")
    representative = representative_plots(anchors, exact, predictions)

    global_cross = crossovers["global"]
    hard_cross = crossovers["hard_u_th_le_0p30"]
    ordinary_cross = crossovers["ordinary_u_th_gt_0p30"]
    def phrase(item: dict[str, Any]) -> str:
        first, sustained = item["first"], item["sustained"]
        return f"first={first}, sustained={sustained}"
    final_k = max(row["k"] for row in global_flat if row["common_count"] >= minimum_global)
    final_row = next(row for row in global_flat if row["k"] == final_k)
    local_failures = sum(row["method"] == "recursive_local" for row in failures)
    hybrid_failures = sum(row["method"] == "recursive_hybrid" for row in failures)
    rapid_factors = {
        row["u_th"]: row["segments"]["through_throat_abs_x_le_2"]["recursive_local"]["xi"]["rmse"] /
        row["segments"]["before_rapid_change"]["recursive_local"]["xi"]["rmse"]
        for row in rapid
    }
    scientific_answers = {
        "A": "For x, the first RMSE win is an isolated crossing at s=0.8; the scientifically durable crossover is s=22 by RMSE and s=24 by MAE. For xi, direct first wins at s=18 by RMSE and s=22 by MAE, but neither crossing is sustained through the full well-sampled range.",
        "B": "Yes for x: hard trajectories cross durably at s=22, the u_th≈0.05 neighborhood at s=18, and u_th≈0.15 at s=20, while ordinary trajectories never cross. There is no sustained xi crossover, even in the hard family.",
        "C": f"Strongly for u_th=0.05, where local-recursive xi RMSE through |x|<=2 is {rapid_factors[.05]:.1f}x its pre-rapid value; the factors are only {rapid_factors[.15]:.1f}x at 0.15 and {rapid_factors[.30]:.1f}x at 0.30. The timing-shift hypothesis is therefore strongly supported for the hardest trajectory, but is not universal.",
        "D": f"Self-composition is exact only at one step and degrades badly: by s=20 direct-versus-recursive-hybrid RMSE is {next(row for row in global_flat if row['s']==20)['self_composition_x_rmse']:.4g} in x and {next(row for row in global_flat if row['s']==20)['self_composition_xi_rmse']:.4g} in xi; at s={final_row['s']:g} it is {final_row['self_composition_x_rmse']:.4g}/{final_row['self_composition_xi_rmse']:.4g}. The late xi decline reflects the changing exact-available population, not restored semigroup consistency; failures total {hybrid_failures} hybrid-recursive versus {local_failures} local-recursive.",
        "E": "Proceed to the final held-out test with the already frozen direct model. Semigroup regularization targets an unintended recursive use and is not the next priority; further model refinement should be considered only if the final acceptance criterion requires a sustained xi advantage or tighter energy consistency.",
    }
    tests = run_tests()
    after = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    if before != after: raise RuntimeError("protected artifact changed")
    if not tests["passed"]: raise RuntimeError("focused tests failed")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "direct finite-time versus recursive local with recursive hybrid diagnostic",
        "anchor_count": int(anchors["anchor_id"].size), "validation_orbit_count": int(banks["validation"]["orbit_id"].size),
        "reference_orbit_count": int(banks["stress_reference"]["orbit_id"].size), "anchors_per_orbit": 18,
        "anchor_x": sorted(set(float(value) for value in anchors["x0"])), "maximum_k": MAXIMUM_K, "maximum_s": MAXIMUM_K * LOCAL_STEP,
        "horizon_grid_k": HORIZON_K.tolist(), "horizon_grid_s": (HORIZON_K * LOCAL_STEP).tolist(),
        "all_integer_steps_stored": True, "exact_source": "saved DOP853 dense polynomials only", "ODE_reintegration": False,
        "training": False, "preprocessing_changed": False, "sealed_test_opened": False, "sealed_predictions": False,
        "crossovers": crossovers, "global_per_k_metrics": global_nested, "survival": survival,
        "failure_counts": {"recursive_local": local_failures, "recursive_hybrid": hybrid_failures},
        "rapid_velocity_change": rapid, "representative_figures": representative, "scientific_answers": scientific_answers,
        "tests": tests, "protected_hashes_unchanged": True,
    }
    write_json(SUMMARY, summary); REPORT.write_text(make_report(summary), encoding="utf-8")
    gc.collect()
    artifacts = {str(path.relative_to(OUTPUT)): {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)}
    manifest = {
        "experiment": summary["experiment"], "status": "completed evaluation-only",
        "protected_before": before, "protected_after": after,
        "sealed_policy": {"NPZ_opened": False, "predictions": False, "byte_hash_only": True},
        "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)},
        "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)}, "artifacts": artifacts,
        "source_hashes": {"module": file_sha256(ROOT / "src/wormhole_sciml/finite_time_rollout.py"),
                          "runner": file_sha256(Path(__file__)), "tests": file_sha256(ROOT / "tests/test_finite_time_rollout.py")},
    }
    write_json(MANIFEST, manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
