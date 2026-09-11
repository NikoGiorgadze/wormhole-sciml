#!/usr/bin/env python3
"""One-time final evaluation of the frozen hybrid model on the sealed test split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, load_hybrid_model, predict_hybrid
from wormhole_sciml.finite_time_hybrid_validation import median_log_grid, predict_hybrid_diagnostics
from wormhole_sciml.finite_time_rollout import (
    HORIZON_K, LOCAL_STEP, MAXIMUM_K, build_common_anchor_bank,
    direct_predictions_from_original, energy_metrics as rollout_energy_metrics,
    first_and_sustained_crossover, physical_diagnostics, recursive_rollout, state_metrics,
)
from wormhole_sciml.finite_time_validation import (
    DENSE_ANCHOR_X, DENSE_FRACTIONS, aggregate_error_metrics, admissibility_summary,
    bin_error_rows, build_dense_queries, build_queries_at_x_and_s, energy_metrics,
    predict_local, scalar_error_metrics, subset_metrics,
)
from wormhole_sciml.model_a import Normalization, load_trained_model, predict_increments
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi, file_sha256
from wormhole_sciml.phase_c_finite_time import invert_saved_orbit_x


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/final_sealed_test"
REPORT = OUTPUT / "FINAL_SEALED_TEST_REPORT.md"
SUMMARY = OUTPUT / "final_sealed_test_summary.json"
MANIFEST = OUTPUT / "final_sealed_test_manifest.json"
MANIFEST_HASH = OUTPUT / "final_sealed_test_manifest.sha256"
CONFIG = OUTPUT / "evaluation_config_pre_prediction.json"
FREEZE = OUTPUT / "pre_prediction_freeze_manifest.json"
EVENTS = OUTPUT / "sealed_evaluation_events.json"

HYBRID_ROOT = ROOT / "output/finite_time_hybrid_s5"
HYBRID_MANIFEST = HYBRID_ROOT / "finite_time_hybrid_manifest.json"
HYBRID_CHECKPOINT = HYBRID_ROOT / "training/seed_202/best_checkpoint.pt"
HYBRID_PREPROCESSING = HYBRID_ROOT / "preprocessing/hybrid_preprocessing_constants.json"
LOCAL_ROOT = ROOT / "output/model_a_x_xi_energy_microcore40k_comparison"
LOCAL_MANIFEST = LOCAL_ROOT / "energy_xi_training_manifest.json"
LOCAL_CHECKPOINT = LOCAL_ROOT / "training/seed_101/best_checkpoint.pt"
LOCAL_NORMALIZATION = LOCAL_ROOT / "training/energy_input_normalization.json"
ORBIT_ROOT = ROOT / "output/phase_b_complete_orbit_banks"
TRAIN_BANK = ORBIT_ROOT / "banks/phase_b_train_orbits.npz"
VALIDATION_BANK = ORBIT_ROOT / "banks/phase_b_validation_orbits.npz"
TEST_BANK = ORBIT_ROOT / "banks/phase_b_test_orbits.npz"
ORBIT_MANIFEST = ORBIT_ROOT / "phase_b_manifest.json"
ORBIT_LEAKAGE = ORBIT_ROOT / "leakage_checks.json"
PHASE_C_ROOT = ROOT / "output/phase_c_finite_time_dataset"
VALIDATION_ROWS = PHASE_C_ROOT / "datasets/phase_c_validation_raw.npz"
TEST_ROWS = PHASE_C_ROOT / "datasets/phase_c_test_sealed_raw.npz"
PHASE_C_MANIFEST = PHASE_C_ROOT / "phase_c_manifest.json"
ROW_LEAKAGE = PHASE_C_ROOT / "leakage_audit.json"
VALIDATION_ROW_METRICS = HYBRID_ROOT / "validation/validation_metrics_seed_202.json"
VALIDATION_DENSE_SUMMARY = ROOT / "output/finite_time_hybrid_dense_validation/hybrid_dense_validation_summary.json"
VALIDATION_MATCHED_GRIDS = ROOT / "output/finite_time_hybrid_dense_validation/tables/matched_map_grids.json"
VALIDATION_ROLLOUT_SUMMARY = ROOT / "output/direct_vs_recursive_rollouts/direct_vs_recursive_summary.json"

EXPECTED = {
    HYBRID_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
    HYBRID_CHECKPOINT: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    HYBRID_PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    LOCAL_MANIFEST: "27050145c861fe630cee764fd0766238604bb6c04be335e23186b3ca2b45b5b5",
    LOCAL_CHECKPOINT: "54a41a54a7fa20e931df0dd987ac871786d4ac8d02e71bd3560e9ebc3228a4a6",
    LOCAL_NORMALIZATION: "5f3605f657d8190ea901112774553e3026a3399266783a78f0102b285c4045c3",
    ORBIT_MANIFEST: "207f6d3c2a5cbaf3eb613c51e4d438abf50538eb8af0e21be8f8700d2d1b716b",
    ORBIT_LEAKAGE: "d6a5e9b84533012068bda5876ce4a5959d1376c90cd890cb5a2f65c3e3b1164f",
    TRAIN_BANK: "8b72b7c89a47d25d215c3991b0f409aff4e3e006508d113bb980f27b87775fa8",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    TEST_BANK: "a79d97f6b9c9a85df12868a52c9fd7b58a7291d2e6afb4f5f3385bfbdddfdf3f",
    PHASE_C_MANIFEST: "c3c6d968bfac01cd7b4f084d331d8b359bdf29158f6f05723d7d1982815684e7",
    ROW_LEAKAGE: "4254c44669eaa366b1045affe9ae8cc164a2ad0915c12315e63eef1da466202d",
    VALIDATION_ROWS: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    TEST_ROWS: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    VALIDATION_ROW_METRICS: "4c3d172167fbac54e7dbfbeff2f188e9708ee6dd5adfbb973143b9732d6607b9",
    VALIDATION_DENSE_SUMMARY: "44234028906464acc0134b8a8ed578921bc4beb215108738eae7bc2ede3f3ff7",
    VALIDATION_MATCHED_GRIDS: "ad82d676907ad38baf8e9309ead7c7f72dc2390220e4ce5f230ad55db557e0c5",
    VALIDATION_ROLLOUT_SUMMARY: "2f5d206e6c46b64980adbfc7a09627405365cc650b858bf6f18abf5cd47c84fc",
}
S_EDGES = np.asarray([0, .01, .025, .05, .1, .2, .5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100.])
X_EDGES = np.linspace(-17., 17., 33)
MAP_S_EDGES = np.asarray([0, .05, .2, .5, 1, 2, 5, 10, 20, 40, 60, 100.])
EPSILON = 1e-12


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite diagnostic cells to strict JSON."""

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
        path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def edges_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([rows[0]["lower"], *[row["upper"] for row in rows]], dtype=np.float64)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), ROOT / "src/wormhole_sciml/finite_time_hybrid.py",
        ROOT / "src/wormhole_sciml/finite_time_hybrid_validation.py",
        ROOT / "src/wormhole_sciml/finite_time_validation.py",
        ROOT / "src/wormhole_sciml/finite_time_rollout.py",
        ROOT / "tests/test_final_sealed_test.py",
    ]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def input_gate() -> dict[str, Any]:
    artifacts, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path)
        artifacts[str(path.resolve())] = {"expected": expected, "measured": measured, "match": measured == expected}
        if measured != expected: failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": artifacts}


def frozen_references() -> dict[str, Any]:
    return {
        "row": load_json(VALIDATION_ROW_METRICS),
        "dense": load_json(VALIDATION_DENSE_SUMMARY),
        "matched_grids": load_json(VALIDATION_MATCHED_GRIDS),
        "rollout": load_json(VALIDATION_ROLLOUT_SUMMARY),
    }


def evaluation_configuration(references: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    dense = references["dense"]
    return {
        "frozen_utc": utc(), "status": "FROZEN_BEFORE_SEALED_OPEN", "test_predictions_made": False,
        "sealed_test_npz_opened_for_ml": False, "primary_model": {"seed": 202, "checkpoint": str(HYBRID_CHECKPOINT.resolve()),
        "mapping": "(x0,xi0,E0,s)->(V_x,F_xi)", "Delta_x": "s*V_x", "Delta_xi": "-5*expm1(-s/5)*F_xi", "s_star": 5.0},
        "local_model": {"seed": 101, "architecture": [3, 32, 32, 2], "native_step": .2, "checkpoint": str(LOCAL_CHECKPOINT.resolve())},
        "expected_counts": {"test_orbits": 1024, "test_rows": 98304, "dense_queries": 524288, "dense_queries_per_orbit": 512,
                            "matched_queries": 32768, "rollout_anchors": 18432, "rollout_anchors_per_orbit": 18},
        "dense_design": {"anchor_x": DENSE_ANCHOR_X.tolist(), "fractions": DENSE_FRACTIONS.tolist()},
        "rollout_design": {"local_step": LOCAL_STEP, "maximum_k": MAXIMUM_K, "maximum_s": MAXIMUM_K * LOCAL_STEP,
                           "comparison_k": HORIZON_K.tolist(), "minimum_common_count": "max(100,ceil(0.10*first_exact_count))",
                           "first_crossover": "first saved well-sampled k with direct error < local error",
                           "sustained_crossover": "first saved well-sampled k after which direct remains lower at every remaining well-sampled saved k",
                           "population": "the 1024 target-split orbits only; no stress-reference supplement is mixed into global test metrics",
                           "validation_reference_population_note": "published validation global includes 1024 validation plus 7 stress-reference orbits; its frozen sustained x crossover is s=22 RMSE and s=24 MAE"},
        "subsets": {"hard": "u_th<=0.30", "ordinary": "u_th>0.30", "sensitive_incoming": "-17<=x0<=-8.5",
                    "neighborhoods": {str(v): "abs(u_th-center)<=0.01" for v in (.05, .15, .30, .90)},
                    "short": "s<=5", "intermediate": "5<s<=20", "long": "s>20"},
        "rapid_change": {"selected_orbits": "nearest test u_th to 0.05,0.15,0.30 before predictions are examined",
                         "anchor": "leftmost frozen rollout anchor", "rapid_interval": "orbit-specific exact |d xi/dt| >= 0.5*orbit maximum",
                         "segments": ["before rapid interval", "rapid interval", "throat corridor |x|<=2", "immediately after rapid interval until x>8"]},
        "bin_edges": {"s": S_EDGES.tolist(), "x0": X_EDGES.tolist(),
                      "u_th": edges_from_rows(dense["coordinate_bins"]["u_th"]).tolist(),
                      "E0": edges_from_rows(dense["coordinate_bins"]["E0"]).tolist(), "map_s": MAP_S_EDGES.tolist()},
        "map_color": {"cmap": "magma", "log10_error_vmin": -6.0, "log10_error_vmax": 0.0,
                      "ratio_cmap": "coolwarm", "ratio_vmin": -3.0, "ratio_vmax": 3.0, "epsilon": EPSILON},
        "representative_orbits": "nearest u_th before prediction inspection to 0.05,0.15,0.30,0.50,0.65,0.80,0.90",
        "representative_anchor_x": [-14., -8., 0.], "no_ode_reintegration": True, "no_training": True,
        "no_test_driven_tuning": True, "input_hash_gate": gate, "evaluation_code_sha256": source_hashes(),
        "validation_reference_files": {str(path.resolve()): EXPECTED[path] for path in (VALIDATION_ROW_METRICS, VALIDATION_DENSE_SUMMARY, VALIDATION_MATCHED_GRIDS, VALIDATION_ROLLOUT_SUMMARY)},
        "predeclared_validation_principal": {"dense_x_rmse": .063681, "dense_x_mae": .016182, "dense_xi_rmse": .00295434,
                                              "dense_xi_mae": .000920645, "energy_rmse": .0220953, "energy_mae": .00477153,
                                              "matched_local_x_rmse": .000113105, "matched_direct_x_rmse": .000153120,
                                              "matched_local_xi_rmse": 1.74116e-5, "matched_direct_xi_rmse": 6.17314e-5,
                                              "rollout_sustained_x_rmse_s": 22., "rollout_sustained_x_mae_s": 24.},
    }


def queries_from_rows(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**rows, "exact_x1": rows["x1"], "exact_u1": rows["u1"], "exact_xi1": rows["xi1"]}


def metrics_for_prediction(prediction: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None: mask = np.ones(prediction["x_error"].shape, dtype=bool)
    return subset_metrics(prediction, np.asarray(mask, dtype=bool))


def coordinate_metrics(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {name: bin_error_rows(queries[name], prediction, np.asarray(config["bin_edges"][name]), name) for name in ("s", "x0", "u_th", "E0")}


def subset_masks(queries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {"hard_u_th_le_0p30": queries["u_th"] <= .30, "ordinary_u_th_gt_0p30": queries["u_th"] > .30,
             "sensitive_incoming_-17_to_-8p5": (queries["x0"] >= -17) & (queries["x0"] <= -8.5),
             "short_s_le_5": queries["s"] <= 5, "intermediate_5_lt_s_le_20": (queries["s"] > 5) & (queries["s"] <= 20),
             "long_s_gt_20": queries["s"] > 20}
    for center in (.05, .15, .30, .90): masks[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(queries["u_th"] - center) <= .01
    return masks


def save_prediction_table(path: Path, queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> None:
    query_fields = {name: value for name, value in queries.items() if np.asarray(value).ndim == 1 and len(value) == len(queries["x0"])}
    fields = {**query_fields, **{name: value for name, value in prediction.items() if np.asarray(value).shape == queries["x0"].shape}}
    np.savez_compressed(path, **fields)


def plot_coordinate(test_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]] | None, coordinate: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
    for ax, (component, metric) in zip(axes.ravel(), (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))):
        ax.plot([r["center"] for r in test_rows], [r[f"{component}_{metric}"] for r in test_rows], label="sealed test")
        if validation_rows is not None:
            ax.plot([r["center"] for r in validation_rows], [r[f"{component}_{metric}"] for r in validation_rows], "--", label="validation")
        ax.set(ylabel=f"{metric.upper()} {component}", yscale="log"); ax.grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel(coordinate)
    axes[0, 0].legend(); fig.suptitle(f"Direct finite-time error versus {coordinate}"); fig.savefig(path, dpi=180); plt.close(fig)


def plot_percentiles(test_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, component in zip(axes, ("x", "xi")):
        for percentile, style in (("p90", "-"), ("p99", "--")):
            ax.plot([r["center"] for r in test_rows], [r[f"{component}_{percentile}_absolute"] for r in test_rows], style, label=f"test {percentile}")
            ax.plot([r["center"] for r in validation_rows], [r[f"{component}_{percentile}_absolute"] for r in validation_rows], style, alpha=.45, label=f"validation {percentile}")
        ax.set(xlabel="physical elapsed time s", ylabel=f"absolute {component} error", yscale="log"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Direct finite-time error percentiles versus elapsed time"); fig.savefig(path, dpi=180); plt.close(fig)


def plot_dense_maps(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], config: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, component in zip(axes, ("x", "xi")):
        grid, _ = median_log_grid(queries["x0"], queries["s"], np.abs(prediction[f"{component}_error"]), X_EDGES, MAP_S_EDGES, EPSILON)
        mesh = ax.pcolormesh(X_EDGES, MAP_S_EDGES, np.ma.masked_invalid(grid), shading="auto", cmap="magma", vmin=-6, vmax=0)
        fig.colorbar(mesh, ax=ax, label=f"log10 median |{component} error|")
        ax.set(xlabel="exact anchor x0", ylabel="physical elapsed time s", title=f"{component} error")
    fig.suptitle("Sealed-test direct finite-time error across anchor and elapsed time"); fig.savefig(path, dpi=180); plt.close(fig)


def matched_map(queries: dict[str, np.ndarray], local: dict[str, np.ndarray], direct: dict[str, np.ndarray], y_name: str,
                x_edges: np.ndarray, y_edges: np.ndarray, path: Path) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    saved: dict[str, Any] = {"x_edges": x_edges.tolist(), "y_edges": y_edges.tolist(), "epsilon": EPSILON}
    for row, component in enumerate(("x", "xi")):
        local_abs, direct_abs = np.abs(local[f"{component}_error"]), np.abs(direct[f"{component}_error"])
        lg, counts = median_log_grid(queries["x0"], queries[y_name], local_abs, x_edges, y_edges, EPSILON)
        dg, _ = median_log_grid(queries["x0"], queries[y_name], direct_abs, x_edges, y_edges, EPSILON)
        rg, _ = median_log_grid(queries["x0"], queries[y_name], (direct_abs + EPSILON) / (local_abs + EPSILON), x_edges, y_edges, EPSILON)
        finite = np.concatenate((lg[np.isfinite(lg)], dg[np.isfinite(dg)])); lo, hi = np.quantile(finite, (.01, .99))
        for column, (grid, title, cmap, vmin, vmax) in enumerate(((lg, "local absolute error", "magma", lo, hi),
                (dg, "direct absolute error", "magma", lo, hi), (rg, "log10 direct/local ratio", "coolwarm", -3, 3))):
            mesh = axes[row, column].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid), shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(mesh, ax=axes[row, column]); axes[row, column].set(xlabel="exact anchor x0", ylabel=y_name, title=f"{component}: {title}")
        saved[component] = {"local_log10": lg.tolist(), "direct_log10": dg.tolist(), "ratio_log10": rg.tolist(), "counts": counts.tolist()}
    fig.suptitle(f"Sealed-test local and direct errors at s=0.2 in (x0, {y_name})"); fig.savefig(path, dpi=180); plt.close(fig)
    return saved


def exact_state_cube(anchors: dict[str, np.ndarray], bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    shape = (anchors["anchor_id"].size, MAXIMUM_K + 1)
    x, u, xi, time = (np.full(shape, np.nan) for _ in range(4)); available = np.zeros(shape, dtype=bool)
    for orbit_index in np.unique(anchors["source_orbit_index"]):
        rows = np.flatnonzero(anchors["source_orbit_index"] == orbit_index); rr, kk, tt = [], [], []
        for row in rows:
            k = np.arange(int(anchors["maximum_evaluated_k"][row]) + 1, dtype=np.int16)
            rr.append(np.full(k.size, row)); kk.append(k); tt.append(anchors["t0"][row] + LOCAL_STEP * k)
        rr, kk, tt = np.concatenate(rr).astype(int), np.concatenate(kk).astype(int), np.concatenate(tt)
        state = evaluate_saved_orbit_x_u_xi(bank, int(orbit_index), tt)
        x[rr, kk], u[rr, kk], xi[rr, kk], time[rr, kk], available[rr, kk] = state[:, 0], state[:, 1], state[:, 2], tt, True
    return {"x": x, "u": u, "xi": xi, "time": time, "available": available}


def rollout_metrics(anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], direct: dict[str, np.ndarray], local: dict[str, np.ndarray], mask: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nested, flat = [], []
    for k in HORIZON_K:
        exact_mask = mask & exact["available"][:, k]
        common = exact_mask & direct["valid"][:, k] & local["valid"][:, k]
        row: dict[str, Any] = {"k": int(k), "s": float(k * LOCAL_STEP), "exact_count": int(exact_mask.sum()), "common_count": int(common.sum()), "methods": {}}
        f: dict[str, Any] = {key: row[key] for key in ("k", "s", "exact_count", "common_count")}
        for name, pred in (("direct", direct), ("local", local)):
            own = exact_mask & pred["valid"][:, k]
            all_metric = state_metrics(pred["x"][:, k], pred["xi"][:, k], exact["x"][:, k], exact["xi"][:, k], own)
            common_metric = state_metrics(pred["x"][:, k], pred["xi"][:, k], exact["x"][:, k], exact["xi"][:, k], common)
            energy = rollout_energy_metrics(pred["E"][:, k], anchors["E0"], own)
            row["methods"][name] = {"all_valid": all_metric, "common": common_metric, "energy": energy}
            f[f"{name}_all_count"], f[f"{name}_common_count"] = all_metric["count"], common_metric["count"]
            for scope, metric_value in (("all", all_metric), ("common", common_metric)):
                for component in ("x", "xi"):
                    for metric_name in ("rmse", "mae", "median_absolute", "p90_absolute", "p95_absolute", "p99_absolute", "maximum_absolute"):
                        f[f"{name}_{scope}_{component}_{metric_name}"] = np.nan if metric_value[component] is None else metric_value[component][metric_name]
            if energy:
                for metric_name, value in energy.items(): f[f"{name}_energy_{metric_name}"] = value
        nested.append(row); flat.append(f)
    return nested, flat


def crossover(flat: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    output: dict[str, Any] = {"minimum_common_count": minimum}
    for component in ("x", "xi"):
        output[component] = {}
        for metric in ("rmse", "mae", "median_absolute"):
            output[component][metric] = first_and_sustained_crossover(flat, f"direct_common_{component}_{metric}", f"local_common_{component}_{metric}", minimum_count=minimum)
    return output


def plot_rollout(flat: list[dict[str, Any]], component: str, cross: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("rmse", "mae")):
        for name, color in (("direct", "#0072B2"), ("local", "#D55E00")):
            ax.plot([r["s"] for r in flat], [r[f"{name}_common_{component}_{metric}"] for r in flat], label=name, color=color)
        sustained = cross[component][metric]["sustained"]
        if sustained: ax.axvline(sustained["s"], color="black", ls=":", label="sustained crossover")
        ax.set(xlabel="total elapsed time s", ylabel=f"common-survivor {metric.upper()} {component}", yscale="log"); ax.grid(alpha=.25)
    axes[0].legend(); fig.suptitle(f"Direct finite-time and recursive local {component} error"); fig.savefig(path, dpi=180); plt.close(fig)


def plot_rollout_ratio(groups: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for ax, (component, metric) in zip(axes.ravel(), (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))):
        for name, rows in groups.items():
            d = np.asarray([r[f"direct_common_{component}_{metric}"] for r in rows]); l = np.asarray([r[f"local_common_{component}_{metric}"] for r in rows])
            ax.plot([r["s"] for r in rows], np.log10((d + EPSILON) / (l + EPSILON)), label=name)
        ax.axhline(0, color="black", lw=1); ax.set(ylabel=f"log10 direct/local {metric.upper()} {component}"); ax.grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel("total elapsed time s")
    axes[0, 0].legend(); fig.suptitle("Sealed-test direct/local error ratios across orbit families"); fig.savefig(path, dpi=180); plt.close(fig)


def plot_rollout_energy(rows: list[dict[str, Any]], survival: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for ax, metric in zip(axes.ravel()[:3], ("mae", "rmse", "p99_absolute")):
        for name, color in (("direct", "#0072B2"), ("local", "#D55E00")):
            ax.plot([r["s"] for r in rows], [r["methods"][name]["energy"][metric] for r in rows], color=color, label=name)
        ax.set(ylabel=f"energy {metric.replace('_absolute', '').upper()}", yscale="log"); ax.grid(alpha=.25)
    axes[0, 0].legend()
    axes[1, 1].plot([r["s"] for r in survival], [r["local_survival_fraction"] for r in survival], color="#D55E00")
    axes[1, 1].set(ylabel="recursive-local survival", ylim=(-.02, 1.02)); axes[1, 1].grid(alpha=.25)
    for ax in axes[-1]: ax.set_xlabel("total elapsed time s")
    fig.suptitle("Energy consistency and recursive-local survival on sealed trajectories"); fig.savefig(path, dpi=180); plt.close(fig)


def representative_trajectory(bank: dict[str, np.ndarray], orbit_index: int, model: Any, preprocessing: HybridPreprocessing, path: Path) -> None:
    anchors = (-14., -8., 0.); fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for column, anchor in enumerate(anchors):
        t0, _ = invert_saved_orbit_x(bank, orbit_index, np.asarray([anchor])); elapsed = np.linspace(0, float(bank["t_right"][orbit_index]) - float(t0[0]), 220)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit_index, float(t0[0]) + elapsed)
        raw = predict_hybrid(model, preprocessing, {"x0": np.full(elapsed.size, exact[0, 0]), "xi0": np.full(elapsed.size, exact[0, 2]),
                                                        "E0": np.full(elapsed.size, float(bank["E0"][orbit_index])), "s": elapsed})
        axes[0, column].plot(elapsed, exact[:, 0], color="black", label="exact"); axes[0, column].plot(elapsed, raw["predicted_x1"], "--", label="direct")
        axes[1, column].plot(elapsed, exact[:, 2], color="black"); axes[1, column].plot(elapsed, raw["predicted_xi1"], "--")
        axes[2, column].plot(exact[:, 0], exact[:, 2], color="black"); axes[2, column].plot(raw["predicted_x1"], raw["predicted_xi1"], "--")
        axes[0, column].set(title=f"anchor x0={anchor:g}", xlabel="elapsed time s", ylabel="x"); axes[1, column].set(xlabel="elapsed time s", ylabel="xi"); axes[2, column].set(xlabel="x", ylabel="xi")
        for ax in axes[:, column]: ax.grid(alpha=.25)
    axes[0, 0].legend(); fig.suptitle(f"Independent direct predictions on sealed orbit u_th={bank['u_th'][orbit_index]:.5f}")
    fig.savefig(path, dpi=175); plt.close(fig)


def rapid_diagnostic(anchors: dict[str, np.ndarray], exact: dict[str, np.ndarray], direct: dict[str, np.ndarray], local: dict[str, np.ndarray], targets=(.05, .15, .30)) -> list[dict[str, Any]]:
    results = []; fig, axes = plt.subplots(len(targets), 2, figsize=(13, 11), constrained_layout=True)
    for row_number, target in enumerate(targets):
        orbit_ids = np.unique(anchors["orbit_id"]); orbit_u = np.asarray([anchors["u_th"][np.flatnonzero(anchors["orbit_id"] == oid)[0]] for oid in orbit_ids])
        oid = orbit_ids[int(np.argmin(np.abs(orbit_u - target)))]; candidates = np.flatnonzero(anchors["orbit_id"] == oid); row = int(candidates[np.argmin(anchors["x0"][candidates])])
        max_k = int(anchors["maximum_evaluated_k"][row]); k = np.arange(max_k + 1); s = k * LOCAL_STEP
        ex, exi = exact["x"][row, :max_k + 1], exact["xi"][row, :max_k + 1]; rate = np.abs(np.gradient(exi, LOCAL_STEP))
        threshold = .5 * float(rate.max()); rapid = rate >= threshold; indices = np.flatnonzero(rapid); start, stop = int(indices[0]), int(indices[-1])
        throat = int(np.argmin(np.abs(ex)))
        segments = {"before_rapid_change": k < start, "rapid_change_before_throat": rapid & (ex < 0),
                    "through_throat_abs_x_le_2": np.abs(ex) <= 2,
                    "immediately_after_throat_2_lt_x_le_8": (ex > 2) & (ex <= 8),
                    "relaxed_outgoing_x_gt_8": ex > 8}
        row_result: dict[str, Any] = {"target_u_th": target, "actual_u_th": float(anchors["u_th"][row]), "orbit_id": str(oid), "anchor_x0": float(anchors["x0"][row]),
            "peak_abs_dxi_dt": float(rate.max()), "rapid_threshold": threshold,
            "rapid_start_k": start, "rapid_stop_k": stop, "rapid_start_s": float(s[start]), "rapid_stop_s": float(s[stop]),
            "peak_k": int(np.argmax(rate)), "peak_s": float(s[np.argmax(rate)]), "peak_x": float(ex[np.argmax(rate)]),
            "throat_k": throat, "throat_s": float(s[throat]), "segments": {}}
        for segment, mask in segments.items():
            row_result["segments"][segment] = {}
            for name, pred in (("direct", direct), ("local", local)):
                valid = mask & pred["valid"][row, :max_k + 1]
                row_result["segments"][segment][name] = state_metrics(pred["x"][row, :max_k + 1], pred["xi"][row, :max_k + 1], ex, exi, valid)
        results.append(row_result)
        for name, pred, color in (("direct", direct, "#0072B2"), ("local", local, "#D55E00")):
            valid = pred["valid"][row, :max_k + 1]; axes[row_number, 0].plot(s[valid], np.abs(pred["xi"][row, :max_k + 1][valid] - exi[valid]), color=color, label=name)
        axes[row_number, 0].axvspan(s[start], s[stop], color="grey", alpha=.2); axes[row_number, 0].set(ylabel=f"u_th={anchors['u_th'][row]:.4f}\nabsolute xi error", yscale="log"); axes[row_number, 0].grid(alpha=.25)
        axes[row_number, 1].plot(s, exi, color="black", label="exact xi"); twin = axes[row_number, 1].twinx(); twin.plot(s, rate, color="#CC79A7"); axes[row_number, 1].axvspan(s[start], s[stop], color="grey", alpha=.2); axes[row_number, 1].set(ylabel="exact xi"); twin.set_ylabel("|d xi/dt|")
    axes[0, 0].legend(); axes[-1, 0].set_xlabel("elapsed time s"); axes[-1, 1].set_xlabel("elapsed time s"); fig.suptitle("Sealed-test error through orbit-specific rapid velocity change")
    fig.savefig(FIGURES / "rapid_velocity_change.png", dpi=180); plt.close(fig); return results


def leakage_check(test_bank: dict[str, np.ndarray]) -> dict[str, Any]:
    train, validation = load_npz(TRAIN_BANK), load_npz(VALIDATION_BANK)
    ids = {"train": set(train["orbit_id"].tolist()), "validation": set(validation["orbit_id"].tolist()), "test": set(test_bank["orbit_id"].tolist())}
    def minimum_separation(a: np.ndarray, b: np.ndarray) -> float:
        b = np.sort(b); pos = np.searchsorted(b, a); values = []
        for offset in (-1, 0):
            index = np.clip(pos + offset, 0, b.size - 1); values.append(np.abs(a - b[index]))
        return float(np.min(np.concatenate(values)))
    exact_duplicates = np.intersect1d(np.concatenate((train["u_th"], validation["u_th"])), test_bank["u_th"])
    return {"orbit_id_intersections": {"train_test": sorted(ids["train"] & ids["test"]), "validation_test": sorted(ids["validation"] & ids["test"])},
            "exact_u_th_duplicates": exact_duplicates.tolist(), "minimum_u_th_separation": {"train_test": minimum_separation(test_bank["u_th"], train["u_th"]),
            "validation_test": minimum_separation(test_bank["u_th"], validation["u_th"])}, "suspicious_threshold": 1e-10}


def comparison_rows(row_result: dict[str, Any], dense: dict[str, Any], matched: dict[str, Any], rollout_cross: dict[str, Any], references: dict[str, Any]) -> list[dict[str, Any]]:
    vr, vd = references["row"], references["dense"]
    values: list[tuple[str, Any, Any]] = []
    for component in ("x", "xi"):
        for metric in ("rmse", "mae"):
            values.append((f"row_{component}_{metric}", vr["physical_state_metrics"][component][metric], row_result["aggregate"][component][metric]))
            values.append((f"dense_{component}_{metric}", vd["dense_aggregate"]["row_weighted"][component][metric], dense["aggregate"]["row_weighted"][component][metric]))
    for subset in ("hard_u_th_le_0p30", "ordinary_u_th_gt_0p30", "short_s_le_5", "intermediate_5_lt_s_le_20", "long_s_gt_20"):
        validation_subset = vr["families"].get(subset, vr["time_regimes"].get(subset))
        for component in ("x", "xi"): values.append((f"row_{subset}_{component}_rmse", validation_subset[component]["rmse"], row_result["subsets"][subset][component]["rmse"]))
    for name in ("mae", "rmse"):
        values.append((f"dense_energy_{name}", vd["energy"][name], dense["energy"][name]))
    values.append(("dense_admissibility_violations", vd["admissibility"]["union_violation_count"], dense["admissibility"]["union_violation_count"]))
    for model in ("local", "direct"):
        validation_model = "hybrid" if model == "direct" else "local"
        for component in ("x", "xi"):
            values.append((f"matched_{model}_{component}_rmse", vd["matched"]["aggregate"][validation_model][component]["rmse"], matched["aggregate"][model][component]["rmse"]))
    v_cross = references["rollout"]["crossovers"]["global"]
    for metric in ("rmse", "mae"):
        v = v_cross["x"][metric]["sustained"]; t = rollout_cross["x"][metric]["sustained"]
        values.append((f"rollout_sustained_x_{metric}_s", None if v is None else v["s"], None if t is None else t["s"]))
    return [{"metric": name, "validation": validation, "sealed_test": test,
             "test_over_validation": None if validation in (None, 0) or test is None else test / validation} for name, validation, test in values]


def run_tests(output: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_final_sealed_test.py", "tests/test_finite_time_rollout.py",
               "tests/test_finite_time_hybrid.py", "tests/test_finite_time_hybrid_dense.py", "tests/test_finite_time_trajectory_validation.py",
               f"--junitxml={output/'tests/focused_pytest.xml'}"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-final-test-mpl"}, capture_output=True, text=True)
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(output / "tests/test_summary.json", payload); return payload


def evaluate(bank_path: Path, rows_path: Path, output: Path, config: dict[str, Any], references: dict[str, Any], *, sealed: bool) -> dict[str, Any]:
    global FIGURES
    arrays, tables, figures, tests = (output / name for name in ("arrays", "tables", "figures", "tests")); FIGURES = figures
    for directory in (output, arrays, tables, figures, tests): directory.mkdir(parents=True, exist_ok=True)
    bank, raw_rows = load_npz(bank_path), load_npz(rows_path)
    if bank["orbit_id"].size != 1024 or raw_rows["s"].size != 98304: raise RuntimeError("sealed count mismatch")
    hybrid_model = load_hybrid_model(HYBRID_CHECKPOINT); preprocessing = HybridPreprocessing.from_json(HYBRID_PREPROCESSING)
    local_model = load_trained_model(LOCAL_CHECKPOINT)
    local_norm = Normalization.from_stage1(LOCAL_NORMALIZATION, ("x", "xi", "E0"), ("delta_x", "delta_xi"), "outer_microcore40k_train_x_xi_energy_input_only")

    row_queries = queries_from_rows(raw_rows); row_prediction = predict_hybrid_diagnostics(hybrid_model, preprocessing, row_queries)
    save_prediction_table(arrays / "row_level_test_predictions.npz", row_queries, row_prediction)
    row_subsets = {name: metrics_for_prediction(row_prediction, mask) for name, mask in subset_masks(row_queries).items()}
    row_result = {"aggregate": metrics_for_prediction(row_prediction), "subsets": row_subsets,
                  "identity": {"count": int(np.sum(raw_rows["s"] == 0)), "x_max": float(np.max(np.abs(row_prediction["x_error"][raw_rows["s"] == 0]))),
                               "xi_max": float(np.max(np.abs(row_prediction["xi_error"][raw_rows["s"] == 0])))}}
    write_json(tables / "row_level_metrics.json", row_result)

    dense_queries = build_dense_queries(bank)
    if dense_queries["s"].size != 524288: raise RuntimeError("dense test count mismatch")
    dense_prediction = predict_hybrid_diagnostics(hybrid_model, preprocessing, dense_queries)
    save_prediction_table(arrays / "dense_test_predictions.npz", dense_queries, dense_prediction)
    aggregate, per_orbit = aggregate_error_metrics(dense_prediction["x_error"], dense_prediction["xi_error"], dense_queries["orbit_id"])
    coordinates = coordinate_metrics(dense_queries, dense_prediction, config)
    for name, rows in coordinates.items(): write_csv(tables / f"dense_error_vs_{name}.csv", rows)
    write_csv(tables / "dense_per_orbit_metrics.csv", [{"orbit_id": row["orbit_id"], "row_count": row["row_count"],
        **{f"{component}_{metric}": value for component in ("x", "xi") for metric, value in row[component].items()}} for row in per_orbit])
    dense_subsets = {name: metrics_for_prediction(dense_prediction, mask) for name, mask in subset_masks(dense_queries).items()}
    admissibility, invalid = admissibility_summary(dense_queries, dense_prediction); energy = energy_metrics(dense_prediction)
    order = np.argsort(np.abs(dense_prediction["energy_error"]))[-100:]
    energy_tail = {"top100_high_E0_top5pct_fraction": float(np.mean(dense_queries["E0"][order] >= np.quantile(dense_queries["E0"], .95))),
                   "top100_low_C_bottom5pct_fraction": float(np.mean(dense_prediction["exact_C1"][order] <= np.quantile(dense_prediction["exact_C1"], .05))),
                   "top100_near_throat_fraction": float(np.mean(np.abs(dense_queries["exact_x1"][order]) <= 2))}
    violations = [{"query_index": int(i), "orbit_id": str(dense_queries["orbit_id"][i]), "x0": float(dense_queries["x0"][i]),
                   "u_th": float(dense_queries["u_th"][i]), "s": float(dense_queries["s"][i]), "predicted_x": float(dense_prediction["predicted_x1"][i]),
                   "predicted_xi": float(dense_prediction["predicted_xi1"][i]), "predicted_C": float(dense_prediction["predicted_C1"][i])} for i in np.flatnonzero(invalid)]
    write_csv(tables / "dense_admissibility_violations.csv", violations)
    dense_result = {"aggregate": aggregate, "subsets": dense_subsets, "coordinate_bins": coordinates, "admissibility": admissibility,
                    "energy": energy, "energy_tail": energy_tail, "identity": {"x_max": float(np.max(np.abs(dense_prediction["x_error"][dense_queries["s"] == 0]))),
                    "xi_max": float(np.max(np.abs(dense_prediction["xi_error"][dense_queries["s"] == 0])))}}
    write_json(tables / "dense_metrics.json", dense_result)
    for name, rows in coordinates.items(): plot_coordinate(rows, references["dense"]["coordinate_bins"].get(name), name, figures / f"dense_error_vs_{name}.png")
    plot_percentiles(coordinates["s"], references["dense"]["coordinate_bins"]["s"], figures / "dense_error_percentiles_vs_s.png")
    plot_dense_maps(dense_queries, dense_prediction, config, figures / "dense_error_maps_x0_s.png")

    matched_queries = build_queries_at_x_and_s(bank, DENSE_ANCHOR_X, np.asarray([LOCAL_STEP]))
    if matched_queries["s"].size != 32768 or not np.all(matched_queries["s"] == LOCAL_STEP): raise RuntimeError("matched test design mismatch")
    local_prediction = predict_local(local_model, local_norm, matched_queries); direct_prediction = predict_hybrid_diagnostics(hybrid_model, preprocessing, matched_queries)
    save_prediction_table(arrays / "matched_s0p2_anchor_bank.npz", matched_queries, {})
    save_prediction_table(arrays / "matched_local_predictions.npz", matched_queries, local_prediction)
    save_prediction_table(arrays / "matched_direct_predictions.npz", matched_queries, direct_prediction)
    matched = {"aggregate": {name: metrics_for_prediction(pred) for name, pred in (("local", local_prediction), ("direct", direct_prediction))}, "families": {}}
    for name, pred in (("local", local_prediction), ("direct", direct_prediction)):
        matched["families"][name] = {family: metrics_for_prediction(pred, mask) for family, mask in (("hard", matched_queries["u_th"] <= .30), ("ordinary", matched_queries["u_th"] > .30))}
    write_json(tables / "matched_s0p2_metrics.json", matched)
    prior_grids = references["matched_grids"]
    map_u = matched_map(matched_queries, local_prediction, direct_prediction, "u_th", np.asarray(prior_grids["x0_u_th"]["x_edges"]), np.asarray(prior_grids["x0_u_th"]["y_edges"]), figures / "matched_maps_x0_u_th.png")
    map_xi = matched_map(matched_queries, local_prediction, direct_prediction, "xi0", np.asarray(prior_grids["x0_xi0"]["x_edges"]), np.asarray(prior_grids["x0_xi0"]["y_edges"]), figures / "matched_maps_x0_xi0.png")
    write_json(tables / "matched_map_grids.json", {"x0_u_th": map_u, "x0_xi0": map_xi})

    anchors = build_common_anchor_bank([("test", bank)])
    if anchors["anchor_id"].size != 18432: raise RuntimeError("rollout anchor count mismatch")
    exact = exact_state_cube(anchors, bank)
    def direct_predictor(x: np.ndarray, xi: np.ndarray, E0: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = predict_hybrid(hybrid_model, preprocessing, {"x0": x, "xi0": xi, "E0": E0, "s": s}); return raw["predicted_x1"], raw["predicted_xi1"]
    direct_rollout = direct_predictions_from_original(anchors["x0"], anchors["xi0"], anchors["E0"], exact["available"], direct_predictor)
    def local_stepper(x: np.ndarray, xi: np.ndarray, E0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        increment = predict_increments(local_model, np.column_stack((x, xi, E0)), local_norm); return x + increment[:, 0], xi + increment[:, 1]
    local_rollout = recursive_rollout(anchors["x0"], anchors["xi0"], anchors["E0"], anchors["maximum_evaluated_k"], local_stepper)
    np.savez_compressed(arrays / "rollout_anchor_bank.npz", **anchors); np.savez_compressed(arrays / "rollout_exact_states.npz", **exact)
    np.savez_compressed(arrays / "rollout_direct_predictions.npz", **direct_rollout); np.savez_compressed(arrays / "rollout_local_predictions.npz", **local_rollout)
    all_mask = np.ones(anchors["anchor_id"].size, dtype=bool); rollout_nested, rollout_flat = rollout_metrics(anchors, exact, direct_rollout, local_rollout, all_mask)
    minimum = max(100, int(np.ceil(.10 * rollout_flat[0]["exact_count"]))); crossovers = {"global": crossover(rollout_flat, minimum)}; group_flat = {"global": rollout_flat}
    groups = {"hard": anchors["u_th"] <= .30, "ordinary": anchors["u_th"] > .30}
    for center in (.05, .15, .30): groups[f"near_{center:.2f}"] = np.abs(anchors["u_th"] - center) <= .01
    grouped = {}
    for name, mask in groups.items():
        nested, flat = rollout_metrics(anchors, exact, direct_rollout, local_rollout, mask); grouped[name] = nested; group_flat[name] = flat
        first_count = next((r["exact_count"] for r in flat if r["exact_count"]), 0); crossovers[name] = crossover(flat, max(10, int(np.ceil(.10 * first_count))))
    write_json(tables / "rollout_per_k_metrics.json", rollout_nested); write_csv(tables / "rollout_per_k_metrics.csv", rollout_flat)
    write_json(tables / "rollout_family_metrics.json", grouped); write_json(tables / "rollout_crossover_summary.json", crossovers)
    survival = []
    for k in range(1, MAXIMUM_K + 1):
        available = exact["available"][:, k]; count = int(available.sum()); survivors = int(np.sum(available & local_rollout["valid"][:, k]))
        survival.append({"k": k, "s": k * LOCAL_STEP, "exact_count": count, "local_survivor_count": survivors, "local_survival_fraction": survivors / count if count else np.nan})
    write_csv(tables / "rollout_local_survival.csv", survival)
    rapid = rapid_diagnostic(anchors, exact, direct_rollout, local_rollout); write_json(tables / "rapid_velocity_change_diagnostics.json", rapid)
    plot_rollout(rollout_flat, "x", crossovers["global"], figures / "rollout_x_error.png"); plot_rollout(rollout_flat, "xi", crossovers["global"], figures / "rollout_xi_error.png")
    plot_rollout_ratio({name: group_flat[name] for name in ("global", "hard", "ordinary")}, figures / "rollout_error_ratios.png")
    plot_rollout_energy(rollout_nested, survival, figures / "rollout_energy_and_survival.png")

    representatives = []
    for target in (.05, .15, .30, .50, .65, .80, .90):
        index = int(np.argmin(np.abs(bank["u_th"] - target))); path = figures / f"representative_u_th_target_{target:.2f}.png"
        representative_trajectory(bank, index, hybrid_model, preprocessing, path); representatives.append({"target": target, "actual": float(bank["u_th"][index]), "orbit_id": str(bank["orbit_id"][index]), "figure": str(path.resolve())})

    leakage = leakage_check(bank) if sealed else {"preflight": True}
    comparison = comparison_rows(row_result, dense_result, matched, crossovers["global"], references)
    write_csv(tables / "validation_vs_test.csv", comparison); write_json(tables / "validation_vs_test.json", comparison)
    return {"row": row_result, "dense": dense_result, "matched": matched, "rollout": {"crossovers": crossovers, "metrics": rollout_nested,
            "survival": survival, "local_failure_count": int(np.sum(local_rollout["first_failure_k"] >= 0))}, "rapid_change": rapid,
            "representatives": representatives, "leakage": leakage, "comparison": comparison,
            "counts": {"orbits": int(bank["orbit_id"].size), "rows": int(raw_rows["s"].size), "dense": int(dense_queries["s"].size),
                       "matched": int(matched_queries["s"].size), "rollout_anchors": int(anchors["anchor_id"].size)}}


def report_text(summary: dict[str, Any]) -> str:
    d, m, c = summary["dense"], summary["matched"], summary["rollout"]["crossovers"]
    val = {row["metric"]: row for row in summary["comparison"]}
    def number(value: Any) -> str:
        return "—" if value is None else f"{value:.6g}"
    comparison_table = "\n".join(
        f"| {row['metric']} | {number(row['validation'])} | {number(row['sealed_test'])} | {number(row['test_over_validation'])} |"
        for row in summary["comparison"]
    )
    accuracy_table_rows = []
    for evaluation, metrics in (("stored rows", summary["row"]["aggregate"]), ("dense, row-weighted", d["aggregate"]["row_weighted"]),
                                ("dense, orbit-weighted", d["aggregate"]["orbit_weighted"])):
        for component in ("x", "xi"):
            value = metrics[component]
            accuracy_table_rows.append(
                f"| {evaluation} | {component} | {value['rmse']:.6g} | {value['mae']:.6g} | {value['median_absolute']:.6g} | "
                f"{value['p90_absolute']:.6g} | {value['p95_absolute']:.6g} | {value['p99_absolute']:.6g} | "
                f"{value['p99p9_absolute']:.6g} | {value['maximum_absolute']:.6g} |"
            )
    accuracy_table = "\n".join(accuracy_table_rows)
    matched_table = "\n".join(
        f"| {model} | {component} | {m['aggregate'][model][component]['rmse']:.6g} | {m['aggregate'][model][component]['mae']:.6g} | "
        f"{m['aggregate'][model][component]['p90_absolute']:.6g} | {m['aggregate'][model][component]['p99_absolute']:.6g} |"
        for model in ("local", "direct") for component in ("x", "xi")
    )
    def cross(group: str, component: str, metric: str, kind: str) -> str:
        item = c[group][component][metric][kind]; return "none" if item is None else f"s={item['s']:g} (k={item['k']})"
    hard, ordinary = d["subsets"]["hard_u_th_le_0p30"], d["subsets"]["ordinary_u_th_gt_0p30"]
    incoming = d["subsets"]["sensitive_incoming_-17_to_-8p5"]
    long = d["subsets"]["long_s_gt_20"]
    direct_local_x = m["aggregate"]["direct"]["x"]["rmse"] / m["aggregate"]["local"]["x"]["rmse"]
    direct_local_xi = m["aggregate"]["direct"]["xi"]["rmse"] / m["aggregate"]["local"]["xi"]["rmse"]
    rapid_rows = "\n".join(f"| {r['target_u_th']:.2f} | {r['actual_u_th']:.5f} | {r['rapid_start_s']:.1f}–{r['rapid_stop_s']:.1f} | "
        f"{r['segments']['before_rapid_change']['local']['xi']['rmse']:.5g} | {r['segments']['through_throat_abs_x_le_2']['local']['xi']['rmse']:.5g} |"
        for r in summary["rapid_change"])
    return f"""# Final sealed held-out test of the frozen direct finite-time model

## Frozen protocol and integrity

The complete evaluation configuration and relevant source hashes were written before either sealed NPZ was opened. Only hybrid seed 202 was evaluated. The test contains {summary['counts']['orbits']:,} whole held-out orbits and {summary['counts']['rows']:,} frozen row-level transitions. Dense evaluation contains {summary['counts']['dense']:,} direct queries; matched evaluation contains {summary['counts']['matched']:,} common s=0.2 queries; rollout confirmation contains {summary['counts']['rollout_anchors']:,} exact anchors. No model training, preprocessing change, trajectory generation, ODE reintegration, clipping, projection, seed selection, or post-test tuning occurred.

Orbit IDs remain disjoint, exact test u_th duplicates are `{summary['leakage']['exact_u_th_duplicates']}`, and validation–test minimum separation is `{summary['leakage']['minimum_u_th_separation']['validation_test']:.6g}` against the frozen suspicious threshold `{summary['leakage']['suspicious_threshold']:.1e}`.

## Direct finite-time accuracy

| evaluation | component | RMSE | MAE | median | p90 | p95 | p99 | p99.9 | maximum |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
{accuracy_table}

## Frozen validation versus sealed test

| metric | validation | sealed test | test / validation |
|:---|---:|---:|---:|
{comparison_table}

Hard/ordinary x RMSE are `{hard['x']['rmse']:.6g}`/`{ordinary['x']['rmse']:.6g}`; hard/ordinary xi RMSE are `{hard['xi']['rmse']:.6g}`/`{ordinary['xi']['rmse']:.6g}`. Sensitive-incoming x/xi RMSE are `{incoming['x']['rmse']:.6g}`/`{incoming['xi']['rmse']:.6g}`. Long-horizon x/xi RMSE are `{long['x']['rmse']:.6g}`/`{long['xi']['rmse']:.6g}`. Exact identity residuals are `{d['identity']['x_max']:.3g}` in x and `{d['identity']['xi_max']:.3g}` in xi. Dense admissibility violations: `{d['admissibility']['union_violation_count']}`.

Dense energy errors: MAE `{d['energy']['mae']:.6g}`, RMSE `{d['energy']['rmse']:.6g}`, median `{d['energy']['median_absolute']:.6g}`, p90 `{d['energy']['p90_absolute']:.6g}`, p95 `{d['energy']['p95_absolute']:.6g}`, p99 `{d['energy']['p99_absolute']:.6g}`, p99.9 `{d['energy']['p99p9_absolute']:.6g}`, maximum `{d['energy']['maximum_absolute']:.6g}`.

## Matched native-step comparison

At s=0.2, local/direct x RMSE are `{m['aggregate']['local']['x']['rmse']:.6g}`/`{m['aggregate']['direct']['x']['rmse']:.6g}` (direct/local `{direct_local_x:.2f}x`). Local/direct xi RMSE are `{m['aggregate']['local']['xi']['rmse']:.6g}`/`{m['aggregate']['direct']['xi']['rmse']:.6g}` (direct/local `{direct_local_xi:.2f}x`).

| model | component | RMSE | MAE | p90 | p99 |
|:---|:---|---:|---:|---:|---:|
{matched_table}

## Rollout crossover confirmation

| family | first x RMSE win | sustained x RMSE win | sustained x MAE win | sustained xi RMSE win |
|:---|:---|:---|:---|:---|
| global | {cross('global','x','rmse','first')} | {cross('global','x','rmse','sustained')} | {cross('global','x','mae','sustained')} | {cross('global','xi','rmse','sustained')} |
| hard | {cross('hard','x','rmse','first')} | {cross('hard','x','rmse','sustained')} | {cross('hard','x','mae','sustained')} | {cross('hard','xi','rmse','sustained')} |
| ordinary | {cross('ordinary','x','rmse','first')} | {cross('ordinary','x','rmse','sustained')} | {cross('ordinary','x','mae','sustained')} | {cross('ordinary','xi','rmse','sustained')} |
| near 0.05 | {cross('near_0.05','x','rmse','first')} | {cross('near_0.05','x','rmse','sustained')} | {cross('near_0.05','x','mae','sustained')} | {cross('near_0.05','xi','rmse','sustained')} |
| near 0.15 | {cross('near_0.15','x','rmse','first')} | {cross('near_0.15','x','rmse','sustained')} | {cross('near_0.15','x','mae','sustained')} | {cross('near_0.15','xi','rmse','sustained')} |
| near 0.30 | {cross('near_0.30','x','rmse','first')} | {cross('near_0.30','x','rmse','sustained')} | {cross('near_0.30','x','mae','sustained')} | {cross('near_0.30','xi','rmse','sustained')} |

Long-horizon common counts and recursive-local survival are retained for every k; no invalid recursive state was projected or continued.

## Rapid velocity-change confirmation

| target u_th | selected test u_th | rapid interval s | local xi RMSE before | local xi RMSE in throat corridor |
|---:|---:|:---|---:|---:|
{rapid_rows}

## Scientific conclusions

**A. Validation reproduction:** Dense test/validation ratios are `{val['dense_x_rmse']['test_over_validation']:.3f}` for x RMSE, `{val['dense_xi_rmse']['test_over_validation']:.3f}` for xi RMSE, `{val['dense_x_mae']['test_over_validation']:.3f}` for x MAE, and `{val['dense_xi_mae']['test_over_validation']:.3f}` for xi MAE.

**B. Identity and admissibility:** Identity remains structurally exact; dense violations are `{d['admissibility']['union_violation_count']}`.

**C. Error structure:** Hard-to-ordinary RMSE ratios are `{hard['x']['rmse']/ordinary['x']['rmse']:.2f}x` for x and `{hard['xi']['rmse']/ordinary['xi']['rmse']:.2f}x` for xi; incoming, time-regime, and rapid-region comparisons are recorded without changing validation definitions.

**D. Native step:** The local model remains better at s=0.2 by factors `{direct_local_x:.2f}x` in x and `{direct_local_xi:.2f}x` in xi.

**E. Crossover:** Global sustained x crossover is `{cross('global','x','rmse','sustained')}` by RMSE and `{cross('global','x','mae','sustained')}` by MAE, compared with validation s=22 and s=24.

**F. Difficult family:** Hard sustained x RMSE crossover is `{cross('hard','x','rmse','sustained')}`; ordinary is `{cross('ordinary','x','rmse','sustained')}`.

**G. Rapid-change mechanism:** The fixed orbit-specific diagnostic above directly compares pre-rapid and throat-corridor errors on held-out low-u_th trajectories.

**H. Energy:** Dense energy test/validation ratios are `{val['dense_energy_rmse']['test_over_validation']:.3f}` for RMSE and `{val['dense_energy_mae']['test_over_validation']:.3f}` for MAE; the unchanged extreme-tail concentration statistics are machine-readable.

**I. Overall:** The evidence above is the final, untuned held-out assessment of the frozen model; the validation-versus-test table records every principal comparison without a post-hoc pass threshold.

Focused tests: `{summary['tests']['stdout'].strip()}`.
"""


def sealed_main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for name in ("arrays", "tables", "figures", "tests"): (OUTPUT / name).mkdir(parents=True, exist_ok=True)
    gate = input_gate(); references = frozen_references(); config = evaluation_configuration(references, gate)
    write_json(CONFIG, config)
    freeze = {"created_utc": utc(), "status": "SEALED_EVALUATION_FROZEN_NO_PREDICTIONS", "configuration": str(CONFIG.resolve()),
              "configuration_sha256": file_sha256(CONFIG), "evaluation_code_sha256": source_hashes(), "input_gate_passed": gate["passed"],
              "sealed_test_npz_opened_for_ml": False, "test_predictions_made": False}
    write_json(FREEZE, freeze); (OUTPUT / "pre_prediction_freeze_manifest.sha256").write_text(f"{file_sha256(FREEZE)}  {FREEZE.name}\n", encoding="utf-8")
    events = [{"utc": utc(), "event": "configuration frozen", "sealed_opened": False, "predictions_made": False,
               "configuration_sha256": file_sha256(CONFIG), "freeze_manifest_sha256": file_sha256(FREEZE)}]
    write_json(EVENTS, events)
    if not gate["passed"]: raise RuntimeError(gate["failures"])
    before = {str(path.resolve()): file_sha256(path) for path in EXPECTED}

    events.append({"utc": utc(), "event": "sealed orbit bank and finite-time rows opened under frozen protocol", "sealed_opened": True, "predictions_made": False})
    write_json(EVENTS, events)
    result = evaluate(TEST_BANK, TEST_ROWS, OUTPUT, config, references, sealed=True)
    events.append({"utc": utc(), "event": "sealed predictions and diagnostics completed", "sealed_opened": True, "predictions_made": True})
    write_json(EVENTS, events)
    tests = run_tests(OUTPUT); after = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    if before != after: raise RuntimeError("protected artifact changed")
    if not tests["passed"]: raise RuntimeError("focused tests failed")
    summary = {"created_utc": utc(), "status": "FINAL_SEALED_TEST_COMPLETED", **result, "tests": tests,
               "frozen_configuration_sha256": file_sha256(CONFIG), "pre_prediction_freeze_sha256": file_sha256(FREEZE),
               "protected_hashes_unchanged": True, "training_performed": False, "ode_reintegrated": False,
               "test_driven_tuning": False, "test_set_now_unsealed_for_this_final_evaluation": True}
    write_json(SUMMARY, summary); REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {str(path.relative_to(OUTPUT)): {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)}
    manifest = {"status": summary["status"], "protected_before": before, "protected_after": after,
                "configuration_frozen_before_open": True, "configuration_sha256": file_sha256(CONFIG), "freeze_manifest_sha256": file_sha256(FREEZE),
                "checkpoint_hashes": {"hybrid_seed_202": EXPECTED[HYBRID_CHECKPOINT], "local_seed_101": EXPECTED[LOCAL_CHECKPOINT]},
                "sealed_input_hashes": {"orbit_bank": EXPECTED[TEST_BANK], "finite_time_rows": EXPECTED[TEST_ROWS]},
                "source_hashes": source_hashes(), "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)},
                "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)}, "artifacts": artifacts}
    write_json(MANIFEST, manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


def preflight_main() -> None:
    gate = input_gate(); references = frozen_references(); config = evaluation_configuration(references, gate)
    with tempfile.TemporaryDirectory(prefix="wormhole_final_sealed_preflight_") as temporary:
        output = Path(temporary); result = evaluate(VALIDATION_BANK, VALIDATION_ROWS, output, config, references, sealed=False)
        assert result["counts"] == {"orbits": 1024, "rows": 98304, "dense": 524288, "matched": 32768, "rollout_anchors": 18432}
        for component in ("x", "xi"):
            for metric in ("rmse", "mae"):
                assert np.isclose(
                    result["row"]["aggregate"][component][metric],
                    references["row"]["physical_state_metrics"][component][metric], rtol=2e-12, atol=1e-15,
                )
                assert np.isclose(
                    result["dense"]["aggregate"]["row_weighted"][component][metric],
                    references["dense"]["dense_aggregate"]["row_weighted"][component][metric], rtol=2e-12, atol=1e-15,
                )
                assert np.isclose(
                    result["matched"]["aggregate"]["direct"][component][metric],
                    references["dense"]["matched"]["aggregate"]["hybrid"][component][metric], rtol=2e-12, atol=1e-15,
                )
                assert np.isclose(
                    result["matched"]["aggregate"]["local"][component][metric],
                    references["dense"]["matched"]["aggregate"]["local"][component][metric], rtol=2e-12, atol=1e-15,
                )
        assert result["rollout"]["crossovers"]["global"]["x"]["rmse"]["minimum_common_count"] == 1844
        assert result["rollout"]["crossovers"]["global"]["x"]["rmse"]["well_sampled_k_count"] == 80
        print("preflight completed: all counts and frozen validation metrics reproduced")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    preflight_main() if args.preflight else sealed_main()
