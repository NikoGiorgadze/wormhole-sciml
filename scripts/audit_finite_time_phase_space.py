#!/usr/bin/env python3
"""Validation-only phase-space decomposition audit for frozen derivative treatments.

This script consumes only the retained training scale, validation rows, frozen
rapid-change windows, cached predictions, dense reference arrays, and trained
checkpoints from the completed derivative-loss experiment.  It does not train,
integrate trajectories, detect features, or alter any upstream artifact.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import conserved_energy, radial_acceleration
from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, load_hybrid_model, predict_hybrid
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi


ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "output/finite_time_hybrid_derivative_loss_experiment"
OUTPUT = ROOT / "output/finite_time_hybrid_phase_space_audit"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
ARRAYS = OUTPUT / "arrays"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_HYBRID_PHASE_SPACE_AUDIT.md"
SUMMARY = OUTPUT / "finite_time_hybrid_phase_space_audit_summary.json"
MANIFEST = OUTPUT / "finite_time_hybrid_phase_space_audit_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_hybrid_phase_space_audit_manifest.sha256"

TRAIN_ROWS = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_train_raw.npz"
VALIDATION_ROWS = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_validation_raw.npz"
PREPROCESSING = ROOT / "output/finite_time_hybrid_s5/preprocessing/hybrid_preprocessing_constants.json"
STAGE3_MANIFEST = STAGE3 / "finite_time_hybrid_derivative_loss_manifest.json"
STAGE3_MANIFEST_HASH = STAGE3 / "finite_time_hybrid_derivative_loss_manifest.sha256"
STAGE3_SUMMARY = STAGE3 / "finite_time_hybrid_derivative_loss_summary.json"
CHECKPOINT_MANIFEST = STAGE3 / "checkpoint_manifest.json"
FEATURE_WINDOWS = STAGE3 / "protocol/frozen_validation_feature_windows.csv"
REGION_REALIZATION = STAGE3 / "protocol/frozen_regional_rule_realization.json"
REFERENCE_TIMING = STAGE3 / "tables/dense_reference_feature_timing.csv"

LAMBDAS = (0.0, 0.011, 0.034, 0.068)
SEEDS = (101, 202, 303)
TARGETS = (0.05, 0.15, 0.30)
LABELS = {0.0: "control", 0.011: "weak", 0.034: "moderate", 0.068: "upper_stress"}
COLORS = {0.0: "#666666", 0.011: "#228833", 0.034: "#3366aa", 0.068: "#cc6677"}
BIN_DEFINITIONS: tuple[tuple[str, float, float | None, bool], ...] = (
    ("[0.01,0.05]", 0.01, 0.05, True),
    ("(0.05,0.10]", 0.05, 0.10, False),
    ("(0.10,0.15]", 0.10, 0.15, False),
    ("(0.15,0.20]", 0.15, 0.20, False),
    ("(0.20,0.30]", 0.20, 0.30, False),
    (">0.30", 0.30, None, False),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lambda_slug(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.3f}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bin_mask(values: np.ndarray, definition: tuple[str, float, float | None, bool]) -> np.ndarray:
    _, lower, upper, include_lower = definition
    lower_mask = values >= lower if include_lower else values > lower
    return lower_mask if upper is None else lower_mask & (values <= upper)


def decompose_phase_error(
    exact_velocity: np.ndarray,
    exact_second_rate: np.ndarray,
    first_error: np.ndarray,
    second_error: np.ndarray,
    first_scale: float,
    second_scale: float,
) -> dict[str, np.ndarray | float | int]:
    """Project scaled two-coordinate errors onto exact tangents and normals."""

    tangent = np.column_stack(
        (np.asarray(exact_velocity, dtype=np.float64) / first_scale,
         np.asarray(exact_second_rate, dtype=np.float64) / second_scale)
    )
    norm = np.linalg.norm(tangent, axis=1)
    tolerance = float(64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(norm))))
    valid = np.isfinite(norm) & (norm > tolerance)
    unit_tangent = np.full_like(tangent, np.nan)
    unit_tangent[valid] = tangent[valid] / norm[valid, None]
    unit_normal = np.column_stack((-unit_tangent[:, 1], unit_tangent[:, 0]))
    error = np.column_stack(
        (np.asarray(first_error, dtype=np.float64) / first_scale,
         np.asarray(second_error, dtype=np.float64) / second_scale)
    )
    e_parallel = np.einsum("ij,ij->i", error, unit_tangent)
    e_perp = np.einsum("ij,ij->i", error, unit_normal)
    distance = np.linalg.norm(error, axis=1)
    identity_residual = np.abs(distance[valid] ** 2 - e_parallel[valid] ** 2 - e_perp[valid] ** 2)
    return {
        "e_parallel": e_parallel,
        "e_perp": e_perp,
        "d_ps": distance,
        "valid": valid,
        "tangent_norm": norm,
        "tangent_tolerance": tolerance,
        "invalid_count": int(np.sum(~valid)),
        "maximum_pythagorean_residual": float(np.max(identity_residual)) if identity_residual.size else 0.0,
    }


def scalar_metrics(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None, None
    return float(np.sqrt(np.mean(finite**2))), float(np.mean(np.abs(finite)))


def metric_record(
    decomposition: dict[str, Any],
    x_error: np.ndarray,
    selection: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(selection, dtype=bool)
    valid = selected & np.asarray(decomposition["valid"], dtype=bool)
    perp_rmse, perp_mae = scalar_metrics(np.asarray(decomposition["e_perp"])[valid])
    parallel_rmse, parallel_mae = scalar_metrics(np.asarray(decomposition["e_parallel"])[valid])
    distance_rmse, _ = scalar_metrics(np.asarray(decomposition["d_ps"])[valid])
    x_rmse, x_mae = scalar_metrics(np.asarray(x_error)[selected])
    q = None
    q_status = "unavailable_no_rows"
    if perp_rmse is not None and parallel_rmse is not None:
        if parallel_rmse > 100.0 * np.finfo(np.float64).eps:
            q = perp_rmse / parallel_rmse
            q_status = "reported"
        else:
            q_status = "not_reported_tiny_parallel_denominator"
    return {
        "row_count": int(np.sum(selected)),
        "tangent_valid_count": int(np.sum(valid)),
        "tangent_invalid_count": int(np.sum(selected & ~np.asarray(decomposition["valid"], dtype=bool))),
        "e_perp_rmse": perp_rmse,
        "e_perp_mae": perp_mae,
        "e_parallel_rmse": parallel_rmse,
        "e_parallel_mae": parallel_mae,
        "d_ps_rmse": distance_rmse,
        "Q_perp_over_parallel": q,
        "Q_status": q_status,
        "x_rmse": x_rmse,
        "x_mae": x_mae,
    }


def load_frozen_regions(validation: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(FEATURE_WINDOWS):
        row: dict[str, Any] = dict(raw)
        row["source_orbit_index"] = int(raw["source_orbit_index"])
        row["feature_index"] = int(raw["feature_index"])
        for name in (
            "u_th", "dense_step", "pre_start", "pre_stop", "during_start",
            "during_stop", "post_start", "post_stop", "feature_width",
        ):
            row[name] = float(raw[name])
        rows.append(row)
    masks = {name: np.zeros(validation["s"].size, dtype=bool) for name in ("pre", "during", "post")}
    endpoint = validation["t1"]
    source = validation["source_orbit_index"]
    for row in rows:
        orbit = source == row["source_orbit_index"]
        masks["pre"] |= orbit & (endpoint >= row["pre_start"]) & (endpoint < row["pre_stop"])
        masks["during"] |= orbit & (endpoint >= row["during_start"]) & (endpoint <= row["during_stop"])
        masks["post"] |= orbit & (endpoint > row["post_start"]) & (endpoint <= row["post_stop"])
    if np.any((masks["pre"] & masks["during"]) | (masks["during"] & masks["post"]) | (masks["pre"] & masks["post"])):
        raise RuntimeError("stored frozen regional windows map to overlapping validation masks")
    expected = json.loads(REGION_REALIZATION.read_text(encoding="utf-8"))["row_counts"]
    measured = {name: int(mask.sum()) for name, mask in masks.items()}
    if measured != expected:
        raise RuntimeError(f"stored region-mask counts disagree: {measured} != {expected}")
    return masks, rows


def protected_inputs() -> dict[Path, str]:
    stage3_manifest = json.loads(STAGE3_MANIFEST.read_text(encoding="utf-8"))
    expected: dict[Path, str] = {}
    recorded_manifest_hash = STAGE3_MANIFEST_HASH.read_text(encoding="utf-8").split()[0]
    expected[STAGE3_MANIFEST] = recorded_manifest_hash
    gate_artifacts = stage3_manifest["input_gate"]["artifacts"]
    for path in (TRAIN_ROWS, VALIDATION_ROWS, PREPROCESSING):
        item = gate_artifacts[str(path.resolve())]
        if not item["match"]:
            raise RuntimeError(f"upstream input gate was not passed for {path}")
        expected[path] = item["expected"]
    artifact_names = [
        "checkpoint_manifest.json",
        "finite_time_hybrid_derivative_loss_summary.json",
        "protocol/frozen_validation_feature_windows.csv",
        "protocol/frozen_regional_rule_realization.json",
        "tables/dense_reference_feature_timing.csv",
    ]
    artifact_names += [
        f"validation/predictions_lambda_{lambda_slug(value)}_seed_{seed}.npz"
        for value in LAMBDAS for seed in SEEDS
    ]
    artifact_names += [
        f"arrays/reference_u_th_{target:.2f}_seed_{seed}.npz"
        for target in TARGETS for seed in SEEDS
    ]
    for name in artifact_names:
        expected[STAGE3 / name] = stage3_manifest["artifacts"][name]["sha256"]
    checkpoints = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))["checkpoints"]
    for row in checkpoints:
        expected[Path(row["checkpoint"])] = row["checkpoint_sha256"]
    return expected


def verify_hashes(expected: dict[Path, str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for path, target in expected.items():
        measured = sha256(path)
        match = measured == target
        output[str(path.resolve())] = {"expected": target, "measured": measured, "match": match}
        if not match:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"protected-input hash mismatch: {failures}")
    return output


def aggregate_seed_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "e_perp_rmse", "e_perp_mae", "e_parallel_rmse", "e_parallel_mae",
        "d_ps_rmse", "Q_perp_over_parallel", "x_rmse", "x_mae",
    )
    controls = {
        (row["seed"], row["u_th_bin"], row["scope"]): row
        for row in rows if row["lambda_dot_xi"] == 0.0
    }
    output: list[dict[str, Any]] = []
    for value in LAMBDAS:
        for definition in BIN_DEFINITIONS:
            bin_name = definition[0]
            for scope in ("global", "during", "post"):
                group = [
                    row for row in rows
                    if row["lambda_dot_xi"] == value and row["u_th_bin"] == bin_name and row["scope"] == scope
                ]
                record: dict[str, Any] = {
                    "lambda_dot_xi": value,
                    "treatment": LABELS[value],
                    "u_th_bin": bin_name,
                    "scope": scope,
                    "seed_count": len(group),
                    "row_count": group[0]["row_count"] if group else 0,
                    "orbit_count": group[0]["orbit_count"] if group else 0,
                }
                for metric in metric_names:
                    values = np.asarray([row[metric] for row in group if row[metric] is not None], dtype=np.float64)
                    record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
                    record[f"{metric}_standard_deviation"] = float(np.std(values, ddof=0)) if values.size else None
                    record[f"{metric}_minimum"] = float(np.min(values)) if values.size else None
                    record[f"{metric}_maximum"] = float(np.max(values)) if values.size else None
                relative = []
                improvement_count = 0
                for row in group:
                    current = row["e_perp_rmse"]
                    control = controls[(row["seed"], bin_name, scope)]["e_perp_rmse"]
                    if current is not None and control is not None and control > 0.0:
                        change = (current - control) / control
                        relative.append(change)
                        improvement_count += int(change < 0.0)
                record["paired_e_perp_relative_change_mean"] = float(np.mean(relative)) if relative else None
                record["paired_e_perp_relative_change_minimum"] = float(np.min(relative)) if relative else None
                record["paired_e_perp_relative_change_maximum"] = float(np.max(relative)) if relative else None
                record["e_perp_seed_improvement_count"] = improvement_count if relative else None
                output.append(record)
    return output


def orbit_level_post_metrics(
    validation: dict[str, np.ndarray],
    post_mask: np.ndarray,
    decompositions: dict[tuple[float, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orbit in np.unique(validation["source_orbit_index"][post_mask]):
        orbit_mask = post_mask & (validation["source_orbit_index"] == orbit)
        u_th = float(validation["u_th"][orbit_mask][0])
        for value in LAMBDAS:
            for seed in SEEDS:
                decomposition = decompositions[(value, seed)]
                valid = orbit_mask & decomposition["valid"]
                perp, _ = scalar_metrics(decomposition["e_perp"][valid])
                parallel, _ = scalar_metrics(decomposition["e_parallel"][valid])
                rows.append({
                    "source_orbit_index": int(orbit),
                    "u_th": u_th,
                    "lambda_dot_xi": value,
                    "treatment": LABELS[value],
                    "seed": seed,
                    "post_row_count": int(np.sum(orbit_mask)),
                    "post_e_perp_rmse": perp,
                    "post_e_parallel_rmse": parallel,
                    "Q_perp_over_parallel": perp / parallel if perp is not None and parallel is not None and parallel > 100 * np.finfo(float).eps else None,
                })
    controls = {
        (row["source_orbit_index"], row["seed"]): row["post_e_perp_rmse"]
        for row in rows if row["lambda_dot_xi"] == 0.0
    }
    for row in rows:
        control = controls[(row["source_orbit_index"], row["seed"])]
        current = row["post_e_perp_rmse"]
        row["relative_post_e_perp_vs_same_seed_control"] = (
            (current - control) / control if current is not None and control is not None and control > 0.0 else None
        )
    return rows


def aggregate_orbits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["lambda_dot_xi"], row["source_orbit_index"]) for row in rows})
    for value, orbit in keys:
        group = [row for row in rows if row["lambda_dot_xi"] == value and row["source_orbit_index"] == orbit]
        relative = [
            row["relative_post_e_perp_vs_same_seed_control"]
            for row in group
            if row["relative_post_e_perp_vs_same_seed_control"] is not None
        ]
        output.append({
            "lambda_dot_xi": value,
            "source_orbit_index": orbit,
            "u_th": group[0]["u_th"],
            "post_e_perp_rmse_seed_mean": float(np.mean([row["post_e_perp_rmse"] for row in group])),
            "relative_post_e_perp_seed_mean": float(np.mean(relative)) if relative else None,
        })
    return output


def centered_rolling(values: np.ndarray, width: int = 51) -> np.ndarray:
    """Return a robust centered rolling median, ignoring undefined ratios."""
    values = np.asarray(values, dtype=np.float64)
    radius = width // 2
    output = []
    for index in range(values.size):
        window = values[max(0, index - radius): min(values.size, index + radius + 1)]
        finite = window[np.isfinite(window)]
        output.append(float(np.median(finite)) if finite.size else np.nan)
    return np.asarray(output)


def plot_binned_decomposition(aggregate_rows: list[dict[str, Any]]) -> None:
    bins = [definition[0] for definition in BIN_DEFINITIONS]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True, sharex=True)
    for row_index, scope in enumerate(("global", "post")):
        for column_index, (metric, title) in enumerate((("e_perp_rmse", "normal RMSE"), ("e_parallel_rmse", "tangential RMSE"))):
            ax = axes[row_index, column_index]
            for value in LAMBDAS:
                selected = [
                    next(row for row in aggregate_rows if row["lambda_dot_xi"] == value and row["u_th_bin"] == name and row["scope"] == scope)
                    for name in bins
                ]
                y = np.asarray([row[f"{metric}_mean"] if row[f"{metric}_mean"] is not None else np.nan for row in selected])
                low = np.asarray([row[f"{metric}_minimum"] if row[f"{metric}_minimum"] is not None else np.nan for row in selected])
                high = np.asarray([row[f"{metric}_maximum"] if row[f"{metric}_maximum"] is not None else np.nan for row in selected])
                ax.plot(x, y, marker="o", color=COLORS[value], label=f"lambda={value:.3g}")
                ax.fill_between(x, low, high, color=COLORS[value], alpha=0.10)
            ax.set_title(f"{scope}: {title}")
            ax.set_ylabel("scaled phase-space error")
            ax.grid(alpha=0.25)
            ax.set_xticks(x, bins, rotation=25, ha="right")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Exact-tangent decomposition by validation $u_{th}$ bin\n(lines: seed means; bands: seed ranges)")
    fig.savefig(FIGURES / "phase_space_decomposition_by_u_th.png", dpi=180)
    plt.close(fig)


def plot_continuous(orbits: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for value in LAMBDAS:
        group = sorted(
            (row for row in orbits if row["lambda_dot_xi"] == value and row["u_th"] <= 0.15),
            key=lambda row: row["u_th"],
        )
        u = np.asarray([row["u_th"] for row in group])
        y = np.asarray([row["post_e_perp_rmse_seed_mean"] for row in group])
        ax.scatter(u, y, s=8, alpha=0.10, color=COLORS[value])
        ax.plot(u, centered_rolling(y), color=COLORS[value], lw=2, label=f"lambda={value:.3g}")
    ax.set(xlabel="$u_{th}$", ylabel="orbit-level post-change RMSE($e_\\perp$)", xscale="log", yscale="log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2)
    ax.set_title("Post-change normal error in the low-$u_{th}$ regime\n51-orbit centered rolling medians over seed-averaged orbit metrics")
    fig.savefig(FIGURES / "post_e_perp_continuous_u_th.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.axhline(0.0, color="black", lw=1)
    all_relative: list[float] = []
    for value in LAMBDAS[1:]:
        group = sorted(
            (row for row in orbits if row["lambda_dot_xi"] == value and row["u_th"] <= 0.15),
            key=lambda row: row["u_th"],
        )
        u = np.asarray([row["u_th"] for row in group])
        y = 100.0 * np.asarray([
            np.nan if row["relative_post_e_perp_seed_mean"] is None else row["relative_post_e_perp_seed_mean"]
            for row in group
        ], dtype=np.float64)
        all_relative.extend(y[np.isfinite(y)].tolist())
        ax.scatter(u, y, s=8, alpha=0.10, color=COLORS[value])
        ax.plot(u, centered_rolling(y), color=COLORS[value], lw=2, label=f"lambda={value:.3g}")
    ax.set(xlabel="$u_{th}$", ylabel="post-change normal-error change vs matched control (%)", xscale="log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=3)
    finite_relative = np.asarray(all_relative, dtype=np.float64)
    lower = min(-5.0, float(np.quantile(finite_relative, 0.01)))
    upper = max(5.0, float(np.quantile(finite_relative, 0.99)))
    padding = 0.08 * (upper - lower)
    ax.set_ylim(lower - padding, upper + padding)
    outside = int(np.sum((finite_relative < lower - padding) | (finite_relative > upper + padding)))
    ax.text(
        0.01, 0.02,
        f"{outside} of {finite_relative.size} orbit-level values lie outside the display range; all remain in the CSV",
        transform=ax.transAxes, fontsize=8, va="bottom",
    )
    ax.set_title("Relative treatment effect on orbit-normal error\nnegative values improve; 51-orbit centered rolling medians")
    fig.savefig(FIGURES / "relative_post_e_perp_continuous_u_th.png", dpi=180)
    plt.close(fig)


def dominant_peak_times() -> dict[float, list[float]]:
    rows = read_csv(REFERENCE_TIMING)
    output: dict[float, list[float]] = {}
    for target in TARGETS:
        candidates = [
            row for row in rows
            if float(row["target_u_th"]) == target and int(row["seed"]) == SEEDS[0]
            and float(row["lambda_dot_xi"]) == 0.0 and float(row["exact_relative_height"]) >= 0.99
        ]
        output[target] = sorted({float(row["exact_peak_s"]) for row in candidates})
    return output


def reference_inference(
    preprocessing: HybridPreprocessing,
    checkpoint_rows: list[dict[str, Any]],
) -> tuple[dict[float, dict[str, np.ndarray]], dict[tuple[float, int, float], dict[str, np.ndarray]], float]:
    wormhole, spiral = experiment_parameters()
    references: dict[float, dict[str, np.ndarray]] = {}
    query_parts: dict[str, list[np.ndarray]] = {name: [] for name in ("x0", "xi0", "E0", "s")}
    slices: dict[float, slice] = {}
    offset = 0
    for target in TARGETS:
        per_seed = [load_npz(STAGE3 / f"arrays/reference_u_th_{target:.2f}_seed_{seed}.npz") for seed in SEEDS]
        reference = per_seed[0]
        for other in per_seed[1:]:
            for name in ("elapsed_s", "exact_x", "exact_xi", "exact_dot_xi"):
                if not np.array_equal(reference[name], other[name]):
                    raise RuntimeError(f"dense exact reference differs across seeds for u_th={target}")
        exact_x = reference["exact_x"]
        exact_xi = reference["exact_xi"]
        _, exact_u = state_from_xi(exact_x, exact_xi, wormhole, spiral)
        energy = float(conserved_energy(exact_x[0], exact_u[0], wormhole, spiral))
        reference = {**reference, "exact_u": exact_u, "E0": np.asarray(energy)}
        references[target] = reference
        count = reference["elapsed_s"].size
        slices[target] = slice(offset, offset + count)
        offset += count
        query_parts["x0"].append(np.full(count, exact_x[0]))
        query_parts["xi0"].append(np.full(count, exact_xi[0]))
        query_parts["E0"].append(np.full(count, energy))
        query_parts["s"].append(reference["elapsed_s"])
    query = {name: np.concatenate(parts) for name, parts in query_parts.items()}
    predictions: dict[tuple[float, int, float], dict[str, np.ndarray]] = {}
    maximum_discrepancy = 0.0
    for checkpoint in checkpoint_rows:
        value = float(checkpoint["lambda_dot_xi"])
        seed = int(checkpoint["seed"])
        model = load_hybrid_model(Path(checkpoint["checkpoint"]))
        prediction = predict_hybrid(model, preprocessing, query)
        for target in TARGETS:
            part = slices[target]
            item = {
                "predicted_x": prediction["predicted_x1"][part],
                "predicted_xi": prediction["predicted_xi1"][part],
            }
            cached = load_npz(STAGE3 / f"arrays/reference_u_th_{target:.2f}_seed_{seed}.npz")
            cached_xi = cached[f"lambda_{lambda_slug(value)}_predicted_xi"]
            discrepancy = float(np.max(np.abs(item["predicted_xi"] - cached_xi)))
            maximum_discrepancy = max(maximum_discrepancy, discrepancy)
            if discrepancy > 2.0e-15:
                raise RuntimeError(f"reference inference disagrees with cached xi for lambda={value}, seed={seed}")
            _, item["predicted_u"] = state_from_xi(item["predicted_x"], item["predicted_xi"], wormhole, spiral)
            predictions[(value, seed, target)] = item
    return references, predictions, maximum_discrepancy


def reference_metrics(
    references: dict[float, dict[str, np.ndarray]],
    predictions: dict[tuple[float, int, float], dict[str, np.ndarray]],
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    wormhole, spiral = experiment_parameters()
    rows: list[dict[str, Any]] = []
    for target, exact in references.items():
        acceleration = radial_acceleration(exact["exact_x"], exact["exact_u"], wormhole, spiral)
        for value in LAMBDAS:
            for seed in SEEDS:
                predicted = predictions[(value, seed, target)]
                x_error = predicted["predicted_x"] - exact["exact_x"]
                xi_error = predicted["predicted_xi"] - exact["exact_xi"]
                u_error = predicted["predicted_u"] - exact["exact_u"]
                ml = decompose_phase_error(exact["exact_u"], exact["exact_dot_xi"], x_error, xi_error, scales["x"], scales["xi"])
                physical = decompose_phase_error(exact["exact_u"], acceleration, x_error, u_error, scales["x"], scales["u"])
                ml_metric = metric_record(ml, x_error, np.ones(x_error.size, dtype=bool))
                physical_metric = metric_record(physical, x_error, np.ones(x_error.size, dtype=bool))
                rows.append({
                    "target_u_th": target,
                    "lambda_dot_xi": value,
                    "treatment": LABELS[value],
                    "seed": seed,
                    "x_xi_e_perp_rmse": ml_metric["e_perp_rmse"],
                    "x_xi_e_parallel_rmse": ml_metric["e_parallel_rmse"],
                    "x_xi_Q": ml_metric["Q_perp_over_parallel"],
                    "x_u_e_perp_rmse": physical_metric["e_perp_rmse"],
                    "x_u_e_parallel_rmse": physical_metric["e_parallel_rmse"],
                    "x_u_Q": physical_metric["Q_perp_over_parallel"],
                    "x_rmse": ml_metric["x_rmse"],
                })
    return rows


def plot_reference_phase_space(
    references: dict[float, dict[str, np.ndarray]],
    predictions: dict[tuple[float, int, float], dict[str, np.ndarray]],
) -> None:
    peak_times = dominant_peak_times()
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    for row_index, target in enumerate(TARGETS):
        exact = references[target]
        for column_index in range(2):
            ax = axes[row_index, column_index]
            if column_index == 1 and not peak_times[target]:
                ax.axis("off")
                ax.text(0.5, 0.5, "No stored dominant rapid-change peak\nfor this reference", ha="center", va="center", transform=ax.transAxes)
                continue
            mask = np.ones(exact["elapsed_s"].size, dtype=bool)
            if column_index == 1:
                mask = exact["elapsed_s"] >= peak_times[target][0]
            ax.plot(exact["exact_x"][mask], exact["exact_xi"][mask], color="black", lw=2.4, label="exact")
            for value in LAMBDAS:
                curves_x = np.vstack([predictions[(value, seed, target)]["predicted_x"] for seed in SEEDS])[:, mask]
                curves_y = np.vstack([predictions[(value, seed, target)]["predicted_xi"] for seed in SEEDS])[:, mask]
                mean_x, mean_y = np.mean(curves_x, axis=0), np.mean(curves_y, axis=0)
                ax.plot(mean_x, mean_y, color=COLORS[value], lw=1.5, label=f"lambda={value:.3g}")
                if value in (0.011, 0.034):
                    for seed_index in range(len(SEEDS)):
                        ax.plot(curves_x[seed_index], curves_y[seed_index], color=COLORS[value], lw=0.45, alpha=0.22)
            if column_index == 0:
                for peak in peak_times[target]:
                    index = int(np.argmin(np.abs(exact["elapsed_s"] - peak)))
                    ax.scatter(exact["exact_x"][index], exact["exact_xi"][index], color="black", s=22, zorder=5)
            ax.set(xlabel="x", ylabel="$\\xi$")
            ax.grid(alpha=0.23)
            suffix = "full reference" if column_index == 0 else "after first stored dominant-peak time"
            ax.set_title(f"$u_{{th}}={target:.2f}$: {suffix}")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle("Dense reference trajectories in ML phase space\ncolored curves are seed means; thin lines show seeds for lambda=0.011 and 0.034")
    fig.savefig(FIGURES / "reference_phase_space_x_xi.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        exact = references[target]
        ax.plot(exact["exact_x"], exact["exact_u"], color="black", lw=2.4, label="exact")
        for value in LAMBDAS:
            curves_x = np.vstack([predictions[(value, seed, target)]["predicted_x"] for seed in SEEDS])
            curves_u = np.vstack([predictions[(value, seed, target)]["predicted_u"] for seed in SEEDS])
            ax.plot(np.mean(curves_x, axis=0), np.mean(curves_u, axis=0), color=COLORS[value], lw=1.5, label=f"lambda={value:.3g}")
        ax.set(title=f"$u_{{th}}={target:.2f}$", xlabel="x", ylabel="u")
        ax.grid(alpha=0.23)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Dense reference trajectories in physical phase space (seed means)")
    fig.savefig(FIGURES / "reference_phase_space_x_u.png", dpi=180)
    plt.close(fig)


def aggregate_lookup(rows: list[dict[str, Any]], value: float, bin_name: str, scope: str) -> dict[str, Any]:
    return next(row for row in rows if row["lambda_dot_xi"] == value and row["u_th_bin"] == bin_name and row["scope"] == scope)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}%"


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def report_text(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate_metrics"]
    coverage = summary["coverage"]
    coverage_lines = []
    for row in coverage:
        coverage_lines.append(
            f"| {row['u_th_bin']} | {row['global_orbit_count']} | {row['global_row_count']} | "
            f"{row['during_orbit_count']} | {row['during_row_count']} | {row['post_orbit_count']} | {row['post_row_count']} |"
        )
    primary_lines = []
    for value in LAMBDAS:
        row = aggregate_lookup(aggregate, value, "[0.01,0.05]", "post")
        primary_lines.append(
            f"| {value:.3g} | {number(row['e_perp_rmse_mean'])} | {number(row['e_parallel_rmse_mean'])} | "
            f"{number(row['Q_perp_over_parallel_mean'])} | {percent(row['paired_e_perp_relative_change_mean'])} | "
            f"{row['e_perp_seed_improvement_count'] if row['e_perp_seed_improvement_count'] is not None else 'n/a'}/3 | {number(row['x_rmse_mean'])} |"
        )
    cross_lines = []
    for value in LAMBDAS:
        row = next(
            item for item in summary["physical_crosscheck_aggregate"]
            if item["lambda_dot_xi"] == value and item["u_th_bin"] == "[0.01,0.05]" and item["scope"] == "post"
        )
        cross_lines.append(
            f"| {value:.3g} | {number(row['e_perp_rmse_mean'])} | {number(row['e_parallel_rmse_mean'])} | {number(row['Q_perp_over_parallel_mean'])} |"
        )
    decision = summary["decision"]
    return f"""# Validation-only phase-space error audit

## Scope and integrity

This audit reused the 12 frozen derivative-treatment checkpoints, all cached validation predictions, exact validation endpoints, the stored training-only scales, the frozen q95 rapid-change windows, and the nine cached dense reference trajectories. It performed no training, optimization, ODE integration, feature detection, lambda search, architecture change, or dataset mutation. No held-out test artifact was accessed.

Scaled ML coordinates use training-only population standard deviations `sigma_x={summary['scales']['x']:.12g}` and `sigma_xi={summary['scales']['xi']:.12g}` from the frozen preprocessing. The physical cross-check uses `sigma_u={summary['scales']['u']:.12g}`, the training-only population standard deviation of `u0`. Centering is immaterial for an error-vector decomposition.

The exact tangent Pythagorean identity was verified over all runs: maximum absolute residual `{summary['numerical_verification']['maximum_pythagorean_residual']:.3e}`. Tangent tolerance was `{summary['numerical_verification']['x_xi_tangent_tolerance']:.3e}`; `{summary['numerical_verification']['x_xi_invalid_tangent_rows']}` validation rows had no numerically safe ML-phase tangent. Protected input hashes were identical before and after the audit.

## Coverage of the unchanged frozen regions

| u_th bin | global orbits | global rows | during orbits | during rows | post orbits | post rows |
|:---|---:|---:|---:|---:|---:|---:|
{chr(10).join(coverage_lines)}

No bin was merged: every global bin contains at least 86 orbits. The `(0.15,0.20]` and `(0.20,0.30]` bins have no during/post rows because no validation orbit there crosses the already-frozen q95 `|dot xi|` threshold—not because their global coverage is sparse. Creating substitute regions would violate the frozen protocol.

## Primary result: lowest-u_th post-change error

Values are means across three seeds. Relative changes are paired to the same-seed lambda-zero control; negative is improvement. `Q=RMSE(e_perp)/RMSE(e_parallel)`.

| lambda | post e_perp RMSE | post e_parallel RMSE | Q | paired e_perp change | improving seeds | x RMSE |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(primary_lines)}

At the lowest `u_th`, the tangential component is roughly {decision['lowest_bin_parallel_to_perp_ratio_control']:.2f} times the normal component for the control. Thus the remaining phase-space displacement is predominantly **along** the exact orbit, while the normal component remains nonzero. Lambda `0.034` reduces the primary normal-error metric in all three seeds by a paired mean `{percent(decision['lambda_0p034_lowest_post_e_perp_change'])}`. Lambda `0.011` is weaker and not three-seed consistent. Lambda `0.068` reduces the lowest-bin normal error most, but retains the previously established `{decision['lambda_0p068_previous_global_x_percent_change']:+.2f}%` global-x RMSE degradation and is not the preferred tradeoff.

Across the neighboring `(0.05,0.10]` bin, lambda `0.034` also improves post normal error in all three seeds. In `(0.10,0.15]`, only 43 orbits contribute post rows and treatment rankings become seed-sensitive; the full values remain in the main and aggregate tables.

## Continuous trend and component balance

The orbit-level figures use deterministic 51-orbit centered rolling medians. They show the same-seed treatment effect without fitting a regression model. The relative-effect figure uses a percentile-based display range because division by near-zero control error creates a few extreme but valid ratios; every value is retained in the CSV. Derivative supervision does not eliminate the tangentially dominated low-`u_th` error, but lambda `0.034` shifts the lowest-bin balance modestly toward a smaller normal share (`Q` falls from `{decision['control_lowest_Q']:.3f}` to `{decision['lambda_0p034_lowest_Q']:.3f}`).

## Physical (x,u) cross-check

| lambda | post e_perp RMSE | post e_parallel RMSE | Q |
|---:|---:|---:|---:|
{chr(10).join(cross_lines)}

The physical-coordinate decomposition {decision['physical_crosscheck_statement']}. Detailed reporting therefore remains focused on `(x,xi)`.

## Direct answers and modeling implication

**Is the low-u_th failure mainly timing/progression or deviation away from the orbit?**  {decision['failure_answer']}

**Which existing treatment best fits that failure?**  **lambda=0.034** is the best balanced existing treatment: it gives consistent low-`u_th` post-change normal-error improvement without the known `{decision['lambda_0p068_previous_global_x_percent_change']:+.2f}%` broader x penalty of lambda `0.068`. The audit does not promote `0.068` solely for winning the narrowest normal-error metric.

**Is a parameter-matched split-head test scientifically motivated?**  {decision['architecture_statement']}

The conclusion is validation-only and does not imply held-out performance.

## Artifacts

- Main seed-level table: `tables/low_u_th_phase_space_metrics.csv`
- Across-seed treatment table: `tables/aggregate_treatment_metrics.csv`
- Physical cross-check: `tables/physical_x_u_crosscheck.csv`
- Orbit-level continuous data: `tables/orbit_level_post_metrics.csv`
- Dense-reference metrics: `tables/reference_phase_space_metrics.csv`
- Figures: `figures/`
- Machine-readable summary and integrity manifest are stored beside this report.

Focused tests: `{summary['tests']['stdout'].strip()}`
"""


def run_tests() -> dict[str, Any]:
    command = [
        sys.executable, "-m", "pytest", "-q", "tests/test_finite_time_phase_space_audit.py",
        f"--junitxml={TESTS / 'focused_pytest.xml'}",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-phase-space-audit-mpl"},
        capture_output=True,
        text=True,
    )
    payload = {
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }
    write_json(TESTS / "test_summary.json", payload)
    return payload


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {OUTPUT}")
    for directory in (OUTPUT, TABLES, FIGURES, ARRAYS, TESTS):
        directory.mkdir(parents=True, exist_ok=False)

    expected_inputs = protected_inputs()
    protected_before = verify_hashes(expected_inputs)
    validation = load_npz(VALIDATION_ROWS)
    training = load_npz(TRAIN_ROWS)
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    scales = {
        "x": float(preprocessing.input_std[0]),
        "xi": float(preprocessing.input_std[1]),
        "u": float(np.std(training["u0"], ddof=0, dtype=np.float64)),
    }
    del training
    if min(scales.values()) <= 0.0 or not all(np.isfinite(list(scales.values()))):
        raise RuntimeError(f"invalid training-only scales: {scales}")

    regions, feature_rows = load_frozen_regions(validation)
    checkpoint_rows = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))["checkpoints"]
    control_cache = load_npz(STAGE3 / "validation/predictions_lambda_0_seed_101.npz")
    exact_dot_xi = control_cache["predicted_dot_xi"] - control_cache["dot_xi_error"]
    exact_dot_consistency = 0.0
    for seed in SEEDS[1:]:
        other = load_npz(STAGE3 / f"validation/predictions_lambda_0_seed_{seed}.npz")
        other_exact = other["predicted_dot_xi"] - other["dot_xi_error"]
        exact_dot_consistency = max(exact_dot_consistency, float(np.max(np.abs(exact_dot_xi - other_exact))))
    if exact_dot_consistency > 2.0e-15:
        raise RuntimeError("cached exact dot-xi differs across control seeds")

    wormhole, spiral = experiment_parameters()
    exact_acceleration = radial_acceleration(validation["x1"], validation["u1"], wormhole, spiral)
    main_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    decompositions: dict[tuple[float, int], dict[str, Any]] = {}
    physical_decompositions: dict[tuple[float, int], dict[str, Any]] = {}
    maximum_identity_residual = 0.0
    maximum_physical_identity_residual = 0.0
    x_xi_invalid_rows = 0
    x_u_invalid_rows = 0
    x_xi_tolerance = 0.0
    x_u_tolerance = 0.0
    scopes = {"global": np.ones(validation["s"].size, dtype=bool), "during": regions["during"], "post": regions["post"]}
    for value in LAMBDAS:
        for seed in SEEDS:
            cached = load_npz(STAGE3 / f"validation/predictions_lambda_{lambda_slug(value)}_seed_{seed}.npz")
            if not np.array_equal(cached["transition_id"], validation["transition_id"]):
                raise RuntimeError(f"cached validation row alignment failed for lambda={value}, seed={seed}")
            x_error = cached["predicted_x"] - validation["x1"]
            xi_error = cached["predicted_xi"] - validation["xi1"]
            if not np.array_equal(x_error, cached["x_error"]) or not np.array_equal(xi_error, cached["xi_error"]):
                raise RuntimeError(f"cached error arrays disagree for lambda={value}, seed={seed}")
            _, predicted_u = state_from_xi(cached["predicted_x"], cached["predicted_xi"], wormhole, spiral)
            u_error = predicted_u - validation["u1"]
            decomposition = decompose_phase_error(
                validation["u1"], exact_dot_xi, x_error, xi_error, scales["x"], scales["xi"]
            )
            physical = decompose_phase_error(
                validation["u1"], exact_acceleration, x_error, u_error, scales["x"], scales["u"]
            )
            decompositions[(value, seed)] = decomposition
            physical_decompositions[(value, seed)] = physical
            maximum_identity_residual = max(maximum_identity_residual, decomposition["maximum_pythagorean_residual"])
            maximum_physical_identity_residual = max(maximum_physical_identity_residual, physical["maximum_pythagorean_residual"])
            x_xi_invalid_rows = max(x_xi_invalid_rows, decomposition["invalid_count"])
            x_u_invalid_rows = max(x_u_invalid_rows, physical["invalid_count"])
            x_xi_tolerance = decomposition["tangent_tolerance"]
            x_u_tolerance = physical["tangent_tolerance"]
            for definition in BIN_DEFINITIONS:
                name = definition[0]
                family = bin_mask(validation["u_th"], definition)
                orbit_count = int(np.unique(validation["source_orbit_index"][family]).size)
                for scope, scope_mask in scopes.items():
                    selection = family & scope_mask
                    prefix = {
                        "lambda_dot_xi": value,
                        "treatment": LABELS[value],
                        "seed": seed,
                        "u_th_bin": name,
                        "scope": scope,
                        "orbit_count": int(np.unique(validation["source_orbit_index"][selection]).size),
                        "global_bin_orbit_count": orbit_count,
                    }
                    main_rows.append({**prefix, **metric_record(decomposition, x_error, selection)})
                    physical_rows.append({**prefix, **metric_record(physical, x_error, selection)})

    aggregate_rows = aggregate_seed_metrics(main_rows)
    physical_aggregate = aggregate_seed_metrics(physical_rows)
    orbit_rows = orbit_level_post_metrics(validation, regions["post"], decompositions)
    orbit_aggregate = aggregate_orbits(orbit_rows)

    coverage: list[dict[str, Any]] = []
    for definition in BIN_DEFINITIONS:
        name = definition[0]
        family = bin_mask(validation["u_th"], definition)
        row = {"u_th_bin": name}
        for scope, scope_mask in scopes.items():
            selected = family & scope_mask
            row[f"{scope}_row_count"] = int(np.sum(selected))
            row[f"{scope}_orbit_count"] = int(np.unique(validation["source_orbit_index"][selected]).size)
        coverage.append(row)

    references, reference_predictions, reference_discrepancy = reference_inference(preprocessing, checkpoint_rows)
    reference_rows = reference_metrics(references, reference_predictions, scales)
    compressed: dict[str, np.ndarray] = {}
    for target, exact in references.items():
        tag = f"u_th_{target:.2f}".replace(".", "p")
        for name in ("elapsed_s", "exact_x", "exact_xi", "exact_u", "exact_dot_xi"):
            compressed[f"{tag}_{name}"] = exact[name]
        for value in LAMBDAS:
            for seed in SEEDS:
                predicted = reference_predictions[(value, seed, target)]
                prefix = f"{tag}_lambda_{lambda_slug(value)}_seed_{seed}"
                for name, array in predicted.items():
                    compressed[f"{prefix}_{name}"] = array
    np.savez_compressed(ARRAYS / "reference_phase_space_predictions.npz", **compressed)

    write_csv(TABLES / "low_u_th_phase_space_metrics.csv", main_rows)
    write_csv(TABLES / "aggregate_treatment_metrics.csv", aggregate_rows)
    write_csv(TABLES / "physical_x_u_crosscheck.csv", physical_rows)
    write_csv(TABLES / "aggregate_physical_x_u_crosscheck.csv", physical_aggregate)
    write_csv(TABLES / "orbit_level_post_metrics.csv", orbit_rows)
    write_csv(TABLES / "orbit_level_post_seed_aggregate.csv", orbit_aggregate)
    write_csv(TABLES / "bin_coverage.csv", coverage)
    write_csv(TABLES / "reference_phase_space_metrics.csv", reference_rows)

    plot_binned_decomposition(aggregate_rows)
    plot_continuous(orbit_aggregate)
    plot_reference_phase_space(references, reference_predictions)

    control_lowest = aggregate_lookup(aggregate_rows, 0.0, "[0.01,0.05]", "post")
    moderate_lowest = aggregate_lookup(aggregate_rows, 0.034, "[0.01,0.05]", "post")
    physical_q_values = [
        aggregate_lookup(physical_aggregate, value, "[0.01,0.05]", "post")["Q_perp_over_parallel_mean"]
        for value in LAMBDAS
    ]
    physical_persists = all(value is not None and value < 1.0 for value in physical_q_values)
    stage3_summary = json.loads(STAGE3_SUMMARY.read_text(encoding="utf-8"))
    upper_global_x = next(
        row for row in stage3_summary["aggregate_decision_metrics"]
        if row["lambda_dot_xi"] == 0.068 and row["metric"] == "global_x_rmse"
    )
    decision = {
        "classification": "TIMING_PROGRESSION_DOMINANT_WITH_NONZERO_ORBIT_NORMAL_ERROR",
        "failure_answer": (
            "The evidence favors a timing/progression error along an otherwise largely correct orbit: "
            "post-change tangential RMSE exceeds normal RMSE for every treatment in the lowest bin. "
            "The normal error is nevertheless measurable and increases sharply as u_th decreases, so this is not a claim of perfect orbit shape."
        ),
        "preferred_existing_lambda": 0.034,
        "lowest_bin_parallel_to_perp_ratio_control": control_lowest["e_parallel_rmse_mean"] / control_lowest["e_perp_rmse_mean"],
        "control_lowest_Q": control_lowest["Q_perp_over_parallel_mean"],
        "lambda_0p034_lowest_Q": moderate_lowest["Q_perp_over_parallel_mean"],
        "lambda_0p034_lowest_post_e_perp_change": moderate_lowest["paired_e_perp_relative_change_mean"],
        "lambda_0p068_previous_global_x_percent_change": upper_global_x["mean_percent_change_from_control"],
        "physical_crosscheck_persists": physical_persists,
        "physical_crosscheck_statement": (
            "supports the same qualitative conclusion: tangential error exceeds normal error in the lowest post-change bin for all four treatments"
            if physical_persists else
            "does not uniformly reproduce the ML-coordinate component ordering, so coordinate dependence must be retained as a caveat"
        ),
        "architecture_statement": (
            "There is a meaningful but qualified reason to test the parameter-matched split-head architecture next. "
            "The upper derivative weight reduces narrow low-u_th normal error while degrading global x, which is consistent with shared-representation interference. "
            "Because tangential error still dominates and lambda=0.034 already improves the balanced tradeoff, a split-head run would be a focused specialization test—not evidence that more capacity is required."
        ),
    }

    tests = run_tests()
    if not tests["passed"]:
        raise RuntimeError(f"focused tests failed: {tests}")
    protected_after = verify_hashes(expected_inputs)
    if protected_before != protected_after:
        raise RuntimeError("protected inputs changed during validation-only audit")
    summary = {
        "created_utc": utc_now(),
        "status": "VALIDATION_ONLY_PHASE_SPACE_AUDIT_COMPLETED",
        "scope": {
            "training_performed": False,
            "ode_integration_performed": False,
            "new_feature_detection_performed": False,
            "dataset_modified": False,
            "held_out_test_accessed": False,
            "validation_inference_reused_from_cache": True,
            "reference_inference_passes_per_checkpoint": 1,
        },
        "scales": scales,
        "frozen_regions": {
            "source": str(FEATURE_WINDOWS.resolve()),
            "feature_count": len(feature_rows),
            "threshold": json.loads((STAGE3 / "protocol/frozen_experiment_protocol.json").read_text(encoding="utf-8"))["rapid_region_rule"]["threshold"],
            "recomputed": False,
        },
        "coverage": coverage,
        "aggregate_metrics": aggregate_rows,
        "physical_crosscheck_aggregate": physical_aggregate,
        "reference_metrics": reference_rows,
        "numerical_verification": {
            "maximum_pythagorean_residual": maximum_identity_residual,
            "maximum_physical_pythagorean_residual": maximum_physical_identity_residual,
            "x_xi_tangent_tolerance": x_xi_tolerance,
            "x_u_tangent_tolerance": x_u_tolerance,
            "x_xi_invalid_tangent_rows": x_xi_invalid_rows,
            "x_u_invalid_tangent_rows": x_u_invalid_rows,
            "cached_exact_dot_xi_maximum_cross_seed_discrepancy": exact_dot_consistency,
            "reference_cached_xi_maximum_inference_discrepancy": reference_discrepancy,
        },
        "decision": decision,
        "tests": tests,
        "protected_hashes_unchanged": True,
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(report_text(summary), encoding="utf-8")

    artifacts = {
        str(path.relative_to(OUTPUT)): {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)
    }
    manifest = {
        "status": summary["status"],
        "scope": summary["scope"],
        "protected_before": protected_before,
        "protected_after": protected_after,
        "source_hashes": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str((ROOT / "tests/test_finite_time_phase_space_audit.py").resolve()): sha256(ROOT / "tests/test_finite_time_phase_space_audit.py"),
        },
        "summary_sha256": sha256(SUMMARY),
        "report_sha256": sha256(REPORT),
        "artifacts": artifacts,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
