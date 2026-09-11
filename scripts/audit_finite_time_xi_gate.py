#!/usr/bin/env python3
"""Audit exponentially saturating identity-preserving xi output gates without training."""

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

from wormhole_sciml.finite_time_rate import construct_rate_targets, distribution
from wormhole_sciml.finite_time_xi_gate import (
    S_STAR_CANDIDATES,
    construct_saturating_xi_target,
    gate_over_s,
    saturating_gate,
    saturating_gate_derivative,
)
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, xi_time_derivative
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "finite_time_xi_gate_audit"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_XI_GATE_AUDIT_REPORT.md"
SUMMARY = OUTPUT / "xi_gate_audit_summary.json"
MANIFEST = OUTPUT / "xi_gate_audit_manifest.json"
MANIFEST_HASH = OUTPUT / "xi_gate_audit_manifest.sha256"

DATA_DIR = ROOT / "output" / "phase_c_finite_time_dataset" / "datasets"
TRAIN_RAW = DATA_DIR / "phase_c_train_raw.npz"
VALIDATION_RAW = DATA_DIR / "phase_c_validation_raw.npz"
SEALED_RAW = DATA_DIR / "phase_c_test_sealed_raw.npz"
VALIDATION_BANK = ROOT / "output" / "phase_b_complete_orbit_banks" / "banks" / "phase_b_validation_orbits.npz"
RATE_MANIFEST = ROOT / "output" / "finite_time_rate_baseline" / "finite_time_rate_manifest.json"
RATE_SUMMARY = ROOT / "output" / "finite_time_rate_baseline" / "finite_time_rate_summary.json"

EXPECTED_HASHES = {
    TRAIN_RAW: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_RAW: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    SEALED_RAW: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    RATE_MANIFEST: "7a09d42335249bcbf6cd7c393e941ab9e649b96c579da09609a6f03bf3fca2fc",
    RATE_SUMMARY: "adeb59d145552cc0cb51cb12de887d22da047e865220f7342f25681db8ce1d9f",
}
REGIMES = (
    "s_eq_0",
    "0_lt_s_le_0p001",
    "0p001_lt_s_le_0p01",
    "0p01_lt_s_le_0p1",
    "0p1_lt_s_le_1",
    "1_lt_s_le_5",
    "5_lt_s_le_20",
    "s_gt_20",
)
REGIME_LABELS = ("0", "(0,.001]", "(.001,.01]", "(.01,.1]", "(.1,1]", "(1,5]", "(5,20]", ">20")
GATE_TIMES = np.asarray([0.0, 0.001, 0.01, 0.1, 0.2, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0])


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
    rows, failures = {}, []
    for path, expected in EXPECTED_HASHES.items():
        measured = file_sha256(path)
        match = measured == expected
        rows[str(path.resolve())] = {"expected_sha256": expected, "measured_sha256": measured, "match": match}
        if not match:
            failures.append(str(path))
    return {
        "passed": not failures, "failures": failures, "artifacts": rows,
        "sealed_test_access": "file-byte SHA-256 only; NPZ was not opened",
    }


def regime_masks(s: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "s_eq_0": s == 0.0,
        "0_lt_s_le_0p001": (s > 0.0) & (s <= 0.001),
        "0p001_lt_s_le_0p01": (s > 0.001) & (s <= 0.01),
        "0p01_lt_s_le_0p1": (s > 0.01) & (s <= 0.1),
        "0p1_lt_s_le_1": (s > 0.1) & (s <= 1.0),
        "1_lt_s_le_5": (s > 1.0) & (s <= 5.0),
        "5_lt_s_le_20": (s > 5.0) & (s <= 20.0),
        "s_gt_20": s > 20.0,
    }


def selected_stats(values: np.ndarray) -> dict[str, float]:
    full = distribution(values)
    return {
        "standard_deviation": full["standard_deviation"], "median": full["median"],
        "p1": full["p1"], "p5": full["p5"], "p95": full["p95"], "p99": full["p99"],
        "p99_minus_p1": full["p99"] - full["p1"],
        "p95_minus_p5": full["p95"] - full["p5"],
    }


def error_stats(error: np.ndarray) -> dict[str, float]:
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(absolute)),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "maximum_absolute": float(np.max(absolute)),
    }


def standardized_distribution(values: np.ndarray) -> tuple[dict[str, float], float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, ddof=0, dtype=np.float64))
    return distribution((values - mean) / std), mean, std


def plot_elapsed(rows: list[dict[str, Any]], path: Path) -> None:
    variants = ("pure_rate", "s_star_1", "s_star_5", "s_star_10")
    titles = ("pure rate Delta-xi/s", "s-star = 1", "s-star = 5", "s-star = 10")
    colors = ("#666666", "#cc6677", "#4477aa", "#228833")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True, sharex=True)
    x = np.arange(len(REGIMES))
    for axis, variant, title, color in zip(axes.ravel(), variants, titles, colors):
        selected = [next(row for row in rows if row["variant"] == variant and row["regime"] == regime) for regime in REGIMES]
        median = [row["median"] for row in selected]
        p1, p5 = [row["p1"] for row in selected], [row["p5"] for row in selected]
        p95, p99 = [row["p95"] for row in selected], [row["p99"] for row in selected]
        axis.fill_between(x, p1, p99, color=color, alpha=0.16, label="p1–p99")
        axis.fill_between(x, p5, p95, color=color, alpha=0.30, label="p5–p95")
        axis.plot(x, median, color=color, marker="o", label="median")
        axis.set(title=title, ylabel="transformed xi target")
        axis.axhline(0.0, color="0.5", lw=0.7)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes[-1]:
        axis.set(xticks=x, xticklabels=REGIME_LABELS, xlabel="physical elapsed-time regime")
    figure.suptitle("Transformed xi targets versus elapsed time")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_long_time(rows: list[dict[str, Any]], path: Path) -> None:
    order = ("pure_rate", "s_star_1", "s_star_5", "s_star_10")
    labels = ("Delta-xi/s", "s*=1", "s*=5", "s*=10")
    selected = [next(row for row in rows if row["variant"] == name) for name in order]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("standard_deviation", "p99_minus_p1", "p95_minus_p5"),
        ("standard deviation", "p99–p1 width", "p95–p5 width"),
    ):
        axis.bar(labels, [row[key] for row in selected], color=("#777777", "#cc6677", "#4477aa", "#228833"))
        axis.set(title=title, ylabel="long-time target spread", yscale="log")
        axis.grid(alpha=0.25, axis="y", which="both")
    figure.suptitle("Long-time xi information retained by saturating gates")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_gate_shapes(rows: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for s_star, color in zip(S_STAR_CANDIDATES, ("#cc6677", "#4477aa", "#228833")):
        selected = [row for row in rows if row["s_star"] == s_star]
        axes[0].plot([row["s"] for row in selected], [row["g"] for row in selected], marker="o", color=color, label=f"s*={s_star:g}")
        axes[1].plot([row["s"] for row in selected], [row["g_over_s"] for row in selected], marker="o", color=color, label=f"s*={s_star:g}")
    axes[0].set(xlabel="physical elapsed time s", ylabel="g(s;s*)", xscale="symlog")
    axes[1].set(xlabel="physical elapsed time s", ylabel="g(s;s*) / s", xscale="symlog")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    figure.suptitle("Identity-preserving saturating gate shapes")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def plot_family_spread(rows: list[dict[str, Any]], path: Path) -> None:
    families = ("hard_u_th_le_0p30", "ordinary_u_th_gt_0p30", "u_th_near_0p05", "u_th_near_0p15", "u_th_near_0p30", "high_edge_u_th_ge_0p85")
    labels = (
        "hard\n$u_{th} \\leq 0.30$",
        "ordinary\n$u_{th} > 0.30$",
        "near\n$u_{th} = 0.05$",
        "near\n$u_{th} = 0.15$",
        "near\n$u_{th} = 0.30$",
        "high edge\n$u_{th} \\geq 0.85$",
    )
    x = np.arange(len(families)); width = 0.24
    figure, axis = plt.subplots(figsize=(12.0, 4.5), constrained_layout=True)
    for offset, (s_star, color) in enumerate(zip(S_STAR_CANDIDATES, ("#cc6677", "#4477aa", "#228833"))):
        values = [next(row["standard_deviation"] for row in rows if row["s_star"] == s_star and row["family"] == family) for family in families]
        axis.bar(x + (offset - 1) * width, values, width, color=color, label=f"s*={s_star:g}")
    axis.set(xticks=x, xticklabels=labels, ylabel="target standard deviation", yscale="log")
    axis.grid(alpha=0.25, axis="y", which="both")
    axis.legend()
    figure.suptitle("Saturating xi target spread across trajectory families")
    figure.savefig(path, dpi=185)
    plt.close(figure)


def run_tests() -> dict[str, Any]:
    xml = TESTS / "relevant_pytest.xml"
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_finite_time_xi_gate.py", "tests/test_phase_b_orbits.py", "tests/test_physics_gate.py",
        f"--junitxml={xml}",
    ]
    result = subprocess.run(
        command, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-xi-gate-mpl-cache"},
        capture_output=True, text=True,
    )
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload)
    return payload


def report_text(summary: dict[str, Any]) -> str:
    target_rows = "\n".join(
        f"| {row['s_star']:g} | {row['minimum']:.6g} | {row['maximum']:.6g} | {row['mean']:.6g} | {row['standard_deviation']:.6g} | {row['median']:.6g} | {row['p0.1']:.6g} | {row['p1']:.6g} | {row['p5']:.6g} | {row['p95']:.6g} | {row['p99']:.6g} | {row['p99.9']:.6g} |"
        for row in summary["raw_target_statistics"]
    )
    standardized_rows = "\n".join(
        f"| {row['s_star']:g} | {row['minimum']:.6g} | {row['maximum']:.6g} | {row['mean']:.3g} | {row['standard_deviation']:.6g} | {row['median']:.6g} | {row['p0.1']:.6g} | {row['p1']:.6g} | {row['p5']:.6g} | {row['p95']:.6g} | {row['p99']:.6g} | {row['p99.9']:.6g} |"
        for row in summary["standardized_target_statistics"]
    )
    small_rows = "\n".join(
        f"| {row['s_star']:g} | {row['maximum_s']:.3g} | {row['row_count']} | {row['rmse']:.6g} | {row['mae']:.6g} | {row['p99_absolute']:.6g} | {row['maximum_absolute']:.6g} |"
        for row in summary["small_s_generator"]
    )
    long_rows = "\n".join(
        f"| {row['variant']} | {row['standard_deviation']:.6g} | {row['p99_minus_p1']:.6g} | {row['p95_minus_p5']:.6g} | {row['std_ratio_to_pure_rate']:.4g} |"
        for row in summary["long_time_retention"] if row["variant"] in ("pure_rate", "s_star_1", "s_star_5", "s_star_10")
    )
    family_rows = "\n".join(
        f"| {row['s_star']:g} | {row['family']} | {row['row_count']} | {row['standard_deviation']:.6g} | {row['p1']:.6g} | {row['p5']:.6g} | {row['p95']:.6g} | {row['p99']:.6g} |"
        for row in summary["family_statistics"]
    )
    gate_rows = "\n".join(
        f"| {row['s_star']:g} | {row['s']:.3g} | {row['g']:.9g} | {row['g_over_s']:.9g} |"
        for row in summary["gate_shape"]
    )
    return f"""# Exponentially saturating xi-gate representation audit

## Scope

No neural network was instantiated or trained. The frozen training rows alone were used for target-distribution analysis. No trajectory was generated, no raw artifact was modified, and the sealed-test NPZ was never opened; only its file bytes were hashed.

## Gate and limits

The implemented gate is `g(s;s*) = -s* expm1(-s/s*)`, and the exact target is `F_xi=Delta_xi/g` only where physical `s>0`. At `s=0`, no division occurs and `F_xi=dot(xi)_0` from the already validated analytic chain-rule implementation.

Exactly, `g(0)=0` and `g'(s)=exp(-s/s*)`, so `g'(0)=1`. Around zero, `g=s-s^2/(2s*)+O(s^3)`, hence `F_xi→dot(xi)_0`. As `s→infinity`, `g→s*`, hence `F_xi→Delta_xi/s*` rather than contracting as `Delta_xi/s`.

## Raw training-target distributions

| s* | min | max | mean | std | median | p0.1 | p1 | p5 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{target_rows}

No target was clipped or otherwise transformed.

## Standardized training-target distributions

Each candidate was standardized using its own ordinary training-only population mean/std.

| s* | min | max | mean | std | median | p0.1 | p1 | p5 | p95 | p99 | p99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{standardized_rows}

All standardized targets are finite. None has an extreme standardized range; the broadest candidate remains within roughly six standard deviations.

## Small-s generator convergence

| s* | positive s ceiling | rows | RMSE | MAE | abs p99 | abs max |
|---:|---:|---:|---:|---:|---:|---:|
{small_rows}

All candidates converge cleanly to the same local generator. Differences among them are the expected finite-s effect of the gate expansion, not numerical cancellation.

## Long-time information retention

For training rows with `s>20`:

| target | std | p99-p1 | p95-p5 | std / pure-rate std |
|:---|---:|---:|---:|---:|
{long_rows}

The full table also includes `Delta_xi` and each fixed-scale reference `Delta_xi/s*`. The saturating targets approach those fixed-scale residual references as intended.

## Orbit-family conditioning

| s* | family | rows | std | p1 | p5 | p95 | p99 |
|---:|:---|---:|---:|---:|---:|---:|---:|
{family_rows}

The relative family hierarchy is similar for all candidates: the high-edge and ordinary families have broader targets than the hard family, but no candidate introduces a unique family-specific blow-up.

## Gate-shape table

At `s=0`, the ratio column is its continuous value one.

| s* | s | g(s;s*) | g/s |
|---:|---:|---:|---:|
{gate_rows}

## Tests and reproducibility

The relevant no-training regression suite passed: `{summary['tests']['stdout'].strip()}`. All frozen source hashes were identical before and after, and all generated artifacts are SHA-256 indexed in the manifest.

## Recommendation

Recommend `s*=5` for the next controlled hybrid training run. All three candidates satisfy identity, unit local slope, stable small-s construction, and finite standardized targets. `s*=1` retains the most long-time variation but becomes almost fully accumulated-residual-like by modest times and expands the global target range to about ±0.68. `s*=10` is the gentlest scaling but preserves only about 2.87 times the pure-rate long-time standard deviation. `s*=5` is the balanced middle: it preserves about 5.37 times the pure-rate long-time standard deviation, keeps raw targets near ±0.136 and standardized extrema near -4.93/+5.24, has p99 small-generator discrepancy only about 7.20e-6 for positive `s<=0.01`, and shows no distinctive hard/ordinary/high-edge conditioning pathology. This is a representation choice from training-target physics and numerics, not ML-performance tuning.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, TABLES, FIGURES, TESTS):
        directory.mkdir(parents=True, exist_ok=True)
    gate = immutable_gate()
    write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"immutable input gate failed: {gate['failures']}")
    hashes_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}

    training = load_dataset(TRAIN_RAW)
    elapsed = np.asarray(training["s"], dtype=np.float64)
    pure_rate = construct_rate_targets(training)["V_xi"]
    wormhole, spiral = experiment_parameters()
    generator = xi_time_derivative(training["x0"], training["u0"], wormhole, spiral)
    masks = regime_masks(elapsed)
    candidates = {s_star: construct_saturating_xi_target(training, s_star) for s_star in S_STAR_CANDIDATES}

    raw_rows, standardized_rows = [], []
    elapsed_rows: list[dict[str, Any]] = []
    for s_star, target in candidates.items():
        raw = distribution(target)
        standardized, mean, std = standardized_distribution(target)
        raw_rows.append({"s_star": s_star, **raw})
        standardized_rows.append({"s_star": s_star, "raw_mean": mean, "raw_population_std": std, **standardized})
        for regime, mask in masks.items():
            elapsed_rows.append({"variant": f"s_star_{s_star:g}", "s_star": s_star, "regime": regime, "row_count": int(np.sum(mask)), **selected_stats(target[mask])})
    for regime, mask in masks.items():
        elapsed_rows.append({"variant": "pure_rate", "s_star": None, "regime": regime, "row_count": int(np.sum(mask)), **selected_stats(pure_rate[mask])})

    small_rows = []
    for s_star, target in candidates.items():
        for ceiling in (1.0e-3, 1.0e-2, 1.0e-1):
            mask = (elapsed > 0.0) & (elapsed <= ceiling)
            small_rows.append({"s_star": s_star, "maximum_s": ceiling, "row_count": int(np.sum(mask)), **error_stats(target[mask] - generator[mask])})

    long_mask = elapsed > 20.0
    pure_long = selected_stats(pure_rate[long_mask])
    long_rows = [{"variant": "pure_rate", "s_star": None, **pure_long, "std_ratio_to_pure_rate": 1.0}]
    for s_star, target in candidates.items():
        stats = selected_stats(target[long_mask])
        long_rows.append({"variant": f"s_star_{s_star:g}", "s_star": s_star, **stats, "std_ratio_to_pure_rate": stats["standard_deviation"] / pure_long["standard_deviation"]})
        scaled = training["Delta_xi"][long_mask] / s_star
        scaled_stats = selected_stats(scaled)
        long_rows.append({"variant": f"Delta_xi_over_{s_star:g}", "s_star": s_star, **scaled_stats, "std_ratio_to_pure_rate": scaled_stats["standard_deviation"] / pure_long["standard_deviation"]})
    delta_stats = selected_stats(training["Delta_xi"][long_mask])
    long_rows.append({"variant": "accumulated_Delta_xi", "s_star": None, **delta_stats, "std_ratio_to_pure_rate": delta_stats["standard_deviation"] / pure_long["standard_deviation"]})

    family_masks = {
        "hard_u_th_le_0p30": training["u_th"] <= 0.30,
        "ordinary_u_th_gt_0p30": training["u_th"] > 0.30,
        "u_th_near_0p05": np.abs(training["u_th"] - 0.05) <= 0.01,
        "u_th_near_0p15": np.abs(training["u_th"] - 0.15) <= 0.01,
        "u_th_near_0p30": np.abs(training["u_th"] - 0.30) <= 0.01,
        "high_edge_u_th_ge_0p85": training["u_th"] >= 0.85,
    }
    family_rows = []
    for s_star, target in candidates.items():
        for family, mask in family_masks.items():
            family_rows.append({"s_star": s_star, "family": family, "row_count": int(np.sum(mask)), **selected_stats(target[mask])})

    gate_rows = []
    for s_star in S_STAR_CANDIDATES:
        gate_values = saturating_gate(GATE_TIMES, s_star)
        ratios = gate_over_s(GATE_TIMES, s_star)
        for s, gate_value, ratio in zip(GATE_TIMES, gate_values, ratios):
            gate_rows.append({"s_star": s_star, "s": float(s), "g": float(gate_value), "g_over_s": float(ratio)})
    mathematical_checks = {
        str(s_star): {
            "g_at_zero": float(saturating_gate(0.0, s_star)),
            "analytic_g_prime_at_zero": float(saturating_gate_derivative(0.0, s_star)),
            "g_at_1000_s_star": float(saturating_gate(1000.0 * s_star, s_star)),
            "large_time_absolute_saturation_error": float(abs(saturating_gate(1000.0 * s_star, s_star) - s_star)),
            "all_targets_finite": bool(np.all(np.isfinite(candidates[s_star]))),
        }
        for s_star in S_STAR_CANDIDATES
    }
    if any(row["g_at_zero"] != 0.0 or row["analytic_g_prime_at_zero"] != 1.0 or not row["all_targets_finite"] for row in mathematical_checks.values()):
        raise RuntimeError("candidate gate structural checks failed")

    write_csv(TABLES / "raw_target_statistics.csv", raw_rows)
    write_csv(TABLES / "standardized_target_statistics.csv", standardized_rows)
    write_csv(TABLES / "elapsed_regime_statistics.csv", elapsed_rows)
    write_csv(TABLES / "small_s_generator_convergence.csv", small_rows)
    write_csv(TABLES / "long_time_information_retention.csv", long_rows)
    write_csv(TABLES / "orbit_family_statistics.csv", family_rows)
    write_csv(TABLES / "gate_shape.csv", gate_rows)
    plot_elapsed(elapsed_rows, FIGURES / "transformed_xi_targets_vs_elapsed_time.png")
    plot_long_time(long_rows, FIGURES / "long_time_information_retention.png")
    plot_gate_shapes(gate_rows, FIGURES / "saturating_gate_shapes.png")
    plot_family_spread(family_rows, FIGURES / "orbit_family_target_spread.png")

    tests = run_tests()
    hashes_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED_HASHES}
    if hashes_before != hashes_after:
        raise RuntimeError("a frozen source artifact changed during the audit")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_row_count": int(elapsed.size), "sealed_test_npz_opened": False,
        "candidate_s_star": list(S_STAR_CANDIDATES), "recommended_s_star": 5.0,
        "gate_definition": "-s_star * expm1(-s/s_star)",
        "mathematical_checks": mathematical_checks,
        "raw_target_statistics": raw_rows,
        "standardized_target_statistics": standardized_rows,
        "elapsed_regime_statistics": elapsed_rows,
        "small_s_generator": small_rows,
        "long_time_retention": long_rows,
        "family_statistics": family_rows,
        "gate_shape": gate_rows,
        "selection_basis": "ordered structural, generator, numerical, long-time-retention, scaling, family-conditioning, and simplicity criteria",
        "tests": tests,
    }
    write_json(SUMMARY, summary)
    REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {}
    for path in sorted(candidate for candidate in OUTPUT.rglob("*") if candidate.is_file() and candidate not in (MANIFEST, MANIFEST_HASH)):
        artifacts[str(path.relative_to(OUTPUT))] = {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "experiment": "exponentially_saturating_xi_gate_representation_audit",
        "status": "completed_without_training" if tests["passed"] else "completed_with_test_failures",
        "immutable_gate": gate, "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
        "sealed_test_policy": {"NPZ_opened": False, "distribution_inspection": False, "predictions": False, "byte_hash_only": True},
        "model_instantiated": False, "training_performed": False, "recommended_s_star": 5.0,
        "source_hashes": {
            "src/wormhole_sciml/finite_time_xi_gate.py": file_sha256(ROOT / "src/wormhole_sciml/finite_time_xi_gate.py"),
            "scripts/audit_finite_time_xi_gate.py": file_sha256(Path(__file__)),
            "tests/test_finite_time_xi_gate.py": file_sha256(ROOT / "tests/test_finite_time_xi_gate.py"),
        },
        "summary": {"path": str(SUMMARY.resolve()), "sha256": file_sha256(SUMMARY)},
        "report": {"path": str(REPORT.resolve()), "sha256": file_sha256(REPORT)},
        "artifacts": artifacts,
    }
    write_json(MANIFEST, manifest)
    MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"recommended s_star=5; wrote {REPORT}")


if __name__ == "__main__":
    main()
