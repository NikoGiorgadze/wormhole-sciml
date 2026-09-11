#!/usr/bin/env python3
"""Generate, validate, and freeze Phase-B complete exact orbit banks only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from wormhole_sciml.dynamics import velocity_bounds
from wormhole_sciml.phase_b_orbits import (
    DIFFICULT_HIGH,
    DUPLICATE_TOLERANCE,
    ENDPOINT_TOLERANCE,
    ENERGY_DRIFT_LIMIT,
    HIGH_EDGE_LOW,
    LOW_EDGE_HIGH,
    MASTER_SEED,
    SPLIT_COUNTS,
    STRESS_U_TH,
    THROAT_TOLERANCE,
    U_TH_LOW,
    XI_DATA_EDGE,
    build_orbit_plan,
    evaluate_saved_orbit,
    file_sha256,
    generate_bank,
    generation_seeds,
    integrate_complete_orbit,
    json_dump,
    leakage_diagnostics,
    metadata_rows,
    save_bank,
    stratum_bounds,
    validated_u_th_high,
)
from wormhole_sciml.physics_gate import (
    VALIDATION_SOLVER,
    experiment_parameters,
    source_manifest,
    xi_from_state,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "phase_b_complete_orbit_banks"
FIGURES = OUTPUT / "figures"
BANKS = OUTPUT / "banks"
SCRIPT_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "src" / "wormhole_sciml" / "phase_b_orbits.py"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["split", "trajectory_stratum", "u_th", "reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def git_provenance() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return {"available": True, "commit": result.stdout.strip()}
    return {
        "available": False,
        "commit": None,
        "reason": result.stderr.strip() or "git metadata unavailable",
    }


def domain_validation() -> list[dict[str, Any]]:
    """Scan the selected closed interval and record the two analytic exclusions."""

    high = validated_u_th_high()
    values = np.linspace(U_TH_LOW, high, 65, dtype=np.float64)
    values = np.unique(np.concatenate((values, np.asarray(STRESS_U_TH))))
    rows: list[dict[str, Any]] = []
    for value in values:
        try:
            metadata, _dense = integrate_complete_orbit(float(value))
            rows.append(
                {
                    "u_th": float(value),
                    "classification": "accepted_throughgoing",
                    "reached_left": True,
                    "reached_right": True,
                    "minimum_C": metadata["minimum_C"],
                    "maximum_relative_energy_drift": metadata["maximum_relative_energy_drift"],
                    "crossing_time": metadata["total_crossing_time"],
                    "reason": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "u_th": float(value),
                    "classification": "failed_inside_selected_interval",
                    "reached_left": False,
                    "reached_right": False,
                    "minimum_C": "",
                    "maximum_relative_energy_drift": "",
                    "crossing_time": "",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    wormhole, spiral = experiment_parameters()
    _lower, upper = velocity_bounds(0.0, wormhole, spiral)
    rows.extend(
        (
            {
                "u_th": 0.0,
                "classification": "excluded_critical_open_boundary",
                "reached_left": False,
                "reached_right": False,
                "minimum_C": "",
                "maximum_relative_energy_drift": "",
                "crossing_time": "",
                "reason": "critical throat state is not a finite-time positive through-going crossing",
            },
            {
                "u_th": float(upper),
                "classification": "excluded_null_open_boundary",
                "reached_left": False,
                "reached_right": False,
                "minimum_C": 0.0,
                "maximum_relative_energy_drift": "",
                "crossing_time": "",
                "reason": "upper throat velocity is the C=0 null root",
            },
        )
    )
    return rows


def descriptive(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def summarize_bank(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    strata = {
        name: int(np.sum(arrays["trajectory_stratum"] == name))
        for name in np.unique(arrays["trajectory_stratum"])
    }
    return {
        "count": int(arrays["orbit_id"].size),
        "stratum_counts": strata,
        "u_th": descriptive(arrays["u_th"]),
        "E0": descriptive(arrays["E0"]),
        "total_crossing_time": descriptive(arrays["total_crossing_time"]),
        "minimum_C": descriptive(arrays["minimum_C"]),
        "maximum_relative_energy_drift": descriptive(arrays["maximum_relative_energy_drift"]),
        "endpoint_residual": descriptive(arrays["endpoint_residual"]),
        "dense_reconstruction_max_abs_error": descriptive(arrays["dense_reconstruction_max_abs_error"]),
        "all_solver_status_terminal_event": bool(np.all(arrays["solver_status"] == 1)),
        "all_monotonic_x": bool(np.all(arrays["monotonic_x"])),
        "all_finite_scalars": bool(
            all(np.all(np.isfinite(arrays[key])) for key in (
                "u_th", "E0", "total_crossing_time", "minimum_C",
                "minimum_one_minus_abs_xi", "maximum_relative_energy_drift",
                "endpoint_residual", "dense_reconstruction_max_abs_error",
            ))
        ),
    }


def plot_coverage(slim: dict[str, dict[str, np.ndarray]], destination: Path) -> None:
    colors = {"broad": "#277da1", "difficult": "#f8961e", "low_edge": "#d62828", "high_edge": "#6a4c93"}
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    for split, bank in slim.items():
        for stratum, color in colors.items():
            mask = bank["trajectory_stratum"] == stratum
            axes[0, 0].hist(bank["u_th"][mask], bins=45, histtype="step", lw=1.2, color=color, alpha=0.9, label=stratum if split == "train" else None)
        axes[0, 1].hist(bank["u_th"], bins=70, histtype="step", lw=1.4, label=split)
        axes[1, 0].hist(bank["E0"], bins=70, histtype="step", lw=1.4, label=split)
        axes[1, 1].scatter(bank["u_th"], bank["E0"], s=3, alpha=0.28, label=split)
    axes[0, 0].set(title="Training coverage by stratum", xlabel=r"$u_{th}$", ylabel="count")
    axes[0, 1].set(title="Complete split coverage", xlabel=r"$u_{th}$", ylabel="count")
    axes[1, 0].set(title="Conserved-energy coverage", xlabel=r"$\mathcal{E}_0$", ylabel="count")
    axes[1, 1].set(title="Generating-coordinate map", xlabel=r"$u_{th}$", ylabel=r"$\mathcal{E}_0$")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
        axis.legend(fontsize=8)
    fig.suptitle("Phase B orbit-bank generating-coordinate coverage")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_diagnostics(slim: dict[str, dict[str, np.ndarray]], destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    for split, bank in slim.items():
        axes[0].hist(bank["total_crossing_time"], bins=70, histtype="step", lw=1.4, label=split)
        axes[1].hist(bank["minimum_C"], bins=70, histtype="step", lw=1.4, label=split)
        axes[2].hist(bank["maximum_relative_energy_drift"], bins=np.logspace(-16, -8.8, 70), histtype="step", lw=1.4, label=split)
    axes[0].set(title="Complete crossing time", xlabel=r"$T_{cross}$", ylabel="count")
    axes[1].set(title="Minimum timelike margin", xlabel=r"$\min C$", ylabel="count")
    axes[2].set(title="Maximum relative energy drift", xlabel="relative drift", ylabel="count", xscale="log")
    axes[2].axvline(ENERGY_DRIFT_LIMIT, color="red", ls="--", lw=1.0, label="acceptance limit")
    for axis in axes:
        axis.grid(alpha=0.16)
        axis.legend(fontsize=8)
    fig.suptitle("Phase B physical and numerical diagnostics")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def stress_paths(stress_path: Path) -> dict[float, dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    output: dict[float, dict[str, np.ndarray]] = {}
    with np.load(stress_path, allow_pickle=False) as bank:
        for index, value in enumerate(bank["u_th"]):
            times = np.linspace(0.0, float(bank["t_right"][index]), 1601)
            state = evaluate_saved_orbit(bank, index, times)
            xi = xi_from_state(state[:, 0], state[:, 1], wormhole, spiral)
            output[float(value)] = {"t": times, "x": state[:, 0], "u": state[:, 1], "xi": xi}
    return output


def plot_representatives(paths: dict[float, dict[str, np.ndarray]], destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.06, 0.94, len(paths)))
    for (value, path), color in zip(sorted(paths.items()), colors, strict=True):
        label = rf"$u_{{th}}={value:.2f}$"
        axes[0].plot(path["x"], path["xi"], color=color, lw=1.5, label=label)
        mask = (path["x"] >= -17.0) & (path["x"] <= -8.5)
        axes[1].plot(path["x"][mask], path["xi"][mask], color=color, lw=1.65, label=label)
    axes[0].set(title="Complete exact trajectories", xlabel="$x$", ylabel=r"$\xi$")
    axes[1].set(title="Sensitive incoming-left branch", xlabel="$x$", ylabel=r"$\xi$", xlim=(-17.0, -8.5))
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Dedicated immutable stress/reference family")
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def plot_difficult_overlay(paths: dict[float, dict[str, np.ndarray]], destination: Path) -> None:
    selected = (0.05, 0.15, 0.30, 0.50, 0.80, 0.90)
    fig, axis = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    for value in selected:
        path = paths[value]
        hard = value <= DIFFICULT_HIGH
        axis.plot(path["x"], path["xi"], lw=2.1 if hard else 1.25, ls="-" if hard else "--", label=rf"$u_{{th}}={value:.2f}$" + (" difficult" if hard else " ordinary"))
    axis.axvspan(-17.0, -8.5, color="#f4a261", alpha=0.10, label="prior sensitive region")
    axis.set(title="Difficult low-velocity family versus ordinary through-going orbits", xlabel="$x$", ylabel=r"$\xi$")
    axis.grid(alpha=0.18)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(destination, dpi=190)
    plt.close(fig)


def aggregate_statistics(summaries: dict[str, dict[str, Any]], slim: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    combined = {
        key: np.concatenate([bank[key] for bank in slim.values()])
        for key in ("maximum_relative_energy_drift", "endpoint_residual", "minimum_C", "total_crossing_time", "dense_reconstruction_max_abs_error")
    }
    return {
        "splits": summaries,
        "entire_main_bank": {key: descriptive(value) for key, value in combined.items()},
    }


def load_existing_bank(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only orbit-level arrays from a completed bank for safe resume."""

    dense_keys = {"segment_end", "dense_y_old", "dense_coefficients"}
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files if key not in dense_keys}
        segment_count = int(arrays["segment_offsets"][-1])
    artifact = {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "orbit_count": int(arrays["orbit_id"].size),
        "dense_segment_count": segment_count,
        "bytes": path.stat().st_size,
        "schema": "unchanged completed NPZ; see generation_config.json",
    }
    return arrays, artifact


def verify_reused_plan(
    arrays: dict[str, np.ndarray], planned: list[dict[str, Any]], label: str
) -> None:
    expected_u_th = np.asarray([row["u_th"] for row in planned], dtype=np.float64)
    expected_strata = np.asarray(
        [row["trajectory_stratum"] for row in planned], dtype="U16"
    )
    if not np.array_equal(arrays["u_th"], expected_u_th):
        raise RuntimeError(f"existing {label} bank does not match the deterministic u_th plan")
    if not np.array_equal(arrays["trajectory_stratum"], expected_strata):
        raise RuntimeError(f"existing {label} bank does not match the deterministic stratum plan")


def report_text(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    leakage: dict[str, Any],
    rejections: list[dict[str, Any]],
) -> str:
    counts = summary["statistics"]["splits"]
    count_rows = []
    for split in ("train", "validation", "test"):
        row = counts[split]
        strata = row["stratum_counts"]
        count_rows.append(
            f"| {split} | {row['count']} | {strata['broad']} | {strata['difficult']} | {strata['low_edge']} | {strata['high_edge']} |"
        )
    energy = summary["statistics"]["entire_main_bank"]["maximum_relative_energy_drift"]
    endpoints = summary["statistics"]["entire_main_bank"]["endpoint_residual"]
    minc = summary["statistics"]["entire_main_bank"]["minimum_C"]
    crossing = summary["statistics"]["entire_main_bank"]["total_crossing_time"]
    separation_rows = "\n".join(
        f"| {pair.replace('__', ' / ')} | {value:.12g} |"
        for pair, value in leakage["minimum_cross_split_u_th_separation"].items()
    )
    artifacts = manifest["artifacts"]
    return rf"""# Phase B complete exact orbit-bank report

## Recommendation

**{summary['recommendation']} for Phase C.** All requested main-bank orbits were accepted as complete physical trajectories, all reusable dense references passed reconstruction checks, and the split-leakage audit passed. Phase C was not started.

## Reused validated implementation

The fixed experiment mapping is `b0=1`, `m=2`, `alpha=A=-2`, `omega=W=1`, and `theta=pi/6`, so `sin(theta)^2=S=1/4`. The bank reuses `parameters.py` for immutable parameters, `geometry.py` for $R_m$ and the reduced metric, `dynamics.py` for $C$, energy, velocity branches, and the radial ODE, `observables.py` for $\Omega$, and `physics_gate.py` for the canonical corridor center/half-width and $\xi=(u-c)/d$ transform.

Complete orbits follow the already validated traversal construction: compute $\mathcal E_0$ at $(0,u_{{th}})$, obtain the `branch=+1` state at $x=-17$, and integrate forward to the terminal $x=+17$ event with DOP853 (`rtol=1e-11`, `atol=1e-13`, `max_step=0.2`). The null-boundary and unique throat-crossing events are checked explicitly. No physics function or tolerance was changed.

## Validated domain and mixture

The exact positive timelike throat interval is open at $u_{{th}}=0$ and at the null root `{summary['domain']['positive_null_root']:.16g}`. The bank uses the conservative closed finite-time interval

`[{U_TH_LOW:.2f}, {validated_u_th_high():.16g}]`,

whose high endpoint is the repository's exact positive $\xi=0.99$ throat edge (the earlier `0.905` was rounded). All `{summary['domain']['scan_passed_count']}` scanned points in this closed interval passed; no scan point failed. The lower `0.01` is a deliberate finite-time coverage cutoff, not a mathematical critical boundary.

The exact coverage rule is seeded-jitter stratification (one sample in the central 70% of every equal-width cell), independently permuted per split/stratum:

- broad: full validated interval;
- difficult: `[0.01, 0.30]`;
- low edge: `[0.01, 0.08]`;
- high edge: `[0.80, {validated_u_th_high():.16g}]`.

The seven exact named cases are stored only in the separate stress/reference bank and were excluded from main-bank labels at the `{DUPLICATE_TOLERANCE:.0e}` duplicate tolerance.

## Counts and rejection audit

| split | total | broad | difficult | low edge | high edge |
|:---|---:|---:|---:|---:|---:|
{chr(10).join(count_rows)}

- Requested main candidates: `{summary['requested_main_orbits']}`
- Accepted main trajectories: `{summary['accepted_main_orbits']}`
- Rejected attempts: `{len(rejections)}`
- Dedicated stress/reference trajectories: `7`

## Entire-bank physical diagnostics

| diagnostic | minimum | mean | median | p90 | p99 | maximum |
|:---|---:|---:|---:|---:|---:|---:|
| maximum relative energy drift | {energy['minimum']:.3e} | {energy['mean']:.3e} | {energy['median']:.3e} | {energy['p90']:.3e} | {energy['p99']:.3e} | {energy['maximum']:.3e} |
| minimum C per orbit | {minc['minimum']:.6g} | {minc['mean']:.6g} | {minc['median']:.6g} | {minc['p90']:.6g} | {minc['p99']:.6g} | {minc['maximum']:.6g} |
| crossing time | {crossing['minimum']:.6g} | {crossing['mean']:.6g} | {crossing['median']:.6g} | {crossing['p90']:.6g} | {crossing['p99']:.6g} | {crossing['maximum']:.6g} |

Every orbit reached both endpoints. The maximum endpoint residual was `{endpoints['maximum']:.3e}`. Every orbit remained finite, had positive $C$, finite valid $\xi$, a strictly through-going monotonic branch, one throat crossing, and energy drift below `{ENERGY_DRIFT_LIMIT:.1e}`.

## Dense reference storage

Each bank stores the accepted orbit metadata plus every adaptive DOP853 segment's end time, old state, and seven dense-polynomial coefficient vectors. `wormhole_sciml.phase_b_orbits.evaluate_saved_orbit` evaluates exact $(x,u)$ arrays with the same recurrence as SciPy's DOP853 dense output, and `evaluate_saved_orbit_x_u_xi` adds the canonical $\xi$ transform. The maximum audit discrepancy against the live solver was `{summary['statistics']['entire_main_bank']['dense_reconstruction_max_abs_error']['maximum']:.3e}`. This is a stable numeric representation: no solver objects are pickled and later transition extraction requires no reintegration.

## Leakage and duplicate audit

All orbit-ID intersections are empty, and there are no exact repeated $u_{{th}}$ values across the main splits.

| split pair | minimum cross-split separation in u_th |
|:---|---:|
{separation_rows}

Suspicious near-duplicate threshold: `{leakage['suspicious_threshold']:.1e}`; flagged pairs: `{len(leakage['suspicious_near_duplicates'])}`. Nearby values are expected in this dense one-parameter design, but no exact or numerical duplicate was admitted.

## Diagnostic artifacts

- [Generating-coordinate coverage](figures/generating_coordinate_coverage.png)
- [Physical/numerical distributions](figures/physical_diagnostics.png)
- [Named representative trajectories and incoming zoom](figures/stress_reference_x_xi.png)
- [Difficult versus ordinary family overlay](figures/difficult_vs_ordinary_x_xi.png)
- [Orbit metadata table](orbit_metadata.csv)
- [Domain scan](domain_validation.csv)
- [Leakage audit](leakage_checks.json)
- [Reproducibility manifest](phase_b_manifest.json)

Bank files are `{artifacts['banks']['train']['path']}`, `{artifacts['banks']['validation']['path']}`, `{artifacts['banks']['test']['path']}`, and `{artifacts['banks']['stress_reference']['path']}`. The test bank is marked sealed for later ML evaluation.

## Scope confirmation

No neural network was trained or evaluated. No 96-row finite-time transition extraction, normalization fitting, supervised-pair construction, or other Phase C operation was performed.
"""


def main(*, reuse_existing: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    BANKS.mkdir(parents=True, exist_ok=True)

    domain_rows = domain_validation()
    failed_scan = [row for row in domain_rows if row["classification"] == "failed_inside_selected_interval"]
    write_csv(OUTPUT / "domain_validation.csv", domain_rows)
    if failed_scan:
        raise RuntimeError("selected through-going interval failed its pre-generation scan")

    plan = build_orbit_plan()
    planned_values = {float(row["u_th"]) for row in plan}
    slim: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    bank_artifacts: dict[str, dict[str, Any]] = {}
    all_metadata: list[dict[str, Any]] = []
    all_rejections: list[dict[str, Any]] = []
    reserved = set(planned_values) | set(STRESS_U_TH)
    for split in ("train", "validation", "test"):
        split_plan = [row for row in plan if row["split"] == split]
        path = BANKS / f"phase_b_{split}_orbits.npz"
        if reuse_existing and path.exists():
            arrays, bank_artifacts[split] = load_existing_bank(path)
            verify_reused_plan(arrays, split_plan, split)
            rejections = []
        else:
            arrays, rejections = generate_bank(
                split_plan,
                replacement_seed=MASTER_SEED + {"train": 71, "validation": 72, "test": 73}[split],
                reserved_values=reserved,
            )
            bank_artifacts[split] = save_bank(path, arrays)
        summaries[split] = summarize_bank(arrays)
        all_metadata.extend(metadata_rows(arrays))
        all_rejections.extend(rejections)
        slim[split] = {
            key: arrays[key].copy()
            for key in (
                "orbit_id", "trajectory_stratum", "u_th", "E0",
                "total_crossing_time", "minimum_C", "maximum_relative_energy_drift",
                "endpoint_residual", "dense_reconstruction_max_abs_error",
            )
        }
        del arrays

    stress_plan = [
        {
            "split": "stress_reference",
            "trajectory_stratum": "named_stress",
            "stratum_ordinal": index,
            "u_th": value,
            "generation_seed": MASTER_SEED,
        }
        for index, value in enumerate(STRESS_U_TH)
    ]
    stress_path = BANKS / "phase_b_stress_reference_orbits.npz"
    if reuse_existing and stress_path.exists():
        stress_arrays, bank_artifacts["stress_reference"] = load_existing_bank(stress_path)
        verify_reused_plan(stress_arrays, stress_plan, "stress_reference")
        stress_rejections = []
    else:
        stress_arrays, stress_rejections = generate_bank(
            stress_plan,
            replacement_seed=MASTER_SEED + 99,
            reserved_values=reserved,
        )
        bank_artifacts["stress_reference"] = save_bank(stress_path, stress_arrays)
    if stress_rejections:
        raise RuntimeError("a named stress/reference trajectory was rejected")
    stress_metadata = metadata_rows(stress_arrays)
    del stress_arrays

    write_csv(OUTPUT / "orbit_metadata.csv", all_metadata)
    write_csv(OUTPUT / "stress_reference_metadata.csv", stress_metadata)
    write_csv(OUTPUT / "rejections.csv", all_rejections)

    leakage = leakage_diagnostics(slim)
    json_dump(OUTPUT / "leakage_checks.json", leakage)
    statistics = aggregate_statistics(summaries, slim)

    plot_coverage(slim, FIGURES / "generating_coordinate_coverage.png")
    plot_diagnostics(slim, FIGURES / "physical_diagnostics.png")
    paths = stress_paths(stress_path)
    plot_representatives(paths, FIGURES / "stress_reference_x_xi.png")
    plot_difficult_overlay(paths, FIGURES / "difficult_vs_ordinary_x_xi.png")

    wormhole, spiral = experiment_parameters()
    _lower_root, upper_root = velocity_bounds(0.0, wormhole, spiral)
    requested = sum(sum(counts.values()) for counts in SPLIT_COUNTS.values())
    accepted = sum(summary["count"] for summary in summaries.values())
    passed = bool(
        accepted == requested
        and leakage["passed"]
        and all(row["all_solver_status_terminal_event"] and row["all_monotonic_x"] and row["all_finite_scalars"] for row in summaries.values())
        and statistics["entire_main_bank"]["maximum_relative_energy_drift"]["maximum"] <= ENERGY_DRIFT_LIMIT
        and statistics["entire_main_bank"]["endpoint_residual"]["maximum"] <= ENDPOINT_TOLERANCE
        and not failed_scan
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "recommendation": "PASS" if passed else "FAIL",
        "requested_main_orbits": requested,
        "accepted_main_orbits": accepted,
        "rejected_attempts": len(all_rejections),
        "stress_reference_orbits": len(STRESS_U_TH),
        "domain": {
            "mathematical_positive_throughgoing_interval": f"(0, {float(upper_root):.17g})",
            "positive_null_root": float(upper_root),
            "selected_closed_interval": [U_TH_LOW, validated_u_th_high()],
            "high_endpoint_definition": "positive throat xi=0.99 physical-data edge",
            "scan_passed_count": sum(row["classification"] == "accepted_throughgoing" for row in domain_rows),
            "scan_failed_count": len(failed_scan),
        },
        "statistics": statistics,
        "leakage_passed": leakage["passed"],
        "protocol": {
            "complete_orbits_generated": True,
            "finite_time_transition_rows_extracted": False,
            "transitions_per_orbit": 0,
            "normalization_fitted": False,
            "machine_learning_training_performed": False,
            "machine_learning_evaluation_performed": False,
            "test_bank_sealed_for_later_ml_evaluation": True,
        },
    }
    json_dump(OUTPUT / "phase_b_summary.json", summary)

    config = {
        "master_seed": MASTER_SEED,
        "stratum_specific_seeds": generation_seeds(),
        "split_counts": SPLIT_COUNTS,
        "stratum_bounds": {key: list(value) for key, value in stratum_bounds().items()},
        "stress_u_th": list(STRESS_U_TH),
        "solver": {
            "method": VALIDATION_SOLVER.method,
            "rtol": VALIDATION_SOLVER.rtol,
            "atol": VALIDATION_SOLVER.atol,
            "max_step": 0.2,
            "maximum_time_guard": 500.0,
        },
        "acceptance": {
            "C": "strictly positive on solver knots and midpoints",
            "xi": "finite with 1-abs(xi)>0 on solver knots and midpoints",
            "maximum_relative_energy_drift": ENERGY_DRIFT_LIMIT,
            "endpoint_residual": ENDPOINT_TOLERANCE,
            "throat_velocity_residual": THROAT_TOLERANCE,
            "monotonic_x": True,
            "duplicate_u_th_tolerance": DUPLICATE_TOLERANCE,
        },
        "storage": {
            "format": "compressed NumPy NPZ, allow_pickle=False",
            "dense_reference": "portable DOP853 segment end times, y_old, and 7x2 F coefficient arrays",
            "evaluators": [
                "wormhole_sciml.phase_b_orbits.evaluate_saved_orbit",
                "wormhole_sciml.phase_b_orbits.evaluate_saved_orbit_x_u_xi",
            ],
        },
    }
    json_dump(OUTPUT / "generation_config.json", config)

    artifact_paths = {
        "summary": OUTPUT / "phase_b_summary.json",
        "configuration": OUTPUT / "generation_config.json",
        "orbit_metadata": OUTPUT / "orbit_metadata.csv",
        "stress_metadata": OUTPUT / "stress_reference_metadata.csv",
        "domain_validation": OUTPUT / "domain_validation.csv",
        "rejections": OUTPUT / "rejections.csv",
        "leakage": OUTPUT / "leakage_checks.json",
        "coverage_figure": FIGURES / "generating_coordinate_coverage.png",
        "diagnostic_figure": FIGURES / "physical_diagnostics.png",
        "stress_figure": FIGURES / "stress_reference_x_xi.png",
        "difficult_figure": FIGURES / "difficult_vs_ordinary_x_xi.png",
    }
    manifest = {
        "phase": "B_complete_reference_orbit_banks",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": summary["status"],
        "repository": git_provenance(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "validated_physics_source_manifest": source_manifest(ROOT),
        "phase_b_source_hashes": {
            str(MODULE_PATH.relative_to(ROOT)): file_sha256(MODULE_PATH),
            str(SCRIPT_PATH.relative_to(ROOT)): file_sha256(SCRIPT_PATH),
        },
        "artifacts": {
            "banks": bank_artifacts,
            "files": {
                name: {"path": str(path.resolve()), "file_sha256": file_sha256(path), "bytes": path.stat().st_size}
                for name, path in artifact_paths.items()
            },
        },
        "test_bank_policy": {
            "sealed": True,
            "permitted_current_use": "orbit-level Phase B acceptance, physical diagnostics, and split leakage checks only",
            "prohibited_future_tuning_use": "ML preprocessing, architecture, loss, or model-choice tuning",
        },
        "protocol": summary["protocol"],
    }
    report_path = OUTPUT / "PHASE_B_ORBIT_BANK_REPORT.md"
    report_path.write_text(report_text(summary, manifest, leakage, all_rejections), encoding="utf-8")
    manifest["artifacts"]["files"]["report"] = {
        "path": str(report_path.resolve()),
        "file_sha256": file_sha256(report_path),
        "bytes": report_path.stat().st_size,
    }
    json_dump(OUTPUT / "phase_b_manifest.json", manifest)
    if not passed:
        raise RuntimeError("Phase B validation failed; inspect phase_b_summary.json")
    print(json.dumps({"status": "PASSED", "accepted_main_orbits": accepted, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="resume report/manifest generation only after exact bank-plan identity checks",
    )
    main(reuse_existing=parser.parse_args().reuse_existing)
