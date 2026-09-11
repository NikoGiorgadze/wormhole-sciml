#!/usr/bin/env python3
"""Generate and validate the frozen raw Phase-C finite-time dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from wormhole_sciml.phase_b_orbits import (
    evaluate_saved_orbit_x_u_xi,
    file_sha256,
)
from wormhole_sciml.phase_c_finite_time import (
    ADDITIONAL_ROWS,
    GLOBAL_ROWS,
    HARD_U_TH_MAX,
    PHASE_C_MASTER_SEED,
    ROWS_PER_ORBIT,
    SPLIT_SEEDS,
    X_RIGHT,
    array_content_sha256,
    extract_split,
    load_dataset,
    save_dataset,
    validate_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE_B = ROOT / "output" / "phase_b_complete_orbit_banks"
PHASE_B_BANKS = PHASE_B / "banks"
OUTPUT = ROOT / "output" / "phase_c_finite_time_dataset"
DATASETS = OUTPUT / "datasets"
FIGURES = OUTPUT / "figures"
MODULE = ROOT / "src" / "wormhole_sciml" / "phase_c_finite_time.py"
SCRIPT = Path(__file__).resolve()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def phase_b_bank_paths() -> dict[str, Path]:
    return {
        "train": PHASE_B_BANKS / "phase_b_train_orbits.npz",
        "validation": PHASE_B_BANKS / "phase_b_validation_orbits.npz",
        "test": PHASE_B_BANKS / "phase_b_test_orbits.npz",
        "stress_reference": PHASE_B_BANKS / "phase_b_stress_reference_orbits.npz",
    }


def frozen_phase_b_hashes() -> dict[str, str]:
    manifest = json.loads(
        (PHASE_B / "phase_b_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["status"] != "PASSED":
        raise RuntimeError("Phase B manifest is not PASSED")
    hashes = {
        name: str(entry["file_sha256"])
        for name, entry in manifest["artifacts"]["banks"].items()
    }
    for name, path in phase_b_bank_paths().items():
        if file_sha256(path) != hashes[name]:
            raise RuntimeError(f"frozen Phase-B {name} bank hash mismatch")
    return hashes


def descriptive(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


COVERAGE_KEYS = (
    "orbit_id",
    "trajectory_stratum",
    "sample_group",
    "sample_subtype",
    "hard_orbit",
    "sensitive_anchor",
    "target_at_endpoint",
    "identity_sample",
    "u_th",
    "E0",
    "horizon_fraction",
    "x0",
    "xi0",
    "x1",
    "xi1",
    "s",
    "Delta_x",
    "Delta_xi",
)


def coverage_copy(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: dataset[name].copy() for name in COVERAGE_KEYS}


def verify_reused_dataset(
    dataset: dict[str, np.ndarray], bank: dict[str, np.ndarray], label: str
) -> None:
    expected_ids = np.repeat(bank["orbit_id"], ROWS_PER_ORBIT)
    if not np.array_equal(dataset["orbit_id"], expected_ids):
        raise RuntimeError(f"existing Phase-C {label} rows do not match Phase-B parents")
    if not np.all(dataset["split"] == label):
        raise RuntimeError(f"existing Phase-C {label} split labels are inconsistent")


def generation_artifact_from_existing(
    path: Path, dataset: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "row_count": int(dataset["orbit_id"].size),
        "orbit_count": int(np.unique(dataset["orbit_id"]).size),
        "bytes": path.stat().st_size,
        "content_sha256": array_content_sha256(dataset),
        "file_sha256": file_sha256(path),
        "schema": {name: str(array.dtype) for name, array in dataset.items()},
    }


def concatenate_coverage(
    coverage: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    return {
        name: np.concatenate((coverage["train"][name], coverage["validation"][name]))
        for name in COVERAGE_KEYS
    }


def coverage_summary(combined: dict[str, np.ndarray]) -> dict[str, Any]:
    hard_targeted = combined["sample_group"] == "hard_targeted32"
    ordinary = combined["sample_group"] == "ordinary_global32"
    endpoint = combined["target_at_endpoint"]
    identity = combined["identity_sample"]
    targeted_s = combined["s"][hard_targeted]
    return {
        "scope": "training_plus_validation_only",
        "row_count": int(combined["x0"].size),
        "input_distributions": {
            name: descriptive(combined[name])
            for name in ("x0", "xi0", "E0", "s", "horizon_fraction")
        },
        "target_distributions": {
            name: descriptive(combined[name]) for name in ("Delta_x", "Delta_xi")
        },
        "hard_targeted": {
            "row_count": int(np.sum(hard_targeted)),
            "x0": descriptive(combined["x0"][hard_targeted]),
            "x1": descriptive(combined["x1"][hard_targeted]),
            "s": descriptive(targeted_s),
            "physical_horizon_bins": {
                "short_s_le_5": int(np.sum(targeted_s <= 5.0)),
                "intermediate_5_lt_s_le_20": int(
                    np.sum((targeted_s > 5.0) & (targeted_s <= 20.0))
                ),
                "long_s_gt_20": int(np.sum(targeted_s > 20.0)),
            },
            "subtype_counts": {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(combined["sample_subtype"][hard_targeted], return_counts=True)
                )
            },
        },
        "ordinary_additional": {
            "row_count": int(np.sum(ordinary)),
            "x0": descriptive(combined["x0"][ordinary]),
            "s": descriptive(combined["s"][ordinary]),
            "horizon_fraction": descriptive(combined["horizon_fraction"][ordinary]),
        },
        "identity": {
            "row_count": int(np.sum(identity)),
            "all_targets_exactly_zero": bool(
                np.all(combined["Delta_x"][identity] == 0.0)
                and np.all(combined["Delta_xi"][identity] == 0.0)
            ),
        },
        "endpoint_and_long_time": {
            "exact_endpoint_row_count": int(np.sum(endpoint)),
            "near_endpoint_x1_ge_16p5_count": int(np.sum(combined["x1"] >= 16.5)),
            "horizon_fraction_ge_0p95_count": int(
                np.sum(combined["horizon_fraction"] >= 0.95)
            ),
            "physical_s": descriptive(combined["s"][endpoint]),
        },
    }


def plot_input_distributions(
    coverage: dict[str, dict[str, np.ndarray]], destination: Path
) -> None:
    specifications = (
        ("x0", 80, "$x_0$"),
        ("xi0", 80, r"$\xi_0$"),
        ("E0", 80, r"$\mathcal{E}_0$"),
        ("s", 80, "physical $s$"),
        ("horizon_fraction", 80, "sampling fraction $f$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.3), constrained_layout=True)
    for axis, (name, bins, label) in zip(axes.ravel(), specifications):
        for split in ("train", "validation"):
            axis.hist(
                coverage[split][name], bins=bins, histtype="step", lw=1.3, label=split
            )
        axis.set(xlabel=label, ylabel="rows", title=f"{label} distribution")
        axis.grid(alpha=0.16)
        axis.legend(fontsize=8)
    axes[1, 2].axis("off")
    fig.suptitle("Phase C raw-input coverage · training and validation only")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_target_distributions(
    coverage: dict[str, dict[str, np.ndarray]], destination: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    for split in ("train", "validation"):
        data = coverage[split]
        axes[0, 0].hist(data["Delta_x"], bins=100, histtype="step", lw=1.3, label=split)
        axes[0, 1].hist(data["Delta_xi"], bins=100, histtype="step", lw=1.3, label=split)
        nonzero_x = np.abs(data["Delta_x"][data["Delta_x"] != 0.0])
        nonzero_xi = np.abs(data["Delta_xi"][data["Delta_xi"] != 0.0])
        axes[1, 0].hist(nonzero_x, bins=np.logspace(-5, 2, 90), histtype="step", lw=1.3, label=split)
        axes[1, 1].hist(nonzero_xi, bins=np.logspace(-8, 1, 90), histtype="step", lw=1.3, label=split)
    axes[0, 0].set(xlabel=r"$\Delta x_s$", ylabel="rows", title="Signed accumulated x target")
    axes[0, 1].set(xlabel=r"$\Delta \xi_s$", ylabel="rows", title="Signed accumulated xi target")
    axes[1, 0].set(xlabel=r"nonzero $|\Delta x_s|$", ylabel="rows", title="x-target tails", xscale="log")
    axes[1, 1].set(xlabel=r"nonzero $|\Delta \xi_s|$", ylabel="rows", title="xi-target tails", xscale="log")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
        axis.legend(fontsize=8)
    fig.suptitle("Phase C raw finite-time target distributions · training and validation")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_joint_coverage(combined: dict[str, np.ndarray], destination: Path) -> None:
    rng = np.random.default_rng(PHASE_C_MASTER_SEED + 700)
    count = min(70_000, combined["x0"].size)
    index = rng.choice(combined["x0"].size, count, replace=False)
    pairs = (
        ("x0", "xi0", "$x_0$", r"$\xi_0$"),
        ("x0", "E0", "$x_0$", r"$\mathcal{E}_0$"),
        ("xi0", "E0", r"$\xi_0$", r"$\mathcal{E}_0$"),
        ("x0", "s", "$x_0$", "physical $s$"),
        ("E0", "s", r"$\mathcal{E}_0$", "physical $s$"),
        ("s", "Delta_x", "physical $s$", r"$\Delta x_s$"),
        ("s", "Delta_xi", "physical $s$", r"$\Delta \xi_s$"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17.5, 8.2), constrained_layout=True)
    for axis, (xkey, ykey, xlabel, ylabel) in zip(axes.ravel(), pairs):
        axis.scatter(combined[xkey][index], combined[ykey][index], s=1.5, alpha=0.12, rasterized=True)
        axis.set(xlabel=xlabel, ylabel=ylabel)
        axis.grid(alpha=0.13)
    axes[1, 3].axis("off")
    fig.suptitle("Phase C joint coverage · deterministic train+validation display sample")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_hard_coverage(combined: dict[str, np.ndarray], destination: Path) -> None:
    mask = combined["sample_group"] == "hard_targeted32"
    subtypes = (
        "within_sensitive",
        "sensitive_exit",
        "sensitive_to_central",
        "sensitive_to_long_outgoing",
    )
    colors = ("#277da1", "#f8961e", "#43aa8b", "#9b5de5")
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), constrained_layout=True)
    axes[0, 0].hist(combined["x0"][mask], bins=64, histtype="step", lw=1.5)
    axes[0, 0].set(title="Sensitive anchor coverage", xlabel="$x_0$", ylabel="rows")
    for subtype, color in zip(subtypes, colors, strict=True):
        selected = combined["sample_subtype"] == subtype
        axes[0, 1].hist(combined["x1"][selected], bins=55, histtype="step", lw=1.4, color=color, label=subtype)
        axes[1, 0].hist(combined["s"][selected], bins=65, histtype="step", lw=1.4, color=color, label=subtype)
    axes[0, 1].set(title="Four targeted x1 classes", xlabel="$x_1$", ylabel="rows")
    axes[1, 0].set(title="Targeted physical horizons", xlabel="physical $s$", ylabel="rows")
    horizons = combined["s"][mask]
    axes[1, 1].bar(
        ("short ≤5", "intermediate 5–20", "long >20"),
        (np.sum(horizons <= 5), np.sum((horizons > 5) & (horizons <= 20)), np.sum(horizons > 20)),
        color=("#90be6d", "#f9c74f", "#f94144"),
    )
    axes[1, 1].set(title="Targeted horizon regimes", ylabel="rows")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(r"Hard-orbit enrichment ($u_{th}\leq0.30$) · train+validation")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_ordinary_coverage(combined: dict[str, np.ndarray], destination: Path) -> None:
    mask = combined["sample_group"] == "ordinary_global32"
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), constrained_layout=True)
    for axis, key, label in zip(
        axes,
        ("x0", "horizon_fraction", "s"),
        ("$x_0$", "sampling fraction $f$", "physical $s$"),
    ):
        axis.hist(combined[key][mask], bins=65, histtype="step", lw=1.5)
        axis.set(xlabel=label, ylabel="rows")
        axis.grid(alpha=0.16)
    fig.suptitle(r"Ordinary-orbit additional global coverage ($u_{th}>0.30$) · train+validation")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def plot_stress_overlays(
    bank: dict[str, np.ndarray], dataset: dict[str, np.ndarray], destination: Path
) -> None:
    requested = (0.05, 0.15, 0.30, 0.50, 0.90)
    hard_colors = {
        "within_sensitive": "#277da1",
        "sensitive_exit": "#f8961e",
        "sensitive_to_central": "#43aa8b",
        "sensitive_to_long_outgoing": "#9b5de5",
    }
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 15.0), constrained_layout=True)
    for axis, value in zip(axes.ravel(), requested):
        orbit_index = int(np.flatnonzero(np.isclose(bank["u_th"], value, rtol=0, atol=1e-14))[0])
        start, stop = orbit_index * ROWS_PER_ORBIT, (orbit_index + 1) * ROWS_PER_ORBIT
        times = np.linspace(0.0, float(bank["t_right"][orbit_index]), 1601)
        exact = evaluate_saved_orbit_x_u_xi(bank, orbit_index, times)
        axis.plot(exact[:, 0], exact[:, 2], color="black", lw=2.2, label="exact orbit")
        global_rows = np.arange(start, start + GLOBAL_ROWS, 8)
        for row in global_rows:
            axis.plot([dataset["x0"][row], dataset["x1"][row]], [dataset["xi0"][row], dataset["xi1"][row]], color="0.6", alpha=0.35, lw=0.8)
            axis.scatter(dataset["x0"][row], dataset["xi0"][row], s=12, color="0.45", alpha=0.6)
            axis.scatter(dataset["x1"][row], dataset["xi1"][row], s=14, marker="^", color="0.45", alpha=0.6)
        additional = np.arange(start + GLOBAL_ROWS, stop)
        for row in additional:
            subtype = str(dataset["sample_subtype"][row])
            color = hard_colors.get(subtype, "#4d908e")
            axis.plot([dataset["x0"][row], dataset["x1"][row]], [dataset["xi0"][row], dataset["xi1"][row]], color=color, alpha=0.28, lw=0.9)
            axis.scatter(dataset["x0"][row], dataset["xi0"][row], s=14, color=color, alpha=0.75)
            axis.scatter(dataset["x1"][row], dataset["xi1"][row], s=18, marker="^", color=color, alpha=0.75)
        axis.axvspan(-17, -8.5, color="#f4a261", alpha=0.07)
        axis.set(title=rf"$u_{{th}}={value:.2f}$", xlabel="$x$", ylabel=r"$\xi$")
        axis.grid(alpha=0.14)
    legend_axis = axes[2, 1]
    legend_axis.axis("off")
    legend_axis.plot([], [], color="black", lw=2.2, label="exact orbit")
    legend_axis.plot([], [], color="0.55", lw=1.0, label="displayed global pairs")
    for subtype, color in hard_colors.items():
        legend_axis.plot([], [], color=color, lw=2, label=subtype)
    legend_axis.plot([], [], color="#4d908e", lw=2, label="ordinary additional global")
    legend_axis.scatter([], [], color="black", marker="o", label="anchor")
    legend_axis.scatter([], [], color="black", marker="^", label="target")
    legend_axis.legend(loc="center", fontsize=10)
    fig.suptitle("Phase C sampling audit on separate stress/reference orbits")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def leakage_audit(
    datasets: dict[str, dict[str, np.ndarray]],
    stress: dict[str, np.ndarray],
) -> dict[str, Any]:
    orbit_sets = {
        split: set(map(str, dataset["orbit_id"]))
        for split, dataset in datasets.items()
    }
    transition_sets = {
        split: set(map(str, dataset["transition_id"]))
        for split, dataset in datasets.items()
    }
    orbit_intersections: dict[str, list[str]] = {}
    transition_intersections: dict[str, list[str]] = {}
    names = list(datasets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            label = f"{left}__{right}"
            orbit_intersections[label] = sorted(orbit_sets[left] & orbit_sets[right])
            transition_intersections[label] = sorted(
                transition_sets[left] & transition_sets[right]
            )
    stress_orbits = set(map(str, stress["orbit_id"]))
    return {
        "orbit_id_intersections": orbit_intersections,
        "transition_id_intersections": transition_intersections,
        "stress_reference_intersection_with_main": sorted(
            stress_orbits & set().union(*orbit_sets.values())
        ),
        "passed": all(not value for value in orbit_intersections.values())
        and all(not value for value in transition_intersections.values())
        and not (stress_orbits & set().union(*orbit_sets.values())),
    }


def write_orbit_index(
    path: Path, banks: dict[str, dict[str, np.ndarray]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "orbit_id", "split", "trajectory_stratum", "u_th", "E0",
                "hard_orbit", "transition_row_count",
            ),
        )
        writer.writeheader()
        for split, bank in banks.items():
            for index, orbit_id in enumerate(bank["orbit_id"]):
                writer.writerow(
                    {
                        "orbit_id": str(orbit_id),
                        "split": split,
                        "trajectory_stratum": str(bank["trajectory_stratum"][index]),
                        "u_th": float(bank["u_th"][index]),
                        "E0": float(bank["E0"][index]),
                        "hard_orbit": bool(bank["u_th"][index] <= HARD_U_TH_MAX),
                        "transition_row_count": ROWS_PER_ORBIT,
                    }
                )


def report_text(
    summary: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    leakage: dict[str, Any],
) -> str:
    audits = summary["split_audits"]
    count_rows = []
    physics_rows = []
    for split in ("train", "validation", "test"):
        row = audits[split]
        count_rows.append(
            f"| {split} | {row['orbit_count']} | {row['row_count']} | {row['global_row_count']} | {row['hard_orbit_row_count']} | {row['hard_targeted_row_count']} | {row['ordinary_additional_row_count']} | {row['hard_orbit_count']} |"
        )
        physics_rows.append(
            f"| {split} | {row['minimum_C']:.6g} | {row['minimum_one_minus_abs_xi']:.6g} | {row['maximum_relative_energy_error']:.3e} | {row['maximum_x_inversion_residual']:.3e} | {row['nan_inf_count']} | {row['invalid_time_count']} |"
        )
    coverage = summary["coverage_gate"]
    target = coverage["target_distributions"]
    inputs = coverage["input_distributions"]
    artifact_lines = "\n".join(
        f"- `{name}`: `{entry['path']}`" for name, entry in artifacts.items()
    )
    return rf"""# Phase C raw finite-time dataset report

## Decision

**{summary['recommendation']} for preprocessing/training.** The raw finite-time dataset contains exactly 96 transitions per immutable parent orbit, passes every physics/integrity/duplication/leakage gate, and is frozen without normalization or neural-network work.

## Frozen Phase B inputs reused

Phase C reads the accepted Phase B train/validation/test and separate stress-reference NPZ banks without modifying them. State evaluation uses the stored DOP853 polynomial through `evaluate_saved_orbit` / `evaluate_saved_orbit_x_u_xi`. Desired x anchors and hard-family x targets are inverted to time using monotonic segment bracketing and vectorized Newton refinement on those same saved coefficients. No ODE was reintegrated.

## Exact 96-row algorithm

Every orbit contributes 64 global rows plus 32 conditional rows. The global component uses one seeded interior anchor from each of 64 equal x bins on `[-17,17]`, paired by an independent seeded permutation with the exact requested horizon-fraction composition: 4 identity, 8 very short, 12 short/intermediate, 12 medium, 12 medium/long, 8 long, 4 near-endpoint, and 4 exact endpoint rows.

`hard_orbit` is defined exclusively by the physical orbit label `u_th <= 0.30`. Its extra 32 rows use eight seeded anchors on the sensitive incoming-left branch `[-17,-8.5]`, each paired with one target in each class: within-sensitive, sensitive-exit `(-8.5,-6]`, sensitive-to-central `[-4,+2]`, and sensitive-to-long-outgoing `[+8,+17]`. Every hard orbit has targets on both sides of the throat and exactly one hard-targeted `+17` endpoint row.

For `u_th > 0.30`, the extra 32 rows are an independent global x-stratified design with the prescribed short-to-endpoint fraction composition and no extra identity rows. `u_th` and all sampling fields remain diagnostics; intended inputs are only `(x0, xi0, E0, s)` and targets only `(Delta_x, Delta_xi)`.

## Counts

| split | parent orbits | rows | global64 rows | all hard-orbit rows | hard targeted rows | ordinary additional rows | hard orbits |
|:---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(count_rows)}

Total main rows: `{summary['total_main_rows']}`. Every orbit has exactly `{ROWS_PER_ORBIT}` rows. Training and validation contain `{coverage['identity']['row_count']}` exact identity examples and `{coverage['endpoint_and_long_time']['exact_endpoint_row_count']}` exact-endpoint targets. Test counts and integrity were checked structurally; test distributions were not used for sampling decisions or coverage tuning.

## Train+validation coverage gate

| quantity | minimum | median | p90 | p99 | maximum |
|:---|---:|---:|---:|---:|---:|
| x0 | {inputs['x0']['minimum']:.6g} | {inputs['x0']['median']:.6g} | {inputs['x0']['p90']:.6g} | {inputs['x0']['p99']:.6g} | {inputs['x0']['maximum']:.6g} |
| xi0 | {inputs['xi0']['minimum']:.6g} | {inputs['xi0']['median']:.6g} | {inputs['xi0']['p90']:.6g} | {inputs['xi0']['p99']:.6g} | {inputs['xi0']['maximum']:.6g} |
| E0 | {inputs['E0']['minimum']:.6g} | {inputs['E0']['median']:.6g} | {inputs['E0']['p90']:.6g} | {inputs['E0']['p99']:.6g} | {inputs['E0']['maximum']:.6g} |
| physical s | {inputs['s']['minimum']:.6g} | {inputs['s']['median']:.6g} | {inputs['s']['p90']:.6g} | {inputs['s']['p99']:.6g} | {inputs['s']['maximum']:.6g} |
| Delta x | {target['Delta_x']['minimum']:.6g} | {target['Delta_x']['median']:.6g} | {target['Delta_x']['p90']:.6g} | {target['Delta_x']['p99']:.6g} | {target['Delta_x']['maximum']:.6g} |
| Delta xi | {target['Delta_xi']['minimum']:.6g} | {target['Delta_xi']['median']:.6g} | {target['Delta_xi']['p90']:.6g} | {target['Delta_xi']['p99']:.6g} | {target['Delta_xi']['maximum']:.6g} |

Hard-targeted train+validation rows number `{coverage['hard_targeted']['row_count']}` with physical horizons split into `{coverage['hard_targeted']['physical_horizon_bins']['short_s_le_5']}` short, `{coverage['hard_targeted']['physical_horizon_bins']['intermediate_5_lt_s_le_20']}` intermediate, and `{coverage['hard_targeted']['physical_horizon_bins']['long_s_gt_20']}` long transitions. Ordinary additional-global rows number `{coverage['ordinary_additional']['row_count']}` and cover the entire orbit.

## Physics and integrity audit

| split | minimum C | minimum 1-|xi| | maximum relative energy error | maximum x→t residual | NaN/Inf | invalid times |
|:---|---:|---:|---:|---:|---:|---:|
{chr(10).join(physics_rows)}

All identities have exactly `s=0`, unchanged source/target state, and exactly zero targets. All endpoint rows use the parent `t_right` and reach `x=+17` to Phase B precision. All hard targeted rows have positive elapsed time and satisfy their target-class x ranges.

## Duplicate and leakage audit

Every split has zero duplicate `(orbit_id,t0,t1)` pairs and zero near-duplicate pairs at the `1e-10` time-coordinate threshold. The minimum within-orbit maximum-coordinate separations are train `{audits['train']['minimum_within_orbit_max_time_coordinate_separation']:.6g}`, validation `{audits['validation']['minimum_within_orbit_max_time_coordinate_separation']:.6g}`, and test `{audits['test']['minimum_within_orbit_max_time_coordinate_separation']:.6g}`.

All row-level parent-orbit and transition-ID intersections between train, validation, and test are empty. The separate stress-reference rows have no parent-orbit overlap with the main dataset. Leakage gate: `{leakage['passed']}`.

## Coverage and visual-audit figures

- [Input distributions](figures/input_distributions_train_validation.png)
- [Target distributions](figures/target_distributions_train_validation.png)
- [Joint input/target coverage](figures/joint_coverage_train_validation.png)
- [Hard-family targeted coverage](figures/hard_family_coverage_train_validation.png)
- [Ordinary-orbit additional coverage](figures/ordinary_orbit_coverage_train_validation.png)
- [Stress/reference transition overlays](figures/stress_reference_sampling_overlays.png)

## Frozen artifacts

{artifact_lines}

The master sampling seed is `{PHASE_C_MASTER_SEED}` and split seeds are `{SPLIT_SEEDS}`. File and deterministic array-content hashes are recorded in the manifest. Phase B bank hashes were identical before and after extraction.

## Scope confirmation

No normalization statistics were fitted, no physical column was overwritten, no DataLoader or neural network was constructed, and no training, loss inspection, or prediction evaluation occurred. The sealed test transition table was generated only to freeze its deterministic definition and was used only for schema, counts, parent IDs, timing, physicality, finiteness, duplication, leakage, and hash checks.
"""


def main(*, reuse_existing: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    phase_b_before = frozen_phase_b_hashes()
    bank_paths = phase_b_bank_paths()
    dataset_paths = {
        "train": DATASETS / "phase_c_train_raw.npz",
        "validation": DATASETS / "phase_c_validation_raw.npz",
        "test": DATASETS / "phase_c_test_sealed_raw.npz",
        "stress_reference": DATASETS / "phase_c_stress_reference_diagnostic_raw.npz",
    }
    banks: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, np.ndarray]] = {}
    main_for_leakage: dict[str, dict[str, np.ndarray]] = {}

    for split in ("train", "validation", "test"):
        bank = load_bank(bank_paths[split])
        banks[split] = {
            key: bank[key]
            for key in ("orbit_id", "trajectory_stratum", "u_th", "E0")
        }
        path = dataset_paths[split]
        if reuse_existing and path.exists():
            dataset = load_dataset(path)
            verify_reused_dataset(dataset, bank, split)
            artifacts[split] = generation_artifact_from_existing(path, dataset)
        else:
            dataset = extract_split(bank)
            artifacts[split] = save_dataset(path, dataset)
        audits[split] = validate_dataset(dataset, bank)
        main_for_leakage[split] = {
            "orbit_id": dataset["orbit_id"].copy(),
            "transition_id": dataset["transition_id"].copy(),
        }
        if split in ("train", "validation"):
            coverage[split] = coverage_copy(dataset)
        del dataset
        del bank

    stress_bank = load_bank(bank_paths["stress_reference"])
    stress_path = dataset_paths["stress_reference"]
    if reuse_existing and stress_path.exists():
        stress_dataset = load_dataset(stress_path)
        verify_reused_dataset(stress_dataset, stress_bank, "stress_reference")
        artifacts["stress_reference"] = generation_artifact_from_existing(
            stress_path, stress_dataset
        )
    else:
        stress_dataset = extract_split(stress_bank)
        artifacts["stress_reference"] = save_dataset(stress_path, stress_dataset)
    audits["stress_reference"] = validate_dataset(stress_dataset, stress_bank)

    leakage = leakage_audit(main_for_leakage, stress_dataset)
    json_dump(OUTPUT / "leakage_audit.json", leakage)
    write_orbit_index(OUTPUT / "orbit_transition_index.csv", banks)
    combined = concatenate_coverage(coverage)
    substantive = coverage_summary(combined)

    plot_input_distributions(coverage, FIGURES / "input_distributions_train_validation.png")
    plot_target_distributions(coverage, FIGURES / "target_distributions_train_validation.png")
    plot_joint_coverage(combined, FIGURES / "joint_coverage_train_validation.png")
    plot_hard_coverage(combined, FIGURES / "hard_family_coverage_train_validation.png")
    plot_ordinary_coverage(combined, FIGURES / "ordinary_orbit_coverage_train_validation.png")
    plot_stress_overlays(stress_bank, stress_dataset, FIGURES / "stress_reference_sampling_overlays.png")

    phase_b_after = {name: file_sha256(path) for name, path in bank_paths.items()}
    phase_b_unchanged = phase_b_before == phase_b_after
    total_main_rows = sum(audits[split]["row_count"] for split in ("train", "validation", "test"))
    passed = bool(
        total_main_rows == 589_824
        and all(audits[name]["passed"] for name in audits)
        and leakage["passed"]
        and phase_b_unchanged
    )
    summary = {
        "status": "PASSED" if passed else "FAILED",
        "recommendation": "PASS" if passed else "FAIL",
        "total_main_rows": total_main_rows,
        "split_audits": audits,
        "coverage_gate": substantive,
        "leakage_passed": leakage["passed"],
        "phase_b_hashes_before": phase_b_before,
        "phase_b_hashes_after": phase_b_after,
        "phase_b_unchanged": phase_b_unchanged,
        "protocol": {
            "ode_reintegrated": False,
            "raw_physical_rows_extracted": True,
            "rows_per_orbit": ROWS_PER_ORBIT,
            "normalization_fitted": False,
            "preprocessing_performed": False,
            "neural_network_constructed": False,
            "training_performed": False,
            "prediction_evaluation_performed": False,
            "test_used_for_sampling_tuning": False,
            "test_structural_integrity_only": True,
        },
    }
    json_dump(OUTPUT / "phase_c_summary.json", summary)

    config = {
        "master_seed": PHASE_C_MASTER_SEED,
        "split_seeds": SPLIT_SEEDS,
        "orbit_seed_derivation": "uint64 little-endian first 8 bytes of SHA256(master:split_seed:orbit_id:design)",
        "rows_per_orbit": ROWS_PER_ORBIT,
        "global_rows": GLOBAL_ROWS,
        "additional_rows": ADDITIONAL_ROWS,
        "hard_orbit_rule": "u_th <= 0.30",
        "sensitive_incoming_left_branch": [-17.0, -8.5],
        "network_inputs_reserved_for_next_phase": ["x0", "xi0", "E0", "s"],
        "network_targets_reserved_for_next_phase": ["Delta_x", "Delta_xi"],
        "format": "compressed NumPy NPZ, float64 physical columns, allow_pickle=False",
        "test_policy": "generated and structurally/physically validated; excluded from substantive coverage plots and design tuning",
    }
    json_dump(OUTPUT / "phase_c_sampling_config.json", config)

    auxiliary_paths = {
        "summary": OUTPUT / "phase_c_summary.json",
        "sampling_config": OUTPUT / "phase_c_sampling_config.json",
        "leakage_audit": OUTPUT / "leakage_audit.json",
        "orbit_index": OUTPUT / "orbit_transition_index.csv",
        "input_figure": FIGURES / "input_distributions_train_validation.png",
        "target_figure": FIGURES / "target_distributions_train_validation.png",
        "joint_figure": FIGURES / "joint_coverage_train_validation.png",
        "hard_figure": FIGURES / "hard_family_coverage_train_validation.png",
        "ordinary_figure": FIGURES / "ordinary_orbit_coverage_train_validation.png",
        "stress_overlay_figure": FIGURES / "stress_reference_sampling_overlays.png",
    }
    report_path = OUTPUT / "PHASE_C_FINITE_TIME_DATASET_REPORT.md"
    report_path.write_text(report_text(summary, artifacts, leakage), encoding="utf-8")
    auxiliary_paths["report"] = report_path
    manifest = {
        "phase": "C_raw_finite_time_transition_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": summary["status"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "phase_b_input_bank_hashes": phase_b_before,
        "phase_b_inputs_unchanged": phase_b_unchanged,
        "phase_c_source_hashes": {
            str(MODULE.relative_to(ROOT)): file_sha256(MODULE),
            str(SCRIPT.relative_to(ROOT)): file_sha256(SCRIPT),
        },
        "datasets": artifacts,
        "files": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "file_sha256": file_sha256(path),
            }
            for name, path in auxiliary_paths.items()
        },
        "protocol": summary["protocol"],
        "test_dataset_policy": {
            "sealed": True,
            "substantive_statistics_or_plots_used": False,
            "allowed_checks_completed": [
                "schema", "row_count", "orbit_ids", "time_order", "exact_state_consistency",
                "physical_validity", "finite_values", "duplicates", "split_leakage", "hashes",
            ],
        },
    }
    json_dump(OUTPUT / "phase_c_manifest.json", manifest)
    if not passed:
        raise RuntimeError("Phase C gate failed; inspect phase_c_summary.json")
    print(json.dumps({"status": "PASSED", "total_main_rows": total_main_rows, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse raw split artifacts only after exact parent and full integrity checks",
    )
    main(reuse_existing=parser.parse_args().reuse_existing)
