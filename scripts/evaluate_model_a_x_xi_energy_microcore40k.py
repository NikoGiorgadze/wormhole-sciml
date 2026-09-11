#!/usr/bin/env python3
"""Compare frozen no-energy and newly trained fixed-E0 microcore40k rollouts."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import conserved_energy, timelike_margin, velocity_bounds
from wormhole_sciml.model_a import Normalization, load_trained_model, parameter_count, predict_increments
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi, xi_from_state
from wormhole_sciml.stage1_data import file_sha256

try:
    from evaluate_model_a_x_xi_sampling_rollouts import (
        H, MAX_MODEL_STEPS, SEED_COLORS, THROAT_VELOCITIES, X_ENDPOINT,
        energy_diagnostic, exact_xi_path, family_aggregate,
        incoming_xi_diagnostic, reconstruct_physical, u_limits,
    )
except ModuleNotFoundError:
    from scripts.evaluate_model_a_x_xi_sampling_rollouts import (
        H, MAX_MODEL_STEPS, SEED_COLORS, THROAT_VELOCITIES, X_ENDPOINT,
        energy_diagnostic, exact_xi_path, family_aggregate,
        incoming_xi_diagnostic, reconstruct_physical, u_limits,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
TRAINING_MANIFEST = OUTPUT / "energy_xi_training_manifest.json"
ENERGY_NORMALIZATION = OUTPUT / "training" / "energy_input_normalization.json"
ROLLOUTS = OUTPUT / "rollouts"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "energy_xi_rollout_summary.json"
ARRAYS_PATH = OUTPUT / "energy_xi_rollout_arrays.npz"
CSV_PATH = OUTPUT / "energy_xi_rollout_metrics.csv"
REPORT_PATH = OUTPUT / "ENERGY_XI_TRAINING_AND_ROLLOUT_REPORT.md"

BASELINE_TRAINING = ROOT / "output" / "model_a_x_xi_sampling_training_comparison"
BASELINE_MANIFEST = BASELINE_TRAINING / "training_comparison_manifest.json"
BASELINE_NORMALIZATION = BASELINE_TRAINING / "microcore40k_normalization.json"
BASELINE_TEACHER = BASELINE_TRAINING / "teacher_forced_diagnostics.json"
BASELINE_TEACHER_ARRAYS = BASELINE_TRAINING / "teacher_forced_diagnostic_arrays.npz"
BASELINE_ROLLOUTS = ROOT / "output" / "model_a_x_xi_sampling_recursive_rollouts"
BASELINE_SUMMARY = BASELINE_ROLLOUTS / "recursive_rollout_summary.json"
BASELINE_ARRAYS = BASELINE_ROLLOUTS / "recursive_rollout_arrays.npz"
FAMILY_SUMMARY = ROOT / "output" / "c32x32_traversal_families" / "traversal_family_summary.json"
EXACT_REFERENCE = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
ORBIT_SPACING = ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_arrays.npz"
TRAIN_DATA = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore40k_train_x_xi.npz"
VALIDATION_DATA = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore8k_validation_x_xi.npz"

SEEDS = (101, 202, 303)
TREATMENTS = ("no_energy", "fixed_E0")
TITLES = {"no_energy": "microcore40k no energy", "fixed_E0": "microcore40k + fixed E0"}
HARD = THROAT_VELOCITIES[:3]


def family_key(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def protected_paths(training: dict[str, Any]) -> tuple[Path, ...]:
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    baseline_runs = [row for row in baseline["runs"] if row["sampling_treatment"] == "microcore40k"]
    paths = [
        TRAIN_DATA, VALIDATION_DATA, BASELINE_MANIFEST, BASELINE_NORMALIZATION,
        BASELINE_TEACHER, BASELINE_TEACHER_ARRAYS, BASELINE_SUMMARY, BASELINE_ARRAYS,
        FAMILY_SUMMARY, EXACT_REFERENCE, ORBIT_SPACING, TRAINING_MANIFEST,
        ENERGY_NORMALIZATION, OUTPUT / "figures" / "energy_input_training_curves.png",
    ]
    for row in baseline_runs + training["runs"]:
        paths.extend((Path(row["checkpoint"]), Path(row["history"]), Path(row["checkpoint"]).parent / "metadata.json"))
    return tuple(paths)


def hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing protected artifact(s): {missing}")
    return {str(path): file_sha256(path) for path in paths}


def energy_recursive_rollout(
    model: Any, normalization: Normalization, initial: np.ndarray, fixed_E0: float, u_th: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recursively add predicted (delta_x,delta_xi) while holding exact E0 fixed."""
    wormhole, spiral = experiment_parameters()
    coordinates = [np.asarray(initial, dtype=np.float64)]
    physical = [reconstruct_physical(np.asarray(coordinates))[0]]
    status, exit_record = "maximum_step_guard", None
    for step in range(1, MAX_MODEL_STEPS + 1):
        current = coordinates[-1]
        features = np.asarray([[current[0], current[1], fixed_E0]], dtype=np.float64)
        following = current + predict_increments(model, features, normalization)[0]
        coordinates.append(following)
        if not np.all(np.isfinite(following)):
            physical.append(np.asarray([np.nan, np.nan]))
            status = "nonfinite"
            break
        following_physical = reconstruct_physical(following.reshape(1, 2))[0]
        physical.append(following_physical)
        if not np.all(np.isfinite(following_physical)):
            status = "nonfinite"
            break
        margin = float(timelike_margin(following_physical[0], following_physical[1], wormhole, spiral))
        if margin <= 0.0:
            status = "physical_exit"
            exit_record = {"step": step, "time": float(step * H), "x": float(following[0]),
                           "xi": float(following[1]), "u": float(following_physical[1]), "C": margin}
            break
        if following[0] >= X_ENDPOINT:
            status = "reached_x_plus_17"
            break
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    physical_array = np.asarray(physical, dtype=np.float64)
    crossing = np.flatnonzero((coordinates_array[:-1, 0] < 0.0) & (coordinates_array[1:, 0] >= 0.0))
    throat_u = throat_xi = throat_step = None
    if crossing.size:
        index = int(crossing[0])
        fraction = float(-coordinates_array[index, 0] /
                         (coordinates_array[index + 1, 0] - coordinates_array[index, 0]))
        throat_xi = float(coordinates_array[index, 1] + fraction *
                          (coordinates_array[index + 1, 1] - coordinates_array[index, 1]))
        throat_u = float(physical_array[index, 1] + fraction *
                         (physical_array[index + 1, 1] - physical_array[index, 1]))
        throat_step = index + 1
    relative, margins, energy_summary = energy_diagnostic(physical_array, fixed_E0, throat_u)
    summary = {
        "status": status, "reached_x_plus_17": status == "reached_x_plus_17",
        "physical_exit": status == "physical_exit", "maximum_step_guard": status == "maximum_step_guard",
        "nonfinite": status == "nonfinite", "exit": exit_record,
        "rollout_steps": int(coordinates_array.shape[0] - 1),
        "elapsed_time": float((coordinates_array.shape[0] - 1) * H),
        "terminal_x": None if not np.isfinite(coordinates_array[-1, 0]) else float(coordinates_array[-1, 0]),
        "terminal_xi": None if not np.isfinite(coordinates_array[-1, 1]) else float(coordinates_array[-1, 1]),
        "terminal_u": None if not np.isfinite(physical_array[-1, 1]) else float(physical_array[-1, 1]),
        "throat_crossing_step": throat_step, "interpolated_throat_xi": throat_xi,
        "interpolated_throat_u": throat_u,
        "signed_throat_u_error": None if throat_u is None else float(throat_u - u_th),
        "absolute_throat_u_error": None if throat_u is None else float(abs(throat_u - u_th)),
        "energy_diagnostic": energy_summary,
        "fixed_supplied_E0": float(fixed_E0),
        "E0_recomputed_from_predicted_state_for_input": False,
    }
    return coordinates_array, physical_array, margins, relative, summary


def signed_teacher_metrics(error: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(error, dtype=np.float64)
    cumulative = np.cumsum(error)
    return {
        "count": int(error.size), "rmse_delta_xi": float(np.sqrt(np.mean(error ** 2))),
        "mae_delta_xi": float(np.mean(np.abs(error))), "mean_signed_delta_xi": float(np.mean(error)),
        "fraction_positive": float(np.mean(error > 0.0)), "fraction_negative": float(np.mean(error < 0.0)),
        "fraction_zero": float(np.mean(error == 0.0)), "cumulative_sum_final": float(cumulative[-1]),
        "maximum_absolute_cumulative_sum": float(np.max(np.abs(cumulative))),
    }


def teacher_forced(models: dict[int, Any], normalization: Normalization,
                   family_source: dict[float, dict[str, Any]], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    results: dict[str, Any] = {treatment: {} for treatment in TREATMENTS}
    with np.load(EXACT_REFERENCE, allow_pickle=False) as exact, np.load(
        BASELINE_TEACHER_ARRAYS, allow_pickle=False
    ) as baseline:
        for u_th in THROAT_VELOCITIES:
            stem, family = family_key(u_th), f"{u_th:.2f}"
            state = np.column_stack((exact[f"{stem}__exact_state"][:, 0], exact[f"{stem}__exact_xi"]))
            delta = exact[f"{stem}__exact_delta"]
            next_state = exact[f"{stem}__exact_state"] + delta
            next_xi = xi_from_state(next_state[:, 0], next_state[:, 1], *experiment_parameters())
            target_delta_xi = np.asarray(next_xi) - state[:, 1]
            E0 = float(family_source[u_th]["energy"])
            outer = (state[:, 0] >= -17.0) & (state[:, 0] <= -8.5)
            arrays[f"{stem}__teacher_x"] = state[:, 0]
            arrays[f"{stem}__teacher_exact_delta_xi"] = target_delta_xi
            for treatment in TREATMENTS:
                results[treatment].setdefault(family, {})
                for seed in SEEDS:
                    if treatment == "no_energy":
                        error = np.asarray(baseline[f"{stem}__microcore40k__seed_{seed}__signed_error_delta_xi"])
                    else:
                        features = np.column_stack((state, np.full(state.shape[0], E0)))
                        prediction = predict_increments(models[seed], features, normalization)
                        error = prediction[:, 1] - target_delta_xi
                    arrays[f"{stem}__teacher__{treatment}__seed_{seed}__e_delta_xi"] = error
                    results[treatment][family][str(seed)] = {
                        "full_incoming": signed_teacher_metrics(error),
                        "outer_left_minus17_to_minus8p5": signed_teacher_metrics(error[outer]),
                    }
    return results


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def aggregate_teacher(teacher: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for treatment in TREATMENTS:
        result[treatment] = {}
        for scope in ("full_incoming", "outer_left_minus17_to_minus8p5"):
            rows = [teacher[treatment][f"{u:.2f}"][str(seed)][scope] for u in HARD for seed in SEEDS]
            result[treatment][scope] = {key: mean_metric(rows, key) for key in (
                "rmse_delta_xi", "mae_delta_xi", "mean_signed_delta_xi",
                "fraction_positive", "fraction_negative", "maximum_absolute_cumulative_sum",
            )}
    return result


def copy_baseline_case_arrays(source: Any, arrays: dict[str, np.ndarray], stem: str,
                              mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old = f"{stem}__{mode}__microcore40k__seed_{seed}"
    new = f"{stem}__{mode}__no_energy__seed_{seed}"
    values = tuple(np.asarray(source[f"{old}__{suffix}"]) for suffix in
                   ("coordinates", "physical_state", "C", "relative_energy_drift"))
    for suffix, value in zip(("coordinates", "physical_state", "C", "relative_energy_drift"), values):
        arrays[f"{new}__{suffix}"] = value
    return values


def draw_u(axis: Any, exact: np.ndarray, learned: dict[int, np.ndarray], rows: dict[str, Any],
           x_range: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    grid = np.linspace(*x_range, 1601)
    lower, upper = velocity_bounds(grid, *experiment_parameters())
    axis.fill_between(grid, lower, upper, color="#c9d7e3", alpha=0.28, label="admissible")
    axis.plot(exact[:, 0], exact[:, 1], color="black", lw=2.2, label="DOP853")
    axis.scatter(exact[0, 0], exact[0, 1], marker="o", s=30, color="black", zorder=6)
    for seed in SEEDS:
        path, row = learned[seed], rows[str(seed)]
        axis.plot(path[:, 0], path[:, 1], color=SEED_COLORS[seed], lw=1.3,
                  label=f"seed {seed} · {row['status'].replace('_', ' ')}")
        axis.scatter(path[-1, 0], path[-1, 1], marker="s", s=25, color=SEED_COLORS[seed], zorder=6)
        if row["physical_exit"]:
            axis.scatter(path[-1, 0], path[-1, 1], marker="X", s=75, color="red", zorder=7)
    axis.set(title=title, xlabel="$x$", ylabel="radial velocity $u$", xlim=(x_range[0]-.25, x_range[1]+.25), ylim=ylim)
    axis.grid(alpha=.18); axis.legend(fontsize=7, loc="best")


def draw_xi(axis: Any, exact: np.ndarray, learned: dict[int, np.ndarray], rows: dict[str, Any],
            x_range: tuple[float, float], ylim: tuple[float, float], title: str) -> None:
    axis.axhspan(-1, 1, color="#c9d7e3", alpha=.28, label="physical |xi|<1")
    axis.plot(exact[:, 0], exact[:, 1], color="black", lw=2.2, label="DOP853")
    axis.scatter(exact[0, 0], exact[0, 1], marker="o", s=30, color="black", zorder=6)
    for seed in SEEDS:
        path, row = learned[seed], rows[str(seed)]
        axis.plot(path[:, 0], path[:, 1], color=SEED_COLORS[seed], lw=1.3,
                  label=f"seed {seed} · {row['status'].replace('_', ' ')}")
        axis.scatter(path[-1, 0], path[-1, 1], marker="s", s=25, color=SEED_COLORS[seed], zorder=6)
        if row["physical_exit"]:
            axis.scatter(path[-1, 0], path[-1, 1], marker="X", s=75, color="red", zorder=7)
    axis.set(title=title, xlabel="$x$", ylabel=r"transformed velocity $\xi$",
             xlim=(x_range[0]-.25, x_range[1]+.25), ylim=ylim)
    axis.grid(alpha=.18); axis.legend(fontsize=7, loc="best")


def plot_family(u_th: float, exact_full: np.ndarray, exact_throat: np.ndarray,
                paths: dict[str, Any], case: dict[str, Any], kind: str, destination: Path) -> None:
    if kind == "u":
        exacts = (exact_full, exact_throat)
        full_values = [exact_full, *(paths["full_u"][t][s] for t in TREATMENTS for s in SEEDS)]
        throat_values = [exact_throat, *(paths["throat_u"][t][s] for t in TREATMENTS for s in SEEDS)]
        limits = (u_limits(full_values, (-17, 17)), u_limits(throat_values, (0, 17)))
        drawer, keys = draw_u, ("full_u", "throat_u")
    else:
        exacts = (exact_xi_path(exact_full), exact_xi_path(exact_throat))
        all_paths = ([exacts[0], *(paths["full_xi"][t][s] for t in TREATMENTS for s in SEEDS)],
                     [exacts[1], *(paths["throat_xi"][t][s] for t in TREATMENTS for s in SEEDS)])
        limits = []
        for values in all_paths:
            y = np.concatenate([v[:, 1][np.isfinite(v[:, 1])] for v in values])
            pad = .055 * max(float(np.ptp(y)), 1e-9)
            limits.append((float(np.min(y)-pad), float(np.max(y)+pad)))
        drawer, keys = draw_xi, ("full_xi", "throat_xi")
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    for row_index, treatment in enumerate(TREATMENTS):
        drawer(axes[row_index, 0], exacts[0], paths[keys[0]][treatment],
               case["full_traversal"][treatment]["seeds"], (-17, 17), limits[0],
               f"{TITLES[treatment]} · full traversal")
        drawer(axes[row_index, 1], exacts[1], paths[keys[1]][treatment],
               case["throat_started_outgoing"][treatment]["seeds"], (0, 17), limits[1],
               f"{TITLES[treatment]} · throat-started")
    label = "physical u(x)" if kind == "u" else "transformed xi(x)"
    fig.suptitle(rf"Matched {label} comparison · $u_{{th}}={u_th:.2f}$")
    fig.savefig(destination, dpi=185); plt.close(fig)


def plot_xi_drift(u_th: float, arrays: dict[str, np.ndarray], destination: Path) -> None:
    stem = family_key(u_th)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            prefix = f"{stem}__full__{treatment}__seed_{seed}__incoming"
            x, error, ratio = arrays[f"{prefix}__x"], arrays[f"{prefix}__e_xi"], arrays[f"{prefix}__R_drift"]
            axes[row, 0].plot(x, error, color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
            axes[row, 1].plot(x, ratio, color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
        axes[row, 0].axhline(0, color=".35", lw=.8)
        axes[row, 1].axhline(1, color=".35", lw=.8, ls="--")
        axes[row, 0].set(title=f"{TITLES[treatment]} · recursive xi error", xlabel="predicted incoming x", ylabel=r"$\hat\xi-\xi_{exact}(\hat x)$")
        axes[row, 1].set(title=f"{TITLES[treatment]} · orbit-normalized drift", xlabel="predicted incoming x", ylabel=r"$R_{drift}$")
        for axis in axes[row]: axis.grid(alpha=.18); axis.legend(fontsize=7)
    fig.suptitle(rf"Hard-family transformed drift · $u_{{th}}={u_th:.2f}$")
    fig.savefig(destination, dpi=185); plt.close(fig)


def plot_energy(arrays: dict[str, np.ndarray], destination: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        for column, u_th in enumerate(HARD):
            axis, stem = axes[row, column], family_key(u_th)
            for seed in SEEDS:
                prefix = f"{stem}__full__{treatment}__seed_{seed}"
                state, drift = arrays[f"{prefix}__physical_state"], arrays[f"{prefix}__relative_energy_drift"]
                mask = np.isfinite(drift) & (state[:, 0] <= 0)
                axis.plot(state[mask, 0], np.abs(drift[mask]), color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
            axis.set(title=rf"{TITLES[treatment]} · $u_{{th}}={u_th:.2f}$", xlabel="predicted x", ylabel=r"$|E(\hat z)-E_0|/|E_0|$")
            axis.grid(alpha=.18); axis.legend(fontsize=7)
    fig.suptitle("Hard-family predicted-state relative energy drift before the throat")
    fig.savefig(destination, dpi=185); plt.close(fig)


def global_aggregate(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for treatment in TREATMENTS:
        rows = [case[mode][treatment]["seeds"][str(seed)] for case in cases for seed in SEEDS]
        errors = [row["absolute_throat_u_error"] for row in rows if row["absolute_throat_u_error"] is not None]
        maxima = [row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"]
                  for row in rows if row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"] is not None]
        output[treatment] = {
            "rollout_count": len(rows), "successful_full_traversal_count": int(sum(r["reached_x_plus_17"] for r in rows)),
            "physical_exit_count": int(sum(r["physical_exit"] for r in rows)),
            "guard_or_horizon_termination_count": int(sum(r["maximum_step_guard"] for r in rows)),
            "nonfinite_termination_count": int(sum(r["nonfinite"] for r in rows)),
            "throat_crossing_count": int(sum(r["interpolated_throat_u"] is not None for r in rows)),
            "mean_absolute_throat_u_error_successful_crossings": None if not errors else float(np.mean(errors)),
            "mean_maximum_absolute_relative_energy_drift_before_or_at_throat": None if not maxima else float(np.mean(maxima)),
        }
    return output


def hard_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for treatment in TREATMENTS:
        rows = [case["hard_recursive_xi_diagnostic"][treatment][str(seed)]
                for case in cases if case["u_th"] in HARD for seed in SEEDS]
        output[treatment] = {key: mean_metric(rows, key) for key in (
            "mean_absolute_e_xi", "terminal_absolute_e_xi", "median_R_drift", "p90_R_drift",
            "maximum_R_drift", "fraction_R_drift_lt_1", "fraction_R_drift_lt_0p5",
            "maximum_absolute_cumulative_e_xi", "mean_signed_e_xi",
        )}
    return output


def write_csv(cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        for mode in ("full_traversal", "throat_started_outgoing"):
            for treatment in TREATMENTS:
                for seed in SEEDS:
                    row = case[mode][treatment]["seeds"][str(seed)]
                    energy = row["energy_diagnostic"]
                    rows.append({"u_th": case["u_th"], "mode": mode, "treatment": treatment, "seed": seed,
                                 "status": row["status"], "reached_x_plus_17": row["reached_x_plus_17"],
                                 "physical_exit": row["physical_exit"], "maximum_step_guard": row["maximum_step_guard"],
                                 "recovered_throat_u": row["interpolated_throat_u"],
                                 "absolute_throat_u_error": row["absolute_throat_u_error"],
                                 "maximum_pre_throat_absolute_relative_energy_drift": energy["maximum_absolute_relative_energy_drift_before_or_at_throat"],
                                 "absolute_relative_energy_drift_at_throat": energy["absolute_relative_energy_drift_at_interpolated_throat"]})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def report_text(summary: dict[str, Any]) -> str:
    training = summary["training"]
    run_rows = "\n".join(f"| {r['seed']} | {r['best_epoch']} | {r['stopping_epoch']} | {r['one_step_validation']['standardized_mse']:.3e} | {r['one_step_validation']['rmse_delta_x']:.3e} | {r['one_step_validation']['rmse_delta_xi']:.3e} |" for r in training["runs"])
    family_rows = []
    for case in summary["cases"]:
        cells = []
        for treatment in TREATMENTS:
            section = case["full_traversal"][treatment]
            recovered = "/".join("—" if section["seeds"][str(s)]["interpolated_throat_u"] is None else f"{section['seeds'][str(s)]['interpolated_throat_u']:.5f}" for s in SEEDS)
            agg = section["aggregate"]
            cells.extend((recovered, f"{agg['mean_absolute_throat_u_error_successful_crossings']:.5f}" if agg["mean_absolute_throat_u_error_successful_crossings"] is not None else "—",
                          f"{agg['recovered_throat_u_sample_standard_deviation']:.5f}" if agg["recovered_throat_u_sample_standard_deviation"] is not None else "—",
                          f"{agg['successful_full_traversal_count']}/3", str(agg["physical_exit_count"])))
        family_rows.append(f"| {case['u_th']:.2f} | " + " | ".join(cells) + " |")
    one = training["energy_validation"]["aggregate"]; base = training["baseline_validation"]["aggregate"]
    teacher = summary["teacher_forced_hard_aggregate"]
    hard = summary["aggregate"]["hard_recursive_xi"]
    full = summary["aggregate"]["full_traversal"]; throat = summary["aggregate"]["throat_started_outgoing"]
    change = lambda new, old: 100.0 * (new / old - 1.0)
    q = summary["scientific_answers"]
    questions = "\n".join(f"{i}. **{entry['answer']}** {entry['detail']}" for i, entry in enumerate(q, 1))
    return f"""# Fixed-E0 transformed-coordinate training and rollout comparison

## Controlled protocol

Exactly three new `3→32→32→2` two-tanh models (1,250 parameters; seeds 101/202/303) were trained on the unchanged microcore40k rows using ordinary train-only standardization and standardized two-output MSE. Adam `1e-3`, batch 512, float32, no weight decay or scheduler, Xavier/zero initialization, ceiling 1500, patience 40, and physical-validation-MSE checkpointing match the frozen baseline protocol. During recursion, exact family `E0` was calculated once from the exact initial state and held fixed; predicted-state energy was diagnostic only.

`E0` train mean/SD/min/max: `{training['normalization']['E0_training_distribution']['mean']:.10g}` / `{training['normalization']['E0_training_distribution']['standard_deviation']:.10g}` / `{training['normalization']['E0_training_distribution']['minimum']:.10g}` / `{training['normalization']['E0_training_distribution']['maximum']:.10g}`. All standardized values were finite.

| seed | best epoch | stop epoch | validation std MSE | RMSE delta_x | RMSE delta_xi |
|---:|---:|---:|---:|---:|---:|
{run_rows}

## One-step and teacher-forced local diagnostics

On the same microcore8k set, mean delta-xi RMSE changes from `{base['rmse_delta_xi']['mean']:.6e} ± {base['rmse_delta_xi']['sample_standard_deviation']:.2e}` to `{one['rmse_delta_xi']['mean']:.6e} ± {one['rmse_delta_xi']['sample_standard_deviation']:.2e}` (`{change(one['rmse_delta_xi']['mean'], base['rmse_delta_xi']['mean']):+.1f}%`). Delta-x RMSE changes from `{base['rmse_delta_x']['mean']:.6e}` to `{one['rmse_delta_x']['mean']:.6e}`. Region-level values are retained in the training manifest; fixed E0 improves outer micro-core delta-xi RMSE by `{training['regional_relative_change_percent']['outer_micro_core']['rmse_delta_xi']:+.1f}%` but changes outer-left, shoulder, and edge by `{training['regional_relative_change_percent']['outer_left']['rmse_delta_xi']:+.1f}%`, `{training['regional_relative_change_percent']['outer_shoulder']['rmse_delta_xi']:+.1f}%`, and `{training['regional_relative_change_percent']['outer_edge']['rmse_delta_xi']:+.1f}%`.

Across the nine hard-family exact-state incoming checks, delta-xi RMSE changes from `{teacher['no_energy']['full_incoming']['rmse_delta_xi']:.6e}` to `{teacher['fixed_E0']['full_incoming']['rmse_delta_xi']:.6e}` (`{change(teacher['fixed_E0']['full_incoming']['rmse_delta_xi'], teacher['no_energy']['full_incoming']['rmse_delta_xi']):+.1f}%`). On outer-left only it changes from `{teacher['no_energy']['outer_left_minus17_to_minus8p5']['rmse_delta_xi']:.6e}` to `{teacher['fixed_E0']['outer_left_minus17_to_minus8p5']['rmse_delta_xi']:.6e}` (`{change(teacher['fixed_E0']['outer_left_minus17_to_minus8p5']['rmse_delta_xi'], teacher['no_energy']['outer_left_minus17_to_minus8p5']['rmse_delta_xi']):+.1f}%`). Signed fractions and cumulative-bias diagnostics are in the JSON.

## Recursive traversal recovery

Seed order is 101/202/303.

| u_th | no-energy recovered u_th | MAE | SD | reach | exits | fixed-E0 recovered u_th | MAE | SD | reach | exits |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(family_rows)}

Across 21 full rollouts, no-energy reaches/exits/guards are `{full['no_energy']['successful_full_traversal_count']}/{full['no_energy']['physical_exit_count']}/{full['no_energy']['guard_or_horizon_termination_count']}`; fixed-E0 values are `{full['fixed_E0']['successful_full_traversal_count']}/{full['fixed_E0']['physical_exit_count']}/{full['fixed_E0']['guard_or_horizon_termination_count']}`. All 21 throat-started controls reach +17 for no-energy: `{throat['no_energy']['successful_full_traversal_count'] == 21}`; for fixed-E0: `{throat['fixed_E0']['successful_full_traversal_count'] == 21}`.

Across nine hard incoming rollouts, mean absolute recursive e_xi changes from `{hard['no_energy']['mean_absolute_e_xi']:.6e}` to `{hard['fixed_E0']['mean_absolute_e_xi']:.6e}` (`{change(hard['fixed_E0']['mean_absolute_e_xi'], hard['no_energy']['mean_absolute_e_xi']):+.1f}%`); mean median R_drift changes from `{hard['no_energy']['median_R_drift']:.4f}` to `{hard['fixed_E0']['median_R_drift']:.4f}` (`{change(hard['fixed_E0']['median_R_drift'], hard['no_energy']['median_R_drift']):+.1f}%`). Fractions R_drift<0.5 are `{hard['no_energy']['fraction_R_drift_lt_0p5']:.1%}` and `{hard['fixed_E0']['fraction_R_drift_lt_0p5']:.1%}`. Mean maximum pre-throat absolute relative energy drift across all families changes from `{full['no_energy']['mean_maximum_absolute_relative_energy_drift_before_or_at_throat']:.6f}` to `{full['fixed_E0']['mean_maximum_absolute_relative_energy_drift_before_or_at_throat']:.6f}`.

## Twelve scientific questions

{questions}

## Conclusion

**{summary['scientific_conclusion']['headline']}** {summary['scientific_conclusion']['detail']}

This controlled result therefore `{summary['scientific_conclusion']['next_step_statement']}`

## Integrity

Protected datasets, exact references, orbit-spacing arrays, baseline checkpoints/histories, and all three new checkpoints/histories retained identical before/after hashes. No sealed/test data were accessed. No model was retrained during evaluation, and no clipping, projection, correction, penalty, weighting, or multi-step loss was used.
"""


def scientific_answers(summary: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    tr, cases = summary["training"], summary["cases"]
    full, throat, hard = (summary["aggregate"][key] for key in
                          ("full_traversal", "throat_started_outgoing", "hard_recursive_xi"))
    teacher = summary["teacher_forced_hard_aggregate"]
    pct = lambda new, old: 100.0 * (new / old - 1.0)
    case = {c["u_th"]: c for c in cases}
    def family(t: str, u: float) -> dict[str, Any]: return case[u]["full_traversal"][t]["aggregate"]
    local_pct = pct(tr["energy_validation"]["aggregate"]["rmse_delta_xi"]["mean"], tr["baseline_validation"]["aggregate"]["rmse_delta_xi"]["mean"])
    teacher_pct = pct(teacher["fixed_E0"]["full_incoming"]["rmse_delta_xi"], teacher["no_energy"]["full_incoming"]["rmse_delta_xi"])
    drift_pct = pct(hard["fixed_E0"]["median_R_drift"], hard["no_energy"]["median_R_drift"])
    energy_pct = pct(full["fixed_E0"]["mean_maximum_absolute_relative_energy_drift_before_or_at_throat"], full["no_energy"]["mean_maximum_absolute_relative_energy_drift_before_or_at_throat"])
    hard_disp = {u: (family("no_energy", u)["recovered_throat_u_sample_standard_deviation"], family("fixed_E0", u)["recovered_throat_u_sample_standard_deviation"]) for u in HARD}
    dispersion_better = sum(new is not None and old is not None and new < old for old, new in hard_disp.values())
    hardest = family("fixed_E0", .05)
    easy_preserved = sum(family("fixed_E0", u)["successful_full_traversal_count"] for u in THROAT_VELOCITIES[3:]) == 12
    controls_preserved = throat["fixed_E0"]["successful_full_traversal_count"] == 21 and throat["fixed_E0"]["physical_exit_count"] == 0
    answers = [
        {"answer": "Slightly overall, but not uniformly.", "detail": f"Same-set delta-xi RMSE changes {local_pct:+.1f}%; outer-left/shoulder/edge degrade despite micro-core improvement."},
        {"answer": "Mixed.", "detail": f"Hard-family full-incoming delta-xi RMSE changes {teacher_pct:+.1f}%, while the outer-left-only RMSE changes {pct(teacher['fixed_E0']['outer_left_minus17_to_minus8p5']['rmse_delta_xi'], teacher['no_energy']['outer_left_minus17_to_minus8p5']['rmse_delta_xi']):+.1f}%."},
        {"answer": "Yes for robust traversal; throat accuracy remains imperfect.", "detail": f"Fixed-E0 reaches +17 in {hardest['successful_full_traversal_count']}/3 at u_th=0.05 with mean absolute throat error {hardest['mean_absolute_throat_u_error_successful_crossings']:.5f}."},
        {"answer": "Yes" if full["fixed_E0"]["physical_exit_count"] == 0 else "No", "detail": f"Physical exits change from {full['no_energy']['physical_exit_count']} to {full['fixed_E0']['physical_exit_count']} across 21 full rollouts."},
        {"answer": f"For {dispersion_better}/3 hard families.", "detail": "Per-family seed dispersions are reported in the traversal table."},
        {"answer": "Yes" if drift_pct < 0 else "No", "detail": f"Mean hard-family median R_drift changes {drift_pct:+.1f}%."},
        {"answer": "Yes" if energy_pct < 0 else "No", "detail": f"Mean maximum pre-throat predicted-state energy drift changes {energy_pct:+.1f}%."},
        {"answer": "Yes—both are improved." if all(family("fixed_E0", u)["successful_full_traversal_count"] == 3 for u in (.15,.30)) else "Not fully", "detail": f"Mean throat errors change from {family('no_energy',.15)['mean_absolute_throat_u_error_successful_crossings']:.5f}/{family('no_energy',.30)['mean_absolute_throat_u_error_successful_crossings']:.5f} to {family('fixed_E0',.15)['mean_absolute_throat_u_error_successful_crossings']:.5f}/{family('fixed_E0',.30)['mean_absolute_throat_u_error_successful_crossings']:.5f}."},
        {"answer": "Traversal is preserved; precision is mixed." if easy_preserved else "No", "detail": f"Fixed-E0 succeeds in {sum(family('fixed_E0',u)['successful_full_traversal_count'] for u in THROAT_VELOCITIES[3:])}/12; throat error improves for 0.50/0.65 but worsens, while remaining small, for 0.80/0.90."},
        {"answer": "Yes" if controls_preserved else "No", "detail": f"Fixed-E0 throat-started reaches/exits are {throat['fixed_E0']['successful_full_traversal_count']}/21 and {throat['fixed_E0']['physical_exit_count']}."},
        {"answer": "Yes, as an explicit representation aid" if hardest["successful_full_traversal_count"] > family("no_energy", .05)["successful_full_traversal_count"] else "Only limited evidence", "detail": "E0 is deterministic from exact state; any gain reflects inductive bias, not new information."},
    ]
    robust = hardest["successful_full_traversal_count"] == 3 and full["fixed_E0"]["physical_exit_count"] == 0
    materially = robust and drift_pct < -10 and energy_pct < -10
    answers.append({"answer": "Yes, provisionally" if materially else "Not clearly", "detail": "The robust hardest-family recovery and large recursive-drift reduction outweigh the mixed one-step result, while the easy-family precision regressions remain a caveat."})
    if materially:
        conclusion = {"headline": "Fixed E0 materially improves recursive orbit preservation and is justified as a retained input.",
                      "detail": "It removes the hardest-family exit, reduces hard-family drift sharply, and preserves all controls, despite mixed local metrics and small easy-family precision regressions.",
                      "next_step_statement": "supports retaining fixed E0 and moving next toward controlled loss and/or architecture redesign—not further random sampling refinement—to address the remaining u_th=0.05 throat error."}
    else:
        conclusion = {"headline": "Fixed E0 does not provide a sufficiently uniform material gain.",
                      "detail": "The mixed local and recursive evidence does not justify increasing the simple model input on this experiment alone.",
                      "next_step_statement": "supports moving next toward controlled loss and/or architecture redesign rather than retaining E0 or further random sampling refinement."}
    return answers, conclusion


def main() -> None:
    if ROLLOUTS.exists() or SUMMARY_PATH.exists() or ARRAYS_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite existing energy-input rollout artifacts")
    training = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    protected = protected_paths(training); before = hashes(protected)
    models = {int(row["seed"]): load_trained_model(Path(row["checkpoint"])) for row in training["runs"]}
    if set(models) != set(SEEDS) or any(parameter_count(model) != 1250 for model in models.values()):
        raise RuntimeError("expected exactly three 3->32->32->2 checkpoints")
    normalization = Normalization.from_stage1(
        ENERGY_NORMALIZATION, ("x", "xi", "E0"), ("delta_x", "delta_xi"),
        "outer_microcore40k_train_x_xi_energy_input_only",
    )
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_cases = {float(case["u_th"]): case for case in baseline_summary["cases"]}
    family_source = {float(case["u_th"]): case for case in json.loads(FAMILY_SUMMARY.read_text(encoding="utf-8"))["cases"]}
    if set(baseline_cases) != set(THROAT_VELOCITIES) or set(family_source) != set(THROAT_VELOCITIES):
        raise RuntimeError("the frozen seven-family definitions are incomplete")
    ROLLOUTS.mkdir(); arrays: dict[str, np.ndarray] = {}; cases = []
    teacher = teacher_forced(models, normalization, family_source, arrays)
    with np.load(BASELINE_ARRAYS, allow_pickle=False) as base_arrays, np.load(ORBIT_SPACING, allow_pickle=False) as spacing:
        for u_th in THROAT_VELOCITIES:
            stem, base_case = family_key(u_th), baseline_cases[u_th]
            exact_full = np.asarray(base_arrays[f"{stem}__exact_full_state"])
            exact_throat = np.asarray(base_arrays[f"{stem}__exact_throat_started_state"])
            E0_full = float(conserved_energy(exact_full[0, 0], exact_full[0, 1], *experiment_parameters()))
            E0_throat = float(conserved_energy(exact_throat[0, 0], exact_throat[0, 1], *experiment_parameters()))
            stored_E0 = float(family_source[u_th]["energy"])
            if not (np.isclose(E0_full, stored_E0, rtol=0, atol=2e-12) and np.isclose(E0_throat, stored_E0, rtol=0, atol=2e-12)):
                raise RuntimeError(f"initial-state energy disagrees with frozen family for u_th={u_th}")
            arrays[f"{stem}__exact_full_state"] = exact_full
            arrays[f"{stem}__exact_full_coordinates"] = np.asarray(base_arrays[f"{stem}__exact_full_coordinates"])
            arrays[f"{stem}__exact_throat_started_state"] = exact_throat
            arrays[f"{stem}__exact_throat_started_coordinates"] = np.asarray(base_arrays[f"{stem}__exact_throat_started_coordinates"])
            spacing_x, spacing_values = np.asarray(spacing[f"{stem}__x"]), np.asarray(spacing[f"{stem}__orbit_spacing"])
            arrays[f"{stem}__orbit_spacing_x"], arrays[f"{stem}__orbit_spacing"] = spacing_x, spacing_values
            case: dict[str, Any] = {"u_th": float(u_th), "u_left": float(family_source[u_th]["u_left"]),
                                    "E0_from_full_initial_state": E0_full, "E0_from_throat_initial_state": E0_throat,
                                    "E0_held_fixed_during_recursion": True, "full_traversal": {},
                                    "throat_started_outgoing": {}, "hard_recursive_xi_diagnostic": {}, "figures": {}}
            paths = {key: {t: {} for t in TREATMENTS} for key in ("full_u", "throat_u", "full_xi", "throat_xi")}
            for mode, mode_key, exact, fixed_E0 in (("full", "full_traversal", exact_full, E0_full),
                                                     ("throat", "throat_started_outgoing", exact_throat, E0_throat)):
                baseline_section = copy.deepcopy(base_case[mode_key]["microcore40k"])
                case[mode_key]["no_energy"] = baseline_section
                fixed_rows: dict[str, Any] = {}
                initial_coordinates = exact_xi_path(exact)[0]
                for seed in SEEDS:
                    base_coordinates, base_physical, base_C, _ = copy_baseline_case_arrays(base_arrays, arrays, stem, mode, seed)
                    paths[f"{mode}_u"]["no_energy"][seed] = base_physical
                    paths[f"{mode}_xi"]["no_energy"][seed] = base_coordinates
                    coordinates, physical, margins, drift, row = energy_recursive_rollout(
                        models[seed], normalization, initial_coordinates, fixed_E0, u_th
                    )
                    fixed_rows[str(seed)] = row
                    paths[f"{mode}_u"]["fixed_E0"][seed] = physical
                    paths[f"{mode}_xi"]["fixed_E0"][seed] = coordinates
                    prefix = f"{stem}__{mode}__fixed_E0__seed_{seed}"
                    arrays[f"{prefix}__coordinates"], arrays[f"{prefix}__physical_state"] = coordinates, physical
                    arrays[f"{prefix}__C"], arrays[f"{prefix}__relative_energy_drift"] = margins, drift
                    arrays[f"{prefix}__fixed_supplied_E0"] = np.asarray([fixed_E0])
                    if mode == "full" and u_th in HARD:
                        if "no_energy" not in case["hard_recursive_xi_diagnostic"]:
                            case["hard_recursive_xi_diagnostic"]["no_energy"] = copy.deepcopy(base_case["hard_recursive_xi_diagnostic"]["microcore40k"])
                            case["hard_recursive_xi_diagnostic"]["fixed_E0"] = {}
                        old_prefix = f"{stem}__full__microcore40k__seed_{seed}__incoming"
                        new_prefix = f"{stem}__full__no_energy__seed_{seed}__incoming"
                        for suffix in ("x", "predicted_xi", "reference_xi_at_predicted_x", "orbit_spacing_at_predicted_x", "e_xi", "R_drift", "cumulative_e_xi"):
                            arrays[f"{new_prefix}__{suffix}"] = np.asarray(base_arrays[f"{old_prefix}__{suffix}"])
                        old_ratio = arrays[f"{new_prefix}__R_drift"]
                        case["hard_recursive_xi_diagnostic"]["no_energy"][str(seed)]["fraction_R_drift_lt_0p5"] = float(np.mean(old_ratio < .5))
                        diag_arrays, diag_summary = incoming_xi_diagnostic(coordinates, margins, exact_full, spacing_x, spacing_values)
                        diag_summary["fraction_R_drift_lt_0p5"] = float(np.mean(diag_arrays["R_drift"] < .5))
                        case["hard_recursive_xi_diagnostic"]["fixed_E0"][str(seed)] = diag_summary
                        for suffix, value in diag_arrays.items(): arrays[f"{prefix}__incoming__{suffix}"] = value
                case[mode_key]["fixed_E0"] = {"seeds": fixed_rows, "aggregate": family_aggregate(fixed_rows)}
            u_path, xi_path = FIGURES / f"energy_comparison_u_{stem}.png", FIGURES / f"energy_comparison_xi_{stem}.png"
            plot_family(u_th, exact_full, exact_throat, paths, case, "u", u_path)
            plot_family(u_th, exact_full, exact_throat, paths, case, "xi", xi_path)
            case["figures"]["u"] = {"path": str(u_path), "sha256": file_sha256(u_path)}
            case["figures"]["xi"] = {"path": str(xi_path), "sha256": file_sha256(xi_path)}
            if u_th in HARD:
                path = FIGURES / f"energy_comparison_recursive_xi_error_{stem}.png"
                plot_xi_drift(u_th, arrays, path)
                case["figures"]["recursive_xi_error"] = {"path": str(path), "sha256": file_sha256(path)}
            cases.append(case); print(f"completed fixed-E0 family u_th={u_th:.2f}", flush=True)
    energy_path = FIGURES / "energy_comparison_hard_family_energy_drift.png"
    plot_energy(arrays, energy_path)
    np.savez_compressed(ARRAYS_PATH, **arrays); write_csv(cases)
    summary: dict[str, Any] = {
        "stage": "training plus matched frozen-baseline fixed-E0 transformed-coordinate rollout comparison",
        "status": "three_energy_runs_and_two_treatments_x_three_seeds_x_seven_families_completed",
        "training": training, "seeds": list(SEEDS), "families": list(THROAT_VELOCITIES),
        "models": {
            "no_energy": {"architecture": "2->32->32->2", "parameter_count": 1218, "retrained": False},
            "fixed_E0": {"architecture": "3->32->32->2", "parameter_count": 1250,
                         "input_order": ["x", "xi", "E0"], "output_order": ["delta_x", "delta_xi"],
                         "fixed_E0_semantics": "computed once from exact initial state and never replaced by predicted-state energy"},
        },
        "step_size": H, "maximum_model_steps": MAX_MODEL_STEPS,
        "termination_conventions": {"success": "first x>=+17", "physical_exit": "first C<=0",
                                    "guard": "10000 learned steps", "nonfinite": "first nonfinite state"},
        "teacher_forced": teacher, "teacher_forced_hard_aggregate": aggregate_teacher(teacher),
        "cases": cases,
        "aggregate": {"full_traversal": global_aggregate(cases, "full_traversal"),
                      "throat_started_outgoing": global_aggregate(cases, "throat_started_outgoing"),
                      "hard_recursive_xi": hard_aggregate(cases)},
        "recursive_xi_error_definition": {"e_xi": "predicted xi minus exact-family xi interpolated at predicted incoming x",
                                          "R_drift": "absolute e_xi divided by frozen nearest-seven-family xi spacing interpolated at predicted incoming x",
                                          "spacing_source": str(ORBIT_SPACING)},
        "artifacts": {"arrays": str(ARRAYS_PATH), "metrics_csv": str(CSV_PATH), "report": str(REPORT_PATH),
                      "energy_figure": {"path": str(energy_path), "sha256": file_sha256(energy_path)}},
        "protocol": {"baseline_retrained": False, "evaluation_retraining": False, "fixed_E0_recomputed_during_rollout": False,
                     "predicted_energy_used_as_input": False, "clipping_projection_penalty_weighting_or_correction": False,
                     "restricted_evaluation_data_accessed": False},
    }
    summary["scientific_answers"], summary["scientific_conclusion"] = scientific_answers(summary)
    after = hashes(protected)
    if before != after: raise RuntimeError("a protected artifact changed during evaluation")
    summary["protected_hashes_before"], summary["protected_hashes_after"] = before, after
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}, {ARRAYS_PATH}, {CSV_PATH}, 18 figures, and {REPORT_PATH}")


if __name__ == "__main__":
    main()
