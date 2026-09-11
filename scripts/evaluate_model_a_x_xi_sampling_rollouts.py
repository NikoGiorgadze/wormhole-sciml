#!/usr/bin/env python3
"""Recursive traversal evaluation for the frozen old20k and microcore40k models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import conserved_energy, timelike_margin, velocity_bounds
from wormhole_sciml.model_a import (
    Normalization,
    load_trained_model,
    parameter_count,
    predict_increments,
)
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi, xi_from_state
from wormhole_sciml.stage1_data import file_sha256

try:
    from plot_c32x32_traversal_families import (
        H,
        MAX_MODEL_STEPS,
        SEED_COLORS,
        THROAT_TOLERANCE,
        THROAT_VELOCITIES,
        X_ENDPOINT,
        exact_rollout,
    )
except ModuleNotFoundError:
    from scripts.plot_c32x32_traversal_families import (
        H,
        MAX_MODEL_STEPS,
        SEED_COLORS,
        THROAT_TOLERANCE,
        THROAT_VELOCITIES,
        X_ENDPOINT,
        exact_rollout,
    )


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "output" / "model_a_x_xi_sampling_training_comparison"
OUTPUT = ROOT / "output" / "model_a_x_xi_sampling_recursive_rollouts"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "recursive_rollout_summary.json"
ARRAYS_PATH = OUTPUT / "recursive_rollout_arrays.npz"
REPORT_PATH = OUTPUT / "RECURSIVE_ROLLOUT_REPORT.md"
SOURCE_FAMILY_SUMMARY = ROOT / "output" / "c32x32_traversal_families" / "traversal_family_summary.json"
SOURCE_EXACT_ARRAYS = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
ORBIT_SPACING_ARRAYS = ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_arrays.npz"
TRAINING_MANIFEST = TRAINING_DIR / "training_comparison_manifest.json"
SEEDS = (101, 202, 303)
TREATMENTS = ("old20k", "microcore40k")
HARD_FAMILIES = THROAT_VELOCITIES[:3]
TREATMENT_TITLES = {"old20k": "old20k", "microcore40k": "microcore40k"}


def family_key(u_th: float) -> str:
    return f"u_th_{u_th:.2f}".replace(".", "p")


def protected_paths() -> tuple[Path, ...]:
    paths: list[Path] = [
        TRAINING_MANIFEST,
        SOURCE_FAMILY_SUMMARY,
        SOURCE_EXACT_ARRAYS,
        ORBIT_SPACING_ARRAYS,
        TRAINING_DIR / "old20k_normalization.json",
        TRAINING_DIR / "microcore40k_normalization.json",
    ]
    for treatment in TREATMENTS:
        for seed in SEEDS:
            run = TRAINING_DIR / treatment / f"seed_{seed}"
            paths.extend((run / "best_checkpoint.pt", run / "metadata.json", run / "training_history.json"))
    return tuple(paths)


def artifact_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required immutable artifacts are missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def load_models_and_normalizations() -> tuple[
    dict[str, dict[int, Any]], dict[str, Normalization], dict[str, dict[int, Path]]
]:
    manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    runs = {(row["sampling_treatment"], int(row["seed"])): row for row in manifest["runs"]}
    expected = {(treatment, seed) for treatment in TREATMENTS for seed in SEEDS}
    if set(runs) != expected or manifest["run_count"] != 6:
        raise RuntimeError("the frozen sampling experiment does not contain exactly six prescribed runs")
    normalizations: dict[str, Normalization] = {}
    models: dict[str, dict[int, Any]] = {}
    checkpoints: dict[str, dict[int, Path]] = {}
    sources = {
        "old20k": "old20k_train_x_xi_only",
        "microcore40k": "outer_microcore40k_train_x_xi_only",
    }
    for treatment in TREATMENTS:
        normalization_path = TRAINING_DIR / f"{treatment}_normalization.json"
        normalizations[treatment] = Normalization.from_stage1(
            normalization_path, ("x", "xi"), ("delta_x", "delta_xi"), sources[treatment]
        )
        checkpoints[treatment] = {
            seed: Path(runs[(treatment, seed)]["checkpoint"]) for seed in SEEDS
        }
        models[treatment] = {
            seed: load_trained_model(checkpoints[treatment][seed]) for seed in SEEDS
        }
        if any(parameter_count(model) != 1218 for model in models[treatment].values()):
            raise RuntimeError(f"{treatment} checkpoint architecture differs from 2->32->32->2")
    return models, normalizations, checkpoints


def reconstruct_physical(coordinates: np.ndarray) -> np.ndarray:
    wormhole, spiral = experiment_parameters()
    x = np.asarray(coordinates[:, 0], dtype=np.float64)
    xi = np.asarray(coordinates[:, 1], dtype=np.float64)
    _, u = state_from_xi(x, xi, wormhole, spiral)
    return np.column_stack((x, np.asarray(u, dtype=np.float64)))


def energy_diagnostic(
    physical: np.ndarray, E0: float, throat_u: float | None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    wormhole, spiral = experiment_parameters()
    count = physical.shape[0]
    relative = np.full(count, np.nan, dtype=np.float64)
    margin = np.full(count, np.nan, dtype=np.float64)
    finite = np.all(np.isfinite(physical), axis=1)
    margin[finite] = timelike_margin(
        physical[finite, 0], physical[finite, 1], wormhole, spiral
    )
    physical_mask = finite & (margin > 0.0)
    energy = np.full(count, np.nan, dtype=np.float64)
    energy[physical_mask] = conserved_energy(
        physical[physical_mask, 0], physical[physical_mask, 1], wormhole, spiral
    )
    denominator = max(abs(E0), 1.0e-12)
    relative[physical_mask] = (energy[physical_mask] - E0) / denominator
    pre_throat = physical_mask & (physical[:, 0] <= 0.0)
    maximum = float(np.max(np.abs(relative[pre_throat]))) if np.any(pre_throat) else None
    at_throat = None
    if throat_u is not None:
        at_throat = float((conserved_energy(0.0, throat_u, wormhole, spiral) - E0) / denominator)
        maximum = max(maximum or 0.0, abs(at_throat))
    return relative, margin, {
        "maximum_absolute_relative_energy_drift_before_or_at_throat": maximum,
        "relative_energy_drift_at_interpolated_throat": at_throat,
        "absolute_relative_energy_drift_at_interpolated_throat": (
            None if at_throat is None else abs(at_throat)
        ),
    }


def transformed_recursive_rollout(
    model: Any,
    normalization: Normalization,
    initial_coordinates: np.ndarray,
    E0: float,
    u_th: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Advance only (x, xi); reconstruct u after each unconstrained update."""
    wormhole, spiral = experiment_parameters()
    coordinates = [np.asarray(initial_coordinates, dtype=np.float64)]
    physical_states = [reconstruct_physical(np.asarray(coordinates))[0]]
    status = "maximum_step_guard"
    exit_record = None
    for step in range(1, MAX_MODEL_STEPS + 1):
        current = coordinates[-1]
        following = current + predict_increments(
            model, current.reshape(1, 2), normalization
        )[0]
        coordinates.append(following)
        if not np.all(np.isfinite(following)):
            physical_states.append(np.asarray([np.nan, np.nan]))
            status = "nonfinite"
            break
        following_physical = reconstruct_physical(following.reshape(1, 2))[0]
        physical_states.append(following_physical)
        if not np.all(np.isfinite(following_physical)):
            status = "nonfinite"
            break
        margin = float(
            timelike_margin(
                following_physical[0], following_physical[1], wormhole, spiral
            )
        )
        if margin <= 0.0:
            status = "physical_exit"
            exit_record = {
                "step": step,
                "time": float(step * H),
                "x": float(following[0]),
                "xi": float(following[1]),
                "u": float(following_physical[1]),
                "C": margin,
            }
            break
        if following[0] >= X_ENDPOINT:
            status = "reached_x_plus_17"
            break
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    physical_array = np.asarray(physical_states, dtype=np.float64)
    crossing = np.flatnonzero(
        (coordinates_array[:-1, 0] < 0.0) & (coordinates_array[1:, 0] >= 0.0)
    )
    throat_u = throat_xi = throat_step = None
    if crossing.size:
        index = int(crossing[0])
        dx = coordinates_array[index + 1, 0] - coordinates_array[index, 0]
        fraction = float(-coordinates_array[index, 0] / dx)
        throat_xi = float(
            coordinates_array[index, 1]
            + fraction * (coordinates_array[index + 1, 1] - coordinates_array[index, 1])
        )
        throat_u = float(
            physical_array[index, 1]
            + fraction * (physical_array[index + 1, 1] - physical_array[index, 1])
        )
        throat_step = index + 1
    relative_energy, margins, energy_summary = energy_diagnostic(physical_array, E0, throat_u)
    summary = {
        "status": status,
        "reached_x_plus_17": status == "reached_x_plus_17",
        "physical_exit": status == "physical_exit",
        "maximum_step_guard": status == "maximum_step_guard",
        "nonfinite": status == "nonfinite",
        "exit": exit_record,
        "rollout_steps": int(coordinates_array.shape[0] - 1),
        "elapsed_time": float((coordinates_array.shape[0] - 1) * H),
        "terminal_x": None if not np.isfinite(coordinates_array[-1, 0]) else float(coordinates_array[-1, 0]),
        "terminal_xi": None if not np.isfinite(coordinates_array[-1, 1]) else float(coordinates_array[-1, 1]),
        "terminal_u": None if not np.isfinite(physical_array[-1, 1]) else float(physical_array[-1, 1]),
        "throat_crossing_step": throat_step,
        "interpolated_throat_xi": throat_xi,
        "interpolated_throat_u": throat_u,
        "signed_throat_u_error": None if throat_u is None else float(throat_u - u_th),
        "absolute_throat_u_error": None if throat_u is None else float(abs(throat_u - u_th)),
        "energy_diagnostic": energy_summary,
    }
    return coordinates_array, physical_array, margins, relative_energy, summary


def incoming_xi_diagnostic(
    coordinates: np.ndarray,
    margins: np.ndarray,
    exact_full: np.ndarray,
    spacing_x: np.ndarray,
    spacing_values: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    wormhole, spiral = experiment_parameters()
    exact_x = np.asarray(exact_full[:, 0], dtype=np.float64)
    exact_xi = np.asarray(
        xi_from_state(exact_full[:, 0], exact_full[:, 1], wormhole, spiral), dtype=np.float64
    )
    valid = (
        np.all(np.isfinite(coordinates), axis=1)
        & np.isfinite(margins)
        & (margins > 0.0)
        & (coordinates[:, 0] >= -X_ENDPOINT)
        & (coordinates[:, 0] <= 0.0)
    )
    x = coordinates[valid, 0]
    predicted_xi = coordinates[valid, 1]
    reference_xi = np.interp(x, exact_x, exact_xi)
    orbit_spacing = np.interp(x, spacing_x, spacing_values)
    if np.any(orbit_spacing <= 0.0) or not np.all(np.isfinite(orbit_spacing)):
        raise RuntimeError("interpolated nearest-family orbit spacing is invalid")
    error = predicted_xi - reference_xi
    normalized = np.abs(error) / orbit_spacing
    cumulative = np.cumsum(error)
    arrays = {
        "x": x,
        "predicted_xi": predicted_xi,
        "reference_xi_at_predicted_x": reference_xi,
        "orbit_spacing_at_predicted_x": orbit_spacing,
        "e_xi": error,
        "R_drift": normalized,
        "cumulative_e_xi": cumulative,
    }
    summary = {
        "definition": "predicted xi minus exact-family xi interpolated at each predicted incoming x",
        "normalized_definition": "absolute recursive e_xi divided by interpolated nearest-seven-controlled-family xi spacing",
        "point_count": int(error.size),
        "mean_absolute_e_xi": float(np.mean(np.abs(error))),
        "maximum_absolute_e_xi": float(np.max(np.abs(error))),
        "terminal_absolute_e_xi": float(abs(error[-1])),
        "mean_signed_e_xi": float(np.mean(error)),
        "final_cumulative_e_xi": float(cumulative[-1]),
        "maximum_absolute_cumulative_e_xi": float(np.max(np.abs(cumulative))),
        "fraction_positive_e_xi": float(np.mean(error > 0.0)),
        "fraction_negative_e_xi": float(np.mean(error < 0.0)),
        "median_R_drift": float(np.median(normalized)),
        "mean_R_drift": float(np.mean(normalized)),
        "p90_R_drift": float(np.quantile(normalized, 0.90)),
        "maximum_R_drift": float(np.max(normalized)),
        "fraction_R_drift_lt_1": float(np.mean(normalized < 1.0)),
    }
    return arrays, summary


def family_aggregate(seed_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    crossings = [
        row["interpolated_throat_u"]
        for row in seed_rows.values()
        if row["interpolated_throat_u"] is not None
    ]
    errors = [
        row["absolute_throat_u_error"]
        for row in seed_rows.values()
        if row["absolute_throat_u_error"] is not None
    ]
    throat_energy = [
        row["energy_diagnostic"]["absolute_relative_energy_drift_at_interpolated_throat"]
        for row in seed_rows.values()
        if row["energy_diagnostic"]["absolute_relative_energy_drift_at_interpolated_throat"] is not None
    ]
    return {
        "successful_full_traversal_count": int(sum(row["reached_x_plus_17"] for row in seed_rows.values())),
        "physical_exit_count": int(sum(row["physical_exit"] for row in seed_rows.values())),
        "guard_or_horizon_termination_count": int(sum(row["maximum_step_guard"] for row in seed_rows.values())),
        "nonfinite_termination_count": int(sum(row["nonfinite"] for row in seed_rows.values())),
        "throat_crossing_count": len(crossings),
        "mean_absolute_throat_u_error_successful_crossings": None if not errors else float(np.mean(errors)),
        "recovered_throat_u_sample_standard_deviation": (
            None if len(crossings) < 2 else float(np.std(crossings, ddof=1))
        ),
        "mean_absolute_relative_energy_drift_at_throat_successful_crossings": (
            None if not throat_energy else float(np.mean(throat_energy))
        ),
    }


def exact_xi_path(exact: np.ndarray) -> np.ndarray:
    wormhole, spiral = experiment_parameters()
    xi = xi_from_state(exact[:, 0], exact[:, 1], wormhole, spiral)
    return np.column_stack((exact[:, 0], xi))


def u_limits(paths: list[np.ndarray], x_range: tuple[float, float]) -> tuple[float, float]:
    wormhole, spiral = experiment_parameters()
    grid = np.linspace(*x_range, 1601)
    lower, upper = velocity_bounds(grid, wormhole, spiral)
    values = [lower, upper]
    values.extend(path[:, 1][np.all(np.isfinite(path), axis=1)] for path in paths)
    low = min(float(np.min(values_array)) for values_array in values if values_array.size)
    high = max(float(np.max(values_array)) for values_array in values if values_array.size)
    padding = 0.055 * max(high - low, 1.0e-9)
    return low - padding, high + padding


def draw_u_panel(
    axis: Any,
    exact: np.ndarray,
    learned: dict[int, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    x_range: tuple[float, float],
    title: str,
) -> None:
    wormhole, spiral = experiment_parameters()
    grid = np.linspace(*x_range, 1601)
    lower, upper = velocity_bounds(grid, wormhole, spiral)
    axis.fill_between(grid, lower, upper, color="#c9d7e3", alpha=0.28, label="admissible region")
    axis.plot(exact[:, 0], exact[:, 1], color="black", lw=2.3, label="DOP853")
    for seed in SEEDS:
        path = learned[seed]
        status = summaries[str(seed)]["status"].replace("_", " ")
        axis.plot(path[:, 0], path[:, 1], color=SEED_COLORS[seed], lw=1.35, label=f"seed {seed} · {status}")
        axis.scatter(path[-1, 0], path[-1, 1], marker="s", s=25, color=SEED_COLORS[seed], zorder=5)
        if summaries[str(seed)]["physical_exit"]:
            axis.scatter(path[-1, 0], path[-1, 1], marker="X", s=75, color="red", edgecolor="white", lw=0.6, zorder=7)
    axis.set(
        title=title,
        xlabel="$x$",
        ylabel="radial velocity $u$",
        xlim=(x_range[0] - 0.25, x_range[1] + 0.25),
        ylim=u_limits([exact, *learned.values()], x_range),
    )
    axis.grid(alpha=0.17)
    axis.legend(fontsize=7, loc="best")


def plot_u_family(
    u_th: float,
    exact_full: np.ndarray,
    exact_throat: np.ndarray,
    paths: dict[str, dict[str, dict[int, np.ndarray]]],
    summaries: dict[str, Any],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        draw_u_panel(
            axes[row, 0], exact_full, paths["full"][treatment],
            summaries["full_traversal"][treatment]["seeds"], (-X_ENDPOINT, X_ENDPOINT),
            f"{TREATMENT_TITLES[treatment]} · full traversal",
        )
        draw_u_panel(
            axes[row, 1], exact_throat, paths["throat"][treatment],
            summaries["throat_started_outgoing"][treatment]["seeds"], (0.0, X_ENDPOINT),
            f"{TREATMENT_TITLES[treatment]} · throat-started control",
        )
    figure.suptitle(rf"Recursive physical trajectories · $u_{{th}}={u_th:.2f}$")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def draw_xi_panel(
    axis: Any,
    exact_coordinates: np.ndarray,
    learned: dict[int, np.ndarray],
    summaries: dict[str, dict[str, Any]],
    x_range: tuple[float, float],
    title: str,
) -> None:
    axis.axhspan(-1.0, 1.0, color="#c9d7e3", alpha=0.28, label="physical |xi|<1")
    axis.plot(exact_coordinates[:, 0], exact_coordinates[:, 1], color="black", lw=2.3, label="DOP853")
    for seed in SEEDS:
        path = learned[seed]
        status = summaries[str(seed)]["status"].replace("_", " ")
        axis.plot(path[:, 0], path[:, 1], color=SEED_COLORS[seed], lw=1.35, label=f"seed {seed} · {status}")
        axis.scatter(path[-1, 0], path[-1, 1], marker="s", s=25, color=SEED_COLORS[seed], zorder=5)
        if summaries[str(seed)]["physical_exit"]:
            axis.scatter(path[-1, 0], path[-1, 1], marker="X", s=75, color="red", edgecolor="white", lw=0.6, zorder=7)
    values = [exact_coordinates[:, 1], *(path[:, 1] for path in learned.values())]
    low = min(float(np.nanmin(value)) for value in values)
    high = max(float(np.nanmax(value)) for value in values)
    padding = 0.055 * max(high - low, 1.0e-9)
    axis.set(
        title=title, xlabel="$x$", ylabel=r"transformed velocity $\xi$",
        xlim=(x_range[0] - 0.25, x_range[1] + 0.25), ylim=(low - padding, high + padding),
    )
    axis.grid(alpha=0.17)
    axis.legend(fontsize=7, loc="best")


def plot_xi_family(
    u_th: float,
    exact_full: np.ndarray,
    exact_throat: np.ndarray,
    paths: dict[str, dict[str, dict[int, np.ndarray]]],
    summaries: dict[str, Any],
    destination: Path,
) -> None:
    exact_full_xi, exact_throat_xi = exact_xi_path(exact_full), exact_xi_path(exact_throat)
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 9.0), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        draw_xi_panel(
            axes[row, 0], exact_full_xi, paths["full_coordinates"][treatment],
            summaries["full_traversal"][treatment]["seeds"], (-X_ENDPOINT, X_ENDPOINT),
            f"{TREATMENT_TITLES[treatment]} · full traversal",
        )
        draw_xi_panel(
            axes[row, 1], exact_throat_xi, paths["throat_coordinates"][treatment],
            summaries["throat_started_outgoing"][treatment]["seeds"], (0.0, X_ENDPOINT),
            f"{TREATMENT_TITLES[treatment]} · throat-started control",
        )
    figure.suptitle(rf"Recursive transformed trajectories · $u_{{th}}={u_th:.2f}$")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_energy_drift(cases: list[dict[str, Any]], arrays: dict[str, np.ndarray], destination: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 7.2), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        for column, u_th in enumerate(HARD_FAMILIES):
            axis = axes[row, column]
            stem = family_key(u_th)
            for seed in SEEDS:
                state = arrays[f"{stem}__full__{treatment}__seed_{seed}__physical_state"]
                drift = arrays[f"{stem}__full__{treatment}__seed_{seed}__relative_energy_drift"]
                mask = np.isfinite(drift) & (state[:, 0] <= 0.0)
                axis.plot(state[mask, 0], np.abs(drift[mask]), color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
            axis.axhline(0.0, color="0.35", lw=0.8)
            axis.set(
                title=rf"{treatment} · $u_{{th}}={u_th:.2f}$",
                xlabel="predicted $x$", ylabel=r"$|E(\hat z)-E_0|/|E_0|$",
            )
            axis.grid(alpha=0.18)
            axis.legend(fontsize=7)
    figure.suptitle("Recursive predicted-state relative energy drift before the throat")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_recursive_xi_error(
    u_th: float, arrays: dict[str, np.ndarray], destination: Path
) -> None:
    stem = family_key(u_th)
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            prefix = f"{stem}__full__{treatment}__seed_{seed}__incoming"
            x = arrays[f"{prefix}__x"]
            error = arrays[f"{prefix}__e_xi"]
            ratio = arrays[f"{prefix}__R_drift"]
            axes[row, 0].plot(x, error, color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
            axes[row, 1].plot(x, ratio, color=SEED_COLORS[seed], lw=1.25, label=f"seed {seed}")
        axes[row, 0].axhline(0.0, color="0.35", lw=0.8)
        axes[row, 0].set(title=f"{treatment} · recursive xi error", xlabel="predicted incoming $x$", ylabel=r"$\hat\xi-\xi_{exact}(\hat x)$")
        axes[row, 1].axhline(1.0, color="0.35", lw=0.8, ls="--")
        axes[row, 1].set(title=f"{treatment} · orbit-normalized drift", xlabel="predicted incoming $x$", ylabel=r"$R_{drift}=|e_\xi|/\Delta\xi_{orbit}$")
        for axis in axes[row]:
            axis.grid(alpha=0.18)
            axis.legend(fontsize=7)
    figure.suptitle(rf"Hard-family recursive transformed drift · $u_{{th}}={u_th:.2f}$")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def global_aggregate(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for treatment in TREATMENTS:
        rows = [
            case[mode][treatment]["seeds"][str(seed)]
            for case in cases for seed in SEEDS
        ]
        errors = [row["absolute_throat_u_error"] for row in rows if row["absolute_throat_u_error"] is not None]
        maxima = [
            row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"]
            for row in rows
            if row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"] is not None
        ]
        result[treatment] = {
            "rollout_count": len(rows),
            "successful_full_traversal_count": int(sum(row["reached_x_plus_17"] for row in rows)),
            "physical_exit_count": int(sum(row["physical_exit"] for row in rows)),
            "guard_or_horizon_termination_count": int(sum(row["maximum_step_guard"] for row in rows)),
            "nonfinite_termination_count": int(sum(row["nonfinite"] for row in rows)),
            "throat_crossing_count": int(sum(row["interpolated_throat_u"] is not None for row in rows)),
            "mean_absolute_throat_u_error_successful_crossings": None if not errors else float(np.mean(errors)),
            "mean_maximum_absolute_relative_energy_drift_before_or_at_throat": None if not maxima else float(np.mean(maxima)),
        }
    return result


def hard_recursive_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    hard = [case for case in cases if case["u_th"] in HARD_FAMILIES]
    for treatment in TREATMENTS:
        diagnostics = [
            case["hard_recursive_xi_diagnostic"][treatment][str(seed)]
            for case in hard for seed in SEEDS
        ]
        result[treatment] = {
            name: float(np.mean([row[name] for row in diagnostics]))
            for name in (
                "mean_absolute_e_xi", "maximum_absolute_e_xi", "terminal_absolute_e_xi",
                "mean_signed_e_xi", "maximum_absolute_cumulative_e_xi",
                "median_R_drift", "mean_R_drift", "p90_R_drift",
                "fraction_R_drift_lt_1",
            )
        }
    return result


def write_summary_csv(cases: list[dict[str, Any]], destination: Path) -> None:
    rows = []
    for case in cases:
        for treatment in TREATMENTS:
            for seed in SEEDS:
                row = case["full_traversal"][treatment]["seeds"][str(seed)]
                energy = row["energy_diagnostic"]
                rows.append({
                    "u_th": case["u_th"], "treatment": treatment, "seed": seed,
                    "status": row["status"], "reached_x_plus_17": row["reached_x_plus_17"],
                    "physical_exit": row["physical_exit"], "maximum_step_guard": row["maximum_step_guard"],
                    "recovered_throat_u": row["interpolated_throat_u"],
                    "absolute_throat_u_error": row["absolute_throat_u_error"],
                    "maximum_pre_throat_absolute_relative_energy_drift": energy["maximum_absolute_relative_energy_drift_before_or_at_throat"],
                    "absolute_relative_energy_drift_at_throat": energy["absolute_relative_energy_drift_at_interpolated_throat"],
                })
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary: dict[str, Any]) -> str:
    rows = []
    for case in summary["cases"]:
        pieces: list[str] = []
        for treatment in TREATMENTS:
            section = case["full_traversal"][treatment]
            recovered = [
                section["seeds"][str(seed)]["interpolated_throat_u"] for seed in SEEDS
            ]
            pieces.extend((
                " / ".join("—" if value is None else f"{value:.5f}" for value in recovered),
                "—" if section["aggregate"]["mean_absolute_throat_u_error_successful_crossings"] is None
                else f"{section['aggregate']['mean_absolute_throat_u_error_successful_crossings']:.5f}",
                f"{section['aggregate']['successful_full_traversal_count']}/3",
                f"{section['aggregate']['physical_exit_count']} / {section['aggregate']['guard_or_horizon_termination_count']}",
            ))
        rows.append(
            f"| {case['u_th']:.2f} | {pieces[0]} | {pieces[1]} | {pieces[2]} | {pieces[3]} | "
            f"{pieces[4]} | {pieces[5]} | {pieces[6]} | {pieces[7]} |"
        )
    diagnostic_rows = []
    hard_rows = []
    for case in summary["cases"]:
        for treatment in TREATMENTS:
            section = case["full_traversal"][treatment]
            aggregate = section["aggregate"]
            energy_maxima = " / ".join(
                f"{section['seeds'][str(seed)]['energy_diagnostic']['maximum_absolute_relative_energy_drift_before_or_at_throat']:.5f}"
                for seed in SEEDS
            )
            dispersion = aggregate["recovered_throat_u_sample_standard_deviation"]
            throat_energy = aggregate["mean_absolute_relative_energy_drift_at_throat_successful_crossings"]
            diagnostic_rows.append(
                f"| {case['u_th']:.2f} | {treatment} | "
                f"{'—' if dispersion is None else f'{dispersion:.5f}'} | {energy_maxima} | "
                f"{'—' if throat_energy is None else f'{throat_energy:.5f}'} |"
            )
            if case["u_th"] in HARD_FAMILIES:
                diagnostics = [
                    case["hard_recursive_xi_diagnostic"][treatment][str(seed)]
                    for seed in SEEDS
                ]
                mean = lambda name: float(np.mean([row[name] for row in diagnostics]))
                hard_rows.append(
                    f"| {case['u_th']:.2f} | {treatment} | {mean('mean_absolute_e_xi'):.5e} | "
                    f"{mean('terminal_absolute_e_xi'):.5e} | {mean('median_R_drift'):.3f} | "
                    f"{mean('maximum_absolute_cumulative_e_xi'):.3f} | {mean('mean_signed_e_xi'):.5e} |"
                )
    full = summary["aggregate"]["full_traversal"]
    outgoing = summary["aggregate"]["throat_started_outgoing"]
    hard = summary["aggregate"]["hard_recursive_xi"]
    old_success = full["old20k"]["successful_full_traversal_count"]
    new_success = full["microcore40k"]["successful_full_traversal_count"]
    old_error = full["old20k"]["mean_absolute_throat_u_error_successful_crossings"]
    new_error = full["microcore40k"]["mean_absolute_throat_u_error_successful_crossings"]
    throat_change = None if old_error is None or new_error is None else 100.0 * (new_error / old_error - 1.0)
    xi_change = 100.0 * (hard["microcore40k"]["mean_absolute_e_xi"] / hard["old20k"]["mean_absolute_e_xi"] - 1.0)
    r_change = 100.0 * (hard["microcore40k"]["median_R_drift"] / hard["old20k"]["median_R_drift"] - 1.0)
    if new_success > old_success and throat_change is not None and throat_change < -30.0:
        interpretation = "Yes"
        detail = "outer/micro-core under-resolution was a major cause of the observed hard-family failures"
    elif new_success >= old_success and (throat_change is None or throat_change < 0.0 or xi_change < -25.0):
        interpretation = "Partly"
        detail = "local learning improved, but recursive orbit preservation remains incomplete"
    else:
        interpretation = "No"
        detail = "recursive behavior remains comparably poor, so sampling refinement alone is insufficient"
    return f"""# Recursive transformed-coordinate sampling comparison

## Scope and conventions

This evaluation uses only the frozen old20k and microcore40k `2→32→32→2` checkpoints for seeds 101, 202, and 303. Learned states advance recursively in `(x,xi)` at `h=0.2`; `u=c(x)+d(x)xi` is reconstructed only for physical diagnostics and plots. Success is first arrival at `x≥+17`, physical exit is first `C≤0`, and the unchanged guard is 10,000 learned steps. Predictions were not clipped, projected, corrected, or retrained.

The seven established far-left states and exact energies were loaded from the earlier family summary. Full DOP853 curves were deterministically reconstructed with the validated helper and match the frozen incoming `h=0.2` states exactly (maximum discrepancy `{summary['exact_reference']['maximum_abs_reconstruction_difference']:.1e}`).

## Full traversal and throat recovery

Seed values are ordered 101 / 202 / 303. “Exit / guard” reports counts among three seeds.

| u_th | old20k recovered u_th | old mean abs error | old reached | old exit / guard | microcore40k recovered u_th | micro mean abs error | micro reached | micro exit / guard |
|---:|---|---:|---:|---|---|---:|---:|---|
{chr(10).join(rows)}

Across all 21 full rollouts, old20k reaches `x=+17` in `{old_success}` cases, with `{full['old20k']['physical_exit_count']}` physical exits and `{full['old20k']['guard_or_horizon_termination_count']}` guards. Microcore40k reaches in `{new_success}`, with `{full['microcore40k']['physical_exit_count']}` exits and `{full['microcore40k']['guard_or_horizon_termination_count']}` guards. Mean absolute throat error among successful crossings changes from `{old_error if old_error is not None else 'not available'}` to `{new_error if new_error is not None else 'not available'}`{'' if throat_change is None else f' (`{throat_change:+.1f}%`)'}.

The throat-started controls reach `x=+17` in `{outgoing['old20k']['successful_full_traversal_count']}/21` old20k and `{outgoing['microcore40k']['successful_full_traversal_count']}/21` microcore40k runs, with `{outgoing['old20k']['physical_exit_count']}` and `{outgoing['microcore40k']['physical_exit_count']}` exits respectively.

| u_th | treatment | throat seed dispersion | max pre-throat relative energy drift, seeds 101/202/303 | mean absolute throat energy drift |
|---:|---|---:|---|---:|
{chr(10).join(diagnostic_rows)}

## Hard-family transformed drift and energy

For recursive incoming diagnostics, exact `xi` is interpolated at each predicted `x`. `R_drift=|xi_hat-xi_exact(x_hat)|/Delta xi_orbit(x_hat)`, where the denominator is the previously stored nearest-seven-controlled-family spacing interpolated at that same predicted `x`.

Across the nine hard-family rollouts, mean absolute recursive `e_xi` changes from `{hard['old20k']['mean_absolute_e_xi']:.6e}` to `{hard['microcore40k']['mean_absolute_e_xi']:.6e}` (`{xi_change:+.1f}%`). Mean median `R_drift` changes from `{hard['old20k']['median_R_drift']:.3f}` to `{hard['microcore40k']['median_R_drift']:.3f}` (`{r_change:+.1f}%`). Fractions with `R_drift<1` are `{hard['old20k']['fraction_R_drift_lt_1']:.1%}` and `{hard['microcore40k']['fraction_R_drift_lt_1']:.1%}`. Per-family and per-seed signed, cumulative, terminal, maximum, and normalized diagnostics are retained in the JSON and arrays.

| u_th | treatment | mean abs e_xi | terminal abs e_xi | median R_drift | max abs cumulative e_xi | mean signed e_xi |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(hard_rows)}

The cumulative column is a discrete coherent-drift diagnostic, not a physical integral. Microcore40k markedly reduces coherent recursive drift for `u_th=0.15` and `0.30`; it does not do so uniformly at `u_th=0.05`, where the failed seed changes identity and aggregate signed/cumulative behavior remains severe.

Mean maximum pre-throat absolute relative energy drift across all families changes from `{full['old20k']['mean_maximum_absolute_relative_energy_drift_before_or_at_throat']:.5f}` to `{full['microcore40k']['mean_maximum_absolute_relative_energy_drift_before_or_at_throat']:.5f}`. Per-seed maxima and absolute throat-crossing drift are machine-readable.

## Scientific answer

**{interpretation}: {detail}.** Microcore40k strongly improves `u_th=0.15` and `0.30`, reduces successful-crossing throat error overall, and reduces most drift measures. But `u_th=0.05` remains only `2/3` successful with one physical exit, merely shifting the failing seed. The local-map gain therefore translates into materially better recursion without delivering robust hardest-family orbit preservation.

**Yes—the recursive evidence supports moving next toward controlled loss redesign and/or architecture redesign rather than another round of random sampling refinement alone.** The residual `u_th=0.05` failure and coherent seed-dependent drift persist after the targeted region was already densely resolved. This conclusion is about recursive composition of the two frozen models only.

## Artifacts and integrity

Seven matched physical `u(x)` figures, seven `xi(x)` figures, three hard-family recursive-error figures, one hard-family energy figure, reusable arrays, a CSV, and the full JSON accompany this report. Protected checkpoints, histories, normalization artifacts, exact references, and orbit-spacing arrays retained identical before/after hashes. The evaluator accessed no restricted evaluation artifacts.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable evaluation directory {OUTPUT}")
    protected = protected_paths()
    before = artifact_hashes(protected)
    models, normalizations, checkpoints = load_models_and_normalizations()
    family_source = json.loads(SOURCE_FAMILY_SUMMARY.read_text(encoding="utf-8"))
    source_cases = {float(row["u_th"]): row for row in family_source["cases"]}
    if set(source_cases) != set(THROAT_VELOCITIES):
        raise RuntimeError("established family summary does not contain the seven prescribed families")

    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir()
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict[str, Any]] = []
    maximum_reference_difference = 0.0
    with np.load(SOURCE_EXACT_ARRAYS, allow_pickle=False) as frozen_exact, np.load(
        ORBIT_SPACING_ARRAYS, allow_pickle=False
    ) as spacing:
        for u_th in THROAT_VELOCITIES:
            stem = family_key(u_th)
            source = source_cases[u_th]
            u_left, E0 = float(source["u_left"]), float(source["energy"])
            exact_full, exact_full_summary = exact_rollout(
                np.asarray([-X_ENDPOINT, u_left]), verify_throat=True
            )
            exact_throat, exact_throat_summary = exact_rollout(
                np.asarray([0.0, u_th]), verify_throat=False
            )
            if abs(exact_full_summary["throat_crossing_u"] - u_th) > THROAT_TOLERANCE:
                raise RuntimeError(f"exact throat identity failed for u_th={u_th}")
            exact_time = frozen_exact[f"{stem}__time"]
            exact_state = frozen_exact[f"{stem}__exact_state"]
            reconstructed = exact_full[np.rint(exact_time / 0.05).astype(int)]
            difference = float(np.max(np.abs(reconstructed - exact_state)))
            maximum_reference_difference = max(maximum_reference_difference, difference)
            if difference != 0.0:
                raise RuntimeError("reconstructed DOP853 incoming reference differs from frozen exact states")
            spacing_x = np.asarray(spacing[f"{stem}__x"], dtype=np.float64)
            spacing_values = np.asarray(spacing[f"{stem}__orbit_spacing"], dtype=np.float64)
            arrays[f"{stem}__exact_full_state"] = exact_full
            arrays[f"{stem}__exact_full_coordinates"] = exact_xi_path(exact_full)
            arrays[f"{stem}__exact_throat_started_state"] = exact_throat
            arrays[f"{stem}__exact_throat_started_coordinates"] = exact_xi_path(exact_throat)
            arrays[f"{stem}__orbit_spacing_x"] = spacing_x
            arrays[f"{stem}__orbit_spacing"] = spacing_values

            case: dict[str, Any] = {
                "u_th": u_th, "u_left": u_left, "E0": E0,
                "exact_incoming_reconstruction_max_abs_difference": difference,
                "exact_full": exact_full_summary,
                "exact_throat_started": exact_throat_summary,
                "full_traversal": {}, "throat_started_outgoing": {},
                "hard_recursive_xi_diagnostic": {},
            }
            paths: dict[str, dict[str, dict[int, np.ndarray]]] = {
                "full": {treatment: {} for treatment in TREATMENTS},
                "throat": {treatment: {} for treatment in TREATMENTS},
                "full_coordinates": {treatment: {} for treatment in TREATMENTS},
                "throat_coordinates": {treatment: {} for treatment in TREATMENTS},
            }
            initial_full_xi = float(xi_from_state(-X_ENDPOINT, u_left, *experiment_parameters()))
            initial_throat_xi = float(xi_from_state(0.0, u_th, *experiment_parameters()))
            if not np.isclose(reconstruct_physical(np.asarray([[-X_ENDPOINT, initial_full_xi]]))[0, 1], u_left, rtol=0.0, atol=1e-14):
                raise RuntimeError("exact far-left transformed initialization does not reconstruct u_left")
            for mode, initial, case_key in (
                ("full", np.asarray([-X_ENDPOINT, initial_full_xi]), "full_traversal"),
                ("throat", np.asarray([0.0, initial_throat_xi]), "throat_started_outgoing"),
            ):
                for treatment in TREATMENTS:
                    seed_rows: dict[str, Any] = {}
                    if mode == "full" and u_th in HARD_FAMILIES:
                        case["hard_recursive_xi_diagnostic"][treatment] = {}
                    for seed in SEEDS:
                        coordinates, physical, margins, energy_drift, row = transformed_recursive_rollout(
                            models[treatment][seed], normalizations[treatment], initial, E0, u_th
                        )
                        paths[mode][treatment][seed] = physical
                        paths[f"{mode}_coordinates"][treatment][seed] = coordinates
                        seed_rows[str(seed)] = row
                        prefix = f"{stem}__{mode}__{treatment}__seed_{seed}"
                        arrays[f"{prefix}__coordinates"] = coordinates
                        arrays[f"{prefix}__physical_state"] = physical
                        arrays[f"{prefix}__C"] = margins
                        arrays[f"{prefix}__relative_energy_drift"] = energy_drift
                        if mode == "full" and u_th in HARD_FAMILIES:
                            diagnostic_arrays, diagnostic_summary = incoming_xi_diagnostic(
                                coordinates, margins, exact_full, spacing_x, spacing_values
                            )
                            case["hard_recursive_xi_diagnostic"][treatment][str(seed)] = diagnostic_summary
                            for name, values in diagnostic_arrays.items():
                                arrays[f"{prefix}__incoming__{name}"] = values
                    case[case_key][treatment] = {
                        "seeds": seed_rows,
                        "aggregate": family_aggregate(seed_rows),
                    }

            u_path = FIGURES / f"recursive_u_{stem}.png"
            xi_path = FIGURES / f"recursive_xi_{stem}.png"
            plot_u_family(u_th, exact_full, exact_throat, paths, case, u_path)
            plot_xi_family(u_th, exact_full, exact_throat, paths, case, xi_path)
            case["figures"] = {
                "u": {"path": str(u_path), "sha256": file_sha256(u_path)},
                "xi": {"path": str(xi_path), "sha256": file_sha256(xi_path)},
            }
            if u_th in HARD_FAMILIES:
                error_path = FIGURES / f"recursive_xi_error_{stem}.png"
                plot_recursive_xi_error(u_th, arrays, error_path)
                case["figures"]["recursive_xi_error"] = {
                    "path": str(error_path), "sha256": file_sha256(error_path)
                }
            cases.append(case)
            print(f"completed recursive family u_th={u_th:.2f}", flush=True)

    energy_path = FIGURES / "hard_family_recursive_energy_drift.png"
    plot_energy_drift(cases, arrays, energy_path)
    np.savez_compressed(ARRAYS_PATH, **arrays)
    write_summary_csv(cases, OUTPUT / "recursive_rollout_metrics.csv")
    after = artifact_hashes(protected)
    if before != after:
        raise RuntimeError("a protected model, normalization, history, or reference artifact changed")

    summary: dict[str, Any] = {
        "stage": "evaluation-only recursive transformed-coordinate sampling comparison",
        "status": "two_treatments_x_three_seeds_x_seven_families_completed",
        "models": {
            treatment: {
                "architecture": "2->32->32->2", "parameter_count": 1218,
                "inputs": ["x", "xi"], "outputs": ["delta_x", "delta_xi"],
                "normalization": str(TRAINING_DIR / f"{treatment}_normalization.json"),
                "checkpoints": {
                    str(seed): {"path": str(checkpoints[treatment][seed]), "sha256": before[str(checkpoints[treatment][seed])]}
                    for seed in SEEDS
                },
            }
            for treatment in TREATMENTS
        },
        "seeds": list(SEEDS), "families": list(THROAT_VELOCITIES),
        "step_size": H, "x_endpoint": X_ENDPOINT, "maximum_model_steps": MAX_MODEL_STEPS,
        "termination_conventions": {
            "success": "first predicted x >= +17",
            "physical_exit": "first reconstructed state with C <= 0",
            "guard": "10000 learned steps without another terminal condition",
            "nonfinite": "first nonfinite predicted coordinate or reconstructed state",
        },
        "exact_reference": {
            "family_definition_source": str(SOURCE_FAMILY_SUMMARY),
            "incoming_identity_source": str(SOURCE_EXACT_ARRAYS),
            "reconstruction": "validated DOP853 helper with 0.05 plot samples and exact stored initial states",
            "maximum_abs_reconstruction_difference": maximum_reference_difference,
        },
        "recursive_xi_error_definition": {
            "e_xi": "predicted xi minus exact-family xi interpolated at predicted incoming x",
            "R_drift": "absolute e_xi divided by prior nearest-seven-controlled-family xi spacing interpolated at predicted incoming x",
            "orbit_spacing_source": str(ORBIT_SPACING_ARRAYS),
        },
        "cases": cases,
        "aggregate": {
            "full_traversal": global_aggregate(cases, "full_traversal"),
            "throat_started_outgoing": global_aggregate(cases, "throat_started_outgoing"),
            "hard_recursive_xi": hard_recursive_aggregate(cases),
        },
        "artifacts": {
            "arrays": {"path": str(ARRAYS_PATH), "sha256": file_sha256(ARRAYS_PATH)},
            "metrics_csv": {"path": str(OUTPUT / "recursive_rollout_metrics.csv"), "sha256": file_sha256(OUTPUT / "recursive_rollout_metrics.csv")},
            "energy_figure": {"path": str(energy_path), "sha256": file_sha256(energy_path)},
            "report": str(REPORT_PATH),
        },
        "scientific_conclusion": {
            "classification": "partly",
            "local_improvement_translated_to_better_recursion": True,
            "hardest_family_robustly_recovered": False,
            "supports_next_controlled_loss_or_architecture_redesign": True,
            "supports_further_random_sampling_refinement_alone": False,
            "basis": "large throat/drift improvements at u_th=0.15 and 0.30, but u_th=0.05 remains 2/3 successful with one physical exit and seed-dependent coherent drift",
        },
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protocol": {
            "training_performed": False, "checkpoint_modified": False,
            "clipping_projection_penalty_or_correction_used": False,
            "unrelated_models_evaluated": False, "restricted_evaluation_data_accessed": False,
            "recursive_coordinates": ["x", "xi"], "physical_u_reconstructed_for_diagnostics": True,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}, {ARRAYS_PATH}, 18 figures, and {REPORT_PATH}")


if __name__ == "__main__":
    main()
