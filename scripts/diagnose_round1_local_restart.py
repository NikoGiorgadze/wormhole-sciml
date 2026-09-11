#!/usr/bin/env python3
"""Frozen-checkpoint local-map and exact-restart Round-I diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from wormhole_sciml import (
    integrate_trajectory,
    timelike_margin,
    total_speed_squared,
)
from wormhole_sciml.model_a import (
    TRAINING_SEEDS,
    TREATMENTS,
    Normalization,
    load_trained_model,
    predict_increments,
    recursive_rollout,
)
from wormhole_sciml.physics_gate import (
    PRODUCTION_SOLVER,
    TIGHT_SOLVER,
    experiment_parameters,
    frozen_member_identity_sha256,
    state_from_xi,
    xi_from_state,
)
from wormhole_sciml.stage1_data import H


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_ROOT / "output" / "stage1_model_a"
ROUND1_DIR = PROJECT_ROOT / "output" / "round1_model_a"
OUTPUT_DIR = PROJECT_ROOT / "output" / "round1_local_restart"
REPORT_DIR = PROJECT_ROOT / "reports" / "round1_local_restart"
FIGURE_DIR = REPORT_DIR / "figures"
VALIDATION_SUITE_PATH = PROJECT_ROOT / "reports" / "physics_gate" / "rollout_suites.json"

GRID_POINTS = 201
REPRESENTATIVE_IDS = (
    "validation-below_terminal",
    "validation-boundary_throat",
)
ERROR_LEVELS = (0.01, 0.05, 0.1)
TREATMENT_LABELS = {
    "physical_only": "physical-only",
    "collar_0p20": "0.20 collar",
    "collar_0p25": "0.25 collar",
}
TREATMENT_COLORS = {
    "physical_only": "#277da1",
    "collar_0p20": "#7b2cbf",
    "collar_0p25": "#2a9d8f",
}


def extract_json_object(path: Path, key: str) -> dict[str, Any]:
    """Read one named object and stop as soon as that object is complete."""

    decoder = json.JSONDecoder()
    marker = json.dumps(key) + ":"
    started = False
    buffer = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not started:
                position = line.find(marker)
                if position < 0:
                    continue
                started = True
                buffer = line[position + len(marker) :].lstrip()
            else:
                buffer += line
            try:
                value, _ = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{key} is not an object")
            return value
    raise ValueError(f"could not extract {key} from {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_runs() -> list[dict[str, Any]]:
    runs = []
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            checkpoint = ROUND1_DIR / treatment / f"seed_{seed}" / "best_checkpoint.pt"
            metadata = ROUND1_DIR / treatment / f"seed_{seed}" / "metadata.json"
            if not checkpoint.exists() or not metadata.exists():
                raise FileNotFoundError(f"missing frozen checkpoint for {treatment} seed {seed}")
            runs.append(
                {
                    "treatment": treatment,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": sha256(checkpoint),
                    "training_metadata": json.loads(metadata.read_text(encoding="utf-8")),
                }
            )
    if len(runs) != 9:
        raise RuntimeError("expected exactly nine frozen checkpoints")
    return runs


def dense_reference_grid() -> dict[str, np.ndarray]:
    wormhole, spiral = experiment_parameters()
    x_axis = np.linspace(-17.0, 17.0, GRID_POINTS, dtype=np.float64)
    xi_axis = np.linspace(-0.99, 0.99, GRID_POINTS, dtype=np.float64)
    x_grid, xi_grid = np.meshgrid(x_axis, xi_axis, indexing="xy")
    _, u_grid = state_from_xi(x_grid, xi_grid, wormhole, spiral)
    c_grid = timelike_margin(x_grid, u_grid, wormhole, spiral)
    if not np.all(c_grid > 0.0):
        raise RuntimeError("dense diagnostic grid contains a nonphysical initial state")
    states = np.column_stack((x_grid.ravel(), u_grid.ravel())).astype(np.float64)
    targets = np.empty_like(states)
    for index, state in enumerate(states):
        solution = integrate_trajectory(
            state,
            (0.0, H),
            wormhole,
            spiral,
            t_eval=(H,),
            stop_at_null_boundary=True,
            **PRODUCTION_SOLVER.kwargs(),
        )
        if solution.y.shape != (2, 1) or float(solution.t[-1]) != H:
            raise RuntimeError(f"dense-grid reference failed at row {index}")
        targets[index] = solution.y[:, -1] - state
        if (index + 1) % 10_000 == 0:
            print(f"DOP853 grid references: {index + 1}/{states.shape[0]}", flush=True)
    if not np.all(np.isfinite(targets)):
        raise RuntimeError("dense-grid targets contain nonfinite values")
    return {
        "x_axis": x_axis,
        "xi_axis": xi_axis,
        "x": x_grid,
        "xi": xi_grid,
        "u": np.asarray(u_grid, dtype=np.float64),
        "C": np.asarray(c_grid, dtype=np.float64),
        "delta_x_reference": targets[:, 0].reshape(x_grid.shape),
        "delta_u_reference": targets[:, 1].reshape(x_grid.shape),
    }


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def region_summaries(
    error: np.ndarray, x_grid: np.ndarray, xi_grid: np.ndarray
) -> dict[str, dict[str, float]]:
    absolute_xi = np.abs(xi_grid)
    masks = {
        "core": absolute_xi < 0.5,
        "shoulder": (absolute_xi >= 0.5) & (absolute_xi < 0.9),
        "edge": (absolute_xi >= 0.9) & (absolute_xi <= 0.99),
        "central_abs_x_le_8p5": np.abs(x_grid) <= 8.5,
        "outer_abs_x_gt_8p5": np.abs(x_grid) > 8.5,
    }
    return {name: statistics(error[mask]) for name, mask in masks.items()}


def adjacent_sign_agreement(values: np.ndarray) -> float:
    signs = np.sign(values)
    horizontal = signs[:, 1:] == signs[:, :-1]
    vertical = signs[1:, :] == signs[:-1, :]
    return float((np.sum(horizontal) + np.sum(vertical)) / (horizontal.size + vertical.size))


def evaluate_dense_models(
    runs: list[dict[str, Any]],
    grid: dict[str, np.ndarray],
    normalization: Normalization,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    states = np.column_stack((grid["x"].ravel(), grid["u"].ravel()))
    reference = np.column_stack(
        (grid["delta_x_reference"].ravel(), grid["delta_u_reference"].ravel())
    )
    shape = grid["x"].shape
    arrays: dict[str, np.ndarray] = {}
    per_treatment: dict[str, list[dict[str, np.ndarray]]] = {
        treatment: [] for treatment in TREATMENTS
    }
    for run in runs:
        model = load_trained_model(run["checkpoint"])
        prediction = predict_increments(model, states, normalization)
        error = prediction - reference
        combined = np.sqrt(
            (error[:, 0] / normalization.target_std[0]) ** 2
            + (error[:, 1] / normalization.target_std[1]) ** 2
        )
        prefix = f"{run['treatment']}__seed_{run['seed']}"
        arrays[f"{prefix}__delta_x_prediction"] = prediction[:, 0].reshape(shape)
        arrays[f"{prefix}__delta_u_prediction"] = prediction[:, 1].reshape(shape)
        arrays[f"{prefix}__error_x"] = error[:, 0].reshape(shape)
        arrays[f"{prefix}__error_u"] = error[:, 1].reshape(shape)
        arrays[f"{prefix}__combined_error"] = combined.reshape(shape)
        per_treatment[run["treatment"]].append(
            {
                "error_x": error[:, 0].reshape(shape),
                "error_u": error[:, 1].reshape(shape),
                "combined_error": combined.reshape(shape),
            }
        )

    summaries: dict[str, Any] = {}
    for treatment, seeds in per_treatment.items():
        treatment_arrays = {}
        for key in ("error_x", "error_u", "combined_error"):
            stack = np.stack([seed[key] for seed in seeds])
            treatment_arrays[f"mean_{key}"] = np.mean(stack, axis=0)
            treatment_arrays[f"std_{key}"] = np.std(stack, axis=0, ddof=1)
            arrays[f"{treatment}__mean_{key}"] = treatment_arrays[f"mean_{key}"]
            arrays[f"{treatment}__std_{key}"] = treatment_arrays[f"std_{key}"]
        mean_combined = treatment_arrays["mean_combined_error"]
        maximum_index = np.unravel_index(np.argmax(mean_combined), mean_combined.shape)
        summaries[treatment] = {
            "combined_error": statistics(mean_combined),
            "maximum_location": {
                "x": float(grid["x"][maximum_index]),
                "xi": float(grid["xi"][maximum_index]),
                "mean_error_x": float(treatment_arrays["mean_error_x"][maximum_index]),
                "mean_error_u": float(treatment_arrays["mean_error_u"][maximum_index]),
            },
            "seed_spread_combined_error": statistics(
                treatment_arrays["std_combined_error"]
            ),
            "regions": region_summaries(
                mean_combined, grid["x"], grid["xi"]
            ),
            "signed_error_u": {
                "mean": float(np.mean(treatment_arrays["mean_error_u"])),
                "positive_fraction": float(np.mean(treatment_arrays["mean_error_u"] > 0)),
                "negative_fraction": float(np.mean(treatment_arrays["mean_error_u"] < 0)),
                "adjacent_sign_agreement": adjacent_sign_agreement(
                    treatment_arrays["mean_error_u"]
                ),
            },
        }
    return arrays, summaries


def reference_representatives() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
    str,
]:
    suite = extract_json_object(VALIDATION_SUITE_PATH, "validation")
    if suite["count"] != 24:
        raise RuntimeError("frozen validation suite does not contain 24 members")
    if suite["frozen_member_identity_sha256"] != frozen_member_identity_sha256(
        suite["members"]
    ):
        raise RuntimeError("frozen validation suite identity mismatch")
    metadata = {member["id"]: member for member in suite["members"]}
    if any(identifier not in metadata for identifier in REPRESENTATIVE_IDS):
        raise RuntimeError("documented representative identity is missing")
    wormhole, spiral = experiment_parameters()
    references = {}
    for identifier in REPRESENTATIVE_IDS:
        member = metadata[identifier]
        times = np.arange(int(member["step_count"]) + 1, dtype=np.float64) * H
        solution = integrate_trajectory(
            (member["x0"], member["u0"]),
            (0.0, float(member["T_i"])),
            wormhole,
            spiral,
            t_eval=times,
            stop_at_null_boundary=True,
            **TIGHT_SOLVER.kwargs(),
        )
        if solution.y.shape != (2, times.size):
            raise RuntimeError(f"could not evaluate frozen representative {identifier}")
        state = np.asarray(solution.y.T, dtype=np.float64)
        references[identifier] = {
            "time": times,
            "state": state,
            "xi": xi_from_state(state[:, 0], state[:, 1], wormhole, spiral),
            "v_total": np.sqrt(
                total_speed_squared(state[:, 0], state[:, 1], wormhole, spiral)
            ),
        }
    return (
        {identifier: metadata[identifier] for identifier in REPRESENTATIVE_IDS},
        references,
        suite["frozen_member_identity_sha256"],
    )


def snapped_restart_indices(step_count: int) -> np.ndarray:
    return np.asarray(
        [0, int(math.floor(step_count / 3.0 + 0.5)), int(math.floor(2.0 * step_count / 3.0 + 0.5))],
        dtype=np.int64,
    )


def first_threshold_time(
    time_since_restart: np.ndarray, error: np.ndarray, level: float
) -> float | None:
    indices = np.flatnonzero(error > level)
    return None if not indices.size else float(time_since_restart[int(indices[0])])


def value_after_steps(values: np.ndarray, steps: int) -> float | None:
    return None if steps >= len(values) else float(values[steps])


def evaluate_restarts(
    runs: list[dict[str, Any]],
    representative_metadata: dict[str, dict[str, Any]],
    references: dict[str, dict[str, np.ndarray]],
    normalization: Normalization,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    wormhole, spiral = experiment_parameters()
    arrays: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    local_summaries: list[dict[str, Any]] = []
    models = {
        (run["treatment"], run["seed"]): load_trained_model(run["checkpoint"])
        for run in runs
    }
    for identifier, reference in references.items():
        times = reference["time"]
        states = reference["state"]
        restart_indices = snapped_restart_indices(len(times) - 1)
        prefix = identifier.replace("-", "_")
        arrays[f"{prefix}__reference_time"] = times
        arrays[f"{prefix}__reference_state"] = states
        arrays[f"{prefix}__reference_xi"] = reference["xi"]
        arrays[f"{prefix}__reference_v_total"] = reference["v_total"]
        arrays[f"{prefix}__restart_indices"] = restart_indices
        arrays[f"{prefix}__restart_times"] = times[restart_indices]

        exact_reference_increment = states[1:] - states[:-1]
        for run in runs:
            treatment = run["treatment"]
            seed = run["seed"]
            model = models[(treatment, seed)]
            local_prediction = predict_increments(model, states[:-1], normalization)
            local_error = local_prediction - exact_reference_increment
            local_combined = np.sqrt(
                (local_error[:, 0] / normalization.target_std[0]) ** 2
                + (local_error[:, 1] / normalization.target_std[1]) ** 2
            )
            run_prefix = f"{prefix}__{treatment}__seed_{seed}"
            arrays[f"{run_prefix}__local_error_x"] = local_error[:, 0]
            arrays[f"{run_prefix}__local_error_u"] = local_error[:, 1]
            arrays[f"{run_prefix}__local_combined_error"] = local_combined
            local_summaries.append(
                {
                    "representative": identifier,
                    "treatment": treatment,
                    "seed": seed,
                    "mean_local_error": float(np.mean(local_combined)),
                    "maximum_local_error": float(np.max(local_combined)),
                    "maximum_local_error_time": float(times[int(np.argmax(local_combined))]),
                }
            )
            for restart_number, restart_index in enumerate(restart_indices):
                initial_state = states[restart_index].copy()
                steps = len(times) - 1 - int(restart_index)
                predicted = recursive_rollout(model, initial_state, steps, normalization)
                expected = states[restart_index:]
                if not np.array_equal(predicted[0], initial_state):
                    raise RuntimeError("recursive restart did not preserve exact initial state")
                error = predicted - expected
                combined = np.sqrt(
                    (error[:, 0] / normalization.input_std[0]) ** 2
                    + (error[:, 1] / normalization.input_std[1]) ** 2
                )
                absolute_time = times[restart_index:]
                elapsed = absolute_time - absolute_time[0]
                predicted_v_total = np.sqrt(
                    total_speed_squared(
                        predicted[:, 0], predicted[:, 1], wormhole, spiral
                    )
                )
                predicted_c = timelike_margin(
                    predicted[:, 0], predicted[:, 1], wormhole, spiral
                )
                exits = np.flatnonzero(predicted_c <= 0.0)
                first_exit = None if not exits.size else float(absolute_time[int(exits[0])])
                restart_prefix = f"{run_prefix}__restart_{restart_number}"
                arrays[f"{restart_prefix}__absolute_time"] = absolute_time
                arrays[f"{restart_prefix}__elapsed_time"] = elapsed
                arrays[f"{restart_prefix}__initial_state"] = initial_state
                arrays[f"{restart_prefix}__predicted_state"] = predicted
                arrays[f"{restart_prefix}__error"] = error
                arrays[f"{restart_prefix}__combined_error"] = combined
                arrays[f"{restart_prefix}__predicted_v_total"] = predicted_v_total
                arrays[f"{restart_prefix}__predicted_C"] = predicted_c
                summaries.append(
                    {
                        "representative": identifier,
                        "reference_class": representative_metadata[identifier][
                            "reference_class"
                        ],
                        "treatment": treatment,
                        "seed": seed,
                        "restart_number": restart_number,
                        "restart_index": int(restart_index),
                        "restart_time": float(times[restart_index]),
                        "restart_state": initial_state.tolist(),
                        "remaining_steps": steps,
                        "error_after_1_step": value_after_steps(combined, 1),
                        "error_after_5_steps": value_after_steps(combined, 5),
                        "error_after_10_steps": value_after_steps(combined, 10),
                        "first_elapsed_time_above_0p01": first_threshold_time(
                            elapsed, combined, 0.01
                        ),
                        "first_elapsed_time_above_0p05": first_threshold_time(
                            elapsed, combined, 0.05
                        ),
                        "first_elapsed_time_above_0p1": first_threshold_time(
                            elapsed, combined, 0.1
                        ),
                        "final_error": float(combined[-1]),
                        "physical_exit_time": first_exit,
                    }
                )
    for array in arrays.values():
        if not np.all(np.isfinite(array)):
            raise RuntimeError("restart diagnostic contains a nonfinite saved array")
    return arrays, summaries, local_summaries


def aggregate_restart_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for representative in REPRESENTATIVE_IDS:
        for treatment in TREATMENTS:
            for restart_number in range(3):
                selected = [
                    row
                    for row in rows
                    if row["representative"] == representative
                    and row["treatment"] == treatment
                    and row["restart_number"] == restart_number
                ]
                aggregate: dict[str, Any] = {
                    "representative": representative,
                    "treatment": treatment,
                    "restart_number": restart_number,
                    "restart_time": selected[0]["restart_time"],
                    "restart_state": selected[0]["restart_state"],
                    "seed_count": len(selected),
                    "physical_exit_count": sum(
                        row["physical_exit_time"] is not None for row in selected
                    ),
                    "physical_exit_times": [
                        row["physical_exit_time"]
                        for row in selected
                        if row["physical_exit_time"] is not None
                    ],
                }
                for key in (
                    "error_after_1_step",
                    "error_after_5_steps",
                    "error_after_10_steps",
                    "final_error",
                ):
                    values = [row[key] for row in selected if row[key] is not None]
                    aggregate[f"mean_{key}"] = float(np.mean(values))
                    aggregate[f"std_{key}"] = float(np.std(values, ddof=1))
                for level_key in (
                    "first_elapsed_time_above_0p01",
                    "first_elapsed_time_above_0p05",
                    "first_elapsed_time_above_0p1",
                ):
                    values = [row[level_key] for row in selected if row[level_key] is not None]
                    aggregate[f"{level_key}_reached_count"] = len(values)
                    aggregate[f"mean_{level_key}"] = (
                        None if not values else float(np.mean(values))
                    )
                output.append(aggregate)
    return output


def aggregate_local_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for representative in REPRESENTATIVE_IDS:
        for treatment in TREATMENTS:
            selected = [
                row
                for row in rows
                if row["representative"] == representative
                and row["treatment"] == treatment
            ]
            if len(selected) != len(TRAINING_SEEDS):
                raise RuntimeError("trajectory-local summary does not contain three seeds")
            output.append(
                {
                    "representative": representative,
                    "treatment": treatment,
                    "mean_local_error": float(
                        np.mean([row["mean_local_error"] for row in selected])
                    ),
                    "mean_seed_maximum_local_error": float(
                        np.mean([row["maximum_local_error"] for row in selected])
                    ),
                    "seed_maximum_local_error_times": [
                        row["maximum_local_error_time"] for row in selected
                    ],
                }
            )
    return output


def treatment_seed_stack(
    arrays: dict[str, np.ndarray], key_template: str, treatment: str
) -> np.ndarray:
    return np.stack(
        [arrays[key_template.format(treatment=treatment, seed=seed)] for seed in TRAINING_SEEDS]
    )


def make_local_map_figures(
    grid: dict[str, np.ndarray], arrays: dict[str, np.ndarray]
) -> None:
    extent = (-17.0, 17.0, -0.99, 0.99)
    mean_ex = [arrays[f"{treatment}__mean_error_x"] for treatment in TREATMENTS]
    mean_eu = [arrays[f"{treatment}__mean_error_u"] for treatment in TREATMENTS]
    mean_e = [arrays[f"{treatment}__mean_combined_error"] for treatment in TREATMENTS]
    ex_limit = float(np.quantile(np.abs(np.stack(mean_ex)), 0.995))
    eu_limit = float(np.quantile(np.abs(np.stack(mean_eu)), 0.995))
    e_limit = float(np.quantile(np.stack(mean_e), 0.995))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    images = []
    for row, treatment in enumerate(TREATMENTS):
        images.append(
            axes[row, 0].imshow(
                arrays[f"{treatment}__mean_error_x"], origin="lower", extent=extent,
                aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(0, -ex_limit, ex_limit)
            )
        )
        images.append(
            axes[row, 1].imshow(
                arrays[f"{treatment}__mean_error_u"], origin="lower", extent=extent,
                aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(0, -eu_limit, eu_limit)
            )
        )
        images.append(
            axes[row, 2].imshow(
                arrays[f"{treatment}__mean_combined_error"], origin="lower", extent=extent,
                aspect="auto", cmap="magma", vmin=0, vmax=e_limit
            )
        )
        axes[row, 0].set_ylabel(f"{TREATMENT_LABELS[treatment]}\n$\\xi$")
        for column in range(3):
            axes[row, column].set_xlabel("$x$")
    for column, title in enumerate((r"mean signed $e_x$", r"mean signed $e_u$", r"mean $E$")):
        axes[0, column].set_title(title)
    for column, image in enumerate((images[0], images[1], images[2])):
        fig.colorbar(image, ax=axes[:, column], shrink=0.85)
    fig.suptitle("Dense physical local-flow error (three-seed treatment means)")
    fig.savefig(FIGURE_DIR / "local_flow_error_maps.png")
    plt.close(fig)

    limits = []
    for key in ("std_error_x", "std_error_u", "std_combined_error"):
        limits.append(
            float(
                np.quantile(
                    np.stack([arrays[f"{treatment}__{key}"] for treatment in TREATMENTS]),
                    0.995,
                )
            )
        )
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    column_images = [None, None, None]
    for row, treatment in enumerate(TREATMENTS):
        for column, key in enumerate(("std_error_x", "std_error_u", "std_combined_error")):
            column_images[column] = axes[row, column].imshow(
                arrays[f"{treatment}__{key}"], origin="lower", extent=extent,
                aspect="auto", cmap="viridis", vmin=0, vmax=limits[column]
            )
            axes[row, column].set_xlabel("$x$")
        axes[row, 0].set_ylabel(f"{TREATMENT_LABELS[treatment]}\n$\\xi$")
    for column, title in enumerate((r"seed std($e_x$)", r"seed std($e_u$)", r"seed std($E$)")):
        axes[0, column].set_title(title)
        fig.colorbar(column_images[column], ax=axes[:, column], shrink=0.85)
    fig.suptitle("Dense physical local-flow seed spread")
    fig.savefig(FIGURE_DIR / "local_flow_seed_spread.png")
    plt.close(fig)


def make_restart_physics_figure(
    identifier: str,
    arrays: dict[str, np.ndarray],
    output_name: str,
) -> None:
    prefix = identifier.replace("-", "_")
    restart_indices = arrays[f"{prefix}__restart_indices"]
    full_time = arrays[f"{prefix}__reference_time"]
    full_reference = arrays[f"{prefix}__reference_state"]
    full_v_total = arrays[f"{prefix}__reference_v_total"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for row, restart_index in enumerate(restart_indices):
        time = full_time[restart_index:]
        reference = full_reference[restart_index:]
        reference_v = full_v_total[restart_index:]
        for column, reference_values in enumerate(
            (reference[:, 0], reference[:, 1], reference_v)
        ):
            axes[row, column].plot(time, reference_values, color="black", lw=2, label="DOP853")
        for treatment in TREATMENTS:
            state_stack = treatment_seed_stack(
                arrays,
                f"{prefix}__{{treatment}}__seed_{{seed}}__restart_{row}__predicted_state",
                treatment,
            )
            v_stack = treatment_seed_stack(
                arrays,
                f"{prefix}__{{treatment}}__seed_{{seed}}__restart_{row}__predicted_v_total",
                treatment,
            )
            for column, values in enumerate(
                (state_stack[:, :, 0], state_stack[:, :, 1], v_stack)
            ):
                mean = np.mean(values, axis=0)
                spread = np.std(values, axis=0, ddof=1)
                axes[row, column].plot(
                    time, mean, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment]
                )
                axes[row, column].fill_between(
                    time, mean - spread, mean + spread,
                    color=TREATMENT_COLORS[treatment], alpha=0.15
                )
        axes[row, 0].set_ylabel(f"restart s={full_time[restart_index]:.1f}\n$x$")
        axes[row, 1].set_ylabel("$u$")
        axes[row, 2].set_ylabel(r"$v_{tot}$")
        for column in range(3):
            axes[row, column].set_xlabel("absolute time $s$")
    for column, title in enumerate(("position", "radial velocity", "validated total speed")):
        axes[0, column].set_title(title)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Exact-state restarted rollout: {identifier}")
    fig.savefig(FIGURE_DIR / output_name)
    plt.close(fig)


def make_error_growth_figure(arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for row, identifier in enumerate(REPRESENTATIVE_IDS):
        prefix = identifier.replace("-", "_")
        restart_times = arrays[f"{prefix}__restart_times"]
        for restart_number in range(3):
            ax = axes[row, restart_number]
            elapsed = arrays[
                f"{prefix}__physical_only__seed_101__restart_{restart_number}__elapsed_time"
            ]
            for treatment in TREATMENTS:
                stack = treatment_seed_stack(
                    arrays,
                    f"{prefix}__{{treatment}}__seed_{{seed}}__restart_{restart_number}__combined_error",
                    treatment,
                )
                mean = np.mean(stack, axis=0)
                spread = np.std(stack, axis=0, ddof=1)
                ax.plot(elapsed, mean, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment])
                ax.fill_between(elapsed, mean-spread, mean+spread, color=TREATMENT_COLORS[treatment], alpha=0.15)
            for level in ERROR_LEVELS:
                ax.axhline(level, color="0.5", ls="--", lw=0.6)
            ax.set(
                title=f"restart $s_k$={restart_times[restart_number]:.1f}",
                xlabel=r"elapsed time $\tau$",
                ylabel=f"{identifier.replace('validation-', '')}\n$e_z$" if restart_number == 0 else "$e_z$",
            )
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Restarted recursive error growth on fixed individual trajectories")
    fig.savefig(FIGURE_DIR / "restart_error_growth.png")
    plt.close(fig)


def make_trajectory_local_error_figure(arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for row, identifier in enumerate(REPRESENTATIVE_IDS):
        prefix = identifier.replace("-", "_")
        time = arrays[f"{prefix}__reference_time"][:-1]
        for treatment in TREATMENTS:
            for column, key in enumerate(("local_error_x", "local_error_u", "local_combined_error")):
                stack = treatment_seed_stack(
                    arrays,
                    f"{prefix}__{{treatment}}__seed_{{seed}}__{key}",
                    treatment,
                )
                mean = np.mean(stack, axis=0)
                spread = np.std(stack, axis=0, ddof=1)
                axes[row, column].plot(time, mean, color=TREATMENT_COLORS[treatment], label=TREATMENT_LABELS[treatment])
                axes[row, column].fill_between(time, mean-spread, mean+spread, color=TREATMENT_COLORS[treatment], alpha=0.15)
        axes[row, 0].set_ylabel(f"{identifier.replace('validation-', '')}\nlocal error")
        for column in range(3):
            axes[row, column].set_xlabel("absolute time $s$")
    for column, title in enumerate((r"signed local $\epsilon_x$", r"signed local $\epsilon_u$", r"local $E$")):
        axes[0, column].set_title(title)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("One-step local-map error along exact reference trajectories")
    fig.savefig(FIGURE_DIR / "trajectory_local_error.png")
    plt.close(fig)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def make_report(manifest: dict[str, Any]) -> None:
    local = manifest["local_grid_summary"]
    restarts = manifest["restart_aggregate"]
    trajectory_local = manifest["trajectory_local_error_aggregate"]
    lines = [
        "# Frozen Model-A local-map and exact-restart diagnostic",
        "",
        "This addendum is diagnostic only. It uses the nine existing restored checkpoints without training or modification, constructs new DOP853 references directly from the validated physics, and reads only the frozen validation trajectory object needed for the two documented representatives.",
        "",
        "## Dense physical local map",
        "",
        f"The deterministic grid contains {GRID_POINTS} x {GRID_POINTS} = {GRID_POINTS**2:,} physical states on $x\\in[-17,17]$ and $\\xi\\in[-0.99,0.99]$. Every state produced a finite DOP853 target at $h=0.2$.",
        "",
        "| treatment | mean E | median E | p90 | p99 | max E | max location (x, xi) | e_x there | e_u there | mean seed std(E) |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for treatment in TREATMENTS:
        row = local[treatment]
        stats = row["combined_error"]
        location = row["maximum_location"]
        spread = row["seed_spread_combined_error"]
        lines.append(
            f"| {TREATMENT_LABELS[treatment]} | {fmt(stats['mean'])} | {fmt(stats['median'])} | {fmt(stats['p90'])} | {fmt(stats['p99'])} | {fmt(stats['maximum'])} | ({fmt(location['x'])}, {fmt(location['xi'])}) | {fmt(location['mean_error_x'])} | {fmt(location['mean_error_u'])} | {fmt(spread['mean'])} |"
        )
    lines.extend([
        "",
        "| treatment | core mean E | shoulder mean E | edge mean E | central mean E | outer mean E | e_u adjacent-sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for treatment in TREATMENTS:
        row = local[treatment]
        regions = row["regions"]
        lines.append(
            f"| {TREATMENT_LABELS[treatment]} | {fmt(regions['core']['mean'])} | {fmt(regions['shoulder']['mean'])} | {fmt(regions['edge']['mean'])} | {fmt(regions['central_abs_x_le_8p5']['mean'])} | {fmt(regions['outer_abs_x_gt_8p5']['mean'])} | {fmt(row['signed_error_u']['adjacent_sign_agreement'])} |"
        )
    lines.extend([
        "",
        "The signed $e_u$ maps contain broad, alternating sign domains around the central dynamics rather than isolated high-error pixels. Their 98.2--98.5% adjacent-grid sign agreement confirms strong spatial coherence without dividing by targets that cross zero. The largest treatment-mean $E$ values occur near $x=0.51$ and negative $\\xi$ between -0.68 and -0.81; the core/shoulder and central-$x$ summaries show that the largest errors are not confined to the nominal edge strip.",
        "",
        "![Dense local-flow errors](figures/local_flow_error_maps.png)",
        "",
        "![Dense local-flow seed spread](figures/local_flow_seed_spread.png)",
        "",
        "## Exact-state restarted rollouts",
        "",
        "The representatives are `validation-below_terminal` (escaping) and `validation-boundary_throat` (null-boundary-asymptotic). Each restart begins from the exact DOP853 state at the recorded grid index; earlier neural-network error is discarded.",
        "",
        "| representative | treatment | restart s | E after 1 step | E after 5 | E after 10 | first tau > .01 (count/3) | first tau > .05 (count/3) | first tau > .1 (count/3) | final E | exit absolute s (count/3) |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|",
    ])
    for row in restarts:
        def threshold_cell(key: str) -> str:
            return f"{fmt(row['mean_' + key])} ({row[key + '_reached_count']}/3)"
        exit_times = ", ".join(fmt(value) for value in row["physical_exit_times"])
        exit_cell = f"{exit_times or '---'} ({row['physical_exit_count']}/3)"
        lines.append(
            f"| `{row['representative']}` | {TREATMENT_LABELS[row['treatment']]} | {fmt(row['restart_time'])} | {fmt(row['mean_error_after_1_step'])} | {fmt(row['mean_error_after_5_steps'])} | {fmt(row['mean_error_after_10_steps'])} | {threshold_cell('first_elapsed_time_above_0p01')} | {threshold_cell('first_elapsed_time_above_0p05')} | {threshold_cell('first_elapsed_time_above_0p1')} | {fmt(row['mean_final_error'])} | {exit_cell} |"
        )
    lines.extend([
        "",
        "The threshold times are elapsed times since restart and are descriptive only. Absolute physical times, individual-seed curves, exact restart states, signed component errors, $v_{tot}$, and admissibility crossings are retained in the machine-readable artifacts.",
        "",
        "For `validation-below_terminal`, exact restarts at $s=10.6$ and $21.0$ reset the error to $E\\lesssim0.003$ after one step for every treatment; the $s=21.0$ restarts never reach $E=0.05$ during the remaining horizon. This substantially removes the earlier discrepancy and identifies recursive accumulation as the main source in this inspected escaping example. No restarted escaping rollout leaves the physical domain.",
        "",
        "For `validation-boundary_throat`, the picture is phase dependent. The $s=3.2$ restart immediately has mean one-step $E=0.0105$--$0.0180$, consistent with a locally difficult region. At $s=6.6$, the collar runs reset to $E=0.00064$--$0.00120$ after one step and remain below $0.05$, so much of their earlier discrepancy was cumulative, although 2/3 of 0.20-collar seeds and 1/3 of 0.25-collar seeds still cross $C=0$ near the terminal boundary. Physical-only remains locally inaccurate after the late restart (one-step $E=0.0115$) and all three seeds cross $C=0$. These are observations on two fixed representatives, not treatment-selection criteria.",
        "",
        "![Escaping representative restarts](figures/restart_rollout_escaping.png)",
        "",
        "![Null-boundary representative restarts](figures/restart_rollout_null_boundary.png)",
        "",
        "![Restart error growth](figures/restart_error_growth.png)",
        "",
        "## Local-map connection along the trajectories",
        "",
        "At every exact DOP853 reference state with a following step, the report evaluates the checkpoint's one-step error against the next exact reference state. This separates local flow-map error from recursively accumulated state error.",
        "",
        "Here local $E$ is standardized by the increment standard deviations, whereas restarted state $e_z$ is standardized by the state standard deviations; their numerical magnitudes are therefore not directly comparable.",
        "",
        "| representative | treatment | mean local E | mean of per-seed maximum local E | times of per-seed maxima |",
        "|---|---|---:|---:|---|",
    ])
    for row in trajectory_local:
        times = ", ".join(fmt(value) for value in row["seed_maximum_local_error_times"])
        lines.append(
            f"| `{row['representative']}` | {TREATMENT_LABELS[row['treatment']]} | {fmt(row['mean_local_error'])} | {fmt(row['mean_seed_maximum_local_error'])} | {times} |"
        )
    lines.extend([
        "",
        "Along the escaping reference, the strongest local errors occur early (approximately $s=1.2$--$4.0$), after which exact late restarts track substantially better. Along the null-boundary reference, all seeds reach their maximum local error near $s=2.0$--$2.2$, supporting the immediate-error signal from the middle restart; the later behavior then separates local phase-space difficulty from accumulated recursive drift.",
        "",
        "![Trajectory-local one-step errors](figures/trajectory_local_error.png)",
        "",
        "No changing-cohort aggregate trajectory curve is used here: every plotted rollout belongs to one fixed representative and extends to that representative's fixed horizon.",
        "",
        "No domain treatment is selected, and no training, capacity comparison, constraint, or Model B work was performed.",
    ])
    (REPORT_DIR / "LOCAL_MAP_RESTART_DIAGNOSTIC.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 8.5,
            "axes.grid": False,
        }
    )
    normalization = Normalization.from_stage1(STAGE1_DIR / "normalization.json")
    runs = model_runs()
    checkpoint_hashes_before = {
        f"{run['treatment']}__seed_{run['seed']}": run["checkpoint_sha256"]
        for run in runs
    }

    grid = dense_reference_grid()
    reference_path = OUTPUT_DIR / "dense_grid_reference.npz"
    np.savez_compressed(reference_path, **grid)
    grid_arrays, local_summaries = evaluate_dense_models(runs, grid, normalization)
    grid_path = OUTPUT_DIR / "checkpoint_local_grids.npz"
    np.savez_compressed(grid_path, **grid_arrays)

    representative_metadata, references, validation_suite_identity = (
        reference_representatives()
    )
    restart_arrays, restart_rows, trajectory_local_rows = evaluate_restarts(
        runs, representative_metadata, references, normalization
    )
    restart_path = OUTPUT_DIR / "restarted_rollouts.npz"
    np.savez_compressed(restart_path, **restart_arrays)
    restart_aggregate = aggregate_restart_summaries(restart_rows)
    trajectory_local_aggregate = aggregate_local_summaries(trajectory_local_rows)

    make_local_map_figures(grid, grid_arrays)
    make_restart_physics_figure(
        REPRESENTATIVE_IDS[0], restart_arrays, "restart_rollout_escaping.png"
    )
    make_restart_physics_figure(
        REPRESENTATIVE_IDS[1], restart_arrays, "restart_rollout_null_boundary.png"
    )
    make_error_growth_figure(restart_arrays)
    make_trajectory_local_error_figure(restart_arrays)

    checkpoint_hashes_after = {
        f"{run['treatment']}__seed_{run['seed']}": sha256(run["checkpoint"])
        for run in runs
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("a frozen checkpoint changed during diagnostics")
    manifest = {
        "stage": "post-Round-I diagnostic only",
        "status": "complete",
        "training_performed": False,
        "grid": {
            "shape": [GRID_POINTS, GRID_POINTS],
            "point_count": GRID_POINTS**2,
            "x_range": [-17.0, 17.0],
            "xi_range": [-0.99, 0.99],
            "h": H,
            "reference_solver": PRODUCTION_SOLVER.metadata(),
        },
        "normalization": str(STAGE1_DIR / "normalization.json"),
        "checkpoint_hashes_before_and_after_identical": True,
        "checkpoint_sha256": checkpoint_hashes_after,
        "local_grid_summary": local_summaries,
        "representatives": {
            identifier: {
                "reference_class": representative_metadata[identifier]["reference_class"],
                "T_i": representative_metadata[identifier]["T_i"],
                "restart_indices": restart_arrays[
                    f"{identifier.replace('-', '_')}__restart_indices"
                ].tolist(),
                "restart_times": restart_arrays[
                    f"{identifier.replace('-', '_')}__restart_times"
                ].tolist(),
                "restart_states": [
                    references[identifier]["state"][index].tolist()
                    for index in restart_arrays[
                        f"{identifier.replace('-', '_')}__restart_indices"
                    ]
                ],
            }
            for identifier in REPRESENTATIVE_IDS
        },
        "restart_individual": restart_rows,
        "restart_aggregate": restart_aggregate,
        "trajectory_local_error_summary": trajectory_local_rows,
        "trajectory_local_error_aggregate": trajectory_local_aggregate,
        "v_total_definition": "sqrt(wormhole_sciml.total_speed_squared)",
        "artifacts": {
            "dense_grid_reference": {
                "path": str(reference_path),
                "sha256": sha256(reference_path),
            },
            "checkpoint_local_grids": {
                "path": str(grid_path),
                "sha256": sha256(grid_path),
            },
            "restarted_rollouts": {
                "path": str(restart_path),
                "sha256": sha256(restart_path),
            },
        },
        "validation_suite_identity_sha256": validation_suite_identity,
        "representative_identity_sha256": frozen_member_identity_sha256(
            list(representative_metadata.values())
        ),
        "restricted_data_opened": False,
        "domain_treatment_selected": False,
        "aggregate_curve_convention": (
            "Only fixed individual representatives are plotted; no changing-cohort "
            "trajectory aggregation is used."
        ),
        "anomalies": [],
    }
    manifest_path = OUTPUT_DIR / "diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_report(manifest)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {REPORT_DIR / 'LOCAL_MAP_RESTART_DIAGNOSTIC.md'}")


if __name__ == "__main__":
    main()
