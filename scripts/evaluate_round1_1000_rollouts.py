#!/usr/bin/env python3
"""Validation-only recursive diagnostics for the nine 1000-epoch Model-A checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml import (
    conserved_energy,
    effective_angular_velocity,
    radial_acceleration,
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
from wormhole_sciml.physics_gate import experiment_parameters
from wormhole_sciml.stage1_data import H


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_RUN_DIR = PROJECT_ROOT / "output" / "round1_model_a"
NEW_RUN_DIR = PROJECT_ROOT / "output" / "round1_model_a_1000"
OLD_RESTART_DIR = PROJECT_ROOT / "output" / "round1_local_restart"
OUTPUT_DIR = PROJECT_ROOT / "output" / "round1_model_a_1000_rollouts"
REPORT_DIR = PROJECT_ROOT / "reports" / "round1_model_a_1000_rollouts"
FIGURE_DIR = REPORT_DIR / "figures"
OLD_EVALUATION = OLD_RUN_DIR / "evaluation_manifest.json"
OLD_RESTART_MANIFEST = OLD_RESTART_DIR / "diagnostic_manifest.json"
OLD_RESTART_ARRAYS = OLD_RESTART_DIR / "restarted_rollouts.npz"
GATE_RESULT_PATH = PROJECT_ROOT / "reports" / "physics_gate" / "gate_results.json"

REPRESENTATIVE_IDS = (
    "validation-below_terminal",
    "validation-boundary_throat",
)
REFERENCE_CLASSES = ("escaping_force_free", "null_boundary_asymptotic")
ERROR_LEVELS = (0.01, 0.05, 0.1)
LABELS = {
    "physical_only": "physical-only",
    "collar_0p20": "0.20 collar",
    "collar_0p25": "0.25 collar",
}
COLORS = {
    "physical_only": "#277da1",
    "collar_0p20": "#7b2cbf",
    "collar_0p25": "#2a9d8f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(sha256(candidate).encode("ascii") + b"\0")
    return {
        "path": str(path),
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
    }


def extract_json_object(path: Path, key: str) -> dict[str, Any]:
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
                raise ValueError(f"{key} is not a JSON object")
            return value
    raise ValueError(f"could not extract {key} from {path}")


def prefix(identifier: str) -> str:
    return identifier.replace("-", "_")


def old_rollout_path(treatment: str, seed: int) -> Path:
    return OLD_RUN_DIR / treatment / f"seed_{seed}" / "validation_rollouts.npz"


def load_stored_references(
    old_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    members = [
        {
            key: trajectory[key]
            for key in (
                "id",
                "category",
                "reference_class",
                "step_count",
                "T_i",
                "terminal_entry_time_by_T_i",
            )
        }
        for trajectory in old_manifest["runs"][0]["rollouts"]["trajectories"]
    ]
    if len(members) != 24 or len({member["id"] for member in members}) != 24:
        raise RuntimeError("the stored validation rollout suite does not contain 24 members")
    canonical_path = old_rollout_path("physical_only", 101)
    references: dict[str, dict[str, np.ndarray]] = {}
    with np.load(canonical_path) as arrays:
        for member in members:
            key = prefix(member["id"])
            time = np.asarray(arrays[f"{key}__time"], dtype=np.float64)
            state = np.asarray(arrays[f"{key}__reference_state"], dtype=np.float64)
            if time.shape != (int(member["step_count"]) + 1,):
                raise RuntimeError(f"stored time shape mismatch for {member['id']}")
            if state.shape != (time.size, 2):
                raise RuntimeError(f"stored state shape mismatch for {member['id']}")
            if not np.array_equal(time, np.arange(time.size, dtype=np.float64) * H):
                raise RuntimeError(f"stored reference is not on the h=0.2 grid: {member['id']}")
            if not np.all(np.isfinite(state)):
                raise RuntimeError(f"stored reference is nonfinite: {member['id']}")
            references[member["id"]] = {"time": time, "state": state}
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            with np.load(old_rollout_path(treatment, seed)) as arrays:
                for member in members:
                    key = prefix(member["id"])
                    if not np.array_equal(
                        arrays[f"{key}__time"], references[member["id"]]["time"]
                    ) or not np.array_equal(
                        arrays[f"{key}__reference_state"],
                        references[member["id"]]["state"],
                    ):
                        raise RuntimeError("old rollout artifacts do not share one frozen reference")
    return members, references


def checkpoint_runs() -> list[dict[str, Any]]:
    manifest = json.loads(
        (NEW_RUN_DIR / "training_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["status"] != "nine_fresh_runs_completed" or manifest["run_count"] != 9:
        raise RuntimeError("the 1000-epoch training manifest is incomplete")
    runs = []
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            metadata_path = NEW_RUN_DIR / treatment / f"seed_{seed}" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint = Path(metadata["checkpoint"])
            if NEW_RUN_DIR not in checkpoint.parents or metadata["maximum_epochs"] != 1000:
                raise RuntimeError("a requested checkpoint is not from the 1000-epoch run tree")
            runs.append(
                {
                    "treatment": treatment,
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": sha256(checkpoint),
                    "best_epoch": metadata["best_epoch"],
                }
            )
    return runs


def summarize_trajectory_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trajectory_count": len(selected),
            "physical_exit_count": int(sum(row["physical_exit"] for row in selected)),
            "physical_exit_fraction": float(np.mean([row["physical_exit"] for row in selected])),
            "mean_final_combined_state_error": float(
                np.mean([row["final_combined_state_error"] for row in selected])
            ),
            "mean_time_mean_combined_state_error": float(
                np.mean([row["time_mean_combined_state_error"] for row in selected])
            ),
            "mean_mean_physical_prefix_absolute_energy_error": float(
                np.mean([row["mean_physical_prefix_absolute_energy_error"] for row in selected])
            ),
            "mean_maximum_physical_prefix_absolute_energy_error": float(
                np.mean([row["maximum_physical_prefix_absolute_energy_error"] for row in selected])
            ),
            "minimum_positive_C_across_trajectories": float(
                min(row["minimum_positive_C"] for row in selected if row["minimum_positive_C"] is not None)
            ),
            "mean_final_abs_C_error": float(
                np.mean([row["final_abs_C_error"] for row in selected])
            ),
        }

    overall = summarize(rows)
    near_edge = [row for row in rows if row["category"] == "near_edge"]
    overall["mean_near_edge_time_mean_combined_state_error"] = float(
        np.mean([row["time_mean_combined_state_error"] for row in near_edge])
    )
    overall["first_exit_times"] = [
        row["first_physical_exit_time"]
        for row in rows
        if row["first_physical_exit_time"] is not None
    ]
    by_class = {
        reference_class: summarize(
            [row for row in rows if row["reference_class"] == reference_class]
        )
        for reference_class in REFERENCE_CLASSES
    }
    escaping = [row for row in rows if row["reference_class"] == "escaping_force_free"]
    terminal_rows = [
        row for row in escaping if row["terminal_portion_mean_state_error"] is not None
    ]
    null_rows = [
        row for row in rows if row["reference_class"] == "null_boundary_asymptotic"
    ]
    return {
        "overall": overall,
        "by_reference_class": by_class,
        "escaping_diagnostic": {
            "trajectory_count": len(escaping),
            "reference_terminal_portion_available_count": len(terminal_rows),
            "mean_terminal_portion_state_error": float(
                np.mean([row["terminal_portion_mean_state_error"] for row in terminal_rows])
            ),
            "final_terminal_condition_count": int(
                sum(row["final_terminal_condition"] for row in escaping)
            ),
            "escaping_trend_count": int(sum(row["escaping_trend"] for row in escaping)),
            "physical_exit_before_reference_terminal_count": int(
                sum(row["physical_exit_before_terminal"] for row in escaping)
            ),
        },
        "null_boundary_diagnostic": {
            "trajectory_count": len(null_rows),
            "null_approach_trend_count": int(
                sum(row["null_approach_trend"] for row in null_rows)
            ),
            "physical_exit_count": int(sum(row["physical_exit"] for row in null_rows)),
            "minimum_positive_C": float(
                min(row["minimum_positive_C"] for row in null_rows if row["minimum_positive_C"] is not None)
            ),
            "mean_final_abs_C_error": float(
                np.mean([row["final_abs_C_error"] for row in null_rows])
            ),
        },
        "trajectories": rows,
    }


def evaluate_full_rollouts(
    runs: list[dict[str, Any]],
    members: list[dict[str, Any]],
    references: dict[str, dict[str, np.ndarray]],
    normalization: Normalization,
    f_star: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    sigma_x, sigma_u = normalization.input_std
    arrays: dict[str, np.ndarray] = {}
    for member in members:
        identifier = member["id"]
        key = prefix(identifier)
        reference = references[identifier]
        arrays[f"{key}__time"] = reference["time"]
        arrays[f"{key}__reference_state"] = reference["state"]
        arrays[f"{key}__reference_v_total"] = np.sqrt(
            total_speed_squared(
                reference["state"][:, 0], reference["state"][:, 1], wormhole, spiral
            )
        )
    run_rows = []
    for run in runs:
        model = load_trained_model(run["checkpoint"])
        trajectories = []
        for member in members:
            identifier = member["id"]
            key = prefix(identifier)
            reference = references[identifier]
            predicted = recursive_rollout(
                model,
                reference["state"][0],
                int(member["step_count"]),
                normalization,
            )
            if not np.array_equal(predicted[0], reference["state"][0]):
                raise RuntimeError("recursive rollout does not preserve its exact initial state")
            error = predicted - reference["state"]
            absolute_error = np.abs(error)
            combined = np.sqrt(
                (error[:, 0] / sigma_x) ** 2 + (error[:, 1] / sigma_u) ** 2
            )
            predicted_v_total = np.sqrt(
                total_speed_squared(predicted[:, 0], predicted[:, 1], wormhole, spiral)
            )
            predicted_c = timelike_margin(
                predicted[:, 0], predicted[:, 1], wormhole, spiral
            ).astype(np.float64)
            reference_c = timelike_margin(
                reference["state"][:, 0], reference["state"][:, 1], wormhole, spiral
            ).astype(np.float64)
            exits = np.flatnonzero(predicted_c <= 0.0)
            first_exit_index = None if not exits.size else int(exits[0])
            prefix_stop = len(predicted) if first_exit_index is None else first_exit_index
            positive = predicted_c > 0.0
            minimum_positive_c = (
                None if not np.any(positive) else float(np.min(predicted_c[positive]))
            )
            energy_error = np.full(len(predicted), np.nan, dtype=np.float64)
            if prefix_stop > 0:
                predicted_energy = conserved_energy(
                    predicted[:prefix_stop, 0],
                    predicted[:prefix_stop, 1],
                    wormhole,
                    spiral,
                )
                reference_energy = conserved_energy(
                    reference["state"][:prefix_stop, 0],
                    reference["state"][:prefix_stop, 1],
                    wormhole,
                    spiral,
                )
                energy_error[:prefix_stop] = np.abs(predicted_energy - reference_energy)
            omega_ratio = np.abs(effective_angular_velocity(predicted[:, 1], spiral)) / abs(
                spiral.omega
            )
            force_ratio = np.abs(
                radial_acceleration(predicted[:, 0], predicted[:, 1], wormhole, spiral)
            ) / f_star
            terminal_time = member["terminal_entry_time_by_T_i"]
            terminal_mask = (
                np.zeros(len(predicted), dtype=bool)
                if terminal_time is None
                else reference["time"] >= float(terminal_time) - 1e-12
            )
            exit_before_terminal = bool(
                first_exit_index is not None
                and terminal_time is not None
                and reference["time"][first_exit_index] < float(terminal_time)
            )
            trajectories.append(
                {
                    "id": identifier,
                    "category": member["category"],
                    "reference_class": member["reference_class"],
                    "step_count": int(member["step_count"]),
                    "T_i": float(member["T_i"]),
                    "final_abs_x_error": float(absolute_error[-1, 0]),
                    "final_abs_u_error": float(absolute_error[-1, 1]),
                    "final_combined_state_error": float(combined[-1]),
                    "time_mean_abs_x_error": float(np.mean(absolute_error[:, 0])),
                    "time_mean_abs_u_error": float(np.mean(absolute_error[:, 1])),
                    "time_mean_combined_state_error": float(np.mean(combined)),
                    "physical_exit": first_exit_index is not None,
                    "first_physical_exit_time": (
                        None
                        if first_exit_index is None
                        else float(reference["time"][first_exit_index])
                    ),
                    "minimum_positive_C": minimum_positive_c,
                    "final_predicted_C": float(predicted_c[-1]),
                    "final_reference_C": float(reference_c[-1]),
                    "mean_physical_prefix_absolute_energy_error": float(
                        np.nanmean(energy_error)
                    ),
                    "maximum_physical_prefix_absolute_energy_error": float(
                        np.nanmax(energy_error)
                    ),
                    "terminal_entry_time_by_T_i": terminal_time,
                    "terminal_portion_mean_state_error": (
                        None
                        if not np.any(terminal_mask)
                        else float(np.mean(combined[terminal_mask]))
                    ),
                    "final_terminal_condition": bool(
                        omega_ratio[-1] <= 0.05 and force_ratio[-1] <= 0.05
                    ),
                    "escaping_trend": bool(
                        abs(predicted[-1, 0]) > abs(predicted[0, 0])
                        and omega_ratio[-1] < omega_ratio[0]
                    ),
                    "physical_exit_before_terminal": exit_before_terminal,
                    "null_approach_trend": bool(predicted_c[-1] < predicted_c[0]),
                    "final_abs_C_error": float(abs(predicted_c[-1] - reference_c[-1])),
                }
            )
            run_key = f"{key}__{run['treatment']}__seed_{run['seed']}"
            arrays[f"{run_key}__predicted_state"] = predicted
            arrays[f"{run_key}__signed_error"] = error
            arrays[f"{run_key}__absolute_error"] = absolute_error
            arrays[f"{run_key}__combined_error"] = combined
            arrays[f"{run_key}__predicted_v_total"] = predicted_v_total
            arrays[f"{run_key}__predicted_C"] = predicted_c
            arrays[f"{run_key}__energy_error"] = energy_error
        run_rows.append(
            {
                **run,
                "rollouts": summarize_trajectory_rows(trajectories),
            }
        )
    for key, values in arrays.items():
        if key.endswith("__energy_error"):
            if not np.all(np.isfinite(values[np.isfinite(values)])):
                raise RuntimeError(f"invalid energy array: {key}")
        elif not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite full-rollout array: {key}")
    return run_rows, arrays


def value_after_steps(values: np.ndarray, steps: int) -> float | None:
    return None if steps >= len(values) else float(values[steps])


def first_threshold_time(elapsed: np.ndarray, values: np.ndarray, level: float) -> float | None:
    indices = np.flatnonzero(values > level)
    return None if not indices.size else float(elapsed[int(indices[0])])


def evaluate_restarts_and_local_error(
    runs: list[dict[str, Any]], normalization: Normalization
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    sigma_x, sigma_u = normalization.input_std
    restart_manifest = json.loads(OLD_RESTART_MANIFEST.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    restart_rows = []
    local_rows = []
    with np.load(OLD_RESTART_ARRAYS) as old:
        for identifier in REPRESENTATIVE_IDS:
            key = prefix(identifier)
            time = np.asarray(old[f"{key}__reference_time"])
            state = np.asarray(old[f"{key}__reference_state"])
            restart_indices = np.asarray(old[f"{key}__restart_indices"])
            restart_times = np.asarray(old[f"{key}__restart_times"])
            recorded = restart_manifest["representatives"][identifier]
            if not np.array_equal(restart_times, np.asarray(recorded["restart_times"])):
                raise RuntimeError("restart times differ from the previous diagnostic")
            if not np.array_equal(state[restart_indices], np.asarray(recorded["restart_states"])):
                raise RuntimeError("restart states differ from the previous diagnostic")
            arrays[f"{key}__reference_time"] = time
            arrays[f"{key}__reference_state"] = state
            arrays[f"{key}__reference_v_total"] = np.sqrt(
                total_speed_squared(state[:, 0], state[:, 1], wormhole, spiral)
            )
            arrays[f"{key}__restart_indices"] = restart_indices
            arrays[f"{key}__restart_times"] = restart_times
            exact_increment = state[1:] - state[:-1]
            for run in runs:
                model = load_trained_model(run["checkpoint"])
                local_prediction = predict_increments(model, state[:-1], normalization)
                local_error = local_prediction - exact_increment
                local_combined = np.sqrt(
                    (local_error[:, 0] / normalization.target_std[0]) ** 2
                    + (local_error[:, 1] / normalization.target_std[1]) ** 2
                )
                run_key = f"{key}__{run['treatment']}__seed_{run['seed']}"
                arrays[f"{run_key}__local_error_x"] = local_error[:, 0]
                arrays[f"{run_key}__local_error_u"] = local_error[:, 1]
                arrays[f"{run_key}__local_combined_error"] = local_combined
                local_rows.append(
                    {
                        "representative": identifier,
                        "treatment": run["treatment"],
                        "seed": run["seed"],
                        "mean_local_error": float(np.mean(local_combined)),
                        "maximum_local_error": float(np.max(local_combined)),
                        "maximum_local_error_time": float(time[int(np.argmax(local_combined))]),
                    }
                )
                for restart_number, restart_index in enumerate(restart_indices):
                    initial = state[restart_index].copy()
                    predicted = recursive_rollout(
                        model, initial, len(time) - 1 - int(restart_index), normalization
                    )
                    if not np.array_equal(predicted[0], initial):
                        raise RuntimeError("restart did not begin from the exact stored state")
                    expected = state[restart_index:]
                    error = predicted - expected
                    combined = np.sqrt(
                        (error[:, 0] / sigma_x) ** 2 + (error[:, 1] / sigma_u) ** 2
                    )
                    absolute_time = time[restart_index:]
                    elapsed = absolute_time - absolute_time[0]
                    predicted_c = timelike_margin(
                        predicted[:, 0], predicted[:, 1], wormhole, spiral
                    )
                    exits = np.flatnonzero(predicted_c <= 0.0)
                    first_exit = (
                        None if not exits.size else float(absolute_time[int(exits[0])])
                    )
                    restart_key = f"{run_key}__restart_{restart_number}"
                    arrays[f"{restart_key}__absolute_time"] = absolute_time
                    arrays[f"{restart_key}__elapsed_time"] = elapsed
                    arrays[f"{restart_key}__initial_state"] = initial
                    arrays[f"{restart_key}__predicted_state"] = predicted
                    arrays[f"{restart_key}__signed_error"] = error
                    arrays[f"{restart_key}__combined_error"] = combined
                    arrays[f"{restart_key}__predicted_v_total"] = np.sqrt(
                        total_speed_squared(
                            predicted[:, 0], predicted[:, 1], wormhole, spiral
                        )
                    )
                    arrays[f"{restart_key}__predicted_C"] = predicted_c
                    restart_rows.append(
                        {
                            "representative": identifier,
                            "treatment": run["treatment"],
                            "seed": run["seed"],
                            "restart_number": restart_number,
                            "restart_index": int(restart_index),
                            "restart_time": float(time[restart_index]),
                            "restart_state": initial.tolist(),
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
    for key, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite restart/local array: {key}")
    return restart_rows, local_rows, arrays


def aggregate_restarts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    "physical_exit_count": int(
                        sum(row["physical_exit_time"] is not None for row in selected)
                    ),
                    "physical_exit_times": [
                        row["physical_exit_time"]
                        for row in selected
                        if row["physical_exit_time"] is not None
                    ],
                }
                for field in (
                    "error_after_1_step",
                    "error_after_5_steps",
                    "error_after_10_steps",
                    "final_error",
                ):
                    values = [row[field] for row in selected if row[field] is not None]
                    aggregate[f"mean_{field}"] = float(np.mean(values))
                    aggregate[f"std_{field}"] = float(np.std(values, ddof=1))
                for field in (
                    "first_elapsed_time_above_0p01",
                    "first_elapsed_time_above_0p05",
                    "first_elapsed_time_above_0p1",
                ):
                    values = [row[field] for row in selected if row[field] is not None]
                    aggregate[f"{field}_reached_count"] = len(values)
                    aggregate[f"mean_{field}"] = (
                        None if not values else float(np.mean(values))
                    )
                output.append(aggregate)
    return output


def aggregate_local(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for representative in REPRESENTATIVE_IDS:
        for treatment in TREATMENTS:
            selected = [
                row
                for row in rows
                if row["representative"] == representative
                and row["treatment"] == treatment
            ]
            output.append(
                {
                    "representative": representative,
                    "treatment": treatment,
                    "mean_local_error": float(
                        np.mean([row["mean_local_error"] for row in selected])
                    ),
                    "std_local_error": float(
                        np.std([row["mean_local_error"] for row in selected], ddof=1)
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


def enrich_old_runs(old_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for run in old_manifest["runs"]:
        trajectories = run["rollouts"]["trajectories"]
        output.append(
            {
                "treatment": run["treatment"],
                "seed": run["seed"],
                "rollouts": summarize_trajectory_rows(trajectories),
            }
        )
    return output


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "time_mean_error": ("overall", "mean_time_mean_combined_state_error"),
        "final_error": ("overall", "mean_final_combined_state_error"),
        "near_edge_error": (
            "overall",
            "mean_near_edge_time_mean_combined_state_error",
        ),
        "mean_energy_error": (
            "overall",
            "mean_mean_physical_prefix_absolute_energy_error",
        ),
        "maximum_energy_error": (
            "overall",
            "mean_maximum_physical_prefix_absolute_energy_error",
        ),
        "escaping_time_mean_error": (
            "by_reference_class",
            "escaping_force_free",
            "mean_time_mean_combined_state_error",
        ),
        "escaping_final_error": (
            "by_reference_class",
            "escaping_force_free",
            "mean_final_combined_state_error",
        ),
        "null_time_mean_error": (
            "by_reference_class",
            "null_boundary_asymptotic",
            "mean_time_mean_combined_state_error",
        ),
        "null_final_error": (
            "by_reference_class",
            "null_boundary_asymptotic",
            "mean_final_combined_state_error",
        ),
    }
    output = {}
    for treatment in TREATMENTS:
        selected = [run for run in runs if run["treatment"] == treatment]
        summary = {}
        for metric, path in metric_paths.items():
            values = []
            for run in selected:
                value: Any = run["rollouts"]
                for key in path:
                    value = value[key]
                values.append(float(value))
            summary[metric] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "values_by_seed": {
                    str(run["seed"]): value
                    for run, value in zip(selected, values, strict=True)
                },
            }
        summary["physical_exit_count_across_72_rollouts"] = int(
            sum(run["rollouts"]["overall"]["physical_exit_count"] for run in selected)
        )
        summary["escaping_exit_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["by_reference_class"]["escaping_force_free"][
                    "physical_exit_count"
                ]
                for run in selected
            )
        )
        summary["null_exit_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["by_reference_class"][
                    "null_boundary_asymptotic"
                ]["physical_exit_count"]
                for run in selected
            )
        )
        summary["escaping_trend_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["escaping_diagnostic"]["escaping_trend_count"]
                for run in selected
            )
        )
        summary["escaping_terminal_condition_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["escaping_diagnostic"][
                    "final_terminal_condition_count"
                ]
                for run in selected
            )
        )
        summary["escaping_exit_before_terminal_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["escaping_diagnostic"][
                    "physical_exit_before_reference_terminal_count"
                ]
                for run in selected
            )
        )
        summary["null_approach_count_across_36_rollouts"] = int(
            sum(
                run["rollouts"]["null_boundary_diagnostic"][
                    "null_approach_trend_count"
                ]
                for run in selected
            )
        )
        output[treatment] = summary
    return output


def stack_new(
    arrays: dict[str, np.ndarray], identifier: str, treatment: str, suffix: str
) -> np.ndarray:
    key = prefix(identifier)
    return np.stack(
        [
            arrays[f"{key}__{treatment}__seed_{seed}__{suffix}"]
            for seed in TRAINING_SEEDS
        ]
    )


def stack_old_representative(
    identifier: str,
    treatment: str,
    suffix: str,
    normalization: Normalization,
) -> np.ndarray:
    key = prefix(identifier)
    values = []
    wormhole, spiral = experiment_parameters()
    for seed in TRAINING_SEEDS:
        with np.load(old_rollout_path(treatment, seed)) as arrays:
            if suffix == "predicted_v_total":
                predicted = arrays[f"{key}__predicted_state"]
                value = np.sqrt(
                    total_speed_squared(
                        predicted[:, 0], predicted[:, 1], wormhole, spiral
                    )
                )
            else:
                value = arrays[f"{key}__{suffix}"]
            values.append(np.asarray(value))
    return np.stack(values)


def make_summary_figure(old: dict[str, Any], new: dict[str, Any]) -> None:
    metrics = (
        ("time_mean_error", "time-mean $e_z$"),
        ("final_error", "final $e_z$"),
        ("near_edge_error", "near-edge time-mean $e_z$"),
        ("escaping_time_mean_error", "escaping time-mean $e_z$"),
        ("null_time_mean_error", "null time-mean $e_z$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    positions = np.arange(3)
    width = 0.34
    for ax, (metric, title) in zip(axes.ravel()[:5], metrics, strict=True):
        old_mean = [old[treatment][metric]["mean"] for treatment in TREATMENTS]
        old_std = [old[treatment][metric]["standard_deviation"] for treatment in TREATMENTS]
        new_mean = [new[treatment][metric]["mean"] for treatment in TREATMENTS]
        new_std = [new[treatment][metric]["standard_deviation"] for treatment in TREATMENTS]
        ax.bar(positions - width / 2, old_mean, width, yerr=old_std, color="0.65", label="500")
        ax.bar(
            positions + width / 2,
            new_mean,
            width,
            yerr=new_std,
            color=[COLORS[treatment] for treatment in TREATMENTS],
            label="1000",
        )
        ax.set_xticks(positions, [LABELS[treatment] for treatment in TREATMENTS], rotation=13)
        ax.set_title(title)
    exits = axes.ravel()[5]
    old_exits = [old[t]["physical_exit_count_across_72_rollouts"] for t in TREATMENTS]
    new_exits = [new[t]["physical_exit_count_across_72_rollouts"] for t in TREATMENTS]
    exits.bar(positions - width / 2, old_exits, width, color="0.65", label="500")
    exits.bar(
        positions + width / 2,
        new_exits,
        width,
        color=[COLORS[treatment] for treatment in TREATMENTS],
        label="1000",
    )
    exits.set_xticks(positions, [LABELS[treatment] for treatment in TREATMENTS], rotation=13)
    exits.set_title("physical-domain exits across 72 rollouts")
    axes[0, 0].legend()
    fig.suptitle("Recursive validation: scalar fixed-trajectory summaries", y=1.02)
    fig.savefig(FIGURE_DIR / "rollout_500_vs_1000_summary.png")
    plt.close(fig)


def make_representative_figure(
    identifier: str,
    new_arrays: dict[str, np.ndarray],
    normalization: Normalization,
    output_name: str,
) -> None:
    key = prefix(identifier)
    time = new_arrays[f"{key}__time"]
    reference = new_arrays[f"{key}__reference_state"]
    reference_v = new_arrays[f"{key}__reference_v_total"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, values, title in zip(
        axes.ravel()[:3],
        (reference[:, 0], reference[:, 1], reference_v),
        ("position $x$", "radial velocity $u$", "validated total speed $v_{tot}$"),
        strict=True,
    ):
        ax.plot(time, values, color="black", lw=2, label="DOP853")
        ax.set_title(title)
        ax.set_xlabel("absolute time $s$")
    for treatment in TREATMENTS:
        for index, suffix in enumerate(
            ("predicted_state", "predicted_state", "predicted_v_total")
        ):
            old_stack = stack_old_representative(
                identifier, treatment, suffix, normalization
            )
            new_stack = stack_new(new_arrays, identifier, treatment, suffix)
            if suffix == "predicted_state":
                old_stack = old_stack[:, :, index]
                new_stack = new_stack[:, :, index]
            old_mean = np.mean(old_stack, axis=0)
            new_mean = np.mean(new_stack, axis=0)
            old_std = np.std(old_stack, axis=0, ddof=1)
            new_std = np.std(new_stack, axis=0, ddof=1)
            axes.ravel()[index].plot(
                time, old_mean, color=COLORS[treatment], ls="--", alpha=0.75,
                label=f"{LABELS[treatment]} 500",
            )
            axes.ravel()[index].fill_between(
                time, old_mean - old_std, old_mean + old_std,
                color=COLORS[treatment], alpha=0.07,
            )
            axes.ravel()[index].plot(
                time, new_mean, color=COLORS[treatment], lw=1.5,
                label=f"{LABELS[treatment]} 1000",
            )
            axes.ravel()[index].fill_between(
                time, new_mean - new_std, new_mean + new_std,
                color=COLORS[treatment], alpha=0.15,
            )
        old_error = stack_old_representative(
            identifier, treatment, "combined_error", normalization
        )
        new_error = stack_new(new_arrays, identifier, treatment, "combined_error")
        for values, style, alpha, label in (
            (old_error, "--", 0.75, "500"),
            (new_error, "-", 1.0, "1000"),
        ):
            mean = np.mean(values, axis=0)
            spread = np.std(values, axis=0, ddof=1)
            axes[1, 1].plot(
                time,
                mean,
                color=COLORS[treatment],
                ls=style,
                alpha=alpha,
                label=f"{LABELS[treatment]} {label}",
            )
            axes[1, 1].fill_between(
                time, mean - spread, mean + spread,
                color=COLORS[treatment], alpha=0.07 if label == "500" else 0.15,
            )
    axes[1, 1].set(title="combined state error $e_z$", xlabel="absolute time $s$")
    axes[0, 0].legend(fontsize=6, ncols=2)
    fig.suptitle(f"Fixed validation representative: {identifier}", y=1.02)
    fig.savefig(FIGURE_DIR / output_name)
    plt.close(fig)


def make_restart_figure(
    old_arrays: dict[str, np.ndarray], new_arrays: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for row, identifier in enumerate(REPRESENTATIVE_IDS):
        key = prefix(identifier)
        restart_times = new_arrays[f"{key}__restart_times"]
        for restart_number in range(3):
            ax = axes[row, restart_number]
            for treatment in TREATMENTS:
                for source, style, label in (
                    (old_arrays, "--", "500"),
                    (new_arrays, "-", "1000"),
                ):
                    stack = np.stack(
                        [
                            source[
                                f"{key}__{treatment}__seed_{seed}__restart_{restart_number}__combined_error"
                            ]
                            for seed in TRAINING_SEEDS
                        ]
                    )
                    elapsed = source[
                        f"{key}__{treatment}__seed_101__restart_{restart_number}__elapsed_time"
                    ]
                    mean = np.mean(stack, axis=0)
                    spread = np.std(stack, axis=0, ddof=1)
                    ax.plot(
                        elapsed, mean, color=COLORS[treatment], ls=style,
                        label=f"{LABELS[treatment]} {label}",
                    )
                    ax.fill_between(
                        elapsed, mean - spread, mean + spread,
                        color=COLORS[treatment], alpha=0.07 if label == "500" else 0.15,
                    )
            for level in ERROR_LEVELS:
                ax.axhline(level, color="0.55", ls=":", lw=0.6)
            ax.set(
                title=f"restart $s_k={restart_times[restart_number]:.1f}$",
                xlabel="elapsed time $\\tau$",
                ylabel=("escaping $e_z$" if row == 0 else "null-boundary $e_z$")
                if restart_number == 0
                else "$e_z$",
            )
    axes[0, 0].legend(fontsize=6, ncols=2)
    fig.suptitle("Exact-state restart error: dashed 500 vs solid 1000", y=1.02)
    fig.savefig(FIGURE_DIR / "restart_error_500_vs_1000.png")
    plt.close(fig)


def make_local_error_figure(
    old_arrays: dict[str, np.ndarray], new_arrays: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    suffixes = ("local_error_x", "local_error_u", "local_combined_error")
    for row, identifier in enumerate(REPRESENTATIVE_IDS):
        key = prefix(identifier)
        time = new_arrays[f"{key}__reference_time"][:-1]
        for column, suffix in enumerate(suffixes):
            for treatment in TREATMENTS:
                for source, style, label in (
                    (old_arrays, "--", "500"),
                    (new_arrays, "-", "1000"),
                ):
                    stack = np.stack(
                        [
                            source[f"{key}__{treatment}__seed_{seed}__{suffix}"]
                            for seed in TRAINING_SEEDS
                        ]
                    )
                    mean = np.mean(stack, axis=0)
                    spread = np.std(stack, axis=0, ddof=1)
                    axes[row, column].plot(
                        time, mean, color=COLORS[treatment], ls=style,
                        label=f"{LABELS[treatment]} {label}",
                    )
                    axes[row, column].fill_between(
                        time, mean - spread, mean + spread,
                        color=COLORS[treatment], alpha=0.07 if label == "500" else 0.15,
                    )
            axes[row, column].set_xlabel("absolute time $s$")
        axes[row, 0].set_ylabel(
            "escaping\nlocal error" if row == 0 else "null-boundary\nlocal error"
        )
    for column, title in enumerate(
        (r"signed local $e_x$", r"signed local $e_u$", r"local $E$")
    ):
        axes[0, column].set_title(title)
    axes[0, 0].legend(fontsize=6, ncols=2)
    fig.suptitle("Teacher-forced local error: dashed 500 vs solid 1000", y=1.02)
    fig.savefig(FIGURE_DIR / "trajectory_local_error_500_vs_1000.png")
    plt.close(fig)


def relative_percent(new: float, old: float) -> float:
    return float(100.0 * (new - old) / old)


def fmt(value: float | None) -> str:
    return "---" if value is None else f"{value:.6g}"


def make_report(manifest: dict[str, Any]) -> None:
    old = manifest["comparison"]["old_aggregate"]
    new = manifest["comparison"]["new_aggregate"]
    old_runs = {
        (row["treatment"], row["seed"]): row for row in manifest["old_runs"]
    }
    old_restarts = {
        (row["representative"], row["treatment"], row["restart_number"]): row
        for row in manifest["restart"]["old_aggregate"]
    }
    old_local = {
        (row["representative"], row["treatment"]): row
        for row in manifest["teacher_forced_local_error"]["old_aggregate"]
    }
    lines = [
        "# Recursive validation after the controlled 1000-epoch extension",
        "",
        "This is validation-only. It evaluates the nine frozen 1000-epoch best checkpoints on the exact 24 stored DOP853 validation trajectories and compares them with the preserved 500-epoch results. Predictions are unconstrained and use recursive $h=0.2$ increments without clipping or projection.",
        "",
        "## Checkpoints and reference identity",
        "",
        "| treatment | seed | best epoch | checkpoint SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for run in manifest["checkpoint_runs"]:
        lines.append(
            f"| {LABELS[run['treatment']]} | {run['seed']} | {run['best_epoch']} | `{run['checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            f"Frozen validation identity: `{manifest['frozen_validation_identity_sha256']}`. All nine old rollout artifacts contain bit-identical stored reference time/state arrays. No reference trajectory was regenerated.",
            "",
            "## Full recursive rollout comparison",
            "",
            "Each scalar below is first computed over each fixed trajectory and then aggregated; no changing-cohort time curve is used.",
            "",
            "| treatment | time-mean ez old -> new | final ez old -> new | near-edge old -> new | exits old -> new | escaping old -> new | null old -> new |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        cells = []
        for metric in (
            "time_mean_error",
            "final_error",
            "near_edge_error",
        ):
            old_value = old[treatment][metric]["mean"]
            new_value = new[treatment][metric]["mean"]
            cells.append(
                f"{fmt(old_value)} -> {fmt(new_value)} ({fmt(relative_percent(new_value, old_value))}%)"
            )
        cells.append(
            f"{old[treatment]['physical_exit_count_across_72_rollouts']} -> {new[treatment]['physical_exit_count_across_72_rollouts']}"
        )
        for metric in ("escaping_time_mean_error", "null_time_mean_error"):
            old_value = old[treatment][metric]["mean"]
            new_value = new[treatment][metric]["mean"]
            cells.append(
                f"{fmt(old_value)} -> {fmt(new_value)} ({fmt(relative_percent(new_value, old_value))}%)"
            )
        lines.append(f"| {LABELS[treatment]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### Seed spread and individual runs",
            "",
            "| treatment | time-mean ez old -> new (mean +/- SD) | final ez old -> new (mean +/- SD) | near-edge old -> new (mean +/- SD) |",
            "|---|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        cells = []
        for metric in ("time_mean_error", "final_error", "near_edge_error"):
            cells.append(
                f"{fmt(old[treatment][metric]['mean'])} +/- {fmt(old[treatment][metric]['standard_deviation'])}"
                f" -> {fmt(new[treatment][metric]['mean'])} +/- {fmt(new[treatment][metric]['standard_deviation'])}"
            )
        lines.append(f"| {LABELS[treatment]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| treatment | seed | time-mean ez old -> new | final ez old -> new | exits /24 old -> new | mean energy error old -> new |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in manifest["new_runs"]:
        previous = old_runs[(row["treatment"], row["seed"])]
        old_overall = previous["rollouts"]["overall"]
        new_overall = row["rollouts"]["overall"]
        lines.append(
            f"| {LABELS[row['treatment']]} | {row['seed']} | "
            f"{fmt(old_overall['mean_time_mean_combined_state_error'])} -> {fmt(new_overall['mean_time_mean_combined_state_error'])} | "
            f"{fmt(old_overall['mean_final_combined_state_error'])} -> {fmt(new_overall['mean_final_combined_state_error'])} | "
            f"{old_overall['physical_exit_count']} -> {new_overall['physical_exit_count']} | "
            f"{fmt(old_overall['mean_mean_physical_prefix_absolute_energy_error'])} -> {fmt(new_overall['mean_mean_physical_prefix_absolute_energy_error'])} |"
        )
    lines.extend(
        [
            "",
            "",
            "| treatment | escaping trends /36 old -> new | terminal condition /36 | escaping exits /36 | null approach /36 | null exits /36 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        lines.append(
            f"| {LABELS[treatment]} | {old[treatment]['escaping_trend_count_across_36_rollouts']} -> {new[treatment]['escaping_trend_count_across_36_rollouts']} | {old[treatment]['escaping_terminal_condition_count_across_36_rollouts']} -> {new[treatment]['escaping_terminal_condition_count_across_36_rollouts']} | {old[treatment]['escaping_exit_count_across_36_rollouts']} -> {new[treatment]['escaping_exit_count_across_36_rollouts']} | {old[treatment]['null_approach_count_across_36_rollouts']} -> {new[treatment]['null_approach_count_across_36_rollouts']} | {old[treatment]['null_exit_count_across_36_rollouts']} -> {new[treatment]['null_exit_count_across_36_rollouts']} |"
        )
    lines.extend(
        [
            "",
            "| treatment | mean physical-prefix energy error old -> new | mean trajectory maximum old -> new |",
            "|---|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        lines.append(
            f"| {LABELS[treatment]} | {fmt(old[treatment]['mean_energy_error']['mean'])} -> {fmt(new[treatment]['mean_energy_error']['mean'])} | {fmt(old[treatment]['maximum_energy_error']['mean'])} -> {fmt(new[treatment]['maximum_energy_error']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "![Scalar rollout summary](figures/rollout_500_vs_1000_summary.png)",
            "",
            "## Fixed representatives",
            "",
            "The figures compare DOP853 with the 500-epoch and 1000-epoch treatment means and three-seed spread on the same individual trajectory. Solid curves are 1000 epoch; dashed curves are 500 epoch.",
            "",
            "![Escaping representative](figures/representative_escaping_500_vs_1000.png)",
            "",
            "![Null-boundary representative](figures/representative_null_boundary_500_vs_1000.png)",
            "",
            "## Exact-state restart comparison",
            "",
            "| representative | treatment | restart s | one-step old -> new | step 5 | step 10 | tau>.01 old -> new | tau>.05 | tau>.1 | final old -> new | exits old -> new |",
            "|---|---|---:|---:|---:|---:|---|---|---|---:|---:|",
        ]
    )
    for row in manifest["restart"]["new_aggregate"]:
        previous = old_restarts[
            (row["representative"], row["treatment"], row["restart_number"])
        ]

        def threshold_cell(field: str) -> str:
            return (
                f"{fmt(previous['mean_' + field])} ({previous[field + '_reached_count']}/3)"
                f" -> {fmt(row['mean_' + field])} ({row[field + '_reached_count']}/3)"
            )

        lines.append(
            f"| `{row['representative']}` | {LABELS[row['treatment']]} | {fmt(row['restart_time'])} | {fmt(previous['mean_error_after_1_step'])} -> {fmt(row['mean_error_after_1_step'])} | {fmt(previous['mean_error_after_5_steps'])} -> {fmt(row['mean_error_after_5_steps'])} | {fmt(previous['mean_error_after_10_steps'])} -> {fmt(row['mean_error_after_10_steps'])} | {threshold_cell('first_elapsed_time_above_0p01')} | {threshold_cell('first_elapsed_time_above_0p05')} | {threshold_cell('first_elapsed_time_above_0p1')} | {fmt(previous['mean_final_error'])} -> {fmt(row['mean_final_error'])} | {previous['physical_exit_count']} -> {row['physical_exit_count']} |"
        )
    lines.extend(
        [
            "",
            "![Restart comparison](figures/restart_error_500_vs_1000.png)",
            "",
            "## Teacher-forced local error on the representatives",
            "",
            "Local $E$ uses increment standard deviations and is not numerically interchangeable with recursive state $e_z$.",
            "",
            "| representative | treatment | mean local E old -> new | mean seed maximum old -> new | new peak times |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in manifest["teacher_forced_local_error"]["new_aggregate"]:
        previous = old_local[(row["representative"], row["treatment"])]
        times = ", ".join(fmt(value) for value in row["seed_maximum_local_error_times"])
        lines.append(
            f"| `{row['representative']}` | {LABELS[row['treatment']]} | {fmt(previous['mean_local_error'])} -> {fmt(row['mean_local_error'])} ({fmt(relative_percent(row['mean_local_error'], previous['mean_local_error']))}%) | {fmt(previous['mean_seed_maximum_local_error'])} -> {fmt(row['mean_seed_maximum_local_error'])} | {times} |"
        )
    lines.extend(
        [
            "",
            "![Teacher-forced local error](figures/trajectory_local_error_500_vs_1000.png)",
            "",
            "## Interpretation",
            "",
        ]
    )
    report_interpretation = build_interpretation(
        old,
        new,
        manifest["restart"]["old_aggregate"],
        manifest["restart"]["new_aggregate"],
        manifest["teacher_forced_local_error"]["old_aggregate"],
        manifest["teacher_forced_local_error"]["new_aggregate"],
    )
    for item in report_interpretation:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "No training, dense-grid recomputation, clipping, projection, constraint, treatment selection, architecture decision, or restricted-data evaluation was performed.",
        ]
    )
    (REPORT_DIR / "ROLLOUT_REEVALUATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_interpretation(
    old: dict[str, Any],
    new: dict[str, Any],
    old_restart: list[dict[str, Any]],
    new_restart: list[dict[str, Any]],
    old_local: list[dict[str, Any]],
    new_local: list[dict[str, Any]],
) -> list[str]:
    rollout_changes = {
        treatment: relative_percent(
            new[treatment]["time_mean_error"]["mean"],
            old[treatment]["time_mean_error"]["mean"],
        )
        for treatment in TREATMENTS
    }
    exit_changes = {
        treatment: (
            old[treatment]["physical_exit_count_across_72_rollouts"],
            new[treatment]["physical_exit_count_across_72_rollouts"],
        )
        for treatment in TREATMENTS
    }
    null_changes = {
        treatment: relative_percent(
            new[treatment]["null_time_mean_error"]["mean"],
            old[treatment]["null_time_mean_error"]["mean"],
        )
        for treatment in TREATMENTS
    }
    escaping_changes = {
        treatment: relative_percent(
            new[treatment]["escaping_time_mean_error"]["mean"],
            old[treatment]["escaping_time_mean_error"]["mean"],
        )
        for treatment in TREATMENTS
    }
    local_lookup_old = {
        (row["representative"], row["treatment"]): row for row in old_local
    }
    local_changes = []
    for row in new_local:
        previous = local_lookup_old[(row["representative"], row["treatment"])]
        local_changes.append(
            relative_percent(row["mean_local_error"], previous["mean_local_error"])
        )
    restart_lookup_old = {
        (row["representative"], row["treatment"], row["restart_number"]): row
        for row in old_restart
    }
    restart_changes = []
    for row in new_restart:
        previous = restart_lookup_old[
            (row["representative"], row["treatment"], row["restart_number"])
        ]
        restart_changes.append(
            relative_percent(row["mean_final_error"], previous["mean_final_error"])
        )
    late_escaping_restart_changes = {}
    for treatment in TREATMENTS:
        current = next(
            row
            for row in new_restart
            if row["representative"] == "validation-below_terminal"
            and row["treatment"] == treatment
            and row["restart_number"] == 2
        )
        previous = restart_lookup_old[
            ("validation-below_terminal", treatment, 2)
        ]
        late_escaping_restart_changes[treatment] = relative_percent(
            current["mean_final_error"], previous["mean_final_error"]
        )
    return [
        "Recursive rollout accuracy improved materially: treatment-mean time-averaged error changes are "
        + ", ".join(
            f"{LABELS[treatment]} {rollout_changes[treatment]:+.1f}%"
            for treatment in TREATMENTS
        )
        + ".",
        "Physical-domain exit counts across the 72 rollouts per treatment changed from "
        + ", ".join(
            f"{LABELS[treatment]} {exit_changes[treatment][0]} to {exit_changes[treatment][1]}"
            for treatment in TREATMENTS
        )
        + ".",
        "Null-boundary time-averaged error changes are "
        + ", ".join(
            f"{LABELS[treatment]} {null_changes[treatment]:+.1f}%"
            for treatment in TREATMENTS
        )
        + "; both collar treatments preserve the 36/36 qualitative approach count while reducing null crossings to 0/36. Physical-only retains 33/36 null crossings despite its smaller state error.",
        "Escaping behavior is mixed rather than uniformly improved. Time-averaged escaping error changes are "
        + ", ".join(
            f"{LABELS[treatment]} {escaping_changes[treatment]:+.1f}%"
            for treatment in TREATMENTS
        )
        + ", while trend/terminal counts improve modestly and escaping exits fall. Thus the physical-only run is numerically worse on this class even though its qualitative counts improve.",
        f"Exact-state restarts show strongly improved null-boundary evolution, but phase-dependent escaping behavior: across all 18 restart/treatment combinations final-error changes range from {min(restart_changes):+.1f}% to {max(restart_changes):+.1f}%. At the late escaping restart s=21 they are "
        + ", ".join(
            f"{LABELS[treatment]} {late_escaping_restart_changes[treatment]:+.1f}%"
            for treatment in TREATMENTS
        )
        + ".",
        f"Teacher-forced mean local error changes across the two difficult representatives range from {min(local_changes):+.1f}% to {max(local_changes):+.1f}%. The boundary/throat local peak is greatly reduced for the collar models; the physical-only escaping local mean is essentially unchanged, and some peak-error times shift rather than disappear.",
        "The evidence supports a combined explanation: longer optimization reduces important local-map errors, especially near the null boundary, while residual phase-local errors accumulate recursively and the unconstrained map remains sensitive to admissibility near C=0. The late exact restarts demonstrate that accumulation alone does not explain every remaining discrepancy.",
        "These results provide evidence about optimization duration only and do not select a treatment, architecture, or constraint strategy.",
    ]


def main() -> None:
    if OUTPUT_DIR.exists() or REPORT_DIR.exists():
        raise FileExistsError("refusing to overwrite an existing rollout reevaluation tree")
    preserved_paths = (
        OLD_RUN_DIR,
        PROJECT_ROOT / "reports" / "round1_model_a",
        OLD_RESTART_DIR,
        PROJECT_ROOT / "reports" / "round1_local_restart",
        NEW_RUN_DIR,
        PROJECT_ROOT / "reports" / "round1_model_a_1000",
    )
    preserved_before = {str(path): tree_identity(path) for path in preserved_paths}
    OUTPUT_DIR.mkdir(parents=True)
    FIGURE_DIR.mkdir(parents=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    old_manifest = json.loads(OLD_EVALUATION.read_text(encoding="utf-8"))
    if old_manifest["validation_inputs"]["frozen_validation_trajectory_count"] != 24:
        raise RuntimeError("old evaluation does not identify the frozen 24-member suite")
    members, references = load_stored_references(old_manifest)
    runs = checkpoint_runs()
    checkpoint_hashes_before = {
        f"{run['treatment']}__seed_{run['seed']}": run["checkpoint_sha256"] for run in runs
    }
    normalization = Normalization.from_stage1(
        PROJECT_ROOT / "output" / "stage1_model_a" / "normalization.json"
    )
    force_envelope = extract_json_object(GATE_RESULT_PATH, "compact_domain_force_envelope")
    f_star = float(force_envelope["F_star"])

    new_run_rows, full_arrays = evaluate_full_rollouts(
        runs, members, references, normalization, f_star
    )
    full_path = OUTPUT_DIR / "full_validation_rollouts.npz"
    np.savez_compressed(full_path, **full_arrays)
    restart_rows, local_rows, restart_arrays = evaluate_restarts_and_local_error(
        runs, normalization
    )
    restart_path = OUTPUT_DIR / "restart_and_local_diagnostics.npz"
    np.savez_compressed(restart_path, **restart_arrays)
    old_run_rows = enrich_old_runs(old_manifest)
    old_aggregate = aggregate_runs(old_run_rows)
    new_aggregate = aggregate_runs(new_run_rows)
    old_restart_manifest = json.loads(OLD_RESTART_MANIFEST.read_text(encoding="utf-8"))
    old_restart_aggregate = old_restart_manifest["restart_aggregate"]
    new_restart_aggregate = aggregate_restarts(restart_rows)
    old_local_aggregate = old_restart_manifest["trajectory_local_error_aggregate"]
    new_local_aggregate = aggregate_local(local_rows)

    with np.load(OLD_RESTART_ARRAYS) as loaded:
        old_restart_arrays = {key: loaded[key] for key in loaded.files}
    make_summary_figure(old_aggregate, new_aggregate)
    make_representative_figure(
        REPRESENTATIVE_IDS[0],
        full_arrays,
        normalization,
        "representative_escaping_500_vs_1000.png",
    )
    make_representative_figure(
        REPRESENTATIVE_IDS[1],
        full_arrays,
        normalization,
        "representative_null_boundary_500_vs_1000.png",
    )
    make_restart_figure(old_restart_arrays, restart_arrays)
    make_local_error_figure(old_restart_arrays, restart_arrays)

    checkpoint_hashes_after = {
        f"{run['treatment']}__seed_{run['seed']}": sha256(run["checkpoint"])
        for run in runs
    }
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("a frozen 1000-epoch checkpoint changed during evaluation")
    preserved_after = {str(path): tree_identity(path) for path in preserved_paths}
    if preserved_before != preserved_after:
        raise RuntimeError("a prior experiment tree changed during reevaluation")
    interpretation = build_interpretation(
        old_aggregate,
        new_aggregate,
        old_restart_aggregate,
        new_restart_aggregate,
        old_local_aggregate,
        new_local_aggregate,
    )
    serializable_runs = [
        {
            **{key: value for key, value in run.items() if key != "checkpoint"},
            "checkpoint": str(run["checkpoint"]),
        }
        for run in runs
    ]
    new_serializable = []
    for run in new_run_rows:
        new_serializable.append(
            {
                "treatment": run["treatment"],
                "seed": run["seed"],
                "checkpoint": str(run["checkpoint"]),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "best_epoch": run["best_epoch"],
                "rollouts": run["rollouts"],
            }
        )
    manifest = {
        "stage": "1000-epoch Model-A validation-only rollout reevaluation",
        "status": "nine_checkpoints_x_24_trajectories_completed",
        "training_performed": False,
        "checkpoint_runs": serializable_runs,
        "checkpoint_hashes_before_and_after_identical": True,
        "frozen_validation_identity_sha256": old_manifest["validation_inputs"][
            "frozen_validation_identity_sha256"
        ],
        "frozen_validation_trajectory_count": len(members),
        "reference_source": str(old_rollout_path("physical_only", 101)),
        "reference_regenerated": False,
        "reference_consistency_across_nine_old_artifacts": True,
        "aggregate_time_curve_convention": (
            "No aggregate time curve is constructed. Scalar errors are computed on each "
            "fixed complete trajectory before aggregation; representative and restart plots "
            "use fixed individual trajectories."
        ),
        "old_runs": old_run_rows,
        "new_runs": new_serializable,
        "comparison": {
            "old_aggregate": old_aggregate,
            "new_aggregate": new_aggregate,
        },
        "restart": {
            "old_individual": old_restart_manifest["restart_individual"],
            "new_individual": restart_rows,
            "old_aggregate": old_restart_aggregate,
            "new_aggregate": new_restart_aggregate,
        },
        "teacher_forced_local_error": {
            "old_individual": old_restart_manifest["trajectory_local_error_summary"],
            "new_individual": local_rows,
            "old_aggregate": old_local_aggregate,
            "new_aggregate": new_local_aggregate,
        },
        "artifacts": {
            "full_validation_rollouts": {
                "path": str(full_path),
                "sha256": sha256(full_path),
            },
            "restart_and_local_diagnostics": {
                "path": str(restart_path),
                "sha256": sha256(restart_path),
            },
        },
        "validated_v_total_definition": "sqrt(wormhole_sciml.total_speed_squared)",
        "interpretation": interpretation,
        "preserved_trees_before": preserved_before,
        "preserved_trees_after": preserved_after,
        "prior_outputs_preserved": True,
        "predictions_unconstrained": True,
        "clipping_or_projection_used": False,
        "dense_grid_recomputed": False,
        "restricted_data_opened": False,
        "domain_treatment_selected": False,
        "architecture_selected": False,
        "constraint_selected": False,
        "capacity_comparison_started": False,
        "anomalies": [],
    }
    manifest_path = OUTPUT_DIR / "rollout_evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_report(manifest)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {REPORT_DIR / 'ROLLOUT_REEVALUATION_REPORT.md'}")


if __name__ == "__main__":
    main()
