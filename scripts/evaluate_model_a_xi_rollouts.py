#!/usr/bin/env python3
"""Validation-only recursive comparison of physical-only Model-A coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml import (
    effective_angular_velocity, radial_acceleration, timelike_margin,
    total_speed_squared,
)
from wormhole_sciml.model_a import (
    TRAINING_SEEDS, Normalization, load_trained_model, predict_increments,
)
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi, xi_from_state
from wormhole_sciml.stage1_data import H, file_sha256
from evaluate_round1_1000_rollouts import extract_json_object, prefix


ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "output" / "round1_model_a_1000_rollouts"
OLD_MANIFEST = OLD_DIR / "rollout_evaluation_manifest.json"
OLD_ARRAYS = OLD_DIR / "full_validation_rollouts.npz"
NEW_RUN_DIR = ROOT / "output" / "model_a_x_xi_physical_1000" / "physical_only"
OUTPUT_DIR = ROOT / "output" / "model_a_xi_rollout_comparison"
REPRESENTATIVES = ("validation-below_terminal", "validation-boundary_throat")
COLORS = {"x_u": "#277da1", "x_xi": "#d1495b"}
LABELS = {"x_u": r"$(x,u)$", "x_xi": r"$(x,\xi)$"}


def recursive_xi(checkpoint: Path, initial: np.ndarray, steps: int, norm: Normalization):
    wormhole, spiral = experiment_parameters()
    coordinates = np.empty((steps + 1, 2), dtype=np.float64)
    coordinates[0] = (initial[0], xi_from_state(initial[0], initial[1], wormhole, spiral))
    model = load_trained_model(checkpoint)
    for index in range(steps):
        coordinates[index + 1] = coordinates[index] + predict_increments(
            model, coordinates[index : index + 1], norm
        )[0]
    _, u = state_from_xi(coordinates[:, 0], coordinates[:, 1], wormhole, spiral)
    return coordinates, np.column_stack((coordinates[:, 0], u))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group(selected):
        return {
            "trajectory_count": len(selected),
            "physical_exit_count": int(sum(row["physical_exit"] for row in selected)),
            "mean_final_combined_state_error": float(np.mean([row["final_error"] for row in selected])),
            "mean_time_mean_combined_state_error": float(np.mean([row["time_mean_error"] for row in selected])),
        }
    overall = group(rows)
    overall["mean_near_edge_time_mean_combined_state_error"] = float(np.mean(
        [row["time_mean_error"] for row in rows if row["category"] == "near_edge"]
    ))
    escaping = [row for row in rows if row["reference_class"] == "escaping_force_free"]
    null = [row for row in rows if row["reference_class"] == "null_boundary_asymptotic"]
    return {
        "overall": overall,
        "by_reference_class": {"escaping_force_free": group(escaping), "null_boundary_asymptotic": group(null)},
        "escaping_diagnostic": {
            "escaping_trend_count": int(sum(row["escaping_trend"] for row in escaping)),
            "final_terminal_condition_count": int(sum(row["terminal_condition"] for row in escaping)),
            "physical_exit_before_reference_terminal_count": int(sum(row["exit_before_terminal"] for row in escaping)),
        },
        "null_boundary_diagnostic": {
            "null_approach_trend_count": int(sum(row["null_approach"] for row in null)),
            "physical_exit_count": int(sum(row["physical_exit"] for row in null)),
        },
        "trajectories": rows,
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "time_mean_error": ("overall", "mean_time_mean_combined_state_error"),
        "final_error": ("overall", "mean_final_combined_state_error"),
        "near_edge_error": ("overall", "mean_near_edge_time_mean_combined_state_error"),
        "escaping_time_mean_error": ("by_reference_class", "escaping_force_free", "mean_time_mean_combined_state_error"),
        "null_time_mean_error": ("by_reference_class", "null_boundary_asymptotic", "mean_time_mean_combined_state_error"),
    }
    result = {}
    for name, path in paths.items():
        values = []
        for run in runs:
            value = run["rollouts"]
            for key in path:
                value = value[key]
            values.append(float(value))
        result[name] = {
            "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {str(run["seed"]): value for run, value in zip(runs, values)},
        }
    for name, path in {
        "physical_exit_count_across_72": ("overall", "physical_exit_count"),
        "escaping_exit_count_across_36": ("by_reference_class", "escaping_force_free", "physical_exit_count"),
        "null_exit_count_across_36": ("by_reference_class", "null_boundary_asymptotic", "physical_exit_count"),
        "escaping_trend_count_across_36": ("escaping_diagnostic", "escaping_trend_count"),
        "escaping_terminal_count_across_36": ("escaping_diagnostic", "final_terminal_condition_count"),
        "null_approach_count_across_36": ("null_boundary_diagnostic", "null_approach_trend_count"),
    }.items():
        result[name] = int(sum(np.asarray(run["rollouts"][path[0]][path[1]]) if len(path) == 2 else
                               run["rollouts"][path[0]][path[1]][path[2]] for run in runs))
    return result


def representative_figure(identifier: str, old: Any, new: dict[str, np.ndarray]) -> None:
    key = prefix(identifier)
    time, reference = old[f"{key}__time"], old[f"{key}__reference_state"]
    reference_v = old[f"{key}__reference_v_total"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    references = (reference[:, 0], reference[:, 1], reference_v)
    suffixes = ("predicted_state", "predicted_state", "predicted_v_total")
    for axis, values, title in zip(axes.ravel()[:3], references, ("position $x$", "radial velocity $u$", "total velocity $v_{tot}$")):
        axis.plot(time, values, color="black", lw=2, label="DOP853")
        axis.set(title=title, xlabel="$s$")
    for formulation in ("x_u", "x_xi"):
        source = old if formulation == "x_u" else new
        source_name = "physical_only" if formulation == "x_u" else formulation
        for panel, suffix in enumerate(suffixes):
            stacks = np.stack([source[f"{key}__{source_name}__seed_{seed}__{suffix}"] for seed in TRAINING_SEEDS])
            if suffix == "predicted_state":
                stacks = stacks[:, :, panel]
            mean, spread = np.mean(stacks, axis=0), np.std(stacks, axis=0, ddof=1)
            axes.ravel()[panel].plot(time, mean, color=COLORS[formulation], label=LABELS[formulation])
            axes.ravel()[panel].fill_between(time, mean - spread, mean + spread, color=COLORS[formulation], alpha=0.14)
        errors = np.stack([source[f"{key}__{source_name}__seed_{seed}__combined_error"] for seed in TRAINING_SEEDS])
        mean, spread = np.mean(errors, axis=0), np.std(errors, axis=0, ddof=1)
        axes[1, 1].plot(time, mean, color=COLORS[formulation], label=LABELS[formulation])
        axes[1, 1].fill_between(time, mean - spread, mean + spread, color=COLORS[formulation], alpha=0.14)
    axes[1, 1].set(title="common physical state error $e_z$", xlabel="$s$")
    axes[0, 0].legend()
    fig.suptitle(identifier)
    fig.savefig(OUTPUT_DIR / f"{key}_coordinate_rollout.png", dpi=180)
    plt.close(fig)


def summary_figure(comparison: dict[str, Any]) -> None:
    names = ("time_mean_error", "final_error", "near_edge_error", "escaping_time_mean_error", "null_time_mean_error")
    titles = ("time mean", "final", "near edge", "escaping", "null boundary")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for axis, name, title in zip(axes.ravel()[:5], names, titles):
        means = [comparison[key][name]["mean"] for key in ("x_u", "x_xi")]
        spreads = [comparison[key][name]["standard_deviation"] for key in ("x_u", "x_xi")]
        axis.bar((0, 1), means, yerr=spreads, color=[COLORS["x_u"], COLORS["x_xi"]])
        axis.set(xticks=(0, 1), xticklabels=(LABELS["x_u"], LABELS["x_xi"]), title=f"{title} $e_z$")
    exits = [comparison[key]["physical_exit_count_across_72"] for key in ("x_u", "x_xi")]
    axes.ravel()[5].bar((0, 1), exits, color=[COLORS["x_u"], COLORS["x_xi"]])
    axes.ravel()[5].set(xticks=(0, 1), xticklabels=(LABELS["x_u"], LABELS["x_xi"]), title="physical exits /72")
    fig.savefig(OUTPUT_DIR / "coordinate_rollout_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    old_manifest = json.loads(OLD_MANIFEST.read_text())
    expected_old_hash = old_manifest["artifacts"]["full_validation_rollouts"]["sha256"]
    if file_sha256(OLD_ARRAYS) != expected_old_hash:
        raise RuntimeError("stored (x,u) rollout array hash mismatch")
    old_runs = [run for run in old_manifest["new_runs"] if run["treatment"] == "physical_only"]
    members = old_runs[0]["rollouts"]["trajectories"]
    norm = Normalization.from_stage1(
        ROOT / "output" / "stage1_model_a_x_xi" / "normalization.json",
        ("x", "xi"), ("delta_x", "delta_xi"), "physical_train_x_xi",
    )
    physical_norm = Normalization.from_stage1(ROOT / "output" / "stage1_model_a" / "normalization.json")
    checkpoints = {seed: NEW_RUN_DIR / f"seed_{seed}" / "best_checkpoint.pt" for seed in TRAINING_SEEDS}
    protected = [OLD_MANIFEST, OLD_ARRAYS, *checkpoints.values()]
    before = {str(path): file_sha256(path) for path in protected}
    f_star = float(extract_json_object(ROOT / "reports" / "physics_gate" / "gate_results.json", "compact_domain_force_envelope")["F_star"])
    wormhole, spiral = experiment_parameters()
    new_runs, saved = [], {}
    with np.load(OLD_ARRAYS) as old:
        for seed in TRAINING_SEEDS:
            trajectories = []
            for member in members:
                key, steps = prefix(member["id"]), int(member["step_count"])
                reference, time = old[f"{key}__reference_state"], old[f"{key}__time"]
                coordinates, predicted = recursive_xi(checkpoints[seed], reference[0], steps, norm)
                if not np.array_equal(predicted[0], reference[0]):
                    raise RuntimeError("physical initial state was not reconstructed exactly")
                error = predicted - reference
                combined = np.sqrt((error[:, 0] / physical_norm.input_std[0]) ** 2 + (error[:, 1] / physical_norm.input_std[1]) ** 2)
                c = timelike_margin(predicted[:, 0], predicted[:, 1], wormhole, spiral)
                xi_exit, c_exit = np.abs(coordinates[:, 1]) >= 1.0, c <= 0.0
                if not np.array_equal(xi_exit, c_exit):
                    raise RuntimeError("xi and C physical-exit diagnostics disagree")
                exits = np.flatnonzero(xi_exit)
                terminal_time = member["terminal_entry_time_by_T_i"]
                omega_ratio = np.abs(effective_angular_velocity(predicted[:, 1], spiral)) / abs(spiral.omega)
                force_ratio = np.abs(radial_acceleration(predicted[:, 0], predicted[:, 1], wormhole, spiral)) / f_star
                trajectories.append({
                    "id": member["id"], "category": member["category"], "reference_class": member["reference_class"],
                    "final_error": float(combined[-1]), "time_mean_error": float(np.mean(combined)),
                    "physical_exit": bool(exits.size), "first_exit_step": None if not exits.size else int(exits[0]),
                    "first_exit_time": None if not exits.size else float(time[exits[0]]),
                    "escaping_trend": bool(abs(predicted[-1, 0]) > abs(predicted[0, 0]) and omega_ratio[-1] < omega_ratio[0]),
                    "terminal_condition": bool(omega_ratio[-1] <= 0.05 and force_ratio[-1] <= 0.05),
                    "exit_before_terminal": bool(exits.size and terminal_time is not None and time[exits[0]] < terminal_time),
                    "null_approach": bool(c[-1] < c[0]),
                })
                run_key = f"{key}__x_xi__seed_{seed}"
                saved[f"{run_key}__predicted_coordinates"] = coordinates
                saved[f"{run_key}__predicted_state"] = predicted
                saved[f"{run_key}__combined_error"] = combined
                saved[f"{run_key}__predicted_C"] = c
                saved[f"{run_key}__predicted_v_total"] = np.sqrt(total_speed_squared(predicted[:, 0], predicted[:, 1], wormhole, spiral))
            new_runs.append({"seed": seed, "checkpoint": str(checkpoints[seed]), "checkpoint_sha256": before[str(checkpoints[seed])], "rollouts": summarize(trajectories)})
        OUTPUT_DIR.mkdir(parents=True)
        np.savez_compressed(OUTPUT_DIR / "x_xi_validation_rollouts.npz", **saved)
        for identifier in REPRESENTATIVES:
            representative_figure(identifier, old, saved)
    comparison = {"x_u": aggregate(old_runs), "x_xi": aggregate(new_runs)}
    summary_figure(comparison)
    after = {str(path): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("a checkpoint or stored rollout artifact changed")
    artifacts = ["x_xi_validation_rollouts.npz", "coordinate_rollout_summary.png",
                 "validation_below_terminal_coordinate_rollout.png", "validation_boundary_throat_coordinate_rollout.png"]
    manifest = {
        "stage": "validation-only physical coordinate rollout comparison", "status": "three_x_xi_runs_x_24_completed",
        "checkpoints": {str(seed): {"path": str(checkpoints[seed]), "sha256": before[str(checkpoints[seed])]} for seed in TRAINING_SEEDS},
        "frozen_validation_identity_sha256": old_manifest["frozen_validation_identity_sha256"],
        "old_x_u_rollout_source": {"manifest": str(OLD_MANIFEST), "arrays": str(OLD_ARRAYS), "arrays_sha256": expected_old_hash, "reused_not_recomputed": True},
        "individual_x_u": old_runs, "individual_x_xi": new_runs, "aggregate_comparison": comparison,
        "artifacts": {name: {"path": str(OUTPUT_DIR / name), "sha256": file_sha256(OUTPUT_DIR / name)} for name in artifacts},
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protocol": {"training_performed": False, "predictions_unconstrained": True, "xi_clipped_or_projected": False,
                     "xi_C_exit_agreement_verified": True,
                     "restart_or_teacher_forcing_performed": False, "dense_grid_evaluated": False,
                     "reference_regenerated": False, "collar_models_used": False, "restricted_data_accessed": False},
    }
    path = OUTPUT_DIR / "rollout_comparison_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
