#!/usr/bin/env python3
"""Validation-only recursive rollout comparison for four Model-A architectures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml import (
    effective_angular_velocity, radial_acceleration, timelike_margin, total_speed_squared,
)
from wormhole_sciml.model_a import (
    Normalization, TRAINING_SEEDS, load_trained_model, parameter_count, predict_increments,
)
from wormhole_sciml.physics_gate import experiment_parameters, xi_from_state
from wormhole_sciml.stage1_data import file_sha256
from evaluate_model_a_xi_rollouts import aggregate, summarize
from evaluate_round1_1000_rollouts import extract_json_object, prefix


ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "output" / "round1_model_a_1000_rollouts"
OLD_MANIFEST = OLD_DIR / "rollout_evaluation_manifest.json"
OLD_ARRAYS = OLD_DIR / "full_validation_rollouts.npz"
ARCH_MANIFEST = ROOT / "output" / "model_a_architecture_comparison" / "architecture_comparison_manifest.json"
MATCHED_MANIFEST = ROOT / "output" / "model_a_parameter_matched_depth" / "parameter_matched_manifest.json"
OUTPUT_DIR = ROOT / "output" / "model_a_architecture_rollouts"
MODELS = ("a64", "a32x32", "c16x16", "c32x32")
REPRESENTATION = {"a64": "x_u", "a32x32": "x_u", "c16x16": "x_u_xi", "c32x32": "x_u_xi"}
EXPECTED_PARAMETERS = {"a64": 322, "a32x32": 1218, "c16x16": 370, "c32x32": 1250}
LABELS = {"a64": "A64", "a32x32": "A32×32", "c16x16": "C16×16", "c32x32": "C32×32"}
COLORS = {"a64": "#277da1", "a32x32": "#f8961e", "c16x16": "#2a9d8f", "c32x32": "#9b5de5"}
REPRESENTATIVES = ("validation-below_terminal", "validation-boundary_throat")


def recursive_physical(checkpoint: Path, initial: np.ndarray, steps: int,
                       normalization: Normalization, representation: str) -> tuple[np.ndarray, np.ndarray]:
    wormhole, spiral = experiment_parameters()
    states = np.empty((steps + 1, 2), dtype=np.float64)
    features = np.empty(steps + 1, dtype=np.float64)
    states[0] = initial
    model = load_trained_model(checkpoint)
    for index in range(steps):
        features[index] = xi_from_state(states[index, 0], states[index, 1], wormhole, spiral)
        inputs = states[index:index + 1]
        if representation == "x_u_xi":
            inputs = np.column_stack((inputs, features[index:index + 1]))
        states[index + 1] = states[index] + predict_increments(model, inputs, normalization)[0]
    features[-1] = xi_from_state(states[-1, 0], states[-1, 1], wormhole, spiral)
    return states, features


def selected_runs() -> dict[str, list[dict[str, Any]]]:
    architecture = json.loads(ARCH_MANIFEST.read_text(encoding="utf-8"))["models"]
    matched = json.loads(MATCHED_MANIFEST.read_text(encoding="utf-8"))["models"]
    return {key: (matched if key == "c16x16" else architecture)[key]["runs"] for key in MODELS}


def representative_figure(identifier: str, old: Any, saved: dict[str, np.ndarray]) -> None:
    key = prefix(identifier)
    time, reference = old[f"{key}__time"], old[f"{key}__reference_state"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for axis, values, title in zip(
        axes.ravel()[:3],
        (reference[:, 0], reference[:, 1], old[f"{key}__reference_v_total"]),
        ("position $x$", "radial velocity $u$", "total velocity $v_{tot}$"),
    ):
        axis.plot(time, values, color="black", lw=2, label="DOP853")
        axis.set(title=title, xlabel="$s$")
    for name in MODELS:
        states = np.stack([saved[f"{key}__{name}__seed_{seed}__predicted_state"] for seed in TRAINING_SEEDS])
        velocities = np.stack([saved[f"{key}__{name}__seed_{seed}__predicted_v_total"] for seed in TRAINING_SEEDS])
        errors = np.stack([saved[f"{key}__{name}__seed_{seed}__combined_error"] for seed in TRAINING_SEEDS])
        for panel, values in enumerate((states[:, :, 0], states[:, :, 1], velocities, errors)):
            mean, spread = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
            axes.ravel()[panel].plot(time, mean, color=COLORS[name], label=LABELS[name])
            axes.ravel()[panel].fill_between(time, mean - spread, mean + spread,
                                             color=COLORS[name], alpha=0.12)
    axes[1, 1].set(title="common physical state error $e_z$", xlabel="$s$")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(identifier)
    fig.savefig(OUTPUT_DIR / f"{key}_architecture_rollout.png", dpi=180)
    plt.close(fig)


def summary_figure(comparison: dict[str, Any]) -> None:
    metrics = ("time_mean_error", "final_error", "near_edge_error",
               "escaping_time_mean_error", "null_time_mean_error")
    titles = ("time mean", "final", "near edge", "escaping", "null boundary")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), constrained_layout=True)
    for axis, metric, title in zip(axes.ravel()[:5], metrics, titles):
        means = [comparison[name][metric]["mean"] for name in MODELS]
        spreads = [comparison[name][metric]["standard_deviation"] for name in MODELS]
        axis.bar(range(4), means, yerr=spreads, color=[COLORS[name] for name in MODELS])
        axis.set(xticks=range(4), xticklabels=[LABELS[name] for name in MODELS], title=f"{title} $e_z$")
        axis.tick_params(axis="x", labelrotation=25)
    exits = [comparison[name]["physical_exit_count_across_72"] for name in MODELS]
    axes.ravel()[5].bar(range(4), exits, color=[COLORS[name] for name in MODELS])
    axes.ravel()[5].set(xticks=range(4), xticklabels=[LABELS[name] for name in MODELS],
                        title="physical exits /72")
    axes.ravel()[5].tick_params(axis="x", labelrotation=25)
    fig.savefig(OUTPUT_DIR / "architecture_rollout_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    expected_old_hash = old_manifest["artifacts"]["full_validation_rollouts"]["sha256"]
    if file_sha256(OLD_ARRAYS) != expected_old_hash:
        raise RuntimeError("stored frozen rollout arrays fail their hash guard")
    old_a32_runs = [run for run in old_manifest["new_runs"] if run["treatment"] == "physical_only"]
    members = old_a32_runs[0]["rollouts"]["trajectories"]
    runs_by_model = selected_runs()
    physical_norm = Normalization.from_stage1(ROOT / "output" / "stage1_model_a" / "normalization.json")
    feature_norm = Normalization.from_stage1(
        ROOT / "output" / "model_a_x_u_xi_physical_1000" / "normalization.json",
        ("x", "u", "xi"), ("delta_x", "delta_u"), "physical_train_x_u_xi",
    )
    checkpoints = {name: {run["seed"]: Path(run["checkpoint"]) for run in runs_by_model[name]}
                   for name in MODELS}
    protected = [OLD_MANIFEST, OLD_ARRAYS, ARCH_MANIFEST, MATCHED_MANIFEST,
                 *(path for group in checkpoints.values() for path in group.values())]
    before = {str(path): file_sha256(path) for path in protected}
    f_star = float(extract_json_object(
        ROOT / "reports" / "physics_gate" / "gate_results.json", "compact_domain_force_envelope"
    )["F_star"])
    wormhole, spiral = experiment_parameters()
    evaluated, saved = {}, {}
    with np.load(OLD_ARRAYS) as old:
        for name in MODELS:
            model_runs = []
            for seed in TRAINING_SEEDS:
                model = load_trained_model(checkpoints[name][seed])
                if parameter_count(model) != EXPECTED_PARAMETERS[name]:
                    raise RuntimeError(f"{name} checkpoint parameter count mismatch")
                trajectories = []
                for member in members:
                    key, steps = prefix(member["id"]), int(member["step_count"])
                    reference, time = old[f"{key}__reference_state"], old[f"{key}__time"]
                    predicted, xi = recursive_physical(
                        checkpoints[name][seed], reference[0], steps,
                        physical_norm if REPRESENTATION[name] == "x_u" else feature_norm,
                        REPRESENTATION[name],
                    )
                    error = predicted - reference
                    combined = np.sqrt((error[:, 0] / physical_norm.input_std[0]) ** 2
                                       + (error[:, 1] / physical_norm.input_std[1]) ** 2)
                    c = timelike_margin(predicted[:, 0], predicted[:, 1], wormhole, spiral)
                    if not np.array_equal(np.abs(xi) >= 1.0, c <= 0.0):
                        raise RuntimeError("xi and C physical-exit rules disagree")
                    exits = np.flatnonzero(c <= 0.0)
                    omega_ratio = np.abs(effective_angular_velocity(predicted[:, 1], spiral)) / abs(spiral.omega)
                    force_ratio = np.abs(radial_acceleration(predicted[:, 0], predicted[:, 1], wormhole, spiral)) / f_star
                    terminal_time = member["terminal_entry_time_by_T_i"]
                    trajectories.append({
                        "id": member["id"], "category": member["category"],
                        "reference_class": member["reference_class"], "final_error": float(combined[-1]),
                        "time_mean_error": float(np.mean(combined)), "physical_exit": bool(exits.size),
                        "first_exit_step": None if not exits.size else int(exits[0]),
                        "first_exit_time": None if not exits.size else float(time[exits[0]]),
                        "escaping_trend": bool(abs(predicted[-1, 0]) > abs(predicted[0, 0]) and omega_ratio[-1] < omega_ratio[0]),
                        "terminal_condition": bool(omega_ratio[-1] <= 0.05 and force_ratio[-1] <= 0.05),
                        "exit_before_terminal": bool(exits.size and terminal_time is not None and time[exits[0]] < terminal_time),
                        "null_approach": bool(c[-1] < c[0]),
                    })
                    stem = f"{key}__{name}__seed_{seed}"
                    saved[f"{stem}__predicted_state"] = predicted
                    saved[f"{stem}__predicted_xi"] = xi
                    saved[f"{stem}__combined_error"] = combined
                    saved[f"{stem}__predicted_C"] = c
                    saved[f"{stem}__predicted_v_total"] = np.sqrt(total_speed_squared(
                        predicted[:, 0], predicted[:, 1], wormhole, spiral
                    ))
                source = next(run for run in runs_by_model[name] if run["seed"] == seed)
                model_runs.append({"seed": seed, "checkpoint": str(checkpoints[name][seed]),
                                   "checkpoint_sha256": before[str(checkpoints[name][seed])],
                                   "best_epoch": source["best_epoch"], "parameter_count": EXPECTED_PARAMETERS[name],
                                   "rollouts": summarize(trajectories)})
            evaluated[name] = model_runs
        OUTPUT_DIR.mkdir(parents=True)
        np.savez_compressed(OUTPUT_DIR / "architecture_validation_rollouts.npz", **saved)
        for identifier in REPRESENTATIVES:
            representative_figure(identifier, old, saved)
    comparison = {name: aggregate(evaluated[name]) for name in MODELS}
    summary_figure(comparison)
    after = {str(path): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("a checkpoint or prior rollout artifact changed")
    artifacts = ("architecture_validation_rollouts.npz", "architecture_rollout_summary.png",
                 "validation_below_terminal_architecture_rollout.png",
                 "validation_boundary_throat_architecture_rollout.png")
    manifest = {
        "stage": "validation-only Model-A architecture recursive comparison",
        "status": "four_architectures_x_three_seeds_x_24_completed",
        "models": evaluated, "aggregate_comparison": comparison,
        "stored_a32_context": aggregate(old_a32_runs),
        "c32_context": {"status": "feature_model_rollout_artifact_absent_not_recomputed",
                        "incompatible_x_xi_rollout_not_substituted": True},
        "frozen_validation_identity_sha256": old_manifest["frozen_validation_identity_sha256"],
        "reference_source": str(OLD_ARRAYS), "protected_hashes_before": before,
        "protected_hashes_after": after,
        "artifacts": {name: {"path": str(OUTPUT_DIR / name), "sha256": file_sha256(OUTPUT_DIR / name)}
                      for name in artifacts},
        "protocol": {"training_performed": False, "predictions_unconstrained": True,
                     "clipping_or_projection_performed": False, "collar_models_used": False,
                     "dense_grid_evaluated": False, "restart_evaluated": False,
                     "restricted_data_accessed": False, "reference_regenerated": False,
                     "xi_recomputed_canonically_each_C_step": True},
    }
    path = OUTPUT_DIR / "architecture_rollout_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
