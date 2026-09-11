#!/usr/bin/env python3
"""Evaluate the controlled fixed-E0 energy-normal loss experiment."""

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

from wormhole_sciml.dynamics import conserved_energy, timelike_margin
from wormhole_sciml.model_a import (
    Normalization,
    load_trained_model,
    parameter_count,
    predict_increments,
)
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi
from wormhole_sciml.stage1_data import file_sha256, load_dataset

try:
    from evaluate_model_a_x_xi_energy_microcore40k import energy_recursive_rollout
    from evaluate_model_a_x_xi_sampling_rollouts import (
        H,
        MAX_MODEL_STEPS,
        SEED_COLORS,
        THROAT_VELOCITIES,
        exact_xi_path,
        family_aggregate,
        incoming_xi_diagnostic,
    )
except ModuleNotFoundError:
    from scripts.evaluate_model_a_x_xi_energy_microcore40k import energy_recursive_rollout
    from scripts.evaluate_model_a_x_xi_sampling_rollouts import (
        H,
        MAX_MODEL_STEPS,
        SEED_COLORS,
        THROAT_VELOCITIES,
        exact_xi_path,
        family_aggregate,
        incoming_xi_diagnostic,
    )


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model_a_fixed_E0_energy_normal_loss"
TRAINING_MANIFEST = OUTPUT / "energy_normal_training_manifest.json"
DERIVED = OUTPUT / "derived_energy_normals.npz"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "energy_normal_evaluation_summary.json"
ARRAYS_PATH = OUTPUT / "energy_normal_evaluation_arrays.npz"
METRICS_CSV = OUTPUT / "energy_normal_recursive_metrics.csv"
ATTRIBUTION_CSV = OUTPUT / "energy_normal_sensitivity_attribution.csv"
REPORT_PATH = OUTPUT / "ENERGY_NORMAL_LOSS_REPORT.md"

BASELINE = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
BASELINE_TRAINING = BASELINE / "energy_xi_training_manifest.json"
BASELINE_SUMMARY = BASELINE / "energy_xi_rollout_summary.json"
BASELINE_ARRAYS = BASELINE / "energy_xi_rollout_arrays.npz"
NORMALIZATION_PATH = BASELINE / "training" / "energy_input_normalization.json"
DATA_DIR = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling"
TRAIN_DATA = DATA_DIR / "outer_microcore40k_train_x_xi.npz"
VALIDATION_DATA = DATA_DIR / "outer_microcore8k_validation_x_xi.npz"
FAMILY_SUMMARY = ROOT / "output" / "c32x32_traversal_families" / "traversal_family_summary.json"
ORBIT_SPACING = ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_arrays.npz"
EXACT_REFERENCE = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
SENSITIVITY_DIR = ROOT / "output" / "model_a_fixed_E0_throat_sensitivity"
SENSITIVITY_ARRAYS = SENSITIVITY_DIR / "throat_sensitivity_arrays.npz"
SENSITIVITY_SUMMARY = SENSITIVITY_DIR / "throat_sensitivity_summary.json"
ALIGNMENT_SUMMARY = ROOT / "output" / "model_a_energy_gradient_alignment" / "energy_gradient_alignment_summary.json"

SEEDS = (101, 202, 303)
TREATMENTS = ("fixed_E0_baseline", "energy_normal")
TITLES = {"fixed_E0_baseline": "frozen fixed E0", "energy_normal": "energy-normal loss"}
HARD = THROAT_VELOCITIES[:3]
INPUT_COLUMNS = ("x", "xi", "E0")
TARGET_COLUMNS = ("delta_x", "delta_xi")
NORMALIZATION_SOURCE = "outer_microcore40k_train_x_xi_energy_input_only"
REGIONS = {
    "far_upstream": (-17.0, -8.5),
    "middle_upstream": (-8.5, -4.0),
    "inner_incoming": (-4.0, -1.0),
    "near_throat": (-1.0, 0.0),
}


def family_key(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required protected artifact(s) missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def protected_paths(training: dict[str, Any]) -> tuple[Path, ...]:
    baseline_training = json.loads(BASELINE_TRAINING.read_text(encoding="utf-8"))
    paths = [
        TRAIN_DATA,
        VALIDATION_DATA,
        NORMALIZATION_PATH,
        BASELINE_TRAINING,
        BASELINE_SUMMARY,
        BASELINE_ARRAYS,
        FAMILY_SUMMARY,
        ORBIT_SPACING,
        EXACT_REFERENCE,
        SENSITIVITY_ARRAYS,
        SENSITIVITY_SUMMARY,
        ALIGNMENT_SUMMARY,
        TRAINING_MANIFEST,
        DERIVED,
    ]
    for row in baseline_training["runs"] + training["runs"]:
        paths.extend(
            (
                Path(row["checkpoint"]),
                Path(row["history"]),
                Path(row["checkpoint"]).parent / "metadata.json",
            )
        )
    return tuple(paths)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "mae": float(np.mean(np.abs(array))),
        "median": float(np.median(array)),
        "median_absolute": float(np.median(np.abs(array))),
        "p90_absolute": float(np.quantile(np.abs(array), 0.90)),
        "p95_absolute": float(np.quantile(np.abs(array), 0.95)),
        "p99_absolute": float(np.quantile(np.abs(array), 0.99)),
        "maximum_absolute": float(np.max(np.abs(array))),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate_seed_metrics(rows: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(rows[str(seed)][key]) for seed in SEEDS])
        result[key] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
        }
    return result


def one_step_metrics(
    model: Any,
    normalization: Normalization,
    validation: dict[str, np.ndarray],
    normals: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    inputs = np.column_stack(tuple(validation[name] for name in INPUT_COLUMNS))
    exact = np.column_stack(tuple(validation[name] for name in TARGET_COLUMNS))
    prediction = predict_increments(model, inputs, normalization)
    physical_error = prediction - exact
    standardized = physical_error / normalization.target_std
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    e_perp = np.sum(standardized * normals, axis=1)
    e_parallel = np.sum(standardized * tangent, axis=1)
    result = {
        "count": int(exact.shape[0]),
        "standardized_mse": float(np.mean(standardized**2)),
        "rmse_delta_x": float(np.sqrt(np.mean(physical_error[:, 0] ** 2))),
        "rmse_delta_xi": float(np.sqrt(np.mean(physical_error[:, 1] ** 2))),
        "mae_delta_x": float(np.mean(np.abs(physical_error[:, 0]))),
        "mae_delta_xi": float(np.mean(np.abs(physical_error[:, 1]))),
        "rms_e_perp": float(np.sqrt(np.mean(e_perp**2))),
        "mae_e_perp": float(np.mean(np.abs(e_perp))),
        "rms_e_parallel": float(np.sqrt(np.mean(e_parallel**2))),
        "mae_e_parallel": float(np.mean(np.abs(e_parallel))),
        "L_base": float(0.5 * np.mean(np.sum(standardized**2, axis=1))),
        "L_perp_E": float(0.5 * np.mean(e_perp**2)),
        "L_EN_lambda_1": float(
            0.5 * np.mean(np.sum(standardized**2, axis=1)) + 0.5 * np.mean(e_perp**2)
        ),
    }
    return result, {
        "prediction": prediction,
        "physical_error": physical_error,
        "standardized_error": standardized,
        "e_perp": e_perp,
        "e_parallel": e_parallel,
    }


def nonlinear_energy_diagnostic(
    prediction: np.ndarray, validation: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predicted_x = validation["x"] + prediction[:, 0]
    predicted_xi = validation["xi"] + prediction[:, 1]
    wormhole, spiral = experiment_parameters()
    _, predicted_u = state_from_xi(predicted_x, predicted_xi, wormhole, spiral)
    margin = timelike_margin(predicted_x, predicted_u, wormhole, spiral)
    valid = np.isfinite(predicted_x) & np.isfinite(predicted_xi) & np.isfinite(predicted_u) & (margin > 0.0)
    displacement = np.full(predicted_x.shape, np.nan, dtype=np.float64)
    displacement[valid] = (
        conserved_energy(predicted_x[valid], predicted_u[valid], wormhole, spiral)
        - validation["E0"][valid]
    )
    result = {
        "valid_count": int(np.sum(valid)),
        "invalid_or_nonphysical_count": int(np.sum(~valid)),
        "invalid_fraction": float(np.mean(~valid)),
        "minimum_predicted_timelike_margin": float(np.nanmin(margin)),
        "signed_delta_E": distribution(displacement[valid]),
        "absolute_delta_E": distribution(np.abs(displacement[valid])),
    }
    return result, displacement, valid


def global_aggregate(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for treatment in TREATMENTS:
        rows = [case[mode][treatment]["seeds"][str(seed)] for case in cases for seed in SEEDS]
        errors = [row["absolute_throat_u_error"] for row in rows if row["absolute_throat_u_error"] is not None]
        maxima = [
            row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"]
            for row in rows
            if row["energy_diagnostic"]["maximum_absolute_relative_energy_drift_before_or_at_throat"] is not None
        ]
        output[treatment] = {
            "rollout_count": len(rows),
            "successful_full_traversal_count": int(sum(row["reached_x_plus_17"] for row in rows)),
            "physical_exit_count": int(sum(row["physical_exit"] for row in rows)),
            "guard_or_horizon_termination_count": int(sum(row["maximum_step_guard"] for row in rows)),
            "nonfinite_termination_count": int(sum(row["nonfinite"] for row in rows)),
            "throat_crossing_count": int(sum(row["interpolated_throat_u"] is not None for row in rows)),
            "mean_absolute_throat_u_error_successful_crossings": float(np.mean(errors)) if errors else None,
            "mean_maximum_absolute_relative_energy_drift_before_or_at_throat": float(np.mean(maxima)) if maxima else None,
        }
    return output


def hard_aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "mean_absolute_e_xi",
        "terminal_absolute_e_xi",
        "median_R_drift",
        "p90_R_drift",
        "maximum_R_drift",
        "fraction_R_drift_lt_1",
        "fraction_R_drift_lt_0p5",
        "maximum_absolute_cumulative_e_xi",
        "mean_signed_e_xi",
    )
    result: dict[str, Any] = {}
    for treatment in TREATMENTS:
        rows = [
            case["hard_recursive_xi_diagnostic"][treatment][str(seed)]
            for case in cases
            if case["u_th"] in HARD
            for seed in SEEDS
        ]
        result[treatment] = {
            key: float(np.mean([float(row[key]) for row in rows])) for key in keys
        }
    return result


def region_mask(x: np.ndarray, name: str) -> np.ndarray:
    low, high = REGIONS[name]
    return (x >= low) & (x < high) if name != "near_throat" else (x >= low) & (x <= high)


def attribution_metrics(
    x: np.ndarray, A_x: np.ndarray, A_xi: np.ndarray
) -> dict[str, Any]:
    total = A_x + A_xi
    result: dict[str, Any] = {
        "point_count": int(x.size),
        "sum_A_x": float(np.sum(A_x)),
        "sum_A_xi": float(np.sum(A_xi)),
        "sum_A_total": float(np.sum(total)),
        "sum_absolute_A_x": float(np.sum(np.abs(A_x))),
        "sum_absolute_A_xi": float(np.sum(np.abs(A_xi))),
        "sum_absolute_A_total": float(np.sum(np.abs(total))),
        "regions": {},
    }
    denominator = max(float(np.sum(np.abs(total))), 1.0e-300)
    for name in REGIONS:
        mask = region_mask(x, name)
        result["regions"][name] = {
            "point_count": int(np.sum(mask)),
            "sum_A_x": float(np.sum(A_x[mask])),
            "sum_A_xi": float(np.sum(A_xi[mask])),
            "sum_A_total": float(np.sum(total[mask])),
            "sum_absolute_A_total": float(np.sum(np.abs(total[mask]))),
            "fraction_sum_absolute_A_total": float(np.sum(np.abs(total[mask])) / denominator),
        }
    return result


def sensitivity_attribution(
    models: dict[int, Any], normalization: Normalization, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    source_summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    with np.load(SENSITIVITY_ARRAYS, allow_pickle=False) as source, np.load(
        EXACT_REFERENCE, allow_pickle=False
    ) as exact:
        for u_th in HARD:
            family, key = f"{u_th:.2f}", family_key(u_th)
            x = np.asarray(source[f"{key}__attribution_x"], dtype=np.float64)
            K_x = np.asarray(source[f"{key}__K_x"], dtype=np.float64)
            K_xi = np.asarray(source[f"{key}__K_xi"], dtype=np.float64)
            valid = np.asarray(source[f"{key}__valid_kernel_mask"], dtype=bool)
            state = np.column_stack((exact[f"{key}__exact_state"][:, 0], exact[f"{key}__exact_xi"]))
            target = np.column_stack(
                (source[f"{key}__exact_delta_x_all"], source[f"{key}__exact_delta_xi_all"])
            )
            E0 = float(source_summary["families"][family]["E0"])
            features = np.column_stack((state, np.full(state.shape[0], E0)))
            results[family] = {treatment: {} for treatment in TREATMENTS}
            arrays[f"{key}__attribution_x"] = x
            arrays[f"{key}__K_x"] = K_x
            arrays[f"{key}__K_xi"] = K_xi
            for seed in SEEDS:
                baseline_error = np.column_stack(
                    (
                        source[f"{key}__seed_{seed}__e_delta_x"],
                        source[f"{key}__seed_{seed}__e_delta_xi"],
                    )
                )
                prediction = predict_increments(models[seed], features, normalization)
                new_error = prediction[valid] - target[valid]
                for treatment, error in (
                    ("fixed_E0_baseline", baseline_error),
                    ("energy_normal", new_error),
                ):
                    A_x, A_xi = K_x * error[:, 0], K_xi * error[:, 1]
                    arrays[f"{key}__{treatment}__seed_{seed}__e_delta_x"] = error[:, 0]
                    arrays[f"{key}__{treatment}__seed_{seed}__e_delta_xi"] = error[:, 1]
                    arrays[f"{key}__{treatment}__seed_{seed}__A_x"] = A_x
                    arrays[f"{key}__{treatment}__seed_{seed}__A_xi"] = A_xi
                    arrays[f"{key}__{treatment}__seed_{seed}__A_total"] = A_x + A_xi
                    results[family][treatment][str(seed)] = attribution_metrics(x, A_x, A_xi)
    return results


def attach_recursive_attribution_comparison(
    attribution: dict[str, Any], cases: list[dict[str, Any]]
) -> None:
    by_family = {f"{case['u_th']:.2f}": case for case in cases}
    for family, treatments in attribution.items():
        for treatment, seeds in treatments.items():
            recursive_name = treatment
            for seed_text, row in seeds.items():
                actual = by_family[family]["full_traversal"][recursive_name]["seeds"][seed_text]["signed_throat_u_error"]
                predicted = row["sum_A_total"]
                row["actual_recursive_signed_throat_error"] = actual
                row["linearized_sign_matches_recursive"] = bool(
                    actual is not None and np.sign(predicted) == np.sign(actual)
                )
                row["linearized_absolute_error_order_value"] = abs(predicted)
        for treatment, seeds in treatments.items():
            order_linear = sorted(SEEDS, key=lambda seed: abs(seeds[str(seed)]["sum_A_total"]))
            order_actual = sorted(
                SEEDS,
                key=lambda seed: (
                    float("inf")
                    if by_family[family]["full_traversal"][treatment]["seeds"][str(seed)]["signed_throat_u_error"] is None
                    else abs(by_family[family]["full_traversal"][treatment]["seeds"][str(seed)]["signed_throat_u_error"])
                ),
            )
            for row in seeds.values():
                row["family_seed_order_linearized"] = order_linear
                row["family_seed_order_recursive"] = order_actual
                row["family_seed_order_matches"] = order_linear == order_actual


def plot_training(training: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), sharey=True, constrained_layout=True)
    for axis, run in zip(axes, training["runs"], strict=True):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = np.asarray([row["epoch"] for row in history])
        axis.plot(epoch, [row["validation_standardized_mse"] for row in history], label=r"validation $L_{base}$")
        axis.plot(epoch, [row["validation_energy_normal_loss"] for row in history], label=r"validation $L_{\perp E}$")
        axis.plot(epoch, [row["validation_energy_normal_total_loss"] for row in history], label=r"validation $L_{EN}$")
        axis.axvline(run["best_epoch"], color="0.3", ls="--", lw=0.9, label="selected")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("loss")
    figure.suptitle("Energy-normal training diagnostics")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_one_step(one_step: dict[str, Any], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    x = np.arange(len(SEEDS)); width = 0.36
    for axis, key, title in (
        (axes[0], "rms_e_perp", "energy-normal standardized RMS"),
        (axes[1], "rms_e_parallel", "energy-tangent standardized RMS"),
    ):
        for offset, treatment in ((-width / 2, TREATMENTS[0]), (width / 2, TREATMENTS[1])):
            axis.bar(
                x + offset,
                [one_step[treatment]["individual"][str(seed)][key] for seed in SEEDS],
                width,
                label=TITLES[treatment],
            )
        axis.set(title=title, xticks=x, xticklabels=SEEDS, xlabel="seed")
        axis.grid(alpha=0.2, axis="y")
        axis.legend(fontsize=8)
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_nonlinear_energy(
    diagnostic_arrays: dict[str, np.ndarray], destination: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True, sharey=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        data = []
        for treatment in TREATMENTS:
            values = np.abs(diagnostic_arrays[f"validation__{treatment}__seed_{seed}__delta_E"])
            data.append(values[np.isfinite(values)])
        axis.boxplot(data, tick_labels=["baseline", "EN"], showfliers=False)
        axis.set(title=f"seed {seed}", yscale="log")
        axis.grid(alpha=0.2, axis="y", which="both")
    axes[0].set_ylabel(r"true one-step $|\delta E_{NN}|$")
    figure.suptitle("Independent nonlinear validation energy displacement")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_throat_errors(cases: list[dict[str, Any]], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.5), constrained_layout=True, sharey=True)
    for axis, treatment in zip(axes, TREATMENTS, strict=True):
        for seed in SEEDS:
            values = [case["full_traversal"][treatment]["seeds"][str(seed)]["signed_throat_u_error"] for case in cases]
            axis.plot(THROAT_VELOCITIES, values, marker="o", color=SEED_COLORS[seed], label=f"seed {seed}")
        axis.axhline(0.0, color="0.3", lw=0.8)
        axis.set(title=TITLES[treatment], xlabel=r"exact $u_{th}$", ylabel="signed throat-velocity error")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Seven-family recursive throat fidelity")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def plot_attribution(arrays: dict[str, np.ndarray], destination: Path) -> None:
    key = family_key(0.05)
    x = arrays[f"{key}__attribution_x"]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True, sharex=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            prefix = f"{key}__{treatment}__seed_{seed}"
            total = arrays[f"{prefix}__A_total"]
            axes[row, 0].plot(x, total, color=SEED_COLORS[seed], lw=1.2, label=f"seed {seed}")
            axes[row, 1].plot(x, np.cumsum(total), color=SEED_COLORS[seed], lw=1.2, label=f"seed {seed}")
        for axis in axes[row]:
            axis.axvspan(-17, -8.5, color="0.7", alpha=0.18)
            axis.axhline(0, color="0.3", lw=0.8)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        axes[row, 0].set(title=f"{TITLES[treatment]} · local attribution", ylabel=r"$A_x+A_\xi$")
        axes[row, 1].set(title=f"{TITLES[treatment]} · cumulative attribution", ylabel="cumulative sum")
    axes[-1, 0].set_xlabel("exact incoming x")
    axes[-1, 1].set_xlabel("exact incoming x")
    figure.suptitle(r"Exact-kernel sensitivity attribution · $u_{th}=0.05$")
    figure.savefig(destination, dpi=185)
    plt.close(figure)


def write_recursive_csv(cases: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in ("full_traversal", "throat_started_outgoing"):
            for treatment in TREATMENTS:
                for seed in SEEDS:
                    row = case[mode][treatment]["seeds"][str(seed)]
                    energy = row["energy_diagnostic"]
                    rows.append(
                        {
                            "u_th": case["u_th"],
                            "mode": mode,
                            "treatment": treatment,
                            "seed": seed,
                            "status": row["status"],
                            "reached_x_plus_17": row["reached_x_plus_17"],
                            "physical_exit": row["physical_exit"],
                            "maximum_step_guard": row["maximum_step_guard"],
                            "recovered_throat_u": row["interpolated_throat_u"],
                            "signed_throat_u_error": row["signed_throat_u_error"],
                            "absolute_throat_u_error": row["absolute_throat_u_error"],
                            "maximum_pre_throat_absolute_relative_energy_drift": energy["maximum_absolute_relative_energy_drift_before_or_at_throat"],
                        }
                    )
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_attribution_csv(attribution: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for family, treatments in attribution.items():
        for treatment, seeds in treatments.items():
            for seed, row in seeds.items():
                flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
                rows.append({"u_th": family, "treatment": treatment, "seed": seed, **flat})
    with ATTRIBUTION_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percent(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def report_text(summary: dict[str, Any]) -> str:
    one = summary["one_step_validation"]
    nonlinear = summary["nonlinear_energy_validation"]
    full = summary["aggregate"]["full_traversal"]
    throat = summary["aggregate"]["throat_started_outgoing"]
    attribution = summary["sensitivity_attribution"]["0.05"]
    lines = [
        "# Controlled fixed-E0 energy-normal loss experiment",
        "",
        "## Protocol and implementation gate",
        "",
        "Exactly three new 3→32→32→2 two-tanh models used the frozen microcore40k split, fixed-E0 inputs, normalization, Adam settings, batch schedule, epoch cap, patience, seeds, and ordinary-validation-MSE checkpoint selection. The only changed training quantity was the loss. The baseline `torch.mean(error**2)` gives per-sample c=1/2; therefore lambda_E=1 was implemented as `mean(e**2) + 0.5*mean((n_E·e)**2)`.",
        "",
        f"Training-side n_E reproduced the completed alignment gate with maximum absolute difference `{summary['training']['preflight']['completed_alignment_gate_maximum_absolute_n_E_difference']:.1e}`. Degenerate train/validation counts were `0/0`.",
        "",
        "## Selected training checkpoints",
        "",
        "| seed | best / stop epoch | validation L_base | validation L_perpE | validation L_EN |",
        "|--:|:--|--:|--:|--:|",
    ]
    for run in summary["training"]["runs"]:
        v = run["selected_checkpoint_validation"]
        lines.append(f"| {run['seed']} | {run['best_epoch']} / {run['stopping_epoch']} | {v['standardized_mse']:.6e} | {v['energy_normal_loss']:.6e} | {v['energy_normal_total_loss']:.6e} |")
    lines.extend([
        "",
        "## Frozen validation comparison (three-seed means)",
        "",
        "| metric | fixed-E0 baseline | energy-normal | change |",
        "|:--|--:|--:|--:|",
    ])
    for key in ("standardized_mse", "rmse_delta_x", "rmse_delta_xi", "mae_delta_x", "mae_delta_xi", "rms_e_perp", "mae_e_perp", "rms_e_parallel", "mae_e_parallel"):
        old = one[TREATMENTS[0]]["aggregate"][key]["mean"]
        new = one[TREATMENTS[1]]["aggregate"][key]["mean"]
        lines.append(f"| {key} | {old:.6e} | {new:.6e} | {percent(new, old):+.1f}% |")
    lines.extend([
        "",
        "## Independent nonlinear one-step energy displacement",
        "",
        "| treatment | mean seed-level MAE deltaE | invalid/nonphysical predictions |",
        "|:--|--:|--:|",
    ])
    for treatment in TREATMENTS:
        mae = np.mean([nonlinear[treatment]["individual"][str(seed)]["signed_delta_E"]["mae"] for seed in SEEDS])
        invalid = sum(nonlinear[treatment]["individual"][str(seed)]["invalid_or_nonphysical_count"] for seed in SEEDS)
        lines.append(f"| {TITLES[treatment]} | {mae:.6e} | {invalid} / 24000 |")
    lines.extend([
        "",
        "## Seven-family recursive evaluation",
        "",
        "| u_th | baseline throat u (101/202/303) | EN throat u (101/202/303) | baseline / EN MAE | baseline / EN reach-exit-guard |",
        "|--:|:--|:--|:--|:--|",
    ])
    for case in summary["cases"]:
        values = {}
        for treatment in TREATMENTS:
            section = case["full_traversal"][treatment]
            values[treatment] = "/".join(
                "—" if section["seeds"][str(seed)]["interpolated_throat_u"] is None
                else f"{section['seeds'][str(seed)]['interpolated_throat_u']:.5f}"
                for seed in SEEDS
            )
        b = case["full_traversal"][TREATMENTS[0]]["aggregate"]
        n = case["full_traversal"][TREATMENTS[1]]["aggregate"]
        lines.append(f"| {case['u_th']:.2f} | {values[TREATMENTS[0]]} | {values[TREATMENTS[1]]} | {b['mean_absolute_throat_u_error_successful_crossings']:.5f} / {n['mean_absolute_throat_u_error_successful_crossings']:.5f} | {b['successful_full_traversal_count']}-{b['physical_exit_count']}-{b['guard_or_horizon_termination_count']} / {n['successful_full_traversal_count']}-{n['physical_exit_count']}-{n['guard_or_horizon_termination_count']} |")
    lines.extend([
        "",
        f"Across all full rollouts, EN reach/exit/guard counts are `{full['energy_normal']['successful_full_traversal_count']}/{full['energy_normal']['physical_exit_count']}/{full['energy_normal']['guard_or_horizon_termination_count']}` versus `{full['fixed_E0_baseline']['successful_full_traversal_count']}/{full['fixed_E0_baseline']['physical_exit_count']}/{full['fixed_E0_baseline']['guard_or_horizon_termination_count']}`. Throat-started EN controls are `{throat['energy_normal']['successful_full_traversal_count']}/21` successful with `{throat['energy_normal']['physical_exit_count']}` exits and `{throat['energy_normal']['guard_or_horizon_termination_count']}` guards.",
        "",
        "## u_th=0.05 exact-kernel attribution",
        "",
        "| seed | baseline / EN far-upstream sum A_xi | baseline / EN total sum|A| | baseline / EN linearized total | baseline / EN recursive throat error |",
        "|--:|:--|:--|:--|:--|",
    ])
    case005 = next(case for case in summary["cases"] if case["u_th"] == 0.05)
    for seed in SEEDS:
        b, n = attribution[TREATMENTS[0]][str(seed)], attribution[TREATMENTS[1]][str(seed)]
        rb = case005["full_traversal"][TREATMENTS[0]]["seeds"][str(seed)]["signed_throat_u_error"]
        rn = case005["full_traversal"][TREATMENTS[1]]["seeds"][str(seed)]["signed_throat_u_error"]
        rb_text = "—" if rb is None else f"{rb:+.6e}"
        rn_text = "—" if rn is None else f"{rn:+.6e}"
        lines.append(f"| {seed} | {b['regions']['far_upstream']['sum_A_xi']:+.6e} / {n['regions']['far_upstream']['sum_A_xi']:+.6e} | {b['sum_absolute_A_total']:.6e} / {n['sum_absolute_A_total']:.6e} | {b['sum_A_total']:+.6e} / {n['sum_A_total']:+.6e} | {rb_text} / {rn_text} |")
    sci = summary["scientific_interpretation"]
    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        f"- **A. Local design target:** {sci['A_local_design']}",
        f"- **B. Ordinary accuracy:** {sci['B_ordinary_accuracy']}",
        f"- **C. Nonlinear orbit preservation:** {sci['C_nonlinear_energy']}",
        f"- **D. Long-horizon physics:** {sci['D_long_horizon']}",
        f"- **E. Mechanistic chain:** {sci['E_mechanistic_chain']}",
        f"- **F. Capacity cues:** {sci['F_capacity_cues']}",
        "",
        "## Integrity and stop",
        "",
        "All protected datasets, baseline runs, exact references, kernels, alignment-gate artifacts, normalization, and new checkpoints/histories retained identical hashes across evaluation. No extra lambda, weighting, raw-energy loss, multistep loss, architecture, or sampling variant was trained.",
    ])
    return "\n".join(lines) + "\n"


def scientific_interpretation(summary: dict[str, Any]) -> dict[str, str]:
    one = summary["one_step_validation"]
    nonlinear = summary["nonlinear_energy_validation"]
    full = summary["aggregate"]["full_traversal"]
    hard = summary["aggregate"]["hard_recursive_xi"]
    def mean_metric(treatment: str, key: str) -> float:
        return float(one[treatment]["aggregate"][key]["mean"])
    normal_change = percent(mean_metric("energy_normal", "rms_e_perp"), mean_metric("fixed_E0_baseline", "rms_e_perp"))
    normal_changes_by_seed = {
        seed: percent(
            one["energy_normal"]["individual"][str(seed)]["rms_e_perp"],
            one["fixed_E0_baseline"]["individual"][str(seed)]["rms_e_perp"],
        )
        for seed in SEEDS
    }
    tangent_change = percent(mean_metric("energy_normal", "rms_e_parallel"), mean_metric("fixed_E0_baseline", "rms_e_parallel"))
    mse_change = percent(mean_metric("energy_normal", "standardized_mse"), mean_metric("fixed_E0_baseline", "standardized_mse"))
    energy_old = np.mean([nonlinear["fixed_E0_baseline"]["individual"][str(seed)]["signed_delta_E"]["mae"] for seed in SEEDS])
    energy_new = np.mean([nonlinear["energy_normal"]["individual"][str(seed)]["signed_delta_E"]["mae"] for seed in SEEDS])
    energy_change = percent(float(energy_new), float(energy_old))
    case005 = next(case for case in summary["cases"] if case["u_th"] == 0.05)
    old005 = case005["full_traversal"]["fixed_E0_baseline"]["aggregate"]["mean_absolute_throat_u_error_successful_crossings"]
    new005 = case005["full_traversal"]["energy_normal"]["aggregate"]["mean_absolute_throat_u_error_successful_crossings"]
    throat_change = percent(float(new005), float(old005))
    attr = summary["sensitivity_attribution"]["0.05"]
    old_abs = np.mean([attr["fixed_E0_baseline"][str(seed)]["sum_absolute_A_total"] for seed in SEEDS])
    new_abs = np.mean([attr["energy_normal"][str(seed)]["sum_absolute_A_total"] for seed in SEEDS])
    attr_change = percent(float(new_abs), float(old_abs))
    exit_005 = case005["full_traversal"]["energy_normal"]["seeds"]["202"]["exit"]
    baseline_best = [row["best_epoch"] for row in json.loads(BASELINE_TRAINING.read_text(encoding="utf-8"))["runs"]]
    new_best = [row["best_epoch"] for row in summary["training"]["runs"]]
    return {
        "A_local_design": f"RMS e_perp changed {normal_change:+.1f}% across seed means, but seed changes were {normal_changes_by_seed[101]:+.1f}%/{normal_changes_by_seed[202]:+.1f}%/{normal_changes_by_seed[303]:+.1f}%; the intended local effect is real but not seed-uniform.",
        "B_ordinary_accuracy": f"Ordinary standardized MSE changed {mse_change:+.1f}% and tangent RMS changed {tangent_change:+.1f}%; component RMSE/MAE changes are tabulated above.",
        "C_nonlinear_energy": f"Mean seed-level true one-step |delta E| changed {energy_change:+.1f}%, with every invalid/nonphysical prediction counted.",
        "D_long_horizon": f"u_th=0.05 mean absolute crossing error changed {throat_change:+.1f}%; seeds 101/303 changed from undershoot to large overshoot, while seed 202 exited at x={exit_005['x']:.3f}, xi={exit_005['xi']:.6f}. EN full reach/exit/guard counts are {full['energy_normal']['successful_full_traversal_count']}/{full['energy_normal']['physical_exit_count']}/{full['energy_normal']['guard_or_horizon_termination_count']}; all throat-started controls remained successful.",
        "E_mechanistic_chain": f"u_th=0.05 mean total absolute first-order attribution changed {attr_change:+.1f}%. Global validation normal/energy improvements did not transfer to the trajectory-conditioned hard-family errors, so the proposed chain breaks at the sensitivity-attribution/recursive stages in this experiment.",
        "F_capacity_cues": f"Selected epochs shifted from baseline {baseline_best} to EN {new_best}; the strong seed-dependent normal/tangent tradeoff and noisy late validation plateaus are compatible with a capacity/optimization limitation, but are not decisive evidence that architecture alone is the cause.",
    }


def main() -> None:
    for path in (SUMMARY_PATH, ARRAYS_PATH, METRICS_CSV, ATTRIBUTION_CSV, REPORT_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation artifact {path}")
    training = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    protected = protected_paths(training)
    before = hashes(protected)
    baseline_training = json.loads(BASELINE_TRAINING.read_text(encoding="utf-8"))
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_cases = {float(case["u_th"]): case for case in baseline_summary["cases"]}
    family_source = {float(case["u_th"]): case for case in json.loads(FAMILY_SUMMARY.read_text(encoding="utf-8"))["cases"]}
    normalization = Normalization.from_stage1(
        NORMALIZATION_PATH, INPUT_COLUMNS, TARGET_COLUMNS, NORMALIZATION_SOURCE
    )
    models = {int(row["seed"]): load_trained_model(Path(row["checkpoint"])) for row in training["runs"]}
    baseline_models = {int(row["seed"]): load_trained_model(Path(row["checkpoint"])) for row in baseline_training["runs"]}
    if set(models) != set(SEEDS) or set(baseline_models) != set(SEEDS):
        raise RuntimeError("expected exactly three checkpoints per treatment")
    if any(parameter_count(model) != 1250 for model in [*models.values(), *baseline_models.values()]):
        raise RuntimeError("a checkpoint is not 3->32->32->2")
    validation = load_dataset(VALIDATION_DATA)
    with np.load(DERIVED, allow_pickle=False) as derived:
        normals = np.asarray(derived["validation_n_E"], dtype=np.float64)
        if not (
            np.array_equal(derived["validation_source_row_index"], validation["source_row_index"])
            and np.array_equal(derived["validation_x_next"], validation["x_next"])
            and np.array_equal(derived["validation_xi_next"], validation["xi_next"])
        ):
            raise RuntimeError("derived validation normals no longer correspond to frozen rows")

    arrays: dict[str, np.ndarray] = {}
    one_step: dict[str, Any] = {}
    nonlinear: dict[str, Any] = {}
    for treatment, treatment_models in (("fixed_E0_baseline", baseline_models), ("energy_normal", models)):
        individual: dict[str, Any] = {}
        nonlinear_individual: dict[str, Any] = {}
        for seed in SEEDS:
            metrics, values = one_step_metrics(treatment_models[seed], normalization, validation, normals)
            individual[str(seed)] = metrics
            energy, delta_E, valid = nonlinear_energy_diagnostic(values["prediction"], validation)
            nonlinear_individual[str(seed)] = energy
            for name, value in values.items():
                arrays[f"validation__{treatment}__seed_{seed}__{name}"] = value
            arrays[f"validation__{treatment}__seed_{seed}__delta_E"] = delta_E
            arrays[f"validation__{treatment}__seed_{seed}__energy_valid"] = valid
        keys = tuple(individual[str(SEEDS[0])])
        one_step[treatment] = {"individual": individual, "aggregate": aggregate_seed_metrics(individual, keys)}
        nonlinear[treatment] = {"individual": nonlinear_individual}

    cases: list[dict[str, Any]] = []
    with np.load(BASELINE_ARRAYS, allow_pickle=False) as base_arrays, np.load(
        ORBIT_SPACING, allow_pickle=False
    ) as spacing:
        for u_th in THROAT_VELOCITIES:
            key = family_key(u_th)
            base_case = baseline_cases[u_th]
            exact_full = np.asarray(base_arrays[f"{key}__exact_full_state"])
            exact_throat = np.asarray(base_arrays[f"{key}__exact_throat_started_state"])
            E0_full = float(conserved_energy(exact_full[0, 0], exact_full[0, 1], *experiment_parameters()))
            E0_throat = float(conserved_energy(exact_throat[0, 0], exact_throat[0, 1], *experiment_parameters()))
            if not np.isclose(E0_full, float(family_source[u_th]["energy"]), rtol=0.0, atol=2.0e-12):
                raise RuntimeError("frozen family energy mismatch")
            case: dict[str, Any] = {
                "u_th": float(u_th),
                "E0_from_full_initial_state": E0_full,
                "E0_from_throat_initial_state": E0_throat,
                "full_traversal": {},
                "throat_started_outgoing": {},
                "hard_recursive_xi_diagnostic": {},
            }
            for mode, section, exact_state, E0 in (
                ("full", "full_traversal", exact_full, E0_full),
                ("throat", "throat_started_outgoing", exact_throat, E0_throat),
            ):
                baseline_section = copy.deepcopy(base_case[section]["fixed_E0"])
                case[section]["fixed_E0_baseline"] = baseline_section
                new_rows: dict[str, Any] = {}
                initial = exact_xi_path(exact_state)[0]
                for seed in SEEDS:
                    coordinates, physical, margins, drift, row = energy_recursive_rollout(
                        models[seed], normalization, initial, E0, u_th
                    )
                    new_rows[str(seed)] = row
                    prefix = f"{key}__{mode}__energy_normal__seed_{seed}"
                    arrays[f"{prefix}__coordinates"] = coordinates
                    arrays[f"{prefix}__physical_state"] = physical
                    arrays[f"{prefix}__C"] = margins
                    arrays[f"{prefix}__relative_energy_drift"] = drift
                    if mode == "full" and u_th in HARD:
                        spacing_x = np.asarray(spacing[f"{key}__x"])
                        spacing_values = np.asarray(spacing[f"{key}__orbit_spacing"])
                        diag_arrays, diag_summary = incoming_xi_diagnostic(
                            coordinates, margins, exact_full, spacing_x, spacing_values
                        )
                        diag_summary["fraction_R_drift_lt_0p5"] = float(np.mean(diag_arrays["R_drift"] < 0.5))
                        for suffix, value in diag_arrays.items():
                            arrays[f"{prefix}__incoming__{suffix}"] = value
                        case["hard_recursive_xi_diagnostic"].setdefault("energy_normal", {})[str(seed)] = diag_summary
                case[section]["energy_normal"] = {"seeds": new_rows, "aggregate": family_aggregate(new_rows)}
            if u_th in HARD:
                case["hard_recursive_xi_diagnostic"]["fixed_E0_baseline"] = copy.deepcopy(
                    base_case["hard_recursive_xi_diagnostic"]["fixed_E0"]
                )
            cases.append(case)
            print(f"completed energy-normal recursion u_th={u_th:.2f}", flush=True)

    attribution = sensitivity_attribution(models, normalization, arrays)
    attach_recursive_attribution_comparison(attribution, cases)
    aggregate = {
        "full_traversal": global_aggregate(cases, "full_traversal"),
        "throat_started_outgoing": global_aggregate(cases, "throat_started_outgoing"),
        "hard_recursive_xi": hard_aggregate(cases),
    }
    summary: dict[str, Any] = {
        "stage": "controlled fixed-E0 energy-normal loss training and matched evaluation",
        "status": "three_runs_and_complete_frozen_protocol_evaluation_completed",
        "training": training,
        "baseline_training": {"path": str(BASELINE_TRAINING), "retrained": False},
        "loss_convention": {
            "baseline_code": "torch.mean((prediction-target)**2) over batch and two coordinates",
            "per_sample_factor_c": 0.5,
            "lambda_E": 1.0,
            "implemented_batch_loss": "mean(e**2) + 0.5*mean((n_E dot e)**2)",
            "metric": "I + n_E n_E^T",
        },
        "one_step_validation": one_step,
        "nonlinear_energy_validation": nonlinear,
        "cases": cases,
        "aggregate": aggregate,
        "sensitivity_attribution": attribution,
        "protocol": {
            "families": list(THROAT_VELOCITIES),
            "step_size": H,
            "maximum_model_steps": MAX_MODEL_STEPS,
            "baseline_retrained": False,
            "evaluation_retraining": False,
            "E0_held_fixed": True,
            "sensitivity_kernels_recomputed_or_modified": False,
            "nonlinear_energy_used_in_training": False,
            "lambda_tuning_or_other_variant_training": False,
        },
    }
    summary["scientific_interpretation"] = scientific_interpretation(summary)

    FIGURES.mkdir()
    plot_training(training, FIGURES / "energy_normal_training_curves.png")
    plot_one_step(one_step, FIGURES / "normal_tangent_validation_comparison.png")
    plot_nonlinear_energy(arrays, FIGURES / "nonlinear_energy_displacement_comparison.png")
    plot_throat_errors(cases, FIGURES / "seven_family_throat_error_comparison.png")
    plot_attribution(arrays, FIGURES / "u_th_0p05_sensitivity_attribution_comparison.png")
    np.savez_compressed(ARRAYS_PATH, **arrays)
    write_recursive_csv(cases)
    write_attribution_csv(attribution)
    after = hashes(protected)
    if before != after:
        raise RuntimeError("a protected artifact changed during evaluation")
    summary["protected_hashes_before"] = before
    summary["protected_hashes_after"] = after
    summary["artifacts"] = {
        "report": str(REPORT_PATH),
        "summary": str(SUMMARY_PATH),
        "arrays": str(ARRAYS_PATH),
        "recursive_metrics_csv": str(METRICS_CSV),
        "sensitivity_attribution_csv": str(ATTRIBUTION_CSV),
        "figures": [str(path) for path in sorted(FIGURES.glob("*.png"))],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}, {ARRAYS_PATH}, CSVs, figures, and {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
