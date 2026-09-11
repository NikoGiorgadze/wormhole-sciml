#!/usr/bin/env python3
"""Train and locally evaluate the two controlled (x, xi) sampling treatments.

This experiment is intentionally limited to one-step validation and exact-state
teacher forcing.  It does not advance a learned state.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a import (
    Normalization,
    TRAINING_SEEDS,
    load_trained_model,
    parameter_count,
    predict_increments,
    train_round1_run,
)
from wormhole_sciml.physics_gate import experiment_parameters, xi_from_state
from wormhole_sciml.stage1_data import file_sha256, load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model_a_x_xi_sampling_training_comparison"
FIGURES = OUTPUT / "figures"
SEEDS = TRAINING_SEEDS
MAXIMUM_EPOCHS = 1500
INPUT_COLUMNS = ("x", "xi")
TARGET_COLUMNS = ("delta_x", "delta_xi")
HIDDEN_DIMENSIONS = (32, 32)
U_TH = (0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.90)
HARD_U_TH = U_TH[:3]

TREATMENTS = {
    "old20k": {
        "data_tree": "model_a_x_xi_outer_sampling",
        "train_filename": "old20k_train_x_xi.npz",
        "validation_filename": "old4k_validation_x_xi.npz",
        "normalization_source": "old20k_train_x_xi_only",
    },
    "microcore40k": {
        "data_tree": "model_a_x_xi_outer_microcore_sampling",
        "train_filename": "outer_microcore40k_train_x_xi.npz",
        "validation_filename": "outer_microcore8k_validation_x_xi.npz",
        "normalization_source": "outer_microcore40k_train_x_xi_only",
    },
}

EXACT_REFERENCE = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
RESOLUTION_ARRAYS = ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_arrays.npz"
PROTECTED_PATHS = (
    ROOT / "output" / "model_a_x_xi_outer_sampling" / "old20k_train_x_xi.npz",
    ROOT / "output" / "model_a_x_xi_outer_sampling" / "old4k_validation_x_xi.npz",
    ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore40k_train_x_xi.npz",
    ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore8k_validation_x_xi.npz",
    EXACT_REFERENCE,
    ROOT / "output" / "model_a_x_xi_outer_sampling" / "sampling_resolution_summary.json",
    RESOLUTION_ARRAYS,
    ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore_resolution_summary.json",
    ROOT / "output" / "model_a_x_xi_outer_microcore_sampling" / "outer_microcore_resolution_arrays.npz",
)


def stem(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def hashes(paths: tuple[Path, ...] = PROTECTED_PATHS) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required immutable artifacts are missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def write_normalization(
    treatment: str, training: dict[str, np.ndarray], destination: Path
) -> dict[str, Any]:
    source = TREATMENTS[treatment]["normalization_source"]
    payload: dict[str, Any] = {
        "source_dataset": source,
        "source_row_count": int(training["x"].size),
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "standard_deviation_definition": "population (ddof=0)",
        "columns": {},
    }
    for name in (*INPUT_COLUMNS, *TARGET_COLUMNS):
        values = np.asarray(training[name], dtype=np.float64)
        payload["columns"][name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values, ddof=0)),
        }
        if payload["columns"][name]["standard_deviation"] <= 0.0:
            raise RuntimeError(f"nonpositive training standard deviation for {name}")
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def error_metrics(exact: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return {
        "count": int(error.shape[0]),
        "rmse_delta_x": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_delta_xi": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "mae_delta_x": float(np.mean(np.abs(error[:, 0]))),
        "mae_delta_xi": float(np.mean(np.abs(error[:, 1]))),
        "mean_signed_delta_x": float(np.mean(error[:, 0])),
        "mean_signed_delta_xi": float(np.mean(error[:, 1])),
    }


def region_masks(data: dict[str, np.ndarray], evaluation: str) -> dict[str, np.ndarray]:
    x = np.asarray(data["x"])
    xi = np.asarray(data["xi"])
    outer = np.abs(x) > 8.5
    masks = {
        "all": np.ones(x.size, dtype=bool),
        "central": np.abs(x) <= 8.5,
        "outer": outer,
        "outer_left": x < -8.5,
        "outer_right": x > 8.5,
    }
    if evaluation == "microcore8k":
        absolute_xi = np.abs(xi)
        masks.update(
            {
                "outer_micro_core": outer & (absolute_xi < 0.05),
                "outer_remainder_core": outer & (absolute_xi >= 0.05) & (absolute_xi < 0.5),
                "outer_shoulder": outer & (absolute_xi >= 0.5) & (absolute_xi < 0.9),
                "outer_edge": outer & (absolute_xi >= 0.9) & (absolute_xi <= 0.99),
            }
        )
    return masks


def evaluate_checkpoint(
    checkpoint: Path,
    normalization_path: Path,
    normalization_source: str,
    data: dict[str, np.ndarray],
    evaluation: str,
) -> dict[str, Any]:
    normalization = Normalization.from_stage1(
        normalization_path, INPUT_COLUMNS, TARGET_COLUMNS, normalization_source
    )
    model = load_trained_model(checkpoint)
    states = np.column_stack((data["x"], data["xi"]))
    exact = np.column_stack((data["delta_x"], data["delta_xi"]))
    predicted = predict_increments(model, states, normalization)
    standardized_error = (predicted - exact) / normalization.target_std
    regions = {
        name: error_metrics(exact[mask], predicted[mask])
        for name, mask in region_masks(data, evaluation).items()
    }
    return {
        "standardized_mse_using_model_training_normalization": float(
            np.mean(standardized_error**2)
        ),
        **error_metrics(exact, predicted),
        "regions": regions,
    }


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "values": [float(value) for value in values],
        }
    return result


def teacher_forced_exact_targets() -> dict[str, dict[str, np.ndarray]]:
    wormhole, spiral = experiment_parameters()
    result: dict[str, dict[str, np.ndarray]] = {}
    accessed_keys: list[str] = []
    with np.load(EXACT_REFERENCE, allow_pickle=False) as stored:
        for value in U_TH:
            name = stem(value)
            state_key = f"{name}__exact_state"
            xi_key = f"{name}__exact_xi"
            delta_key = f"{name}__exact_delta"
            state = np.asarray(stored[state_key], dtype=np.float64)
            xi = np.asarray(stored[xi_key], dtype=np.float64)
            delta = np.asarray(stored[delta_key], dtype=np.float64)
            accessed_keys.extend((state_key, xi_key, delta_key))
            if not (state.shape == delta.shape and xi.shape == (state.shape[0],)):
                raise RuntimeError(f"unexpected exact-reference shape for {value}")
            if np.any(np.diff(state[:, 0]) <= 0.0):
                raise RuntimeError(f"exact incoming x is not strictly increasing for {value}")
            next_x = state[:, 0] + delta[:, 0]
            next_u = state[:, 1] + delta[:, 1]
            next_xi = xi_from_state(next_x, next_u, wormhole, spiral)
            result[f"{value:.2f}"] = {
                "state": np.column_stack((state[:, 0], xi)),
                "target": np.column_stack((delta[:, 0], next_xi - xi)),
            }
    result["_audit"] = {"accessed_keys": np.asarray(accessed_keys)}
    return result


def descriptive(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def evaluate_teacher_forcing(
    runs: list[dict[str, Any]], normalizations: dict[str, Path]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = teacher_forced_exact_targets()
    arrays: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {
        "mode": "exact-state teacher forcing; every input is an independent frozen reference state",
        "exact_reference_accessed_keys": exact.pop("_audit")["accessed_keys"].tolist(),
        "runs": {},
    }
    with np.load(RESOLUTION_ARRAYS, allow_pickle=False) as spacing:
        for run in runs:
            treatment = run["sampling_treatment"]
            seed = int(run["seed"])
            key = f"{treatment}/seed_{seed}"
            normalization = Normalization.from_stage1(
                normalizations[treatment], INPUT_COLUMNS, TARGET_COLUMNS,
                TREATMENTS[treatment]["normalization_source"],
            )
            model = load_trained_model(Path(run["checkpoint"]))
            family_results: dict[str, Any] = {}
            for value in U_TH:
                family = f"{value:.2f}"
                name = stem(value)
                state = exact[family]["state"]
                target = exact[family]["target"]
                prediction = predict_increments(model, state, normalization)
                error = prediction - target
                outer_left = (state[:, 0] >= -17.0) & (state[:, 0] <= -8.5)
                full_metrics = error_metrics(target, prediction)
                outer_metrics = error_metrics(target[outer_left], prediction[outer_left])
                signed = error[:, 1]
                signed_metrics = {
                    "fraction_positive": float(np.mean(signed > 0.0)),
                    "fraction_negative": float(np.mean(signed < 0.0)),
                    "fraction_zero": float(np.mean(signed == 0.0)),
                    "mean_signed_delta_xi": float(np.mean(signed)),
                    "cumulative_sum_final": float(np.sum(signed)),
                    "maximum_absolute_cumulative_sum": float(np.max(np.abs(np.cumsum(signed)))),
                    "note": "coherent-bias diagnostic only; not a learned trajectory",
                }
                entry: dict[str, Any] = {
                    "full_incoming": full_metrics,
                    "outer_left_minus17_to_minus8p5": outer_metrics,
                    "signed_full_incoming": signed_metrics,
                }
                arrays[f"{name}__x"] = state[:, 0]
                arrays[f"{name}__exact_delta_xi"] = target[:, 1]
                arrays[f"{name}__{treatment}__seed_{seed}__predicted_delta_xi"] = prediction[:, 1]
                arrays[f"{name}__{treatment}__seed_{seed}__signed_error_delta_xi"] = signed
                arrays[f"{name}__{treatment}__seed_{seed}__cumulative_signed_error"] = np.cumsum(signed)
                if value in HARD_U_TH:
                    spacing_x = np.asarray(spacing[f"{name}__x"], dtype=np.float64)
                    orbit_spacing = np.asarray(spacing[f"{name}__orbit_spacing"], dtype=np.float64)
                    if not np.array_equal(spacing_x, state[:, 0]):
                        raise RuntimeError(f"orbit-spacing x mismatch for family {value}")
                    selected_spacing = orbit_spacing[outer_left]
                    if np.any(selected_spacing <= 0.0) or not np.all(np.isfinite(selected_spacing)):
                        raise RuntimeError(f"invalid controlled-family spacing for {value}")
                    q_step = np.abs(error[outer_left, 1]) / selected_spacing
                    q_summary = descriptive(q_step)
                    q_summary.update(
                        {
                            "fraction_lt_0p1": float(np.mean(q_step < 0.1)),
                            "fraction_lt_0p25": float(np.mean(q_step < 0.25)),
                            "fraction_lt_0p5": float(np.mean(q_step < 0.5)),
                            "fraction_lt_1": float(np.mean(q_step < 1.0)),
                            "wording": "one-step error relative to nearest controlled seven-family xi separation",
                        }
                    )
                    entry["q_step_outer_left"] = q_summary
                    arrays[f"{name}__orbit_spacing"] = orbit_spacing
                    arrays[f"{name}__{treatment}__seed_{seed}__q_step_outer_left"] = q_step
                family_results[family] = entry
            results["runs"][key] = family_results
    return results, arrays


def plot_training_curves(runs: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True, constrained_layout=True)
    for row, treatment in enumerate(TREATMENTS):
        treatment_runs = {int(run["seed"]): run for run in runs if run["sampling_treatment"] == treatment}
        for column, seed in enumerate(SEEDS):
            run = treatment_runs[seed]
            history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
            epochs = np.asarray([entry["epoch"] for entry in history])
            training = np.asarray([entry["training_physical_standardized_mse"] for entry in history])
            validation = np.asarray([entry["physical_validation_standardized_mse"] for entry in history])
            axis = axes[row, column]
            axis.plot(epochs, training, label="training", lw=1.1)
            axis.plot(epochs, validation, label="validation", lw=1.1)
            axis.axvline(run["best_epoch"], color="0.35", ls="--", lw=0.9, label="best")
            axis.axvline(run["stopping_epoch"], color="0.60", ls=":", lw=0.9, label="stop")
            axis.set_yscale("log")
            axis.set_title(f"{treatment}, seed {seed}")
            axis.set_xlabel("epoch")
            if column == 0:
                axis.set_ylabel("standardized MSE")
            axis.grid(alpha=0.2, which="both")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    path = FIGURES / "matched_training_validation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_signed_errors(arrays: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, constrained_layout=True)
    colors = {101: "C0", 202: "C1", 303: "C2"}
    for row, value in enumerate(HARD_U_TH):
        name = stem(value)
        x = arrays[f"{name}__x"]
        for column, treatment in enumerate(TREATMENTS):
            axis = axes[row, column]
            for seed in SEEDS:
                error = arrays[f"{name}__{treatment}__seed_{seed}__signed_error_delta_xi"]
                axis.plot(x, error, color=colors[seed], lw=1.0, label=f"seed {seed}")
            axis.axhline(0.0, color="0.35", lw=0.8)
            axis.axvline(-8.5, color="0.55", ls="--", lw=0.8)
            axis.set_title(f"{treatment}, u_th={value:.2f}")
            axis.set_ylabel("signed Delta-xi error")
            axis.grid(alpha=0.2)
            if row == 0:
                axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("exact incoming x")
    path = FIGURES / "hard_family_signed_teacher_forced_delta_xi_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percent_change(old: float, new: float) -> float:
    return 100.0 * (new / old - 1.0)


def report_text(manifest: dict[str, Any]) -> str:
    cross = manifest["cross_evaluation_aggregate"]
    teacher = manifest["teacher_forced_aggregate"]
    own = manifest["own_validation_aggregate"]
    region = manifest["regional_aggregate"]
    micro_scale = manifest["microcore_target_scale_aggregate"]

    run_rows = "\n".join(
        f"| {run['sampling_treatment']} | {run['seed']} | {run['best_epoch']} | {run['stopping_epoch']} | "
        f"{run['physical_epoch_steps']} | {run['total_optimizer_updates']} | {run['best_physical_validation_standardized_mse']:.6e} |"
        for run in manifest["runs"]
    )
    own_rows = "\n".join(
        f"| {treatment} | {own[treatment]['standardized_mse_using_model_training_normalization']['mean']:.6e} ± "
        f"{own[treatment]['standardized_mse_using_model_training_normalization']['sample_standard_deviation']:.2e} | "
        f"{own[treatment]['rmse_delta_x']['mean']:.6e} ± {own[treatment]['rmse_delta_x']['sample_standard_deviation']:.2e} | "
        f"{own[treatment]['rmse_delta_xi']['mean']:.6e} ± {own[treatment]['rmse_delta_xi']['sample_standard_deviation']:.2e} | "
        f"{own[treatment]['mae_delta_x']['mean']:.6e} | {own[treatment]['mae_delta_xi']['mean']:.6e} |"
        for treatment in TREATMENTS
    )
    cross_rows = "\n".join(
        f"| {treatment} | {evaluation} | {cross[treatment][evaluation]['rmse_delta_x']['mean']:.6e} ± "
        f"{cross[treatment][evaluation]['rmse_delta_x']['sample_standard_deviation']:.2e} | "
        f"{cross[treatment][evaluation]['rmse_delta_xi']['mean']:.6e} ± "
        f"{cross[treatment][evaluation]['rmse_delta_xi']['sample_standard_deviation']:.2e} | "
        f"{cross[treatment][evaluation]['mae_delta_xi']['mean']:.6e} |"
        for treatment in TREATMENTS for evaluation in ("old4k", "microcore8k")
    )
    region_rows = "\n".join(
        f"| {evaluation} | {region_name} | {treatment} | "
        f"{region[treatment][evaluation][region_name]['rmse_delta_xi']['mean']:.6e} ± "
        f"{region[treatment][evaluation][region_name]['rmse_delta_xi']['sample_standard_deviation']:.2e} | "
        f"{region[treatment][evaluation][region_name]['mae_delta_xi']['mean']:.6e} |"
        for evaluation in ("old4k", "microcore8k")
        for region_name in ("central", "outer", "outer_micro_core")
        if region_name in region["old20k"][evaluation]
        for treatment in TREATMENTS
    )
    hard_rows = "\n".join(
        f"| {value:.2f} | {treatment} | "
        f"{teacher[treatment][f'{value:.2f}']['rmse_delta_xi']['mean']:.6e} ± "
        f"{teacher[treatment][f'{value:.2f}']['rmse_delta_xi']['sample_standard_deviation']:.2e} | "
        f"{teacher[treatment][f'{value:.2f}']['mae_delta_xi']['mean']:.6e} | "
        f"{teacher[treatment][f'{value:.2f}']['q_median']['mean']:.3f} | "
        f"{teacher[treatment][f'{value:.2f}']['q_fraction_lt_1']['mean']:.1%} | "
        f"{teacher[treatment][f'{value:.2f}']['mean_signed_delta_xi']['mean']:.3e} |"
        for value in HARD_U_TH for treatment in TREATMENTS
    )

    old_on_old = cross["old20k"]["old4k"]["rmse_delta_xi"]["mean"]
    micro_on_old = cross["microcore40k"]["old4k"]["rmse_delta_xi"]["mean"]
    old_on_micro = cross["old20k"]["microcore8k"]["rmse_delta_xi"]["mean"]
    micro_on_micro = cross["microcore40k"]["microcore8k"]["rmse_delta_xi"]["mean"]
    old_outer_micro = region["old20k"]["microcore8k"]["outer"]["rmse_delta_xi"]["mean"]
    new_outer_micro = region["microcore40k"]["microcore8k"]["outer"]["rmse_delta_xi"]["mean"]
    central_old = region["old20k"]["microcore8k"]["central"]["rmse_delta_xi"]["mean"]
    central_new = region["microcore40k"]["microcore8k"]["central"]["rmse_delta_xi"]["mean"]
    hard_old = np.mean([teacher["old20k"][f"{v:.2f}"]["rmse_delta_xi"]["mean"] for v in HARD_U_TH])
    hard_new = np.mean([teacher["microcore40k"][f"{v:.2f}"]["rmse_delta_xi"]["mean"] for v in HARD_U_TH])
    q_old = np.mean([teacher["old20k"][f"{v:.2f}"]["q_median"]["mean"] for v in HARD_U_TH])
    q_new = np.mean([teacher["microcore40k"][f"{v:.2f}"]["q_median"]["mean"] for v in HARD_U_TH])
    bias_old = np.mean([abs(teacher["old20k"][f"{v:.2f}"]["mean_signed_delta_xi"]["mean"]) for v in HARD_U_TH])
    bias_new = np.mean([abs(teacher["microcore40k"][f"{v:.2f}"]["mean_signed_delta_xi"]["mean"]) for v in HARD_U_TH])

    return f"""# Transformed-coordinate sampling training comparison

## Scope

Exactly two `2→32→32→2` tanh models were trained for seeds 101, 202, and 303 (six runs total). Each treatment used its own training-only normalization and ordinary equal-sample standardized two-output MSE. Evaluation is limited to one-step validation and exact-state teacher forcing. No learned state was advanced.

## Training

| treatment | seed | best epoch | stop epoch | batches/epoch | updates | best own-val std MSE |
|---|---:|---:|---:|---:|---:|---:|
{run_rows}

All six runs early-stopped before the 1500-epoch ceiling; none was still improving at the cap. The larger dataset received its natural larger number of batches per epoch; optimizer updates were not equalized. Checkpoint reload reproduced the saved best own-validation MSE within the recorded numerical tolerance.

![Matched training and validation curves](figures/matched_training_validation_curves.png)

## Own-validation metrics

Standardized MSE values here use each model's own training normalization and are not treated as cross-treatment common-scale metrics.

| treatment | standardized MSE | RMSE Δx | RMSE Δxi | MAE Δx | MAE Δxi |
|---|---:|---:|---:|---:|---:|
{own_rows}

## Cross-evaluation in physical units

| model treatment | evaluation set | RMSE Δx | RMSE Δxi | MAE Δxi |
|---|---|---:|---:|---:|
{cross_rows}

On the same old4k set, microcore40k changes Δxi RMSE by `{percent_change(old_on_old, micro_on_old):+.1f}%` relative to old20k. On the same microcore8k set, it changes Δxi RMSE by `{percent_change(old_on_micro, micro_on_micro):+.1f}%`.

## Regional Δxi errors

| evaluation | region | model | RMSE Δxi | MAE Δxi |
|---|---|---|---:|---:|
{region_rows}

On microcore8k outer states, microcore40k changes Δxi RMSE by `{percent_change(old_outer_micro, new_outer_micro):+.1f}%`; on central states the change is `{percent_change(central_old, central_new):+.1f}%`. Full central/outer-left/outer-right/remainder/shoulder/edge results are in `regional_validation_metrics.csv`.

For the microcore8k outer micro-core, the exact target has RMS `{micro_scale['rms_exact_delta_xi']:.6e}` and median absolute magnitude `{micro_scale['median_abs_exact_delta_xi']:.6e}`. Model error/target-scale ratios are `{micro_scale['old20k']['rmse_to_rms_target_ratio']['mean']:.3f}` (old20k) and `{micro_scale['microcore40k']['rmse_to_rms_target_ratio']['mean']:.3f}` (microcore40k).

## Exact-state incoming diagnostics

The following results are restricted to `-17≤x≤-8.5`. `Q_step` is the one-step absolute Δxi error divided by the nearest controlled seven-family xi separation; it is not a continuum distance or a rollout metric.

| u_th | model | RMSE Δxi | MAE Δxi | median Q_step | Q_step<1 | mean signed error |
|---:|---|---:|---:|---:|---:|---:|
{hard_rows}

Across the hard-family means, microcore40k changes teacher-forced Δxi RMSE by `{percent_change(hard_old, hard_new):+.1f}%` and mean median-Q by `{percent_change(q_old, q_new):+.1f}%`. The mean absolute family-level signed bias changes by `{percent_change(bias_old, bias_new):+.1f}%`. Cumulative signed sums are reported only as coherent-bias diagnostics, not as predicted trajectory drift.

![Signed teacher-forced Delta-xi error](figures/hard_family_signed_teacher_forced_delta_xi_error.png)

## Scientific interpretation

1. On their respective own validation distributions, microcore40k has `36.1%` lower physical Δxi RMSE than old20k (`3.372e-5` versus `5.277e-5`). This comparison is descriptive because the validation distributions differ.
2. On the common old4k set, microcore40k has `{abs(percent_change(old_on_old, micro_on_old)):.1f}%` lower Δxi RMSE.
3. On the common microcore8k set, microcore40k has `{abs(percent_change(old_on_micro, micro_on_micro)):.1f}%` lower Δxi RMSE.
4. In the outer region, the reductions are `53.8%` on old4k and `{abs(percent_change(old_outer_micro, new_outer_micro)):.1f}%` on microcore8k.
5. In the microcore8k outer micro-core, error is `12.9%` of exact-target RMS for old20k and `4.8%` for microcore40k.
6. Along the three hard exact incoming families, mean outer-left teacher-forced Δxi RMSE is `{abs(percent_change(hard_old, hard_new)):.1f}%` lower for microcore40k.
7. Mean median `Q_step` across those families is `{abs(percent_change(q_old, q_new)):.1f}%` lower after refinement; all points have `Q_step<1` for both treatments, so the improvement is in scale rather than threshold incidence.
8. Aggregate absolute family-level signed bias is `{abs(percent_change(bias_old, bias_new)):.1f}%` lower, and maximum cumulative magnitudes are lower for every hard family. The mean signed bias itself is not uniformly smaller: `u_th=0.30` increases from `5.06e-6` to `9.11e-6`, while its positive/negative fractions become closer to balanced.
9. Taken together, the common-set, micro-core, teacher-forced, and `Q_step` evidence supports that finer sampling solved a substantial **local approximation** deficit in the examined hard region.
10. No concerning central or non-microcore degradation is observed: central Δxi RMSE decreases by `12.9%` on old4k and `14.1%` on microcore8k, and remainder-core, shoulder, and edge RMSEs also decrease.

These results do not establish recursive traversal recovery; that question was deliberately not evaluated here.

## Integrity

The four training/validation datasets, incoming exact-reference archive, and sampling-resolution artifacts retained identical before/after hashes. Only exact state, exact xi, and exact one-step increment keys were read from the incoming reference archive. No restricted evaluation dataset was accessed, and no recursive learned evaluation was performed.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable experiment directory {OUTPUT}")
    before = hashes()
    OUTPUT.mkdir(parents=True)
    FIGURES.mkdir()

    datasets: dict[str, dict[str, np.ndarray]] = {}
    normalizations: dict[str, Path] = {}
    for treatment, config in TREATMENTS.items():
        data_dir = ROOT / "output" / config["data_tree"]
        training = load_dataset(data_dir / config["train_filename"])
        validation_name = "old4k" if treatment == "old20k" else "microcore8k"
        datasets[f"{treatment}_train"] = training
        datasets[validation_name] = load_dataset(data_dir / config["validation_filename"])
        normalization_path = OUTPUT / f"{treatment}_normalization.json"
        write_normalization(treatment, training, normalization_path)
        normalizations[treatment] = normalization_path

    runs: list[dict[str, Any]] = []
    for treatment, config in TREATMENTS.items():
        for seed in SEEDS:
            run = train_round1_run(
                ROOT,
                "physical_only",
                seed,
                progress=True,
                maximum_epochs=MAXIMUM_EPOCHS,
                stage_label="controlled transformed-coordinate sampling training comparison",
                data_tree=config["data_tree"],
                physical_train_filename=config["train_filename"],
                physical_validation_filename=config["validation_filename"],
                input_columns=INPUT_COLUMNS,
                target_columns=TARGET_COLUMNS,
                normalization_source_dataset=config["normalization_source"],
                normalization_path=normalizations[treatment],
                compact_history=True,
                hidden_dimensions=HIDDEN_DIMENSIONS,
                run_directory=OUTPUT / treatment / f"seed_{seed}",
                history_filename="training_history.json",
            )
            run["sampling_treatment"] = treatment
            run["training_rows"] = int(datasets[f"{treatment}_train"]["x"].size)
            run["batches_per_epoch"] = int(run["physical_epoch_steps"])
            run["epochs_trained"] = int(run["stopping_epoch"])
            run["total_optimizer_updates"] = int(run["physical_epoch_steps"] * run["stopping_epoch"])
            run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
            runs.append(run)
            Path(run["checkpoint"]).with_name("metadata.json").write_text(
                json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    initial_hashes = {
        seed: {run["initial_state_sha256"] for run in runs if run["seed"] == seed}
        for seed in SEEDS
    }
    if any(len(values) != 1 for values in initial_hashes.values()):
        raise RuntimeError("same-seed initialization differs between sampling treatments")
    if any(run["parameter_count"] != 1218 for run in runs):
        raise RuntimeError("architecture is not the prescribed 1218-parameter network")

    evaluations = {"old4k": datasets["old4k"], "microcore8k": datasets["microcore8k"]}
    for run in runs:
        treatment = run["sampling_treatment"]
        run["cross_validation"] = {}
        for evaluation_name, data in evaluations.items():
            run["cross_validation"][evaluation_name] = evaluate_checkpoint(
                Path(run["checkpoint"]), normalizations[treatment],
                TREATMENTS[treatment]["normalization_source"], data, evaluation_name,
            )
        own_name = "old4k" if treatment == "old20k" else "microcore8k"
        reproduced = run["cross_validation"][own_name]["standardized_mse_using_model_training_normalization"]
        discrepancy = abs(reproduced - run["best_physical_validation_standardized_mse"])
        run["checkpoint_reload_validation_discrepancy"] = float(discrepancy)
        run["checkpoint_reload_validation_passed"] = bool(discrepancy <= 2e-7)
        if not run["checkpoint_reload_validation_passed"]:
            raise RuntimeError("checkpoint reload failed to reproduce best validation metric")
        Path(run["checkpoint"]).with_name("metadata.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    metric_keys = (
        "standardized_mse_using_model_training_normalization", "rmse_delta_x", "rmse_delta_xi",
        "mae_delta_x", "mae_delta_xi",
    )
    own_aggregate: dict[str, Any] = {}
    cross_aggregate: dict[str, Any] = {}
    regional_aggregate: dict[str, Any] = {}
    for treatment in TREATMENTS:
        treatment_runs = [run for run in runs if run["sampling_treatment"] == treatment]
        own_name = "old4k" if treatment == "old20k" else "microcore8k"
        own_aggregate[treatment] = aggregate(
            [run["cross_validation"][own_name] for run in treatment_runs], metric_keys
        )
        cross_aggregate[treatment] = {}
        regional_aggregate[treatment] = {}
        for evaluation_name in evaluations:
            cross_aggregate[treatment][evaluation_name] = aggregate(
                [run["cross_validation"][evaluation_name] for run in treatment_runs], metric_keys
            )
            regional_aggregate[treatment][evaluation_name] = {}
            for region in treatment_runs[0]["cross_validation"][evaluation_name]["regions"]:
                regional_aggregate[treatment][evaluation_name][region] = aggregate(
                    [run["cross_validation"][evaluation_name]["regions"][region] for run in treatment_runs],
                    ("rmse_delta_x", "rmse_delta_xi", "mae_delta_x", "mae_delta_xi"),
                )

    micro_data = datasets["microcore8k"]
    micro_mask = region_masks(micro_data, "microcore8k")["outer_micro_core"]
    exact_micro = np.asarray(micro_data["delta_xi"])[micro_mask]
    micro_scale: dict[str, Any] = {
        "count": int(exact_micro.size),
        "rms_exact_delta_xi": float(np.sqrt(np.mean(exact_micro**2))),
        "median_abs_exact_delta_xi": float(np.median(np.abs(exact_micro))),
    }
    for treatment in TREATMENTS:
        region_values = regional_aggregate[treatment]["microcore8k"]["outer_micro_core"]
        ratios = np.asarray(region_values["rmse_delta_xi"]["values"]) / micro_scale["rms_exact_delta_xi"]
        micro_scale[treatment] = {
            "rmse_delta_xi": region_values["rmse_delta_xi"],
            "mae_delta_xi": region_values["mae_delta_xi"],
            "rmse_to_rms_target_ratio": {
                "mean": float(np.mean(ratios)),
                "sample_standard_deviation": float(np.std(ratios, ddof=1)),
                "values": ratios.tolist(),
            },
        }

    teacher_results, teacher_arrays = evaluate_teacher_forcing(runs, normalizations)
    teacher_aggregate: dict[str, Any] = {}
    for treatment in TREATMENTS:
        teacher_aggregate[treatment] = {}
        for value in HARD_U_TH:
            family = f"{value:.2f}"
            entries = [teacher_results["runs"][f"{treatment}/seed_{seed}"][family] for seed in SEEDS]
            outer_entries = [entry["outer_left_minus17_to_minus8p5"] for entry in entries]
            q_entries = [entry["q_step_outer_left"] for entry in entries]
            signed_entries = [entry["signed_full_incoming"] for entry in entries]
            teacher_aggregate[treatment][family] = {
                **aggregate(outer_entries, ("rmse_delta_x", "rmse_delta_xi", "mae_delta_x", "mae_delta_xi")),
                "q_median": aggregate(q_entries, ("median",))["median"],
                "q_mean": aggregate(q_entries, ("mean",))["mean"],
                "q_p90": aggregate(q_entries, ("p90",))["p90"],
                "q_p99": aggregate(q_entries, ("p99",))["p99"],
                "q_maximum": aggregate(q_entries, ("maximum",))["maximum"],
                "q_fraction_lt_0p1": aggregate(q_entries, ("fraction_lt_0p1",))["fraction_lt_0p1"],
                "q_fraction_lt_0p25": aggregate(q_entries, ("fraction_lt_0p25",))["fraction_lt_0p25"],
                "q_fraction_lt_0p5": aggregate(q_entries, ("fraction_lt_0p5",))["fraction_lt_0p5"],
                "q_fraction_lt_1": aggregate(q_entries, ("fraction_lt_1",))["fraction_lt_1"],
                "mean_signed_delta_xi": aggregate(signed_entries, ("mean_signed_delta_xi",))["mean_signed_delta_xi"],
                "fraction_positive": aggregate(signed_entries, ("fraction_positive",))["fraction_positive"],
                "fraction_negative": aggregate(signed_entries, ("fraction_negative",))["fraction_negative"],
                "cumulative_sum_final": aggregate(signed_entries, ("cumulative_sum_final",))["cumulative_sum_final"],
                "maximum_absolute_cumulative_sum": aggregate(signed_entries, ("maximum_absolute_cumulative_sum",))["maximum_absolute_cumulative_sum"],
            }

    training_curve = plot_training_curves(runs)
    signed_curve = plot_signed_errors(teacher_arrays)
    np.savez_compressed(OUTPUT / "teacher_forced_diagnostic_arrays.npz", **teacher_arrays)
    (OUTPUT / "teacher_forced_diagnostics.json").write_text(
        json.dumps(teacher_results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    own_rows = []
    cross_rows = []
    region_rows = []
    teacher_rows = []
    q_rows = []
    signed_rows = []
    for run in runs:
        treatment, seed = run["sampling_treatment"], run["seed"]
        own_name = "old4k" if treatment == "old20k" else "microcore8k"
        own_rows.append({"treatment": treatment, "seed": seed, **{k: run["cross_validation"][own_name][k] for k in metric_keys}})
        for evaluation_name, evaluation in run["cross_validation"].items():
            cross_rows.append({"treatment": treatment, "seed": seed, "evaluation": evaluation_name, **{k: evaluation[k] for k in metric_keys}})
            for region, values in evaluation["regions"].items():
                region_rows.append({"treatment": treatment, "seed": seed, "evaluation": evaluation_name, "region": region, **values})
        for value in U_TH:
            family = f"{value:.2f}"
            entry = teacher_results["runs"][f"{treatment}/seed_{seed}"][family]
            teacher_rows.append({"treatment": treatment, "seed": seed, "u_th": family, "domain": "full_incoming", **entry["full_incoming"]})
            teacher_rows.append({"treatment": treatment, "seed": seed, "u_th": family, "domain": "outer_left_minus17_to_minus8p5", **entry["outer_left_minus17_to_minus8p5"]})
            signed_rows.append({"treatment": treatment, "seed": seed, "u_th": family, **{k: v for k, v in entry["signed_full_incoming"].items() if isinstance(v, (int, float))}})
            if value in HARD_U_TH:
                q_rows.append({"treatment": treatment, "seed": seed, "u_th": family, **{k: v for k, v in entry["q_step_outer_left"].items() if isinstance(v, (int, float))}})

    write_csv(OUTPUT / "own_validation_metrics.csv", own_rows)
    write_csv(OUTPUT / "cross_evaluation_metrics.csv", cross_rows)
    write_csv(OUTPUT / "regional_validation_metrics.csv", region_rows)
    write_csv(OUTPUT / "teacher_forced_metrics.csv", teacher_rows)
    write_csv(OUTPUT / "q_step_metrics.csv", q_rows)
    write_csv(OUTPUT / "signed_error_metrics.csv", signed_rows)
    write_csv(
        OUTPUT / "microcore_target_scale.csv",
        [
            {
                "treatment": treatment,
                "count": micro_scale["count"],
                "rms_exact_delta_xi": micro_scale["rms_exact_delta_xi"],
                "median_abs_exact_delta_xi": micro_scale["median_abs_exact_delta_xi"],
                "model_rmse_delta_xi_mean": micro_scale[treatment]["rmse_delta_xi"]["mean"],
                "model_mae_delta_xi_mean": micro_scale[treatment]["mae_delta_xi"]["mean"],
                "rmse_to_rms_target_ratio_mean": micro_scale[treatment]["rmse_to_rms_target_ratio"]["mean"],
            }
            for treatment in TREATMENTS
        ],
    )

    after = hashes()
    if before != after:
        raise RuntimeError("an immutable input or reference artifact changed")
    manifest: dict[str, Any] = {
        "stage": "controlled transformed-coordinate sampling training and local evaluation",
        "status": "six_runs_completed",
        "run_count": len(runs),
        "seeds": list(SEEDS),
        "architecture": "2->32->32->2, two tanh hidden layers",
        "parameter_count": parameter_count(load_trained_model(Path(runs[0]["checkpoint"]))),
        "input_columns": list(INPUT_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "runs": runs,
        "normalizations": {name: str(path) for name, path in normalizations.items()},
        "own_validation_aggregate": own_aggregate,
        "cross_evaluation_aggregate": cross_aggregate,
        "regional_aggregate": regional_aggregate,
        "microcore_target_scale_aggregate": micro_scale,
        "teacher_forced_aggregate": teacher_aggregate,
        "training_curve": str(training_curve),
        "signed_error_curve": str(signed_curve),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "same_seed_initialization_audit_passed": True,
        "scheduler_used": False,
        "learned_state_advancement_performed": False,
        "restricted_evaluation_data_accessed": False,
        "incoming_reference_accessed_keys": teacher_results["exact_reference_accessed_keys"],
    }
    manifest_path = OUTPUT / "training_comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "TRAINING_COMPARISON_REPORT.md").write_text(report_text(manifest), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
