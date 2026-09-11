#!/usr/bin/env python3
"""Matched evaluation of the controlled fixed-E0 sensitivity-aware experiment."""

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
from wormhole_sciml.model_a import Normalization, load_trained_model, parameter_count, predict_increments
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi
from wormhole_sciml.stage1_data import file_sha256, load_dataset

try:
    from evaluate_energy_normal_trajectory_postmortem import metric_row, stable_cumulative_sign_onset
    from evaluate_model_a_fixed_E0_energy_normal import attribution_metrics, nonlinear_energy_diagnostic
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
    from scripts.evaluate_energy_normal_trajectory_postmortem import metric_row, stable_cumulative_sign_onset
    from scripts.evaluate_model_a_fixed_E0_energy_normal import attribution_metrics, nonlinear_energy_diagnostic
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
OUTPUT = ROOT / "output" / "model_a_fixed_E0_sensitivity_aware_loss"
TRAINING_MANIFEST = OUTPUT / "sensitivity_aware_training_manifest.json"
DERIVED = OUTPUT / "derived_sensitivity_loss_arrays.npz"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "sensitivity_aware_evaluation_summary.json"
ARRAYS_PATH = OUTPUT / "sensitivity_aware_evaluation_arrays.npz"
ONE_STEP_CSV = OUTPUT / "sensitivity_aware_one_step_subsets.csv"
ENERGY_CSV = OUTPUT / "sensitivity_aware_nonlinear_energy_subsets.csv"
RECURSIVE_CSV = OUTPUT / "sensitivity_aware_recursive_metrics.csv"
ATTRIBUTION_CSV = OUTPUT / "sensitivity_aware_kernel_attribution.csv"
TRAJECTORY_CSV = OUTPUT / "sensitivity_aware_hard_trajectory_metrics.csv"
REPORT_PATH = OUTPUT / "SENSITIVITY_AWARE_LOCAL_LOSS_REPORT.md"

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
ALIGNMENT_DIR = ROOT / "output" / "model_a_energy_gradient_alignment"
ALIGNMENT_ARRAYS = ALIGNMENT_DIR / "energy_gradient_alignment_arrays.npz"
ALIGNMENT_SUMMARY = ALIGNMENT_DIR / "energy_gradient_alignment_summary.json"
GATE = ROOT / "output" / "model_a_sensitivity_map_physics_gate"
GATE_ARRAYS = GATE / "sensitivity_map_arrays.npz"
GATE_SUMMARY = GATE / "sensitivity_map_summary.json"

SEEDS = (101, 202, 303)
TREATMENTS = ("fixed_E0_baseline", "sensitivity_aware")
TITLES = {"fixed_E0_baseline": "frozen fixed E0", "sensitivity_aware": "sensitivity-aware"}
HARD = THROAT_VELOCITIES[:3]
FAR = (-17.0, -8.5)
INPUT_COLUMNS = ("x", "xi", "E0")
TARGET_COLUMNS = ("delta_x", "delta_xi")
NORMALIZATION_SOURCE = "outer_microcore40k_train_x_xi_energy_input_only"


def fkey(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def hashes(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required protected artifact(s) missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def protected_paths(training: dict[str, Any], baseline: dict[str, Any]) -> list[Path]:
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
        ALIGNMENT_ARRAYS,
        ALIGNMENT_SUMMARY,
        GATE_ARRAYS,
        GATE_SUMMARY,
        TRAINING_MANIFEST,
        DERIVED,
    ]
    for row in baseline["runs"] + training["runs"]:
        paths.extend((Path(row["checkpoint"]), Path(row["history"]), Path(row["checkpoint"]).parent / "metadata.json"))
    return paths


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": None, "mae": None, "rmse": None, "median": None, "p95_absolute": None, "p99_absolute": None, "maximum_absolute": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "mae": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "median": float(np.median(array)),
        "p95_absolute": float(np.quantile(np.abs(array), 0.95)),
        "p99_absolute": float(np.quantile(np.abs(array), 0.99)),
        "maximum_absolute": float(np.max(np.abs(array))),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validation_masks(
    validation: dict[str, np.ndarray], eligible: np.ndarray, weight: np.ndarray, gate: dict[str, Any]
) -> dict[str, np.ndarray]:
    thresholds = gate["normalized_abs_S"]["distribution"]
    return {
        "all": np.ones(eligible.size, dtype=bool),
        "eligible": eligible,
        "noneligible": ~eligible,
        "eligible_micro_core": eligible & (np.asarray(validation["outer_xi_stratum"]) == 0),
        "eligible_high_S_top10_train_threshold": eligible & (weight >= float(thresholds["p90"])),
        "eligible_high_S_top1_train_threshold": eligible & (weight >= float(thresholds["p99"])),
        "eligible_high_S_top0p1_train_threshold": eligible & (weight >= float(thresholds["p99_9"])),
    }


def local_metrics(
    exact: np.ndarray,
    prediction: np.ndarray,
    normalization: Normalization,
    normals: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    physical_error = prediction - exact
    error = physical_error / normalization.target_std
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    e_perp = np.sum(error * normals, axis=1)
    e_parallel = np.sum(error * tangent, axis=1)
    selected = np.asarray(mask, dtype=bool)
    result = {
        "count": int(selected.sum()),
        "standardized_mse": float(np.mean(error[selected] ** 2)),
        "L_base": float(0.5 * np.mean(np.sum(error[selected] ** 2, axis=1))),
        "L_sensitivity": float(0.5 * np.mean(weights[selected] * e_perp[selected] ** 2)),
        "rmse_delta_x": float(np.sqrt(np.mean(physical_error[selected, 0] ** 2))),
        "rmse_delta_xi": float(np.sqrt(np.mean(physical_error[selected, 1] ** 2))),
        "mae_delta_x": float(np.mean(np.abs(physical_error[selected, 0]))),
        "mae_delta_xi": float(np.mean(np.abs(physical_error[selected, 1]))),
        "rms_e_perp": float(np.sqrt(np.mean(e_perp[selected] ** 2))),
        "mae_e_perp": float(np.mean(np.abs(e_perp[selected]))),
        "rms_e_parallel": float(np.sqrt(np.mean(e_parallel[selected] ** 2))),
        "mae_e_parallel": float(np.mean(np.abs(e_parallel[selected]))),
    }
    return result, {
        "prediction": prediction,
        "physical_error": physical_error,
        "standardized_error": error,
        "e_perp": e_perp,
        "e_parallel": e_parallel,
    }


def aggregate_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in rows[0]:
        if key == "count":
            result[key] = rows[0][key]
            continue
        values = np.asarray([float(row[key]) for row in rows])
        result[key] = {
            "mean": float(np.mean(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
        }
    return result


def nonlinear_subsets(
    prediction: np.ndarray, validation: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _, delta, valid = nonlinear_energy_diagnostic(prediction, validation)
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        use = mask & valid
        result[name] = {
            "count": int(mask.sum()),
            "valid_count": int(use.sum()),
            "invalid_or_nonphysical_count": int(np.sum(mask & ~valid)),
            "delta_E": distribution(delta[use]),
        }
    return result, delta, valid


def training_stability(training: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": {}}
    for run in training["runs"]:
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        finite = all(
            np.isfinite(
                [
                    row["training_base_standardized_mse"],
                    row["training_sensitivity_loss"],
                    row["training_total_loss"],
                    row["validation_standardized_mse"],
                    row["validation_sensitivity_loss"],
                    row["maximum_gradient_norm_before_step"],
                ]
            ).all()
            for row in history
        )
        selected = history[int(run["best_epoch"]) - 1]
        result["runs"][str(run["seed"])] = {
            "all_recorded_values_finite": bool(finite),
            "epoch_count": len(history),
            "maximum_gradient_norm_before_step": float(max(row["maximum_gradient_norm_before_step"] for row in history)),
            "maximum_batch_sensitivity_loss": float(max(row["maximum_batch_sensitivity_loss"] for row in history)),
            "maximum_batch_total_loss": float(max(row["maximum_batch_total_loss"] for row in history)),
            "maximum_weight_batch_was_maximum_sensitivity_batch_epoch_fraction": float(np.mean([row["maximum_weight_batch_is_maximum_sensitivity_batch"] for row in history])),
            "selected_epoch_top_weight_contribution": selected["training_sensitivity_loss_top_weight_contribution"],
            "selected_epoch_maximum_weight_batch_sensitivity_loss": selected["maximum_weight_batch_sensitivity_loss"],
            "selected_epoch_maximum_batch_sensitivity_loss": selected["maximum_batch_sensitivity_loss"],
            "selected_epoch_validation_base": selected["validation_standardized_mse"],
            "selected_epoch_validation_sensitivity": selected["validation_sensitivity_loss"],
            "stopping_epoch_validation_sensitivity": history[-1]["validation_sensitivity_loss"],
            "minimum_validation_sensitivity_epoch": int(np.argmin([row["validation_sensitivity_loss"] for row in history]) + 1),
            "minimum_validation_sensitivity": float(min(row["validation_sensitivity_loss"] for row in history)),
        }
    result["all_runs_numerically_finite"] = all(row["all_recorded_values_finite"] for row in result["runs"].values())
    return result


def recursive_evaluation(
    models: dict[int, Any], normalization: Normalization, arrays: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    baseline_cases = {float(case["u_th"]): case for case in baseline_summary["cases"]}
    family_source = {float(case["u_th"]): case for case in json.loads(FAMILY_SUMMARY.read_text(encoding="utf-8"))["cases"]}
    cases: list[dict[str, Any]] = []
    with np.load(BASELINE_ARRAYS, allow_pickle=False) as base_arrays, np.load(ORBIT_SPACING, allow_pickle=False) as spacing:
        for u_th in THROAT_VELOCITIES:
            key = fkey(u_th)
            base_case = baseline_cases[u_th]
            exact_full = np.asarray(base_arrays[f"{key}__exact_full_state"])
            exact_throat = np.asarray(base_arrays[f"{key}__exact_throat_started_state"])
            E0_full = float(conserved_energy(exact_full[0, 0], exact_full[0, 1], *experiment_parameters()))
            E0_throat = float(conserved_energy(exact_throat[0, 0], exact_throat[0, 1], *experiment_parameters()))
            if not np.isclose(E0_full, float(family_source[u_th]["energy"]), rtol=0.0, atol=2e-12):
                raise RuntimeError("frozen family energy mismatch")
            case: dict[str, Any] = {
                "u_th": float(u_th),
                "E0_from_full_initial_state": E0_full,
                "E0_from_throat_initial_state": E0_throat,
                "full_traversal": {"fixed_E0_baseline": copy.deepcopy(base_case["full_traversal"]["fixed_E0"])},
                "throat_started_outgoing": {"fixed_E0_baseline": copy.deepcopy(base_case["throat_started_outgoing"]["fixed_E0"])},
                "hard_recursive_xi_diagnostic": {},
            }
            for mode, section, exact_state, E0 in (
                ("full", "full_traversal", exact_full, E0_full),
                ("throat", "throat_started_outgoing", exact_throat, E0_throat),
            ):
                rows: dict[str, Any] = {}
                initial = exact_xi_path(exact_state)[0]
                for seed in SEEDS:
                    coordinates, physical, margins, drift, row = energy_recursive_rollout(
                        models[seed], normalization, initial, E0, u_th
                    )
                    rows[str(seed)] = row
                    prefix = f"{key}__{mode}__sensitivity_aware__seed_{seed}"
                    arrays[f"{prefix}__coordinates"] = coordinates
                    arrays[f"{prefix}__physical_state"] = physical
                    arrays[f"{prefix}__C"] = margins
                    arrays[f"{prefix}__relative_energy_drift"] = drift
                    if mode == "full" and u_th in HARD:
                        diagnostic_arrays, diagnostic_summary = incoming_xi_diagnostic(
                            coordinates,
                            margins,
                            exact_full,
                            np.asarray(spacing[f"{key}__x"]),
                            np.asarray(spacing[f"{key}__orbit_spacing"]),
                        )
                        diagnostic_summary["fraction_R_drift_lt_0p5"] = float(np.mean(diagnostic_arrays["R_drift"] < 0.5))
                        for suffix, value in diagnostic_arrays.items():
                            arrays[f"{prefix}__incoming__{suffix}"] = value
                        case["hard_recursive_xi_diagnostic"].setdefault("sensitivity_aware", {})[str(seed)] = diagnostic_summary
                case[section]["sensitivity_aware"] = {"seeds": rows, "aggregate": family_aggregate(rows)}
            if u_th in HARD:
                case["hard_recursive_xi_diagnostic"]["fixed_E0_baseline"] = copy.deepcopy(base_case["hard_recursive_xi_diagnostic"]["fixed_E0"])
            cases.append(case)
            print(f"completed sensitivity-aware recursion u_th={u_th:.2f}", flush=True)

    aggregate: dict[str, Any] = {}
    for mode in ("full_traversal", "throat_started_outgoing"):
        aggregate[mode] = {}
        for treatment in TREATMENTS:
            rows = [case[mode][treatment]["seeds"][str(seed)] for case in cases for seed in SEEDS]
            errors = [row["absolute_throat_u_error"] for row in rows if row["absolute_throat_u_error"] is not None]
            aggregate[mode][treatment] = {
                "rollout_count": len(rows),
                "successful_full_traversal_count": int(sum(row["reached_x_plus_17"] for row in rows)),
                "physical_exit_count": int(sum(row["physical_exit"] for row in rows)),
                "guard_or_horizon_termination_count": int(sum(row["maximum_step_guard"] for row in rows)),
                "nonfinite_termination_count": int(sum(row["nonfinite"] for row in rows)),
                "throat_crossing_count": int(sum(row["interpolated_throat_u"] is not None for row in rows)),
                "mean_absolute_throat_u_error_successful_crossings": float(np.mean(errors)) if errors else None,
            }
    return cases, aggregate


def teacher_forced_hard_diagnostics(
    all_models: dict[str, dict[int, Any]],
    normalization: Normalization,
    recursive_cases: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sensitivity_summary = json.loads(SENSITIVITY_SUMMARY.read_text(encoding="utf-8"))
    alignment_summary = json.loads(ALIGNMENT_SUMMARY.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    attribution: dict[str, Any] = {}
    signs: list[dict[str, Any]] = []
    wormhole, spiral = experiment_parameters()
    with np.load(EXACT_REFERENCE, allow_pickle=False) as exact, np.load(
        SENSITIVITY_ARRAYS, allow_pickle=False
    ) as sensitivity, np.load(ALIGNMENT_ARRAYS, allow_pickle=False) as alignment:
        for family in HARD:
            key = fkey(family)
            family_text = f"{family:.2f}"
            valid_kernel = np.asarray(sensitivity[f"{key}__valid_kernel_mask"], dtype=bool)
            state_all = np.column_stack((exact[f"{key}__exact_state"][:, 0], exact[f"{key}__exact_xi"]))
            target_all = np.column_stack((sensitivity[f"{key}__exact_delta_x_all"], sensitivity[f"{key}__exact_delta_xi_all"]))
            state = state_all[valid_kernel]
            target = target_all[valid_kernel]
            x = state[:, 0]
            E0 = float(sensitivity_summary["families"][family_text]["E0"])
            features = np.column_stack((state, np.full(x.size, E0)))
            gE = np.asarray(alignment[f"{key}__g_E"])
            nE = np.asarray(alignment[f"{key}__n_E"])
            K = np.column_stack((alignment[f"{key}__K_x"], alignment[f"{key}__K_xi"]))
            if not np.array_equal(x, alignment[f"{key}__diagnostic_x"]):
                raise RuntimeError("hard diagnostic x mismatch")
            if not np.allclose(state + target, np.column_stack((alignment[f"{key}__post_x"], alignment[f"{key}__post_xi"])), rtol=0.0, atol=1e-14):
                raise RuntimeError("hard post-step state mismatch")
            tangent = np.column_stack((-nE[:, 1], nE[:, 0]))
            far = (x >= FAR[0]) & (x < FAR[1])
            du_dE = float(alignment_summary["families"][family_text]["proportionality"]["theoretical_du_th_dE"])
            arrays[f"{key}__diagnostic_x"] = x
            arrays[f"{key}__n_E"] = nE
            arrays[f"{key}__g_E"] = gE
            arrays[f"{key}__K"] = K
            attribution[family_text] = {treatment: {} for treatment in TREATMENTS}
            for treatment in TREATMENTS:
                for seed in SEEDS:
                    prediction = predict_increments(all_models[treatment][seed], features, normalization)
                    raw = prediction - target
                    standard = raw / normalization.target_std
                    e_perp = np.sum(standard * nE, axis=1)
                    e_parallel = np.sum(standard * tangent, axis=1)
                    delta_lin = np.sum(standard * gE, axis=1)
                    predicted_x = state[:, 0] + prediction[:, 0]
                    predicted_xi = state[:, 1] + prediction[:, 1]
                    _, predicted_u = state_from_xi(predicted_x, predicted_xi, wormhole, spiral)
                    margin = timelike_margin(predicted_x, predicted_u, wormhole, spiral)
                    physical = np.isfinite(predicted_x) & np.isfinite(predicted_xi) & np.isfinite(predicted_u) & np.isfinite(margin) & (margin > 0)
                    delta_nl = np.full(x.size, np.nan)
                    delta_nl[physical] = conserved_energy(predicted_x[physical], predicted_u[physical], wormhole, spiral) - E0
                    abs_discrepancy = np.abs(delta_nl - delta_lin)
                    relative_discrepancy = abs_discrepancy / np.maximum.reduce((np.abs(delta_nl), np.abs(delta_lin), np.full(x.size, 1e-14)))
                    values = {
                        "e_perp": e_perp,
                        "e_parallel": e_parallel,
                        "delta_E_lin": delta_lin,
                        "delta_E_NL": delta_nl,
                        "linear_nonlinear_abs_discrepancy": abs_discrepancy,
                        "linear_nonlinear_relative_discrepancy": relative_discrepancy,
                    }
                    rows.append(metric_row(family, treatment, seed, "complete_incoming", np.ones(x.size, dtype=bool), values))
                    rows.append(metric_row(family, treatment, seed, "far_upstream", far, values))
                    prefix = f"{key}__{treatment}__seed_{seed}"
                    arrays[f"{prefix}__e_perp"] = e_perp
                    arrays[f"{prefix}__e_parallel"] = e_parallel
                    arrays[f"{prefix}__delta_E_lin"] = delta_lin
                    arrays[f"{prefix}__delta_E_NL"] = delta_nl
                    A_x = K[:, 0] * raw[:, 0]
                    A_xi = K[:, 1] * raw[:, 1]
                    arrays[f"{prefix}__A_x"] = A_x
                    arrays[f"{prefix}__A_xi"] = A_xi
                    arrays[f"{prefix}__A_total"] = A_x + A_xi
                    attribution[family_text][treatment][str(seed)] = attribution_metrics(x, A_x, A_xi)
                    sign, onset, total = stable_cumulative_sign_onset(x, A_x + A_xi)
                    recursive = next(case for case in recursive_cases if case["u_th"] == family)["full_traversal"][treatment]["seeds"][str(seed)]
                    signed_error = recursive["signed_throat_u_error"]
                    signs.append(
                        {
                            "family_u_th": family,
                            "treatment": treatment,
                            "seed": seed,
                            "linearized_total_sign": sign,
                            "persistent_sign_onset_x": onset,
                            "sum_A_total": total,
                            "recursive_signed_throat_error": signed_error,
                            "sign_matches_recursive": bool(signed_error is not None and np.sign(total) == np.sign(signed_error)),
                            "du_th_dE": du_dE,
                        }
                    )
    for family in HARD:
        family_text = f"{family:.2f}"
        case = next(case for case in recursive_cases if case["u_th"] == family)
        for treatment in TREATMENTS:
            linear_order = sorted(SEEDS, key=lambda seed: abs(attribution[family_text][treatment][str(seed)]["sum_A_total"]))
            recursive_order = sorted(
                SEEDS,
                key=lambda seed: float("inf") if case["full_traversal"][treatment]["seeds"][str(seed)]["signed_throat_u_error"] is None else abs(case["full_traversal"][treatment]["seeds"][str(seed)]["signed_throat_u_error"]),
            )
            for seed in SEEDS:
                attribution[family_text][treatment][str(seed)]["family_seed_order_linearized"] = linear_order
                attribution[family_text][treatment][str(seed)]["family_seed_order_recursive"] = recursive_order
                attribution[family_text][treatment][str(seed)]["family_seed_order_matches"] = linear_order == recursive_order
    return rows, attribution, signs


def plot_training(training: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), sharey=True, constrained_layout=True)
    for axis, run in zip(axes, training["runs"], strict=True):
        history = json.loads(Path(run["history"]).read_text(encoding="utf-8"))
        epoch = [row["epoch"] for row in history]
        axis.plot(epoch, [row["validation_standardized_mse"] for row in history], label=r"validation $L_{base}$")
        axis.plot(epoch, [row["validation_sensitivity_loss"] for row in history], label=r"validation $L_{sens}$")
        axis.plot(epoch, [row["validation_total_loss"] for row in history], label=r"validation $L_{SA}$")
        axis.axvline(run["best_epoch"], color="0.3", ls="--", lw=0.9, label="selected")
        axis.set(title=f"seed {run['seed']}", xlabel="epoch", yscale="log")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("loss")
    figure.suptitle("Sensitivity-aware training diagnostics")
    figure.savefig(FIGURES / "sensitivity_aware_training_curves.png", dpi=185)
    plt.close(figure)


def plot_local_subsets(one_step: dict[str, Any]) -> None:
    subsets = ("all", "eligible", "noneligible", "eligible_micro_core", "eligible_high_S_top1_train_threshold")
    x = np.arange(len(subsets))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.7), constrained_layout=True)
    for axis, metric, title in ((axes[0], "rms_e_perp", "energy-normal RMS"), (axes[1], "rms_e_parallel", "energy-tangent RMS")):
        for offset, treatment in ((-width / 2, TREATMENTS[0]), (width / 2, TREATMENTS[1])):
            values = [one_step[treatment]["aggregate"][subset][metric]["mean"] for subset in subsets]
            axis.bar(x + offset, values, width, label=TITLES[treatment])
        axis.set(xticks=x, xticklabels=["all", "eligible", "noneligible", "eligible\nmicro-core", "eligible\nhigh-S top1"], title=title, yscale="log")
        axis.grid(alpha=0.2, axis="y", which="both")
        axis.legend(fontsize=8)
    figure.savefig(FIGURES / "one_step_subset_error_allocation.png", dpi=185)
    plt.close(figure)


def plot_throat_errors(cases: list[dict[str, Any]]) -> None:
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
    figure.savefig(FIGURES / "seven_family_throat_error_comparison.png", dpi=185)
    plt.close(figure)


def plot_hard_diagnostics(arrays: dict[str, np.ndarray]) -> None:
    key = fkey(0.05)
    x = arrays[f"{key}__diagnostic_x"]
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True, sharex=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            prefix = f"{key}__{treatment}__seed_{seed}"
            total = arrays[f"{prefix}__A_total"]
            axes[row, 0].plot(x, total, color=SEED_COLORS[seed], lw=1.2, label=f"seed {seed}")
            axes[row, 1].plot(x, np.cumsum(total), color=SEED_COLORS[seed], lw=1.2, label=f"seed {seed}")
        for axis in axes[row]:
            axis.axvspan(FAR[0], FAR[1], color="0.7", alpha=0.18)
            axis.axhline(0, color="0.3", lw=0.8)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        axes[row, 0].set(title=f"{TITLES[treatment]} · local attribution", ylabel=r"$A_x+A_\xi$")
        axes[row, 1].set(title=f"{TITLES[treatment]} · cumulative attribution", ylabel="cumulative sum")
    axes[-1, 0].set_xlabel("exact incoming x")
    axes[-1, 1].set_xlabel("exact incoming x")
    figure.suptitle(r"Exact-kernel attribution · $u_{th}=0.05$")
    figure.savefig(FIGURES / "u_th_0p05_kernel_attribution.png", dpi=185)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 7.5), constrained_layout=True, sharex=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            prefix = f"{key}__{treatment}__seed_{seed}"
            axes[row, 0].plot(x, arrays[f"{prefix}__e_perp"], color=SEED_COLORS[seed], lw=1.2, label=f"seed {seed}")
            axes[row, 1].plot(x, arrays[f"{prefix}__delta_E_NL"], color=SEED_COLORS[seed], lw=1.2)
        for axis in axes[row]:
            axis.axvspan(FAR[0], FAR[1], color="0.7", alpha=0.18)
            axis.axhline(0, color="0.3", lw=0.8)
            axis.grid(alpha=0.2)
        axes[row, 0].set_ylabel(TITLES[treatment])
    axes[0, 0].set_title(r"standardized $e_\perp$")
    axes[0, 1].set_title(r"true nonlinear $\delta E$")
    axes[-1, 0].set_xlabel("exact incoming x")
    axes[-1, 1].set_xlabel("exact incoming x")
    axes[0, 0].legend(fontsize=8, ncol=3)
    figure.suptitle(r"Teacher-forced hard trajectory · $u_{th}=0.05$")
    figure.savefig(FIGURES / "u_th_0p05_teacher_forced_energy_diagnostics.png", dpi=185)
    plt.close(figure)


def report_text(summary: dict[str, Any]) -> str:
    one = summary["one_step_validation"]
    full = summary["recursive_aggregate"]["full_traversal"]
    throat = summary["recursive_aggregate"]["throat_started_outgoing"]
    trajectory = summary["trajectory_conditioned_changes"]
    lines = [
        "# Controlled Sensitivity-Aware Local-Loss Training Experiment",
        "",
        "## Outcome",
        "",
        "Exactly three models were trained with the frozen fixed-E0 pipeline. The only intervention was `0.5 * mean(I*w_absS*(n_E·e)^2)` with lambda_S=1, using the saved future-crossing classification and uncapped `|S|/0.3526914868127516`. Checkpoint selection remained ordinary validation standardized MSE.",
        "",
        "## Implementation and optimization",
        "",
        f"The preflight reproduced 10,352/40,000 eligible training rows and 2,122/8,000 eligible validation rows, including the 10,081/271 directed training split. Energy normals matched the completed alignment gate to {summary['training']['preflight']['completed_alignment_gate_maximum_absolute_n_E_difference']:.1e}. The maximum raw normalized training weight was {summary['training']['preflight']['train']['eligible_weight_maximum']:.6g}.",
        "",
        "| seed | best / stop epoch | val base MSE | val sensitivity term | max gradient norm | selected top 0.1/1/5% sensitivity-loss shares |",
        "|--:|:--|--:|--:|--:|:--|",
    ]
    for run in summary["training"]["runs"]:
        stability = summary["optimization_stability"]["runs"][str(run["seed"])]
        shares = stability["selected_epoch_top_weight_contribution"]
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} / {run['stopping_epoch']} | {run['selected_checkpoint_validation']['standardized_mse']:.6e} | {run['selected_checkpoint_validation']['sensitivity_loss']:.6e} | {stability['maximum_gradient_norm_before_step']:.4g} | {shares['top_0.1_percent']:.3f} / {shares['top_1_percent']:.3f} / {shares['top_5_percent']:.3f} |"
        )
    lines.extend(["", "## One-step validation: seed means", "", "| subset | metric | baseline | sensitivity-aware | change |", "|:--|:--|--:|--:|--:|"])
    for subset in ("all", "eligible", "noneligible", "eligible_micro_core", "eligible_high_S_top1_train_threshold"):
        for metric in ("standardized_mse", "rms_e_perp", "rms_e_parallel"):
            old = one["fixed_E0_baseline"]["aggregate"][subset][metric]["mean"]
            new = one["sensitivity_aware"]["aggregate"][subset][metric]["mean"]
            lines.append(f"| {subset} | {metric} | {old:.6e} | {new:.6e} | {pct(new, old):+.1f}% |")
    lines.extend(["", "## Seven-family recursive evaluation", "", "| u_th | baseline throat u (101/202/303) | sensitivity-aware throat u (101/202/303) | baseline / new MAE | new reach-exit-guard |", "|--:|:--|:--|:--|:--|"])
    for case in summary["cases"]:
        values: dict[str, str] = {}
        for treatment in TREATMENTS:
            values[treatment] = "/".join(
                "—" if case["full_traversal"][treatment]["seeds"][str(seed)]["interpolated_throat_u"] is None else f"{case['full_traversal'][treatment]['seeds'][str(seed)]['interpolated_throat_u']:.5f}"
                for seed in SEEDS
            )
        old = case["full_traversal"]["fixed_E0_baseline"]["aggregate"]
        new = case["full_traversal"]["sensitivity_aware"]["aggregate"]
        lines.append(f"| {case['u_th']:.2f} | {values['fixed_E0_baseline']} | {values['sensitivity_aware']} | {old['mean_absolute_throat_u_error_successful_crossings']:.6g} / {new['mean_absolute_throat_u_error_successful_crossings']:.6g} | {new['successful_full_traversal_count']}-{new['physical_exit_count']}-{new['guard_or_horizon_termination_count']} |")
    lines.extend([
        "",
        f"Across the full suite, sensitivity-aware reach/exit/guard counts were {full['sensitivity_aware']['successful_full_traversal_count']}/{full['sensitivity_aware']['physical_exit_count']}/{full['sensitivity_aware']['guard_or_horizon_termination_count']}. Throat-started controls were {throat['sensitivity_aware']['successful_full_traversal_count']}/21 successful with {throat['sensitivity_aware']['physical_exit_count']} exits and {throat['sensitivity_aware']['guard_or_horizon_termination_count']} guards.",
        "",
        "## Hard-family teacher-forced changes (three-seed means)",
        "",
        "| u_th | scope | RMS e_perp change | nonlinear energy MAE change | total absolute A change |",
        "|--:|:--|--:|--:|--:|",
    ])
    for row in trajectory:
        lines.append(f"| {row['family_u_th']:.2f} | {row['scope']} | {row['rms_e_perp_change_percent']:+.1f}% | {row['mean_abs_delta_E_NL_change_percent']:+.1f}% | {row['sum_absolute_A_total_change_percent']:+.1f}% |")
    conclusion = summary["scientific_conclusion"]
    lines.extend([
        "",
        "## Scientific answers",
        "",
        f"1. **Optimization:** {conclusion['A_optimization']}",
        "",
        f"2. **Loss allocation:** {conclusion['B_loss_allocation']}",
        "",
        f"3. **Hard upstream region:** {conclusion['C_hard_upstream']}",
        "",
        f"4. **Recursive physics:** {conclusion['D_recursive_physics']}",
        "",
        f"5. **Robustness/easy controls:** {conclusion['E_robustness']}",
        "",
        f"6. **Noneligible dynamics:** {conclusion['F_noneligible']}",
        "",
        f"7. **Seed dependence/capacity:** {conclusion['G_seed_capacity']}",
        "",
        f"8. **Next scientific decision:** {conclusion['H_next_decision']}",
        "",
        "## Artifacts and stop",
        "",
        "Detailed per-seed/subset, nonlinear-energy, recursive, attribution, and trajectory-conditioned tables are the CSV files beside this report. Figures are under `figures/`; numerical arrays and the complete JSON summary are also preserved.",
        "",
        "All protected hashes were unchanged. Evaluation stops here: no coefficient sweep, clipping, alternate weighting, multi-step training, architecture change, or additional model training was performed.",
    ])
    return "\n".join(lines) + "\n"


def scientific_conclusion(summary: dict[str, Any]) -> dict[str, str]:
    one = summary["one_step_validation"]
    full = summary["recursive_aggregate"]["full_traversal"]
    throat = summary["recursive_aggregate"]["throat_started_outgoing"]
    trajectory = summary["trajectory_conditioned_changes"]
    high_old = one["fixed_E0_baseline"]["aggregate"]["eligible_high_S_top1_train_threshold"]["rms_e_perp"]["mean"]
    high_new = one["sensitivity_aware"]["aggregate"]["eligible_high_S_top1_train_threshold"]["rms_e_perp"]["mean"]
    eligible_old = one["fixed_E0_baseline"]["aggregate"]["eligible"]["rms_e_perp"]["mean"]
    eligible_new = one["sensitivity_aware"]["aggregate"]["eligible"]["rms_e_perp"]["mean"]
    non_old = one["fixed_E0_baseline"]["aggregate"]["noneligible"]["standardized_mse"]["mean"]
    non_new = one["sensitivity_aware"]["aggregate"]["noneligible"]["standardized_mse"]["mean"]
    far005 = next(row for row in trajectory if row["family_u_th"] == 0.05 and row["scope"] == "far_upstream")
    hard_changes = []
    for family in HARD:
        case = next(case for case in summary["cases"] if case["u_th"] == family)
        old = case["full_traversal"]["fixed_E0_baseline"]["aggregate"]["mean_absolute_throat_u_error_successful_crossings"]
        new = case["full_traversal"]["sensitivity_aware"]["aggregate"]["mean_absolute_throat_u_error_successful_crossings"]
        hard_changes.append((family, pct(new, old)))
    baseline_seed_values = [one["fixed_E0_baseline"]["individual"][str(seed)]["eligible"]["rms_e_perp"] for seed in SEEDS]
    new_seed_values = [one["sensitivity_aware"]["individual"][str(seed)]["eligible"]["rms_e_perp"] for seed in SEEDS]
    baseline_cv = float(np.std(baseline_seed_values, ddof=1) / np.mean(baseline_seed_values))
    new_cv = float(np.std(new_seed_values, ddof=1) / np.mean(new_seed_values))
    stable = summary["optimization_stability"]["all_runs_numerically_finite"]
    robust = full["sensitivity_aware"]["successful_full_traversal_count"] == 21 and full["sensitivity_aware"]["physical_exit_count"] == 0 and full["sensitivity_aware"]["guard_or_horizon_termination_count"] == 0
    hard_improved = all(change < 0 for _, change in hard_changes)
    far_improved = far005["rms_e_perp_change_percent"] < 0 and far005["mean_abs_delta_E_NL_change_percent"] < 0 and far005["sum_absolute_A_total_change_percent"] < 0
    if not stable:
        decision = "The raw tail was not numerically trainable; a later bounded/tempered-weight investigation is supported."
    elif not far_improved or not hard_improved:
        decision = "The targeted local formulation did not consistently solve the decisive hard-family physics. Do not adopt it as-is; the evidence supports moving to a separately designed short-horizon/multi-step milestone before any architecture claim."
    else:
        decision = "Keep normalized-|S| sensitivity-aware loss as the supported next baseline; coefficient tuning remains a separate experiment."
    return {
        "A_optimization": f"{'Yes' if stable else 'No'}. All three histories remained finite without clipping; the largest recorded gradient norm was {max(row['maximum_gradient_norm_before_step'] for row in summary['optimization_stability']['runs'].values()):.4g}.",
        "B_loss_allocation": f"Eligible RMS e_perp changed {pct(eligible_new, eligible_old):+.1f}%, and the training-p99 high-sensitivity validation subset changed {pct(high_new, high_old):+.1f}%.",
        "C_hard_upstream": f"For u_th=0.05 far upstream, RMS e_perp changed {far005['rms_e_perp_change_percent']:+.1f}%, nonlinear energy MAE {far005['mean_abs_delta_E_NL_change_percent']:+.1f}%, and total absolute exact-kernel attribution {far005['sum_absolute_A_total_change_percent']:+.1f}%.",
        "D_recursive_physics": "Hard-family mean absolute throat-error changes were " + ", ".join(f"u_th={family:.2f}: {change:+.1f}%" for family, change in hard_changes) + ".",
        "E_robustness": f"{'Preserved' if robust else 'Not preserved'}: full reach/exit/guard={full['sensitivity_aware']['successful_full_traversal_count']}/{full['sensitivity_aware']['physical_exit_count']}/{full['sensitivity_aware']['guard_or_horizon_termination_count']}; throat-started reach/exit/guard={throat['sensitivity_aware']['successful_full_traversal_count']}/{throat['sensitivity_aware']['physical_exit_count']}/{throat['sensitivity_aware']['guard_or_horizon_termination_count']}.",
        "F_noneligible": f"Noneligible standardized MSE changed {pct(non_new, non_old):+.1f}%; this directly measures spillover from applying the extra term only to the eligible 25.88% training subset.",
        "G_seed_capacity": f"Eligible e_perp seed CV changed from {baseline_cv:.3f} to {new_cv:.3f}. Tradeoffs or variability are capacity/optimization cues only; this experiment does not establish an architecture limitation.",
        "H_next_decision": decision,
    }


def main() -> None:
    for path in (SUMMARY_PATH, ARRAYS_PATH, ONE_STEP_CSV, ENERGY_CSV, RECURSIVE_CSV, ATTRIBUTION_CSV, TRAJECTORY_CSV, REPORT_PATH):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evaluation artifact {path}")
    training = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_TRAINING.read_text(encoding="utf-8"))
    gate_summary = json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))
    protected = protected_paths(training, baseline)
    before = hashes(protected)
    normalization = Normalization.from_stage1(NORMALIZATION_PATH, INPUT_COLUMNS, TARGET_COLUMNS, NORMALIZATION_SOURCE)
    models = {int(row["seed"]): load_trained_model(Path(row["checkpoint"])) for row in training["runs"]}
    baseline_models = {int(row["seed"]): load_trained_model(Path(row["checkpoint"])) for row in baseline["runs"]}
    if set(models) != set(SEEDS) or set(baseline_models) != set(SEEDS):
        raise RuntimeError("expected exactly three checkpoints per treatment")
    if any(parameter_count(model) != 1250 for model in [*models.values(), *baseline_models.values()]):
        raise RuntimeError("a checkpoint is not 3->32->32->2")
    validation = load_dataset(VALIDATION_DATA)
    exact_target = np.column_stack(tuple(validation[name] for name in TARGET_COLUMNS))
    features = np.column_stack(tuple(validation[name] for name in INPUT_COLUMNS))
    with np.load(DERIVED, allow_pickle=False) as derived:
        if not (
            np.array_equal(derived["validation_source_row_index"], validation["source_row_index"])
            and np.array_equal(derived["validation_x_next"], validation["x_next"])
            and np.array_equal(derived["validation_xi_next"], validation["xi_next"])
        ):
            raise RuntimeError("derived validation arrays no longer correspond to frozen rows")
        normals = np.asarray(derived["validation_n_E"], dtype=np.float64)
        eligible = np.asarray(derived["validation_eligible"], dtype=bool)
        weights = np.asarray(derived["validation_sensitivity_weight"], dtype=np.float64)
    masks = validation_masks(validation, eligible, weights, gate_summary)

    arrays: dict[str, np.ndarray] = {"validation_eligible": eligible, "validation_sensitivity_weight": weights}
    one_step: dict[str, Any] = {}
    nonlinear: dict[str, Any] = {}
    one_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    all_models = {"fixed_E0_baseline": baseline_models, "sensitivity_aware": models}
    for treatment, treatment_models in all_models.items():
        individual: dict[str, Any] = {}
        nonlinear_individual: dict[str, Any] = {}
        for seed in SEEDS:
            prediction = predict_increments(treatment_models[seed], features, normalization)
            individual[str(seed)] = {}
            values_for_arrays: dict[str, np.ndarray] | None = None
            for subset, mask in masks.items():
                metrics, values = local_metrics(exact_target, prediction, normalization, normals, weights, mask)
                individual[str(seed)][subset] = metrics
                one_rows.append({"treatment": treatment, "seed": seed, "subset": subset, **metrics})
                values_for_arrays = values
            assert values_for_arrays is not None
            for name, value in values_for_arrays.items():
                arrays[f"validation__{treatment}__seed_{seed}__{name}"] = value
            nonlinear_metrics, delta_E, physical = nonlinear_subsets(prediction, validation, masks)
            nonlinear_individual[str(seed)] = nonlinear_metrics
            arrays[f"validation__{treatment}__seed_{seed}__delta_E_NL"] = delta_E
            arrays[f"validation__{treatment}__seed_{seed}__energy_valid"] = physical
            for subset, metrics in nonlinear_metrics.items():
                energy_rows.append({"treatment": treatment, "seed": seed, "subset": subset, **{key: value for key, value in metrics.items() if key != "delta_E"}, **{f"delta_E_{key}": value for key, value in metrics["delta_E"].items()}})
        aggregates = {
            subset: aggregate_local([individual[str(seed)][subset] for seed in SEEDS])
            for subset in masks
        }
        one_step[treatment] = {"individual": individual, "aggregate": aggregates}
        nonlinear[treatment] = {"individual": nonlinear_individual}

    stability = training_stability(training)
    cases, recursive_aggregate = recursive_evaluation(models, normalization, arrays)
    trajectory_rows, attribution, sign_diagnostics = teacher_forced_hard_diagnostics(all_models, normalization, cases, arrays)

    trajectory_changes: list[dict[str, Any]] = []
    for family in HARD:
        for scope in ("complete_incoming", "far_upstream"):
            row: dict[str, Any] = {"family_u_th": family, "scope": scope}
            for metric in ("rms_e_perp", "mean_abs_delta_E_NL"):
                old = float(np.mean([item[metric] for item in trajectory_rows if item["family_u_th"] == family and item["scope"] == scope and item["treatment"] == "fixed_E0_baseline"]))
                new = float(np.mean([item[metric] for item in trajectory_rows if item["family_u_th"] == family and item["scope"] == scope and item["treatment"] == "sensitivity_aware"]))
                row[f"{metric}_baseline_mean"] = old
                row[f"{metric}_sensitivity_aware_mean"] = new
                row[f"{metric}_change_percent"] = pct(new, old)
            attr_key = "regions" if scope == "far_upstream" else None
            old_values = []
            new_values = []
            for seed in SEEDS:
                old_attr = attribution[f"{family:.2f}"]["fixed_E0_baseline"][str(seed)]
                new_attr = attribution[f"{family:.2f}"]["sensitivity_aware"][str(seed)]
                old_values.append(old_attr["regions"]["far_upstream"]["sum_absolute_A_total"] if attr_key else old_attr["sum_absolute_A_total"])
                new_values.append(new_attr["regions"]["far_upstream"]["sum_absolute_A_total"] if attr_key else new_attr["sum_absolute_A_total"])
            row["sum_absolute_A_total_baseline_mean"] = float(np.mean(old_values))
            row["sum_absolute_A_total_sensitivity_aware_mean"] = float(np.mean(new_values))
            row["sum_absolute_A_total_change_percent"] = pct(float(np.mean(new_values)), float(np.mean(old_values)))
            trajectory_changes.append(row)

    recursive_rows: list[dict[str, Any]] = []
    for case in cases:
        for mode in ("full_traversal", "throat_started_outgoing"):
            for treatment in TREATMENTS:
                for seed in SEEDS:
                    row = case[mode][treatment]["seeds"][str(seed)]
                    recursive_rows.append({
                        "u_th": case["u_th"],
                        "mode": mode,
                        "treatment": treatment,
                        "seed": seed,
                        "status": row["status"],
                        "reached_x_plus_17": row["reached_x_plus_17"],
                        "physical_exit": row["physical_exit"],
                        "maximum_step_guard": row["maximum_step_guard"],
                        "predicted_throat_u": row["interpolated_throat_u"],
                        "signed_throat_u_error": row["signed_throat_u_error"],
                        "absolute_throat_u_error": row["absolute_throat_u_error"],
                    })
    attribution_rows: list[dict[str, Any]] = []
    for family, treatments in attribution.items():
        for treatment, seeds in treatments.items():
            for seed, row in seeds.items():
                flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
                far = row["regions"]["far_upstream"]
                attribution_rows.append({"u_th": family, "treatment": treatment, "seed": seed, **flat, **{f"far_upstream_{key}": value for key, value in far.items()}})

    summary: dict[str, Any] = {
        "stage": "controlled fixed-E0 sensitivity-aware local-loss training and matched evaluation",
        "status": "three_runs_and_complete_frozen_protocol_evaluation_completed",
        "training": training,
        "baseline_training": {"path": str(BASELINE_TRAINING), "retrained": False},
        "loss_convention": {
            "lambda_S": 1.0,
            "implemented_batch_loss": "mean(e**2) + 0.5*mean(I*w_absS*(n_E dot e)**2)",
            "weight": "abs(S)/0.3526914868127516 on saved future-throat-crossing rows only",
            "clipping_capping_or_transform": None,
        },
        "optimization_stability": stability,
        "validation_subsets": {name: int(mask.sum()) for name, mask in masks.items()},
        "one_step_validation": one_step,
        "nonlinear_energy_validation": nonlinear,
        "cases": cases,
        "recursive_aggregate": recursive_aggregate,
        "trajectory_conditioned_metrics": trajectory_rows,
        "trajectory_conditioned_changes": trajectory_changes,
        "sensitivity_attribution": attribution,
        "attribution_sign_diagnostics": sign_diagnostics,
        "protocol": {
            "families": list(THROAT_VELOCITIES),
            "step_size": H,
            "maximum_model_steps": MAX_MODEL_STEPS,
            "baseline_retrained": False,
            "evaluation_retraining": False,
            "E0_held_fixed": True,
            "exact_reference_trajectories_regenerated": False,
            "sensitivity_kernels_recomputed_or_modified": False,
            "checkpoint_selected_by_recursive_performance": False,
            "lambda_tuning_or_other_variant_training": False,
        },
    }
    summary["scientific_conclusion"] = scientific_conclusion(summary)

    FIGURES.mkdir()
    plot_training(training)
    plot_local_subsets(one_step)
    plot_throat_errors(cases)
    plot_hard_diagnostics(arrays)
    np.savez_compressed(ARRAYS_PATH, **arrays)
    save_csv(ONE_STEP_CSV, one_rows)
    save_csv(ENERGY_CSV, energy_rows)
    save_csv(RECURSIVE_CSV, recursive_rows)
    save_csv(ATTRIBUTION_CSV, attribution_rows)
    save_csv(TRAJECTORY_CSV, trajectory_rows)
    after = hashes(protected)
    if before != after:
        raise RuntimeError("a protected artifact changed during evaluation")
    summary["protected_hashes_before"] = before
    summary["protected_hashes_after"] = after
    summary["artifacts"] = {
        "report": str(REPORT_PATH),
        "summary": str(SUMMARY_PATH),
        "arrays": str(ARRAYS_PATH),
        "one_step_csv": str(ONE_STEP_CSV),
        "nonlinear_energy_csv": str(ENERGY_CSV),
        "recursive_csv": str(RECURSIVE_CSV),
        "attribution_csv": str(ATTRIBUTION_CSV),
        "trajectory_csv": str(TRAJECTORY_CSV),
        "figures": [str(path) for path in sorted(FIGURES.glob("*.png"))],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report_text(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "recursive": recursive_aggregate, "conclusion": summary["scientific_conclusion"], "output": str(OUTPUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
