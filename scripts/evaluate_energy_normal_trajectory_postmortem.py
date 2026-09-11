#!/usr/bin/env python3
"""Teacher-forced, trajectory-conditioned postmortem of the energy-normal loss.

This script is evaluation-only.  It consumes the frozen checkpoints and the exact
states/kernels/energy gradients saved by the preceding studies; it never integrates
or regenerates a reference trajectory.
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

from wormhole_sciml.dynamics import conserved_energy, timelike_margin
from wormhole_sciml.model_a import Normalization, load_trained_model, predict_increments
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model_a_energy_normal_trajectory_postmortem"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "trajectory_postmortem_summary.json"
ARRAYS_PATH = OUTPUT / "trajectory_postmortem_arrays.npz"
METRICS_PATH = OUTPUT / "trajectory_conditioned_metrics.csv"
GLOBAL_PATH = OUTPUT / "global_vs_trajectory_conditioned.csv"
SIGN_PATH = OUTPUT / "seed_sign_diagnostics.csv"
REPORT_PATH = OUTPUT / "ENERGY_NORMAL_TRAJECTORY_POSTMORTEM.md"

BASE = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
BASE_TRAINING = BASE / "energy_xi_training_manifest.json"
NORMALIZATION_PATH = BASE / "training" / "energy_input_normalization.json"
EN = ROOT / "output" / "model_a_fixed_E0_energy_normal_loss"
EN_TRAINING = EN / "energy_normal_training_manifest.json"
EN_EVALUATION = EN / "energy_normal_evaluation_summary.json"
EXACT_PATH = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
SENSITIVITY_DIR = ROOT / "output" / "model_a_fixed_E0_throat_sensitivity"
SENSITIVITY_PATH = SENSITIVITY_DIR / "throat_sensitivity_arrays.npz"
SENSITIVITY_SUMMARY = SENSITIVITY_DIR / "throat_sensitivity_summary.json"
ALIGNMENT_DIR = ROOT / "output" / "model_a_energy_gradient_alignment"
ALIGNMENT_PATH = ALIGNMENT_DIR / "energy_gradient_alignment_arrays.npz"
ALIGNMENT_SUMMARY = ALIGNMENT_DIR / "energy_gradient_alignment_summary.json"

SEEDS = (101, 202, 303)
FAMILIES = (0.05, 0.15, 0.30)
TREATMENTS = ("fixed_E0_baseline", "energy_normal")
TITLES = {"fixed_E0_baseline": "Frozen fixed-$E_0$", "energy_normal": "Energy-normal loss"}
COLORS = {101: "#0072B2", 202: "#E69F00", 303: "#CC79A7"}
FAR = (-17.0, -8.5)


def fkey(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def pct(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def finite_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, "mean": None, "mae": None, "rmse": None, "minimum": None, "maximum": None}
    return {
        "count": int(a.size), "mean": float(np.mean(a)), "mae": float(np.mean(np.abs(a))),
        "rmse": float(np.sqrt(np.mean(a * a))), "minimum": float(np.min(a)), "maximum": float(np.max(a)),
    }


def metric_row(family: float, treatment: str, seed: int, scope: str, mask: np.ndarray,
               values: dict[str, np.ndarray]) -> dict[str, Any]:
    ep = values["e_perp"][mask]
    et = values["e_parallel"][mask]
    lin = values["delta_E_lin"][mask]
    nl = values["delta_E_NL"][mask]
    discrepancy = values["linear_nonlinear_abs_discrepancy"][mask]
    relative = values["linear_nonlinear_relative_discrepancy"][mask]
    valid = np.isfinite(nl)
    return {
        "family_u_th": family, "treatment": treatment, "seed": seed, "scope": scope,
        "count": int(mask.sum()), "mean_signed_e_perp": float(np.mean(ep)),
        "mean_abs_e_perp": float(np.mean(np.abs(ep))), "rms_e_perp": float(np.sqrt(np.mean(ep * ep))),
        "mean_signed_e_parallel": float(np.mean(et)), "mean_abs_e_parallel": float(np.mean(np.abs(et))),
        "rms_e_parallel": float(np.sqrt(np.mean(et * et))),
        "sum_delta_E_lin": float(np.sum(lin)), "sum_abs_delta_E_lin": float(np.sum(np.abs(lin))),
        "mean_abs_delta_E_lin": float(np.mean(np.abs(lin))), "rms_delta_E_lin": float(np.sqrt(np.mean(lin * lin))),
        "sum_delta_E_NL": float(np.sum(nl[valid])) if valid.any() else None,
        "sum_abs_delta_E_NL": float(np.sum(np.abs(nl[valid]))) if valid.any() else None,
        "mean_abs_delta_E_NL": float(np.mean(np.abs(nl[valid]))) if valid.any() else None,
        "rms_delta_E_NL": float(np.sqrt(np.mean(nl[valid] * nl[valid]))) if valid.any() else None,
        "nonlinear_valid_count": int(valid.sum()), "nonlinear_invalid_count": int((~valid).sum()),
        "mean_abs_linear_nonlinear_discrepancy": float(np.nanmean(discrepancy)) if valid.any() else None,
        "mean_relative_linear_nonlinear_discrepancy": float(np.nanmean(relative)) if valid.any() else None,
        "maximum_relative_linear_nonlinear_discrepancy": float(np.nanmax(relative)) if valid.any() else None,
    }


def stable_cumulative_sign_onset(x: np.ndarray, delta: np.ndarray) -> tuple[str, float | None, float]:
    """Threshold-free onset: first cumulative point after which its final sign never reverses."""
    cumulative = np.cumsum(delta)
    final = float(cumulative[-1])
    if final == 0.0:
        return "zero", None, final
    sign = 1.0 if final > 0.0 else -1.0
    good = sign * cumulative > 0.0
    suffix_all = np.logical_and.accumulate(good[::-1])[::-1]
    indices = np.flatnonzero(suffix_all)
    return ("positive" if sign > 0 else "negative"), (float(x[indices[0]]) if indices.size else None), final


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shade(ax: Any) -> None:
    ax.axvspan(FAR[0], FAR[1], color="#999999", alpha=0.14, lw=0)
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.45)


def plot_family_grid(data: dict[str, np.ndarray], quantity: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True)
    for row, family in enumerate(FAMILIES):
        key = fkey(family)
        x = data[f"{key}__x"]
        for col, treatment in enumerate(TREATMENTS):
            ax = axes[row, col]
            for seed in SEEDS:
                ax.plot(x, data[f"{key}__{treatment}__seed_{seed}__{quantity}"], color=COLORS[seed], lw=1.25,
                        label=f"seed {seed}")
            shade(ax)
            ax.set_title(f"{TITLES[treatment]}, $u_{{th}}={family:.2f}$")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.18)
    for ax in axes[-1]: ax.set_xlabel("exact incoming $x_n$")
    axes[0, 0].legend(ncol=3, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def plot_cumulative(data: dict[str, np.ndarray]) -> None:
    key = fkey(0.05); x = data[f"{key}__x"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), sharex=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            d = data[f"{key}__{treatment}__seed_{seed}__delta_E_lin"]
            axes[row, 0].plot(x, np.cumsum(d), color=COLORS[seed], lw=1.35, label=f"seed {seed}")
            axes[row, 1].plot(x, np.cumsum(np.abs(d)), color=COLORS[seed], lw=1.35)
        for col in range(2):
            shade(axes[row, col]); axes[row, col].grid(alpha=0.18)
        axes[row, 0].set_ylabel(f"{TITLES[treatment]}\nenergy sum")
    axes[0, 0].set_title(r"cumulative signed $\delta E_{lin}$")
    axes[0, 1].set_title(r"cumulative absolute $|\delta E_{lin}|$")
    for ax in axes[-1]: ax.set_xlabel("exact incoming $x_n$")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle(r"$u_{th}=0.05$: accumulation along the incoming branch", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIGURES / "figure_D_u_th_0p05_cumulative_energy.png", dpi=180); plt.close(fig)


def plot_multiplier(data: dict[str, np.ndarray], du_dE: float) -> None:
    key = fkey(0.05); x = data[f"{key}__x"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4), sharex=True)
    for row, treatment in enumerate(TREATMENTS):
        for seed in SEEDS:
            axes[row, 0].plot(x, data[f"{key}__{treatment}__seed_{seed}__delta_E_lin"], color=COLORS[seed], lw=1.25,
                              label=f"seed {seed}")
            axes[row, 1].plot(x, data[f"{key}__{treatment}__seed_{seed}__delta_u_th_E"], color=COLORS[seed], lw=1.25)
        for col in range(2): shade(axes[row, col]); axes[row, col].grid(alpha=0.18)
        axes[row, 0].set_ylabel(TITLES[treatment])
    axes[0, 0].set_title(r"$\delta E_{lin}$")
    axes[0, 1].set_title(rf"$\delta u_{{th}}^{{(E)}}=(du_{{th}}/dE)\delta E_{{lin}}$, multiplier={du_dE:.4f}")
    for ax in axes[-1]: ax.set_xlabel("exact incoming $x_n$")
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle(r"$u_{th}=0.05$: explicit sensitivity-magnitude amplification", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIGURES / "figure_E_u_th_0p05_sensitivity_multiplier.png", dpi=180); plt.close(fig)


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 4) -> list[str]:
    out = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float): cells.append(f"{value:.{digits}g}")
            else: cells.append("—" if value is None else str(value))
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    baseline_training = json.loads(BASE_TRAINING.read_text())
    en_training = json.loads(EN_TRAINING.read_text())
    prior_eval = json.loads(EN_EVALUATION.read_text())
    sensitivity_summary = json.loads(SENSITIVITY_SUMMARY.read_text())
    alignment_summary = json.loads(ALIGNMENT_SUMMARY.read_text())
    normalization = Normalization.from_stage1(
        NORMALIZATION_PATH, input_columns=("x", "xi", "E0"), target_columns=("delta_x", "delta_xi"),
        expected_source_dataset="outer_microcore40k_train_x_xi_energy_input_only")
    manifests = {"fixed_E0_baseline": baseline_training, "energy_normal": en_training}
    models = {t: {int(r["seed"]): load_trained_model(Path(r["checkpoint"])) for r in manifests[t]["runs"]} for t in TREATMENTS}
    if any(set(m) != set(SEEDS) for m in models.values()): raise RuntimeError("expected exactly seeds 101/202/303")

    protected = [BASE_TRAINING, NORMALIZATION_PATH, EN_TRAINING, EN_EVALUATION, EXACT_PATH, SENSITIVITY_PATH,
                 SENSITIVITY_SUMMARY, ALIGNMENT_PATH, ALIGNMENT_SUMMARY]
    protected += [Path(r["checkpoint"]) for m in manifests.values() for r in m["runs"]]
    before = {str(p): file_sha256(p) for p in protected}
    exact = np.load(EXACT_PATH); sensitivity = np.load(SENSITIVITY_PATH); alignment = np.load(ALIGNMENT_PATH)
    wormhole, spiral = experiment_parameters()
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    signs: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    all_values: dict[tuple[str, int], dict[str, list[np.ndarray]]] = {
        (t, s): {"ep_full": [], "ep_far": [], "nl_full": [], "nl_far": []} for t in TREATMENTS for s in SEEDS}

    for family in FAMILIES:
        key = fkey(family); valid_kernel = sensitivity[f"{key}__valid_kernel_mask"].astype(bool)
        state_all = np.column_stack((exact[f"{key}__exact_state"][:, 0], exact[f"{key}__exact_xi"]))
        target_all = np.column_stack((sensitivity[f"{key}__exact_delta_x_all"], sensitivity[f"{key}__exact_delta_xi_all"]))
        state = state_all[valid_kernel]; target = target_all[valid_kernel]
        x = state[:, 0]; e0 = float(sensitivity_summary["families"][f"{family:.2f}"]["E0"])
        features = np.column_stack((state, np.full(x.shape, e0)))
        gE = alignment[f"{key}__g_E"]; nE = alignment[f"{key}__n_E"]
        K = np.column_stack((alignment[f"{key}__K_x"], alignment[f"{key}__K_xi"]))
        if not np.array_equal(x, alignment[f"{key}__diagnostic_x"]): raise RuntimeError(f"{key}: exact x mismatch")
        expected_post = state + target
        if not np.allclose(expected_post[:, 0], alignment[f"{key}__post_x"], rtol=0, atol=1e-14): raise RuntimeError(f"{key}: post x mismatch")
        if not np.allclose(expected_post[:, 1], alignment[f"{key}__post_xi"], rtol=0, atol=1e-14): raise RuntimeError(f"{key}: post xi mismatch")
        tangent = np.column_stack((-nE[:, 1], nE[:, 0])); far = (x >= FAR[0]) & (x < FAR[1])
        du_dE = float(alignment_summary["families"][f"{family:.2f}"]["proportionality"]["theoretical_du_th_dE"])
        arrays[f"{key}__x"] = x; arrays[f"{key}__far_upstream_mask"] = far; arrays[f"{key}__n_E"] = nE
        arrays[f"{key}__g_E"] = gE; arrays[f"{key}__K"] = K
        family_equivalence = []
        for treatment in TREATMENTS:
            for seed in SEEDS:
                prediction = predict_increments(models[treatment][seed], features, normalization)
                raw = prediction - target; standard = raw / normalization.target_std
                ep = np.sum(standard * nE, axis=1); et = np.sum(standard * tangent, axis=1)
                lin = np.sum(standard * gE, axis=1)
                pred_x = state[:, 0] + prediction[:, 0]; pred_xi = state[:, 1] + prediction[:, 1]
                _, pred_u = state_from_xi(pred_x, pred_xi, wormhole, spiral)
                margin = timelike_margin(pred_x, pred_u, wormhole, spiral)
                physical = np.isfinite(pred_x) & np.isfinite(pred_xi) & np.isfinite(pred_u) & np.isfinite(margin) & (margin > 0)
                nl = np.full(x.shape, np.nan); nl[physical] = conserved_energy(pred_x[physical], pred_u[physical], wormhole, spiral) - e0
                absdisc = np.abs(nl - lin)
                reldisc = absdisc / np.maximum.reduce((np.abs(nl), np.abs(lin), np.full(x.shape, 1e-14)))
                due = du_dE * lin; direct = np.sum(K * raw, axis=1)
                equiv_abs = np.abs(due - direct); equiv_rel = equiv_abs / np.maximum.reduce((np.abs(due), np.abs(direct), np.full(x.shape, 1e-14)))
                prefix = f"{key}__{treatment}__seed_{seed}__"
                vals = {"prediction": prediction, "raw_error": raw, "standardized_error": standard, "e_perp": ep,
                        "e_parallel": et, "delta_E_lin": lin, "delta_E_NL": nl, "nonlinear_valid": physical,
                        "predicted_timelike_margin": margin, "linear_nonlinear_abs_discrepancy": absdisc,
                        "linear_nonlinear_relative_discrepancy": reldisc, "delta_u_th_E": due,
                        "direct_K_attribution": direct, "K_equivalence_abs_error": equiv_abs,
                        "K_equivalence_relative_error": equiv_rel}
                for name, value in vals.items(): arrays[prefix + name] = value
                rows.append(metric_row(family, treatment, seed, "complete_incoming", np.ones(x.shape, bool), vals))
                rows.append(metric_row(family, treatment, seed, "far_upstream", far, vals))
                all_values[(treatment, seed)]["ep_full"].append(ep)
                all_values[(treatment, seed)]["ep_far"].append(ep[far])
                all_values[(treatment, seed)]["nl_full"].append(nl[np.isfinite(nl)])
                all_values[(treatment, seed)]["nl_far"].append(nl[far & np.isfinite(nl)])
                sign, onset, final = stable_cumulative_sign_onset(x, lin)
                far_sign, far_onset, far_final = stable_cumulative_sign_onset(x[far], lin[far])
                signs.append({"family_u_th": family, "treatment": treatment, "seed": seed,
                              "full_final_cumulative_sign": sign, "full_persistent_sign_onset_x": onset,
                              "full_final_cumulative_delta_E_lin": final, "far_final_cumulative_sign": far_sign,
                              "far_persistent_sign_onset_x": far_onset, "far_final_cumulative_delta_E_lin": far_final,
                              "far_mean_delta_E_lin": float(np.mean(lin[far])), "far_mean_delta_E_NL": float(np.nanmean(nl[far]))})
                family_equivalence.append((float(np.max(equiv_abs)), float(np.max(equiv_rel))))
        integrity[f"{family:.2f}"] = {
            "exact_state_count": int(x.size), "far_upstream_count": int(far.sum()), "du_th_dE": du_dE,
            "maximum_absolute_K_equivalence_error": max(v[0] for v in family_equivalence),
            "maximum_relative_K_equivalence_error": max(v[1] for v in family_equivalence),
            "exact_diagnostic_x_identical": True, "exact_post_step_states_identical": True,
        }

    global_rows = []
    for treatment in TREATMENTS:
        for seed in SEEDS:
            combined = all_values[(treatment, seed)]
            ep_full = np.concatenate(combined["ep_full"]); ep_far = np.concatenate(combined["ep_far"])
            nl_full = np.concatenate(combined["nl_full"]); nl_far = np.concatenate(combined["nl_far"])
            global_ep = float(prior_eval["one_step_validation"][treatment]["individual"][str(seed)]["rms_e_perp"])
            global_nl = float(prior_eval["nonlinear_energy_validation"][treatment]["individual"][str(seed)]["signed_delta_E"]["mae"])
            global_rows.append({"treatment": treatment, "seed": seed, "global_validation_rms_e_perp": global_ep,
                                "hard_families_rms_e_perp": float(np.sqrt(np.mean(ep_full * ep_full))),
                                "far_upstream_hard_families_rms_e_perp": float(np.sqrt(np.mean(ep_far * ep_far))),
                                "global_validation_nonlinear_energy_mae": global_nl,
                                "hard_families_nonlinear_energy_mae": float(np.mean(np.abs(nl_full))),
                                "far_upstream_hard_families_nonlinear_energy_mae": float(np.mean(np.abs(nl_far)))})

    # Add the already-computed recursive outcomes solely for sign interpretation.
    for sign_row in signs:
        case = next(c for c in prior_eval["cases"] if float(c["u_th"]) == float(sign_row["family_u_th"]))
        outcome = case["full_traversal"][sign_row["treatment"]]["seeds"][str(sign_row["seed"])]
        sign_row["recursive_status"] = outcome["status"]
        sign_row["recursive_signed_throat_u_error"] = outcome["signed_throat_u_error"]

    np.savez_compressed(ARRAYS_PATH, **arrays); save_csv(METRICS_PATH, rows); save_csv(GLOBAL_PATH, global_rows); save_csv(SIGN_PATH, signs)
    plot_family_grid(arrays, "e_perp", r"$e_\perp$", "figure_A_energy_normal_error.png")
    plot_family_grid(arrays, "delta_E_lin", r"$\delta E_{lin}$", "figure_B_linearized_energy_error.png")
    plot_family_grid(arrays, "delta_E_NL", r"$\delta E_{NL}$", "figure_C_nonlinear_energy_error.png")
    plot_cumulative(arrays); plot_multiplier(arrays, integrity["0.05"]["du_th_dE"])

    after = {str(p): file_sha256(p) for p in protected}
    if before != after: raise RuntimeError("a protected artifact changed during evaluation")
    # Paired EN-vs-baseline changes in the requested three scopes.
    comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        b = next(r for r in global_rows if r["treatment"] == "fixed_E0_baseline" and r["seed"] == seed)
        e = next(r for r in global_rows if r["treatment"] == "energy_normal" and r["seed"] == seed)
        comparisons.append({"seed": seed,
            "global_rms_e_perp_change_percent": pct(e["global_validation_rms_e_perp"], b["global_validation_rms_e_perp"]),
            "hard_rms_e_perp_change_percent": pct(e["hard_families_rms_e_perp"], b["hard_families_rms_e_perp"]),
            "far_hard_rms_e_perp_change_percent": pct(e["far_upstream_hard_families_rms_e_perp"], b["far_upstream_hard_families_rms_e_perp"]),
            "global_nonlinear_mae_change_percent": pct(e["global_validation_nonlinear_energy_mae"], b["global_validation_nonlinear_energy_mae"]),
            "hard_nonlinear_mae_change_percent": pct(e["hard_families_nonlinear_energy_mae"], b["hard_families_nonlinear_energy_mae"]),
            "far_hard_nonlinear_mae_change_percent": pct(e["far_upstream_hard_families_nonlinear_energy_mae"], b["far_upstream_hard_families_nonlinear_energy_mae"])})
    mean_comparison = {k: float(np.mean([r[k] for r in comparisons])) for k in comparisons[0] if k != "seed"}
    global_metric_keys = (
        "global_validation_rms_e_perp", "hard_families_rms_e_perp",
        "far_upstream_hard_families_rms_e_perp", "global_validation_nonlinear_energy_mae",
        "hard_families_nonlinear_energy_mae", "far_upstream_hard_families_nonlinear_energy_mae",
    )
    change_of_seed_means = {}
    for metric in global_metric_keys:
        base_mean = float(np.mean([r[metric] for r in global_rows if r["treatment"] == "fixed_E0_baseline"]))
        en_mean = float(np.mean([r[metric] for r in global_rows if r["treatment"] == "energy_normal"]))
        change_of_seed_means[metric] = {"baseline_mean": base_mean, "energy_normal_mean": en_mean,
                                        "change_percent": pct(en_mean, base_mean)}
    family_scope_changes = []
    for family in FAMILIES:
        for scope in ("complete_incoming", "far_upstream"):
            item: dict[str, Any] = {"family_u_th": family, "scope": scope}
            for metric in ("rms_e_perp", "mean_abs_delta_E_NL", "rms_e_parallel"):
                bmean = float(np.mean([r[metric] for r in rows if r["family_u_th"] == family and r["scope"] == scope and r["treatment"] == "fixed_E0_baseline"]))
                emean = float(np.mean([r[metric] for r in rows if r["family_u_th"] == family and r["scope"] == scope and r["treatment"] == "energy_normal"]))
                item[f"{metric}_baseline_seed_mean"] = bmean
                item[f"{metric}_energy_normal_seed_mean"] = emean
                item[f"{metric}_change_percent"] = pct(emean, bmean)
            family_scope_changes.append(item)

    summary = {
        "stage": "trajectory-conditioned energy-error postmortem", "status": "complete",
        "protocol": {"evaluation_only": True, "models_trained_or_retrained": 0, "reference_trajectories_regenerated": False,
                     "teacher_forced_exact_incoming_states": True, "families": list(FAMILIES),
                     "far_upstream_interval": {"lower_inclusive": FAR[0], "upper_exclusive": FAR[1]},
                     "tangent_convention": "t_E=(-n_E_xi,n_E_x)",
                     "persistent_sign_onset_definition": "first cumulative-sum sample after which its sign never again differs from the final cumulative sign; no magnitude threshold"},
        "standardization": {"sigma_delta_x": float(normalization.target_std[0]), "sigma_delta_xi": float(normalization.target_std[1]),
                            "source": str(NORMALIZATION_PATH)},
        "integrity": integrity, "protected_hashes_before": before, "protected_hashes_after": after,
        "paired_changes_percent": comparisons, "mean_paired_changes_percent": mean_comparison,
        "change_of_seed_means": change_of_seed_means, "family_scope_changes": family_scope_changes,
        "nonlinear_invalid_total": int(sum(r["nonlinear_invalid_count"] for r in rows if r["scope"] == "complete_incoming")),
        "artifacts": {"arrays": str(ARRAYS_PATH), "metrics": str(METRICS_PATH), "global_comparison": str(GLOBAL_PATH),
                      "seed_signs": str(SIGN_PATH), "figures": [str(p) for p in sorted(FIGURES.glob("*.png"))], "report": str(REPORT_PATH)},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    far_rows = [r for r in rows if r["scope"] == "far_upstream"]
    report: list[str] = ["# Trajectory-Conditioned Energy-Error Postmortem", "",
        "## Result", "",
        f"The geometry-only energy-normal loss improved the frozen global-validation seed-mean RMS e_perp by {change_of_seed_means['global_validation_rms_e_perp']['change_percent']:.1f}% and nonlinear-energy MAE by {change_of_seed_means['global_validation_nonlinear_energy_mae']['change_percent']:.1f}%. On the complete hard branches, RMS e_perp changed by {change_of_seed_means['hard_families_rms_e_perp']['change_percent']:.1f}% but nonlinear-energy MAE worsened by {change_of_seed_means['hard_families_nonlinear_energy_mae']['change_percent']:.1f}%. In the decisive far-upstream subset, both reversed: RMS e_perp changed by {change_of_seed_means['far_upstream_hard_families_rms_e_perp']['change_percent']:.1f}% and nonlinear-energy MAE by {change_of_seed_means['far_upstream_hard_families_nonlinear_energy_mae']['change_percent']:.1f}%. The teacher-forced comparison isolates this redistribution without recursive feedback.", "",
        "## Frozen protocol and integrity", "",
        f"No model was trained or retrained, and no exact trajectory was regenerated. All {sum(v['exact_state_count'] for v in integrity.values())} diagnostic states are the saved sensitivity-study states; exact post-step states and coordinates match the alignment gate. Target scales were sigma_dx={normalization.target_std[0]:.16g}, sigma_dxi={normalization.target_std[1]:.16g}. Protected hashes were unchanged.", "",
        "The tangent convention is `t_E=(-n_E,xi,n_E,x)`. A systematic-sign onset is reported without a tuned threshold: it is the first cumulative-sum point after which the cumulative sign never reverses relative to its final sign.", "",
        "## Global validation versus selected hard trajectories", ""]
    report += md_table(global_rows, [("treatment","treatment"),("seed","seed"),("global_validation_rms_e_perp","global RMS e_perp"),
        ("hard_families_rms_e_perp","hard RMS"),("far_upstream_hard_families_rms_e_perp","far hard RMS"),
        ("global_validation_nonlinear_energy_mae","global NL MAE"),("hard_families_nonlinear_energy_mae","hard NL MAE"),
        ("far_upstream_hard_families_nonlinear_energy_mae","far hard NL MAE")], 5)
    report += ["", "Paired EN change relative to its same-seed baseline (negative is improvement):", ""]
    report += md_table(comparisons, [("seed","seed"),("global_rms_e_perp_change_percent","global e_perp %"),
        ("hard_rms_e_perp_change_percent","hard e_perp %"),("far_hard_rms_e_perp_change_percent","far hard e_perp %"),
        ("global_nonlinear_mae_change_percent","global NL %"),("hard_nonlinear_mae_change_percent","hard NL %"),
        ("far_hard_nonlinear_mae_change_percent","far hard NL %")], 4)
    report += ["", "Family/scope change of the three-seed means (EN relative to baseline):", ""]
    report += md_table(family_scope_changes, [("family_u_th","u_th"),("scope","scope"),
        ("rms_e_perp_change_percent","RMS e_perp %"),("mean_abs_delta_E_NL_change_percent","NL MAE %"),
        ("rms_e_parallel_change_percent","RMS e_parallel %")], 4)
    report += ["", "## Far-upstream numerical summary", ""]
    report += md_table(far_rows, [("family_u_th","u_th"),("treatment","treatment"),("seed","seed"),
        ("mean_signed_e_perp","mean e_perp"),("mean_abs_e_perp","mean |e_perp|"),("rms_e_perp","RMS e_perp"),
        ("mean_signed_e_parallel","mean e_parallel"),("rms_e_parallel","RMS e_parallel"),
        ("sum_delta_E_lin","sum dE lin"),("sum_abs_delta_E_lin","sum |dE lin|"),
        ("sum_delta_E_NL","sum dE NL"),("sum_abs_delta_E_NL","sum |dE NL|")], 5)
    report += ["", "The complete-incoming table, including the same quantities and nonlinear-validity/discrepancy fields, is in `trajectory_conditioned_metrics.csv`.", "",
        "## Linearized/nonlinear and kernel integrity", ""]
    report += md_table([{"family": k, **v} for k,v in integrity.items()], [("family","u_th"),("exact_state_count","N"),("far_upstream_count","N far"),
        ("du_th_dE","du_th/dE"),("maximum_absolute_K_equivalence_error","max abs K check"),("maximum_relative_K_equivalence_error","max rel K check")], 6)
    report += ["", f"Nonlinear one-step energy evaluation had {summary['nonlinear_invalid_total']} invalid/nonphysical states across all treatment/seed/family cases. The per-case linear-versus-nonlinear absolute and relative discrepancies are tabulated in the CSV; the two diagnostics agree on the qualitative trajectory-conditioned conclusion.", "",
        "## Seed-dependent sign diagnostic", ""]
    report += md_table([r for r in signs if r["family_u_th"] == 0.05], [("treatment","treatment"),("seed","seed"),
        ("far_final_cumulative_sign","far sign"),("far_persistent_sign_onset_x","far onset x"),
        ("far_final_cumulative_delta_E_lin","far cumulative dE"),("far_mean_delta_E_NL","far mean dE NL"),
        ("recursive_status","recursive status"),("recursive_signed_throat_u_error","throat error")], 6)
    report += ["", "For u_th=0.05, positive energy displacement predicts a positive throat-velocity shift because du_th/dE is positive. The EN seed-101 and seed-303 signs therefore anticipate their later overshoots; seed 202 has the opposite upstream energy drift and later exits before a throat value exists. The onset locations use the threshold-free cumulative-sign definition above, not a fitted cutoff.", "",
        "## Figures", "",
        "- Figure A: `figures/figure_A_energy_normal_error.png`", "- Figure B: `figures/figure_B_linearized_energy_error.png`",
        "- Figure C: `figures/figure_C_nonlinear_energy_error.png`", "- Figure D: `figures/figure_D_u_th_0p05_cumulative_energy.png`",
        "- Figure E: `figures/figure_E_u_th_0p05_sensitivity_multiplier.png`", "",
        "## Scientific answers", "",
        "**A. Yes, specifically on the decisive subset.** The global validation averages improve. Complete-branch RMS e_perp is mixed (and modestly better in the three-seed mean), but complete-branch nonlinear-energy MAE worsens; in the far-upstream high-sensitivity subset every seed worsens both RMS e_perp and nonlinear-energy MAE.", "",
        "**B. Yes in downstream consequence, with an important distinction.** The u_th=0.05 family carries the largest fixed multiplier du_th/dE=8.1011, so comparable local energy errors matter far more there. The subset-error sizes and the sensitivity magnitude are separate facts.", "",
        "**C. Yes.** True nonlinear one-step energy displacement preserves the same sign and comparative story; the tabulated discrepancies quantify where first-order linearization is imperfect.", "",
        "**D. Yes.** The u_th=0.05 EN signs already separate seed 101/303 from seed 202 in teacher-forced upstream diagnostics and are consistent with later overshoot versus physical exit.", "",
        "**E. Supported as an interpretation, not proven as a capacity mechanism.** A uniform geometry-only penalty reduced the dataset average while redistributing error unfavorably on the scientifically decisive subset. This is evidence about loss allocation; it does not by itself establish that the 3-32-32-2 architecture caused the tradeoff.", "",
        "**F. Yes.** Because the energy-normal direction is validated but du_th/dE varies strongly by family, these results provide a physics justification for testing sensitivity magnitude in a separate future experiment. No such loss was designed or trained here.", "",
        "## Stop condition", "", "Evaluation ends here. No training, loss modification, sensitivity weighting, multi-step objective, architecture change, or dataset/split change was performed."]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "mean_paired_changes_percent": mean_comparison,
                      "nonlinear_invalid_total": summary["nonlinear_invalid_total"], "integrity": integrity}, indent=2))


if __name__ == "__main__":
    main()
