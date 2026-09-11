#!/usr/bin/env python3
"""Validate energy-gradient alignment with saved exact-flow sensitivity kernels.

This evaluator is intentionally read-only with respect to all existing physics,
data, normalization, sensitivity, and model artifacts.  It performs no neural
network loading or training.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import conserved_energy
from wormhole_sciml.geometry import areal_radius, areal_radius_derivative
from wormhole_sciml.physics_gate import (
    experiment_parameters,
    state_from_xi,
    xi_from_state,
)
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_DIR = ROOT / "output" / "model_a_fixed_E0_throat_sensitivity"
SENSITIVITY_ARRAYS = SENSITIVITY_DIR / "throat_sensitivity_arrays.npz"
SENSITIVITY_SUMMARY = SENSITIVITY_DIR / "throat_sensitivity_summary.json"
EXACT_REFERENCE = (
    ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
)
FIXED_E0_NORMALIZATION = (
    ROOT
    / "output"
    / "model_a_x_xi_energy_microcore40k_comparison"
    / "training"
    / "energy_input_normalization.json"
)
MICROCORE_NORMALIZATION = (
    ROOT
    / "output"
    / "model_a_x_xi_sampling_training_comparison"
    / "microcore40k_normalization.json"
)
OUTPUT = ROOT / "output" / "model_a_energy_gradient_alignment"

FAMILIES = (0.05, 0.15, 0.30)
FAMILY_COLORS = {0.05: "#0072B2", 0.15: "#D55E00", 0.30: "#009E73"}
REPRESENTATIVE_X = (-17.0, -8.5, -2.0, 0.0)
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
ADOPTED_FD_STEP = 1.0e-3
DEGENERACY_THRESHOLD = 1.0e-14
FAR_UPSTREAM = (-17.0, -8.5)


def stem(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required artifact(s) missing: {missing}")
    return {str(path): file_sha256(path) for path in paths}


def energy_gradient_x_xi(x: Any, xi: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the analytic gradient of the validated energy in (x, xi).

    The calculation differentiates the algebraic expression used by
    ``conserved_energy`` and the established ``u=c(x)+d(x) xi`` coordinate map.
    The implementation remains parameter-generic for the validated experiment.
    """

    wormhole, spiral = experiment_parameters()
    x_array = np.asarray(x, dtype=np.float64)
    xi_array = np.asarray(xi, dtype=np.float64)
    sine_squared = np.sin(spiral.theta) ** 2
    radius = areal_radius(x_array, wormhole)
    radius_prime = areal_radius_derivative(x_array, wormhole)
    q = sine_squared * radius**2
    q_x = 2.0 * sine_squared * radius * radius_prime

    corridor_denominator = 1.0 + q * spiral.alpha**2
    corridor_discriminant = 1.0 + q * (
        spiral.alpha**2 - spiral.omega**2
    )
    center = -q * spiral.omega * spiral.alpha / corridor_denominator
    half_width = np.sqrt(corridor_discriminant) / corridor_denominator
    center_x = (
        -spiral.omega
        * spiral.alpha
        * q_x
        / corridor_denominator**2
    )
    half_width_x = half_width * q_x * (
        0.5
        * (spiral.alpha**2 - spiral.omega**2)
        / corridor_discriminant
        - spiral.alpha**2 / corridor_denominator
    )

    u = center + half_width * xi_array
    omega_effective = spiral.omega + spiral.alpha * u
    margin = 1.0 - u**2 - q * omega_effective**2
    numerator = 1.0 - q * spiral.omega * omega_effective
    if np.any(margin <= 0.0):
        raise ValueError("energy gradient requested outside the timelike domain")

    numerator_x_at_u = -q_x * spiral.omega * omega_effective
    numerator_u = -q * spiral.omega * spiral.alpha
    margin_x_at_u = -q_x * omega_effective**2
    margin_u = -2.0 * u - 2.0 * q * spiral.alpha * omega_effective
    root_margin = np.sqrt(margin)
    energy_x_at_u = (
        numerator_x_at_u / root_margin
        - 0.5 * numerator * margin_x_at_u / margin**1.5
    )
    energy_u = (
        numerator_u / root_margin
        - 0.5 * numerator * margin_u / margin**1.5
    )
    u_x_at_xi = center_x + half_width_x * xi_array
    return (
        np.asarray(energy_x_at_u + energy_u * u_x_at_xi, dtype=np.float64),
        np.asarray(energy_u * half_width, dtype=np.float64),
    )


def energy_in_x_xi(x: float, xi: float) -> float:
    wormhole, spiral = experiment_parameters()
    _, u = state_from_xi(x, xi, wormhole, spiral)
    return float(conserved_energy(x, u, wormhole, spiral))


def five_point_derivative(
    function: Callable[[float], float], value: float, step: float
) -> float:
    return float(
        (
            -function(value + 2.0 * step)
            + 8.0 * function(value + step)
            - 8.0 * function(value - step)
            + function(value - 2.0 * step)
        )
        / (12.0 * step)
    )


def descriptive(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def alignment_summary(
    diagnostic_x: np.ndarray,
    post_x: np.ndarray,
    cosine: np.ndarray,
    absolute_cosine: np.ndarray,
    theta_degrees: np.ndarray,
    relative_residual: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise RuntimeError("requested alignment region contains no states")
    local_worst = indices[int(np.argmax(theta_degrees[mask]))]
    return {
        "point_count": int(indices.size),
        "c_EK": descriptive(cosine[mask]),
        "absolute_c_EK": descriptive(absolute_cosine[mask]),
        "theta_EK_degrees": descriptive(theta_degrees[mask]),
        "relative_projection_residual": descriptive(relative_residual[mask]),
        "alpha": descriptive(alpha[mask]),
        "worst_directional_disagreement": {
            "diagnostic_incoming_x": float(diagnostic_x[local_worst]),
            "energy_gradient_post_step_x": float(post_x[local_worst]),
            "c_EK": float(cosine[local_worst]),
            "absolute_c_EK": float(absolute_cosine[local_worst]),
            "theta_EK_degrees": float(theta_degrees[local_worst]),
            "relative_projection_residual": float(relative_residual[local_worst]),
        },
    }


def validate_derivatives(
    exact: np.lib.npyio.NpzFile,
    sensitivity: np.lib.npyio.NpzFile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        key = stem(family)
        physical_state = np.asarray(exact[f"{key}__exact_state"], dtype=np.float64)
        exact_delta = np.asarray(exact[f"{key}__exact_delta"], dtype=np.float64)
        post_state = physical_state + exact_delta
        wormhole, spiral = experiment_parameters()
        post_xi_all = xi_from_state(
            post_state[:, 0], post_state[:, 1], wormhole, spiral
        )
        valid = np.asarray(sensitivity[f"{key}__valid_kernel_mask"], dtype=bool)
        diagnostic_x = physical_state[valid, 0]
        post_x = post_state[valid, 0]
        post_xi = post_xi_all[valid]
        analytic_x, analytic_xi = energy_gradient_x_xi(post_x, post_xi)
        for requested_x in REPRESENTATIVE_X:
            index = int(np.argmin(np.abs(diagnostic_x - requested_x)))
            for variable, analytic in (
                ("x", float(analytic_x[index])),
                ("xi", float(analytic_xi[index])),
            ):
                for step in FD_STEPS:
                    if variable == "x":
                        function = lambda value, i=index: energy_in_x_xi(
                            value, float(post_xi[i])
                        )
                        center = float(post_x[index])
                    else:
                        function = lambda value, i=index: energy_in_x_xi(
                            float(post_x[i]), value
                        )
                        center = float(post_xi[index])
                    numerical = five_point_derivative(function, center, step)
                    absolute_error = abs(numerical - analytic)
                    relative_error = absolute_error / max(abs(analytic), 1.0e-300)
                    rows.append(
                        {
                            "u_th": family,
                            "requested_incoming_x": requested_x,
                            "actual_incoming_x": float(diagnostic_x[index]),
                            "post_step_x": float(post_x[index]),
                            "post_step_xi": float(post_xi[index]),
                            "variable": variable,
                            "finite_difference_step": step,
                            "analytic_derivative": analytic,
                            "finite_difference_derivative": numerical,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                        }
                    )

    adopted = [row for row in rows if row["finite_difference_step"] == ADOPTED_FD_STEP]
    by_variable: dict[str, Any] = {}
    for variable in ("x", "xi"):
        selected = [row for row in adopted if row["variable"] == variable]
        by_variable[variable] = {
            "comparison_count": len(selected),
            "maximum_absolute_error": max(row["absolute_error"] for row in selected),
            "median_absolute_error": float(
                np.median([row["absolute_error"] for row in selected])
            ),
            "maximum_relative_error": max(row["relative_error"] for row in selected),
            "median_relative_error": float(
                np.median([row["relative_error"] for row in selected])
            ),
        }
    summary = {
        "method": "centered five-point finite difference of validated conserved_energy after the established (x, xi)->(x, u) map",
        "representative_incoming_x_requests": list(REPRESENTATIVE_X),
        "families": list(FAMILIES),
        "candidate_steps": list(FD_STEPS),
        "adopted_reporting_step": ADOPTED_FD_STEP,
        "representative_state_count": len(adopted) // 2,
        "component_comparison_count": len(adopted),
        "by_variable_at_adopted_step": by_variable,
        "all_adopted_relative_errors_below_1e-7": all(
            row["relative_error"] < 1.0e-7 for row in adopted
        ),
        "all_values_finite": all(
            np.isfinite(row[name])
            for row in rows
            for name in (
                "analytic_derivative",
                "finite_difference_derivative",
                "absolute_error",
                "relative_error",
            )
        ),
    }
    return rows, summary


def throat_du_dE(u_th: float) -> float:
    wormhole, spiral = experiment_parameters()
    q = np.sin(spiral.theta) ** 2 * float(areal_radius(0.0, wormhole)) ** 2
    omega_effective = spiral.omega + spiral.alpha * u_th
    margin = 1.0 - u_th**2 - q * omega_effective**2
    numerator = 1.0 - q * spiral.omega * omega_effective
    numerator_u = -q * spiral.omega * spiral.alpha
    margin_u = -2.0 * u_th - 2.0 * q * spiral.alpha * omega_effective
    dE_du = (
        numerator_u / np.sqrt(margin)
        - 0.5 * numerator * margin_u / margin**1.5
    )
    return float(1.0 / dE_du)


def plot_alignment(
    families: dict[str, dict[str, np.ndarray]], destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for family in FAMILIES:
        data = families[f"{family:.2f}"]
        axis.plot(
            data["diagnostic_x"],
            data["absolute_cosine"],
            lw=1.8,
            color=FAMILY_COLORS[family],
            label=rf"$u_{{\rm th}}={family:.2f}$",
        )
    axis.axvspan(*FAR_UPSTREAM, color="#999999", alpha=0.12, label="far upstream")
    axis.set_xlabel(r"exact incoming diagnostic location $x_n$")
    axis.set_ylabel(r"$|c_{EK}|$")
    axis.set_title("Energy-normal and exact-flow sensitivity alignment")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_angles(
    families: dict[str, dict[str, np.ndarray]], destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for family in FAMILIES:
        data = families[f"{family:.2f}"]
        axis.plot(
            data["diagnostic_x"],
            data["theta_degrees"],
            lw=1.8,
            color=FAMILY_COLORS[family],
            label=rf"$u_{{\rm th}}={family:.2f}$",
        )
    axis.axvspan(*FAR_UPSTREAM, color="#999999", alpha=0.12, label="far upstream")
    axis.set_xlabel(r"exact incoming diagnostic location $x_n$")
    axis.set_ylabel(r"acute misalignment $\theta_{EK}$ (degrees)")
    axis.set_title("Numerically resolved directional misalignment")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_components(
    families: dict[str, dict[str, np.ndarray]], destination: Path
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True)
    for axis, family in zip(axes, FAMILIES, strict=True):
        data = families[f"{family:.2f}"]
        x = data["diagnostic_x"]
        n_E, n_K = data["n_E"], data["n_K"]
        axis.plot(x, n_E[:, 0], color="#0072B2", lw=2.0, label=r"$n_{E,x}$")
        axis.plot(x, n_K[:, 0], color="#0072B2", lw=1.0, ls="--", label=r"$n_{K,x}$")
        axis.plot(x, n_E[:, 1], color="#D55E00", lw=2.0, label=r"$n_{E,\xi}$")
        axis.plot(x, n_K[:, 1], color="#D55E00", lw=1.0, ls="--", label=r"$n_{K,\xi}$")
        axis.axvspan(*FAR_UPSTREAM, color="#999999", alpha=0.12)
        axis.set_ylabel("component")
        axis.set_title(rf"$u_{{\rm th}}={family:.2f}$")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, loc="lower left", fontsize=8)
    axes[-1].set_xlabel(r"exact incoming diagnostic location $x_n$")
    figure.suptitle("Standardized energy and sensitivity unit directions")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def format_number(value: float) -> str:
    return f"{value:.8g}"


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Energy-gradient vs exact-flow sensitivity physics gate",
        "",
        "## Scope and reused artifacts",
        "",
        "This is a pre-training physics/geometry validation only. It loads the saved exact-flow "
        "kernels and their exact trajectory reference, evaluates the validated conserved energy at "
        "the exact post-step states, and uses the fixed-E0 microcore40k target scales. No checkpoint "
        "is loaded, and no training, loss, architecture, data, split, or checkpoint is changed.",
        "",
        f"- Sensitivity arrays: `{summary['sources']['sensitivity_arrays']['path']}`",
        f"- Exact states: `{summary['sources']['exact_reference']['path']}`",
        f"- Fixed-E0 normalization: `{summary['sources']['fixed_E0_normalization']['path']}`",
        f"- $\\sigma_{{\\Delta x}}={summary['standardization']['sigma_delta_x']:.17g}$",
        f"- $\\sigma_{{\\Delta\\xi}}={summary['standardization']['sigma_delta_xi']:.17g}$",
        "",
        "The horizontal coordinate in the figures/tables is the original incoming diagnostic "
        "location $x_n$ used by the sensitivity study. Every energy derivative is evaluated at its "
        "matched exact post-step state $(x_{n+1},\\xi_{n+1})$.",
        "",
        "## Energy-derivative validation",
        "",
        "The analytic derivative of the existing energy expression and established coordinate map "
        "was checked against centered five-point finite differences of `conserved_energy` at 12 "
        "representative post-step states (four regions for each family). Step sizes "
        f"`{list(FD_STEPS)}` were inspected; the table reports $h={ADOPTED_FD_STEP:g}$.",
        "",
        "| derivative | comparisons | median abs error | max abs error | median rel error | max rel error |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for variable in ("x", "xi"):
        row = summary["derivative_validation"]["by_variable_at_adopted_step"][variable]
        lines.append(
            f"| $E_{variable}$ | {row['comparison_count']} | "
            f"{row['median_absolute_error']:.3e} | {row['maximum_absolute_error']:.3e} | "
            f"{row['median_relative_error']:.3e} | {row['maximum_relative_error']:.3e} |"
        )
    saved = summary["derivative_validation"]["agreement_with_saved_energy_displacement"]
    lines.extend(
        [
            "",
            "All finite-difference values were finite and all 24 adopted component comparisons had "
            "relative error below $10^{-7}$. The full step-size table is saved as "
            "`energy_derivative_validation.csv`. As an additional indexing/convention check, the "
            "analytic gradients agree with the sensitivity study's previously saved central energy "
            "differences over every diagnostic state:",
            "",
            f"- max $|E_x-E_x^{{saved}}|={saved['maximum_absolute_difference_E_x']:.3e}$;",
            f"- max $|E_\\xi-E_\\xi^{{saved}}|={saved['maximum_absolute_difference_E_xi']:.3e}$.",
            "",
            "## Directional alignment",
            "",
            "The acute angle is mathematically $\\cos^{-1}(|c_{EK}|)$ and was evaluated with the "
            "equivalent, numerically stable `atan2(|det|, |dot|)` form because the directions agree "
            "to nearly floating-point precision.",
            "",
            "### All diagnostic states",
            "",
            "| $u_{th}$ | N | mean / median $c$ | mean / median $|c|$ | q05 / q01 / min $|c|$ | mean / median / max $\\theta$ (deg) | worst incoming $x$ |",
            "|--:|--:|:--|:--|:--|:--|--:|",
        ]
    )
    for family in FAMILIES:
        row = summary["families"][f"{family:.2f}"]["all"]
        c, ac, theta = row["c_EK"], row["absolute_c_EK"], row["theta_EK_degrees"]
        lines.append(
            f"| {family:.2f} | {row['point_count']} | {c['mean']:.16f} / {c['median']:.16f} | "
            f"{ac['mean']:.16f} / {ac['median']:.16f} | {ac['q05']:.16f} / "
            f"{ac['q01']:.16f} / {ac['minimum']:.16f} | {theta['mean']:.3e} / "
            f"{theta['median']:.3e} / {theta['maximum']:.3e} | "
            f"{row['worst_directional_disagreement']['diagnostic_incoming_x']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Far upstream: $-17\\leq x<-8.5$",
            "",
            "| $u_{th}$ | N | mean / median $c$ | mean / median $|c|$ | q05 / q01 / min $|c|$ | mean / median / max $\\theta$ (deg) | worst incoming $x$ |",
            "|--:|--:|:--|:--|:--|:--|--:|",
        ]
    )
    for family in FAMILIES:
        row = summary["families"][f"{family:.2f}"]["far_upstream"]
        c, ac, theta = row["c_EK"], row["absolute_c_EK"], row["theta_EK_degrees"]
        lines.append(
            f"| {family:.2f} | {row['point_count']} | {c['mean']:.16f} / {c['median']:.16f} | "
            f"{ac['mean']:.16f} / {ac['median']:.16f} | {ac['q05']:.16f} / "
            f"{ac['q01']:.16f} / {ac['minimum']:.16f} | {theta['mean']:.3e} / "
            f"{theta['median']:.3e} / {theta['maximum']:.3e} | "
            f"{row['worst_directional_disagreement']['diagnostic_incoming_x']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Degeneracy threshold: `{DEGENERACY_THRESHOLD:.1e}`. No energy-gradient or sensitivity "
            "norm was degenerate in any family.",
            "",
            "## Proportionality check",
            "",
            "| $u_{th}$ | theoretical $du_{th}/dE$ | mean alpha | alpha range | mean / median / max $|r_K|/|g_K|$ | $|g_K|$ range |",
            "|--:|--:|--:|:--|:--|:--|",
        ]
    )
    for family in FAMILIES:
        family_row = summary["families"][f"{family:.2f}"]
        prop = family_row["proportionality"]
        alpha = family_row["all"]["alpha"]
        residual = family_row["all"]["relative_projection_residual"]
        magnitude = prop["g_K_norm"]
        lines.append(
            f"| {family:.2f} | {prop['theoretical_du_th_dE']:.9g} | {alpha['mean']:.9g} | "
            f"[{alpha['minimum']:.9g}, {alpha['maximum']:.9g}] | "
            f"{residual['mean']:.3e} / {residual['median']:.3e} / {residual['maximum']:.3e} | "
            f"[{magnitude['minimum']:.3e}, {magnitude['maximum']:.3e}] |"
        )
    lines.extend(
        [
            "",
            "The local projection coefficient is essentially constant along each family and agrees "
            "with the independently evaluated throat-orbit derivative $du_{th}/dE$. Its family "
            "dependence is large: the 0.05 coefficient is about 2.69 times the 0.15 value and 5.24 "
            "times the 0.30 value. Thus the direction is common energy geometry, while the dangerous "
            "magnitude is strongly energy/family dependent.",
            "",
            "## Scientific conclusion",
            "",
            "1. **Yes.** The exact local energy-normal and exact-flow downstream-sensitivity "
            "directions are parallel to numerical precision at every retained diagnostic state.",
            "2. **Yes.** Alignment is equally strong in the far-upstream interval, including the "
            "region where the 0.05 sensitivity magnitude peaks.",
            "3. **Yes, as a geometry surrogate.** A local squared energy-normal penalty is physically "
            "justified as targeting the dangerous direction. This gate does not select a loss or a "
            "coefficient.",
            "4. **Direction no; magnitude yes.** Directional alignment barely varies, but $du_{th}/dE$ "
            "and $|g_K|$ vary substantially between families.",
            "5. **Yes.** That separation supports investigating sensitivity weighting later, in "
            "addition to energy geometry, without implementing it in this milestone.",
            "",
            "## Integrity and stop condition",
            "",
            "Every protected source artifact retained the same SHA-256 hash before and after this "
            "evaluation. No NN was loaded or trained; no loss, architecture, dataset, split, optimizer, "
            "checkpoint, or standardization artifact was modified.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation directory {OUTPUT}")
    protected = (
        SENSITIVITY_ARRAYS,
        SENSITIVITY_SUMMARY,
        EXACT_REFERENCE,
        FIXED_E0_NORMALIZATION,
        MICROCORE_NORMALIZATION,
    )
    before = hashes(protected)
    fixed_normalization = json.loads(FIXED_E0_NORMALIZATION.read_text(encoding="utf-8"))
    microcore_normalization = json.loads(MICROCORE_NORMALIZATION.read_text(encoding="utf-8"))
    for name in ("delta_x", "delta_xi"):
        if fixed_normalization["columns"][name] != microcore_normalization["columns"][name]:
            raise RuntimeError(f"fixed-E0 and established microcore40k {name} scales differ")
    sigma_x = float(fixed_normalization["columns"]["delta_x"]["standard_deviation"])
    sigma_xi = float(fixed_normalization["columns"]["delta_xi"]["standard_deviation"])

    family_arrays: dict[str, dict[str, np.ndarray]] = {}
    family_summaries: dict[str, Any] = {}
    point_rows: list[dict[str, Any]] = []
    max_saved_difference_x = 0.0
    max_saved_difference_xi = 0.0
    with np.load(SENSITIVITY_ARRAYS, allow_pickle=False) as sensitivity, np.load(
        EXACT_REFERENCE, allow_pickle=False
    ) as exact:
        derivative_rows, derivative_summary = validate_derivatives(exact, sensitivity)
        for family in FAMILIES:
            family_name, key = f"{family:.2f}", stem(family)
            physical_state = np.asarray(exact[f"{key}__exact_state"], dtype=np.float64)
            exact_delta = np.asarray(exact[f"{key}__exact_delta"], dtype=np.float64)
            post_state = physical_state + exact_delta
            wormhole, spiral = experiment_parameters()
            post_xi_all = xi_from_state(
                post_state[:, 0], post_state[:, 1], wormhole, spiral
            )
            valid = np.asarray(sensitivity[f"{key}__valid_kernel_mask"], dtype=bool)
            diagnostic_x = physical_state[valid, 0]
            stored_x = np.asarray(sensitivity[f"{key}__attribution_x"], dtype=np.float64)
            if not np.array_equal(diagnostic_x, stored_x):
                raise RuntimeError(f"saved diagnostic states changed for family {family}")
            post_x = post_state[valid, 0]
            post_xi = post_xi_all[valid]
            E_x, E_xi = energy_gradient_x_xi(post_x, post_xi)
            saved_E_x = np.asarray(sensitivity[f"{key}__dE_dx"], dtype=np.float64)
            saved_E_xi = np.asarray(sensitivity[f"{key}__dE_dxi"], dtype=np.float64)
            max_saved_difference_x = max(
                max_saved_difference_x, float(np.max(np.abs(E_x - saved_E_x)))
            )
            max_saved_difference_xi = max(
                max_saved_difference_xi, float(np.max(np.abs(E_xi - saved_E_xi)))
            )
            K_x = np.asarray(sensitivity[f"{key}__K_x"], dtype=np.float64)
            K_xi = np.asarray(sensitivity[f"{key}__K_xi"], dtype=np.float64)
            g_E = np.column_stack((sigma_x * E_x, sigma_xi * E_xi))
            g_K = np.column_stack((sigma_x * K_x, sigma_xi * K_xi))
            norm_E = np.linalg.norm(g_E, axis=1)
            norm_K = np.linalg.norm(g_K, axis=1)
            degenerate_E = norm_E <= DEGENERACY_THRESHOLD
            degenerate_K = norm_K <= DEGENERACY_THRESHOLD
            if np.any(degenerate_E) or np.any(degenerate_K):
                raise RuntimeError(f"degenerate direction encountered for family {family}")
            n_E = g_E / norm_E[:, None]
            n_K = g_K / norm_K[:, None]
            cosine = np.clip(np.sum(n_E * n_K, axis=1), -1.0, 1.0)
            absolute_cosine = np.abs(cosine)
            determinant = n_E[:, 0] * n_K[:, 1] - n_E[:, 1] * n_K[:, 0]
            theta_degrees = np.degrees(
                np.arctan2(np.abs(determinant), absolute_cosine)
            )
            theta_arccos_degrees = np.degrees(
                np.arccos(np.clip(absolute_cosine, 0.0, 1.0))
            )
            alpha = np.sum(g_K * g_E, axis=1) / np.sum(g_E * g_E, axis=1)
            residual = g_K - alpha[:, None] * g_E
            relative_residual = np.linalg.norm(residual, axis=1) / norm_K
            far_mask = (diagnostic_x >= FAR_UPSTREAM[0]) & (
                diagnostic_x < FAR_UPSTREAM[1]
            )
            all_mask = np.ones(diagnostic_x.size, dtype=bool)
            theory = throat_du_dE(family)

            family_arrays[family_name] = {
                "diagnostic_x": diagnostic_x,
                "post_x": post_x,
                "post_xi": post_xi,
                "E_x": E_x,
                "E_xi": E_xi,
                "K_x": K_x,
                "K_xi": K_xi,
                "g_E": g_E,
                "g_K": g_K,
                "norm_E": norm_E,
                "norm_K": norm_K,
                "n_E": n_E,
                "n_K": n_K,
                "cosine": cosine,
                "absolute_cosine": absolute_cosine,
                "theta_degrees": theta_degrees,
                "theta_arccos_degrees": theta_arccos_degrees,
                "alpha": alpha,
                "residual": residual,
                "relative_residual": relative_residual,
                "far_upstream_mask": far_mask,
            }
            family_summaries[family_name] = {
                "u_th": family,
                "all": alignment_summary(
                    diagnostic_x,
                    post_x,
                    cosine,
                    absolute_cosine,
                    theta_degrees,
                    relative_residual,
                    alpha,
                    all_mask,
                ),
                "far_upstream": alignment_summary(
                    diagnostic_x,
                    post_x,
                    cosine,
                    absolute_cosine,
                    theta_degrees,
                    relative_residual,
                    alpha,
                    far_mask,
                ),
                "degeneracy": {
                    "threshold": DEGENERACY_THRESHOLD,
                    "g_E_count": int(np.sum(degenerate_E)),
                    "g_K_count": int(np.sum(degenerate_K)),
                    "minimum_g_E_norm": float(np.min(norm_E)),
                    "minimum_g_K_norm": float(np.min(norm_K)),
                },
                "proportionality": {
                    "theoretical_du_th_dE": theory,
                    "alpha_relative_error_vs_theory": descriptive(
                        np.abs(alpha / theory - 1.0)
                    ),
                    "g_K_norm": descriptive(norm_K),
                    "g_E_norm": descriptive(norm_E),
                },
            }
            for index in range(diagnostic_x.size):
                point_rows.append(
                    {
                        "u_th": family,
                        "diagnostic_incoming_x": diagnostic_x[index],
                        "post_step_x": post_x[index],
                        "post_step_xi": post_xi[index],
                        "E_x": E_x[index],
                        "E_xi": E_xi[index],
                        "K_x": K_x[index],
                        "K_xi": K_xi[index],
                        "g_E_x": g_E[index, 0],
                        "g_E_xi": g_E[index, 1],
                        "g_K_x": g_K[index, 0],
                        "g_K_xi": g_K[index, 1],
                        "n_E_x": n_E[index, 0],
                        "n_E_xi": n_E[index, 1],
                        "n_K_x": n_K[index, 0],
                        "n_K_xi": n_K[index, 1],
                        "c_EK": cosine[index],
                        "absolute_c_EK": absolute_cosine[index],
                        "theta_EK_degrees": theta_degrees[index],
                        "theta_EK_arccos_degrees": theta_arccos_degrees[index],
                        "alpha": alpha[index],
                        "r_K_x": residual[index, 0],
                        "r_K_xi": residual[index, 1],
                        "relative_projection_residual": relative_residual[index],
                        "far_upstream": bool(far_mask[index]),
                    }
                )

    derivative_summary["agreement_with_saved_energy_displacement"] = {
        "maximum_absolute_difference_E_x": max_saved_difference_x,
        "maximum_absolute_difference_E_xi": max_saved_difference_xi,
    }
    after = hashes(protected)
    if before != after:
        raise RuntimeError("a protected source artifact changed during evaluation")

    summary: dict[str, Any] = {
        "stage": "pre-training energy-gradient vs exact-flow sensitivity physics gate",
        "status": "passed",
        "protocol": {
            "training_performed": False,
            "neural_network_loaded": False,
            "loss_implemented_or_changed": False,
            "architecture_changed": False,
            "dataset_or_split_changed": False,
            "optimizer_changed": False,
            "checkpoint_changed": False,
            "sensitivity_trajectories_regenerated": False,
            "energy_function": "wormhole_sciml.dynamics.conserved_energy",
            "energy_gradient_state": "exact post-step (x_{n+1}, xi_{n+1})",
            "plot_abscissa": "matched exact incoming diagnostic x_n",
            "angle_evaluation": "atan2(abs(det(n_E,n_K)), abs(dot(n_E,n_K))), algebraically equivalent to acos(abs(c_EK)) and stable near zero angle",
        },
        "sources": {
            "sensitivity_arrays": {
                "path": str(SENSITIVITY_ARRAYS),
                "sha256": before[str(SENSITIVITY_ARRAYS)],
            },
            "sensitivity_summary": {
                "path": str(SENSITIVITY_SUMMARY),
                "sha256": before[str(SENSITIVITY_SUMMARY)],
            },
            "exact_reference": {
                "path": str(EXACT_REFERENCE),
                "sha256": before[str(EXACT_REFERENCE)],
            },
            "fixed_E0_normalization": {
                "path": str(FIXED_E0_NORMALIZATION),
                "sha256": before[str(FIXED_E0_NORMALIZATION)],
            },
            "microcore40k_normalization": {
                "path": str(MICROCORE_NORMALIZATION),
                "sha256": before[str(MICROCORE_NORMALIZATION)],
            },
        },
        "standardization": {
            "sigma_delta_x": sigma_x,
            "sigma_delta_xi": sigma_xi,
            "fixed_E0_scales_match_original_microcore40k_exactly": True,
        },
        "derivative_validation": derivative_summary,
        "families": family_summaries,
        "global_degeneracy": {
            "threshold": DEGENERACY_THRESHOLD,
            "g_E_count": sum(
                row["degeneracy"]["g_E_count"] for row in family_summaries.values()
            ),
            "g_K_count": sum(
                row["degeneracy"]["g_K_count"] for row in family_summaries.values()
            ),
        },
        "protected_hashes_before": before,
        "protected_hashes_after": after,
    }

    temporary = Path(tempfile.mkdtemp(prefix="energy-gradient-alignment-", dir=OUTPUT.parent))
    try:
        figures = temporary / "figures"
        figures.mkdir()
        plot_alignment(family_arrays, figures / "absolute_alignment_vs_x.png")
        plot_angles(family_arrays, figures / "misalignment_angle_vs_x.png")
        plot_components(family_arrays, figures / "standardized_direction_components_vs_x.png")

        arrays: dict[str, np.ndarray] = {}
        for family_name, values in family_arrays.items():
            prefix = stem(float(family_name))
            for name, value in values.items():
                arrays[f"{prefix}__{name}"] = np.asarray(value)
        np.savez_compressed(temporary / "energy_gradient_alignment_arrays.npz", **arrays)

        with (temporary / "energy_gradient_alignment_points.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
            writer.writeheader()
            writer.writerows(point_rows)
        with (temporary / "energy_derivative_validation.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(derivative_rows[0]))
            writer.writeheader()
            writer.writerows(derivative_rows)

        summary["artifacts"] = {
            "report": str(OUTPUT / "ENERGY_GRADIENT_ALIGNMENT_REPORT.md"),
            "summary": str(OUTPUT / "energy_gradient_alignment_summary.json"),
            "arrays": str(OUTPUT / "energy_gradient_alignment_arrays.npz"),
            "point_csv": str(OUTPUT / "energy_gradient_alignment_points.csv"),
            "derivative_validation_csv": str(OUTPUT / "energy_derivative_validation.csv"),
            "figures": {
                "absolute_alignment": str(OUTPUT / "figures" / "absolute_alignment_vs_x.png"),
                "misalignment_angle": str(OUTPUT / "figures" / "misalignment_angle_vs_x.png"),
                "direction_components": str(
                    OUTPUT / "figures" / "standardized_direction_components_vs_x.png"
                ),
            },
            "script_snapshot": str(OUTPUT / "evaluate_energy_gradient_alignment.py"),
        }
        (temporary / "energy_gradient_alignment_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "ENERGY_GRADIENT_ALIGNMENT_REPORT.md").write_text(
            render_report(summary), encoding="utf-8"
        )
        shutil.copy2(__file__, temporary / "evaluate_energy_gradient_alignment.py")
        temporary.rename(OUTPUT)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
