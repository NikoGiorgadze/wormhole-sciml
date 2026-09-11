#!/usr/bin/env python3
"""Build the small figure set used by the public README.

The large numerical inputs live under ``output/`` and are intentionally not
committed.  This script turns the retained experiment arrays and tables into
the compact, versioned figures in ``figures/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import velocity_bounds
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi
from wormhole_sciml.physics_gate import experiment_parameters


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"

REFERENCE_BANK = (
    OUTPUT
    / "phase_b_complete_orbit_banks"
    / "banks"
    / "phase_b_stress_reference_orbits.npz"
)
FINITE_TIME_SOURCE = (
    OUTPUT
    / "finite_time_hybrid_phase_space_audit"
    / "arrays"
    / "reference_phase_space_predictions.npz"
)
ERROR_GROWTH_SOURCE = (
    OUTPUT
    / "direct_vs_recursive_rollouts"
    / "tables"
    / "per_k_metrics.csv"
)
LOCAL_ENERGY_ROLLOUT_SOURCE = (
    OUTPUT
    / "model_a_x_xi_energy_microcore40k_comparison"
    / "energy_xi_rollout_arrays.npz"
)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"required retained experiment artifact is missing: {path}\n"
            "Regenerate it with the experiment script listed in scripts/README.md."
        )
    return path


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_reference_geometry() -> Path:
    with np.load(require(REFERENCE_BANK), allow_pickle=False) as bank:
        colors = plt.cm.viridis(np.linspace(0.08, 0.95, bank["u_th"].size))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        for index, (u_th, color) in enumerate(zip(bank["u_th"], colors)):
            times = np.linspace(0.0, float(bank["t_right"][index]), 1201)
            states = evaluate_saved_orbit_x_u_xi(bank, index, times)
            x, xi = states[:, 0], states[:, 2]
            label = rf"$u_{{\rm th}}={u_th:.2f}$"
            axes[0].plot(x, xi, color=color, linewidth=1.6, label=label)
            incoming = x <= -8.5
            axes[1].plot(x[incoming], xi[incoming], color=color, linewidth=1.6, label=label)
    axes[0].set_title("Complete trajectories")
    axes[1].set_title("Compressed incoming branch")
    for axis in axes:
        axis.set_xlabel(r"position $x$")
        axis.set_ylabel(r"normalized velocity $\xi$")
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Exact trajectory-family geometry")
    fig.tight_layout()
    destination = FIGURES / "stress_reference_x_xi.png"
    _finish(fig, destination)
    return destination


def _add_physical_corridor(axis: plt.Axes) -> None:
    wormhole, spiral = experiment_parameters()
    position = np.linspace(-17.0, 17.0, 1001)
    lower, upper = velocity_bounds(position, wormhole, spiral)
    axis.fill_between(
        position,
        lower,
        upper,
        color="#dce6ef",
        alpha=0.55,
        label="Timelike region",
    )
    axis.plot(position, lower, color="#718096", linewidth=0.9)
    axis.plot(position, upper, color="#718096", linewidth=0.9)


def build_local_energy_rollouts() -> Path:
    seed_colors = {101: "#2878b5", 202: "#e6861a", 303: "#8f63c7"}
    with np.load(require(LOCAL_ENERGY_ROLLOUT_SOURCE), allow_pickle=False) as data:
        reference_state = np.asarray(data["u_th_0p05__exact_full_state"])
        reference_coordinates = np.asarray(
            data["u_th_0p05__exact_full_coordinates"]
        )
        predicted = {
            seed: (
                np.asarray(
                    data[
                        f"u_th_0p05__full__fixed_E0__seed_{seed}__physical_state"
                    ]
                ),
                np.asarray(
                    data[
                        f"u_th_0p05__full__fixed_E0__seed_{seed}__coordinates"
                    ]
                ),
            )
            for seed in seed_colors
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    _add_physical_corridor(axes[0])
    axes[0].plot(
        reference_state[:, 0],
        reference_state[:, 1],
        color="black",
        linewidth=2.2,
        label="Numerical reference",
    )
    axes[1].plot(
        reference_coordinates[:, 0],
        reference_coordinates[:, 1],
        color="black",
        linewidth=2.2,
        label="Numerical reference",
    )

    for seed, color in seed_colors.items():
        physical_state, coordinates = predicted[seed]
        axes[0].plot(
            physical_state[:, 0],
            physical_state[:, 1],
            color=color,
            linewidth=1.6,
            label=f"Seed {seed}",
        )
        axes[1].plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=color,
            linewidth=1.6,
            label=f"Seed {seed}",
        )

    axes[0].set(
        title="Recursive rollout in physical phase space",
        xlabel=r"position $x$",
        ylabel=r"radial velocity $u$",
        xlim=(-17.4, 17.4),
    )
    axes[1].set(
        title="Recursive rollout in normalized phase space",
        xlabel=r"position $x$",
        ylabel=r"normalized velocity $\xi$",
        xlim=(-17.4, 17.4),
        ylim=(-0.37, 0.025),
    )
    for axis in axes:
        axis.grid(alpha=0.18)
    axes[1].legend(fontsize=8.1, loc="lower right", frameon=False)
    fig.suptitle(
        r"Fixed-energy local model on the difficult $u_{\rm th}=0.05$ orbit"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    destination = FIGURES / "local_energy_input_rollouts_u_th_0p05.png"
    _finish(fig, destination)
    return destination


def build_finite_time_reference() -> Path:
    with np.load(require(FINITE_TIME_SOURCE)) as arrays:
        s = arrays["u_th_0p05_elapsed_s"]
        exact_x = arrays["u_th_0p05_exact_x"]
        exact_xi = arrays["u_th_0p05_exact_xi"]
        predicted_x = arrays["u_th_0p05_lambda_0_seed_202_predicted_x"]
        predicted_xi = arrays["u_th_0p05_lambda_0_seed_202_predicted_xi"]

    reference_style = {"color": "black", "linewidth": 2.2, "label": "Numerical reference"}
    model_style = {
        "color": "#d1495b",
        "linewidth": 1.6,
        "linestyle": "--",
        "label": "Time-rescaled finite-time model",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    axes[0].plot(s, exact_x, **reference_style)
    axes[0].plot(s, predicted_x, **model_style)
    axes[0].axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].set(xlabel=r"elapsed time $s$", ylabel=r"position $x$")

    axes[1].plot(s, exact_xi, **reference_style)
    axes[1].plot(s, predicted_xi, **model_style)
    axes[1].set(xlabel=r"elapsed time $s$", ylabel=r"normalized velocity $\xi$")

    axes[2].plot(exact_x, exact_xi, **reference_style)
    axes[2].plot(predicted_x, predicted_xi, **model_style)
    axes[2].axvline(0.0, color="0.75", linewidth=0.8)
    axes[2].set(xlabel=r"position $x$", ylabel=r"normalized velocity $\xi$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(r"Direct finite-time prediction for the difficult $u_{\rm th}=0.05$ orbit", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    destination = FIGURES / "finite_time_reference_u_th_0p05.png"
    _finish(fig, destination)
    return destination


def _read_error_growth() -> dict[str, np.ndarray]:
    with require(ERROR_GROWTH_SOURCE).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = (
        "s",
        "common_count",
        "direct_hybrid_common_x_rmse",
        "recursive_local_common_x_rmse",
        "direct_hybrid_common_xi_rmse",
        "recursive_local_common_xi_rmse",
    )
    return {
        column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        for column in columns
    }


def build_direct_vs_recursive() -> Path:
    data = _read_error_growth()
    valid = data["common_count"] > 0
    s = data["s"][valid]
    direct_style = {"color": "#00798c", "linewidth": 2.0, "label": "Direct finite-time"}
    recursive_style = {"color": "#d1495b", "linewidth": 1.8, "label": "Recursive local"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharex=True)
    for axis, coordinate in zip(axes, ("x", "xi")):
        axis.plot(s, data[f"direct_hybrid_common_{coordinate}_rmse"][valid], **direct_style)
        axis.plot(s, data[f"recursive_local_common_{coordinate}_rmse"][valid], **recursive_style)
        axis.set_yscale("log")
        axis.set_xlabel(r"forecast horizon $s$")
        axis.set_ylabel(rf"{coordinate} RMSE")
        axis.grid(alpha=0.18)
    axes[0].axvline(22.0, color="0.4", linestyle=":", linewidth=1.1)
    axes[0].text(22.5, axes[0].get_ylim()[0] * 1.5, r"sustained $x$-RMSE crossover", fontsize=8)
    axes[0].legend(frameon=False)
    fig.suptitle("Direct and recursive error growth on the common-survivor cohort")
    fig.tight_layout()
    destination = FIGURES / "direct_vs_recursive_error_growth.png"
    _finish(fig, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=(
            "all",
            "geometry",
            "local-energy",
            "finite-time",
            "error-growth",
        ),
        default="all",
        help="build one figure family instead of the complete README set",
    )
    selection = parser.parse_args().only
    FIGURES.mkdir(parents=True, exist_ok=True)
    builders = {
        "geometry": build_reference_geometry,
        "local-energy": build_local_energy_rollouts,
        "finite-time": build_finite_time_reference,
        "error-growth": build_direct_vs_recursive,
    }
    chosen = builders.values() if selection == "all" else (builders[selection],)
    for builder in chosen:
        print(builder().relative_to(ROOT))


if __name__ == "__main__":
    main()
