#!/usr/bin/env python3
"""Calculate the main trajectories and create the scientific figures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Matplotlib writes a font cache. Keep that cache outside the project.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wormhole-sciml-matplotlib")

import matplotlib

matplotlib.use("Agg")  # Create files without opening a graphical window.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from wormhole_sciml import (
    SpiralParameters,
    WormholeParameters,
    conserved_energy,
    coordinate_visualization,
    energy_branch_for_state,
    integrate_trajectory,
    radial_acceleration,
    total_speed_squared,
    velocity_bounds,
    velocity_from_energy,
)


# All time integrations in this script use the same explicit solver settings.
SOLVER = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}
M_VALUES = (2, 4, 6, 8, 10)


def make_radial_dynamics(output_dir: Path) -> None:
    """Plot acceleration, radial velocity, and total speed versus time."""

    spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 2)
    cases = ((0.1, 20.0), (0.95, 10.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(M_VALUES)))
    figure, axes = plt.subplots(3, 2, figsize=(10, 10), constrained_layout=True)

    for column, (initial_velocity, end_time) in enumerate(cases):
        times = np.linspace(0.0, end_time, 1001)
        for color, m in zip(colors, M_VALUES):
            wormhole = WormholeParameters(m=m)
            solution = integrate_trajectory(
                initial_state=(0.0, initial_velocity),
                t_span=(0.0, end_time),
                wormhole=wormhole,
                spiral=spiral,
                t_eval=times,
                **SOLVER,
            )

            # Derived quantities are calculated from the solver's l(t), v(t).
            acceleration = radial_acceleration(
                solution.y[0], solution.y[1], wormhole, spiral
            )
            total_speed = np.sqrt(
                total_speed_squared(
                    solution.y[0], solution.y[1], wormhole, spiral
                )
            )

            axes[0, column].plot(times, acceleration, color=color, label=f"m={m}")
            axes[1, column].plot(times, solution.y[1], color=color)
            axes[2, column].plot(times, total_speed, color=color)

        axes[0, column].set_title(f"$l_0=0$, $v_0={initial_velocity}$")
        axes[0, column].legend(frameon=False)
        axes[2, column].set_xlabel("laboratory time $t$")

    for row, label in enumerate(("$d^2l/dt^2$", "$dl/dt$", "$v_{tot}$")):
        for column in range(2):
            axes[row, column].set_ylabel(label)
            axes[row, column].grid(alpha=0.2)

    figure.suptitle(
        "Radial dynamics\n"
        "$\\omega=1$, $\\alpha=-1.25$, $b_0=1$, $\\theta=\\pi/2$"
    )
    figure.savefig(output_dir / "radial_dynamics.png", dpi=180)
    plt.close(figure)


def make_phase_space_energy_curves(output_dir: Path) -> None:
    """Plot several conserved-energy trajectories inside the allowed region."""

    spiral = SpiralParameters(omega=1.0, alpha=-2.0, theta=math.pi / 2)
    initial_velocities = np.linspace(0.05, 0.75, 8)
    colors = plt.cm.plasma(np.linspace(0.08, 0.88, len(initial_velocities)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)

    for axis, m, limit in zip(axes, (2, 10), (20.0, 10.0)):
        wormhole = WormholeParameters(m=m)
        proper_radius = np.linspace(-limit, limit, 1201)
        lower, upper = velocity_bounds(proper_radius, wormhole, spiral)
        axis.fill_between(
            proper_radius,
            lower,
            upper,
            color="#9db7d5",
            alpha=0.45,
            label="timelike region",
        )

        # Each v0 sets E at l=0. Equation (14) then gives the whole phase curve.
        for color, initial_velocity in zip(colors, initial_velocities):
            energy = float(
                conserved_energy(0.0, initial_velocity, wormhole, spiral)
            )
            branch = energy_branch_for_state(
                0.0, initial_velocity, wormhole, spiral
            )
            velocity = velocity_from_energy(
                proper_radius,
                energy,
                wormhole,
                spiral,
                branch=branch,
            )
            axis.plot(
                proper_radius,
                velocity,
                color=color,
                lw=1.5,
                label=f"$v_0={initial_velocity:.2f}$",
            )

        axis.set(
            xlim=(-limit, limit),
            ylim=(0.0, 0.82),
            xlabel="proper radial coordinate $l$",
            title=f"m={m}",
        )
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("radial velocity $dl/dt$")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
        fontsize=8,
        ncol=5,
    )
    figure.suptitle(
        "Conserved-energy trajectories in the timelike phase space\n"
        "$\\omega=1$, $\\alpha=-2$, $b_0=1$, $\\theta=\\pi/2$"
    )
    figure.savefig(output_dir / "phase_space_energy_curves.png", dpi=180)
    plt.close(figure)


def make_vector_fields(output_dir: Path) -> None:
    """Plot the phase-space direction field (dl/dt, dv/dt) = (v, a)."""

    spiral = SpiralParameters(omega=1.0, alpha=-2.0, theta=math.pi / 6)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for axis, m, limit in zip(axes, (2, 10), (10.0, 8.0)):
        wormhole = WormholeParameters(m=m)
        l_values = np.linspace(0.05, limit, 24)
        v_values = np.linspace(0.02, 0.92, 23)
        l_grid, v_grid = np.meshgrid(l_values, v_values)

        dl_dt = v_grid
        dv_dt = radial_acceleration(l_grid, v_grid, wormhole, spiral)

        # Normalize arrows so that direction remains visible where |a| is large.
        length = np.hypot(dl_dt, dv_dt)
        direction_l = np.divide(dl_dt, length, out=np.zeros_like(dl_dt), where=length > 0)
        direction_v = np.divide(dv_dt, length, out=np.zeros_like(dv_dt), where=length > 0)

        boundary_l = np.linspace(0.0, limit, 800)
        lower, upper = velocity_bounds(boundary_l, wormhole, spiral)
        axis.fill_between(
            boundary_l,
            np.maximum(lower, 0.0),
            upper,
            color="#9db7d5",
            alpha=0.45,
        )
        axis.quiver(
            l_grid,
            v_grid,
            direction_l,
            direction_v,
            color="#365f91",
            angles="xy",
            scale_units="xy",
            scale=9.0,
            width=0.0025,
            pivot="mid",
        )
        axis.set(
            xlim=(0.0, limit),
            ylim=(0.0, 0.94),
            xlabel="proper radial coordinate $l$",
            title=f"m={m}",
        )
        axis.grid(alpha=0.15)

    axes[0].set_ylabel("radial velocity $v$")
    figure.suptitle(
        "Phase-space vector field $(dl/dt,dv/dt)=(v,a)$\n"
        "$\\omega=1$, $\\alpha=-2$, $b_0=1$, $\\theta=\\pi/6$"
    )
    figure.savefig(output_dir / "vector_fields.png", dpi=180)
    plt.close(figure)


def make_outgoing_trajectory_3d(output_dir: Path) -> None:
    """Plot one outgoing trajectory and a phase-shifted particle jet in 3-D."""

    wormhole = WormholeParameters(m=2)
    spiral = SpiralParameters(omega=1.0, alpha=-2.0, theta=math.pi / 6)
    times = np.linspace(0.0, 100.0, 1801)
    solution = integrate_trajectory(
        initial_state=(0.0, 0.9),
        t_span=(0.0, 100.0),
        wormhole=wormhole,
        spiral=spiral,
        t_eval=times,
        **SOLVER,
    )

    figure = plt.figure(figsize=(11, 5.5))
    single_axis = figure.add_subplot(1, 2, 1, projection="3d")
    jet_axis = figure.add_subplot(1, 2, 2, projection="3d")

    single = coordinate_visualization(
        times,
        solution.y[0],
        wormhole,
        spiral,
        frame="laboratory",
        radial_scale="proper",
    )
    single_axis.plot(single[:, 0], single[:, 1], single[:, 2], lw=1.8)

    # The same radial solution is copied with twelve equally spaced phases.
    for phase in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
        particle = coordinate_visualization(
            times,
            solution.y[0],
            wormhole,
            spiral,
            phase=phase,
            frame="laboratory",
            radial_scale="proper",
        )
        jet_axis.plot(particle[:, 0], particle[:, 1], particle[:, 2], lw=1.1)

    for axis, title in (
        (single_axis, "single particle"),
        (jet_axis, "12 phase-shifted particles"),
    ):
        axis.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title=title)
        axis.view_init(elev=18, azim=-62)
        axis.set_box_aspect((1.0, 1.0, 1.35))
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.zaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(labelsize=8, pad=1)

    figure.suptitle(
        "Outgoing laboratory-frame trajectories\n"
        "$\\omega=1$, $\\alpha=-2$, $b_0=1$, $\\theta=\\pi/6$, "
        "$m=2$, $l_0=0$, $v_0=0.9$"
    )
    figure.subplots_adjust(top=0.79, bottom=0.06, wspace=0.08)
    figure.savefig(
        output_dir / "outgoing_trajectory_3d.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def make_through_wormhole_3d(output_dir: Path) -> None:
    """Plot a trajectory entering one side and leaving the other side."""

    wormhole = WormholeParameters(m=2)
    spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 6)
    times = np.linspace(0.0, 150.0, 2201)
    solution = integrate_trajectory(
        initial_state=(-30.0, 0.827),
        t_span=(0.0, 150.0),
        wormhole=wormhole,
        spiral=spiral,
        t_eval=times,
        **SOLVER,
    )

    figure = plt.figure(figsize=(11, 5.5))
    local_axis = figure.add_subplot(1, 2, 1, projection="3d")
    lab_axis = figure.add_subplot(1, 2, 2, projection="3d")

    local = coordinate_visualization(
        times,
        solution.y[0],
        wormhole,
        spiral,
        frame="co_rotating",
        radial_scale="proper",
    )
    local_axis.plot(local[:, 0], local[:, 1], local[:, 2], lw=1.5)

    for phase in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
        particle = coordinate_visualization(
            times,
            solution.y[0],
            wormhole,
            spiral,
            phase=phase,
            frame="laboratory",
            radial_scale="proper",
        )
        lab_axis.plot(particle[:, 0], particle[:, 1], particle[:, 2], lw=1.0)

    for axis, title in (
        (local_axis, "co-rotating frame: one particle"),
        (lab_axis, "laboratory frame: 12 particles"),
    ):
        axis.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title=title)
        axis.view_init(elev=20, azim=-60)
        axis.set_box_aspect((1.0, 1.0, 1.35))
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.zaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(labelsize=8, pad=1)

    figure.suptitle(
        "Motion through the wormhole\n"
        "$\\omega=1$, $\\alpha=-1.25$, $b_0=1$, $\\theta=\\pi/6$, "
        "$m=2$, $l_0=-30$, $v_0=0.827$"
    )
    figure.subplots_adjust(top=0.79, bottom=0.06, wspace=0.08)
    figure.savefig(
        output_dir / "through_wormhole_3d.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def make_through_wormhole_dynamics(output_dir: Path) -> None:
    """Plot radial and total speeds for the through-going trajectory."""

    wormhole = WormholeParameters(m=2)
    spiral = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 6)
    times = np.linspace(0.0, 200.0, 2001)
    solution = integrate_trajectory(
        initial_state=(-30.0, 0.827),
        t_span=(0.0, 200.0),
        wormhole=wormhole,
        spiral=spiral,
        t_eval=times,
        **SOLVER,
    )
    total_speed = np.sqrt(
        total_speed_squared(solution.y[0], solution.y[1], wormhole, spiral)
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot(times, solution.y[1], color="#365f91")
    axes[0].set(xlabel="laboratory time $t$", ylabel="$dl/dt$")
    axes[1].plot(times, total_speed, color="#365f91")
    axes[1].set(xlabel="laboratory time $t$", ylabel="$v_{tot}$", ylim=(0, 1.01))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Velocity during motion through the wormhole\n"
        "$\\omega=1$, $\\alpha=-1.25$, $b_0=1$, $\\theta=\\pi/6$, "
        "$m=2$, $l_0=-30$, $v_0=0.827$"
    )
    figure.savefig(output_dir / "through_wormhole_dynamics.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/figures")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # One call per figure keeps this entry point easy to edit.
    make_radial_dynamics(args.output_dir)
    make_phase_space_energy_curves(args.output_dir)
    make_vector_fields(args.output_dir)
    make_outgoing_trajectory_3d(args.output_dir)
    make_through_wormhole_3d(args.output_dir)
    make_through_wormhole_dynamics(args.output_dir)

    settings = {
        "solver": SOLVER,
        "radial_dynamics.png": {
            "m": list(M_VALUES),
            "initial_states": [[0.0, 0.1], [0.0, 0.95]],
        },
        "phase_space_energy_curves.png": {
            "m": [2, 10],
            "initial_velocities_at_l_0": np.linspace(0.05, 0.75, 8).tolist(),
        },
        "vector_fields.png": {"m": [2, 10]},
        "outgoing_trajectory_3d.png": {
            "initial_state": [0.0, 0.9],
            "particles": 12,
        },
        "through_wormhole_3d.png": {
            "initial_state": [-30.0, 0.827],
            "particles": 12,
        },
        "coordinate_note": (
            "The 3-D figures use signed proper l as the historical plotting "
            "radius; they are coordinate pictures, not isometric embeddings."
        ),
    }
    (args.output_dir / "figure_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
