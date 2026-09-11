#!/usr/bin/env python3
"""Run quantitative scientific validation beyond the fast unit suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from wormhole_sciml import (
    SpiralParameters,
    WormholeParameters,
    conserved_energy,
    energy_branch_for_state,
    integrate_trajectory,
    radial_acceleration,
    radial_acceleration_from_metric,
    terminal_radial_velocity,
    timelike_margin,
    total_speed_squared,
    velocity_from_energy,
)


SOLVER = {
    "method": "DOP853",
    "rtol": 1e-11,
    "atol": 1e-13,
}


def first_crossing_time(times: np.ndarray, values: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(values >= level)
    if not len(indices):
        raise AssertionError(f"trajectory did not reach level {level}")
    index = int(indices[0])
    if index == 0:
        return float(times[0])
    fraction = (level - values[index - 1]) / (values[index] - values[index - 1])
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def run_validation() -> dict[str, object]:
    equatorial = SpiralParameters(omega=1.0, alpha=-1.25, theta=math.pi / 2)
    m2 = WormholeParameters(throat_radius=1.0, m=2)

    legacy_energy = 0.0072950858953519365
    calculated_energy = float(conserved_energy(-40.0, 0.7995025, m2, equatorial))
    energy_notebook_error = abs(calculated_energy - legacy_energy)

    l_grid = np.linspace(-12.0, 12.0, 121)[:, None]
    v_grid = np.linspace(0.02, 0.98, 73)[None, :]
    acceleration_errors: dict[str, float] = {}
    for m in (2, 4, 6, 8, 10):
        geometry = WormholeParameters(m=m)
        closed = radial_acceleration(l_grid, v_grid, geometry, equatorial)
        metric = radial_acceleration_from_metric(l_grid, v_grid, geometry, equatorial)
        acceleration_errors[str(m)] = float(np.max(np.abs(closed - metric)))

    times = np.linspace(0.0, 20.0, 1001)
    plateau_times: dict[str, float] = {}
    energy_drifts: dict[str, float] = {}
    max_total_speeds: dict[str, float] = {}
    convergence_errors: dict[str, float] = {}
    terminal = terminal_radial_velocity(equatorial)
    midpoint = 0.5 * (0.1 + terminal)
    for m in (2, 4, 6, 8, 10):
        geometry = WormholeParameters(m=m)
        reference = integrate_trajectory(
            (0.0, 0.1),
            (0.0, 20.0),
            geometry,
            equatorial,
            t_eval=times,
            **SOLVER,
        )
        tighter = integrate_trajectory(
            (0.0, 0.1),
            (0.0, 20.0),
            geometry,
            equatorial,
            t_eval=times,
            rtol=2e-13,
            atol=2e-15,
            method="DOP853",
        )
        energy = conserved_energy(reference.y[0], reference.y[1], geometry, equatorial)
        energy_drifts[str(m)] = float(np.max(np.abs(energy / energy[0] - 1.0)))
        convergence_errors[str(m)] = float(np.max(np.abs(reference.y - tighter.y)))
        plateau_times[str(m)] = first_crossing_time(times, reference.y[1], midpoint)
        max_total_speeds[str(m)] = float(
            np.sqrt(
                np.max(
                    total_speed_squared(
                        reference.y[0], reference.y[1], geometry, equatorial
                    )
                )
            )
        )

    # Independent energy first integral versus the integrated second-order ODE.
    geometry = WormholeParameters(m=10)
    phase_times = np.linspace(0.0, 20.0, 801)
    phase_solution = integrate_trajectory(
        (0.0, 0.1),
        (0.0, 20.0),
        geometry,
        equatorial,
        t_eval=phase_times,
        **SOLVER,
    )
    phase_energy = float(conserved_energy(0.0, 0.1, geometry, equatorial))
    phase_branch = energy_branch_for_state(0.0, 0.1, geometry, equatorial)
    phase_velocity = velocity_from_energy(
        phase_solution.y[0],
        phase_energy,
        geometry,
        equatorial,
        branch=phase_branch,
    )
    phase_integral_error = float(np.nanmax(np.abs(phase_velocity - phase_solution.y[1])))

    # Paper figures 7-8: transit from one asymptotic region through the throat.
    traversal_spiral = SpiralParameters(
        omega=1.0, alpha=-1.25, theta=math.pi / 6
    )
    traversal_times = np.linspace(0.0, 200.0, 2001)
    traversal = integrate_trajectory(
        (-30.0, 0.827),
        (0.0, 200.0),
        m2,
        traversal_spiral,
        t_eval=traversal_times,
        **SOLVER,
    )
    crossing_index = int(np.flatnonzero(traversal.y[0] >= 0.0)[0])
    throat_crossing_time = float(traversal.t[crossing_index])
    traversal_min_margin = float(
        np.min(
            timelike_margin(
                traversal.y[0], traversal.y[1], m2, traversal_spiral
            )
        )
    )
    traversal_peak_velocity = float(np.max(traversal.y[1]))
    traversal_final_velocity = float(traversal.y[1, -1])

    asymptotic_energy = float(conserved_energy(0.0, 0.1, m2, equatorial))
    asymptotic_branch = energy_branch_for_state(0.0, 0.1, m2, equatorial)
    asymptotic_velocity = float(
        velocity_from_energy(
            1e5,
            asymptotic_energy,
            m2,
            equatorial,
            branch=asymptotic_branch,
        )
    )
    asymptotic_terminal_error = abs(asymptotic_velocity - terminal)

    metrics: dict[str, object] = {
        "solver": SOLVER,
        "legacy_energy": legacy_energy,
        "calculated_energy": calculated_energy,
        "energy_notebook_absolute_error": energy_notebook_error,
        "acceleration_metric_identity_max_abs_error": acceleration_errors,
        "m_halfway_velocity_times": plateau_times,
        "trajectory_relative_energy_drift": energy_drifts,
        "tight_tolerance_trajectory_max_abs_difference": convergence_errors,
        "outward_case_max_total_speed": max_total_speeds,
        "energy_integral_vs_ode_max_abs_velocity_error": phase_integral_error,
        "traversal": {
            "initial_state": [-30.0, 0.827],
            "throat_crossing_time": throat_crossing_time,
            "peak_radial_velocity": traversal_peak_velocity,
            "radial_velocity_at_t_200": traversal_final_velocity,
            "minimum_timelike_margin": traversal_min_margin,
        },
        "velocity_at_l_1e5": asymptotic_velocity,
        "terminal_velocity": terminal,
        "asymptotic_terminal_absolute_error": asymptotic_terminal_error,
    }

    assert energy_notebook_error < 3e-13
    assert max(acceleration_errors.values()) < 5e-13
    ordered_times = [plateau_times[str(m)] for m in (2, 4, 6, 8, 10)]
    assert all(later > earlier for earlier, later in zip(ordered_times, ordered_times[1:]))
    assert max(energy_drifts.values()) < 2e-8
    assert max(convergence_errors.values()) < 2e-8
    assert max(max_total_speeds.values()) < 1.0
    assert phase_integral_error < 2e-8
    assert traversal.y[0, -1] > 0.0
    assert traversal_min_margin > 0.0
    assert asymptotic_terminal_error < 2e-9
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/validation.json"),
    )
    args = parser.parse_args()
    metrics = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nValidation passed; wrote {args.output}")


if __name__ == "__main__":
    main()
