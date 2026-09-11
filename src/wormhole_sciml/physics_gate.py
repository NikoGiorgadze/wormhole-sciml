"""Deterministic physics-only calibration for the Simple GEB ML experiment.

This module contains no machine-learning models or final dataset generation.
It maps the experiment specification onto the validated physics package,
calibrates the compact domain and local flow step, verifies the mathematical
exterior continuation, and freezes reference-only rollout suites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from . import dynamics as dynamics_module
from . import integrate as integrate_module
from .dynamics import (
    conserved_energy,
    radial_acceleration,
    radial_system,
    timelike_margin,
    total_speed_squared,
    velocity_bounds,
)
from .geometry import areal_radius
from .integrate import integrate_trajectory, integrate_vector_field_continuation
from .observables import effective_angular_velocity, terminal_radial_velocity
from .parameters import SpiralParameters, WormholeParameters


class GateBlocked(RuntimeError):
    """Raised when an unchanged scientific gate condition fails."""


@dataclass(frozen=True, slots=True)
class SolverSettings:
    method: str
    rtol: float
    atol: float
    max_step: float = math.inf

    def kwargs(self) -> dict[str, float | str]:
        return asdict(self)

    def metadata(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": "inf" if math.isinf(self.max_step) else self.max_step,
        }


PRODUCTION_SOLVER = SolverSettings("DOP853", 1e-10, 1e-12)
TIGHT_SOLVER = SolverSettings("DOP853", 2e-13, 2e-15)
VALIDATION_SOLVER = SolverSettings("DOP853", 1e-11, 1e-13)

DIAGNOSTIC_STEP = 0.01
PHYSICAL_XI_POINTS = 401
EXTERIOR_XI_POINTS = 501
H_CANDIDATES = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25)
CALIBRATION_STATE_COUNT = 2048
SURVEY_CAP = 100.0
SURVEY_BOUNDARY_FACTOR = 2.0

SEEDS = {
    "learning_step": 314159,
    "validation_suite": 271828,
    "sealed_test_suite": 161803,
}

ESCAPING_ROLLOUT_STATUSES = (
    "terminal_reached_by_T_i",
    "right_censored_at_T_i",
    "physical_failure_before_terminal_entry",
    "nonfinite_numerical_failure_before_terminal_entry",
)
BOUNDARY_ROLLOUT_STATUSES = (
    "finite_inside_common_domain_through_T_i",
    "premature_state_domain_exit_while_timelike",
    "physical_failure_by_T_i",
    "nonfinite_by_T_i",
)

PHYSICS_MANIFEST = (
    "src/wormhole_sciml/parameters.py",
    "src/wormhole_sciml/geometry.py",
    "src/wormhole_sciml/dynamics.py",
    "src/wormhole_sciml/observables.py",
    "src/wormhole_sciml/integrate.py",
    "src/wormhole_sciml/__init__.py",
    "pyproject.toml",
)


def experiment_parameters() -> tuple[WormholeParameters, SpiralParameters]:
    """Return the fixed experiment parameters without using package defaults."""

    return (
        WormholeParameters(throat_radius=1.0, m=2),
        SpiralParameters(omega=1.0, alpha=-2.0, theta=math.pi / 6.0),
    )


def state_geometry(
    x: Any,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return admissible-corridor center ``c`` and half-width ``d``."""

    lower, upper = velocity_bounds(x, wormhole, spiral)
    return 0.5 * (lower + upper), 0.5 * (upper - lower)


def state_geometry_derivative(
    x: Any,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact physical-coordinate derivatives ``dc/dx`` and ``dd/dx``.

    This differentiates the same closed-form null-boundary midpoint and
    half-width used by :func:`state_geometry`; it is not an independent
    dynamics implementation.
    """

    coordinate = np.asarray(x, dtype=np.float64)
    base = wormhole.throat_radius**wormhole.m + coordinate**wormhole.m
    radius_squared = base ** (2.0 / wormhole.m)
    radius_squared_derivative = (
        2.0
        * coordinate ** (wormhole.m - 1)
        * base ** (2.0 / wormhole.m - 1.0)
    )
    sine_squared = np.sin(spiral.theta) ** 2
    q = sine_squared * radius_squared
    q_derivative = sine_squared * radius_squared_derivative
    coefficient = 1.0 + q * spiral.alpha**2
    beta = spiral.alpha**2 - spiral.omega**2
    discriminant = 1.0 + q * beta
    root = np.sqrt(discriminant)
    center_derivative = (
        -spiral.omega * spiral.alpha * q_derivative / coefficient**2
    )
    half_width_derivative = q_derivative * (
        0.5 * beta * coefficient / root - spiral.alpha**2 * root
    ) / coefficient**2
    return (
        np.asarray(center_derivative, dtype=np.float64),
        np.asarray(half_width_derivative, dtype=np.float64),
    )


def state_from_xi(
    x: Any,
    xi: Any,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[np.ndarray, np.ndarray]:
    center, half_width = state_geometry(x, wormhole, spiral)
    return np.asarray(x, dtype=np.float64), np.asarray(
        center + half_width * np.asarray(xi, dtype=np.float64),
        dtype=np.float64,
    )


def xi_from_state(
    x: Any,
    u: Any,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> np.ndarray:
    center, half_width = state_geometry(x, wormhole, spiral)
    return np.asarray((np.asarray(u, dtype=np.float64) - center) / half_width)


def xi_time_derivative(
    x: Any,
    u: Any,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> np.ndarray:
    """Return ``d xi / dt`` along the validated radial dynamics.

    The result applies the chain rule to ``xi=(u-c(x))/d(x)`` using the
    established radial acceleration and the exact derivatives of the same
    corridor geometry used by :func:`xi_from_state`.
    """

    coordinate = np.asarray(x, dtype=np.float64)
    velocity = np.asarray(u, dtype=np.float64)
    center, half_width = state_geometry(coordinate, wormhole, spiral)
    center_prime, half_width_prime = state_geometry_derivative(
        coordinate, wormhole, spiral
    )
    acceleration = radial_acceleration(coordinate, velocity, wormhole, spiral)
    return np.asarray(
        (acceleration - center_prime * velocity) / half_width
        - (velocity - center) * half_width_prime * velocity / half_width**2,
        dtype=np.float64,
    )


def source_manifest(project_root: Path) -> dict[str, Any]:
    """Hash the ordered validated-physics source manifest reproducibly."""

    combined = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for relative in PHYSICS_MANIFEST:
        path = project_root / relative
        data = path.read_bytes()
        file_hashes[relative] = hashlib.sha256(data).hexdigest()
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
    return {
        "algorithm": "sha256(path + NUL + contents + NUL, ordered)",
        "sha256": combined.hexdigest(),
        "files": file_hashes,
    }


def package_mapping() -> dict[str, Any]:
    return {
        "R_m": "wormhole_sciml.areal_radius",
        "F": "wormhole_sciml.radial_acceleration",
        "radial_system": "wormhole_sciml.dynamics.radial_system",
        "physical_flow": "wormhole_sciml.integrate_trajectory",
        "exterior_flow": "wormhole_sciml.integrate_vector_field_continuation",
        "C": "wormhole_sciml.timelike_margin",
        "c_d_u_bounds": "midpoint/half-width of wormhole_sciml.velocity_bounds",
        "xi": "(u-c(x))/d(x)",
        "Omega": "wormhole_sciml.effective_angular_velocity",
        "u_infinity": "wormhole_sciml.terminal_radial_velocity",
        "v_total_squared": "wormhole_sciml.total_speed_squared",
        "energy": "wormhole_sciml.conserved_energy (C>0 only)",
        "coordinates": {"x": "l/b0=l", "s": "t/b0=t", "u": "dl/dt"},
        "parameters": {
            "throat_radius": 1.0,
            "m": 2,
            "alpha": -2.0,
            "omega": 1.0,
            "theta": "pi/6",
            "A": -2.0,
            "W": 1.0,
            "S": 0.25,
        },
    }


def _grid_count(limit: float, spacing: float) -> int:
    return int(round(2.0 * limit / spacing)) + 1


def force_envelope_scan(
    limit: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    x_spacing: float = 0.01,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate the specified physical phase-space acceleration envelope."""

    x = np.linspace(-limit, limit, _grid_count(limit, x_spacing), dtype=np.float64)
    xi = np.linspace(-0.99, 0.99, PHYSICAL_XI_POINTS, dtype=np.float64)
    center, half_width = state_geometry(x, wormhole, spiral)
    envelope = np.empty_like(x)
    omega_envelope = np.empty_like(x)
    for start in range(0, x.size, 512):
        stop = min(start + 512, x.size)
        u = center[start:stop, None] + half_width[start:stop, None] * xi[None, :]
        acceleration = radial_acceleration(x[start:stop, None], u, wormhole, spiral)
        omega = effective_angular_velocity(u, spiral)
        envelope[start:stop] = np.max(np.abs(acceleration), axis=1)
        omega_envelope[start:stop] = np.max(np.abs(omega), axis=1)

    f_star = float(np.max(envelope))
    threshold = 0.05 * f_star
    center_index = x.size // 2
    paired = np.maximum(envelope[center_index:], envelope[center_index::-1])
    bad = np.flatnonzero(paired > threshold)
    if bad.size == 0:
        x_acc = 0.0
    elif bad[-1] + 1 >= paired.size:
        x_acc = math.nan
    else:
        x_acc = float((bad[-1] + 1) * x_spacing)
    start_index = int(round(x_acc / x_spacing)) if math.isfinite(x_acc) else 0
    recrossing = bool(np.any(paired[start_index:] > threshold))
    summary = {
        "limit": float(limit),
        "x_spacing": x_spacing,
        "x_points": int(x.size),
        "xi_points": PHYSICAL_XI_POINTS,
        "xi_range": [-0.99, 0.99],
        "F_star": f_star,
        "threshold_5pct": threshold,
        "X_acc": x_acc,
        "threshold_recrossing": recrossing,
        "max_abs_Omega": float(np.max(omega_envelope)),
        "min_d": float(np.min(half_width)),
        "max_d": float(np.max(half_width)),
    }
    return summary, x, envelope


def calibrate_force_envelope(
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Apply the L-to-2L tail certification without weakening thresholds."""

    limit = 20.0
    attempts: list[dict[str, Any]] = []
    while limit <= 1280.0:
        base, _, _ = force_envelope_scan(limit, wormhole, spiral)
        expanded, x_expanded, envelope_expanded = force_envelope_scan(
            2.0 * limit, wormhole, spiral
        )
        outer_mask = np.abs(x_expanded) >= 1.8 * limit
        outer_max = float(np.max(envelope_expanded[outer_mask]))
        x_acc_change = abs(float(expanded["X_acc"]) - float(base["X_acc"]))
        passed = bool(
            math.isfinite(float(base["X_acc"]))
            and x_acc_change <= 0.02 + 1e-15
            and not bool(expanded["threshold_recrossing"])
            and outer_max <= 0.01 * float(expanded["F_star"])
        )
        attempt = {
            "L": limit,
            "base": base,
            "expanded": expanded,
            "X_acc_change": x_acc_change,
            "outer_1p8L_to_2L_max": outer_max,
            "outer_ratio_to_F_star": outer_max / float(expanded["F_star"]),
            "passed": passed,
        }
        attempts.append(attempt)
        if passed:
            return {
                "passed": True,
                "L": limit,
                "certification_extent": 2.0 * limit,
                "F_star": float(expanded["F_star"]),
                "X_acc": float(expanded["X_acc"]),
                "attempts": attempts,
            }
        limit *= 2.0
    raise GateBlocked("compact-domain tail certification failed through L=1280")


def diagnostic_times(end_time: float) -> np.ndarray:
    count = int(math.floor((end_time + 1e-12) / DIAGNOSTIC_STEP))
    return np.arange(count + 1, dtype=np.float64) * DIAGNOSTIC_STEP


def _first_dense_event(
    solution: Any,
    end_time: float,
    event_value: Callable[[float], float],
) -> float | None:
    times = diagnostic_times(end_time)
    values = np.asarray([event_value(float(time)) for time in times])
    hits = np.flatnonzero(values >= 0.0)
    if not hits.size:
        return None
    index = int(hits[0])
    if index == 0:
        return float(times[0])
    left, right = float(times[index - 1]), float(times[index])
    if values[index - 1] == 0.0:
        return left
    return float(brentq(event_value, left, right, xtol=1e-13, rtol=1e-13))


def _terminal_entry_time(
    solution: Any,
    usable_end: float,
    f_star: float,
    spiral: SpiralParameters,
    wormhole: WormholeParameters,
) -> float | None:
    times = diagnostic_times(usable_end)
    if times.size < 101:
        return None
    states = solution.sol(times)
    omega_ok = (
        np.abs(effective_angular_velocity(states[1], spiral)) / abs(spiral.omega)
        <= 0.05
    )
    acceleration_ok = (
        np.abs(radial_acceleration(states[0], states[1], wormhole, spiral)) / f_star
        <= 0.05
    )
    good = omega_ok & acceleration_ok
    rolling = np.convolve(good.astype(np.int64), np.ones(101, dtype=np.int64), "valid")
    hits = np.flatnonzero(rolling == 101)
    return None if not hits.size else float(times[int(hits[0])])


def _classify_once(
    initial_state: tuple[float, float],
    survey_cap: float,
    spatial_boundary: float,
    x_acc: float,
    f_star: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    solver: SolverSettings,
) -> dict[str, Any]:
    solution = integrate_trajectory(
        initial_state,
        (0.0, survey_cap),
        wormhole,
        spiral,
        dense_output=True,
        stop_at_null_boundary=True,
        **solver.kwargs(),
    )
    null_time = None
    if solution.t_events is not None and len(solution.t_events[0]):
        null_time = float(solution.t_events[0][0])
    available_end = null_time if null_time is not None else survey_cap

    def xi_event(time: float) -> float:
        state = solution.sol(time)
        return float(abs(xi_from_state(state[0], state[1], wormhole, spiral)) - 0.99)

    def spatial_event(time: float) -> float:
        return float(abs(solution.sol(time)[0]) - spatial_boundary)

    tau_099 = _first_dense_event(solution, available_end, xi_event)
    tau_spatial = _first_dense_event(solution, available_end, spatial_event)
    boundary_candidates = [
        value for value in (tau_099, tau_spatial) if value is not None
    ]
    terminal_search_end = min(boundary_candidates) if boundary_candidates else available_end
    terminal_entry = _terminal_entry_time(
        solution,
        terminal_search_end,
        f_star,
        spiral,
        wormhole,
    )

    classification = "unresolved"
    event_time: float | None = None
    event_name = "survey_cap"
    monotone_c = None
    monotone_abs_u = None
    endpoint_omega = None
    if terminal_entry is not None:
        classification = "escaping_force_free"
        event_time = terminal_entry
        event_name = "terminal_entry"
        endpoint_time = min(terminal_entry + 1.0, available_end)
    elif tau_099 is not None:
        endpoint_time = tau_099
        endpoint = solution.sol(endpoint_time)
        grid = diagnostic_times(endpoint_time)
        window = grid[grid >= max(0.0, endpoint_time - 1.0 - DIAGNOSTIC_STEP)]
        window_states = solution.sol(window)
        c_values = timelike_margin(
            window_states[0], window_states[1], wormhole, spiral
        )
        abs_u_values = np.abs(window_states[1])
        monotone_c = bool(
            c_values[-1] < c_values[0]
            and np.all(np.diff(c_values) <= 1e-12)
        )
        monotone_abs_u = bool(
            abs_u_values[-1] < abs_u_values[0]
            and np.all(np.diff(abs_u_values) <= 1e-12)
        )
        endpoint_omega = float(
            abs(effective_angular_velocity(endpoint[1], spiral)) / abs(spiral.omega)
        )
        if (
            abs(float(endpoint[0])) < x_acc
            and monotone_c
            and monotone_abs_u
            and endpoint_omega >= 0.5
        ):
            classification = "null_boundary_asymptotic"
            event_time = tau_099
            event_name = "abs_xi_0.99"
    else:
        endpoint_time = available_end

    endpoint = solution.sol(endpoint_time)
    grid = diagnostic_times(endpoint_time)
    grid_states = solution.sol(grid)
    c_grid = timelike_margin(grid_states[0], grid_states[1], wormhole, spiral)
    return {
        "class": classification,
        "survey_cap": survey_cap,
        "event": event_name,
        "event_time": event_time,
        "terminal_entry_time": terminal_entry,
        "tau_abs_xi_0.99": tau_099,
        "tau_spatial_boundary": tau_spatial,
        "null_event_time": null_time,
        "endpoint_time": float(endpoint_time),
        "endpoint": {
            "x": float(endpoint[0]),
            "u": float(endpoint[1]),
            "xi": float(xi_from_state(endpoint[0], endpoint[1], wormhole, spiral)),
            "C": float(timelike_margin(endpoint[0], endpoint[1], wormhole, spiral)),
            "abs_Omega_over_abs_W": float(
                abs(effective_angular_velocity(endpoint[1], spiral))
                / abs(spiral.omega)
            ),
        },
        "minimum_C": float(np.min(c_grid)),
        "boundary_window_monotone_C": monotone_c,
        "boundary_window_monotone_abs_u": monotone_abs_u,
        "boundary_endpoint_abs_Omega_over_abs_W": endpoint_omega,
        "still_physical_at_cap": bool(null_time is None and endpoint_time == survey_cap),
    }


def classify_reference(
    x0: float,
    xi0: float,
    spatial_boundary: float,
    x_acc: float,
    f_star: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Classify a physical reference and confirm the class under refinement."""

    _, u_array = state_from_xi(x0, xi0, wormhole, spiral)
    initial_state = (float(x0), float(u_array))
    by_solver: dict[str, dict[str, Any]] = {}
    for label, solver in (
        ("production", PRODUCTION_SOLVER),
        ("tight", TIGHT_SOLVER),
    ):
        result = _classify_once(
            initial_state,
            SURVEY_CAP,
            spatial_boundary,
            x_acc,
            f_star,
            wormhole,
            spiral,
            solver,
        )
        if result["class"] == "unresolved" and result["still_physical_at_cap"]:
            result = _classify_once(
                initial_state,
                2.0 * SURVEY_CAP,
                spatial_boundary,
                x_acc,
                f_star,
                wormhole,
                spiral,
                solver,
            )
        by_solver[label] = result
    consistent = by_solver["production"]["class"] == by_solver["tight"]["class"]
    final_class = by_solver["tight"]["class"] if consistent else "unresolved"
    tight = by_solver["tight"]
    return {
        "x0": float(x0),
        "xi0": float(xi0),
        "u0": initial_state[1],
        "C0": float(timelike_margin(*initial_state, wormhole, spiral)),
        "class": final_class,
        "tolerance_class_consistent": consistent,
        "production": by_solver["production"],
        "tight": tight,
    }


def trajectory_reconnaissance(
    force_calibration: dict[str, Any],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Build the revised complete reference trajectory-class table."""

    x_acc = float(force_calibration["X_acc"])
    f_star = float(force_calibration["F_star"])
    spatial_boundary = SURVEY_BOUNDARY_FACTOR * float(force_calibration["L"])
    candidates: list[tuple[str, float, float]] = [
        ("throat", 0.0, xi)
        for xi in (-0.95, -0.75, -0.25, 0.25, 0.75, 0.95)
    ]
    for x_fraction in (1.0, 0.75, 0.5):
        for xi in (-0.5, -0.25, 0.0, 0.25, 0.5):
            candidates.append(("incoming_grid", -x_fraction * x_acc, xi))

    table: list[dict[str, Any]] = []
    for identifier, (source, x0, xi0) in enumerate(candidates, start=1):
        result = classify_reference(
            x0,
            xi0,
            spatial_boundary,
            x_acc,
            f_star,
            wormhole,
            spiral,
        )
        result["id"] = f"recon-{identifier:02d}"
        result["source"] = source
        table.append(result)

    escaping = [entry for entry in table if entry["class"] == "escaping_force_free"]
    if not any(entry["source"] == "throat" for entry in escaping):
        raise GateBlocked("escaping reconnaissance subset has no throat launch")
    if not any(
        entry["source"] == "incoming_grid" and entry["tight"]["endpoint"]["x"] > 0.0
        for entry in escaping
    ):
        raise GateBlocked("escaping reconnaissance subset has no incoming crossing case")
    if any(not entry["tolerance_class_consistent"] for entry in table):
        inconsistent = [entry["id"] for entry in table if not entry["tolerance_class_consistent"]]
        raise GateBlocked(f"trajectory classifications inconsistent under refinement: {inconsistent}")

    terminal_positions = []
    terminal_times = []
    for entry in escaping:
        terminal_time = entry["tight"]["terminal_entry_time"]
        if terminal_time is None:
            raise GateBlocked(f"escaping reference lacks terminal time: {entry['id']}")
        _, u0 = state_from_xi(entry["x0"], entry["xi0"], wormhole, spiral)
        solution = integrate_trajectory(
            (entry["x0"], float(u0)),
            (0.0, float(terminal_time) + 1.0),
            wormhole,
            spiral,
            dense_output=True,
            stop_at_null_boundary=True,
            **TIGHT_SOLVER.kwargs(),
        )
        terminal_positions.append(abs(float(solution.sol(terminal_time)[0])))
        terminal_times.append(float(terminal_time))
    x_term = float(max(terminal_positions))
    x_domain = int(math.ceil(1.25 * max(x_acc, x_term)))
    x_central = min(x_acc, x_domain / 2.0)
    counts = {
        name: sum(entry["class"] == name for entry in table)
        for name in ("escaping_force_free", "null_boundary_asymptotic", "unresolved")
    }
    return {
        "survey_cap_initial": SURVEY_CAP,
        "survey_cap_max": 2.0 * SURVEY_CAP,
        "spatial_survey_boundary": spatial_boundary,
        "incoming_grid": {
            "x_fractions_of_X_acc": [1.0, 0.75, 0.5],
            "xi_values": [-0.5, -0.25, 0.0, 0.25, 0.5],
            "ordering": "x fraction outer-to-inner, then xi ascending",
        },
        "counts": counts,
        "table": table,
        "escaping_terminal_times": terminal_times,
        "X_term": x_term,
        "X": x_domain,
        "X_c": x_central,
    }


def allocate_strata(total: int) -> list[dict[str, Any]]:
    """Allocate 30/35/35 counts and globally balanced xi signs."""

    raw = np.asarray([0.30, 0.35, 0.35]) * total
    counts = np.floor(raw).astype(int)
    remainder_order = np.argsort(-(raw - counts), kind="stable")
    for index in remainder_order[: total - int(np.sum(counts))]:
        counts[index] += 1
    positive = counts // 2
    odd = np.flatnonzero(counts % 2)
    for offset, index in enumerate(odd):
        positive[index] += 1 if offset % 2 == 0 else 0
    target_positive = total // 2
    while int(np.sum(positive)) < target_positive:
        for index in range(len(positive)):
            if positive[index] < counts[index] and int(np.sum(positive)) < target_positive:
                positive[index] += 1
    while int(np.sum(positive)) > target_positive:
        for index in range(len(positive) - 1, -1, -1):
            if positive[index] > 0 and int(np.sum(positive)) > target_positive:
                positive[index] -= 1
    names = ("core", "shoulder", "edge")
    ranges = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99))
    return [
        {
            "name": name,
            "range": list(interval),
            "count": int(count),
            "positive": int(pos),
            "negative": int(count - pos),
        }
        for name, interval, count, pos in zip(names, ranges, counts, positive)
    ]


def sample_calibration_states(
    total: int,
    x_domain: float,
    x_central: float,
    seed: int,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw the disposable physical calibration design exactly once."""

    rng = np.random.default_rng(seed)
    broad_count = total // 2
    central_count = total - broad_count
    x = np.concatenate(
        (
            rng.uniform(-x_central, x_central, central_count),
            rng.uniform(-x_domain, x_domain, broad_count),
        )
    ).astype(np.float64)
    x_component = np.asarray(
        ["central"] * central_count + ["broad"] * broad_count,
        dtype="U8",
    )
    strata = allocate_strata(total)
    xi_parts = []
    labels = []
    for stratum in strata:
        low, high = stratum["range"]
        magnitudes = rng.uniform(low, high, stratum["count"])
        signs = np.concatenate(
            (
                np.ones(stratum["positive"]),
                -np.ones(stratum["negative"]),
            )
        )
        rng.shuffle(signs)
        xi_parts.append(magnitudes * signs)
        labels.extend([stratum["name"]] * stratum["count"])
    xi = np.concatenate(xi_parts).astype(np.float64)
    labels_array = np.asarray(labels, dtype="U8")
    permutation = rng.permutation(total)
    x = x[permutation]
    x_component = x_component[permutation]
    xi = xi[permutation]
    labels_array = labels_array[permutation]
    _, u = state_from_xi(x, xi, wormhole, spiral)
    states = np.column_stack((x, u)).astype(np.float64)
    design_hash = hashlib.sha256()
    design_hash.update(states.tobytes())
    design_hash.update(xi.tobytes())
    return states, {
        "seed": seed,
        "count": total,
        "x_mixture_counts": {"central": central_count, "broad": broad_count},
        "strata": strata,
        "positive_xi": int(np.sum(xi > 0.0)),
        "negative_xi": int(np.sum(xi < 0.0)),
        "state_sha256": design_hash.hexdigest(),
        "x_component_counts": {
            "central": int(np.sum(x_component == "central")),
            "broad": int(np.sum(x_component == "broad")),
        },
        "stratum_counts_after_shuffle": {
            name: int(np.sum(labels_array == name))
            for name in ("core", "shoulder", "edge")
        },
    }


def _flow_state(
    state: np.ndarray,
    duration: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    solver: SolverSettings,
    *,
    exterior: bool = False,
) -> np.ndarray:
    integrator = (
        integrate_vector_field_continuation if exterior else integrate_trajectory
    )
    kwargs: dict[str, Any] = solver.kwargs()
    if not exterior:
        kwargs["stop_at_null_boundary"] = True
    solution = integrator(
        state,
        (0.0, duration),
        wormhole,
        spiral,
        t_eval=(duration,),
        **kwargs,
    )
    if solution.y.shape[1] != 1:
        raise GateBlocked(f"reference flow did not reach duration {duration}")
    return np.asarray(solution.y[:, -1], dtype=np.float64)


def _normalized_rows(delta: np.ndarray, x: np.ndarray, d: np.ndarray, X: float) -> np.ndarray:
    return np.sqrt((delta[:, 0] / X) ** 2 + (delta[:, 1] / d) ** 2)


def calibrate_learning_step(
    states: np.ndarray,
    x_domain: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Compute all h diagnostics using physical reference flow only."""

    x = states[:, 0]
    _, d = state_geometry(x, wormhole, spiral)
    vector_field = np.column_stack(
        (
            states[:, 1],
            radial_acceleration(states[:, 0], states[:, 1], wormhole, spiral),
        )
    )
    candidates: list[dict[str, Any]] = []
    for h in H_CANDIDATES:
        production = np.empty_like(states)
        tight = np.empty_like(states)
        two_half = np.empty_like(states)
        for index, state in enumerate(states):
            production[index] = _flow_state(
                state, h, wormhole, spiral, PRODUCTION_SOLVER
            )
            tight[index] = _flow_state(state, h, wormhole, spiral, TIGHT_SOLVER)
            midpoint = _flow_state(
                state, 0.5 * h, wormhole, spiral, TIGHT_SOLVER
            )
            two_half[index] = _flow_state(
                midpoint, 0.5 * h, wormhole, spiral, TIGHT_SOLVER
            )
        increment = tight - states
        residual = increment - h * vector_field
        increment_norm = _normalized_rows(increment, x, d, x_domain)
        residual_norm = _normalized_rows(residual, x, d, x_domain)
        prod_tight_delta = production - tight
        tight_half_delta = tight - two_half
        prod_tight_norm = _normalized_rows(prod_tight_delta, x, d, x_domain)
        tight_half_norm = _normalized_rows(tight_half_delta, x, d, x_domain)
        rms_increment = float(np.sqrt(np.mean(increment_norm**2)))
        epsilon = float(max(np.max(prod_tight_norm), np.max(tight_half_norm)))
        q_value = float(np.sqrt(np.mean(residual_norm**2)) / rms_increment)
        rho_99 = float(np.quantile(increment_norm, 0.99, method="linear"))
        convergence_absolute = epsilon <= 1e-9
        convergence_relative = epsilon <= 1e-6 * rms_increment
        candidate = {
            "h": h,
            "q_h": q_value,
            "rho_99_h": rho_99,
            "rms_increment_norm": rms_increment,
            "epsilon_ref": epsilon,
            "componentwise_max_abs": {
                "production_vs_tight": {
                    "x": float(np.max(np.abs(prod_tight_delta[:, 0]))),
                    "u": float(np.max(np.abs(prod_tight_delta[:, 1]))),
                },
                "tight_vs_two_half": {
                    "x": float(np.max(np.abs(tight_half_delta[:, 0]))),
                    "u": float(np.max(np.abs(tight_half_delta[:, 1]))),
                },
            },
            "normalized_max": {
                "production_vs_tight": float(np.max(prod_tight_norm)),
                "tight_vs_two_half": float(np.max(tight_half_norm)),
            },
            "worst_indices": {
                "production_vs_tight": int(np.argmax(prod_tight_norm)),
                "tight_vs_two_half": int(np.argmax(tight_half_norm)),
            },
            "passes_nontriviality": q_value >= 0.02,
            "passes_locality": rho_99 <= 0.10,
            "passes_convergence_absolute": convergence_absolute,
            "passes_convergence_relative": convergence_relative,
        }
        candidate["passes_all"] = bool(
            candidate["passes_nontriviality"]
            and candidate["passes_locality"]
            and convergence_absolute
            and convergence_relative
        )
        candidates.append(candidate)
    selected = next((row["h"] for row in candidates if row["passes_all"]), None)
    return {
        "production_solver": PRODUCTION_SOLVER.metadata(),
        "tight_solver": TIGHT_SOLVER.metadata(),
        "candidates": candidates,
        "selected_h": selected,
        "passed": selected is not None,
    }


def verify_exterior_contract(
    x_domain: float,
    selected_h: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Regression-check F and finite-step continuation through |xi|=1.25."""

    x = np.linspace(
        -x_domain,
        x_domain,
        _grid_count(x_domain, 0.01),
        dtype=np.float64,
    )
    xi = np.linspace(-1.25, 1.25, EXTERIOR_XI_POINTS, dtype=np.float64)
    center, half_width = state_geometry(x, wormhole, spiral)
    finite = True
    maximum_abs_f = 0.0
    worst_grid_state = None
    for start in range(0, x.size, 512):
        stop = min(start + 512, x.size)
        u = center[start:stop, None] + half_width[start:stop, None] * xi[None, :]
        acceleration = radial_acceleration(x[start:stop, None], u, wormhole, spiral)
        if not np.all(np.isfinite(acceleration)):
            finite = False
        flat_index = int(np.argmax(np.abs(acceleration)))
        value = float(np.abs(acceleration).flat[flat_index])
        if value > maximum_abs_f:
            local_x, local_xi = np.unravel_index(flat_index, acceleration.shape)
            maximum_abs_f = value
            worst_grid_state = {
                "x": float(x[start + local_x]),
                "xi": float(xi[local_xi]),
                "F": float(acceleration[local_x, local_xi]),
            }
    if not finite:
        raise GateBlocked("exterior F grid contains nonfinite values")

    design_x = np.linspace(-x_domain, x_domain, 17)
    design_xi = np.asarray([-1.25, -1.13, -1.01, 1.01, 1.13, 1.25])
    xx, xixi = np.meshgrid(design_x, design_xi, indexing="ij")
    _, uu = state_from_xi(xx.ravel(), xixi.ravel(), wormhole, spiral)
    states = np.column_stack((xx.ravel(), uu)).astype(np.float64)
    _, d = state_geometry(states[:, 0], wormhole, spiral)
    production = np.empty_like(states)
    tight = np.empty_like(states)
    two_half = np.empty_like(states)
    for index, state in enumerate(states):
        production[index] = _flow_state(
            state,
            selected_h,
            wormhole,
            spiral,
            PRODUCTION_SOLVER,
            exterior=True,
        )
        tight[index] = _flow_state(
            state,
            selected_h,
            wormhole,
            spiral,
            TIGHT_SOLVER,
            exterior=True,
        )
        midpoint = _flow_state(
            state,
            0.5 * selected_h,
            wormhole,
            spiral,
            TIGHT_SOLVER,
            exterior=True,
        )
        two_half[index] = _flow_state(
            midpoint,
            0.5 * selected_h,
            wormhole,
            spiral,
            TIGHT_SOLVER,
            exterior=True,
        )
    increment = tight - states
    rms_increment = float(
        np.sqrt(np.mean(_normalized_rows(increment, states[:, 0], d, x_domain) ** 2))
    )
    prod_delta = production - tight
    half_delta = tight - two_half
    prod_norm = _normalized_rows(prod_delta, states[:, 0], d, x_domain)
    half_norm = _normalized_rows(half_delta, states[:, 0], d, x_domain)
    epsilon = float(max(np.max(prod_norm), np.max(half_norm)))
    prod_worst = int(np.argmax(prod_norm))
    half_worst = int(np.argmax(half_norm))

    continuation_source = inspect.getsource(
        integrate_module.integrate_vector_field_continuation
    )
    radial_source = inspect.getsource(radial_system) + inspect.getsource(
        radial_acceleration
    )
    forbidden_executable_dependencies = any(
        name in radial_source
        for name in ("conserved_energy(", "lorentz_factor(", "timelike_margin(")
    )
    result = {
        "grid": {
            "x_range": [-x_domain, x_domain],
            "x_spacing": 0.01,
            "x_points": int(x.size),
            "xi_range": [-1.25, 1.25],
            "xi_points": EXTERIOR_XI_POINTS,
            "all_F_finite": finite,
            "max_abs_F": maximum_abs_f,
            "worst_F_state": worst_grid_state,
        },
        "call_path": [
            "integrate_vector_field_continuation",
            "_solve_radial_initial_value_problem",
            "radial_system",
            "radial_acceleration",
        ],
        "continuation_doc_declares_nonphysical": "not physical" in continuation_source,
        "requires_sqrt_C_energy_or_lorentz": forbidden_executable_dependencies,
        "design_state_count": int(states.shape[0]),
        "design_x_points": design_x.tolist(),
        "design_xi_points": design_xi.tolist(),
        "rms_increment_norm": rms_increment,
        "epsilon_ref": epsilon,
        "componentwise_max_abs": {
            "production_vs_tight": {
                "x": float(np.max(np.abs(prod_delta[:, 0]))),
                "u": float(np.max(np.abs(prod_delta[:, 1]))),
            },
            "tight_vs_two_half": {
                "x": float(np.max(np.abs(half_delta[:, 0]))),
                "u": float(np.max(np.abs(half_delta[:, 1]))),
            },
        },
        "normalized_max": {
            "production_vs_tight": float(np.max(prod_norm)),
            "tight_vs_two_half": float(np.max(half_norm)),
        },
        "worst_states": {
            "production_vs_tight": {
                "x": float(states[prod_worst, 0]),
                "xi": float(xixi.ravel()[prod_worst]),
            },
            "tight_vs_two_half": {
                "x": float(states[half_worst, 0]),
                "xi": float(xixi.ravel()[half_worst]),
            },
        },
        "passes_absolute": epsilon <= 1e-9,
        "passes_relative": epsilon <= 1e-6 * rms_increment,
    }
    result["passed"] = bool(
        finite
        and not forbidden_executable_dependencies
        and result["passes_absolute"]
        and result["passes_relative"]
    )
    if not result["passed"]:
        raise GateBlocked("exterior continuation certification failed")
    result["status"] = "passed"
    return result


def verify_integrator_contract(
    selected_h: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Regression-check physical equivalence and exterior rejection semantics."""

    physical_cases = ((-2.0, -0.75), (0.0, 0.0), (2.0, 0.75))
    componentwise = np.zeros(2, dtype=np.float64)
    for x0, xi0 in physical_cases:
        _, u0 = state_from_xi(x0, xi0, wormhole, spiral)
        state = np.asarray([x0, float(u0)], dtype=np.float64)
        physical = _flow_state(
            state, selected_h, wormhole, spiral, PRODUCTION_SOLVER
        )
        continuation = _flow_state(
            state,
            selected_h,
            wormhole,
            spiral,
            PRODUCTION_SOLVER,
            exterior=True,
        )
        componentwise = np.maximum(componentwise, np.abs(physical - continuation))

    probe_x, probe_xi = 0.0, 1.25
    _, probe_u = state_from_xi(probe_x, probe_xi, wormhole, spiral)
    probe = np.asarray([probe_x, float(probe_u)], dtype=np.float64)
    continuation_probe = _flow_state(
        probe,
        selected_h,
        wormhole,
        spiral,
        PRODUCTION_SOLVER,
        exterior=True,
    )
    rejection_message = None
    try:
        _flow_state(probe, selected_h, wormhole, spiral, PRODUCTION_SOLVER)
    except ValueError as error:
        rejection_message = str(error)
    if rejection_message is None:
        raise GateBlocked("physical integrator no longer rejects an exterior state")
    result = {
        "physical_case_count": len(physical_cases),
        "physical_vs_continuation_max_abs": {
            "x": float(componentwise[0]),
            "u": float(componentwise[1]),
        },
        "exterior_probe": {
            "x0": probe_x,
            "xi0": probe_xi,
            "u0": float(probe_u),
            "C0": float(timelike_margin(*probe, wormhole, spiral)),
            "continuation_final": continuation_probe.tolist(),
            "physical_rejection": rejection_message,
        },
        "passed": bool(
            np.max(componentwise) <= 2e-14
            and "not timelike" in rejection_message
            and np.all(np.isfinite(continuation_probe))
        ),
    }
    if not result["passed"]:
        raise GateBlocked("physical/continuation integrator contract regression failed")
    result["status"] = "passed"
    return result


def escaping_rollout_cap(
    reconnaissance: dict[str, Any], selected_h: float
) -> float:
    terminal_times = [
        float(entry["tight"]["terminal_entry_time"])
        for entry in reconnaissance["table"]
        if entry["class"] == "escaping_force_free"
    ]
    if not terminal_times:
        raise GateBlocked("escaping reconnaissance subset is empty")
    return float(selected_h * math.ceil((max(terminal_times) + 1.0) / selected_h))


def _reference_data_event(
    solution: Any,
    available_end: float,
    x_domain: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[float | None, str | None]:
    def spatial_value(time: float) -> float:
        return float(abs(solution.sol(time)[0]) - 0.9 * x_domain)

    def xi_value(time: float) -> float:
        state = solution.sol(time)
        return float(abs(xi_from_state(state[0], state[1], wormhole, spiral)) - 0.99)

    spatial_time = _first_dense_event(solution, available_end, spatial_value)
    xi_time = _first_dense_event(solution, available_end, xi_value)
    events = [
        (time, label)
        for time, label in ((spatial_time, "abs_x_0.9X"), (xi_time, "abs_xi_0.99"))
        if time is not None
    ]
    return (None, None) if not events else min(events, key=lambda item: item[0])


def reference_rollout_metadata(
    x0: float,
    xi0: float,
    reference_class: str,
    selected_h: float,
    s_esc: float,
    x_domain: float,
    f_star: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Freeze one reference-only horizon and class-appropriate diagnostics."""

    _, u0_array = state_from_xi(x0, xi0, wormhole, spiral)
    u0 = float(u0_array)
    solution = integrate_trajectory(
        (x0, u0),
        (0.0, s_esc),
        wormhole,
        spiral,
        dense_output=True,
        stop_at_null_boundary=True,
        **TIGHT_SOLVER.kwargs(),
    )
    null_time = None
    if solution.t_events is not None and len(solution.t_events[0]):
        null_time = float(solution.t_events[0][0])
    available_end = null_time if null_time is not None else s_esc
    tau_data, event_type = _reference_data_event(
        solution, available_end, x_domain, wormhole, spiral
    )
    if tau_data is None and null_time is not None:
        raise GateBlocked(
            f"reference ({x0}, {xi0}) reaches C=0 before a physical-data event"
        )
    horizon_source = s_esc if tau_data is None else min(s_esc, tau_data)
    steps = int(math.floor((horizon_source + 1e-12) / selected_h))
    horizon = float(steps * selected_h)
    times = diagnostic_times(horizon)
    states = solution.sol(times)
    states_finite = bool(solution.success and np.all(np.isfinite(states)))
    if not states_finite:
        raise GateBlocked(
            f"reference ({x0}, {xi0}) has a numerical failure by T_i"
        )
    c_values = timelike_margin(states[0], states[1], wormhole, spiral)
    minimum_c = float(np.min(c_values))
    if not minimum_c > 0.0:
        raise GateBlocked(f"reference ({x0}, {xi0}) violates C>0 by T_i")
    energy = conserved_energy(states[0], states[1], wormhole, spiral)
    energy0 = float(energy[0])
    relative_energy_error = np.abs(energy - energy0) / (abs(energy0) + 1e-12)
    terminal_entry_by_horizon = None
    escaping_status: str | None = None
    boundary_status: str | None = None
    if reference_class == "escaping_force_free":
        terminal_entry_by_horizon = _terminal_entry_time(
            solution, horizon, f_star, spiral, wormhole
        )
        escaping_status = (
            "terminal_reached_by_T_i"
            if terminal_entry_by_horizon is not None
            else "right_censored_at_T_i"
        )
    elif reference_class == "null_boundary_asymptotic":
        boundary_status = "finite_inside_common_domain_through_T_i"
    else:
        raise GateBlocked(f"rollout candidate has unsupported class {reference_class}")

    crossing_time = None
    if x0 < 0.0:
        crossing_time = _first_dense_event(
            solution,
            min(horizon, available_end),
            lambda time: float(solution.sol(time)[0]),
        )
    final_state = solution.sol(horizon)
    return {
        "x0": float(x0),
        "xi0": float(xi0),
        "u0": u0,
        "C0": float(timelike_margin(x0, u0, wormhole, spiral)),
        "reference_class": reference_class,
        "T_i": horizon,
        "step_count": steps,
        "tau_data_ref": tau_data,
        "tau_data_event": event_type,
        "null_event_after_cutoff": null_time,
        "minimum_reference_C": minimum_c,
        "reference_energy_initial": energy0,
        "reference_energy_relative_error_max": float(
            np.max(relative_energy_error)
        ),
        "reference_energy_relative_error_final": float(relative_energy_error[-1]),
        "terminal_entry_time_by_T_i": terminal_entry_by_horizon,
        "terminal_entry_search_interval": [0.0, horizon],
        "exclusive_escaping_rollout_status": escaping_status,
        "exclusive_boundary_rollout_status": boundary_status,
        "reference_integration_success_through_T_i": bool(solution.success),
        "reference_states_finite_through_T_i": states_finite,
        "reference_physical_through_T_i": True,
        "throat_crossing_time": crossing_time,
        "final_reference_state": {
            "x": float(final_state[0]),
            "u": float(final_state[1]),
            "xi": float(
                xi_from_state(final_state[0], final_state[1], wormhole, spiral)
            ),
            "C": float(
                timelike_margin(final_state[0], final_state[1], wormhole, spiral)
            ),
            "boundary_distance": float(
                1.0
                - abs(
                    xi_from_state(
                        final_state[0], final_state[1], wormhole, spiral
                    )
                )
            ),
        },
    }


def rollout_count_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Count and enforce mutually exclusive frozen-horizon reference outcomes."""

    categories = (
        "escaping_force_free",
        "null_boundary_asymptotic",
        "near_edge",
    )
    classes = ("escaping_force_free", "null_boundary_asymptotic")
    category_counts = {
        category: sum(member["category"] == category for member in members)
        for category in categories
    }
    reference_class_counts = {
        reference_class: sum(
            member["reference_class"] == reference_class for member in members
        )
        for reference_class in classes
    }
    escaping_counts = {status: 0 for status in ESCAPING_ROLLOUT_STATUSES}
    boundary_counts = {status: 0 for status in BOUNDARY_ROLLOUT_STATUSES}
    for member in members:
        reference_class = member["reference_class"]
        escaping_status = member["exclusive_escaping_rollout_status"]
        boundary_status = member["exclusive_boundary_rollout_status"]
        if reference_class == "escaping_force_free":
            if escaping_status not in escaping_counts or boundary_status is not None:
                raise GateBlocked(
                    f"{member['id']} has invalid escaping rollout status fields"
                )
            escaping_counts[escaping_status] += 1
        elif reference_class == "null_boundary_asymptotic":
            if boundary_status not in boundary_counts or escaping_status is not None:
                raise GateBlocked(
                    f"{member['id']} has invalid boundary rollout status fields"
                )
            boundary_counts[boundary_status] += 1
        else:
            raise GateBlocked(
                f"{member['id']} has unsupported reference class {reference_class}"
            )
    if sum(escaping_counts.values()) != reference_class_counts["escaping_force_free"]:
        raise GateBlocked("exclusive escaping rollout-status identity failed")
    if sum(boundary_counts.values()) != reference_class_counts[
        "null_boundary_asymptotic"
    ]:
        raise GateBlocked("exclusive boundary rollout-status identity failed")
    return {
        "category_counts": category_counts,
        "reference_class_counts": reference_class_counts,
        "exclusive_escaping_rollout_status_counts": escaping_counts,
        "exclusive_boundary_rollout_status_counts": boundary_counts,
    }


def frozen_member_identity_sha256(members: list[dict[str, Any]]) -> str:
    """Hash the frozen identifiers, states, classes, categories, and horizons."""

    identity = [
        {
            key: member[key]
            for key in ("id", "category", "reference_class", "x0", "xi0", "T_i")
        }
        for member in members
    ]
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_pools(
    seed: int, x_central: float, size: int = 256
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    interior = np.column_stack(
        (
            rng.uniform(-0.5 * x_central, 0.5 * x_central, size),
            rng.uniform(-0.85, 0.85, size),
        )
    ).astype(np.float64)
    edge_signs = np.tile(np.asarray([-1.0, 1.0]), size // 2)
    if edge_signs.size < size:
        edge_signs = np.append(edge_signs, -1.0)
    edge = np.column_stack(
        (
            rng.uniform(-0.5 * x_central, 0.5 * x_central, size),
            edge_signs * rng.uniform(0.90, 0.97, size),
        )
    ).astype(np.float64)
    interior_hash = hashlib.sha256(interior.tobytes()).hexdigest()
    edge_hash = hashlib.sha256(edge.tobytes()).hexdigest()
    return interior, edge, {
        "seed": seed,
        "size_per_pool": size,
        "interior_rule": "x~U[-X_c/2,X_c/2], xi~U[-0.85,0.85]",
        "near_edge_rule": (
            "x~U[-X_c/2,X_c/2], alternating xi signs, "
            "abs(xi)~U[0.90,0.97]"
        ),
        "interior_sha256": interior_hash,
        "near_edge_sha256": edge_hash,
    }


def _horizon_requirement(category: str, s_esc: float) -> float:
    if category == "near_edge":
        return min(2.0, 0.25 * s_esc)
    return min(5.0, 0.5 * s_esc)


def _build_suite_member(
    split: str,
    identifier: str,
    category: str,
    role: str,
    origin: str,
    x0: float,
    xi0: float,
    expected_class: str,
    selected_h: float,
    s_esc: float,
    x_domain: float,
    x_acc: float,
    f_star: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    classification = classify_reference(
        x0,
        xi0,
        40.0,
        x_acc,
        f_star,
        wormhole,
        spiral,
    )
    if classification["class"] != expected_class:
        raise GateBlocked(
            f"{identifier} expected {expected_class}, got {classification['class']}"
        )
    metadata = reference_rollout_metadata(
        x0,
        xi0,
        expected_class,
        selected_h,
        s_esc,
        x_domain,
        f_star,
        wormhole,
        spiral,
    )
    minimum_horizon = _horizon_requirement(category, s_esc)
    if metadata["T_i"] + 1e-12 < minimum_horizon:
        raise GateBlocked(
            f"{identifier} horizon {metadata['T_i']} < {minimum_horizon}"
        )
    metadata.update(
        {
            "id": identifier,
            "split": split,
            "category": category,
            "presentation_role": role,
            "origin": origin,
            "minimum_horizon_required": minimum_horizon,
            "classification_tolerance_consistent": classification[
                "tolerance_class_consistent"
            ],
            "classification_production_event_time": classification["production"][
                "event_time"
            ],
            "classification_tight_event_time": classification["tight"][
                "event_time"
            ],
            "classification_spatial_boundary_abs_x": 40.0,
            "classification_production_time_domain": [
                0.0,
                classification["production"]["survey_cap"],
            ],
            "classification_tight_time_domain": [
                0.0,
                classification["tight"]["survey_cap"],
            ],
            "classification_production_terminal_entry_time": classification[
                "production"
            ]["terminal_entry_time"],
            "classification_tight_terminal_entry_time": classification["tight"][
                "terminal_entry_time"
            ],
        }
    )
    return metadata


def build_rollout_suite(
    split: str,
    seed: int,
    anchors: dict[str, tuple[float, float]],
    crossing_x: float,
    selected_h: float,
    s_esc: float,
    x_domain: float,
    x_central: float,
    x_acc: float,
    f_star: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    """Freeze one 24-member suite from fixed anchors and prespecified pools."""

    members: list[dict[str, Any]] = []

    def add_anchor(
        key: str,
        category: str,
        role: str,
        expected_class: str,
    ) -> None:
        x0, xi0 = anchors[key]
        members.append(
            _build_suite_member(
                split,
                f"{split}-{key}",
                category,
                role,
                "fixed_anchor",
                x0,
                xi0,
                expected_class,
                selected_h,
                s_esc,
                x_domain,
                x_acc,
                f_star,
                wormhole,
                spiral,
            )
        )

    add_anchor(
        "boundary_throat",
        "null_boundary_asymptotic",
        "boundary_asymptotic_throat",
        "null_boundary_asymptotic",
    )
    add_anchor(
        "below_terminal",
        "escaping_force_free",
        "below_terminal_escaping",
        "escaping_force_free",
    )
    add_anchor(
        "above_terminal",
        "escaping_force_free",
        "above_terminal_escaping",
        "escaping_force_free",
    )
    add_anchor(
        "negative_edge",
        "near_edge",
        "negative_edge",
        "null_boundary_asymptotic",
    )
    add_anchor(
        "positive_edge",
        "near_edge",
        "positive_edge",
        "escaping_force_free",
    )

    crossing_member = None
    crossing_pool_evidence = []
    for xi0 in (-0.5, -0.25, 0.0, 0.25, 0.5):
        try:
            candidate = _build_suite_member(
                split,
                f"{split}-crossing-anchor",
                "escaping_force_free",
                "incoming_throat_crossing",
                "ordered_crossing_pool",
                crossing_x,
                xi0,
                "escaping_force_free",
                selected_h,
                s_esc,
                x_domain,
                x_acc,
                f_star,
                wormhole,
                spiral,
            )
            qualifies = bool(
                candidate["throat_crossing_time"] is not None
                and candidate["throat_crossing_time"] <= candidate["T_i"] + 1e-12
            )
            crossing_pool_evidence.append(
                {"xi0": xi0, "class": candidate["reference_class"], "qualifies": qualifies}
            )
            if qualifies:
                crossing_member = candidate
                break
        except GateBlocked as error:
            crossing_pool_evidence.append(
                {"xi0": xi0, "class": "nonqualifying", "qualifies": False, "reason": str(error)}
            )
    if crossing_member is None:
        raise GateBlocked(f"{split} ordered crossing pool has no qualifying member")
    members.append(crossing_member)

    interior_pool, edge_pool, pool_metadata = _candidate_pools(seed, x_central)
    needed = {"escaping_force_free": 5, "null_boundary_asymptotic": 7}
    selected_interior = {"escaping_force_free": [], "null_boundary_asymptotic": []}
    for pool_index, (x0, xi0) in enumerate(interior_pool):
        if all(len(selected_interior[key]) >= value for key, value in needed.items()):
            break
        classification = classify_reference(
            float(x0), float(xi0), 40.0, x_acc, f_star, wormhole, spiral
        )
        reference_class = classification["class"]
        if reference_class not in needed or len(selected_interior[reference_class]) >= needed[reference_class]:
            continue
        category = reference_class
        try:
            candidate = _build_suite_member(
                split,
                f"{split}-{category}-generated-{len(selected_interior[reference_class]) + 1:02d}",
                category,
                "generated",
                f"interior_pool_index_{pool_index}",
                float(x0),
                float(xi0),
                reference_class,
                selected_h,
                s_esc,
                x_domain,
                x_acc,
                f_star,
                wormhole,
                spiral,
            )
        except GateBlocked:
            continue
        selected_interior[reference_class].append(candidate)
    for reference_class, count in needed.items():
        if len(selected_interior[reference_class]) != count:
            raise GateBlocked(
                f"{split} interior pool supplied {len(selected_interior[reference_class])}/{count} "
                f"{reference_class} members"
            )
        members.extend(selected_interior[reference_class])

    edge_candidates = {"escaping_force_free": [], "null_boundary_asymptotic": []}
    for pool_index, (x0, xi0) in enumerate(edge_pool):
        if all(len(values) >= 3 for values in edge_candidates.values()):
            break
        classification = classify_reference(
            float(x0), float(xi0), 40.0, x_acc, f_star, wormhole, spiral
        )
        reference_class = classification["class"]
        if reference_class not in edge_candidates or len(edge_candidates[reference_class]) >= 3:
            continue
        try:
            candidate = _build_suite_member(
                split,
                f"{split}-near-edge-{reference_class}-generated-{len(edge_candidates[reference_class]) + 1:02d}",
                "near_edge",
                "generated",
                f"near_edge_pool_index_{pool_index}",
                float(x0),
                float(xi0),
                reference_class,
                selected_h,
                s_esc,
                x_domain,
                x_acc,
                f_star,
                wormhole,
                spiral,
            )
        except GateBlocked:
            continue
        edge_candidates[reference_class].append(candidate)
    if not all(len(values) >= 3 for values in edge_candidates.values()):
        available = {key: len(value) + 1 for key, value in edge_candidates.items()}
        raise GateBlocked(
            f"{split} near-edge pool cannot supply four-per-class target: {available}"
        )
    members.extend(edge_candidates["escaping_force_free"][:3])
    members.extend(edge_candidates["null_boundary_asymptotic"][:3])

    counts = rollout_count_summary(members)
    category_counts = counts["category_counts"]
    edge_class_counts = {
        reference_class: sum(
            member["category"] == "near_edge"
            and member["reference_class"] == reference_class
            for member in members
        )
        for reference_class in (
            "escaping_force_free",
            "null_boundary_asymptotic",
        )
    }
    if len(members) != 24 or any(count != 8 for count in category_counts.values()):
        raise GateBlocked(f"{split} suite category counts invalid: {category_counts}")
    if any(count != 4 for count in edge_class_counts.values()):
        raise GateBlocked(f"{split} near-edge class balance invalid: {edge_class_counts}")
    expected_reference_classes = {
        "escaping_force_free": 12,
        "null_boundary_asymptotic": 12,
    }
    if counts["reference_class_counts"] != expected_reference_classes:
        raise GateBlocked(
            f"{split} reference-class counts invalid: "
            f"{counts['reference_class_counts']}"
        )
    escaping_failures = counts["exclusive_escaping_rollout_status_counts"]
    if (
        escaping_failures["physical_failure_before_terminal_entry"] != 0
        or escaping_failures[
            "nonfinite_numerical_failure_before_terminal_entry"
        ]
        != 0
    ):
        raise GateBlocked(
            f"{split} escaping references contain a physical/numerical failure: "
            f"{escaping_failures}"
        )
    boundary_failures = counts["exclusive_boundary_rollout_status_counts"]
    if any(
        boundary_failures[status] != 0
        for status in BOUNDARY_ROLLOUT_STATUSES[1:]
    ):
        raise GateBlocked(
            f"{split} boundary references contain an invalid reference outcome: "
            f"{boundary_failures}"
        )
    return {
        "split": split,
        "seed": seed,
        "candidate_pool": pool_metadata,
        "ordered_crossing_pool": {
            "x0": crossing_x,
            "xi_values": [-0.5, -0.25, 0.0, 0.25, 0.5],
            "evidence": crossing_pool_evidence,
        },
        "count": len(members),
        "category_counts": category_counts,
        "reference_class_counts": counts["reference_class_counts"],
        "exclusive_escaping_rollout_status_counts": counts[
            "exclusive_escaping_rollout_status_counts"
        ],
        "exclusive_boundary_rollout_status_counts": counts[
            "exclusive_boundary_rollout_status_counts"
        ],
        "near_edge_reference_class_counts": edge_class_counts,
        "frozen_member_identity_sha256": frozen_member_identity_sha256(members),
        "members": members,
    }


def construct_rollout_suites(
    force: dict[str, Any],
    reconnaissance: dict[str, Any],
    selected_h: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, Any]:
    s_esc = escaping_rollout_cap(reconnaissance, selected_h)
    validation_anchors = {
        "boundary_throat": (0.0, -0.70),
        "below_terminal": (0.0, 0.20),
        "above_terminal": (0.0, 0.70),
        "negative_edge": (0.0, -0.94),
        "positive_edge": (0.0, 0.94),
    }
    test_anchors = {
        "boundary_throat": (0.0, -0.75),
        "below_terminal": (0.0, 0.25),
        "above_terminal": (0.0, 0.75),
        "negative_edge": (0.0, -0.95),
        "positive_edge": (0.0, 0.95),
    }
    validation = build_rollout_suite(
        "validation",
        SEEDS["validation_suite"],
        validation_anchors,
        -float(reconnaissance["X_c"]),
        selected_h,
        s_esc,
        float(reconnaissance["X"]),
        float(reconnaissance["X_c"]),
        float(force["X_acc"]),
        float(force["F_star"]),
        wormhole,
        spiral,
    )
    sealed_test = build_rollout_suite(
        "sealed_test",
        SEEDS["sealed_test_suite"],
        test_anchors,
        -1.25 * float(reconnaissance["X_c"]),
        selected_h,
        s_esc,
        float(reconnaissance["X"]),
        float(reconnaissance["X_c"]),
        float(force["X_acc"]),
        float(force["F_star"]),
        wormhole,
        spiral,
    )
    all_members = validation["members"] + sealed_test["members"]
    combined_counts = rollout_count_summary(all_members)
    if combined_counts["reference_class_counts"] != {
        "escaping_force_free": 24,
        "null_boundary_asymptotic": 24,
    }:
        raise GateBlocked(
            "combined rollout reference-class identity failed: "
            f"{combined_counts['reference_class_counts']}"
        )
    invariant_floor = max(
        member["reference_energy_relative_error_max"] for member in all_members
    )
    minimum_c = min(member["minimum_reference_C"] for member in all_members)
    return {
        "status": "frozen",
        "S_esc": s_esc,
        "selected_h": selected_h,
        "validation": validation,
        "sealed_test": sealed_test,
        "combined_counts": combined_counts,
        "global_minimum_reference_C": minimum_c,
        "reference_invariant_error_floor": invariant_floor,
    }


def run_physics_gate(project_root: Path) -> dict[str, Any]:
    """Run the revised gate in order and stop before dependent stages."""

    wormhole, spiral = experiment_parameters()
    result: dict[str, Any] = {
        "gate": "Simple GEB Wormhole ML physics-only gate",
        "specification_revision": "physics-gate-h-grid-v2",
        "specification_amendment": (
            "docs/specification_revisions/physics_gate_h_grid_v2.md"
        ),
        "status": "RUNNING",
        "package_version": "0.1.0",
        "package_mapping": package_mapping(),
        "physics_source_manifest": source_manifest(project_root),
        "solvers": {
            "production": PRODUCTION_SOLVER.metadata(),
            "validation": VALIDATION_SOLVER.metadata(),
            "tight": TIGHT_SOLVER.metadata(),
        },
        "seeds": SEEDS,
        "thresholds": {
            "q_min": 0.02,
            "rho_99_max": 0.10,
            "epsilon_absolute_max": 1e-9,
            "epsilon_relative_factor": 1e-6,
            "terminal_abs_Omega_ratio_max": 0.05,
            "terminal_abs_F_ratio_max": 0.05,
            "data_abs_xi_max": 0.99,
            "exterior_abs_xi_max": 1.25,
        },
    }
    force = calibrate_force_envelope(wormhole, spiral)
    result["compact_domain_force_envelope"] = force
    reconnaissance = trajectory_reconnaissance(force, wormhole, spiral)
    result["trajectory_reconnaissance"] = reconnaissance
    states, design = sample_calibration_states(
        CALIBRATION_STATE_COUNT,
        reconnaissance["X"],
        reconnaissance["X_c"],
        SEEDS["learning_step"],
        wormhole,
        spiral,
    )
    result["learning_step_design"] = design
    step = calibrate_learning_step(
        states,
        reconnaissance["X"],
        wormhole,
        spiral,
    )
    result["learning_step_calibration"] = step
    if not step["passed"]:
        result["status"] = "BLOCKED"
        result["blocker"] = {
            "stage": "learning_step_calibration",
            "condition": "no candidate h satisfies q_h >= 0.02",
            "smallest_follow_up": (
                "Revise the h-candidate grid explicitly in a new specification; "
                "do not generate final data under the current grid."
            ),
        }
        result["exterior_certification"] = {
            "status": "not_run",
            "reason": "selected h is undefined",
        }
        result["rollout_suites"] = {
            "status": "not_constructed",
            "reason": "selected h and S_esc are undefined",
            "validation": [],
            "sealed_test": [],
        }
        result["S_esc"] = None
        result["reference_invariant_error_floor"] = None
        return result

    selected_h = float(step["selected_h"])
    result["exterior_certification"] = verify_exterior_contract(
        reconnaissance["X"],
        selected_h,
        wormhole,
        spiral,
    )
    result["integrator_contract_regression"] = verify_integrator_contract(
        selected_h, wormhole, spiral
    )
    suites = construct_rollout_suites(
        force,
        reconnaissance,
        selected_h,
        wormhole,
        spiral,
    )
    result["rollout_suites"] = suites
    result["S_esc"] = suites["S_esc"]
    result["reference_invariant_error_floor"] = suites[
        "reference_invariant_error_floor"
    ]
    result["status"] = "PASSED"
    result["blocker"] = None
    return result
