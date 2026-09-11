"""Numerical integration for the published radial equation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from scipy.integrate import solve_ivp

from .dynamics import radial_system, timelike_margin
from .parameters import SpiralParameters, WormholeParameters


def _validated_initial_value_problem(
    initial_state: Sequence[float],
    t_span: tuple[float, float],
) -> np.ndarray:
    """Return a validated float64 state shared by both integration APIs."""

    state = np.asarray(initial_state, dtype=np.float64)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be two finite values: (l0, v0)")
    if not t_span[0] < t_span[1]:
        raise ValueError("t_span must be increasing")
    return state


def _solve_radial_initial_value_problem(
    state: np.ndarray,
    t_span: tuple[float, float],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    t_eval: Sequence[float] | None,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    dense_output: bool,
    events: Any,
) -> Any:
    """Run the shared SciPy solve without adding physical interpretation."""

    # args are appended to every call of radial_system and any event function.
    result = solve_ivp(
        radial_system,
        t_span,
        state,
        args=(wormhole, spiral),
        method=method,
        t_eval=None if t_eval is None else np.asarray(t_eval, dtype=np.float64),
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events,
    )
    if not result.success:
        raise RuntimeError(f"trajectory integration failed: {result.message}")
    return result


def integrate_trajectory(
    initial_state: Sequence[float],
    t_span: tuple[float, float],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    t_eval: Sequence[float] | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    dense_output: bool = False,
    stop_at_null_boundary: bool = True,
) -> Any:
    r"""Integrate ``(l, dl/dt)`` with SciPy's adaptive ``solve_ivp``.

    DOP853 is the default because the published parameter regime gives a
    smooth, non-stiff two-variable ODE and the high-order dense solution makes
    tolerance-convergence and energy-residual checks inexpensive. Solver,
    tolerances, output times, and maximum step are all configurable.

    Initial states must satisfy the massive-particle condition
    :math:`v_\mathrm{tot}<1`. By default integration terminates if numerical
    evolution reaches its null boundary.
    """

    state = _validated_initial_value_problem(initial_state, t_span)
    if float(timelike_margin(state[0], state[1], wormhole, spiral)) <= 0.0:
        raise ValueError("initial_state is not timelike under paper equation (19)")

    events = None
    if stop_at_null_boundary:

        # solve_ivp calls this after each accepted step and locates a zero.
        def null_boundary(
            _time: float,
            y: np.ndarray,
            _wormhole: WormholeParameters,
            _spiral: SpiralParameters,
        ) -> float:
            return float(timelike_margin(y[0], y[1], wormhole, spiral))

        null_boundary.terminal = True  # type: ignore[attr-defined]
        null_boundary.direction = -1.0  # type: ignore[attr-defined]
        events = null_boundary

    return _solve_radial_initial_value_problem(
        state,
        t_span,
        wormhole,
        spiral,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=events,
    )


def integrate_vector_field_continuation(
    initial_state: Sequence[float],
    t_span: tuple[float, float],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    t_eval: Sequence[float] | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    dense_output: bool = False,
) -> Any:
    r"""Integrate the mathematical continuation of the radial vector field.

    Unlike :func:`integrate_trajectory`, this diagnostic/SciML entry point
    accepts finite states with a nonpositive timelike margin and imposes no
    null-boundary event. Such exterior states are not physical timelike
    particle trajectories. The solve uses the package's validated
    :func:`~wormhole_sciml.dynamics.radial_system` directly and therefore does
    not evaluate energy, a Lorentz factor, or ``sqrt(C)``.

    DOP853 and the default physical-integrator tolerances are retained for a
    consistent numerical interface. ``method``, ``rtol``, ``atol``, and
    ``max_step`` remain explicit and configurable for convergence studies.
    """

    state = _validated_initial_value_problem(initial_state, t_span)
    return _solve_radial_initial_value_problem(
        state,
        t_span,
        wormhole,
        spiral,
        method=method,
        t_eval=t_eval,
        dense_output=dense_output,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=None,
    )
