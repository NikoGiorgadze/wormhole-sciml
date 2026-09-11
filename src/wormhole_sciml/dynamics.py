"""Published equations of motion and physical constraints."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    areal_radius,
    metric_determinant,
    reduced_metric,
    reduced_metric_derivative,
)
from .parameters import SpiralParameters, WormholeParameters


def total_speed_squared(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return :math:`v_\mathrm{tot}^2` from paper equations (18)-(19)."""

    l = np.asarray(proper_radius, dtype=np.float64)
    v = np.asarray(radial_velocity, dtype=np.float64)
    radius = areal_radius(l, wormhole)
    effective_omega = spiral.omega + spiral.alpha * v
    return np.asarray(
        v**2 + np.sin(spiral.theta) ** 2 * radius**2 * effective_omega**2,
        dtype=np.float64,
    )


def timelike_margin(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return :math:`1-v_\mathrm{tot}^2`; massive motion requires it > 0."""

    return 1.0 - total_speed_squared(
        proper_radius, radial_velocity, wormhole, spiral
    )


def is_admissible(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.bool_]:
    """Return whether each phase-space point is timelike."""

    return np.asarray(
        timelike_margin(proper_radius, radial_velocity, wormhole, spiral) > 0.0
    )


def velocity_bounds(
    proper_radius: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the lower and upper radial velocities on the null boundary.

    This solves paper equation (19) at arbitrary ``l``. At ``l=0`` it reduces
    to equation (20). Values are NaN where the quadratic has no real root.
    """

    radius = areal_radius(proper_radius, wormhole)
    q = np.sin(spiral.theta) ** 2 * radius**2
    coefficient = 1.0 + q * spiral.alpha**2
    # Complete the square in v to obtain the center and half-width of the roots.
    center = -q * spiral.omega * spiral.alpha / coefficient
    discriminant = 1.0 + q * (spiral.alpha**2 - spiral.omega**2)
    with np.errstate(invalid="ignore"):
        half_width = np.sqrt(discriminant) / coefficient
    return (
        np.asarray(center - half_width, dtype=np.float64),
        np.asarray(center + half_width, dtype=np.float64),
    )


def lorentz_factor(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return :math:`\gamma=dt/d\tau` from paper equation (13)."""

    margin = timelike_margin(proper_radius, radial_velocity, wormhole, spiral)
    if np.any(margin <= 0.0):
        raise ValueError("Lorentz factor is real only for timelike states")
    return np.asarray(1.0 / np.sqrt(margin), dtype=np.float64)


def conserved_energy(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    """Return the conserved specific energy :math:`E` (paper equation 12)."""

    velocity = np.asarray(radial_velocity, dtype=np.float64)
    gamma = lorentz_factor(proper_radius, velocity, wormhole, spiral)
    radius = areal_radius(proper_radius, wormhole)
    q = np.sin(spiral.theta) ** 2 * radius**2
    effective_omega = spiral.omega + spiral.alpha * velocity
    # This algebraic form avoids subtracting two very large metric terms.
    return np.asarray(
        gamma * (1.0 - q * spiral.omega * effective_omega),
        dtype=np.float64,
    )


def velocity_from_energy(
    proper_radius: ArrayLike,
    energy: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    branch: int = 1,
) -> NDArray[np.float64]:
    """Return a radial-velocity branch of the energy first integral (eq. 14).

    ``branch=1`` selects the plus sign and ``branch=-1`` the minus sign in the
    paper. NaN marks radii where the selected energy has no real state.
    """

    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    metric = reduced_metric(proper_radius, wormhole, spiral)
    energy_array = np.asarray(energy, dtype=np.float64)
    g00 = metric[..., 0, 0]
    g01 = metric[..., 0, 1]
    g11 = metric[..., 1, 1]
    radicand = g00 + energy_array**2
    determinant_term = g01**2 - g00 * g11
    denominator = g01**2 + energy_array**2 * g11
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(radicand)
        velocity = root / denominator * (
            -g01 * root + branch * energy_array * np.sqrt(determinant_term)
        )
    return np.asarray(velocity, dtype=np.float64)


def energy_branch_for_state(
    proper_radius: float,
    radial_velocity: float,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> int:
    """Return the equation-(14) sign that passes through a given state."""

    energy = conserved_energy(proper_radius, radial_velocity, wormhole, spiral)
    candidates = {
        branch: float(
            velocity_from_energy(
                proper_radius, energy, wormhole, spiral, branch=branch
            )
        )
        for branch in (-1, 1)
    }
    finite = {key: value for key, value in candidates.items() if np.isfinite(value)}
    if not finite:
        raise ValueError("state is not represented by a real energy branch")
    return min(finite, key=lambda key: abs(finite[key] - radial_velocity))


def radial_acceleration(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    """Return :math:`d^2l/dt^2` in the closed form of paper equation (24)."""

    l = np.asarray(proper_radius, dtype=np.float64)
    velocity = np.asarray(radial_velocity, dtype=np.float64)
    b0 = wormhole.throat_radius
    m = wormhole.m
    sine_squared = np.sin(spiral.theta) ** 2
    # Keeping these repeated pieces named makes equation (24) visible in code.
    base = b0**m + l**m
    radius_squared = base ** (2.0 / m)
    effective_omega = spiral.omega + spiral.alpha * velocity
    bracket = (
        spiral.alpha * velocity
        - spiral.omega
        + 2.0 * spiral.omega * velocity**2
        + sine_squared * effective_omega**2 * spiral.omega * radius_squared
    )
    numerator = (
        -sine_squared
        * effective_omega
        * base ** (2.0 / m - 1.0)
        * l ** (m - 1)
        * bracket
    )
    denominator = 1.0 + sine_squared * (
        spiral.alpha**2 - spiral.omega**2
    ) * radius_squared
    return np.asarray(numerator / denominator, dtype=np.float64)


def radial_acceleration_from_metric(
    proper_radius: ArrayLike,
    radial_velocity: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    """Evaluate the coordinate-time geodesic equation independently of (24).

    Direct reduction gives corrected coefficients
    ``A2 = 2*g01*d01 + 2*g11*d00 - g00*d11`` and
    ``A3 = 2*g11*d01 - g01*d11``. The factors multiplying ``g11`` are missing
    in the paper's printed equations (10)-(11); the corrected result agrees
    with its equation (24) and the trajectory equations in both legacy
    Mathematica notebooks.
    """

    velocity = np.asarray(radial_velocity, dtype=np.float64)
    metric = reduced_metric(proper_radius, wormhole, spiral)
    derivative = reduced_metric_derivative(proper_radius, wormhole, spiral)
    g00 = metric[..., 0, 0]
    g01 = metric[..., 0, 1]
    g11 = metric[..., 1, 1]
    d00 = derivative[..., 0, 0]
    d01 = derivative[..., 0, 1]
    d11 = derivative[..., 1, 1]
    a0 = g00 * d00
    a1 = 3.0 * g01 * d00
    a2 = 2.0 * g01 * d01 + 2.0 * g11 * d00 - g00 * d11
    a3 = 2.0 * g11 * d01 - g01 * d11
    determinant = metric_determinant(proper_radius, wormhole, spiral)
    return np.asarray(
        (a0 + a1 * velocity + a2 * velocity**2 + a3 * velocity**3)
        / (2.0 * determinant),
        dtype=np.float64,
    )


def radial_system(
    _time: float,
    state: NDArray[np.float64],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    """First-order form ``(dl/dt, dv/dt)`` of the radial dynamics."""

    proper_radius, velocity = state
    # solve_ivp expects derivatives in the same order as the state: (l, v).
    return np.array(
        [velocity, radial_acceleration(proper_radius, velocity, wormhole, spiral)],
        dtype=np.float64,
    )
