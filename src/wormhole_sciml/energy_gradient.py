"""Energy derivatives in the established standardized-coordinate geometry.

This implementation is the reusable form of the analytic derivative validated
by ``scripts/evaluate_energy_gradient_alignment.py`` against the repository's
validated ``conserved_energy`` implementation and finite differences.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import areal_radius, areal_radius_derivative
from .physics_gate import experiment_parameters


def energy_gradient_x_xi(x: Any, xi: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dE/dx at fixed xi, dE/dxi at fixed x)`` for the experiment."""

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
        -spiral.omega * spiral.alpha * q_x / corridor_denominator**2
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


def standardized_energy_normal(
    x: Any,
    xi: Any,
    sigma_delta_x: float,
    sigma_delta_xi: float,
    *,
    degeneracy_threshold: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_E, |g_E|, degenerate_mask)`` in standardized error space."""

    E_x, E_xi = energy_gradient_x_xi(x, xi)
    gradient = np.column_stack(
        (sigma_delta_x * E_x, sigma_delta_xi * E_xi)
    )
    norm = np.linalg.norm(gradient, axis=1)
    degenerate = norm <= degeneracy_threshold
    normal = np.full_like(gradient, np.nan)
    normal[~degenerate] = gradient[~degenerate] / norm[~degenerate, None]
    return normal, norm, degenerate
