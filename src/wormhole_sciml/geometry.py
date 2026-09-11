"""Geometry of the generalized Ellis-Bronnikov baseline problem."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import SpiralParameters, WormholeParameters


def areal_radius(
    proper_radius: ArrayLike, wormhole: WormholeParameters
) -> NDArray[np.float64]:
    r"""Return the paper's areal radius :math:`R(l)` (equation 16)."""

    l = np.asarray(proper_radius, dtype=np.float64)
    b0 = wormhole.throat_radius
    m = wormhole.m
    # NumPy applies the same formula to a scalar or to every element of an array.
    return np.asarray((b0**m + l**m) ** (1.0 / m), dtype=np.float64)


def areal_radius_derivative(
    proper_radius: ArrayLike, wormhole: WormholeParameters
) -> NDArray[np.float64]:
    r"""Return :math:`dR/dl` on the full proper-radial domain."""

    l = np.asarray(proper_radius, dtype=np.float64)
    b0 = wormhole.throat_radius
    m = wormhole.m
    base = b0**m + l**m
    return np.asarray(l ** (m - 1) * base ** (1.0 / m - 1.0), dtype=np.float64)


def reduced_metric(
    proper_radius: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return the constrained :math:`(t,l)` metric from paper equation (21).

    The returned array has shape ``proper_radius.shape + (2, 2)`` with index
    order ``(t, l)``.
    """

    radius = areal_radius(proper_radius, wormhole)
    # q is the common angular factor R(l)^2 sin(theta)^2.
    q = np.sin(spiral.theta) ** 2 * radius**2
    # The final two axes hold the 2x2 matrix; earlier axes follow the input l.
    metric = np.empty(q.shape + (2, 2), dtype=np.float64)
    metric[..., 0, 0] = -1.0 + q * spiral.omega**2
    metric[..., 0, 1] = q * spiral.omega * spiral.alpha
    metric[..., 1, 0] = metric[..., 0, 1]
    metric[..., 1, 1] = 1.0 + q * spiral.alpha**2
    return metric


def reduced_metric_derivative(
    proper_radius: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return :math:`\partial_l g_{ab}` for the reduced metric."""

    radius = areal_radius(proper_radius, wormhole)
    radius_prime = areal_radius_derivative(proper_radius, wormhole)
    q_prime = 2.0 * np.sin(spiral.theta) ** 2 * radius * radius_prime
    derivative = np.empty(q_prime.shape + (2, 2), dtype=np.float64)
    derivative[..., 0, 0] = q_prime * spiral.omega**2
    derivative[..., 0, 1] = q_prime * spiral.omega * spiral.alpha
    derivative[..., 1, 0] = derivative[..., 0, 1]
    derivative[..., 1, 1] = q_prime * spiral.alpha**2
    return derivative


def metric_determinant(
    proper_radius: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> NDArray[np.float64]:
    r"""Return :math:`\det g` in the analytic form of paper equation (23)."""

    radius = areal_radius(proper_radius, wormhole)
    q = np.sin(spiral.theta) ** 2 * radius**2
    return np.asarray(
        -(1.0 + q * (spiral.alpha**2 - spiral.omega**2)),
        dtype=np.float64,
    )
