"""Derived observables for the constrained particle trajectory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import areal_radius
from .parameters import SpiralParameters, WormholeParameters


def effective_angular_velocity(
    radial_velocity: ArrayLike, spiral: SpiralParameters
) -> NDArray[np.float64]:
    r"""Return :math:`\Omega=d\Phi/dt=\omega+\alpha\,dl/dt`."""

    velocity = np.asarray(radial_velocity, dtype=np.float64)
    return np.asarray(spiral.omega + spiral.alpha * velocity, dtype=np.float64)


def terminal_radial_velocity(spiral: SpiralParameters) -> float:
    r"""Return the force-free terminal value :math:`v_\infty=-\omega/\alpha`."""

    if spiral.alpha == 0.0:
        raise ValueError("a nonzero alpha is required for a terminal velocity")
    return -spiral.omega / spiral.alpha


def azimuth(
    time: ArrayLike,
    proper_radius: ArrayLike,
    spiral: SpiralParameters,
    *,
    phase: float = 0.0,
    frame: str = "laboratory",
) -> NDArray[np.float64]:
    """Return spiral azimuth in the laboratory or co-rotating frame."""

    t = np.asarray(time, dtype=np.float64)
    l = np.asarray(proper_radius, dtype=np.float64)
    if frame == "laboratory":
        rotation = spiral.omega * t
    elif frame == "co_rotating":
        rotation = 0.0
    else:
        raise ValueError("frame must be 'laboratory' or 'co_rotating'")
    return np.asarray(spiral.alpha * l + rotation + phase, dtype=np.float64)


def coordinate_visualization(
    time: ArrayLike,
    proper_radius: ArrayLike,
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
    *,
    phase: float = 0.0,
    frame: str = "laboratory",
    radial_scale: str = "proper",
) -> NDArray[np.float64]:
    r"""Return Cartesian-like coordinates used to visualize the trajectory.

    ``radial_scale='proper'`` matches the legacy/paper plotting convention,
    which substitutes the signed proper coordinate ``l`` as a Euclidean radial
    scale. ``radial_scale='areal'`` uses :math:`R(l)` instead. Neither choice is
    claimed to be an isometric embedding of a wormhole spatial slice.
    """

    l = np.asarray(proper_radius, dtype=np.float64)
    phi = azimuth(time, l, spiral, phase=phase, frame=frame)
    # Broadcasting allows one time with many radii or matching time/radius arrays.
    l, phi = np.broadcast_arrays(l, phi)
    if radial_scale == "proper":
        scale = l
    elif radial_scale == "areal":
        scale = areal_radius(l, wormhole)
    else:
        raise ValueError("radial_scale must be 'proper' or 'areal'")
    sine = np.sin(spiral.theta)
    coordinates = np.stack(
        (
            scale * sine * np.cos(phi),
            scale * sine * np.sin(phi),
            scale * np.cos(spiral.theta),
        ),
        axis=-1,
    )
    return np.asarray(coordinates, dtype=np.float64)
