"""Physical parameters for the published baseline model."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class WormholeParameters:
    r"""Generalized Ellis-Bronnikov geometry parameters.

    ``throat_radius`` is :math:`b_0` and ``m`` is the even positive integer in
    :math:`R(l)=(b_0^m+l^m)^{1/m}`. Even ``m >= 2`` makes the areal-radius
    function real and smooth on the full proper-radial domain
    :math:`l\in(-\infty,\infty)` used in the paper.
    """

    throat_radius: float = 1.0
    m: int = 2

    def __post_init__(self) -> None:
        # Reject invalid geometry immediately instead of failing inside a solver.
        if not math.isfinite(self.throat_radius) or self.throat_radius <= 0.0:
            raise ValueError("throat_radius must be finite and positive")
        if isinstance(self.m, bool) or not isinstance(self.m, int):
            raise TypeError("m must be an even integer")
        if self.m < 2 or self.m % 2:
            raise ValueError("m must be even and at least 2")


@dataclass(frozen=True, slots=True)
class SpiralParameters:
    r"""Rigidly rotating Archimedean spiral parameters.

    The constrained field line has constant polar angle ``theta`` and
    laboratory-frame azimuth

    .. math:: \Phi(t,l)=\alpha l+\omega t.

    The paper's force-free escaping regime uses ``abs(omega) < abs(alpha)``;
    this is exposed as :attr:`has_subluminal_terminal_speed` rather than
    enforced so that admissibility failures can be studied explicitly.
    """

    omega: float = 1.0
    alpha: float = -1.25
    theta: float = math.pi / 2.0

    def __post_init__(self) -> None:
        # Keeping validation here makes every later function safe to call.
        values = (self.omega, self.alpha, self.theta)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("spiral parameters must be finite")
        if not 0.0 <= self.theta <= math.pi:
            raise ValueError("theta must lie in [0, pi]")

    @property
    def has_subluminal_terminal_speed(self) -> bool:
        r"""Whether :math:`|v_\infty|=|\omega/\alpha|<1`."""

        return self.alpha != 0.0 and abs(self.omega) < abs(self.alpha)
