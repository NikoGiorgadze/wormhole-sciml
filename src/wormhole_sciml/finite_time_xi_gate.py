"""Exponentially saturating identity-preserving gate for finite-time xi residuals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .physics_gate import experiment_parameters, xi_time_derivative


S_STAR_CANDIDATES = (1.0, 5.0, 10.0)


def saturating_gate(s: Any, s_star: float) -> np.ndarray:
    """Return ``s_star * (1-exp(-s/s_star))`` using stable ``expm1``."""

    if not np.isfinite(s_star) or s_star <= 0.0:
        raise ValueError("s_star must be finite and positive")
    elapsed = np.asarray(s, dtype=np.float64)
    if np.any(elapsed < 0.0):
        raise ValueError("physical elapsed time cannot be negative")
    return np.asarray(-s_star * np.expm1(-elapsed / s_star), dtype=np.float64)


def saturating_gate_derivative(s: Any, s_star: float) -> np.ndarray:
    """Return the analytic derivative ``exp(-s/s_star)``."""

    if not np.isfinite(s_star) or s_star <= 0.0:
        raise ValueError("s_star must be finite and positive")
    elapsed = np.asarray(s, dtype=np.float64)
    return np.asarray(np.exp(-elapsed / s_star), dtype=np.float64)


def gate_over_s(s: Any, s_star: float) -> np.ndarray:
    """Return ``g(s;s_star)/s`` with its continuous value one at zero."""

    elapsed = np.asarray(s, dtype=np.float64)
    gate = saturating_gate(elapsed, s_star)
    ratio = np.ones_like(elapsed)
    np.divide(gate, elapsed, out=ratio, where=elapsed > 0.0)
    return ratio


def construct_saturating_xi_target(
    data: dict[str, np.ndarray], s_star: float
) -> np.ndarray:
    """Construct exact ``F_xi`` without division on identity rows."""

    elapsed = np.asarray(data["s"], dtype=np.float64)
    delta_xi = np.asarray(data["Delta_xi"], dtype=np.float64)
    gate = saturating_gate(elapsed, s_star)
    positive = elapsed > 0.0
    if np.any(gate[positive] <= 0.0):
        raise FloatingPointError("positive elapsed time produced a zero gate")
    target = np.empty_like(elapsed)
    np.divide(delta_xi, gate, out=target, where=positive)
    identity = ~positive
    if np.any(identity):
        wormhole, spiral = experiment_parameters()
        target[identity] = xi_time_derivative(
            np.asarray(data["x0"], dtype=np.float64)[identity],
            np.asarray(data["u0"], dtype=np.float64)[identity],
            wormhole,
            spiral,
        )
    if not np.all(np.isfinite(target)):
        raise FloatingPointError("saturating xi target contains NaN or Inf")
    return target

