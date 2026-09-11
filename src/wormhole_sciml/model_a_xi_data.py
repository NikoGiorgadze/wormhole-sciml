"""Deterministic coordinate transformation for the ``(x, xi)`` Model-A task."""

from __future__ import annotations

from typing import Any

import numpy as np

from .parameters import SpiralParameters, WormholeParameters
from .physics_gate import xi_from_state


IDENTITY_FIELDS = ("x_component", "xi_sign", "stratum", "is_physical")


def transform_state_pairs_to_x_xi(
    data: dict[str, np.ndarray],
    wormhole: WormholeParameters,
    spiral: SpiralParameters,
) -> dict[str, np.ndarray]:
    """Transform stored finite-step ``(x, u)`` pairs without approximation."""

    required = {"x", "u", "delta_x", "delta_u", *IDENTITY_FIELDS}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"missing source fields: {sorted(missing)}")
    count = data["x"].size
    if any(np.asarray(data[name]).size != count for name in required):
        raise ValueError("source fields do not have a common row count")

    x = np.asarray(data["x"], dtype=np.float64)
    u = np.asarray(data["u"], dtype=np.float64)
    delta_x = np.asarray(data["delta_x"], dtype=np.float64)
    delta_u = np.asarray(data["delta_u"], dtype=np.float64)
    xi = xi_from_state(x, u, wormhole, spiral)
    next_xi = xi_from_state(x + delta_x, u + delta_u, wormhole, spiral)
    transformed = {
        "x": x,
        "xi": np.asarray(xi, dtype=np.float64),
        "delta_x": delta_x,
        "delta_xi": np.asarray(next_xi - xi, dtype=np.float64),
    }
    transformed.update({name: np.asarray(data[name]) for name in IDENTITY_FIELDS})
    return transformed


def x_xi_normalization(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return population statistics for transformed physical training rows only."""

    columns = ("x", "xi", "delta_x", "delta_xi")
    return {
        "source_dataset": "physical_train_x_xi",
        "source_row_count": int(data["x"].size),
        "dtype": "float64",
        "standard_deviation_definition": "population (ddof=0)",
        "input_columns": ["x", "xi"],
        "target_columns": ["delta_x", "delta_xi"],
        "columns": {
            name: {
                "mean": float(np.mean(data[name], dtype=np.float64)),
                "standard_deviation": float(
                    np.std(data[name], ddof=0, dtype=np.float64)
                ),
            }
            for name in columns
        },
    }
