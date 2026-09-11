"""Evaluation-only helpers for direct and recursively composed frozen models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .dynamics import conserved_energy, timelike_margin
from .finite_time_validation import scalar_error_metrics
from .finite_time_validation import DENSE_ANCHOR_X
from .phase_b_orbits import evaluate_saved_orbit_x_u_xi
from .phase_c_finite_time import invert_saved_orbit_x
from .physics_gate import experiment_parameters, state_from_xi


LOCAL_STEP = 0.2
MAXIMUM_K = 300
ANCHOR_INDICES = np.asarray(
    [0, 2, 4, 6, 7, 8, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 28, 31],
    dtype=np.int16,
)
HORIZON_K = np.asarray(
    sorted(set(range(1, 51)) | set(range(55, 101, 5)) | set(range(110, 301, 10))),
    dtype=np.int16,
)


def build_common_anchor_bank(
    banks: list[tuple[str, dict[str, np.ndarray]]],
    *,
    local_step: float = LOCAL_STEP,
) -> dict[str, np.ndarray]:
    """Build deterministic on-manifold anchors from frozen saved-orbit banks."""

    anchor_x = DENSE_ANCHOR_X[ANCHOR_INDICES]
    rows: dict[str, list[Any]] = {name: [] for name in (
        "anchor_id", "source_bank", "orbit_id", "trajectory_stratum",
        "source_orbit_index", "anchor_index", "u_th", "E0", "x0", "u0", "xi0",
        "t0", "t_right", "remaining_time", "maximum_physical_k", "maximum_evaluated_k",
        "inversion_residual",
    )}
    for source_bank, bank in banks:
        for orbit_index in range(int(bank["orbit_id"].size)):
            t0, residual = invert_saved_orbit_x(bank, orbit_index, anchor_x)
            state = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t0)
            remaining = float(bank["t_right"][orbit_index]) - t0
            maximum_physical = np.floor((remaining + 1.0e-12) / local_step).astype(np.int64)
            orbit_id = str(bank["orbit_id"][orbit_index])
            for position, anchor_index in enumerate(ANCHOR_INDICES):
                values = {
                    "anchor_id": f"{orbit_id}__a{int(anchor_index):02d}",
                    "source_bank": source_bank,
                    "orbit_id": orbit_id,
                    "trajectory_stratum": str(bank["trajectory_stratum"][orbit_index]),
                    "source_orbit_index": orbit_index,
                    "anchor_index": int(anchor_index),
                    "u_th": float(bank["u_th"][orbit_index]),
                    "E0": float(bank["E0"][orbit_index]),
                    "x0": float(state[position, 0]),
                    "u0": float(state[position, 1]),
                    "xi0": float(state[position, 2]),
                    "t0": float(t0[position]),
                    "t_right": float(bank["t_right"][orbit_index]),
                    "remaining_time": float(remaining[position]),
                    "maximum_physical_k": int(maximum_physical[position]),
                    "maximum_evaluated_k": int(min(maximum_physical[position], MAXIMUM_K)),
                    "inversion_residual": float(residual[position]),
                }
                for name, value in values.items():
                    rows[name].append(value)
    integer = {"source_orbit_index": np.int32, "anchor_index": np.int16,
               "maximum_physical_k": np.int32, "maximum_evaluated_k": np.int16}
    string = {"anchor_id": "U64", "source_bank": "U20", "orbit_id": "U27",
              "trajectory_stratum": "U16"}
    output = {
        name: np.asarray(values, dtype=integer.get(name, string.get(name, np.float64)))
        for name, values in rows.items()
    }
    if np.max(np.abs(output["inversion_residual"])) > 5.0e-12:
        raise RuntimeError("anchor inversion residual exceeded frozen gate")
    return output


def physical_diagnostics(x: np.ndarray, xi: np.ndarray) -> dict[str, np.ndarray]:
    """Reconstruct physical velocity, margin, energy, and accepted-domain validity."""

    x_array = np.asarray(x, dtype=np.float64)
    xi_array = np.asarray(xi, dtype=np.float64)
    wormhole, spiral = experiment_parameters()
    _, u = state_from_xi(x_array, xi_array, wormhole, spiral)
    margin = timelike_margin(x_array, u, wormhole, spiral)
    finite = np.isfinite(x_array) & np.isfinite(xi_array) & np.isfinite(u) & np.isfinite(margin)
    valid = finite & (np.abs(xi_array) < 1.0) & (margin > 0.0) & (x_array >= -17.0) & (x_array <= 17.0)
    energy = np.full(x_array.shape, np.nan, dtype=np.float64)
    if np.any(valid):
        energy[valid] = conserved_energy(x_array[valid], u[valid], wormhole, spiral)
    return {
        "u": np.asarray(u, dtype=np.float64),
        "C": np.asarray(margin, dtype=np.float64),
        "E": energy,
        "valid": np.asarray(valid, dtype=np.bool_),
    }


def recursive_rollout(
    initial_x: np.ndarray,
    initial_xi: np.ndarray,
    invariant_E0: np.ndarray,
    maximum_available_k: np.ndarray,
    stepper: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    maximum_k: int = MAXIMUM_K,
) -> dict[str, np.ndarray]:
    """Compose a one-step predictor without clipping, projection, or exact restarts."""

    initial_x = np.asarray(initial_x, dtype=np.float64)
    initial_xi = np.asarray(initial_xi, dtype=np.float64)
    invariant_E0 = np.asarray(invariant_E0, dtype=np.float64)
    maximum_available_k = np.asarray(maximum_available_k, dtype=np.int64)
    count = initial_x.size
    if not (initial_xi.shape == invariant_E0.shape == maximum_available_k.shape == initial_x.shape):
        raise ValueError("rollout inputs must have identical one-dimensional shapes")
    shape = (count, maximum_k + 1)
    x = np.full(shape, np.nan, dtype=np.float64)
    xi = np.full(shape, np.nan, dtype=np.float64)
    u = np.full(shape, np.nan, dtype=np.float64)
    margin = np.full(shape, np.nan, dtype=np.float64)
    energy = np.full(shape, np.nan, dtype=np.float64)
    valid = np.zeros(shape, dtype=np.bool_)
    evaluated = np.zeros(shape, dtype=np.bool_)
    first_failure_k = np.full(count, -1, dtype=np.int16)

    initial = physical_diagnostics(initial_x, initial_xi)
    x[:, 0], xi[:, 0] = initial_x, initial_xi
    u[:, 0], margin[:, 0], energy[:, 0] = initial["u"], initial["C"], initial["E"]
    valid[:, 0] = initial["valid"]
    evaluated[:, 0] = True
    first_failure_k[~initial["valid"]] = 0

    for k in range(1, maximum_k + 1):
        active = valid[:, k - 1] & (maximum_available_k >= k)
        if not np.any(active):
            continue
        indices = np.flatnonzero(active)
        next_x, next_xi = stepper(x[indices, k - 1], xi[indices, k - 1], invariant_E0[indices])
        next_x = np.asarray(next_x, dtype=np.float64)
        next_xi = np.asarray(next_xi, dtype=np.float64)
        if next_x.shape != indices.shape or next_xi.shape != indices.shape:
            raise ValueError("stepper output shape mismatch")
        diagnostics = physical_diagnostics(next_x, next_xi)
        x[indices, k], xi[indices, k] = next_x, next_xi
        u[indices, k], margin[indices, k], energy[indices, k] = (
            diagnostics["u"], diagnostics["C"], diagnostics["E"]
        )
        evaluated[indices, k] = True
        valid[indices, k] = diagnostics["valid"]
        failed = indices[~diagnostics["valid"]]
        first_failure_k[failed[first_failure_k[failed] < 0]] = k
    return {
        "x": x,
        "xi": xi,
        "u": u,
        "C": margin,
        "E": energy,
        "valid": valid,
        "evaluated": evaluated,
        "first_failure_k": first_failure_k,
        "invariant_E0": invariant_E0.copy(),
    }


def direct_predictions_from_original(
    initial_x: np.ndarray,
    initial_xi: np.ndarray,
    invariant_E0: np.ndarray,
    exact_available: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    local_step: float = LOCAL_STEP,
) -> dict[str, np.ndarray]:
    """Evaluate each direct horizon independently from its original exact anchor."""

    initial_x = np.asarray(initial_x, dtype=np.float64)
    initial_xi = np.asarray(initial_xi, dtype=np.float64)
    invariant_E0 = np.asarray(invariant_E0, dtype=np.float64)
    available = np.asarray(exact_available, dtype=np.bool_)
    if available.ndim != 2 or available.shape[0] != initial_x.size:
        raise ValueError("exact_available must be [anchor, k]")
    x = np.full(available.shape, np.nan, dtype=np.float64)
    xi = np.full(available.shape, np.nan, dtype=np.float64)
    x[:, 0], xi[:, 0] = initial_x, initial_xi
    row, k = np.nonzero(available & (np.arange(available.shape[1])[None, :] > 0))
    elapsed = k.astype(np.float64) * float(local_step)
    predicted_x, predicted_xi = predictor(
        initial_x[row], initial_xi[row], invariant_E0[row], elapsed
    )
    x[row, k] = np.asarray(predicted_x, dtype=np.float64)
    xi[row, k] = np.asarray(predicted_xi, dtype=np.float64)
    diagnostics = physical_diagnostics(x, xi)
    valid = diagnostics["valid"] & available
    return {"x": x, "xi": xi, **diagnostics, "valid": valid}


def common_survivor_mask(
    exact_available: np.ndarray,
    direct_valid: np.ndarray,
    local_valid: np.ndarray,
    hybrid_recursive_valid: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(exact_available, dtype=np.bool_)
        & np.asarray(direct_valid, dtype=np.bool_)
        & np.asarray(local_valid, dtype=np.bool_)
        & np.asarray(hybrid_recursive_valid, dtype=np.bool_)
    )


def state_metrics(
    predicted_x: np.ndarray,
    predicted_xi: np.ndarray,
    exact_x: np.ndarray,
    exact_xi: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=np.bool_)
    if not np.any(selected):
        return {"count": 0, "x": None, "xi": None}
    return {
        "count": int(np.sum(selected)),
        "x": scalar_error_metrics(np.asarray(predicted_x)[selected] - np.asarray(exact_x)[selected]),
        "xi": scalar_error_metrics(np.asarray(predicted_xi)[selected] - np.asarray(exact_xi)[selected]),
    }


def energy_metrics(energy: np.ndarray, invariant_E0: np.ndarray, mask: np.ndarray) -> dict[str, float] | None:
    selected = np.asarray(mask, dtype=np.bool_) & np.isfinite(energy)
    if not np.any(selected):
        return None
    error = np.asarray(energy)[selected] - np.asarray(invariant_E0)[selected]
    metrics = scalar_error_metrics(error)
    return {
        "count": int(np.sum(selected)),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "p99_absolute": metrics["p99_absolute"],
        "maximum_absolute": metrics["maximum_absolute"],
    }


def first_and_sustained_crossover(
    rows: list[dict[str, Any]],
    direct_key: str,
    local_key: str,
    *,
    minimum_count: int,
) -> dict[str, Any]:
    """Locate isolated and remainder-wide direct-over-local improvements."""

    usable = [
        row for row in rows
        if int(row.get("common_count", 0)) >= minimum_count
        and np.isfinite(row.get(direct_key, np.nan))
        and np.isfinite(row.get(local_key, np.nan))
    ]
    winners = [float(row[direct_key]) < float(row[local_key]) for row in usable]
    first = next((row for row, wins in zip(usable, winners) if wins), None)
    sustained = None
    for index, row in enumerate(usable):
        if winners[index] and all(winners[index:]):
            sustained = row
            break
    def payload(row: dict[str, Any] | None) -> dict[str, float | int] | None:
        return None if row is None else {"k": int(row["k"]), "s": float(row["s"])}
    return {
        "minimum_common_count": int(minimum_count),
        "well_sampled_k_count": len(usable),
        "first": payload(first),
        "sustained": payload(sustained),
        "crossing_count": int(sum(winners)),
    }
