"""Phase-B complete-orbit generation for the finite-time Model-A project.

This module contains no supervised-row extraction and no machine-learning
code.  It reuses the validated Simple GEB dynamics and stores the numerical
representation used by SciPy's DOP853 dense output without pickling solver
objects.  The stored coefficients can therefore be queried later without
rerunning an ODE integration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import (
    conserved_energy,
    radial_system,
    timelike_margin,
    velocity_from_energy,
)
from .physics_gate import (
    VALIDATION_SOLVER,
    experiment_parameters,
    state_geometry,
    xi_from_state,
)


X_ENDPOINT = 17.0
U_TH_LOW = 0.01
DIFFICULT_HIGH = 0.30
LOW_EDGE_HIGH = 0.08
HIGH_EDGE_LOW = 0.80
XI_DATA_EDGE = 0.99
MAX_EXACT_TIME = 500.0
ENERGY_DRIFT_LIMIT = 1.0e-9
ENDPOINT_TOLERANCE = 5.0e-10
THROAT_TOLERANCE = 1.0e-9
DUPLICATE_TOLERANCE = 1.0e-12
SUSPICIOUS_DUPLICATE_TOLERANCE = 1.0e-10
MASTER_SEED = 2_026_083_101
STRESS_U_TH = (0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.90)

SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "train": {"broad": 2458, "difficult": 1229, "low_edge": 205, "high_edge": 204},
    "validation": {"broad": 614, "difficult": 307, "low_edge": 52, "high_edge": 51},
    "test": {"broad": 614, "difficult": 307, "low_edge": 51, "high_edge": 52},
}


def validated_u_th_high() -> float:
    """Return the repository's exact positive ``xi=0.99`` throat edge."""

    wormhole, spiral = experiment_parameters()
    center, half_width = state_geometry(0.0, wormhole, spiral)
    return float(center + XI_DATA_EDGE * half_width)


def stratum_bounds() -> dict[str, tuple[float, float]]:
    high = validated_u_th_high()
    return {
        "broad": (U_TH_LOW, high),
        "difficult": (U_TH_LOW, DIFFICULT_HIGH),
        "low_edge": (U_TH_LOW, LOW_EDGE_HIGH),
        "high_edge": (HIGH_EDGE_LOW, high),
    }


def _seed_for(label: str) -> int:
    digest = hashlib.sha256(f"{MASTER_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def generation_seeds() -> dict[str, int]:
    return {
        f"{split}:{stratum}": _seed_for(f"{split}:{stratum}")
        for split, counts in SPLIT_COUNTS.items()
        for stratum in counts
    }


def _stratified_values(
    low: float,
    high: float,
    count: int,
    seed: int,
) -> np.ndarray:
    """Return one seeded-jitter point from each equal-width stratum."""

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.15, 0.85, count)
    values = low + (np.arange(count, dtype=np.float64) + jitter) * (
        (high - low) / count
    )
    return values[rng.permutation(count)]


def build_orbit_plan() -> list[dict[str, Any]]:
    """Construct deterministic, unique split/stratum orbit labels."""

    bounds = stratum_bounds()
    seeds = generation_seeds()
    plan: list[dict[str, Any]] = []
    used: list[float] = list(STRESS_U_TH)
    for split, counts in SPLIT_COUNTS.items():
        for stratum, count in counts.items():
            low, high = bounds[stratum]
            rng = np.random.default_rng(seeds[f"{split}:{stratum}"])
            values = _stratified_values(
                low, high, count, seeds[f"{split}:{stratum}"]
            )
            for ordinal, original in enumerate(values):
                value = float(original)
                attempts = 0
                while min(abs(value - previous) for previous in used) <= DUPLICATE_TOLERANCE:
                    cell = int(math.floor((value - low) / (high - low) * count))
                    cell = min(max(cell, 0), count - 1)
                    value = low + (cell + rng.uniform(0.15, 0.85)) * (
                        (high - low) / count
                    )
                    attempts += 1
                    if attempts > 100:
                        raise RuntimeError("could not construct a unique stratified orbit label")
                used.append(value)
                plan.append(
                    {
                        "split": split,
                        "trajectory_stratum": stratum,
                        "stratum_ordinal": ordinal,
                        "u_th": value,
                        "generation_seed": seeds[f"{split}:{stratum}"],
                    }
                )
    return plan


def orbit_id_from_u_th(u_th: float) -> str:
    payload = np.asarray([u_th], dtype="<f8").tobytes()
    return "phaseb-" + hashlib.sha256(payload).hexdigest()[:20]


def _right_endpoint_event(_time: float, state: np.ndarray, *_args: Any) -> float:
    return float(state[0] - X_ENDPOINT)


_right_endpoint_event.terminal = True  # type: ignore[attr-defined]
_right_endpoint_event.direction = 1.0  # type: ignore[attr-defined]


def _null_boundary_event(
    _time: float, state: np.ndarray, wormhole: Any, spiral: Any
) -> float:
    return float(timelike_margin(state[0], state[1], wormhole, spiral))


_null_boundary_event.terminal = True  # type: ignore[attr-defined]
_null_boundary_event.direction = -1.0  # type: ignore[attr-defined]


def _throat_event(_time: float, state: np.ndarray, *_args: Any) -> float:
    return float(state[0])


_throat_event.terminal = False  # type: ignore[attr-defined]
_throat_event.direction = 1.0  # type: ignore[attr-defined]


@dataclass(slots=True)
class DenseOrbit:
    """Portable arrays for the exact DOP853 dense polynomial."""

    segment_end: np.ndarray
    y_old: np.ndarray
    coefficients: np.ndarray

    def evaluate(self, times: Any) -> np.ndarray:
        return evaluate_dop853_dense(
            self.segment_end, self.y_old, self.coefficients, times
        )


def dense_orbit_from_solution(solution: Any) -> DenseOrbit:
    interpolants = solution.sol.interpolants
    return DenseOrbit(
        segment_end=np.asarray([item.t for item in interpolants], dtype=np.float64),
        y_old=np.asarray([item.y_old for item in interpolants], dtype=np.float64),
        coefficients=np.asarray([item.F for item in interpolants], dtype=np.float64),
    )


def evaluate_dop853_dense(
    segment_end: np.ndarray,
    y_old: np.ndarray,
    coefficients: np.ndarray,
    times: Any,
) -> np.ndarray:
    """Evaluate serialized SciPy DOP853 dense-output coefficients.

    The recurrence is the public numerical representation copied from the
    solver result, not a pickled SciPy object.  Output has shape ``(2,)`` for
    a scalar time and ``times.shape + (2,)`` otherwise.
    """

    requested = np.asarray(times, dtype=np.float64)
    flat = requested.reshape(-1)
    if segment_end.ndim != 1 or segment_end.size == 0:
        raise ValueError("segment_end must be a nonempty one-dimensional array")
    if np.any(flat < -1e-13) or np.any(flat > segment_end[-1] + 1e-13):
        raise ValueError("requested time lies outside the stored orbit")
    indices = np.searchsorted(segment_end, flat, side="left")
    indices = np.minimum(indices, segment_end.size - 1)
    starts = np.zeros_like(flat)
    positive = indices > 0
    starts[positive] = segment_end[indices[positive] - 1]
    step = segment_end[indices] - starts
    coordinate = (flat - starts) / step
    selected = coefficients[indices]
    value = np.zeros((flat.size, 2), dtype=np.float64)
    for order, coefficient in enumerate(selected[:, ::-1, :].transpose(1, 0, 2)):
        value += coefficient
        factor = coordinate if order % 2 == 0 else 1.0 - coordinate
        value *= factor[:, None]
    value += y_old[indices]
    shaped = value.reshape(requested.shape + (2,))
    return shaped if requested.ndim else shaped.reshape(2)


def integrate_complete_orbit(u_th: float) -> tuple[dict[str, Any], DenseOrbit]:
    """Integrate and validate one complete ``x=-17`` to ``x=+17`` orbit."""

    wormhole, spiral = experiment_parameters()
    energy0 = float(conserved_energy(0.0, u_th, wormhole, spiral))
    u_left = float(
        velocity_from_energy(-X_ENDPOINT, energy0, wormhole, spiral, branch=1)
    )
    if not np.isfinite(u_left):
        raise ValueError("energy branch has no finite left-end state")
    solution = solve_ivp(
        radial_system,
        (0.0, MAX_EXACT_TIME),
        np.asarray([-X_ENDPOINT, u_left], dtype=np.float64),
        args=(wormhole, spiral),
        method=VALIDATION_SOLVER.method,
        rtol=VALIDATION_SOLVER.rtol,
        atol=VALIDATION_SOLVER.atol,
        max_step=0.2,
        dense_output=True,
        events=(_right_endpoint_event, _null_boundary_event, _throat_event),
    )
    if not solution.success:
        raise RuntimeError(f"solver failure: {solution.message}")
    if len(solution.t_events[0]) != 1:
        raise RuntimeError("right endpoint was not reached exactly once")
    if len(solution.t_events[1]) != 0:
        raise RuntimeError("trajectory reached the null boundary")
    if len(solution.t_events[2]) != 1:
        raise RuntimeError("trajectory did not cross the throat exactly once")

    dense = dense_orbit_from_solution(solution)
    midpoints = 0.5 * (solution.t[:-1] + solution.t[1:])
    diagnostic_times = np.unique(np.concatenate((solution.t, midpoints)))
    states = np.asarray(solution.sol(diagnostic_times), dtype=np.float64)
    if not np.all(np.isfinite(states)):
        raise FloatingPointError("dense reference contains NaN or Inf")
    margins = np.asarray(
        timelike_margin(states[0], states[1], wormhole, spiral), dtype=np.float64
    )
    xi = np.asarray(
        xi_from_state(states[0], states[1], wormhole, spiral), dtype=np.float64
    )
    energies = np.asarray(
        conserved_energy(states[0], states[1], wormhole, spiral), dtype=np.float64
    )
    relative_energy = np.abs(energies - energy0) / (abs(energy0) + 1.0e-12)
    maximum_energy_drift = float(np.max(relative_energy))
    minimum_c = float(np.min(margins))
    minimum_xi_margin = float(np.min(1.0 - np.abs(xi)))
    endpoint_state = np.asarray(solution.y_events[0][0], dtype=np.float64)
    throat_state = np.asarray(solution.y_events[2][0], dtype=np.float64)
    endpoint_residual = float(abs(endpoint_state[0] - X_ENDPOINT))
    throat_residual = float(abs(throat_state[1] - u_th))
    monotonic = bool(
        np.all(np.diff(states[0]) > -1.0e-12) and np.min(states[1]) > 0.0
    )
    finite_dense_arrays = bool(
        np.all(np.isfinite(dense.segment_end))
        and np.all(np.isfinite(dense.y_old))
        and np.all(np.isfinite(dense.coefficients))
    )
    audit_times = np.linspace(0.0, float(solution.t[-1]), 19, dtype=np.float64)
    portable = dense.evaluate(audit_times)
    scipy_values = np.asarray(solution.sol(audit_times).T, dtype=np.float64)
    dense_reconstruction_error = float(np.max(np.abs(portable - scipy_values)))

    failures: list[str] = []
    if minimum_c <= 0.0:
        failures.append("nonpositive_C")
    if minimum_xi_margin <= 0.0 or not np.all(np.isfinite(xi)):
        failures.append("invalid_xi")
    if maximum_energy_drift > ENERGY_DRIFT_LIMIT:
        failures.append("energy_drift_limit")
    if endpoint_residual > ENDPOINT_TOLERANCE:
        failures.append("endpoint_residual")
    if throat_residual > THROAT_TOLERANCE:
        failures.append("throat_state_mismatch")
    if not monotonic:
        failures.append("nonmonotonic_throughgoing_branch")
    if not finite_dense_arrays or dense_reconstruction_error > 5.0e-14:
        failures.append("dense_reference_invalid")
    if failures:
        raise RuntimeError(",".join(failures))

    left_xi = float(xi_from_state(-X_ENDPOINT, u_left, wormhole, spiral))
    right_xi = float(
        xi_from_state(endpoint_state[0], endpoint_state[1], wormhole, spiral)
    )
    throat_xi = float(xi_from_state(0.0, throat_state[1], wormhole, spiral))
    metadata = {
        "E0": energy0,
        "throat_state": [0.0, float(throat_state[1]), throat_xi, float(timelike_margin(0.0, throat_state[1], wormhole, spiral))],
        "left_endpoint_state": [-X_ENDPOINT, u_left, left_xi, float(timelike_margin(-X_ENDPOINT, u_left, wormhole, spiral))],
        "right_endpoint_state": [float(endpoint_state[0]), float(endpoint_state[1]), right_xi, float(timelike_margin(endpoint_state[0], endpoint_state[1], wormhole, spiral))],
        "t_left": 0.0,
        "t_throat": float(solution.t_events[2][0]),
        "t_right": float(solution.t_events[0][0]),
        "total_crossing_time": float(solution.t_events[0][0]),
        "minimum_C": minimum_c,
        "minimum_one_minus_abs_xi": minimum_xi_margin,
        "maximum_relative_energy_drift": maximum_energy_drift,
        "endpoint_residual": endpoint_residual,
        "throat_velocity_residual": throat_residual,
        "minimum_radial_velocity": float(np.min(states[1])),
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "solver_steps": int(solution.t.size - 1),
        "dense_reconstruction_max_abs_error": dense_reconstruction_error,
        "monotonic_x": monotonic,
    }
    return metadata, dense


_SCALAR_FLOAT_KEYS = (
    "u_th",
    "E0",
    "t_left",
    "t_throat",
    "t_right",
    "total_crossing_time",
    "minimum_C",
    "minimum_one_minus_abs_xi",
    "maximum_relative_energy_drift",
    "endpoint_residual",
    "throat_velocity_residual",
    "minimum_radial_velocity",
    "dense_reconstruction_max_abs_error",
)


def generate_bank(
    rows: Iterable[dict[str, Any]],
    *,
    replacement_seed: int,
    reserved_values: set[float] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate one split, replacing any rejected candidate in its stratum."""

    planned = list(rows)
    if not planned:
        raise ValueError("bank plan is empty")
    split = str(planned[0]["split"])
    bounds = stratum_bounds()
    rng = np.random.default_rng(replacement_seed)
    reserved = set() if reserved_values is None else reserved_values
    accepted: list[dict[str, Any]] = []
    dense_rows: list[DenseOrbit] = []
    rejections: list[dict[str, Any]] = []
    for requested in planned:
        candidate = dict(requested)
        replacement_index = 0
        while True:
            u_th = float(candidate["u_th"])
            try:
                metadata, dense = integrate_complete_orbit(u_th)
            except Exception as error:
                rejections.append(
                    {
                        "split": split,
                        "trajectory_stratum": candidate["trajectory_stratum"],
                        "u_th": u_th,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                low, high = bounds[str(candidate["trajectory_stratum"])]
                while True:
                    replacement = float(rng.uniform(low, high))
                    if all(
                        abs(replacement - value) > DUPLICATE_TOLERANCE
                        for value in reserved
                    ) and all(
                        abs(replacement - value) > DUPLICATE_TOLERANCE
                        for value in STRESS_U_TH
                    ):
                        break
                candidate["u_th"] = replacement
                candidate["replacement_for_u_th"] = u_th
                replacement_index += 1
                if replacement_index > 100:
                    raise RuntimeError("more than 100 replacements were needed")
                continue
            reserved.add(u_th)
            accepted.append(
                {
                    **candidate,
                    **metadata,
                    "orbit_id": orbit_id_from_u_th(u_th),
                }
            )
            dense_rows.append(dense)
            break

    offsets = np.zeros(len(dense_rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([row.segment_end.size for row in dense_rows])
    arrays: dict[str, np.ndarray] = {
        "orbit_id": np.asarray([row["orbit_id"] for row in accepted], dtype="U27"),
        "split": np.asarray([row["split"] for row in accepted], dtype="U16"),
        "trajectory_stratum": np.asarray([row["trajectory_stratum"] for row in accepted], dtype="U16"),
        "stratum_ordinal": np.asarray([row["stratum_ordinal"] for row in accepted], dtype=np.int64),
        "generation_seed": np.asarray([row["generation_seed"] for row in accepted], dtype=np.uint64),
        "throat_state": np.asarray([row["throat_state"] for row in accepted], dtype=np.float64),
        "left_endpoint_state": np.asarray([row["left_endpoint_state"] for row in accepted], dtype=np.float64),
        "right_endpoint_state": np.asarray([row["right_endpoint_state"] for row in accepted], dtype=np.float64),
        "solver_status": np.asarray([row["solver_status"] for row in accepted], dtype=np.int8),
        "solver_message": np.asarray([row["solver_message"] for row in accepted], dtype="U80"),
        "solver_steps": np.asarray([row["solver_steps"] for row in accepted], dtype=np.int32),
        "monotonic_x": np.asarray([row["monotonic_x"] for row in accepted], dtype=np.bool_),
        "segment_offsets": offsets,
        "segment_end": np.concatenate([row.segment_end for row in dense_rows]),
        "dense_y_old": np.concatenate([row.y_old for row in dense_rows]),
        "dense_coefficients": np.concatenate([row.coefficients for row in dense_rows]),
    }
    for key in _SCALAR_FLOAT_KEYS:
        arrays[key] = np.asarray([row[key] for row in accepted], dtype=np.float64)
    return arrays, rejections


def orbit_dense_arrays(
    bank: dict[str, np.ndarray] | Any, orbit_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(bank["segment_offsets"][orbit_index])
    stop = int(bank["segment_offsets"][orbit_index + 1])
    return (
        np.asarray(bank["segment_end"][start:stop]),
        np.asarray(bank["dense_y_old"][start:stop]),
        np.asarray(bank["dense_coefficients"][start:stop]),
    )


def evaluate_saved_orbit(
    bank: dict[str, np.ndarray] | Any, orbit_index: int, times: Any
) -> np.ndarray:
    return evaluate_dop853_dense(*orbit_dense_arrays(bank, orbit_index), times)


def evaluate_saved_orbit_x_u_xi(
    bank: dict[str, np.ndarray] | Any, orbit_index: int, times: Any
) -> np.ndarray:
    """Return saved exact ``(x, u, xi)`` states at arbitrary valid times."""

    state = evaluate_saved_orbit(bank, orbit_index, times)
    wormhole, spiral = experiment_parameters()
    xi = xi_from_state(state[..., 0], state[..., 1], wormhole, spiral)
    return np.concatenate((state, np.asarray(xi)[..., None]), axis=-1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bank(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "orbit_count": int(arrays["orbit_id"].size),
        "dense_segment_count": int(arrays["segment_end"].size),
        "bytes": path.stat().st_size,
        "schema": {key: str(value.dtype) for key, value in arrays.items()},
    }


def metadata_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    output = []
    for index in range(arrays["orbit_id"].size):
        row: dict[str, Any] = {
            "orbit_id": str(arrays["orbit_id"][index]),
            "split": str(arrays["split"][index]),
            "trajectory_stratum": str(arrays["trajectory_stratum"][index]),
            "stratum_ordinal": int(arrays["stratum_ordinal"][index]),
            "generation_seed": int(arrays["generation_seed"][index]),
            "solver_status": int(arrays["solver_status"][index]),
            "solver_message": str(arrays["solver_message"][index]),
            "solver_steps": int(arrays["solver_steps"][index]),
            "monotonic_x": bool(arrays["monotonic_x"][index]),
        }
        row.update({key: float(arrays[key][index]) for key in _SCALAR_FLOAT_KEYS})
        for key in ("throat_state", "left_endpoint_state", "right_endpoint_state"):
            row[key] = [float(value) for value in arrays[key][index]]
        output.append(row)
    return output


def leakage_diagnostics(banks: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    ids = {name: set(map(str, bank["orbit_id"])) for name, bank in banks.items()}
    values = {name: np.sort(np.asarray(bank["u_th"], dtype=np.float64)) for name, bank in banks.items()}
    intersections: dict[str, list[str]] = {}
    separations: dict[str, float] = {}
    suspicious: list[dict[str, Any]] = []
    names = list(banks)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            label = f"{left}__{right}"
            intersections[label] = sorted(ids[left] & ids[right])
            a, b = values[left], values[right]
            joined = np.concatenate((a, b))
            origin = np.concatenate((np.zeros(a.size, dtype=np.int8), np.ones(b.size, dtype=np.int8)))
            order = np.argsort(joined)
            sorted_values, sorted_origin = joined[order], origin[order]
            cross = sorted_origin[1:] != sorted_origin[:-1]
            differences = np.diff(sorted_values)[cross]
            separations[label] = float(np.min(differences))
            for position in np.flatnonzero(cross & (np.diff(sorted_values) <= SUSPICIOUS_DUPLICATE_TOLERANCE)):
                suspicious.append(
                    {
                        "split_pair": label,
                        "u_th_left": float(sorted_values[position]),
                        "u_th_right": float(sorted_values[position + 1]),
                        "separation": float(sorted_values[position + 1] - sorted_values[position]),
                    }
                )
    exact_u_duplicates = []
    combined = []
    for split, bank in banks.items():
        combined.extend((float(value), split) for value in bank["u_th"])
    counts = Counter(value for value, _split in combined)
    for value, count in counts.items():
        if count > 1:
            exact_u_duplicates.append({"u_th": value, "count": count})
    return {
        "orbit_id_intersections": intersections,
        "all_orbit_id_intersections_empty": all(not value for value in intersections.values()),
        "exact_u_th_duplicates": exact_u_duplicates,
        "minimum_cross_split_u_th_separation": separations,
        "suspicious_threshold": SUSPICIOUS_DUPLICATE_TOLERANCE,
        "suspicious_near_duplicates": suspicious,
        "passed": all(not value for value in intersections.values()) and not exact_u_duplicates,
    }


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
