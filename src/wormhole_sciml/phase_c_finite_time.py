"""Phase-C finite-time transition extraction from frozen Phase-B orbits.

This module performs deterministic sampling, dense-reference evaluation, and
raw physical-data validation only.  It contains no ODE integration, feature
normalization, neural-network code, or training utilities.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics import conserved_energy, timelike_margin
from .phase_b_orbits import (
    evaluate_dop853_dense,
    evaluate_saved_orbit,
    file_sha256,
    orbit_dense_arrays,
)
from .physics_gate import experiment_parameters, xi_from_state


ROWS_PER_ORBIT = 96
GLOBAL_ROWS = 64
ADDITIONAL_ROWS = 32
HARD_U_TH_MAX = 0.30
SENSITIVE_X_LOW = -17.0
SENSITIVE_X_HIGH = -8.5
X_LEFT = -17.0
X_RIGHT = 17.0
PHASE_C_MASTER_SEED = 2_026_083_102
ENERGY_TOLERANCE = 1.0e-9
TIME_TOLERANCE = 2.0e-12
X_INVERSION_TOLERANCE = 5.0e-12
ENDPOINT_X_TOLERANCE = 5.0e-10
NEAR_DUPLICATE_TIME_TOLERANCE = 1.0e-10

SPLIT_SEEDS = {
    "train": 2_026_083_121,
    "validation": 2_026_083_122,
    "test": 2_026_083_123,
    "stress_reference": 2_026_083_129,
}

FLOAT_COLUMNS = (
    "u_th",
    "E0",
    "t_left",
    "t_right",
    "t0",
    "t1",
    "s",
    "x0",
    "u0",
    "xi0",
    "x1",
    "u1",
    "xi1",
    "Delta_x",
    "Delta_xi",
    "C0",
    "C1",
    "E_state_0",
    "E_state_1",
    "relative_energy_error_0",
    "relative_energy_error_1",
    "horizon_fraction",
    "x0_inversion_residual",
    "x1_inversion_residual",
)


def seed_for(split: str, orbit_id: str, design: str) -> int:
    """Derive a stable uint64 sampling seed from saved identifiers."""

    if split not in SPLIT_SEEDS:
        raise ValueError(f"unsupported Phase-C split {split}")
    payload = (
        f"{PHASE_C_MASTER_SEED}:{SPLIT_SEEDS[split]}:{orbit_id}:{design}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _stratified_interior(
    low: float, high: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    jitter = rng.uniform(0.15, 0.85, count)
    return low + (np.arange(count, dtype=np.float64) + jitter) * (
        (high - low) / count
    )


def _fraction_design(
    specifications: tuple[tuple[int, float, float, str, bool], ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independently permuted fractions and their subtype labels."""

    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for count, low, high, label, fixed in specifications:
        if fixed:
            part = np.full(count, low, dtype=np.float64)
        else:
            part = _stratified_interior(low, high, count, rng)
        values.append(part)
        labels.append(np.full(count, label, dtype="U40"))
    fractions = np.concatenate(values)
    subtypes = np.concatenate(labels)
    permutation = rng.permutation(fractions.size)
    return fractions[permutation], subtypes[permutation]


GLOBAL_FRACTIONS = (
    (4, 0.0, 0.0, "global_identity", True),
    (8, 0.0, 0.05, "global_f_0_0p05", False),
    (12, 0.05, 0.25, "global_f_0p05_0p25", False),
    (12, 0.25, 0.50, "global_f_0p25_0p50", False),
    (12, 0.50, 0.80, "global_f_0p50_0p80", False),
    (8, 0.80, 0.95, "global_f_0p80_0p95", False),
    (4, 0.95, 1.00, "global_f_0p95_1", False),
    (4, 1.0, 1.0, "global_endpoint", True),
)

ORDINARY_FRACTIONS = (
    (4, 0.0, 0.05, "additional_f_0_0p05", False),
    (8, 0.05, 0.25, "additional_f_0p05_0p25", False),
    (8, 0.25, 0.50, "additional_f_0p25_0p50", False),
    (6, 0.50, 0.80, "additional_f_0p50_0p80", False),
    (4, 0.80, 1.00, "additional_f_0p80_1", False),
    (2, 1.0, 1.0, "additional_endpoint", True),
)


def global_design(split: str, orbit_id: str) -> dict[str, np.ndarray]:
    anchor_rng = np.random.default_rng(seed_for(split, orbit_id, "global64_anchor"))
    fraction_rng = np.random.default_rng(seed_for(split, orbit_id, "global64_fraction"))
    anchors = _stratified_interior(X_LEFT, X_RIGHT, GLOBAL_ROWS, anchor_rng)
    fractions, subtypes = _fraction_design(GLOBAL_FRACTIONS, fraction_rng)
    return {
        "desired_x0": anchors,
        "desired_x1": np.zeros(GLOBAL_ROWS, dtype=np.float64),
        "fraction": fractions,
        "target_x_inverted": np.zeros(GLOBAL_ROWS, dtype=np.bool_),
        "sample_group": np.full(GLOBAL_ROWS, "global64", dtype="U24"),
        "sample_subtype": subtypes,
        "anchor_bin": np.arange(GLOBAL_ROWS, dtype=np.int16),
        "target_bin": np.full(GLOBAL_ROWS, -1, dtype=np.int16),
    }


def hard_additional_design(split: str, orbit_id: str) -> dict[str, np.ndarray]:
    anchor_rng = np.random.default_rng(seed_for(split, orbit_id, "hard32_anchor"))
    anchors = _stratified_interior(
        SENSITIVE_X_LOW, SENSITIVE_X_HIGH, 8, anchor_rng
    )
    target_values: dict[str, np.ndarray] = {}
    target_bins: dict[str, np.ndarray] = {}

    within_rng = np.random.default_rng(seed_for(split, orbit_id, "hard32_within"))
    progress = _stratified_interior(0.0, 1.0, 8, within_rng)
    target_values["within_sensitive"] = anchors + progress * (
        SENSITIVE_X_HIGH - anchors
    )
    target_bins["within_sensitive"] = np.arange(8, dtype=np.int16)

    for label, low, high, salt in (
        ("sensitive_exit", -8.5, -6.0, "hard32_exit"),
        ("sensitive_to_central", -4.0, 2.0, "hard32_central"),
    ):
        rng = np.random.default_rng(seed_for(split, orbit_id, salt))
        values = _stratified_interior(low, high, 8, rng)
        permutation = rng.permutation(8)
        target_values[label] = values[permutation]
        target_bins[label] = np.arange(8, dtype=np.int16)[permutation]

    long_rng = np.random.default_rng(seed_for(split, orbit_id, "hard32_long"))
    long_values = _stratified_interior(8.0, 17.0, 8, long_rng)
    long_values[-1] = X_RIGHT
    permutation = long_rng.permutation(8)
    target_values["sensitive_to_long_outgoing"] = long_values[permutation]
    target_bins["sensitive_to_long_outgoing"] = np.arange(8, dtype=np.int16)[
        permutation
    ]

    labels = (
        "within_sensitive",
        "sensitive_exit",
        "sensitive_to_central",
        "sensitive_to_long_outgoing",
    )
    desired_x0 = np.repeat(anchors, 4)
    desired_x1 = np.empty(ADDITIONAL_ROWS, dtype=np.float64)
    subtypes = np.empty(ADDITIONAL_ROWS, dtype="U40")
    target_bin = np.empty(ADDITIONAL_ROWS, dtype=np.int16)
    for anchor_index in range(8):
        for class_index, label in enumerate(labels):
            row = 4 * anchor_index + class_index
            desired_x1[row] = target_values[label][anchor_index]
            subtypes[row] = label
            target_bin[row] = target_bins[label][anchor_index]
    return {
        "desired_x0": desired_x0,
        "desired_x1": desired_x1,
        "fraction": np.full(ADDITIONAL_ROWS, -1.0, dtype=np.float64),
        "target_x_inverted": np.ones(ADDITIONAL_ROWS, dtype=np.bool_),
        "sample_group": np.full(
            ADDITIONAL_ROWS, "hard_targeted32", dtype="U24"
        ),
        "sample_subtype": subtypes,
        "anchor_bin": np.repeat(np.arange(8, dtype=np.int16), 4),
        "target_bin": target_bin,
    }


def ordinary_additional_design(split: str, orbit_id: str) -> dict[str, np.ndarray]:
    anchor_rng = np.random.default_rng(
        seed_for(split, orbit_id, "ordinary32_anchor")
    )
    fraction_rng = np.random.default_rng(
        seed_for(split, orbit_id, "ordinary32_fraction")
    )
    anchors = _stratified_interior(X_LEFT, X_RIGHT, ADDITIONAL_ROWS, anchor_rng)
    fractions, subtypes = _fraction_design(ORDINARY_FRACTIONS, fraction_rng)
    return {
        "desired_x0": anchors,
        "desired_x1": np.zeros(ADDITIONAL_ROWS, dtype=np.float64),
        "fraction": fractions,
        "target_x_inverted": np.zeros(ADDITIONAL_ROWS, dtype=np.bool_),
        "sample_group": np.full(
            ADDITIONAL_ROWS, "ordinary_global32", dtype="U24"
        ),
        "sample_subtype": subtypes,
        "anchor_bin": np.arange(ADDITIONAL_ROWS, dtype=np.int16),
        "target_bin": np.full(ADDITIONAL_ROWS, -1, dtype=np.int16),
    }


def orbit_design(
    split: str, orbit_id: str, u_th: float
) -> dict[str, np.ndarray]:
    global_rows = global_design(split, orbit_id)
    additional = (
        hard_additional_design(split, orbit_id)
        if u_th <= HARD_U_TH_MAX
        else ordinary_additional_design(split, orbit_id)
    )
    return {
        key: np.concatenate((global_rows[key], additional[key]))
        for key in global_rows
    }


def invert_saved_orbit_x(
    bank: dict[str, np.ndarray] | Any,
    orbit_index: int,
    targets: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert monotonic saved ``x(t)`` by bracketed vectorized Newton steps."""

    requested = np.asarray(targets, dtype=np.float64)
    flat = requested.reshape(-1)
    if np.any(flat < X_LEFT) or np.any(flat > X_RIGHT):
        raise ValueError("x inversion target lies outside [-17, 17]")
    segment_end, y_old, coefficients = orbit_dense_arrays(bank, orbit_index)
    segment_start = np.zeros_like(segment_end)
    segment_start[1:] = segment_end[:-1]
    start_x = y_old[:, 0]
    end_x = evaluate_dop853_dense(
        segment_end, y_old, coefficients, segment_end
    )[:, 0]
    if np.any(np.diff(end_x) <= 0.0):
        raise RuntimeError("saved orbit segment endpoints are not monotonic in x")

    indices = np.searchsorted(end_x, flat, side="left")
    indices = np.minimum(indices, end_x.size - 1)
    low_t = segment_start[indices].copy()
    high_t = segment_end[indices].copy()
    low_x = start_x[indices]
    high_x = end_x[indices]
    fraction = (flat - low_x) / (high_x - low_x)
    times = low_t + fraction * (high_t - low_t)
    left = flat == X_LEFT
    right = flat == X_RIGHT
    times[left] = 0.0
    times[right] = float(bank["t_right"][orbit_index])

    interior = ~(left | right)
    for _iteration in range(8):
        state = evaluate_dop853_dense(segment_end, y_old, coefficients, times)
        residual = state[:, 0] - flat
        active = interior & (np.abs(residual) > X_INVERSION_TOLERANCE)
        if not np.any(active):
            break
        above = residual > 0.0
        high_t[active & above] = times[active & above]
        low_t[active & ~above] = times[active & ~above]
        proposed = times - residual / state[:, 1]
        invalid = active & (
            (proposed <= low_t) | (proposed >= high_t) | ~np.isfinite(proposed)
        )
        proposed[invalid] = 0.5 * (low_t[invalid] + high_t[invalid])
        times[active] = proposed[active]
    final_state = evaluate_dop853_dense(segment_end, y_old, coefficients, times)
    residual = final_state[:, 0] - flat
    if np.max(np.abs(residual)) > X_INVERSION_TOLERANCE:
        raise RuntimeError(
            f"x-to-time inversion residual {np.max(np.abs(residual)):.3e} exceeds gate"
        )
    return times.reshape(requested.shape), residual.reshape(requested.shape)


def _empty_dataset(row_count: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "transition_id": np.empty(row_count, dtype="U44"),
        "orbit_id": np.empty(row_count, dtype="U27"),
        "split": np.empty(row_count, dtype="U16"),
        "trajectory_stratum": np.empty(row_count, dtype="U16"),
        "sample_group": np.empty(row_count, dtype="U24"),
        "sample_subtype": np.empty(row_count, dtype="U40"),
        "source_orbit_index": np.empty(row_count, dtype=np.int32),
        "row_within_orbit": np.empty(row_count, dtype=np.int16),
        "anchor_bin": np.empty(row_count, dtype=np.int16),
        "target_bin": np.empty(row_count, dtype=np.int16),
        "hard_orbit": np.empty(row_count, dtype=np.bool_),
        "sensitive_anchor": np.empty(row_count, dtype=np.bool_),
        "target_at_endpoint": np.empty(row_count, dtype=np.bool_),
        "identity_sample": np.empty(row_count, dtype=np.bool_),
        "target_x_inverted": np.empty(row_count, dtype=np.bool_),
    }
    arrays.update(
        {name: np.empty(row_count, dtype=np.float64) for name in FLOAT_COLUMNS}
    )
    return arrays


def _transition_ids(orbit_id: str) -> np.ndarray:
    return np.asarray(
        [
            "phasec-"
            + hashlib.sha256(f"{orbit_id}:{row}".encode("utf-8")).hexdigest()[:36]
            for row in range(ROWS_PER_ORBIT)
        ],
        dtype="U44",
    )


def extract_orbit_rows(
    bank: dict[str, np.ndarray] | Any, orbit_index: int
) -> dict[str, np.ndarray]:
    orbit_id = str(bank["orbit_id"][orbit_index])
    split = str(bank["split"][orbit_index])
    u_th = float(bank["u_th"][orbit_index])
    energy0 = float(bank["E0"][orbit_index])
    t_left = float(bank["t_left"][orbit_index])
    t_right = float(bank["t_right"][orbit_index])
    hard = u_th <= HARD_U_TH_MAX
    design = orbit_design(split, orbit_id, u_th)

    unique_x0, x0_inverse = np.unique(design["desired_x0"], return_inverse=True)
    unique_t0, unique_residual0 = invert_saved_orbit_x(
        bank, orbit_index, unique_x0
    )
    t0 = unique_t0[x0_inverse]
    residual0 = unique_residual0[x0_inverse]
    t1 = np.empty(ROWS_PER_ORBIT, dtype=np.float64)
    residual1 = np.zeros(ROWS_PER_ORBIT, dtype=np.float64)
    target_x_mask = design["target_x_inverted"]
    time_fraction_mask = ~target_x_mask
    fraction = design["fraction"].copy()
    t1[time_fraction_mask] = t0[time_fraction_mask] + fraction[
        time_fraction_mask
    ] * (t_right - t0[time_fraction_mask])
    if np.any(target_x_mask):
        unique_x1, x1_inverse = np.unique(
            design["desired_x1"][target_x_mask], return_inverse=True
        )
        unique_t1, unique_residual1 = invert_saved_orbit_x(
            bank, orbit_index, unique_x1
        )
        t1[target_x_mask] = unique_t1[x1_inverse]
        residual1[target_x_mask] = unique_residual1[x1_inverse]
        fraction[target_x_mask] = (t1[target_x_mask] - t0[target_x_mask]) / (
            t_right - t0[target_x_mask]
        )

    state0 = evaluate_saved_orbit(bank, orbit_index, t0)
    state1 = evaluate_saved_orbit(bank, orbit_index, t1)
    wormhole, spiral = experiment_parameters()
    xi0 = xi_from_state(state0[:, 0], state0[:, 1], wormhole, spiral)
    xi1 = xi_from_state(state1[:, 0], state1[:, 1], wormhole, spiral)
    identity = fraction == 0.0
    if np.any(identity):
        t1[identity] = t0[identity]
        state1[identity] = state0[identity]
        xi1[identity] = xi0[identity]
    elapsed = t1 - t0
    delta_x = state1[:, 0] - state0[:, 0]
    delta_xi = xi1 - xi0
    delta_x[identity] = 0.0
    delta_xi[identity] = 0.0
    c0 = timelike_margin(state0[:, 0], state0[:, 1], wormhole, spiral)
    c1 = timelike_margin(state1[:, 0], state1[:, 1], wormhole, spiral)
    e0 = conserved_energy(state0[:, 0], state0[:, 1], wormhole, spiral)
    e1 = conserved_energy(state1[:, 0], state1[:, 1], wormhole, spiral)
    denominator = abs(energy0) + 1.0e-12
    endpoint = t1 == t_right

    output = _empty_dataset(ROWS_PER_ORBIT)
    output.update(
        {
            "transition_id": _transition_ids(orbit_id),
            "orbit_id": np.full(ROWS_PER_ORBIT, orbit_id, dtype="U27"),
            "split": np.full(ROWS_PER_ORBIT, split, dtype="U16"),
            "trajectory_stratum": np.full(
                ROWS_PER_ORBIT,
                str(bank["trajectory_stratum"][orbit_index]),
                dtype="U16",
            ),
            "sample_group": design["sample_group"],
            "sample_subtype": design["sample_subtype"],
            "source_orbit_index": np.full(
                ROWS_PER_ORBIT, orbit_index, dtype=np.int32
            ),
            "row_within_orbit": np.arange(ROWS_PER_ORBIT, dtype=np.int16),
            "anchor_bin": design["anchor_bin"],
            "target_bin": design["target_bin"],
            "hard_orbit": np.full(ROWS_PER_ORBIT, hard, dtype=np.bool_),
            "sensitive_anchor": np.asarray(
                hard
                & (state0[:, 0] >= SENSITIVE_X_LOW - 1e-12)
                & (state0[:, 0] <= SENSITIVE_X_HIGH + 1e-12),
                dtype=np.bool_,
            ),
            "target_at_endpoint": endpoint,
            "identity_sample": identity,
            "target_x_inverted": target_x_mask,
            "u_th": np.full(ROWS_PER_ORBIT, u_th, dtype=np.float64),
            "E0": np.full(ROWS_PER_ORBIT, energy0, dtype=np.float64),
            "t_left": np.full(ROWS_PER_ORBIT, t_left, dtype=np.float64),
            "t_right": np.full(ROWS_PER_ORBIT, t_right, dtype=np.float64),
            "t0": t0,
            "t1": t1,
            "s": elapsed,
            "x0": state0[:, 0],
            "u0": state0[:, 1],
            "xi0": np.asarray(xi0),
            "x1": state1[:, 0],
            "u1": state1[:, 1],
            "xi1": np.asarray(xi1),
            "Delta_x": delta_x,
            "Delta_xi": delta_xi,
            "C0": np.asarray(c0),
            "C1": np.asarray(c1),
            "E_state_0": np.asarray(e0),
            "E_state_1": np.asarray(e1),
            "relative_energy_error_0": np.abs(e0 - energy0) / denominator,
            "relative_energy_error_1": np.abs(e1 - energy0) / denominator,
            "horizon_fraction": fraction,
            "x0_inversion_residual": residual0,
            "x1_inversion_residual": residual1,
        }
    )
    return output


def extract_split(bank: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    orbit_count = int(bank["orbit_id"].size)
    dataset = _empty_dataset(orbit_count * ROWS_PER_ORBIT)
    for orbit_index in range(orbit_count):
        rows = extract_orbit_rows(bank, orbit_index)
        destination = slice(
            orbit_index * ROWS_PER_ORBIT, (orbit_index + 1) * ROWS_PER_ORBIT
        )
        for name in dataset:
            dataset[name][destination] = rows[name]
    return dataset


def _pair_duplicate_audit(dataset: dict[str, np.ndarray]) -> dict[str, Any]:
    exact_duplicates = 0
    near_duplicates = 0
    minimum_pair_separation = np.inf
    orbit_count = dataset["orbit_id"].size // ROWS_PER_ORBIT
    for orbit_index in range(orbit_count):
        start = orbit_index * ROWS_PER_ORBIT
        stop = start + ROWS_PER_ORBIT
        pair = np.column_stack((dataset["t0"][start:stop], dataset["t1"][start:stop]))
        exact_duplicates += ROWS_PER_ORBIT - np.unique(pair, axis=0).shape[0]
        difference = np.max(np.abs(pair[:, None, :] - pair[None, :, :]), axis=2)
        difference[np.tril_indices(ROWS_PER_ORBIT)] = np.inf
        local_minimum = float(np.min(difference))
        minimum_pair_separation = min(minimum_pair_separation, local_minimum)
        near_duplicates += int(
            np.sum(difference <= NEAR_DUPLICATE_TIME_TOLERANCE)
        )
    return {
        "exact_orbit_t0_t1_duplicate_count": exact_duplicates,
        "near_duplicate_threshold": NEAR_DUPLICATE_TIME_TOLERANCE,
        "near_duplicate_pair_count": near_duplicates,
        "minimum_within_orbit_max_time_coordinate_separation": minimum_pair_separation,
    }


def validate_dataset(
    dataset: dict[str, np.ndarray], bank: dict[str, np.ndarray]
) -> dict[str, Any]:
    row_count = dataset["orbit_id"].size
    orbit_count = bank["orbit_id"].size
    expected_rows = orbit_count * ROWS_PER_ORBIT
    failures: list[str] = []
    if row_count != expected_rows:
        failures.append("row_count")
    parent_ids = set(map(str, bank["orbit_id"]))
    row_ids = set(map(str, dataset["orbit_id"]))
    if row_ids != parent_ids:
        failures.append("parent_orbit_ids")
    counts = Counter(map(str, dataset["orbit_id"]))
    if set(counts.values()) != {ROWS_PER_ORBIT}:
        failures.append("rows_per_orbit")
    if len(set(map(str, dataset["transition_id"]))) != row_count:
        failures.append("transition_id_uniqueness")
    numeric_finite = all(
        np.all(np.isfinite(dataset[name])) for name in FLOAT_COLUMNS
    )
    if not numeric_finite:
        failures.append("nonfinite_numeric_value")
    invalid_time = (
        (dataset["t0"] < dataset["t_left"] - TIME_TOLERANCE)
        | (dataset["t1"] < dataset["t0"] - TIME_TOLERANCE)
        | (dataset["t1"] > dataset["t_right"] + TIME_TOLERANCE)
        | (dataset["s"] < -TIME_TOLERANCE)
    )
    if np.any(invalid_time):
        failures.append("invalid_time_order")
    if not np.array_equal(dataset["s"], dataset["t1"] - dataset["t0"]):
        failures.append("elapsed_time_identity")
    if not np.array_equal(dataset["Delta_x"], dataset["x1"] - dataset["x0"]):
        failures.append("delta_x_identity")
    if not np.array_equal(dataset["Delta_xi"], dataset["xi1"] - dataset["xi0"]):
        failures.append("delta_xi_identity")
    if np.any(dataset["x1"] < dataset["x0"] - X_INVERSION_TOLERANCE):
        failures.append("nonmonotonic_transition")
    if np.any(dataset["C0"] <= 0.0) or np.any(dataset["C1"] <= 0.0):
        failures.append("nonpositive_C")
    if np.any(np.abs(dataset["xi0"]) >= 1.0) or np.any(np.abs(dataset["xi1"]) >= 1.0):
        failures.append("invalid_xi")
    maximum_energy_error = float(
        max(
            np.max(dataset["relative_energy_error_0"]),
            np.max(dataset["relative_energy_error_1"]),
        )
    )
    if maximum_energy_error > ENERGY_TOLERANCE:
        failures.append("energy_consistency")
    maximum_inversion_residual = float(
        max(
            np.max(np.abs(dataset["x0_inversion_residual"])),
            np.max(np.abs(dataset["x1_inversion_residual"])),
        )
    )
    if maximum_inversion_residual > X_INVERSION_TOLERANCE:
        failures.append("x_to_time_inversion")

    identity = dataset["identity_sample"]
    identity_ok = bool(
        np.all(dataset["s"][identity] == 0.0)
        and np.array_equal(dataset["x1"][identity], dataset["x0"][identity])
        and np.array_equal(dataset["xi1"][identity], dataset["xi0"][identity])
        and np.all(dataset["Delta_x"][identity] == 0.0)
        and np.all(dataset["Delta_xi"][identity] == 0.0)
    )
    if not identity_ok:
        failures.append("identity_rows")
    endpoint = dataset["target_at_endpoint"]
    endpoint_ok = bool(
        np.array_equal(dataset["t1"][endpoint], dataset["t_right"][endpoint])
        and np.max(np.abs(dataset["x1"][endpoint] - X_RIGHT), initial=0.0)
        <= ENDPOINT_X_TOLERANCE
    )
    if not endpoint_ok:
        failures.append("endpoint_rows")

    global_counts = np.bincount(
        dataset["source_orbit_index"][dataset["sample_group"] == "global64"],
        minlength=orbit_count,
    )
    additional_counts = np.bincount(
        dataset["source_orbit_index"][dataset["sample_group"] != "global64"],
        minlength=orbit_count,
    )
    if not np.all(global_counts == GLOBAL_ROWS):
        failures.append("global64_count")
    if not np.all(additional_counts == ADDITIONAL_ROWS):
        failures.append("additional32_count")

    hard_mask = dataset["hard_orbit"]
    if np.any(dataset["u_th"][hard_mask] > HARD_U_TH_MAX) or np.any(
        dataset["u_th"][~hard_mask] <= HARD_U_TH_MAX
    ):
        failures.append("hard_orbit_definition")
    hard_additional = dataset["sample_group"] == "hard_targeted32"
    ordinary_additional = dataset["sample_group"] == "ordinary_global32"
    if np.any(~hard_mask[hard_additional]) or np.any(hard_mask[ordinary_additional]):
        failures.append("additional_group_assignment")
    subtype_counts = Counter(map(str, dataset["sample_subtype"][hard_additional]))
    hard_orbit_count = int(np.sum(np.asarray(bank["u_th"]) <= HARD_U_TH_MAX))
    ordinary_orbit_count = orbit_count - hard_orbit_count
    for subtype in (
        "within_sensitive",
        "sensitive_exit",
        "sensitive_to_central",
        "sensitive_to_long_outgoing",
    ):
        if subtype_counts[subtype] != 8 * hard_orbit_count:
            failures.append(f"hard_subtype_count:{subtype}")
    if np.any(~dataset["sensitive_anchor"][hard_additional]):
        failures.append("hard_sensitive_anchor")
    if np.any(dataset["s"][hard_additional] <= 0.0):
        failures.append("hard_positive_horizon")
    range_rules = {
        "within_sensitive": (
            dataset["x1"] > dataset["x0"],
            dataset["x1"] <= SENSITIVE_X_HIGH + X_INVERSION_TOLERANCE,
        ),
        "sensitive_exit": (
            dataset["x1"] > SENSITIVE_X_HIGH,
            dataset["x1"] <= -6.0 + X_INVERSION_TOLERANCE,
        ),
        "sensitive_to_central": (
            dataset["x1"] >= -4.0 - X_INVERSION_TOLERANCE,
            dataset["x1"] <= 2.0 + X_INVERSION_TOLERANCE,
        ),
        "sensitive_to_long_outgoing": (
            dataset["x1"] >= 8.0 - X_INVERSION_TOLERANCE,
            dataset["x1"] <= X_RIGHT + ENDPOINT_X_TOLERANCE,
        ),
    }
    for subtype, conditions in range_rules.items():
        mask = dataset["sample_subtype"] == subtype
        if not all(np.all(condition[mask]) for condition in conditions):
            failures.append(f"hard_target_range:{subtype}")
    central = dataset["sample_subtype"] == "sensitive_to_central"
    long_outgoing = dataset["sample_subtype"] == "sensitive_to_long_outgoing"
    for orbit_index in range(orbit_count):
        orbit_mask = dataset["source_orbit_index"] == orbit_index
        if not np.any(hard_mask & orbit_mask):
            continue
        central_targets = dataset["x1"][central & orbit_mask]
        if not (np.any(central_targets < 0.0) and np.any(central_targets > 0.0)):
            failures.append("hard_central_both_sides")
            break
        if np.sum(long_outgoing & orbit_mask & endpoint) != 1:
            failures.append("hard_long_exact_endpoint_count")
            break

    all_subtypes = Counter(map(str, dataset["sample_subtype"]))
    expected_global = {
        "global_identity": 4,
        "global_f_0_0p05": 8,
        "global_f_0p05_0p25": 12,
        "global_f_0p25_0p50": 12,
        "global_f_0p50_0p80": 12,
        "global_f_0p80_0p95": 8,
        "global_f_0p95_1": 4,
        "global_endpoint": 4,
    }
    expected_ordinary = {
        "additional_f_0_0p05": 4,
        "additional_f_0p05_0p25": 8,
        "additional_f_0p25_0p50": 8,
        "additional_f_0p50_0p80": 6,
        "additional_f_0p80_1": 4,
        "additional_endpoint": 2,
    }
    for subtype, per_orbit in expected_global.items():
        if all_subtypes[subtype] != per_orbit * orbit_count:
            failures.append(f"global_fraction_composition:{subtype}")
    for subtype, per_orbit in expected_ordinary.items():
        if all_subtypes[subtype] != per_orbit * ordinary_orbit_count:
            failures.append(f"ordinary_fraction_composition:{subtype}")

    duplicate = _pair_duplicate_audit(dataset)
    if duplicate["exact_orbit_t0_t1_duplicate_count"]:
        failures.append("exact_duplicate_transition")
    if duplicate["near_duplicate_pair_count"]:
        failures.append("near_duplicate_transition")
    result = {
        "passed": not failures,
        "failures": failures,
        "row_count": row_count,
        "orbit_count": orbit_count,
        "rows_per_orbit_minimum": min(counts.values()),
        "rows_per_orbit_maximum": max(counts.values()),
        "global_row_count": int(np.sum(dataset["sample_group"] == "global64")),
        "hard_targeted_row_count": int(np.sum(hard_additional)),
        "ordinary_additional_row_count": int(np.sum(ordinary_additional)),
        "hard_orbit_count": hard_orbit_count,
        "hard_orbit_row_count": int(np.sum(hard_mask)),
        "ordinary_orbit_row_count": int(np.sum(~hard_mask)),
        "identity_row_count": int(np.sum(identity)),
        "endpoint_row_count": int(np.sum(endpoint)),
        "minimum_C": float(min(np.min(dataset["C0"]), np.min(dataset["C1"]))),
        "minimum_one_minus_abs_xi": float(
            min(np.min(1.0 - np.abs(dataset["xi0"])), np.min(1.0 - np.abs(dataset["xi1"])))
        ),
        "maximum_relative_energy_error": maximum_energy_error,
        "maximum_x_inversion_residual": maximum_inversion_residual,
        "nan_inf_count": int(
            sum(np.size(dataset[name]) - np.count_nonzero(np.isfinite(dataset[name])) for name in FLOAT_COLUMNS)
        ),
        "invalid_time_count": int(np.sum(invalid_time)),
        "endpoint_order_failure_count": int(
            np.sum(dataset["t1"] > dataset["t_right"] + TIME_TOLERANCE)
        ),
        "identity_rows_valid": identity_ok,
        "endpoint_rows_valid": endpoint_ok,
        "sample_group_counts": {
            str(name): int(count)
            for name, count in zip(*np.unique(dataset["sample_group"], return_counts=True))
        },
        "sample_subtype_counts": {
            str(name): int(count)
            for name, count in zip(*np.unique(dataset["sample_subtype"], return_counts=True))
        },
        "trajectory_stratum_counts": {
            str(name): int(count)
            for name, count in zip(*np.unique(dataset["trajectory_stratum"], return_counts=True))
        },
        **duplicate,
    }
    if failures:
        raise RuntimeError("Phase-C dataset validation failed: " + ", ".join(failures))
    return result


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_dataset(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "row_count": int(arrays["orbit_id"].size),
        "orbit_count": int(np.unique(arrays["orbit_id"]).size),
        "bytes": path.stat().st_size,
        "content_sha256": array_content_sha256(arrays),
        "file_sha256": file_sha256(path),
        "schema": {name: str(array.dtype) for name, array in arrays.items()},
    }


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}
