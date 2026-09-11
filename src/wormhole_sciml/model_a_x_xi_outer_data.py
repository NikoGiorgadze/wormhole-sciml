"""Deterministic random data for the transformed-coordinate outer-x study."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .dynamics import conserved_energy, timelike_margin
from .integrate import integrate_trajectory
from .physics_gate import PRODUCTION_SOLVER, experiment_parameters, state_from_xi, xi_from_state
from .stage1_data import H, STRATUM_BOUNDS


X_COMPONENTS = (
    ("central", -8.5, 8.5),
    ("broad", -17.0, 17.0),
    ("outer_left", -17.0, -8.5),
    ("outer_right", 8.5, 17.0),
)
STRATUM_FRACTIONS = (0.30, 0.35, 0.35)
TRAIN_SEED = 2_026_082_201
VALIDATION_SEED = 2_026_082_202


def sample_outer_enriched_states(count: int, seed: int) -> dict[str, np.ndarray]:
    """Sample the exact four-component x mixture and xi quotas."""

    if count % 4:
        raise ValueError("row count must be divisible by four")
    per_component = count // 4
    stratum_counts = [int(per_component * fraction) for fraction in STRATUM_FRACTIONS]
    if sum(stratum_counts) != per_component or any(value % 2 for value in stratum_counts):
        raise ValueError("component size does not permit exact stratum/sign quotas")
    rng = np.random.default_rng(seed)
    pieces: dict[str, list[np.ndarray]] = {
        "x": [], "xi": [], "x_component": [], "xi_sign": [], "stratum": [],
    }
    for component_id, (_name, low_x, high_x) in enumerate(X_COMPONENTS):
        for stratum_id, ((low_xi, high_xi), stratum_count) in enumerate(
            zip(STRATUM_BOUNDS, stratum_counts)
        ):
            x = rng.uniform(low_x, high_x, stratum_count).astype(np.float64)
            magnitude = rng.uniform(low_xi, high_xi, stratum_count).astype(np.float64)
            signs = np.concatenate((
                -np.ones(stratum_count // 2, dtype=np.int8),
                np.ones(stratum_count // 2, dtype=np.int8),
            ))
            rng.shuffle(signs)
            pieces["x"].append(x)
            pieces["xi"].append(magnitude * signs)
            pieces["x_component"].append(np.full(stratum_count, component_id, dtype=np.int8))
            pieces["xi_sign"].append(signs)
            pieces["stratum"].append(np.full(stratum_count, stratum_id, dtype=np.uint8))
    sampled = {name: np.concatenate(parts) for name, parts in pieces.items()}
    permutation = rng.permutation(count)
    sampled = {name: values[permutation] for name, values in sampled.items()}
    wormhole, spiral = experiment_parameters()
    _, u = state_from_xi(sampled["x"], sampled["xi"], wormhole, spiral)
    sampled.update({
        "u": np.asarray(u, dtype=np.float64),
        "is_physical": np.ones(count, dtype=np.bool_),
        "source_row_index": np.arange(count, dtype=np.int64),
    })
    return sampled


def integrate_transformed_targets(
    sampled: dict[str, np.ndarray],
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Integrate each physical state for h=0.2 and return transformed targets."""

    count = sampled["x"].size
    next_state = np.empty((count, 2), dtype=np.float64)
    wormhole, spiral = experiment_parameters()
    kwargs = PRODUCTION_SOLVER.kwargs()
    kwargs["stop_at_null_boundary"] = True
    for index, (x, u) in enumerate(zip(sampled["x"], sampled["u"])):
        solution = integrate_trajectory(
            (float(x), float(u)), (0.0, H), wormhole, spiral, t_eval=(H,), **kwargs
        )
        if solution.y.shape != (2, 1) or float(solution.t[-1]) != H:
            raise RuntimeError(f"row {index} did not reach h=0.2")
        next_state[index] = solution.y[:, -1]
        if progress is not None and (index + 1) % 5_000 == 0:
            progress(f"integrated {index + 1}/{count}")
    next_x, next_u = next_state.T
    next_xi = xi_from_state(next_x, next_u, wormhole, spiral)
    energy = conserved_energy(sampled["x"], sampled["u"], wormhole, spiral)
    return {
        "x": np.asarray(sampled["x"], dtype=np.float64),
        "xi": np.asarray(sampled["xi"], dtype=np.float64),
        "E0": np.asarray(energy, dtype=np.float64),
        "delta_x": np.asarray(next_x - sampled["x"], dtype=np.float64),
        "delta_xi": np.asarray(next_xi - sampled["xi"], dtype=np.float64),
        "u": np.asarray(sampled["u"], dtype=np.float64),
        "x_next": np.asarray(next_x, dtype=np.float64),
        "u_next": np.asarray(next_u, dtype=np.float64),
        "xi_next": np.asarray(next_xi, dtype=np.float64),
        "x_component": np.asarray(sampled["x_component"]),
        "xi_sign": np.asarray(sampled["xi_sign"]),
        "stratum": np.asarray(sampled["stratum"]),
        "is_physical": np.asarray(sampled["is_physical"]),
        "source_row_index": np.asarray(sampled["source_row_index"]),
    }


def transform_old_control(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform existing state pairs without performing new integration."""

    wormhole, spiral = experiment_parameters()
    x = np.asarray(source["x"], dtype=np.float64)
    u = np.asarray(source["u"], dtype=np.float64)
    x_next = x + np.asarray(source["delta_x"], dtype=np.float64)
    u_next = u + np.asarray(source["delta_u"], dtype=np.float64)
    xi = xi_from_state(x, u, wormhole, spiral)
    xi_next = xi_from_state(x_next, u_next, wormhole, spiral)
    count = x.size
    return {
        "x": x,
        "xi": np.asarray(xi, dtype=np.float64),
        "E0": np.asarray(conserved_energy(x, u, wormhole, spiral), dtype=np.float64),
        "delta_x": np.asarray(source["delta_x"], dtype=np.float64),
        "delta_xi": np.asarray(xi_next - xi, dtype=np.float64),
        "u": u,
        "x_next": x_next,
        "u_next": u_next,
        "xi_next": np.asarray(xi_next, dtype=np.float64),
        "x_component": np.asarray(source["x_component"]),
        "xi_sign": np.asarray(source["xi_sign"]),
        "stratum": np.asarray(source["stratum"]),
        "is_physical": np.asarray(source["is_physical"]),
        "source_row_index": np.arange(count, dtype=np.int64),
    }


def integrity_statistics(data: dict[str, np.ndarray], energy_relative_tolerance: float = 1e-9) -> dict:
    """Apply the established physical and energy gates to a transformed dataset."""

    wormhole, spiral = experiment_parameters()
    _, u_reconstructed = state_from_xi(data["x"], data["xi"], wormhole, spiral)
    _, next_u_reconstructed = state_from_xi(
        data["x_next"], data["xi_next"], wormhole, spiral
    )
    initial_margin = timelike_margin(data["x"], data["u"], wormhole, spiral)
    next_margin = timelike_margin(data["x_next"], data["u_next"], wormhole, spiral)
    next_energy = conserved_energy(data["x_next"], data["u_next"], wormhole, spiral)
    energy_absolute = np.abs(next_energy - data["E0"])
    energy_relative = energy_absolute / np.maximum(np.abs(data["E0"]), 1e-12)
    reconstruction = np.maximum(
        np.abs(u_reconstructed - data["u"]),
        np.abs(next_u_reconstructed - data["u_next"]),
    )
    numeric = ("x", "xi", "E0", "delta_x", "delta_xi", "u", "x_next", "u_next", "xi_next")
    passed = (
        all(np.all(np.isfinite(data[name])) for name in numeric)
        and np.all(initial_margin > 0.0) and np.all(next_margin > 0.0)
        and np.all(np.abs(data["xi"]) <= 0.99)
        and float(np.max(energy_relative)) <= energy_relative_tolerance
    )
    if not passed:
        raise RuntimeError("transformed dataset failed the established exact-target gate")
    def distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "maximum": float(np.max(values)), "median": float(np.median(values)),
            "p99": float(np.quantile(values, 0.99)),
        }
    return {
        "passed": True,
        "all_initial_states_physical": True,
        "all_next_states_physical": True,
        "minimum_initial_margin": float(np.min(initial_margin)),
        "minimum_next_margin": float(np.min(next_margin)),
        "all_numeric_arrays_finite": True,
        "initial_xi_within_prescribed_range": True,
        "reconstruction_absolute_error": distribution(reconstruction),
        "energy_absolute_mismatch": distribution(energy_absolute),
        "energy_relative_mismatch": {**distribution(energy_relative), "tolerance": energy_relative_tolerance},
    }
