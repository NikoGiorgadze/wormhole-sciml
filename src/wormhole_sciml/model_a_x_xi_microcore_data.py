"""Random outer micro-core sampling for the transformed-coordinate study."""

from __future__ import annotations

import numpy as np

from .model_a_x_xi_outer_data import X_COMPONENTS, integrate_transformed_targets
from .physics_gate import experiment_parameters, state_from_xi


TRAIN_SEED = 2_026_082_203
VALIDATION_SEED = 2_026_082_204
OUTER_STRATUM_NAMES = ("micro_core", "remainder_core", "shoulder", "edge")
OUTER_STRATUM_BOUNDS = ((0.0, 0.05), (0.05, 0.5), (0.5, 0.9), (0.9, 0.99))
OUTER_STRATUM_FRACTIONS = (0.20, 0.20, 0.30, 0.30)
STANDARD_STRATUM_BOUNDS = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99))
STANDARD_STRATUM_FRACTIONS = (0.30, 0.35, 0.35)


def sample_outer_microcore_states(count: int, seed: int) -> dict[str, np.ndarray]:
    """Sample one independent four-component random design with outer micro-cores."""

    if count % 4:
        raise ValueError("row count must be divisible by four")
    per_component = count // 4
    rng = np.random.default_rng(seed)
    pieces: dict[str, list[np.ndarray]] = {
        "x": [], "xi": [], "x_component": [], "xi_sign": [], "stratum": [],
        "outer_xi_stratum": [],
    }
    for component, (_name, low_x, high_x) in enumerate(X_COMPONENTS):
        outer = component in (2, 3)
        bounds = OUTER_STRATUM_BOUNDS if outer else STANDARD_STRATUM_BOUNDS
        fractions = OUTER_STRATUM_FRACTIONS if outer else STANDARD_STRATUM_FRACTIONS
        counts = [int(per_component * fraction) for fraction in fractions]
        if sum(counts) != per_component or any(value % 2 for value in counts):
            raise ValueError("component size does not permit exact quotas and sign balance")
        for local_label, ((low_xi, high_xi), size) in enumerate(zip(bounds, counts)):
            x = rng.uniform(low_x, high_x, size).astype(np.float64)
            magnitude = rng.uniform(low_xi, high_xi, size).astype(np.float64)
            signs = np.concatenate((
                -np.ones(size // 2, dtype=np.int8),
                np.ones(size // 2, dtype=np.int8),
            ))
            rng.shuffle(signs)
            conventional_label = (0, 0, 1, 2)[local_label] if outer else local_label
            pieces["x"].append(x)
            pieces["xi"].append(magnitude * signs)
            pieces["x_component"].append(np.full(size, component, dtype=np.int8))
            pieces["xi_sign"].append(signs)
            pieces["stratum"].append(np.full(size, conventional_label, dtype=np.uint8))
            pieces["outer_xi_stratum"].append(
                np.full(size, local_label if outer else 255, dtype=np.uint8)
            )
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


def generate_outer_microcore_targets(
    count: int, seed: int, progress=print,
) -> dict[str, np.ndarray]:
    """Sample and integrate the transformed h=0.2 targets once."""

    sampled = sample_outer_microcore_states(count, seed)
    transformed = integrate_transformed_targets(sampled, progress)
    transformed["outer_xi_stratum"] = sampled["outer_xi_stratum"]
    return transformed
