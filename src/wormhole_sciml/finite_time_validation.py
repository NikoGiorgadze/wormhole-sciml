"""Validation-only diagnostics for the direct finite-time flow model.

All exact targets are queried from frozen Phase-B dense-output coefficients.
No ODE integration, training, preprocessing fit, or sealed-test prediction is
performed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .dynamics import conserved_energy, timelike_margin
from .finite_time import (
    FiniteTimePreprocessing,
    INPUT_COLUMNS,
    stack_columns,
)
from .model_a import ModelA, Normalization, load_trained_model, predict_increments, predict_standardized
from .phase_b_orbits import evaluate_saved_orbit_x_u_xi
from .phase_c_finite_time import invert_saved_orbit_x
from .physics_gate import experiment_parameters, state_from_xi


DENSE_FRACTIONS = np.asarray(
    [0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40,
     0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00],
    dtype=np.float64,
)
ANCHOR_BIN_COUNT = 32
ANCHOR_WIDTH = 34.0 / ANCHOR_BIN_COUNT
DENSE_ANCHOR_X = -17.0 + (np.arange(ANCHOR_BIN_COUNT) + 0.5) * ANCHOR_WIDTH
SMALL_S_VALUES = np.asarray([0.0, 1.0e-3, 1.0e-2, 0.05, 0.10, 0.20, 0.50, 1.00])
SMALL_S_ANCHOR_X = np.asarray([-15.0, -12.0, -9.0, -4.0, 0.0, 4.0, 10.0, 15.0])
COMPOSITION_ANCHOR_X = SMALL_S_ANCHOR_X
COMPOSITION_FRACTION_PAIRS = np.asarray(
    [(0.10, 0.10), (0.20, 0.20), (0.20, 0.40), (0.40, 0.40)],
    dtype=np.float64,
)
LOCAL_STEP = 0.2


def _allocate_query_rows(count: int) -> dict[str, np.ndarray]:
    return {
        "query_index": np.arange(count, dtype=np.int64),
        "orbit_index": np.empty(count, dtype=np.int32),
        "orbit_id": np.empty(count, dtype="U27"),
        "trajectory_stratum": np.empty(count, dtype="U16"),
        "anchor_index": np.empty(count, dtype=np.int16),
        "fraction_index": np.empty(count, dtype=np.int16),
        **{name: np.empty(count, dtype=np.float64) for name in (
            "u_th", "E0", "t0", "t1", "f", "s", "s_remaining",
            "x0", "u0", "xi0", "exact_x1", "exact_u1", "exact_xi1",
        )},
    }


def build_dense_queries(bank: Any) -> dict[str, np.ndarray]:
    """Construct exactly 32 x-centered anchors times 16 fractions per orbit."""

    orbit_count = int(bank["orbit_id"].size)
    rows_per_orbit = ANCHOR_BIN_COUNT * DENSE_FRACTIONS.size
    output = _allocate_query_rows(orbit_count * rows_per_orbit)
    for orbit_index in range(orbit_count):
        start = orbit_index * rows_per_orbit
        stop = start + rows_per_orbit
        t0, residual = invert_saved_orbit_x(bank, orbit_index, DENSE_ANCHOR_X)
        if float(np.max(np.abs(residual))) > 5.0e-12:
            raise RuntimeError("dense anchor inversion residual exceeded frozen gate")
        state0 = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t0)
        remaining = float(bank["t_right"][orbit_index]) - t0
        t1_matrix = t0[:, None] + remaining[:, None] * DENSE_FRACTIONS[None, :]
        state1 = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t1_matrix.reshape(-1))
        anchor_index = np.repeat(np.arange(ANCHOR_BIN_COUNT, dtype=np.int16), DENSE_FRACTIONS.size)
        fraction_index = np.tile(np.arange(DENSE_FRACTIONS.size, dtype=np.int16), ANCHOR_BIN_COUNT)
        output["orbit_index"][start:stop] = orbit_index
        output["orbit_id"][start:stop] = str(bank["orbit_id"][orbit_index])
        output["trajectory_stratum"][start:stop] = str(bank["trajectory_stratum"][orbit_index])
        output["anchor_index"][start:stop] = anchor_index
        output["fraction_index"][start:stop] = fraction_index
        output["u_th"][start:stop] = float(bank["u_th"][orbit_index])
        output["E0"][start:stop] = float(bank["E0"][orbit_index])
        output["t0"][start:stop] = np.repeat(t0, DENSE_FRACTIONS.size)
        output["t1"][start:stop] = t1_matrix.reshape(-1)
        output["f"][start:stop] = np.tile(DENSE_FRACTIONS, ANCHOR_BIN_COUNT)
        output["s_remaining"][start:stop] = np.repeat(remaining, DENSE_FRACTIONS.size)
        output["s"][start:stop] = (t1_matrix - t0[:, None]).reshape(-1)
        output["x0"][start:stop] = np.repeat(state0[:, 0], DENSE_FRACTIONS.size)
        output["u0"][start:stop] = np.repeat(state0[:, 1], DENSE_FRACTIONS.size)
        output["xi0"][start:stop] = np.repeat(state0[:, 2], DENSE_FRACTIONS.size)
        output["exact_x1"][start:stop] = state1[:, 0]
        output["exact_u1"][start:stop] = state1[:, 1]
        output["exact_xi1"][start:stop] = state1[:, 2]
    if output["query_index"].size != orbit_count * 512:
        raise RuntimeError("dense query count mismatch")
    return output


def build_queries_at_x_and_s(
    bank: Any,
    anchor_x: np.ndarray,
    elapsed_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct exact-anchor queries at fixed physical elapsed times when valid."""

    rows: dict[str, list[Any]] = {
        name: [] for name in (
            "orbit_index", "orbit_id", "trajectory_stratum", "anchor_index",
            "u_th", "E0", "t0", "t1", "s", "x0", "u0", "xi0",
            "exact_x1", "exact_u1", "exact_xi1",
        )
    }
    for orbit_index in range(int(bank["orbit_id"].size)):
        t0, _ = invert_saved_orbit_x(bank, orbit_index, anchor_x)
        state0 = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t0)
        for anchor_index, (anchor_time, anchor_state) in enumerate(zip(t0, state0)):
            valid = elapsed_values <= float(bank["t_right"][orbit_index]) - anchor_time + 1.0e-13
            elapsed = elapsed_values[valid]
            target_time = anchor_time + elapsed
            state1 = evaluate_saved_orbit_x_u_xi(bank, orbit_index, target_time)
            count = elapsed.size
            for key, value in (
                ("orbit_index", orbit_index), ("orbit_id", str(bank["orbit_id"][orbit_index])),
                ("trajectory_stratum", str(bank["trajectory_stratum"][orbit_index])),
                ("anchor_index", anchor_index), ("u_th", float(bank["u_th"][orbit_index])),
                ("E0", float(bank["E0"][orbit_index])), ("t0", float(anchor_time)),
                ("x0", float(anchor_state[0])), ("u0", float(anchor_state[1])),
                ("xi0", float(anchor_state[2])),
            ):
                rows[key].extend([value] * count)
            rows["t1"].extend(target_time.tolist())
            rows["s"].extend(elapsed.tolist())
            rows["exact_x1"].extend(state1[:, 0].tolist())
            rows["exact_u1"].extend(state1[:, 1].tolist())
            rows["exact_xi1"].extend(state1[:, 2].tolist())
    count = len(rows["s"])
    output: dict[str, np.ndarray] = {"query_index": np.arange(count, dtype=np.int64)}
    integer = {"orbit_index": np.int32, "anchor_index": np.int16}
    string = {"orbit_id": "U27", "trajectory_stratum": "U16"}
    for name, values in rows.items():
        output[name] = np.asarray(values, dtype=integer.get(name, string.get(name, np.float64)))
    return output


@torch.no_grad()
def predict_finite_time(
    model: ModelA,
    preprocessing: FiniteTimePreprocessing,
    queries: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    features = np.column_stack(tuple(queries[name] for name in INPUT_COLUMNS))
    standardized = torch.from_numpy(preprocessing.standardize_inputs(features))
    predicted_standardized = predict_standardized(model, standardized, batch_size=16384).cpu().numpy()
    increments = preprocessing.unstandardize_targets(predicted_standardized)
    predicted_x = queries["x0"] + increments[:, 0]
    predicted_xi = queries["xi0"] + increments[:, 1]
    return prediction_diagnostics(queries, increments, predicted_x, predicted_xi)


def prediction_diagnostics(
    queries: dict[str, np.ndarray],
    increments: np.ndarray,
    predicted_x: np.ndarray,
    predicted_xi: np.ndarray,
) -> dict[str, np.ndarray]:
    wormhole, spiral = experiment_parameters()
    _, predicted_u = state_from_xi(predicted_x, predicted_xi, wormhole, spiral)
    predicted_c = timelike_margin(predicted_x, predicted_u, wormhole, spiral)
    exact_c = timelike_margin(queries["exact_x1"], queries["exact_u1"], wormhole, spiral)
    energy = np.full(predicted_x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(predicted_c) & (predicted_c > 0.0)
    if np.any(valid):
        energy[valid] = conserved_energy(
            predicted_x[valid], predicted_u[valid], wormhole, spiral
        )
    x_error = predicted_x - queries["exact_x1"]
    xi_error = predicted_xi - queries["exact_xi1"]
    exact_delta = np.column_stack((
        queries["exact_x1"] - queries["x0"],
        queries["exact_xi1"] - queries["xi0"],
    ))
    residual_error = increments - exact_delta
    residual_state_difference = np.column_stack((x_error, xi_error)) - residual_error
    return {
        "predicted_Delta_x": increments[:, 0],
        "predicted_Delta_xi": increments[:, 1],
        "predicted_x1": np.asarray(predicted_x, dtype=np.float64),
        "predicted_xi1": np.asarray(predicted_xi, dtype=np.float64),
        "predicted_u1": np.asarray(predicted_u, dtype=np.float64),
        "predicted_C1": np.asarray(predicted_c, dtype=np.float64),
        "exact_C1": np.asarray(exact_c, dtype=np.float64),
        "predicted_E1": energy,
        "energy_error": energy - queries["E0"],
        "x_error": x_error,
        "xi_error": xi_error,
        "residual_state_error_max_difference": np.asarray(
            [np.max(np.abs(residual_state_difference))], dtype=np.float64
        ),
    }


@torch.no_grad()
def predict_local(
    model: ModelA,
    normalization: Normalization,
    queries: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    features = np.column_stack((queries["x0"], queries["xi0"], queries["E0"]))
    increments = predict_increments(model, features, normalization)
    return prediction_diagnostics(
        queries, increments, queries["x0"] + increments[:, 0], queries["xi0"] + increments[:, 1]
    )


def scalar_error_metrics(error: np.ndarray) -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    q = np.quantile(absolute, (0.50, 0.90, 0.95, 0.99, 0.999))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "median_absolute": float(q[0]),
        "p90_absolute": float(q[1]),
        "p95_absolute": float(q[2]),
        "p99_absolute": float(q[3]),
        "p99p9_absolute": float(q[4]),
        "maximum_absolute": float(np.max(absolute)),
    }


def aggregate_error_metrics(
    x_error: np.ndarray, xi_error: np.ndarray, orbit_ids: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unique = np.unique(orbit_ids)
    per_orbit: list[dict[str, Any]] = []
    for orbit_id in unique:
        mask = orbit_ids == orbit_id
        per_orbit.append({
            "orbit_id": str(orbit_id), "row_count": int(np.sum(mask)),
            "x": scalar_error_metrics(x_error[mask]),
            "xi": scalar_error_metrics(xi_error[mask]),
        })
    orbit_weighted: dict[str, Any] = {}
    for component in ("x", "xi"):
        orbit_weighted[component] = {
            key: float(np.mean([row[component][key] for row in per_orbit]))
            for key in per_orbit[0][component]
        }
        orbit_weighted[component]["rmse"] = float(np.sqrt(np.mean([
            row[component]["rmse"] ** 2 for row in per_orbit
        ])))
    return {
        "row_weighted": {
            "x": scalar_error_metrics(x_error), "xi": scalar_error_metrics(xi_error)
        },
        "orbit_weighted": orbit_weighted,
        "orbit_count": int(unique.size),
        "rows_per_orbit_minimum": int(min(row["row_count"] for row in per_orbit)),
        "rows_per_orbit_maximum": int(max(row["row_count"] for row in per_orbit)),
    }, per_orbit


def subset_metrics(
    predictions: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "row_count": int(np.sum(mask)),
        "x": scalar_error_metrics(predictions["x_error"][mask]),
        "xi": scalar_error_metrics(predictions["xi_error"][mask]),
    }


def bin_error_rows(
    coordinate: np.ndarray,
    predictions: dict[str, np.ndarray],
    edges: np.ndarray,
    coordinate_name: str,
) -> list[dict[str, Any]]:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    indices = np.digitize(coordinate, edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for index in range(edges.size - 1):
        mask = indices == index
        if not np.any(mask):
            continue
        x = scalar_error_metrics(predictions["x_error"][mask])
        xi = scalar_error_metrics(predictions["xi_error"][mask])
        rows.append({
            "bin_index": index, "coordinate": coordinate_name,
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "center": float(np.mean(coordinate[mask])), "row_count": int(np.sum(mask)),
            **{f"x_{key}": value for key, value in x.items()},
            **{f"xi_{key}": value for key, value in xi.items()},
        })
    return rows


def energy_metrics(predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    error = predictions["energy_error"]
    finite = np.isfinite(error)
    return {
        "finite_count": int(np.sum(finite)),
        "invalid_count": int(np.sum(~finite)),
        **scalar_error_metrics(error[finite]),
        "signed_mean": float(np.mean(error[finite])),
    }


def admissibility_summary(
    queries: dict[str, np.ndarray], predictions: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray]:
    xi_bad = np.abs(predictions["predicted_xi1"]) >= 1.0
    c_bad = predictions["predicted_C1"] <= 0.0
    invalid = xi_bad | c_bad | ~np.isfinite(predictions["predicted_C1"])
    return {
        "row_count": int(invalid.size),
        "absolute_xi_ge_1_count": int(np.sum(xi_bad)),
        "absolute_xi_ge_1_fraction": float(np.mean(xi_bad)),
        "C_le_0_count": int(np.sum(c_bad)),
        "C_le_0_fraction": float(np.mean(c_bad)),
        "union_violation_count": int(np.sum(invalid)),
        "union_violation_fraction": float(np.mean(invalid)),
        "orbit_ids_involved": sorted(set(queries["orbit_id"][invalid].tolist())),
    }, invalid


def build_composition_queries(bank: Any) -> dict[str, np.ndarray]:
    """Build deterministic exact-anchor composition intervals."""

    rows: dict[str, list[Any]] = {name: [] for name in (
        "orbit_index", "orbit_id", "anchor_index", "pair_index", "u_th", "E0",
        "t0", "s1", "s2", "s_total", "x0", "u0", "xi0",
        "exact_x_mid", "exact_u_mid", "exact_xi_mid",
        "exact_x1", "exact_u1", "exact_xi1",
    )}
    for orbit_index in range(int(bank["orbit_id"].size)):
        t0, _ = invert_saved_orbit_x(bank, orbit_index, COMPOSITION_ANCHOR_X)
        state0 = evaluate_saved_orbit_x_u_xi(bank, orbit_index, t0)
        remaining = float(bank["t_right"][orbit_index]) - t0
        for anchor_index, (anchor_time, anchor_state, rem) in enumerate(zip(t0, state0, remaining)):
            for pair_index, (fraction1, fraction2) in enumerate(COMPOSITION_FRACTION_PAIRS):
                s1, s2 = float(fraction1 * rem), float(fraction2 * rem)
                mid = evaluate_saved_orbit_x_u_xi(bank, orbit_index, anchor_time + s1)
                final = evaluate_saved_orbit_x_u_xi(bank, orbit_index, anchor_time + s1 + s2)
                values = {
                    "orbit_index": orbit_index, "orbit_id": str(bank["orbit_id"][orbit_index]),
                    "anchor_index": anchor_index, "pair_index": pair_index,
                    "u_th": float(bank["u_th"][orbit_index]), "E0": float(bank["E0"][orbit_index]),
                    "t0": float(anchor_time), "s1": s1, "s2": s2, "s_total": s1 + s2,
                    "x0": float(anchor_state[0]), "u0": float(anchor_state[1]), "xi0": float(anchor_state[2]),
                    "exact_x_mid": float(mid[0]), "exact_u_mid": float(mid[1]), "exact_xi_mid": float(mid[2]),
                    "exact_x1": float(final[0]), "exact_u1": float(final[1]), "exact_xi1": float(final[2]),
                }
                for name, value in values.items():
                    rows[name].append(value)
    output: dict[str, np.ndarray] = {"query_index": np.arange(len(rows["s1"]), dtype=np.int64)}
    integer = {"orbit_index": np.int32, "anchor_index": np.int16, "pair_index": np.int16}
    for name, values in rows.items():
        output[name] = np.asarray(values, dtype="U27" if name == "orbit_id" else integer.get(name, np.float64))
    return output


@torch.no_grad()
def evaluate_composition(
    model: ModelA,
    preprocessing: FiniteTimePreprocessing,
    queries: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    direct_queries = {
        "x0": queries["x0"], "xi0": queries["xi0"], "E0": queries["E0"], "s": queries["s_total"],
    }
    direct_features = stack_columns(direct_queries, INPUT_COLUMNS)
    direct_delta = preprocessing.unstandardize_targets(
        predict_standardized(
            model, torch.from_numpy(preprocessing.standardize_inputs(direct_features)), 16384
        ).cpu().numpy()
    )
    direct = np.column_stack((queries["x0"], queries["xi0"])) + direct_delta

    exact_restart_features = np.column_stack((
        queries["exact_x_mid"], queries["exact_xi_mid"], queries["E0"], queries["s2"]
    ))
    exact_restart_delta = preprocessing.unstandardize_targets(
        predict_standardized(
            model, torch.from_numpy(preprocessing.standardize_inputs(exact_restart_features)), 16384
        ).cpu().numpy()
    )
    exact_restart = np.column_stack((queries["exact_x_mid"], queries["exact_xi_mid"])) + exact_restart_delta

    first_features = np.column_stack((queries["x0"], queries["xi0"], queries["E0"], queries["s1"]))
    first_delta = preprocessing.unstandardize_targets(
        predict_standardized(
            model, torch.from_numpy(preprocessing.standardize_inputs(first_features)), 16384
        ).cpu().numpy()
    )
    learned_mid = np.column_stack((queries["x0"], queries["xi0"])) + first_delta
    second_features = np.column_stack((learned_mid[:, 0], learned_mid[:, 1], queries["E0"], queries["s2"]))
    second_delta = preprocessing.unstandardize_targets(
        predict_standardized(
            model, torch.from_numpy(preprocessing.standardize_inputs(second_features)), 16384
        ).cpu().numpy()
    )
    self_composed = learned_mid + second_delta
    exact_final = np.column_stack((queries["exact_x1"], queries["exact_xi1"]))
    return {
        "direct_x": direct[:, 0], "direct_xi": direct[:, 1],
        "exact_restart_x": exact_restart[:, 0], "exact_restart_xi": exact_restart[:, 1],
        "self_composed_x": self_composed[:, 0], "self_composed_xi": self_composed[:, 1],
        "exact_restart_x_error": exact_restart[:, 0] - exact_final[:, 0],
        "exact_restart_xi_error": exact_restart[:, 1] - exact_final[:, 1],
        "exact_restart_minus_direct_x": exact_restart[:, 0] - direct[:, 0],
        "exact_restart_minus_direct_xi": exact_restart[:, 1] - direct[:, 1],
        "self_x_error": self_composed[:, 0] - exact_final[:, 0],
        "self_xi_error": self_composed[:, 1] - exact_final[:, 1],
        "self_minus_direct_x": self_composed[:, 0] - direct[:, 0],
        "self_minus_direct_xi": self_composed[:, 1] - direct[:, 1],
    }


def composition_summary(result: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "exact_restart_vs_exact": {
            "x": scalar_error_metrics(result["exact_restart_x_error"]),
            "xi": scalar_error_metrics(result["exact_restart_xi_error"]),
        },
        "exact_restart_vs_direct": {
            "x": scalar_error_metrics(result["exact_restart_minus_direct_x"]),
            "xi": scalar_error_metrics(result["exact_restart_minus_direct_xi"]),
        },
        "self_composition_vs_exact": {
            "x": scalar_error_metrics(result["self_x_error"]),
            "xi": scalar_error_metrics(result["self_xi_error"]),
        },
        "self_composition_vs_direct": {
            "x": scalar_error_metrics(result["self_minus_direct_x"]),
            "xi": scalar_error_metrics(result["self_minus_direct_xi"]),
        },
    }


def load_finite_time_preprocessing(path: Path) -> FiniteTimePreprocessing:
    return FiniteTimePreprocessing.from_json(path)


def load_checkpoint(path: Path) -> ModelA:
    return load_trained_model(path)
