"""Validation-only regional and physical diagnostics for derivative treatments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from .dynamics import conserved_energy, timelike_margin
from .finite_time_derivative_audit import (
    predict_xi_and_physical_s_derivative,
    scalar_error_metrics,
)
from .finite_time_hybrid import HybridPreprocessing, predict_hybrid
from .model_a import ModelA
from .phase_b_orbits import evaluate_saved_orbit_x_u_xi
from .physics_gate import experiment_parameters, state_from_xi, xi_time_derivative


RAPID_THRESHOLD = 0.03651413248214974
REGION_GRID_POINTS = 4801
REFERENCE_GRID_POINTS = 2401
REFERENCE_ANCHOR_X = -14.0
REFERENCE_TARGETS = (0.05, 0.15, 0.30)


def interval_error_metrics(error: np.ndarray) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)
    if not values.size:
        return {"rmse": None, "mae": None, "median_absolute": None, "maximum_absolute": None}
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(absolute)),
        "median_absolute": float(np.median(absolute)),
        "maximum_absolute": float(np.max(absolute)),
    }


def _crossing_time(t0: float, t1: float, y0: float, y1: float, threshold: float) -> float:
    if y1 == y0:
        return 0.5 * (t0 + t1)
    fraction = (threshold - y0) / (y1 - y0)
    return float(t0 + np.clip(fraction, 0.0, 1.0) * (t1 - t0))


def rapid_intervals(
    times: np.ndarray,
    exact_dot_xi: np.ndarray,
    *,
    threshold: float = RAPID_THRESHOLD,
) -> list[dict[str, float | int]]:
    """Freeze exact-only during/pre/post intervals using feature-scaled widths.

    Exact above-threshold components are linearly interpolated at their
    boundaries.  Components are merged iteratively when the separating gap is
    no wider than the sum of their current widths; this guarantees that the
    subsequent width-matched post/pre windows do not overlap.
    """

    time = np.asarray(times, dtype=np.float64)
    sharpness = np.abs(np.asarray(exact_dot_xi, dtype=np.float64))
    if time.ndim != 1 or sharpness.shape != time.shape or time.size < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("aligned, strictly increasing dense arrays are required")
    above = sharpness > threshold
    changes = np.diff(np.pad(above.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    components: list[list[float]] = []
    for first, last in zip(starts, stops):
        left = float(time[first]) if first == 0 else _crossing_time(time[first - 1], time[first], sharpness[first - 1], sharpness[first], threshold)
        right = float(time[last]) if last == time.size - 1 else _crossing_time(time[last], time[last + 1], sharpness[last], sharpness[last + 1], threshold)
        components.append([left, right])
    merged = True
    while merged and len(components) > 1:
        merged = False
        output: list[list[float]] = []
        index = 0
        while index < len(components):
            current = components[index]
            if index + 1 < len(components):
                following = components[index + 1]
                gap = following[0] - current[1]
                if gap <= (current[1] - current[0]) + (following[1] - following[0]):
                    output.append([current[0], following[1]])
                    index += 2
                    merged = True
                    continue
            output.append(current)
            index += 1
        components = output
    records = []
    for feature_index, (start, stop) in enumerate(components):
        width = stop - start
        records.append({
            "feature_index": feature_index,
            "pre_start": max(float(time[0]), start - width),
            "pre_stop": start,
            "during_start": start,
            "during_stop": stop,
            "post_start": stop,
            "post_stop": min(float(time[-1]), stop + width),
            "feature_width": width,
        })
    return records


def validation_feature_table(
    validation: dict[str, np.ndarray],
    bank: dict[str, np.ndarray],
    *,
    threshold: float = RAPID_THRESHOLD,
    grid_points: int = REGION_GRID_POINTS,
) -> list[dict[str, Any]]:
    """Construct feature windows from exact validation-bank dynamics only."""

    wormhole, spiral = experiment_parameters()
    table: list[dict[str, Any]] = []
    for orbit_index in np.unique(validation["source_orbit_index"]):
        orbit_index = int(orbit_index)
        times = np.linspace(float(bank["t_left"][orbit_index]), float(bank["t_right"][orbit_index]), grid_points)
        state = evaluate_saved_orbit_x_u_xi(bank, orbit_index, times)
        exact_dot = xi_time_derivative(state[:, 0], state[:, 1], wormhole, spiral)
        for feature in rapid_intervals(times, exact_dot, threshold=threshold):
            table.append({
                "source_orbit_index": orbit_index,
                "orbit_id": str(bank["orbit_id"][orbit_index]),
                "u_th": float(bank["u_th"][orbit_index]),
                "dense_step": float(times[1] - times[0]),
                **feature,
            })
    return table


def regional_masks(
    rows: dict[str, np.ndarray], feature_table: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    """Map exact feature windows to validation endpoint rows."""

    regions = {name: np.zeros(rows["s"].size, dtype=bool) for name in ("pre", "during", "post")}
    features: list[dict[str, np.ndarray]] = []
    endpoint = np.asarray(rows["t1"], dtype=np.float64)
    source = np.asarray(rows["source_orbit_index"])
    for feature in feature_table:
        orbit = source == int(feature["source_orbit_index"])
        masks = {
            "pre": orbit & (endpoint >= feature["pre_start"]) & (endpoint < feature["pre_stop"]),
            "during": orbit & (endpoint >= feature["during_start"]) & (endpoint <= feature["during_stop"]),
            "post": orbit & (endpoint > feature["post_start"]) & (endpoint <= feature["post_stop"]),
        }
        for name, mask in masks.items():
            regions[name] |= mask
        features.append(masks)
    if np.any((regions["pre"] & regions["during"]) | (regions["during"] & regions["post"]) | (regions["pre"] & regions["post"])):
        raise RuntimeError("frozen regional masks overlap")
    return regions, features


def physical_diagnostics(
    validation: dict[str, np.ndarray], prediction: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Reuse the established admissibility and conserved-energy diagnostics."""

    wormhole, spiral = experiment_parameters()
    x_hat, xi_hat = prediction["predicted_x1"], prediction["predicted_xi1"]
    _, u_hat = state_from_xi(x_hat, xi_hat, wormhole, spiral)
    margin = timelike_margin(x_hat, u_hat, wormhole, spiral)
    valid = np.isfinite(margin) & (margin > 0.0)
    energy_error = np.full(x_hat.shape, np.nan, dtype=np.float64)
    energy_error[valid] = conserved_energy(x_hat[valid], u_hat[valid], wormhole, spiral) - validation["E0"][valid]
    finite = energy_error[np.isfinite(energy_error)]
    absolute = np.abs(finite)
    return {
        "row_count": int(x_hat.size),
        "absolute_xi_ge_1_count": int(np.sum(np.abs(xi_hat) >= 1.0)),
        "C_le_0_count": int(np.sum(margin <= 0.0)),
        "union_violation_count": int(np.sum((np.abs(xi_hat) >= 1.0) | (margin <= 0.0) | ~np.isfinite(margin))),
        "energy_finite_count": int(finite.size),
        "energy_invalid_count": int(x_hat.size - finite.size),
        "energy_mae": float(np.mean(absolute)),
        "energy_rmse": float(np.sqrt(np.mean(finite**2))),
        "energy_p99_absolute": float(np.quantile(absolute, .99)),
        "energy_maximum_absolute": float(np.max(absolute)),
    }


def treatment_validation_diagnostics(
    model: ModelA,
    preprocessing: HybridPreprocessing,
    validation: dict[str, np.ndarray],
    exact_dot_xi: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute global, regional, family, horizon, and safeguard diagnostics."""

    state = predict_hybrid(model, preprocessing, validation)
    derivative = predict_xi_and_physical_s_derivative(model, preprocessing, validation)
    inference_discrepancy = float(np.max(np.abs(state["predicted_xi1"] - derivative["predicted_xi1"])))
    if inference_discrepancy > 2.0e-15:
        raise RuntimeError("state and derivative inference paths disagree beyond floating-point roundoff")
    x_error = state["predicted_x1"] - validation["x1"]
    xi_error = state["predicted_xi1"] - validation["xi1"]
    dot_error = derivative["predicted_dot_xi1"] - np.asarray(exact_dot_xi)
    prediction = {**state, **derivative, "predicted_xi1": state["predicted_xi1"],
                  "x_error": x_error, "xi_error": xi_error, "dot_xi_error": dot_error}

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        return {
            "row_count": int(np.sum(mask)),
            "x": interval_error_metrics(x_error[mask]),
            "xi": interval_error_metrics(xi_error[mask]),
            "dot_xi": interval_error_metrics(dot_error[mask]),
        }

    all_rows = np.ones(x_error.size, dtype=bool)
    families = {
        "hard_u_th_le_0p30": validation["u_th"] <= .30,
        "ordinary_u_th_gt_0p30": validation["u_th"] > .30,
    }
    for center in REFERENCE_TARGETS:
        families[f"u_th_within_0p01_of_{center:.2f}"] = np.abs(validation["u_th"] - center) <= .01
    horizons = {
        "short_s_le_5": validation["s"] <= 5.0,
        "intermediate_5_lt_s_le_20": (validation["s"] > 5.0) & (validation["s"] <= 20.0),
        "long_s_gt_20": validation["s"] > 20.0,
        "hard_long_s_gt_20": (validation["u_th"] <= .30) & (validation["s"] > 20.0),
    }
    return {
        "global": metrics(all_rows),
        "regions": {name: metrics(mask) for name, mask in region_masks.items()},
        "families": {name: metrics(mask) for name, mask in families.items()},
        "horizons": {name: metrics(mask) for name, mask in horizons.items()},
        "physical": physical_diagnostics(validation, state),
        "maximum_state_vs_derivative_inference_xi_discrepancy": inference_discrepancy,
    }, prediction


def feature_level_metrics(
    feature_table: list[dict[str, Any]],
    feature_masks: list[dict[str, np.ndarray]],
    prediction: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    output = []
    for feature, masks in zip(feature_table, feature_masks):
        during = interval_error_metrics(prediction["dot_xi_error"][masks["during"]])
        post = interval_error_metrics(prediction["xi_error"][masks["post"]])
        output.append({
            **feature,
            "during_row_count": int(np.sum(masks["during"])),
            "post_row_count": int(np.sum(masks["post"])),
            "dot_xi_during_rmse": during["rmse"],
            "dot_xi_during_mae": during["mae"],
            "xi_post_rmse": post["rmse"],
            "xi_post_mae": post["mae"],
        })
    return output


def dense_feature_diagnostics(
    elapsed: np.ndarray,
    exact_dot_xi: np.ndarray,
    predicted_dot_xi: np.ndarray,
    *,
    threshold: float = RAPID_THRESHOLD,
) -> list[dict[str, Any]]:
    """Match significant peaks by time and report timing/amplitude/width safely."""

    time = np.asarray(elapsed, dtype=np.float64)
    exact = np.abs(np.asarray(exact_dot_xi, dtype=np.float64))
    predicted = np.abs(np.asarray(predicted_dot_xi, dtype=np.float64))
    exact_peaks, _ = find_peaks(exact, prominence=max(float(exact.max()) * .05, 1e-15))
    predicted_peaks, _ = find_peaks(predicted, prominence=max(float(predicted.max()) * .05, 1e-15))
    if not exact_peaks.size or not predicted_peaks.size:
        return []
    exact_assignment, predicted_assignment = linear_sum_assignment(np.abs(time[exact_peaks, None] - time[predicted_peaks][None, :]))

    def half_crossings(values: np.ndarray, peak: int, peaks: np.ndarray) -> tuple[float, float] | None:
        half = .5 * float(values[peak])
        competitors = peaks[(peaks != peak) & (values[peaks] >= .8 * values[peak])]
        if competitors.size:
            return None
        left = np.flatnonzero(values[:peak] <= half)
        right = np.flatnonzero(values[peak + 1:] <= half)
        if not left.size or not right.size:
            return None
        i0, i1 = int(left[-1]), int(left[-1] + 1)
        j1 = int(peak + 1 + right[0]); j0 = j1 - 1
        rising = _crossing_time(time[i0], time[i1], values[i0], values[i1], half)
        falling = _crossing_time(time[j0], time[j1], values[j0], values[j1], half)
        return rising, falling

    rows = []
    for exact_position, predicted_position in zip(exact_assignment, predicted_assignment):
        e = int(exact_peaks[exact_position]); p = int(predicted_peaks[predicted_position])
        ec, pc = half_crossings(exact, e, exact_peaks), half_crossings(predicted, p, predicted_peaks)
        status = "reported" if ec is not None and pc is not None else "ambiguous_not_reported"
        row: dict[str, Any] = {
            "exact_peak_s": float(time[e]), "predicted_peak_s": float(time[p]),
            "delta_peak_s": float(time[p] - time[e]),
            "exact_peak_amplitude": float(exact[e]), "predicted_peak_amplitude": float(predicted[p]),
            "amplitude_error": float(predicted[p] - exact[e]),
            "exact_relative_height": float(exact[e] / exact.max()),
            "predicted_relative_height": float(predicted[p] / predicted.max()),
            "half_max_status": status,
        }
        if status == "reported" and ec is not None and pc is not None:
            row.update({
                "exact_half_max_width": ec[1] - ec[0], "predicted_half_max_width": pc[1] - pc[0],
                "half_max_width_error": (pc[1] - pc[0]) - (ec[1] - ec[0]),
                "rising_half_max_offset": pc[0] - ec[0], "falling_half_max_offset": pc[1] - ec[1],
            })
        rows.append(row)
    return sorted(rows, key=lambda row: row["exact_peak_s"])
