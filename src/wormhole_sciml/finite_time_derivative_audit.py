"""Evaluation-only derivative diagnostics for the frozen finite-time hybrid.

This module contains no fitting or training entry points.  Exact derivatives
reuse :func:`wormhole_sciml.physics_gate.xi_time_derivative`; model derivatives
are taken with respect to a leaf tensor containing physical elapsed time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
import torch

from .finite_time_hybrid import HybridPreprocessing, S_STAR
from .phase_b_orbits import evaluate_saved_orbit_x_u_xi


def predict_xi_and_physical_s_derivative(
    model: torch.nn.Module,
    preprocessing: HybridPreprocessing,
    data: dict[str, np.ndarray],
    *,
    batch_size: int = 8192,
) -> dict[str, np.ndarray]:
    """Return frozen ``xi_hat`` and ``d xi_hat / d physical_s``.

    Physical inputs and normalization constants remain float64 until the
    standardized feature tensor is cast to the checkpoint's float32 dtype.
    Autograd therefore includes the physical-to-standardized time Jacobian;
    the differentiated leaf is never standardized time.
    """

    count = int(np.asarray(data["s"]).size)
    for name in ("x0", "xi0", "E0", "s"):
        value = np.asarray(data[name])
        if value.ndim != 1 or value.size != count:
            raise ValueError(f"{name} must be a one-dimensional array of length {count}")
    if np.any(np.asarray(data["s"], dtype=np.float64) < 0.0):
        raise ValueError("physical elapsed time cannot be negative")

    predicted_xi = np.empty(count, dtype=np.float64)
    predicted_dot_xi = np.empty(count, dtype=np.float64)
    predicted_f_xi = np.empty(count, dtype=np.float64)
    gate_values = np.empty(count, dtype=np.float64)
    input_mean = torch.as_tensor(preprocessing.input_mean, dtype=torch.float64)
    input_std = torch.as_tensor(preprocessing.input_std, dtype=torch.float64)
    target_mean = torch.as_tensor(preprocessing.target_mean, dtype=torch.float64)
    target_std = torch.as_tensor(preprocessing.target_std, dtype=torch.float64)
    model.eval()

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        x0 = torch.as_tensor(np.asarray(data["x0"])[start:stop], dtype=torch.float64)
        xi0 = torch.as_tensor(np.asarray(data["xi0"])[start:stop], dtype=torch.float64)
        energy = torch.as_tensor(np.asarray(data["E0"])[start:stop], dtype=torch.float64)
        physical_s = torch.as_tensor(
            np.asarray(data["s"])[start:stop], dtype=torch.float64
        ).clone().detach().requires_grad_(True)
        physical = torch.stack((x0, xi0, energy, physical_s), dim=1)
        standardized = ((physical - input_mean) / input_std).to(dtype=torch.float32)
        standardized_output = model(standardized)
        physical_output = standardized_output.to(dtype=torch.float64) * target_std + target_mean
        f_xi = physical_output[:, 1]
        gate = -S_STAR * torch.expm1(-physical_s / S_STAR)
        xi_hat = xi0 + gate * f_xi
        derivative = torch.autograd.grad(
            xi_hat,
            physical_s,
            grad_outputs=torch.ones_like(xi_hat),
            create_graph=False,
            retain_graph=False,
        )[0]
        predicted_xi[start:stop] = xi_hat.detach().cpu().numpy()
        predicted_dot_xi[start:stop] = derivative.detach().cpu().numpy()
        predicted_f_xi[start:stop] = f_xi.detach().cpu().numpy()
        gate_values[start:stop] = gate.detach().cpu().numpy()
    return {
        "predicted_xi1": predicted_xi,
        "predicted_dot_xi1": predicted_dot_xi,
        "predicted_F_xi": predicted_f_xi,
        "xi_gate": gate_values,
    }


def finite_difference_saved_xi_derivative(
    bank: dict[str, np.ndarray] | Any,
    orbit_index: int,
    times: np.ndarray,
    *,
    base_step: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate saved xi using five-point centered/one-sided stencils."""

    requested = np.asarray(times, dtype=np.float64)
    if requested.ndim != 1:
        raise ValueError("times must be one-dimensional")
    left = float(bank["t_left"][orbit_index])
    right = float(bank["t_right"][orbit_index])
    if np.any(requested < left - 1e-13) or np.any(requested > right + 1e-13):
        raise ValueError("finite-difference time lies outside saved orbit")
    derivative = np.empty(requested.size, dtype=np.float64)
    step_used = np.empty(requested.size, dtype=np.float64)
    method = np.empty(requested.size, dtype="U16")

    for index, time in enumerate(requested):
        if time - 2.0 * base_step >= left and time + 2.0 * base_step <= right:
            h = base_step
            values = evaluate_saved_orbit_x_u_xi(
                bank, orbit_index, time + h * np.asarray([-2.0, -1.0, 1.0, 2.0])
            )[:, 2]
            derivative[index] = (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * h)
            method[index] = "centered_5point"
        elif time - left < right - time:
            h = min(base_step, (right - time) / 4.0)
            if h <= 0.0:
                raise ValueError("forward stencil has no available interval")
            values = evaluate_saved_orbit_x_u_xi(bank, orbit_index, time + h * np.arange(5.0))[:, 2]
            derivative[index] = (-25.0 * values[0] + 48.0 * values[1] - 36.0 * values[2] + 16.0 * values[3] - 3.0 * values[4]) / (12.0 * h)
            method[index] = "forward_5point"
        else:
            h = min(base_step, (time - left) / 4.0)
            if h <= 0.0:
                raise ValueError("backward stencil has no available interval")
            values = evaluate_saved_orbit_x_u_xi(bank, orbit_index, time - h * np.arange(5.0))[:, 2]
            derivative[index] = (25.0 * values[0] - 48.0 * values[1] + 36.0 * values[2] - 16.0 * values[3] + 3.0 * values[4]) / (12.0 * h)
            method[index] = "backward_5point"
        step_used[index] = h
    return derivative, method, step_used


def scalar_error_metrics(error: np.ndarray) -> dict[str, float]:
    values = np.asarray(error, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mae": float(np.mean(absolute)),
        "median_absolute": float(np.quantile(absolute, 0.50)),
        "p90_absolute": float(np.quantile(absolute, 0.90)),
        "p95_absolute": float(np.quantile(absolute, 0.95)),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "p99p9_absolute": float(np.quantile(absolute, 0.999)),
        "maximum_absolute": float(np.max(absolute)),
    }


def value_distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    absolute = np.abs(array)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        **{f"absolute_p{label}": float(np.quantile(absolute, quantile)) for label, quantile in (
            ("0", 0.0), ("25", .25), ("50", .50), ("75", .75), ("90", .90),
            ("95", .95), ("99", .99), ("99p5", .995), ("99p9", .999), ("100", 1.0),
        )},
    }


def training_sharpness_edges(exact_dot_xi: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return fixed validation bins derived only from training |dot(xi)|."""

    quantiles = np.asarray([0.50, 0.75, 0.90, 0.95, 0.99], dtype=np.float64)
    internal = np.quantile(np.abs(np.asarray(exact_dot_xi, dtype=np.float64)), quantiles)
    if np.any(np.diff(internal) <= 0.0):
        raise RuntimeError("training sharpness quantiles are not strictly increasing")
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    labels = ["q00_q50", "q50_q75", "q75_q90", "q90_q95", "q95_q99", "q99_q100"]
    return edges, labels


def binned_error_metrics(
    exact_dot_xi: np.ndarray,
    predicted_dot_xi: np.ndarray,
    xi_error: np.ndarray,
    edges: np.ndarray,
    labels: list[str],
) -> list[dict[str, Any]]:
    sharpness = np.abs(np.asarray(exact_dot_xi, dtype=np.float64))
    dot_error = np.asarray(predicted_dot_xi, dtype=np.float64) - np.asarray(exact_dot_xi, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        mask = (sharpness > edges[index]) & (sharpness <= edges[index + 1])
        if index == 0:
            mask = sharpness <= edges[index + 1]
        rows.append({
            "bin": label,
            "lower_exclusive": None if not np.isfinite(edges[index]) else float(edges[index]),
            "upper_inclusive": None if not np.isfinite(edges[index + 1]) else float(edges[index + 1]),
            "row_count": int(np.sum(mask)),
            "row_fraction": float(np.mean(mask)),
            **{f"dot_xi_{key}": value for key, value in scalar_error_metrics(dot_error[mask]).items()},
            **{f"xi_{key}": value for key, value in scalar_error_metrics(np.asarray(xi_error)[mask]).items()},
        })
    return rows


def _quadratic_peak_time(times: np.ndarray, values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= times.size - 1:
        return float(times[index])
    x = times[index - 1:index + 2]
    y = values[index - 1:index + 2]
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] >= 0.0:
        return float(times[index])
    vertex = -coefficients[1] / (2.0 * coefficients[0])
    return float(vertex) if x[0] <= vertex <= x[-1] else float(times[index])


def _half_max_crossings(times: np.ndarray, values: np.ndarray, peak: int) -> tuple[float, float] | None:
    half = 0.5 * float(values[peak])
    left_candidates = np.flatnonzero(values[:peak] < half)
    right_candidates = np.flatnonzero(values[peak + 1:] < half)
    if not left_candidates.size or not right_candidates.size:
        return None
    left_below = int(left_candidates[-1]); left_above = left_below + 1
    right_below = int(peak + 1 + right_candidates[0]); right_above = right_below - 1
    def interpolate(i0: int, i1: int) -> float:
        fraction = (half - values[i0]) / (values[i1] - values[i0])
        return float(times[i0] + fraction * (times[i1] - times[i0]))
    return interpolate(left_below, left_above), interpolate(right_above, right_below)


def dominant_peak_timing(
    times: np.ndarray,
    exact_dot_xi: np.ndarray,
    predicted_dot_xi: np.ndarray,
    *,
    ambiguity_ratio: float = 0.80,
) -> dict[str, Any]:
    """Compare global |dot(xi)| peaks and conditionally report half maxima."""

    time = np.asarray(times, dtype=np.float64)
    exact = np.abs(np.asarray(exact_dot_xi, dtype=np.float64))
    predicted = np.abs(np.asarray(predicted_dot_xi, dtype=np.float64))
    if not (time.ndim == exact.ndim == predicted.ndim == 1 and time.size == exact.size == predicted.size):
        raise ValueError("timing arrays must be one-dimensional and aligned")
    exact_peak = int(np.argmax(exact)); predicted_peak = int(np.argmax(predicted))
    exact_peaks, _ = find_peaks(exact, prominence=max(float(exact.max()) * .05, 1e-15))
    predicted_peaks, _ = find_peaks(predicted, prominence=max(float(predicted.max()) * .05, 1e-15))
    def second_ratio(values: np.ndarray, peaks: np.ndarray, primary: int) -> float:
        candidates = values[peaks[peaks != primary]]
        return 0.0 if not candidates.size else float(np.max(candidates) / values[primary])
    exact_ratio = second_ratio(exact, exact_peaks, exact_peak)
    predicted_ratio = second_ratio(predicted, predicted_peaks, predicted_peak)
    exact_cross = _half_max_crossings(time, exact, exact_peak)
    predicted_cross = _half_max_crossings(time, predicted, predicted_peak)
    clean = exact_ratio < ambiguity_ratio and predicted_ratio < ambiguity_ratio and exact_cross is not None and predicted_cross is not None
    exact_time = _quadratic_peak_time(time, exact, exact_peak)
    predicted_time = _quadratic_peak_time(time, predicted, predicted_peak)
    output: dict[str, Any] = {
        "exact_peak_s": exact_time,
        "predicted_peak_s": predicted_time,
        "delta_peak_s": predicted_time - exact_time,
        "exact_peak_abs_dot_xi": float(exact[exact_peak]),
        "predicted_peak_abs_dot_xi": float(predicted[predicted_peak]),
        "exact_significant_peak_count": int(exact_peaks.size),
        "predicted_significant_peak_count": int(predicted_peaks.size),
        "exact_second_peak_ratio": exact_ratio,
        "predicted_second_peak_ratio": predicted_ratio,
        "half_max_status": "reported" if clean else "ambiguous_not_reported",
        "ambiguity_reason": None if clean else "competing peak >= 0.8 of dominant peak or an unbounded half-maximum crossing",
    }
    if exact_peaks.size and predicted_peaks.size:
        exact_assignment, predicted_assignment = linear_sum_assignment(
            np.abs(time[exact_peaks, None] - time[predicted_peaks][None, :])
        )
        pairs = []
        for exact_position, predicted_position in zip(exact_assignment, predicted_assignment):
            exact_index = int(exact_peaks[exact_position]); predicted_index = int(predicted_peaks[predicted_position])
            exact_feature_time = _quadratic_peak_time(time, exact, exact_index)
            predicted_feature_time = _quadratic_peak_time(time, predicted, predicted_index)
            pairs.append({
                "exact_s": exact_feature_time,
                "predicted_s": predicted_feature_time,
                "delta_s": predicted_feature_time - exact_feature_time,
                "exact_relative_height": float(exact[exact_index] / exact[exact_peak]),
                "predicted_relative_height": float(predicted[predicted_index] / predicted[predicted_peak]),
            })
        output["matched_significant_peaks"] = sorted(pairs, key=lambda row: row["exact_s"])
        output["unmatched_exact_peak_count"] = int(exact_peaks.size - len(pairs))
        output["unmatched_predicted_peak_count"] = int(predicted_peaks.size - len(pairs))
    else:
        output["matched_significant_peaks"] = []
        output["unmatched_exact_peak_count"] = int(exact_peaks.size)
        output["unmatched_predicted_peak_count"] = int(predicted_peaks.size)
    if clean and exact_cross is not None and predicted_cross is not None:
        output.update({
            "exact_rising_half_max_s": exact_cross[0],
            "predicted_rising_half_max_s": predicted_cross[0],
            "delta_rising_half_max_s": predicted_cross[0] - exact_cross[0],
            "exact_falling_half_max_s": exact_cross[1],
            "predicted_falling_half_max_s": predicted_cross[1],
            "delta_falling_half_max_s": predicted_cross[1] - exact_cross[1],
        })
    return output
