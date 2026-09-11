"""Evaluation-only helpers for the frozen hybrid finite-time checkpoint."""

from __future__ import annotations

import numpy as np

from .finite_time_hybrid import HybridPreprocessing, predict_hybrid
from .finite_time_validation import prediction_diagnostics
from .model_a import ModelA


def predict_hybrid_diagnostics(
    model: ModelA, preprocessing: HybridPreprocessing, queries: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    raw = predict_hybrid(model, preprocessing, queries)
    increments = np.column_stack((raw["predicted_Delta_x"], raw["predicted_Delta_xi"]))
    diagnostic = prediction_diagnostics(
        queries, increments, raw["predicted_x1"], raw["predicted_xi1"]
    )
    return {**raw, **diagnostic}


def median_log_grid(
    x: np.ndarray,
    y: np.ndarray,
    absolute_error: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log10 median absolute error and sample counts; empty cells stay NaN."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    x_index = np.digitize(x, x_edges[1:-1])
    y_index = np.digitize(y, y_edges[1:-1])
    grid = np.full((y_edges.size - 1, x_edges.size - 1), np.nan)
    count = np.zeros_like(grid, dtype=np.int64)
    for yi in range(grid.shape[0]):
        for xi in range(grid.shape[1]):
            mask = (x_index == xi) & (y_index == yi)
            count[yi, xi] = int(np.sum(mask))
            if count[yi, xi]:
                grid[yi, xi] = np.log10(float(np.median(absolute_error[mask])) + epsilon)
    return grid, count


def log_error_ratio(hybrid_absolute: np.ndarray, local_absolute: np.ndarray, epsilon: float) -> np.ndarray:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return np.log10((np.asarray(hybrid_absolute) + epsilon) / (np.asarray(local_absolute) + epsilon))
