"""Gradient-scale diagnostics for the frozen finite-time hybrid model.

The functions here use ``torch.autograd.grad`` only.  They expose no optimizer
and never mutate parameters, datasets, or preprocessing constants.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .finite_time import INPUT_COLUMNS, stack_columns
from .finite_time_hybrid import HYBRID_TARGET_COLUMNS, HybridPreprocessing, S_STAR
from .model_a import ModelA


PARAMETER_GROUPS = ("input_layer", "hidden_layers.0", "output_layer")


def _parameter_group(name: str) -> str:
    for group in PARAMETER_GROUPS:
        if name.startswith(group):
            return group
    raise ValueError(f"unrecognized parameter group for {name}")


def _gradient_geometry(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradient0: tuple[torch.Tensor | None, ...],
    gradient_d: tuple[torch.Tensor | None, ...],
    *,
    near_zero: float,
) -> dict[str, Any]:
    group_sums = {group: [0.0, 0.0, 0.0] for group in PARAMETER_GROUPS}
    sum0 = sumd = dot = 0.0
    unused0 = []; unusedd = []
    for (name, parameter), g0, gd in zip(named_parameters, gradient0, gradient_d):
        if g0 is None:
            g0 = torch.zeros_like(parameter); unused0.append(name)
        if gd is None:
            gd = torch.zeros_like(parameter); unusedd.append(name)
        a = g0.detach().double(); b = gd.detach().double()
        a2 = float(torch.sum(a * a)); b2 = float(torch.sum(b * b)); ab = float(torch.sum(a * b))
        sum0 += a2; sumd += b2; dot += ab
        group = _parameter_group(name); group_sums[group][0] += a2; group_sums[group][1] += b2; group_sums[group][2] += ab
    norm0, normd = math.sqrt(sum0), math.sqrt(sumd)
    cosine = None if norm0 <= near_zero or normd <= near_zero else dot / (norm0 * normd)
    ratio = None if norm0 <= near_zero else normd / norm0
    groups = {}
    for name, (a2, b2, ab) in group_sums.items():
        a, b = math.sqrt(a2), math.sqrt(b2)
        groups[name] = {"gradient0_norm": a, "gradient_d_norm": b,
                        "ratio": None if a <= near_zero else b / a,
                        "cosine": None if a <= near_zero or b <= near_zero else ab / (a * b),
                        "near_zero_gradient0": a <= near_zero, "near_zero_gradient_d": b <= near_zero}
    return {"gradient0_norm": norm0, "gradient_d_norm": normd, "ratio": ratio, "cosine": cosine,
            "near_zero_gradient0": norm0 <= near_zero, "near_zero_gradient_d": normd <= near_zero,
            "unused_gradient0_parameters": unused0, "unused_gradient_d_parameters": unusedd, "parameter_groups": groups}


def audit_gradient_batch(
    model: ModelA,
    preprocessing: HybridPreprocessing,
    batch: dict[str, np.ndarray],
    exact_dot_xi: np.ndarray,
    *,
    dot_xi_mean: float,
    dot_xi_std: float,
    near_zero: float = 1.0e-20,
) -> dict[str, Any]:
    """Measure independent parameter gradients of current and derivative loss."""

    if not np.isfinite(dot_xi_std) or dot_xi_std <= 0.0:
        raise ValueError("dot_xi_std must be finite and positive")
    count = int(np.asarray(batch["s"]).size)
    exact = np.asarray(exact_dot_xi, dtype=np.float64)
    if exact.shape != (count,):
        raise ValueError("exact_dot_xi must align with the batch")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = tuple(parameter for _, parameter in named)
    before = {name: parameter.detach().clone() for name, parameter in named}

    standardized_inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(batch, INPUT_COLUMNS)))
    standardized_targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(batch, HYBRID_TARGET_COLUMNS)))
    start0 = perf_counter()
    output0 = model(standardized_inputs)
    loss0 = torch.mean((output0 - standardized_targets) ** 2)
    gradient0 = torch.autograd.grad(loss0, parameters, create_graph=False, retain_graph=False, allow_unused=True)
    elapsed0 = perf_counter() - start0

    input_mean = torch.as_tensor(preprocessing.input_mean, dtype=torch.float64)
    input_std = torch.as_tensor(preprocessing.input_std, dtype=torch.float64)
    target_mean = torch.as_tensor(preprocessing.target_mean, dtype=torch.float64)
    target_std = torch.as_tensor(preprocessing.target_std, dtype=torch.float64)
    x0 = torch.as_tensor(batch["x0"], dtype=torch.float64)
    xi0 = torch.as_tensor(batch["xi0"], dtype=torch.float64)
    energy = torch.as_tensor(batch["E0"], dtype=torch.float64)
    physical_s = torch.as_tensor(batch["s"], dtype=torch.float64).clone().detach().requires_grad_(True)
    exact_tensor = torch.as_tensor(exact, dtype=torch.float64)
    startd = perf_counter()
    physical = torch.stack((x0, xi0, energy, physical_s), dim=1)
    standardized = ((physical - input_mean) / input_std).to(dtype=torch.float32)
    standardized_output = model(standardized)
    physical_output = standardized_output.to(dtype=torch.float64) * target_std + target_mean
    gate = -S_STAR * torch.expm1(-physical_s / S_STAR)
    xi_hat = xi0 + gate * physical_output[:, 1]
    dot_xi_hat = torch.autograd.grad(
        xi_hat, physical_s, grad_outputs=torch.ones_like(xi_hat), create_graph=True, retain_graph=True,
    )[0]
    normalized_prediction = (dot_xi_hat - dot_xi_mean) / dot_xi_std
    normalized_exact = (exact_tensor - dot_xi_mean) / dot_xi_std
    lossd = torch.mean((normalized_prediction - normalized_exact) ** 2)
    gradientd = torch.autograd.grad(lossd, parameters, create_graph=False, retain_graph=False, allow_unused=True)
    elapsedd = perf_counter() - startd

    for name, parameter in named:
        if not torch.equal(parameter.detach(), before[name]):
            raise RuntimeError(f"parameter changed during gradient-only audit: {name}")
    geometry = _gradient_geometry(named, gradient0, gradientd, near_zero=near_zero)
    return {"batch_size": count, "loss0": float(loss0.detach()), "loss_d": float(lossd.detach()),
            "loss0_forward_backward_seconds": elapsed0, "loss_d_forward_mixed_backward_seconds": elapsedd,
            "time_ratio_d_over_0": elapsedd / elapsed0 if elapsed0 > 0 else None, **geometry}


def deterministic_batch_indices(
    eligible_indices: np.ndarray,
    *,
    batch_size: int,
    batch_count: int,
    seed: int,
) -> list[np.ndarray]:
    """Choose existing rows without replacement within a diagnostic category."""

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    required = batch_size * batch_count
    if eligible.ndim != 1 or eligible.size < required:
        raise ValueError(f"category has {eligible.size} rows but {required} are required")
    permutation = np.random.default_rng(seed).permutation(eligible)
    selected = permutation[:required]
    return [selected[start:start + batch_size] for start in range(0, required, batch_size)]


def distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "standard_deviation": float(np.std(array, ddof=0)), "minimum": float(np.min(array)),
            "p10": float(np.quantile(array, .10)), "p25": float(np.quantile(array, .25)),
            "p75": float(np.quantile(array, .75)), "p90": float(np.quantile(array, .90)), "maximum": float(np.max(array))}


def rounded_fixed_lambda(value: float, significant_digits: int = 2) -> float:
    """Round a positive calibration value to a fixed readable scalar."""

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("lambda candidate must be finite and positive")
    decimals = significant_digits - 1 - int(math.floor(math.log10(abs(value))))
    return float(round(value, decimals))
