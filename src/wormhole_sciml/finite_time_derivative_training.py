"""Controlled derivative-loss training for the finite-time hybrid model.

Only the scalar derivative-loss coefficient differs from the frozen hybrid
training protocol.  Checkpoint selection remains the existing validation
orbit-averaged standardized hybrid-target MSE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import resource
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .finite_time import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PATIENCE,
    TRAINING_SEEDS,
    orbit_averaged_standardized_mse,
    stack_columns,
)
from .finite_time_hybrid import (
    HYBRID_TARGET_COLUMNS,
    S_STAR,
    HybridPreprocessing,
    _set_deterministic_seed,
    build_hybrid_model,
    expected_hybrid_parameter_count,
)
from .model_a import parameter_count, predict_standardized, state_dict_sha256


DERIVATIVE_SCALE = 0.014438580924341927
DERIVATIVE_MEAN = -0.0018535215398591892
TREATMENT_LAMBDAS = (0.0, 0.011, 0.034, 0.068)


def derivative_training_dataset(
    data: dict[str, np.ndarray],
    preprocessing: HybridPreprocessing,
    exact_dot_xi: np.ndarray,
) -> TensorDataset:
    """Return the unchanged standardized tensors plus physical derivative data."""

    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, ("x0", "xi0", "E0", "s"))))
    targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(data, HYBRID_TARGET_COLUMNS)))
    exact = np.asarray(exact_dot_xi, dtype=np.float64)
    if exact.shape != np.asarray(data["s"]).shape or not np.all(np.isfinite(exact)):
        raise ValueError("exact derivative targets must be finite and row-aligned")
    return TensorDataset(
        inputs,
        targets,
        torch.as_tensor(np.asarray(data["xi0"], dtype=np.float64)),
        torch.as_tensor(np.asarray(data["s"], dtype=np.float64)),
        torch.as_tensor(exact),
    )


def derivative_loss_components(
    model: nn.Module,
    preprocessing: HybridPreprocessing,
    standardized_inputs: torch.Tensor,
    standardized_targets: torch.Tensor,
    xi0: torch.Tensor,
    physical_s_values: torch.Tensor,
    exact_dot_xi: torch.Tensor,
    lambda_dot_xi: float,
    *,
    derivative_mean: float = DERIVATIVE_MEAN,
    derivative_scale: float = DERIVATIVE_SCALE,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Construct the existing loss and optional physical-time derivative loss."""

    if lambda_dot_xi < 0.0 or not np.isfinite(lambda_dot_xi):
        raise ValueError("lambda_dot_xi must be finite and nonnegative")
    if derivative_scale <= 0.0 or not np.isfinite(derivative_scale):
        raise ValueError("derivative_scale must be finite and positive")
    if lambda_dot_xi == 0.0:
        prediction = model(standardized_inputs)
        current = torch.mean((prediction - standardized_targets) ** 2)
        return current, None, current

    physical_s = physical_s_values.clone().detach().to(dtype=torch.float64).requires_grad_(True)
    standardized_s = ((physical_s - float(preprocessing.input_mean[3])) / float(preprocessing.input_std[3])).to(dtype=torch.float32)
    inputs = torch.cat((standardized_inputs[:, :3], standardized_s[:, None]), dim=1)
    if not torch.equal(inputs.detach(), standardized_inputs):
        raise RuntimeError("physical-s preprocessing does not reproduce the frozen standardized input")
    prediction = model(inputs)
    current = torch.mean((prediction - standardized_targets) ** 2)
    f_xi = prediction[:, 1].to(dtype=torch.float64) * float(preprocessing.target_std[1]) + float(preprocessing.target_mean[1])
    gate = -S_STAR * torch.expm1(-physical_s / S_STAR)
    xi_hat = xi0.to(dtype=torch.float64) + gate * f_xi
    predicted_dot_xi = torch.autograd.grad(
        xi_hat,
        physical_s,
        grad_outputs=torch.ones_like(xi_hat),
        create_graph=True,
        retain_graph=True,
    )[0]
    normalized_prediction = (predicted_dot_xi - derivative_mean) / derivative_scale
    normalized_exact = (exact_dot_xi.to(dtype=torch.float64) - derivative_mean) / derivative_scale
    derivative = torch.mean((normalized_prediction - normalized_exact) ** 2)
    total = current.to(dtype=torch.float64) + lambda_dot_xi * derivative
    return current, derivative, total


def _gradient_norm(model: nn.Module) -> tuple[float, float]:
    squared = 0.0
    maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().double()
        squared += float(torch.sum(gradient * gradient))
        maximum = max(maximum, float(torch.max(torch.abs(gradient))))
    return math.sqrt(squared), maximum


def train_derivative_treatment_seed(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    preprocessing: HybridPreprocessing,
    training_exact_dot_xi: np.ndarray,
    seed: int,
    lambda_dot_xi: float,
    output_dir: Path,
    *,
    architecture: str = "shared",
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience: int = PATIENCE,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one fixed treatment without altering the legacy optimizer protocol."""

    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported finite-time seed {seed}")
    if lambda_dot_xi not in TREATMENT_LAMBDAS:
        raise ValueError(f"unsupported controlled lambda {lambda_dot_xi}")
    reproducibility = _set_deterministic_seed(seed)
    train_data = derivative_training_dataset(training, preprocessing, training_exact_dot_xi)
    validation_inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(validation, ("x0", "xi0", "E0", "s"))))
    validation_targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(validation, HYBRID_TARGET_COLUMNS)))
    model = build_hybrid_model(architecture).to(dtype=torch.float32)
    expected_parameters = expected_hybrid_parameter_count(architecture)
    if parameter_count(model) != expected_parameters:
        raise RuntimeError(f"{architecture} hybrid parameter count mismatch")
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
    started = perf_counter()
    nan_inf_events = 0
    for epoch in range(1, maximum_epochs + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 10_000_000 + epoch * 10 + 1)
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, drop_last=False)
        model.train()
        current_squared_sum = 0.0
        current_element_count = 0
        derivative_weighted_sum = 0.0
        derivative_row_count = 0
        total_weighted_sum = 0.0
        gradient_norm_sum = 0.0
        maximum_gradient_norm = 0.0
        maximum_absolute_gradient = 0.0
        for batch_inputs, batch_targets, batch_xi0, batch_s, batch_exact_dot in loader:
            current, derivative, total = derivative_loss_components(
                model, preprocessing, batch_inputs, batch_targets, batch_xi0, batch_s,
                batch_exact_dot, lambda_dot_xi,
            )
            if not torch.isfinite(total):
                nan_inf_events += 1
                raise FloatingPointError(f"nonfinite training loss at seed={seed}, lambda={lambda_dot_xi}, epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm, maximum_absolute = _gradient_norm(model)
            if not np.isfinite(gradient_norm) or not np.isfinite(maximum_absolute):
                nan_inf_events += 1
                raise FloatingPointError(f"nonfinite gradient at seed={seed}, lambda={lambda_dot_xi}, epoch={epoch}")
            optimizer.step()
            # Use the forward prediction's exact current loss identity: MSE has two target components.
            batch_rows = int(batch_inputs.shape[0])
            current_squared_sum += float(current.detach()) * batch_rows * 2
            current_element_count += batch_rows * 2
            if derivative is not None:
                derivative_weighted_sum += float(derivative.detach()) * batch_rows
                derivative_row_count += batch_rows
            total_weighted_sum += float(total.detach()) * batch_rows
            gradient_norm_sum += gradient_norm
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            maximum_absolute_gradient = max(maximum_absolute_gradient, maximum_absolute)
        training_current = current_squared_sum / current_element_count
        training_derivative = None if derivative_row_count == 0 else derivative_weighted_sum / derivative_row_count
        training_total = total_weighted_sum / len(train_data)
        model.eval()
        with torch.no_grad():
            validation_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
        measured = orbit_averaged_standardized_mse(validation_prediction, validation_targets.cpu().numpy(), validation["orbit_id"])
        current_validation = float(measured["orbit_averaged_standardized_mse"])
        history.append({
            "epoch": epoch,
            "training_standardized_hybrid_mse": training_current,
            "training_normalized_derivative_mse": training_derivative,
            "training_total_loss": training_total,
            "validation_orbit_averaged_standardized_hybrid_mse": current_validation,
            "validation_row_averaged_standardized_hybrid_mse": float(measured["ordinary_row_averaged_standardized_mse"]),
            "mean_batch_total_gradient_l2": gradient_norm_sum / steps_per_epoch,
            "maximum_batch_total_gradient_l2": maximum_gradient_norm,
            "maximum_absolute_gradient_element": maximum_absolute_gradient,
        })
        if current_validation < best_metric:
            best_metric = current_validation
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            without_improvement = 0
        else:
            without_improvement += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            print(f"derivative seed={seed} lambda={lambda_dot_xi:.3g} epoch={epoch} val={current_validation:.8g} best={best_metric:.8g}", flush=True)
        if without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("derivative treatment produced no checkpoint")
    stopping_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
    restored = orbit_averaged_standardized_mse(restored_prediction, validation_targets.cpu().numpy(), validation["orbit_id"])
    if float(restored["orbit_averaged_standardized_mse"]) != best_metric:
        raise RuntimeError("restored derivative-treatment checkpoint does not reproduce best metric")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
    return {
        "seed": seed,
        "lambda_dot_xi": lambda_dot_xi,
        "architecture_id": architecture,
        "architecture": (
            "4->64->64->2 with two tanh hidden layers and no output activation"
            if architecture == "shared" else
            "4->64 shared tanh trunk; independent 64->32->1 tanh/linear heads for V_x and F_xi"
        ),
        "parameter_count": parameter_count(model),
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "dtype": "float32",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": 0.0,
        "scheduler": None,
        "loss": "equal two-component standardized MSE on (V_x,F_xi) plus fixed lambda times normalized physical-time dot_xi MSE",
        "derivative_mean": DERIVATIVE_MEAN,
        "derivative_scale": DERIVATIVE_SCALE,
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "checkpoint_selection_metric": "validation orbit-averaged standardized hybrid-target MSE only",
        "best_epoch": best_epoch,
        "stopping_epoch": stopping_epoch,
        "best_validation_orbit_averaged_standardized_hybrid_mse": best_metric,
        "final_training_standardized_hybrid_mse": float(history[-1]["training_standardized_hybrid_mse"]),
        "final_training_normalized_derivative_mse": history[-1]["training_normalized_derivative_mse"],
        "final_validation_orbit_averaged_standardized_hybrid_mse": float(history[-1]["validation_orbit_averaged_standardized_hybrid_mse"]),
        "early_stopping_triggered": stopping_epoch < maximum_epochs,
        "batches_per_epoch": steps_per_epoch,
        "total_optimizer_updates": steps_per_epoch * stopping_epoch,
        "training_wall_seconds": perf_counter() - started,
        "process_peak_rss_mib": peak_mib,
        "maximum_observed_batch_gradient_l2": float(max(row["maximum_batch_total_gradient_l2"] for row in history)),
        "maximum_observed_absolute_gradient_element": float(max(row["maximum_absolute_gradient_element"] for row in history)),
        "nan_inf_events": nan_inf_events,
        "gradient_clipping": False,
        "checkpoint": str(checkpoint.resolve()),
        "history": str(history_path.resolve()),
        "restored_validation": restored,
        "reproducibility": reproducibility,
    }
