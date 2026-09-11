"""Minimal Model-A network and Round-I training loop.

The module deliberately contains only the prescribed 2->32->2 tanh model,
physical-training normalization, and the matched three-treatment trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .stage1_data import load_dataset


TRAINING_SEEDS = (101, 202, 303)
TREATMENTS = {
    "physical_only": None,
    "collar_0p20": "exterior_train_0p20",
    "collar_0p25": "exterior_train_0p25",
}

LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 40
PHYSICAL_BATCH_SIZE = 512
COLLAR_BATCH_SIZE = 128
COLLAR_COEFFICIENT = 0.25


@dataclass(frozen=True, slots=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def from_stage1(
        cls,
        path: Path,
        input_columns: tuple[str, ...] = ("x", "u"),
        target_columns: tuple[str, str] = ("delta_x", "delta_u"),
        expected_source_dataset: str = "physical_train",
    ) -> "Normalization":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["source_dataset"] != expected_source_dataset:
            raise ValueError(
                f"normalization must come only from {expected_source_dataset}"
            )
        columns = payload["columns"]
        return cls(
            input_mean=np.asarray(
                tuple(columns[name]["mean"] for name in input_columns), dtype=np.float64
            ),
            input_std=np.asarray(
                tuple(columns[name]["standard_deviation"] for name in input_columns),
                dtype=np.float64,
            ),
            target_mean=np.asarray(
                tuple(columns[name]["mean"] for name in target_columns),
                dtype=np.float64,
            ),
            target_std=np.asarray(
                tuple(columns[name]["standard_deviation"] for name in target_columns),
                dtype=np.float64,
            ),
        )

    def standardize_inputs(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((values - self.input_mean) / self.input_std, dtype=np.float32)

    def standardize_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray((values - self.target_mean) / self.target_std, dtype=np.float32)

    def unstandardize_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values * self.target_std + self.target_mean, dtype=np.float64)


class ModelA(nn.Module):
    """A small tanh increment MLP with configurable hidden dimensions."""

    def __init__(self, input_dim: int = 2, hidden_dimensions: tuple[int, ...] = (32,)) -> None:
        super().__init__()
        if not hidden_dimensions or any(width < 1 for width in hidden_dimensions):
            raise ValueError("Model-A requires positive hidden dimensions")
        self.input_layer = nn.Linear(input_dim, hidden_dimensions[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(hidden_dimensions, hidden_dimensions[1:])
        )
        self.output_layer = nn.Linear(hidden_dimensions[-1], 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, standardized_state: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input_layer(standardized_state))
        for layer in self.hidden_layers:
            hidden = torch.tanh(layer(hidden))
        return self.output_layer(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        values = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(values.tobytes())
    return digest.hexdigest()


def _tensor_dataset(
    values: dict[str, np.ndarray],
    normalization: Normalization,
    input_columns: tuple[str, ...] = ("x", "u"),
    target_columns: tuple[str, str] = ("delta_x", "delta_u"),
    feature_builder: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = (
        np.column_stack(tuple(values[name] for name in input_columns))
        if feature_builder is None
        else feature_builder(values)
    )
    targets = np.column_stack(tuple(values[name] for name in target_columns))
    return (
        torch.from_numpy(normalization.standardize_inputs(states)),
        torch.from_numpy(normalization.standardize_targets(targets)),
    )


def collar_objective(
    physical_prediction: torch.Tensor,
    physical_target: torch.Tensor,
    collar_prediction: torch.Tensor | None = None,
    collar_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    physical_loss = torch.mean((physical_prediction - physical_target) ** 2)
    if collar_prediction is None or collar_target is None:
        return physical_loss, None, physical_loss
    collar_loss = torch.mean((collar_prediction - collar_target) ** 2)
    return physical_loss, collar_loss, physical_loss + COLLAR_COEFFICIENT * collar_loss


def _permutation(count: int, seed: int, epoch: int, stream: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 10_000_000 + epoch * 10 + stream)
    return torch.randperm(count, generator=generator)


def _batch_slices(count: int, batch_size: int) -> list[slice]:
    return [slice(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


@torch.no_grad()
def predict_standardized(
    model: nn.Module, inputs: torch.Tensor, batch_size: int = 4096
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        outputs.append(model(inputs[start : start + batch_size]))
    return torch.cat(outputs, dim=0)


def _validation_metrics(
    model: ModelA,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    target_std: np.ndarray,
    target_columns: tuple[str, str] = ("delta_x", "delta_u"),
) -> dict[str, float]:
    predictions = predict_standardized(model, inputs)
    errors = predictions - targets
    mse = float(torch.mean(errors**2).item())
    component_mse = torch.mean(errors**2, dim=0).cpu().numpy().astype(np.float64)
    rmse = np.sqrt(component_mse) * target_std
    result = {"standardized_mse": mse}
    result.update(
        {f"physical_rmse_{name}": float(rmse[index]) for index, name in enumerate(target_columns)}
    )
    return result


def train_round1_run(
    project_root: Path,
    treatment: str,
    seed: int,
    *,
    progress: bool = True,
    maximum_epochs: int = MAX_EPOCHS,
    output_tree: str = "round1_model_a",
    reference_history_path: Path | None = None,
    reference_prefix_epochs: int = 0,
    deterministic_history_atol: float = 0.0,
    stage_label: str = "Round I (Stage 2A/2B)",
    data_tree: str = "stage1_model_a",
    physical_train_filename: str = "physical_train.npz",
    physical_validation_filename: str = "physical_validation.npz",
    input_columns: tuple[str, ...] = ("x", "u"),
    target_columns: tuple[str, str] = ("delta_x", "delta_u"),
    normalization_source_dataset: str = "physical_train",
    normalization_path: Path | None = None,
    feature_builder: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    compact_history: bool = False,
    hidden_dimensions: tuple[int, ...] = (32,),
    run_directory: Path | None = None,
    history_filename: str = "history.json",
) -> dict[str, Any]:
    """Train one prescribed run and restore/save its physical-validation best."""

    if treatment not in TREATMENTS:
        raise ValueError(f"unsupported Round-I treatment: {treatment}")
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported Round-I seed: {seed}")
    if maximum_epochs < 1:
        raise ValueError("maximum_epochs must be positive")
    if reference_prefix_epochs:
        if reference_history_path is None:
            raise ValueError("a reference history is required for a prefix gate")
        if reference_prefix_epochs > maximum_epochs:
            raise ValueError("reference prefix exceeds the maximum epoch count")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    data_dir = project_root / "output" / data_tree
    output_dir = (
        project_root / "output" / output_tree / treatment / f"seed_{seed}"
        if run_directory is None else run_directory
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = data_dir / "normalization.json" if normalization_path is None else normalization_path
    normalization = Normalization.from_stage1(
        normalization_path,
        input_columns,
        target_columns,
        normalization_source_dataset,
    )
    physical_train = load_dataset(data_dir / physical_train_filename)
    physical_validation = load_dataset(data_dir / physical_validation_filename)
    train_inputs, train_targets = _tensor_dataset(
        physical_train, normalization, input_columns, target_columns, feature_builder
    )
    validation_inputs, validation_targets = _tensor_dataset(
        physical_validation, normalization, input_columns, target_columns, feature_builder
    )
    collar_name = TREATMENTS[treatment]
    collar_inputs = collar_targets = None
    if collar_name is not None:
        collar = load_dataset(data_dir / f"{collar_name}.npz")
        collar_inputs, collar_targets = _tensor_dataset(
            collar, normalization, input_columns, target_columns, feature_builder
        )

    model = ModelA(len(input_columns), hidden_dimensions).to(dtype=torch.float32)
    layer_dimensions = (len(input_columns), *hidden_dimensions, 2)
    expected_parameter_count = sum(
        (source + 1) * target
        for source, target in zip(layer_dimensions, layer_dimensions[1:])
    )
    if parameter_count(model) != expected_parameter_count:
        raise RuntimeError("Model-A parameter count does not match its prescribed shape")
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    physical_slices = _batch_slices(train_inputs.shape[0], PHYSICAL_BATCH_SIZE)
    collar_slices = (
        []
        if collar_inputs is None
        else _batch_slices(collar_inputs.shape[0], COLLAR_BATCH_SIZE)
    )
    if collar_inputs is not None and len(collar_slices) != len(physical_slices):
        raise RuntimeError("physical and collar schedules do not have matched step counts")

    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    target_std = normalization.target_std

    prefix_comparison: dict[str, Any] | None = None
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        physical_order = _permutation(train_inputs.shape[0], seed, epoch, 1)
        collar_order = (
            None
            if collar_inputs is None
            else _permutation(collar_inputs.shape[0], seed, epoch, 2)
        )
        physical_squared = np.zeros(2, dtype=np.float64)
        collar_squared = np.zeros(2, dtype=np.float64)
        physical_count = 0
        collar_count = 0
        for step, physical_slice in enumerate(physical_slices):
            physical_indices = physical_order[physical_slice]
            x_physical = train_inputs[physical_indices]
            y_physical = train_targets[physical_indices]
            prediction_physical = model(x_physical)
            prediction_collar = target_collar = None
            if collar_inputs is not None and collar_order is not None:
                collar_indices = collar_order[collar_slices[step]]
                prediction_collar = model(collar_inputs[collar_indices])
                target_collar = collar_targets[collar_indices]
            physical_loss, collar_loss, objective = collar_objective(
                prediction_physical,
                y_physical,
                prediction_collar,
                target_collar,
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()

            physical_errors = (prediction_physical.detach() - y_physical).cpu().numpy()
            physical_squared += np.sum(physical_errors.astype(np.float64) ** 2, axis=0)
            physical_count += physical_errors.shape[0]
            if collar_loss is not None and prediction_collar is not None and target_collar is not None:
                collar_errors = (prediction_collar.detach() - target_collar).cpu().numpy()
                collar_squared += np.sum(collar_errors.astype(np.float64) ** 2, axis=0)
                collar_count += collar_errors.shape[0]

        physical_component_mse = physical_squared / physical_count
        physical_mse = float(np.mean(physical_component_mse))
        physical_rmse = np.sqrt(physical_component_mse) * target_std
        collar_mse_value = None
        if collar_count:
            collar_component_mse = collar_squared / collar_count
            collar_mse_value = float(np.mean(collar_component_mse))
        total_value = physical_mse + (
            0.0 if collar_mse_value is None else COLLAR_COEFFICIENT * collar_mse_value
        )
        validation = _validation_metrics(
            model, validation_inputs, validation_targets, target_std, target_columns
        )
        history_row = {
            "epoch": epoch,
            "training_physical_standardized_mse": physical_mse,
            "physical_validation_standardized_mse": validation["standardized_mse"],
        }
        if not compact_history:
            history_row["training_collar_standardized_mse"] = collar_mse_value
            history_row["training_total_objective"] = total_value
        for index, name in enumerate(target_columns):
            if not compact_history:
                history_row[f"training_physical_rmse_{name}"] = float(physical_rmse[index])
            history_row[f"physical_validation_rmse_{name}"] = validation[
                f"physical_rmse_{name}"
            ]
        history.append(history_row)

        if reference_prefix_epochs and epoch == reference_prefix_epochs:
            reference_history = json.loads(
                reference_history_path.read_text(encoding="utf-8")
            )
            if len(reference_history) < reference_prefix_epochs:
                raise RuntimeError("reference history is shorter than the prefix gate")
            comparison_fields = [
                "training_physical_standardized_mse",
                "physical_validation_standardized_mse",
                *(f"physical_validation_rmse_{name}" for name in target_columns),
            ]
            maximum_absolute_discrepancy = {}
            for field in comparison_fields:
                discrepancy = max(
                    abs(float(history[index][field]) - float(reference_history[index][field]))
                    for index in range(reference_prefix_epochs)
                )
                maximum_absolute_discrepancy[field] = discrepancy
            prefix_comparison = {
                "reference_history": str(reference_history_path),
                "epochs_compared": reference_prefix_epochs,
                "absolute_tolerance": deterministic_history_atol,
                "maximum_absolute_discrepancy": maximum_absolute_discrepancy,
                "passed": all(
                    value <= deterministic_history_atol
                    for value in maximum_absolute_discrepancy.values()
                ),
            }
            if not prefix_comparison["passed"]:
                raise RuntimeError(
                    "deterministic history prefix does not reproduce the original run: "
                    f"{maximum_absolute_discrepancy}"
                )

        if validation["standardized_mse"] < best_validation:
            best_validation = validation["standardized_mse"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            print(
                f"{treatment} seed={seed} epoch={epoch} "
                f"val_mse={validation['standardized_mse']:.8g} best={best_validation:.8g}",
                flush=True,
            )
        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("training produced no physical-validation checkpoint")
    stopping_epoch = history[-1]["epoch"]
    model.load_state_dict(best_state)
    restored_validation = _validation_metrics(
        model, validation_inputs, validation_targets, target_std, target_columns
    )
    if restored_validation["standardized_mse"] != best_validation:
        raise RuntimeError("restored checkpoint does not match best validation metric")

    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint_path)
    history_path = output_dir / history_filename
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "stage": stage_label,
        "input_columns": list(input_columns),
        "target_columns": list(target_columns),
        "treatment": treatment,
        "collar_dataset": collar_name,
        "seed": seed,
        "architecture": "->".join(str(width) for width in layer_dimensions) + (
            ", one tanh hidden layer" if len(hidden_dimensions) == 1
            else f", {len(hidden_dimensions)} tanh hidden layers"
        ),
        "hidden_dimensions": list(hidden_dimensions),
        "parameter_count": parameter_count(model),
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "dtype": "float32",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": PATIENCE,
        "checkpoint_selection_metric": "physical_validation_standardized_increment_mse_only",
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "collar_batch_size": None if collar_name is None else COLLAR_BATCH_SIZE,
        "collar_coefficient": None if collar_name is None else COLLAR_COEFFICIENT,
        "physical_epoch_steps": len(physical_slices),
        "collar_epoch_steps": len(collar_slices),
        "physical_order_scheme": "torch.randperm seed*10000000 + epoch*10 + 1",
        "collar_order_scheme": "torch.randperm seed*10000000 + epoch*10 + 2",
        "best_epoch": best_epoch,
        "best_physical_validation_standardized_mse": best_validation,
        "stopping_epoch": stopping_epoch,
        "early_stopping_triggered": stopping_epoch < maximum_epochs,
        "best_epoch_at_maximum_limit": best_epoch == maximum_epochs,
        "deterministic_history_prefix_comparison": prefix_comparison,
        "restored_validation": restored_validation,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "compact_history": compact_history,
        "normalization": str(normalization_path),
        "normalization_source": f"{normalization_source_dataset} only",
        "physical_train_dataset": str(data_dir / physical_train_filename),
        "physical_validation_dataset": str(data_dir / physical_validation_filename),
        "sealed_data_used": False,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_trained_model(checkpoint_path: Path) -> ModelA:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden_indices = sorted(
        int(name.split(".")[1]) for name in state
        if name.startswith("hidden_layers.") and name.endswith(".weight")
    )
    hidden_dimensions = (
        int(state["input_layer.weight"].shape[0]),
        *(int(state[f"hidden_layers.{index}.weight"].shape[0]) for index in hidden_indices),
    )
    model = ModelA(int(state["input_layer.weight"].shape[1]), hidden_dimensions).to(dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_increments(
    model: ModelA, states: np.ndarray, normalization: Normalization
) -> np.ndarray:
    standardized = torch.from_numpy(normalization.standardize_inputs(states))
    predictions = predict_standardized(model, standardized).cpu().numpy()
    return normalization.unstandardize_targets(predictions)


def recursive_rollout(
    model: ModelA,
    initial_state: np.ndarray,
    steps: int,
    normalization: Normalization,
) -> np.ndarray:
    """Apply z[n+1] = z[n] + predicted increment without clipping."""

    states = np.empty((steps + 1, 2), dtype=np.float64)
    states[0] = np.asarray(initial_state, dtype=np.float64)
    for index in range(steps):
        increment = predict_increments(model, states[index : index + 1], normalization)[0]
        states[index + 1] = states[index] + increment
    return states
