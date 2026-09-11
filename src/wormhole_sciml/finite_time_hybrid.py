"""Time-rescaled finite-time model.

The original experiment and saved artifacts used the internal name ``hybrid``.
That name is retained in the implementation API so old checkpoints and
manifests remain reproducible.  Public-facing code may use the equivalent
``time_rescaled`` aliases defined at the end of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .finite_time import (
    BATCH_SIZE,
    EXPECTED_PARAMETER_COUNT,
    HIDDEN_DIMENSIONS,
    INPUT_COLUMNS,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PATIENCE,
    TRAINING_SEEDS,
    orbit_averaged_standardized_mse,
    stack_columns,
)
from .finite_time_rate import RatePreprocessing
from .finite_time_xi_gate import construct_saturating_xi_target, saturating_gate
from .model_a import (
    ModelA,
    Normalization,
    load_trained_model,
    parameter_count,
    predict_standardized,
    state_dict_sha256,
)


S_STAR = 5.0
HYBRID_TARGET_COLUMNS = ("V_x", "F_xi")
HYBRID_ARCHITECTURES = ("shared", "split_head")
SHARED_PARAMETER_COUNT = 4610
SPLIT_HEAD_PARAMETER_COUNT = 4546
HYBRID_PREPROCESSING_SCHEMA = "wormhole_finite_time_hybrid_standardization/v1"
HYBRID_PREPROCESSING_IMPLEMENTATION = "wormhole_sciml.finite_time_hybrid/v1"


class SplitHeadFiniteTimeModel(nn.Module):
    """Parameter-count-controlled finite-time MLP with one trunk and two heads."""

    def __init__(
        self,
        input_dim: int = 4,
        trunk_width: int = 64,
        head_width: int = 32,
    ) -> None:
        super().__init__()
        if min(input_dim, trunk_width, head_width) < 1:
            raise ValueError("split-head dimensions must be positive")
        self.shared_trunk = nn.Linear(input_dim, trunk_width)
        self.vx_hidden = nn.Linear(trunk_width, head_width)
        self.vx_output = nn.Linear(head_width, 1)
        self.fxi_hidden = nn.Linear(trunk_width, head_width)
        self.fxi_output = nn.Linear(head_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.shared_trunk,
            self.vx_hidden,
            self.vx_output,
            self.fxi_hidden,
            self.fxi_output,
        ):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, standardized_state: torch.Tensor) -> torch.Tensor:
        shared = torch.tanh(self.shared_trunk(standardized_state))
        vx = self.vx_output(torch.tanh(self.vx_hidden(shared)))
        fxi = self.fxi_output(torch.tanh(self.fxi_hidden(shared)))
        return torch.cat((vx, fxi), dim=1)

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Expose the scientific gradient topology for explicit audits."""

        return {
            "shared_trunk": tuple(self.shared_trunk.parameters()),
            "x_head": tuple(self.vx_hidden.parameters()) + tuple(self.vx_output.parameters()),
            "xi_head": tuple(self.fxi_hidden.parameters()) + tuple(self.fxi_output.parameters()),
        }


def build_hybrid_model(architecture: str = "shared") -> nn.Module:
    """Construct one of the two controlled finite-time architectures."""

    if architecture == "shared":
        return ModelA(4, HIDDEN_DIMENSIONS)
    if architecture == "split_head":
        return SplitHeadFiniteTimeModel(4, trunk_width=64, head_width=32)
    raise ValueError(f"unsupported hybrid architecture {architecture!r}; choose from {HYBRID_ARCHITECTURES}")


def expected_hybrid_parameter_count(architecture: str) -> int:
    if architecture == "shared":
        return SHARED_PARAMETER_COUNT
    if architecture == "split_head":
        return SPLIT_HEAD_PARAMETER_COUNT
    raise ValueError(f"unsupported hybrid architecture {architecture!r}; choose from {HYBRID_ARCHITECTURES}")


def construct_hybrid_targets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add exact ``V_x`` and ``F_xi`` targets without division at identity rows."""

    elapsed = np.asarray(data["s"], dtype=np.float64)
    if np.any(elapsed < 0.0):
        raise ValueError("physical elapsed time cannot be negative")
    positive = elapsed > 0.0
    rate_x = np.empty_like(elapsed)
    np.divide(np.asarray(data["Delta_x"], dtype=np.float64), elapsed, out=rate_x, where=positive)
    rate_x[~positive] = np.asarray(data["u0"], dtype=np.float64)[~positive]
    f_xi = construct_saturating_xi_target(data, S_STAR)
    if not np.all(np.isfinite(rate_x)) or not np.all(np.isfinite(f_xi)):
        raise FloatingPointError("hybrid targets contain NaN or Inf")
    return {**data, "V_x": rate_x, "F_xi": f_xi}


@dataclass(frozen=True, slots=True)
class HybridPreprocessing:
    """Frozen inputs, verified reused V_x constants, and new training-only F_xi constants."""

    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    source_training_sha256: str
    source_rate_preprocessing_sha256: str

    @classmethod
    def fit(
        cls,
        training: dict[str, np.ndarray],
        rate_preprocessing: RatePreprocessing,
        source_training_sha256: str,
        source_rate_preprocessing_sha256: str,
        *,
        vx_array_identity_verified: bool,
    ) -> "HybridPreprocessing":
        if not vx_array_identity_verified:
            raise ValueError("V_x array identity must be verified before constants are reused")
        targets = stack_columns(training, HYBRID_TARGET_COLUMNS)
        target_mean = np.mean(targets, axis=0, dtype=np.float64)
        target_std = np.std(targets, axis=0, ddof=0, dtype=np.float64)
        vx_mean, f_xi_mean = (float(value) for value in target_mean)
        vx_std, f_xi_std = (float(value) for value in target_std)
        if vx_mean != float(rate_preprocessing.target_mean[0]) or vx_std != float(rate_preprocessing.target_std[0]):
            raise RuntimeError("reconstructed V_x constants differ from the frozen rate experiment")
        if not np.isfinite(f_xi_std) or f_xi_std <= 0.0:
            raise ValueError("F_xi training standard deviation must be finite and positive")
        return cls(
            rate_preprocessing.input_mean.copy(),
            rate_preprocessing.input_std.copy(),
            np.asarray([rate_preprocessing.target_mean[0], f_xi_mean], dtype=np.float64),
            np.asarray([rate_preprocessing.target_std[0], f_xi_std], dtype=np.float64),
            source_training_sha256,
            source_rate_preprocessing_sha256,
        )

    @classmethod
    def from_json(cls, path: Path) -> "HybridPreprocessing":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema"] != HYBRID_PREPROCESSING_SCHEMA:
            raise ValueError("unsupported hybrid preprocessing schema")
        if tuple(payload["input_order"]) != INPUT_COLUMNS:
            raise ValueError("hybrid input ordering mismatch")
        if tuple(payload["target_order"]) != HYBRID_TARGET_COLUMNS:
            raise ValueError("hybrid target ordering mismatch")
        columns = payload["columns"]
        return cls(
            np.asarray([columns[name]["mean"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in INPUT_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["mean"] for name in HYBRID_TARGET_COLUMNS], dtype=np.float64),
            np.asarray([columns[name]["standard_deviation"] for name in HYBRID_TARGET_COLUMNS], dtype=np.float64),
            str(payload["source_training_dataset_sha256"]),
            str(payload["source_rate_preprocessing_sha256"]),
        )

    def as_normalization(self) -> Normalization:
        return Normalization(
            self.input_mean.copy(), self.input_std.copy(),
            self.target_mean.copy(), self.target_std.copy(),
        )

    def standardize_inputs(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().standardize_inputs(values)

    def standardize_targets(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().standardize_targets(values)

    def unstandardize_targets(self, values: np.ndarray) -> np.ndarray:
        return self.as_normalization().unstandardize_targets(values)

    def payload(self, source_path: Path, source_rows: int) -> dict[str, Any]:
        names = INPUT_COLUMNS + HYBRID_TARGET_COLUMNS
        means = np.concatenate((self.input_mean, self.target_mean))
        stds = np.concatenate((self.input_std, self.target_std))
        return {
            "schema": HYBRID_PREPROCESSING_SCHEMA,
            "implementation": HYBRID_PREPROCESSING_IMPLEMENTATION,
            "source_split": "train only",
            "source_training_dataset": str(source_path.resolve()),
            "source_training_dataset_sha256": self.source_training_sha256,
            "source_training_row_count": int(source_rows),
            "source_rate_preprocessing_sha256": self.source_rate_preprocessing_sha256,
            "input_constants_policy": "copied exactly from frozen finite-time preprocessing",
            "V_x_constants_policy": "reused after exact array and mean/std identity checks",
            "F_xi_constants_policy": "new population mean/std fitted on frozen training rows only",
            "input_order": list(INPUT_COLUMNS),
            "target_order": list(HYBRID_TARGET_COLUMNS),
            "statistics_dtype": "float64",
            "model_tensor_dtype": "float32",
            "standard_deviation_definition": "population (ddof=0)",
            "physical_output_gates": {
                "Delta_x": "physical s * V_x",
                "Delta_xi": "-5*expm1(-physical s/5) * F_xi",
            },
            "columns": {
                name: {"mean": float(mean), "standard_deviation": float(std)}
                for name, mean, std in zip(names, means, stds)
            },
        }


def tensor_dataset(data: dict[str, np.ndarray], preprocessing: HybridPreprocessing) -> TensorDataset:
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, INPUT_COLUMNS)))
    targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(data, HYBRID_TARGET_COLUMNS)))
    return TensorDataset(inputs, targets)


def _set_deterministic_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python_random_seed": seed,
        "numpy_legacy_seed": seed,
        "torch_manual_seed": seed,
        "torch_deterministic_algorithms": True,
        "torch_num_threads": 1,
        "dataloader_num_workers": 0,
    }


@torch.no_grad()
def predict_hybrid(
    model: nn.Module, preprocessing: HybridPreprocessing, data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, INPUT_COLUMNS)))
    standardized = predict_standardized(model, inputs, batch_size=8192).cpu().numpy()
    outputs = preprocessing.unstandardize_targets(standardized)
    elapsed = np.asarray(data["s"], dtype=np.float64)
    xi_gate = saturating_gate(elapsed, S_STAR)
    delta_x = elapsed * outputs[:, 0]
    delta_xi = xi_gate * outputs[:, 1]
    return {
        "standardized_V_x": standardized[:, 0],
        "standardized_F_xi": standardized[:, 1],
        "predicted_V_x": outputs[:, 0],
        "predicted_F_xi": outputs[:, 1],
        "xi_gate": xi_gate,
        "predicted_Delta_x": delta_x,
        "predicted_Delta_xi": delta_xi,
        "predicted_x1": np.asarray(data["x0"], dtype=np.float64) + delta_x,
        "predicted_xi1": np.asarray(data["xi0"], dtype=np.float64) + delta_xi,
    }


def train_hybrid_seed(
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    preprocessing: HybridPreprocessing,
    seed: int,
    output_dir: Path,
    *,
    maximum_epochs: int = MAXIMUM_EPOCHS,
    patience: int = PATIENCE,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one prescribed seed and select only by orbit-averaged validation loss."""

    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported finite-time seed {seed}")
    reproducibility = _set_deterministic_seed(seed)
    train_data = tensor_dataset(training, preprocessing)
    validation_data = tensor_dataset(validation, preprocessing)
    validation_inputs, validation_targets = validation_data.tensors
    model = ModelA(4, HIDDEN_DIMENSIONS).to(dtype=torch.float32)
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("4->64->64->2 parameter count mismatch")
    initial_hash = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    without_improvement = 0
    history: list[dict[str, float | int]] = []
    steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
    for epoch in range(1, maximum_epochs + 1):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed * 10_000_000 + epoch * 10 + 1)
        loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
            num_workers=0, drop_last=False,
        )
        model.train()
        squared_sum = 0.0
        element_count = 0
        for batch_inputs, batch_targets in loader:
            prediction = model(batch_inputs)
            loss = torch.mean((prediction - batch_targets) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            error = prediction.detach() - batch_targets
            squared_sum += float(torch.sum(error.double() ** 2).item())
            element_count += int(error.numel())
        training_loss = squared_sum / element_count
        model.eval()
        with torch.no_grad():
            validation_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
        measured = orbit_averaged_standardized_mse(
            validation_prediction, validation_targets.cpu().numpy(), validation["orbit_id"]
        )
        current = float(measured["orbit_averaged_standardized_mse"])
        history.append({
            "epoch": epoch,
            "training_standardized_hybrid_mse": training_loss,
            "validation_orbit_averaged_standardized_hybrid_mse": current,
            "validation_row_averaged_standardized_hybrid_mse": float(
                measured["ordinary_row_averaged_standardized_mse"]
            ),
        })
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            without_improvement = 0
        else:
            without_improvement += 1
        if progress and (epoch == 1 or epoch % 25 == 0):
            print(
                f"hybrid seed={seed} epoch={epoch} val={current:.8g} best={best_metric:.8g}",
                flush=True,
            )
        if without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("hybrid training produced no checkpoint")
    stopping_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored_prediction = predict_standardized(model, validation_inputs, 8192).cpu().numpy()
    restored = orbit_averaged_standardized_mse(
        restored_prediction, validation_targets.cpu().numpy(), validation["orbit_id"]
    )
    if float(restored["orbit_averaged_standardized_mse"]) != best_metric:
        raise RuntimeError("restored hybrid checkpoint does not reproduce best metric")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_checkpoint.pt"
    torch.save(best_state, checkpoint)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "seed": seed,
        "architecture": "4->64->64->2 with two tanh hidden layers and no output activation",
        "parameter_count": parameter_count(model),
        "initialization": "Xavier-uniform weights, zero biases",
        "initial_state_sha256": initial_hash,
        "dtype": "float32",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": 0.0,
        "scheduler": None,
        "loss": "equal two-component standardized MSE on V_x and F_xi",
        "maximum_epochs": maximum_epochs,
        "early_stopping_patience": patience,
        "checkpoint_selection_metric": "validation orbit-averaged standardized hybrid-target MSE only",
        "best_epoch": best_epoch,
        "stopping_epoch": stopping_epoch,
        "best_validation_orbit_averaged_standardized_hybrid_mse": best_metric,
        "final_training_standardized_hybrid_mse": float(history[-1]["training_standardized_hybrid_mse"]),
        "final_validation_orbit_averaged_standardized_hybrid_mse": float(
            history[-1]["validation_orbit_averaged_standardized_hybrid_mse"]
        ),
        "early_stopping_triggered": stopping_epoch < maximum_epochs,
        "batches_per_epoch": steps_per_epoch,
        "total_optimizer_updates": steps_per_epoch * stopping_epoch,
        "restored_validation": restored,
        "checkpoint": str(checkpoint.resolve()),
        "history": str(history_path.resolve()),
        "reproducibility": reproducibility,
        "sealed_test_predictions_computed": False,
    }


def load_hybrid_model(path: Path) -> nn.Module:
    """Load legacy shared or new split-head checkpoints by state-dict schema."""

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "shared_trunk.weight" not in state:
        return load_trained_model(path)
    required = {
        "shared_trunk.weight", "vx_hidden.weight", "vx_output.weight",
        "fxi_hidden.weight", "fxi_output.weight",
    }
    if not required.issubset(state):
        raise ValueError("unrecognized incomplete split-head checkpoint")
    model = SplitHeadFiniteTimeModel(
        input_dim=int(state["shared_trunk.weight"].shape[1]),
        trunk_width=int(state["shared_trunk.weight"].shape[0]),
        head_width=int(state["vx_hidden.weight"].shape[0]),
    ).to(dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    return model


# Public scientific terminology.  The legacy names above remain authoritative
# for saved experiment manifests and checkpoint-loading compatibility.
TIME_RESCALED_TARGET_COLUMNS = HYBRID_TARGET_COLUMNS
TIME_RESCALED_ARCHITECTURES = HYBRID_ARCHITECTURES
TimeRescaledPreprocessing = HybridPreprocessing
build_time_rescaled_model = build_hybrid_model
expected_time_rescaled_parameter_count = expected_hybrid_parameter_count
construct_time_rescaled_targets = construct_hybrid_targets
predict_time_rescaled = predict_hybrid
train_time_rescaled_seed = train_hybrid_seed
load_time_rescaled_model = load_hybrid_model
