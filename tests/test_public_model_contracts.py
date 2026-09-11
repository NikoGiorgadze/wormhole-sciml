from __future__ import annotations

import numpy as np
import torch

from wormhole_sciml.finite_time_derivative_training import derivative_loss_components
from wormhole_sciml.finite_time_hybrid import (
    HybridPreprocessing,
    TimeRescaledPreprocessing,
    build_hybrid_model,
    build_time_rescaled_model,
    expected_hybrid_parameter_count,
    expected_time_rescaled_parameter_count,
    predict_hybrid,
    predict_time_rescaled,
)
from wormhole_sciml.finite_time_xi_gate import saturating_gate
from wormhole_sciml.model_a import ModelA, parameter_count
from wormhole_sciml.physics_gate import (
    experiment_parameters,
    state_from_xi,
    xi_from_state,
)


def _preprocessing() -> HybridPreprocessing:
    return HybridPreprocessing(
        input_mean=np.zeros(4),
        input_std=np.ones(4),
        target_mean=np.zeros(2),
        target_std=np.ones(2),
        source_training_sha256="synthetic",
        source_rate_preprocessing_sha256="synthetic",
    )


def test_xi_coordinate_round_trip_stays_inside_admissible_phase_space() -> None:
    wormhole, spiral = experiment_parameters()
    x = np.linspace(-20.0, 20.0, 101)
    xi = np.linspace(-0.95, 0.95, 101)
    _, u = state_from_xi(x, xi, wormhole, spiral)
    np.testing.assert_allclose(xi_from_state(x, u, wormhole, spiral), xi, atol=2e-15)


def test_public_model_interfaces_and_parameter_counts() -> None:
    local_model = ModelA(3, (32, 32))
    assert local_model(torch.zeros((5, 3))).shape == (5, 2)
    assert parameter_count(local_model) == 1250

    for architecture in ("shared", "split_head"):
        model = build_hybrid_model(architecture)
        assert model(torch.zeros((5, 4))).shape == (5, 2)
        assert parameter_count(model) == expected_hybrid_parameter_count(architecture)


def test_public_time_rescaled_aliases_preserve_the_checkpoint_contract() -> None:
    assert TimeRescaledPreprocessing is HybridPreprocessing
    for architecture in ("shared", "split_head"):
        model = build_time_rescaled_model(architecture)
        assert parameter_count(model) == expected_time_rescaled_parameter_count(architecture)
        assert expected_time_rescaled_parameter_count(architecture) == expected_hybrid_parameter_count(
            architecture
        )


def test_time_rescaled_model_has_exact_zero_time_identity() -> None:
    model = build_time_rescaled_model("shared")
    data = {
        "x0": np.array([-2.0, 3.0]),
        "xi0": np.array([-0.4, 0.7]),
        "E0": np.array([0.9, 0.95]),
        "s": np.zeros(2),
    }
    prediction = predict_time_rescaled(model, _preprocessing(), data)
    np.testing.assert_array_equal(saturating_gate(data["s"], 5.0), np.zeros(2))
    np.testing.assert_array_equal(prediction["predicted_x1"], data["x0"])
    np.testing.assert_array_equal(prediction["predicted_xi1"], data["xi0"])


def test_derivative_loss_backpropagates_through_physical_time() -> None:
    model = build_hybrid_model("shared")
    standardized_inputs = torch.tensor(
        [[-1.0, -0.3, 0.9, 0.2], [0.5, 0.4, 1.0, 1.2]], dtype=torch.float32
    )
    standardized_targets = torch.zeros((2, 2), dtype=torch.float32)
    current, derivative, total = derivative_loss_components(
        model,
        _preprocessing(),
        standardized_inputs,
        standardized_targets,
        xi0=torch.tensor([-0.3, 0.4], dtype=torch.float64),
        physical_s_values=torch.tensor([0.2, 1.2], dtype=torch.float64),
        exact_dot_xi=torch.zeros(2, dtype=torch.float64),
        lambda_dot_xi=0.034,
    )
    assert current.isfinite()
    assert derivative is not None and derivative.isfinite()
    total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
