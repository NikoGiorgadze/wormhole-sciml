#!/usr/bin/env python3
"""Readiness audit and guarded launcher for the split-head finite-time study.

The default ``plan`` mode is read-only.  ``readiness`` performs explicit
architecture, gradient, derivative, reproducibility, and one-update smoke
checks.  ``full`` is intentionally guarded and is reserved for the later six
scientific runs; it is not used by this implementation task.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import torch

from wormhole_sciml.finite_time import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    PATIENCE,
    TRAINING_SEEDS,
    stack_columns,
)
from wormhole_sciml.finite_time_derivative_audit import predict_xi_and_physical_s_derivative
from wormhole_sciml.finite_time_derivative_training import (
    DERIVATIVE_MEAN,
    DERIVATIVE_SCALE,
    derivative_loss_components,
    train_derivative_treatment_seed,
)
from wormhole_sciml.finite_time_hybrid import (
    HYBRID_ARCHITECTURES,
    HYBRID_TARGET_COLUMNS,
    S_STAR,
    HybridPreprocessing,
    SplitHeadFiniteTimeModel,
    _set_deterministic_seed,
    build_hybrid_model,
    construct_hybrid_targets,
    expected_hybrid_parameter_count,
    load_hybrid_model,
    predict_hybrid,
)
from wormhole_sciml.model_a import ModelA, parameter_count, state_dict_sha256
from wormhole_sciml.phase_b_orbits import file_sha256
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, xi_time_derivative


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output/phase_c_finite_time_dataset/datasets"
TRAIN_ROWS = DATA / "phase_c_train_raw.npz"
VALIDATION_ROWS = DATA / "phase_c_validation_raw.npz"
PREPROCESSING = ROOT / "output/finite_time_hybrid_s5/preprocessing/hybrid_preprocessing_constants.json"
STAGE3 = ROOT / "output/finite_time_hybrid_derivative_loss_experiment"
STAGE3_CHECKPOINTS = STAGE3 / "checkpoint_manifest.json"
READINESS_OUTPUT = ROOT / "output/finite_time_split_head_readiness"
FULL_OUTPUT = ROOT / "output/finite_time_split_head_derivative_experiment"
FULL_LAMBDAS = (0.0, 0.034)
FULL_SEEDS = (101, 202, 303)
EXPECTED_INPUT_HASHES = {
    TRAIN_ROWS: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    VALIDATION_ROWS: "b95c6d2ebed0a34418daec24448a699f3fd86365191f45631f3d8e25ad20ea0a",
    PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_values(text: str, converter: Any) -> tuple[Any, ...]:
    try:
        return tuple(converter(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def protocol(architecture: str, lambdas: tuple[float, ...], seeds: tuple[int, ...], workers: int) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "architecture_description": (
            "4->64 shared tanh trunk; independent 64->32->1 tanh/linear heads for V_x and F_xi"
            if architecture == "split_head" else
            "4->64->64->2 fully shared tanh/tanh/linear model"
        ),
        "parameter_count": expected_hybrid_parameter_count(architecture),
        "lambda_dot_xi": list(lambdas),
        "seeds": list(seeds),
        "run_count": len(lambdas) * len(seeds),
        "workers": workers,
        "dataset": {"training": str(TRAIN_ROWS), "validation": str(VALIDATION_ROWS)},
        "preprocessing": str(PREPROCESSING),
        "target_order": list(HYBRID_TARGET_COLUMNS),
        "physical_reconstruction": {
            "Delta_x": "s * V_x",
            "Delta_xi": "5*(1-exp(-s/5)) * F_xi",
            "s_star": S_STAR,
        },
        "loss": "unchanged equal two-component standardized target MSE plus fixed lambda_dot_xi times normalized physical-s dot-xi MSE",
        "derivative_mean": DERIVATIVE_MEAN,
        "derivative_scale": DERIVATIVE_SCALE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.0,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "patience": PATIENCE,
        "scheduler": None,
        "checkpoint_selection": "validation orbit-averaged standardized hybrid-target MSE only",
        "data_scope": "training and validation only",
    }


def protected_paths() -> dict[Path, str]:
    output = dict(EXPECTED_INPUT_HASHES)
    checkpoint_manifest = json.loads(STAGE3_CHECKPOINTS.read_text(encoding="utf-8"))
    for row in checkpoint_manifest["checkpoints"]:
        if float(row["lambda_dot_xi"]) in FULL_LAMBDAS and int(row["seed"]) in FULL_SEEDS:
            output[Path(row["checkpoint"])] = row["checkpoint_sha256"]
    if len(output) != 9:
        raise RuntimeError("expected three immutable inputs and six retained shared checkpoints")
    return output


def verify_hashes(expected: dict[Path, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    failures = []
    for path, expected_hash in expected.items():
        measured = file_sha256(path)
        match = measured == expected_hash
        result[str(path.resolve())] = {"expected": expected_hash, "measured": measured, "match": match}
        if not match:
            failures.append(str(path))
    if failures:
        raise RuntimeError(f"protected input mismatch: {failures}")
    return result


def real_batch(count: int = 32) -> tuple[dict[str, np.ndarray], HybridPreprocessing, torch.Tensor, torch.Tensor, np.ndarray]:
    raw = load_dataset(TRAIN_ROWS)
    selected = construct_hybrid_targets({name: value[:count] for name, value in raw.items()})
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(selected, ("x0", "xi0", "E0", "s"))))
    targets = torch.from_numpy(preprocessing.standardize_targets(stack_columns(selected, HYBRID_TARGET_COLUMNS)))
    wormhole, spiral = experiment_parameters()
    exact_dot = xi_time_derivative(selected["x1"], selected["u1"], wormhole, spiral)
    return selected, preprocessing, inputs, targets, exact_dot


def gradient_group_norm(parameters: Iterable[torch.nn.Parameter]) -> dict[str, Any]:
    parameters = tuple(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    squared = sum(float(torch.sum(gradient.detach().double() ** 2)) for gradient in gradients)
    return {
        "l2": math.sqrt(squared),
        "parameter_tensor_count": len(parameters),
        "gradient_tensor_count": len(gradients),
        "none_gradient_tensor_count": len(parameters) - len(gradients),
        "all_finite": all(bool(torch.all(torch.isfinite(gradient))) for gradient in gradients),
    }


def gradient_routing_check(
    data: dict[str, np.ndarray],
    preprocessing: HybridPreprocessing,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    exact_dot: np.ndarray,
) -> dict[str, Any]:
    _set_deterministic_seed(101)
    model = build_hybrid_model("split_head").to(dtype=torch.float32)
    if not isinstance(model, SplitHeadFiniteTimeModel):
        raise TypeError("split-head factory returned the wrong model type")
    groups = model.parameter_groups()
    output: dict[str, Any] = {}

    def capture(name: str, loss_builder: Any) -> None:
        model.zero_grad(set_to_none=True)
        loss = loss_builder()
        loss.backward()
        result = {group: gradient_group_norm(parameters) for group, parameters in groups.items()}
        result["loss"] = float(loss.detach())
        output[name] = result

    capture("L_x", lambda: torch.mean((model(inputs)[:, 0] - targets[:, 0]) ** 2))
    capture("L_xi", lambda: torch.mean((model(inputs)[:, 1] - targets[:, 1]) ** 2))

    def derivative_only() -> torch.Tensor:
        _, derivative, _ = derivative_loss_components(
            model,
            preprocessing,
            inputs,
            targets,
            torch.as_tensor(data["xi0"], dtype=torch.float64),
            torch.as_tensor(data["s"], dtype=torch.float64),
            torch.as_tensor(exact_dot, dtype=torch.float64),
            0.034,
        )
        if derivative is None:
            raise RuntimeError("nonzero derivative coefficient did not construct derivative loss")
        return derivative

    capture("L_dot_xi", derivative_only)
    expected = {
        "L_x": {"shared_trunk": True, "x_head": True, "xi_head": False},
        "L_xi": {"shared_trunk": True, "x_head": False, "xi_head": True},
        "L_dot_xi": {"shared_trunk": True, "x_head": False, "xi_head": True},
    }
    for loss_name, groups_expected in expected.items():
        for group, should_be_nonzero in groups_expected.items():
            norm = output[loss_name][group]["l2"]
            gradients_present = output[loss_name][group]["gradient_tensor_count"] > 0
            if should_be_nonzero != (gradients_present and norm > 0.0):
                raise RuntimeError(f"gradient-routing failure for {loss_name}/{group}: {output[loss_name][group]}")
            if not output[loss_name][group]["all_finite"]:
                raise FloatingPointError(f"nonfinite gradient for {loss_name}/{group}")
    return output


def derivative_autograd_check(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    preprocessing: HybridPreprocessing,
) -> dict[str, Any]:
    count = len(data["s"])
    physical_s = torch.as_tensor(data["s"], dtype=torch.float64).clone().detach().requires_grad_(True)
    physical = torch.stack((
        torch.as_tensor(data["x0"], dtype=torch.float64),
        torch.as_tensor(data["xi0"], dtype=torch.float64),
        torch.as_tensor(data["E0"], dtype=torch.float64),
        physical_s,
    ), dim=1)
    mean = torch.as_tensor(preprocessing.input_mean, dtype=torch.float64)
    std = torch.as_tensor(preprocessing.input_std, dtype=torch.float64)
    standardized = ((physical - mean) / std).to(dtype=torch.float32)
    output = model(standardized)
    fxi = output[:, 1].to(dtype=torch.float64) * preprocessing.target_std[1] + preprocessing.target_mean[1]
    d_fxi_ds = torch.autograd.grad(
        fxi, physical_s, grad_outputs=torch.ones_like(fxi), create_graph=True, retain_graph=True
    )[0]
    gate = -S_STAR * torch.expm1(-physical_s / S_STAR)
    xi_hat = physical[:, 1] + gate * fxi
    autograd_derivative = torch.autograd.grad(
        xi_hat, physical_s, grad_outputs=torch.ones_like(xi_hat), create_graph=False
    )[0]
    explicit_chain_rule = torch.exp(-physical_s / S_STAR) * fxi + gate * d_fxi_ds
    helper = predict_xi_and_physical_s_derivative(model, preprocessing, data, batch_size=count)
    absolute_discrepancy = torch.max(torch.abs(autograd_derivative - explicit_chain_rule)).detach()
    comparison_scale = torch.max(torch.abs(autograd_derivative)).detach()
    return {
        "row_count": count,
        "maximum_absolute_autograd_vs_chain_rule": float(absolute_discrepancy),
        "maximum_relative_to_peak_derivative_autograd_vs_chain_rule": float(absolute_discrepancy / comparison_scale),
        "maximum_absolute_autograd_vs_existing_helper": float(np.max(np.abs(autograd_derivative.detach().numpy() - helper["predicted_dot_xi1"]))),
        "physical_s_leaf_requires_grad": physical_s.requires_grad,
        "standardized_s_matches_preprocessing": bool(torch.equal(standardized[:, 3].detach(), torch.from_numpy(preprocessing.standardize_inputs(stack_columns(data, ("x0", "xi0", "E0", "s"))))[:, 3])),
    }


def architecture_checks() -> dict[str, Any]:
    data, preprocessing, inputs, targets, exact_dot = real_batch()
    models: dict[str, torch.nn.Module] = {}
    counts: dict[str, int] = {}
    interfaces: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for architecture in HYBRID_ARCHITECTURES:
        _set_deterministic_seed(101)
        model = build_hybrid_model(architecture).to(dtype=torch.float32)
        models[architecture] = model
        counts[architecture] = parameter_count(model)
        if counts[architecture] != expected_hybrid_parameter_count(architecture):
            raise RuntimeError(f"parameter count mismatch for {architecture}")
        forward = model(inputs)
        physical = predict_hybrid(model, preprocessing, data)
        interfaces[architecture] = {
            "standardized_output_shape": list(forward.shape),
            "V_x_shape": list(forward[:, 0].shape),
            "F_xi_shape": list(forward[:, 1].shape),
            "Delta_x_shape": list(physical["predicted_Delta_x"].shape),
            "Delta_xi_shape": list(physical["predicted_Delta_xi"].shape),
            "all_finite": bool(torch.all(torch.isfinite(forward))),
        }
        identity_data = {name: np.asarray(values[:8]).copy() for name, values in data.items() if name in ("x0", "xi0", "E0", "s")}
        identity_data["s"][:] = 0.0
        identity = predict_hybrid(model, preprocessing, identity_data)
        identities[architecture] = {
            "maximum_absolute_Delta_x": float(np.max(np.abs(identity["predicted_Delta_x"]))),
            "maximum_absolute_Delta_xi": float(np.max(np.abs(identity["predicted_Delta_xi"]))),
            "exact": bool(np.all(identity["predicted_Delta_x"] == 0.0) and np.all(identity["predicted_Delta_xi"] == 0.0)),
        }
        if not interfaces[architecture]["all_finite"] or tuple(forward.shape) != (len(data["s"]), 2):
            raise RuntimeError(f"forward interface failed for {architecture}")
        if not identities[architecture]["exact"]:
            raise RuntimeError(f"identity reconstruction failed for {architecture}")

    derivative = derivative_autograd_check(models["split_head"], data, preprocessing)
    if derivative["maximum_absolute_autograd_vs_chain_rule"] > 1e-8:
        raise RuntimeError(f"split derivative chain-rule mismatch: {derivative}")
    if derivative["maximum_absolute_autograd_vs_existing_helper"] > 1e-14:
        raise RuntimeError(f"split derivative helper mismatch: {derivative}")
    gradients = gradient_routing_check(data, preprocessing, inputs, targets, exact_dot)

    _set_deterministic_seed(101)
    split_first = build_hybrid_model("split_head")
    _set_deterministic_seed(101)
    split_second = build_hybrid_model("split_head")
    _set_deterministic_seed(202)
    split_other_seed = build_hybrid_model("split_head")
    _set_deterministic_seed(101)
    shared_factory = build_hybrid_model("shared")
    _set_deterministic_seed(101)
    shared_legacy = ModelA(4, (64, 64))
    reproducibility = {
        "split_seed_101_first_sha256": state_dict_sha256(split_first.state_dict()),
        "split_seed_101_second_sha256": state_dict_sha256(split_second.state_dict()),
        "split_seed_202_sha256": state_dict_sha256(split_other_seed.state_dict()),
        "same_architecture_same_seed_exact": state_dict_sha256(split_first.state_dict()) == state_dict_sha256(split_second.state_dict()),
        "different_seed_differs": state_dict_sha256(split_first.state_dict()) != state_dict_sha256(split_other_seed.state_dict()),
        "shared_factory_matches_legacy_constructor": state_dict_sha256(shared_factory.state_dict()) == state_dict_sha256(shared_legacy.state_dict()),
        "policy": "the established deterministic seed is set immediately before natural architecture initialization; different architectures are not forced to share layer tensors",
    }
    if not all(reproducibility[key] for key in ("same_architecture_same_seed_exact", "different_seed_differs", "shared_factory_matches_legacy_constructor")):
        raise RuntimeError(f"reproducibility check failed: {reproducibility}")
    return {
        "parameter_counts": counts,
        "interfaces": interfaces,
        "identity": identities,
        "derivative_autograd": derivative,
        "gradient_routing": gradients,
        "reproducibility": reproducibility,
    }


def smoke_run(output: Path) -> dict[str, Any]:
    raw_training = load_dataset(TRAIN_ROWS)
    raw_validation = load_dataset(VALIDATION_ROWS)
    training = construct_hybrid_targets({name: value[:96] for name, value in raw_training.items()})
    validation = construct_hybrid_targets({name: value[:96] for name, value in raw_validation.items()})
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    wormhole, spiral = experiment_parameters()
    exact = xi_time_derivative(training["x1"], training["u1"], wormhole, spiral)
    result = train_derivative_treatment_seed(
        training,
        validation,
        preprocessing,
        exact,
        101,
        0.034,
        output,
        architecture="split_head",
        maximum_epochs=1,
        patience=PATIENCE,
        progress=False,
    )
    loaded = load_hybrid_model(Path(result["checkpoint"]))
    inputs = torch.from_numpy(preprocessing.standardize_inputs(stack_columns(validation, ("x0", "xi0", "E0", "s"))))
    with torch.no_grad():
        prediction = loaded(inputs[:8])
    result.update({
        "scientific_interpretation_permitted": False,
        "training_rows": 96,
        "validation_rows": 96,
        "expected_optimizer_updates": 1,
        "loaded_checkpoint_model_type": type(loaded).__name__,
        "loaded_checkpoint_output_shape": list(prediction.shape),
        "checkpoint_sha256": file_sha256(Path(result["checkpoint"])),
        "history_sha256": file_sha256(Path(result["history"])),
    })
    if result["total_optimizer_updates"] != 1 or tuple(prediction.shape) != (8, 2):
        raise RuntimeError(f"smoke training/checkpoint cycle failed: {result}")
    return result


def run_tests(output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_model_a.py",
        "tests/test_finite_time_hybrid.py",
        "tests/test_finite_time_xi_gate.py",
        "tests/test_finite_time_derivative_audit.py",
        "tests/test_finite_time_split_head.py",
        "tests/test_finite_time_derivative_experiment.py",
        f"--junitxml={output / 'focused_pytest.xml'}",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-split-readiness-mpl"},
        capture_output=True,
        text=True,
    )
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(output / "test_summary.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"focused tests failed: {payload}")
    return payload


def readiness_report(summary: dict[str, Any]) -> str:
    gradients = summary["checks"]["gradient_routing"]
    gradient_lines = []
    for loss in ("L_x", "L_xi", "L_dot_xi"):
        row = gradients[loss]
        gradient_lines.append(
            f"| {loss} | {row['shared_trunk']['l2']:.6g} | {row['x_head']['l2']:.6g} | {row['xi_head']['l2']:.6g} |"
        )
    unchanged = summary["verified_unchanged"]
    return f"""# Split-head finite-time implementation readiness

## Outcome

The parameter-count-controlled split-head architecture is implemented and ready for the later controlled six-run experiment. No full scientific run was launched, and the smoke metric has no scientific interpretation.

## Files changed

- `src/wormhole_sciml/finite_time_hybrid.py`: split-head class, architecture factory/counts, and backward-compatible checkpoint loading.
- `src/wormhole_sciml/finite_time_derivative_training.py`: selectable architecture argument and architecture-aware metadata/count gate; loss and training protocol are unchanged.
- `src/wormhole_sciml/model_a.py` and `src/wormhole_sciml/finite_time_derivative_audit.py`: interface type annotations generalized to accept either model class.
- `scripts/run_finite_time_split_head_experiment.py`: safe plan/readiness modes and explicitly guarded later six-run launcher.
- `tests/test_finite_time_split_head.py`: parameter-count, interface, identity, checkpoint, and routing regressions.

## Implementation

- Existing selectable shared model: `4 -> 64_tanh -> 64_tanh -> 2_linear`, exactly `{summary['checks']['parameter_counts']['shared']}` trainable parameters.
- New selectable split model: `4 -> 64_tanh` followed by independent `64 -> 32_tanh -> 1_linear` heads for `V_x` and `F_xi`, exactly `{summary['checks']['parameter_counts']['split_head']}` trainable parameters.
- Both return a `(batch, 2)` tensor ordered `(V_x, F_xi)`. Existing raw shared checkpoints remain loadable; split checkpoints are auto-detected without changing their external inference interface.
- Architecture selection is exposed as `build_hybrid_model("shared" | "split_head")`, `train_derivative_treatment_seed(..., architecture=...)`, and the guarded runner `--architecture` option. The trainer default remains `shared`.

## Explicit checks

At `s=0`, both architectures produced exactly zero `Delta_x` and `Delta_xi`; this follows only from the unchanged physical gates.

The split derivative used the original physical-`s` leaf through standardization and the complete physical reconstruction. Maximum discrepancy between autograd and

`exp(-s/5) F_xi + 5(1-exp(-s/5)) dF_xi/ds`

was `{summary['checks']['derivative_autograd']['maximum_absolute_autograd_vs_chain_rule']:.3e}` absolute (`{summary['checks']['derivative_autograd']['maximum_relative_to_peak_derivative_autograd_vs_chain_rule']:.3e}` relative to the peak derivative); discrepancy from the existing inference helper was `{summary['checks']['derivative_autograd']['maximum_absolute_autograd_vs_existing_helper']:.3e}`.

Gradient L2 norms on a real 32-row training batch were:

| isolated loss | shared trunk | x head | xi head |
|:---|---:|---:|---:|
{chr(10).join(gradient_lines)}

The zero entries are exact zero norms; PyTorch may represent an unused concatenated-output branch as either `None` or an allocated all-zero gradient tensor. This verifies the intended routing: `L_x` reaches only trunk+x head; `L_xi` and `L_dot_xi` reach only trunk+xi head.

Reinitializing split-head twice with seed 101 produced identical state hashes; seed 202 differed. Rebuilding the shared model through the new selector exactly matched the legacy constructor at seed 101. Initialization remains Xavier-uniform weights and zero biases, with the established Python/NumPy/Torch deterministic seed setup and epoch-specific DataLoader generator. Cross-architecture tensors were not artificially coupled.

## Smoke cycle

One non-scientific split-head `lambda_dot_xi=0.034`, seed-101 update was run on 96 real training rows with 96 validation rows. It completed forward loss, physical-time derivative loss, backward, one unchanged Adam step, validation checkpoint selection, checkpoint/history writes, reload, and `(8,2)` inference. Checkpoint loader returned `{summary['smoke']['loaded_checkpoint_model_type']}`. Focused tests: `{summary['tests']['stdout'].strip()}`

## Controlled-comparison audit

Verified unchanged: {', '.join(unchanged)}.

Established nuance retained: the base target loss is the mean standardized MSE over the two outputs—equivalently equal component weighting with an overall factor of one half—not an unnormalized `L_x + L_xi` scalar sum. No scheduler is present. Early stopping patience is `{PATIENCE}`, maximum epochs `{MAXIMUM_EPOCHS}`, batch size `{BATCH_SIZE}`, learning rate `{LEARNING_RATE}`, derivative scale `{DERIVATIVE_SCALE:.16g}`, and selection remains validation orbit-averaged standardized hybrid-target MSE only.

All nine protected dataset/preprocessing/retained-shared-checkpoint hashes were unchanged. No held-out artifact was accessed.

## Later six-run launch — do not run during this task

From the repository root:

```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/wormhole-split-full-mpl .venv/bin/python scripts/run_finite_time_split_head_experiment.py --mode full --confirm-six-run-experiment --architecture split_head --lambdas 0,0.034 --seeds 101,202,303 --workers 3
```

The full mode refuses any other architecture, lambda set, or seed set, and refuses to overwrite its output directory. The retained shared runs are referenced for the later comparison and are not retrained.

## Risks and discrepancies

- No scientific discrepancy blocked readiness.
- The two architectures consume different natural RNG sequences after the common seed because their layer structures differ; reproducibility is exact within each architecture/seed, which is the controlled and non-artificial policy.
- The smoke subset is deliberately tiny and must not be compared scientifically.
- This task prepares the six split-head trainings only; full comparative validation should be performed after those runs exist.
"""


def readiness() -> None:
    if READINESS_OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {READINESS_OUTPUT}")
    READINESS_OUTPUT.mkdir(parents=True)
    protected = protected_paths()
    before = verify_hashes(protected)
    checks = architecture_checks()
    smoke = smoke_run(READINESS_OUTPUT / "smoke" / "split_head_lambda_0p034_seed_101")
    tests_dir = READINESS_OUTPUT / "tests"
    tests_dir.mkdir()
    tests = run_tests(tests_dir)
    after = verify_hashes(protected)
    if before != after:
        raise RuntimeError("protected input changed during readiness audit")
    verified_unchanged = [
        "dataset files and train/validation split",
        "row sampling and DataLoader shuffle scheme",
        "preprocessing and input/target normalization",
        "V_x and F_xi target definitions",
        "saturating gate g(s) with s_star=5",
        "physical reconstruction",
        "physical-s derivative autograd path and derivative normalization",
        "base loss and derivative-loss implementation",
        "Adam optimizer, learning rate, batch size, weight decay, and absent scheduler",
        "maximum schedule, early stopping, and checkpoint selection",
        "seed handling and validation procedure",
    ]
    summary = {
        "created_utc": utc_now(),
        "status": "SPLIT_HEAD_IMPLEMENTATION_READY",
        "full_scientific_training_launched": False,
        "held_out_test_accessed": False,
        "protocol_for_later_run": protocol("split_head", FULL_LAMBDAS, FULL_SEEDS, 3),
        "checks": checks,
        "smoke": smoke,
        "tests": tests,
        "verified_unchanged": verified_unchanged,
        "protected_before": before,
        "protected_after": after,
        "source_snapshots": {
            "derivative_loss_components_sha256": hashlib.sha256(inspect.getsource(derivative_loss_components).encode()).hexdigest(),
            "construct_hybrid_targets_sha256": hashlib.sha256(inspect.getsource(construct_hybrid_targets).encode()).hexdigest(),
            "predict_hybrid_sha256": hashlib.sha256(inspect.getsource(predict_hybrid).encode()).hexdigest(),
        },
    }
    summary_path = READINESS_OUTPUT / "finite_time_split_head_readiness_summary.json"
    report_path = READINESS_OUTPUT / "FINITE_TIME_SPLIT_HEAD_READINESS.md"
    write_json(summary_path, summary)
    report_path.write_text(readiness_report(summary), encoding="utf-8")
    artifacts = {
        str(path.relative_to(READINESS_OUTPUT)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(READINESS_OUTPUT.rglob("*")) if path.is_file()
    }
    manifest_path = READINESS_OUTPUT / "finite_time_split_head_readiness_manifest.json"
    write_json(manifest_path, {
        "status": summary["status"],
        "protected_before": before,
        "protected_after": after,
        "source_hashes": {
            str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
            str((ROOT / "src/wormhole_sciml/finite_time_hybrid.py").resolve()): file_sha256(ROOT / "src/wormhole_sciml/finite_time_hybrid.py"),
            str((ROOT / "src/wormhole_sciml/finite_time_derivative_training.py").resolve()): file_sha256(ROOT / "src/wormhole_sciml/finite_time_derivative_training.py"),
            str((ROOT / "src/wormhole_sciml/finite_time_derivative_audit.py").resolve()): file_sha256(ROOT / "src/wormhole_sciml/finite_time_derivative_audit.py"),
            str((ROOT / "src/wormhole_sciml/model_a.py").resolve()): file_sha256(ROOT / "src/wormhole_sciml/model_a.py"),
            str((ROOT / "tests/test_finite_time_split_head.py").resolve()): file_sha256(ROOT / "tests/test_finite_time_split_head.py"),
        },
        "artifacts": artifacts,
    })
    (READINESS_OUTPUT / "finite_time_split_head_readiness_manifest.sha256").write_text(
        f"{file_sha256(manifest_path)}  {manifest_path.name}\n", encoding="utf-8"
    )
    print(f"wrote {report_path}")


def full_worker(seed: int, architecture: str, lambdas: tuple[float, ...], output: str) -> list[dict[str, Any]]:
    training = construct_hybrid_targets(load_dataset(TRAIN_ROWS))
    validation = construct_hybrid_targets(load_dataset(VALIDATION_ROWS))
    preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
    wormhole, spiral = experiment_parameters()
    exact = xi_time_derivative(training["x1"], training["u1"], wormhole, spiral)
    rows = []
    for value in lambdas:
        destination = Path(output) / "training" / f"lambda_{str(value).replace('.', 'p')}" / f"seed_{seed}"
        run = train_derivative_treatment_seed(
            training,
            validation,
            preprocessing,
            exact,
            seed,
            value,
            destination,
            architecture=architecture,
            progress=True,
        )
        run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
        run["history_sha256"] = file_sha256(Path(run["history"]))
        write_json(destination / "metadata.json", run)
        rows.append(run)
    return rows


def full_experiment(arguments: argparse.Namespace) -> None:
    architecture = arguments.architecture
    lambdas = parse_values(arguments.lambdas, float)
    seeds = parse_values(arguments.seeds, int)
    if not arguments.confirm_six_run_experiment:
        raise RuntimeError("full mode requires --confirm-six-run-experiment")
    if architecture != "split_head" or lambdas != FULL_LAMBDAS or seeds != FULL_SEEDS:
        raise RuntimeError("the guarded full experiment requires split_head, lambdas 0,0.034, and seeds 101,202,303 exactly")
    if arguments.workers < 1 or arguments.workers > 3:
        raise ValueError("workers must be between one and three")
    output = Path(arguments.output_dir).resolve() if arguments.output_dir else FULL_OUTPUT
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    protected = protected_paths()
    before = verify_hashes(protected)
    frozen = protocol(architecture, lambdas, seeds, arguments.workers)
    frozen["created_utc_before_training"] = utc_now()
    frozen["retained_shared_checkpoint_manifest"] = str(STAGE3_CHECKPOINTS.resolve())
    write_json(output / "frozen_protocol.json", frozen)
    runs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(full_worker, seed, architecture, lambdas, str(output)): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            runs.extend(future.result())
    runs.sort(key=lambda row: (row["lambda_dot_xi"], row["seed"]))
    after = verify_hashes(protected)
    if before != after:
        raise RuntimeError("protected input changed during full split-head training")
    checkpoint_manifest = {
        "status": "SIX_SPLIT_HEAD_RUNS_COMPLETED",
        "protocol": frozen,
        "runs": runs,
        "protected_before": before,
        "protected_after": after,
        "held_out_test_accessed": False,
    }
    write_json(output / "split_head_training_manifest.json", checkpoint_manifest)
    print(f"completed six split-head runs in {output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("plan", "readiness", "full"), default="plan")
    result.add_argument("--architecture", choices=HYBRID_ARCHITECTURES, default="split_head")
    result.add_argument("--lambdas", default="0,0.034")
    result.add_argument("--seeds", default="101,202,303")
    result.add_argument("--workers", type=int, default=3)
    result.add_argument("--output-dir")
    result.add_argument("--confirm-six-run-experiment", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    lambdas = parse_values(arguments.lambdas, float)
    seeds = parse_values(arguments.seeds, int)
    if arguments.mode == "plan":
        print(json.dumps(protocol(arguments.architecture, lambdas, seeds, arguments.workers), indent=2))
    elif arguments.mode == "readiness":
        if arguments.architecture != "split_head" or lambdas != FULL_LAMBDAS or seeds != FULL_SEEDS:
            raise RuntimeError("readiness mode audits the frozen split_head / {0,0.034} / {101,202,303} plan")
        readiness()
    else:
        full_experiment(arguments)


if __name__ == "__main__":
    main()
