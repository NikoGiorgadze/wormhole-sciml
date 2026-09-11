#!/usr/bin/env python3
"""Train the four prescribed wider/deeper physical-only Model-A variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a import Normalization, TRAINING_SEEDS, train_round1_run
from wormhole_sciml.physics_gate import experiment_parameters, xi_from_state
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "output" / "stage1_model_a"
OUTPUT_TREE = "model_a_architecture_comparison"
OUTPUT_DIR = ROOT / "output" / OUTPUT_TREE
A_MANIFEST = ROOT / "output" / "round1_model_a_1000" / "training_manifest.json"
C_DIR = ROOT / "output" / "model_a_x_u_xi_physical_1000"
C_MANIFEST = C_DIR / "training_manifest.json"
ARCHITECTURE_MANIFEST = OUTPUT_DIR / "architecture_comparison_manifest.json"
ARCHITECTURES = {
    "a64": {"representation": "x_u", "inputs": ("x", "u"), "hidden": (64,)},
    "a32x32": {"representation": "x_u", "inputs": ("x", "u"), "hidden": (32, 32)},
    "c64": {"representation": "x_u_xi", "inputs": ("x", "u", "xi"), "hidden": (64,)},
    "c32x32": {"representation": "x_u_xi", "inputs": ("x", "u", "xi"), "hidden": (32, 32)},
}
PARAMETER_MATCHED_ARCHITECTURES = {
    "a15x15": {"representation": "x_u", "inputs": ("x", "u"), "hidden": (15, 15)},
    "c16x16": {"representation": "x_u_xi", "inputs": ("x", "u", "xi"), "hidden": (16, 16)},
}
METRICS = ("standardized_mse", "physical_rmse_delta_x", "physical_rmse_delta_u")


def augmented_inputs(values: dict[str, np.ndarray]) -> np.ndarray:
    wormhole, spiral = experiment_parameters()
    return np.column_stack((
        values["x"], values["u"],
        xi_from_state(values["x"], values["u"], wormhole, spiral),
    ))


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in METRICS:
        values = np.asarray([run["restored_validation"][metric] for run in runs])
        result[metric] = {
            "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)),
            "values_by_seed": {str(run["seed"]): float(value) for run, value in zip(runs, values)},
        }
    return result


def stored_baselines() -> dict[str, dict[str, Any]]:
    a_payload = json.loads(A_MANIFEST.read_text(encoding="utf-8"))
    c_payload = json.loads(C_MANIFEST.read_text(encoding="utf-8"))
    a_runs = [row for row in a_payload["runs"] if row["treatment"] == "physical_only"]
    c_runs = c_payload["runs"]
    if [row["seed"] for row in a_runs] != list(TRAINING_SEEDS):
        raise RuntimeError("stored A32 baseline is incomplete")
    if [row["seed"] for row in c_runs] != list(TRAINING_SEEDS):
        raise RuntimeError("stored C32 baseline is incomplete")
    return {
        "a32": {"representation": "x_u", "hidden": [32], "runs": a_runs,
                "aggregate": aggregate(a_runs), "reused": True},
        "c32": {"representation": "x_u_xi", "hidden": [32], "runs": c_runs,
                "aggregate": aggregate(c_runs), "reused": True},
    }


def stored_matched_context() -> dict[str, dict[str, Any]]:
    payload = json.loads(ARCHITECTURE_MANIFEST.read_text(encoding="utf-8"))
    return {key: payload["models"][key] for key in ("a64", "a32x32", "c64", "c32x32")}


def relative(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        metric: 100.0 * (candidate[metric]["mean"] / reference[metric]["mean"] - 1.0)
        for metric in METRICS
    }


def plot_histories(
    models: dict[str, dict[str, Any]], architectures: dict[str, dict[str, Any]], output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, len(architectures), figsize=(5 * len(architectures), 3.8),
                             constrained_layout=True, squeeze=False)
    for axis, key in zip(axes.ravel(), architectures):
        histories = [json.loads(Path(run["history"]).read_text()) for run in models[key]["runs"]]
        common_length = min(len(history) for history in histories)
        train = np.empty((3, common_length))
        validation = np.empty((3, common_length))
        for index, history in enumerate(histories):
            train[index] = [row["training_physical_standardized_mse"] for row in history[:common_length]]
            validation[index] = [row["physical_validation_standardized_mse"] for row in history[:common_length]]
        epoch = np.arange(1, common_length + 1)
        axis.plot(epoch, np.mean(train, axis=0), alpha=0.55, label="training mean")
        axis.plot(epoch, np.mean(validation, axis=0), label="validation mean")
        axis.set(title=key.upper(), xlabel="epoch", ylabel="standardized MSE", yscale="log")
        axis.grid(alpha=0.2)
        axis.legend()
    path = output_dir / "training_validation_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_table(models: dict[str, dict[str, Any]], model_order: tuple[str, ...], output_dir: Path) -> Path:
    path = output_dir / "validation_summary.csv"
    fields = ["model", "representation", "hidden_dimensions", "parameter_count", "summary",
              "standardized_mse", "rmse_delta_x", "rmse_delta_u"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in model_order:
            model = models[key]
            common = {"model": key, "representation": model["representation"],
                      "hidden_dimensions": "x".join(map(str, model["hidden"])),
                      "parameter_count": model["runs"][0]["parameter_count"]}
            for run in model["runs"]:
                restored = run["restored_validation"]
                writer.writerow({**common, "summary": f"seed_{run['seed']}",
                                 "standardized_mse": restored["standardized_mse"],
                                 "rmse_delta_x": restored["physical_rmse_delta_x"],
                                 "rmse_delta_u": restored["physical_rmse_delta_u"]})
            for summary, field in (("mean", "mean"), ("sample_sd", "standard_deviation")):
                writer.writerow({**common, "summary": summary,
                                 "standardized_mse": model["aggregate"]["standardized_mse"][field],
                                 "rmse_delta_x": model["aggregate"]["physical_rmse_delta_x"][field],
                                 "rmse_delta_u": model["aggregate"]["physical_rmse_delta_u"][field]})
    return path


def main(parameter_matched: bool = False) -> None:
    if parameter_matched:
        architectures = PARAMETER_MATCHED_ARCHITECTURES
        output_tree = "model_a_parameter_matched_depth"
        output_dir = ROOT / "output" / output_tree
        models = stored_matched_context()
        references = {"a15x15": "a64", "c16x16": "c64"}
        model_order = ("a64", "a15x15", "a32x32", "c64", "c16x16", "c32x32")
    else:
        architectures, output_tree, output_dir = ARCHITECTURES, OUTPUT_TREE, OUTPUT_DIR
        models = stored_baselines()
        references = {key: "a32" if row["representation"] == "x_u" else "c32"
                      for key, row in architectures.items()}
        model_order = ("a32", "a64", "a32x32", "c32", "c64", "c32x32")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    normalization_paths = {
        "x_u": DATA_DIR / "normalization.json", "x_u_xi": C_DIR / "normalization.json",
    }
    normalizations = {
        "x_u": Normalization.from_stage1(normalization_paths["x_u"]),
        "x_u_xi": Normalization.from_stage1(
            normalization_paths["x_u_xi"], ("x", "u", "xi"), ("delta_x", "delta_u"),
            "physical_train_x_u_xi",
        ),
    }
    if not np.array_equal(normalizations["x_u"].target_mean, normalizations["x_u_xi"].target_mean):
        raise RuntimeError("target means differ across representations")
    if not np.array_equal(normalizations["x_u"].target_std, normalizations["x_u_xi"].target_std):
        raise RuntimeError("target standard deviations differ across representations")
    protected = [DATA_DIR / "physical_train.npz", DATA_DIR / "physical_validation.npz",
                 *normalization_paths.values(), A_MANIFEST, C_MANIFEST]
    if parameter_matched:
        protected.append(ARCHITECTURE_MANIFEST)
    for baseline in models.values():
        protected.extend(Path(run["checkpoint"]) for run in baseline["runs"])
    before = {str(path): file_sha256(path) for path in protected}
    audit_fields = ("optimizer", "learning_rate", "weight_decay", "maximum_epochs",
                    "early_stopping_patience", "checkpoint_selection_metric",
                    "physical_batch_size", "physical_epoch_steps", "physical_order_scheme")
    for key, specification in architectures.items():
        reference = models[references[key]]
        reference_by_seed = {run["seed"]: run for run in reference["runs"]}
        runs = []
        for seed in TRAINING_SEEDS:
            feature_builder = None if specification["representation"] == "x_u" else augmented_inputs
            expected_source = "physical_train" if feature_builder is None else "physical_train_x_u_xi"
            run = train_round1_run(
                ROOT, "physical_only", seed, maximum_epochs=1000,
                output_tree=f"{output_tree}/{key}",
                stage_label=f"Model-A architecture comparison: {key}",
                input_columns=specification["inputs"], target_columns=("delta_x", "delta_u"),
                normalization_source_dataset=expected_source,
                normalization_path=normalization_paths[specification["representation"]],
                feature_builder=feature_builder, compact_history=True,
                hidden_dimensions=specification["hidden"],
            )
            if any(run[field] != reference_by_seed[seed][field] for field in audit_fields):
                raise RuntimeError(f"{key} training protocol differs from its stored baseline")
            if run["collar_dataset"] is not None:
                raise RuntimeError("collar data entered the architecture comparison")
            run["checkpoint_sha256"] = file_sha256(Path(run["checkpoint"]))
            runs.append(run)
        models[key] = {**specification, "hidden": list(specification["hidden"]), "runs": runs,
                       "aggregate": aggregate(runs), "reused": False}
        models[key]["relative_to_reference_percent"] = relative(
            models[key]["aggregate"], reference["aggregate"]
        )
    after = {str(path): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("a physical dataset or stored baseline changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    curve = plot_histories(models, architectures, output_dir)
    table = write_table(models, model_order, output_dir)
    comparisons = (
        {
            "a15x15_relative_to_a64_percent": models["a15x15"]["relative_to_reference_percent"],
            "c16x16_relative_to_c64_percent": models["c16x16"]["relative_to_reference_percent"],
            "a32x32_relative_to_a15x15_percent": relative(models["a32x32"]["aggregate"], models["a15x15"]["aggregate"]),
            "c32x32_relative_to_c16x16_percent": relative(models["c32x32"]["aggregate"], models["c16x16"]["aggregate"]),
        }
        if parameter_matched else {
            "cross_representation_c_relative_to_a_percent": {
                shape: relative(models[c_key]["aggregate"], models[a_key]["aggregate"])
                for shape, a_key, c_key in (("64", "a64", "c64"), ("32x32", "a32x32", "c32x32"))
            }
        }
    )
    manifest = {
        "stage": "Model-A physical-only architecture training and basic validation",
        "status": "six_new_runs_completed" if parameter_matched else "twelve_new_runs_completed",
        "new_run_count": 6 if parameter_matched else 12,
        "models": models, "training_seeds": list(TRAINING_SEEDS),
        "target_normalization_identical_across_all_models": True,
        "comparisons": comparisons,
        "training_curve": str(curve), "validation_table": str(table),
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protocol": {"stored_baselines_retrained": False, "model_unconstrained": True,
                     "collar_data_used": False, "loss_changed": False,
                     "dense_grid_evaluation_performed": False,
                     "rollout_evaluation_performed": False, "restricted_data_accessed": False},
    }
    path = output_dir / (
        "parameter_matched_manifest.json" if parameter_matched else "architecture_comparison_manifest.json"
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-matched", action="store_true")
    main(parser.parse_args().parameter_matched)
