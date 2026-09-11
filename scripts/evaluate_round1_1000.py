#!/usr/bin/env python3
"""Compare frozen 500-epoch and controlled 1000-cap Model-A local maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from wormhole_sciml.model_a import (
    TRAINING_SEEDS,
    TREATMENTS,
    Normalization,
    load_trained_model,
    predict_increments,
)
from wormhole_sciml.stage1_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "output" / "stage1_model_a"
OLD_RUN_DIR = PROJECT_ROOT / "output" / "round1_model_a"
NEW_RUN_DIR = PROJECT_ROOT / "output" / "round1_model_a_1000"
OLD_EVALUATION = OLD_RUN_DIR / "evaluation_manifest.json"
GRID_REFERENCE = (
    PROJECT_ROOT / "output" / "round1_local_restart" / "dense_grid_reference.npz"
)
OLD_GRID_ARRAYS = (
    PROJECT_ROOT / "output" / "round1_local_restart" / "checkpoint_local_grids.npz"
)
GRID_MANIFEST = (
    PROJECT_ROOT / "output" / "round1_local_restart" / "diagnostic_manifest.json"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "round1_model_a_1000"
FIGURE_DIR = REPORT_DIR / "figures"
LABELS = {
    "physical_only": "physical-only",
    "collar_0p20": "0.20 collar",
    "collar_0p25": "0.25 collar",
}
COLORS = {
    "physical_only": "#277da1",
    "collar_0p20": "#7b2cbf",
    "collar_0p25": "#2a9d8f",
}
PHYSICAL_METRICS = (
    "standardized_combined_mse",
    "rmse_delta_x",
    "rmse_delta_u",
)
GROUPS = (
    "core",
    "shoulder",
    "edge",
    "central_abs_x_le_8p5",
    "outer_abs_x_gt_8p5",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_metrics(errors: np.ndarray, target_std: np.ndarray) -> dict[str, Any]:
    standardized = errors / target_std
    return {
        "count": int(errors.shape[0]),
        "standardized_combined_mse": float(np.mean(standardized**2)),
        "rmse_delta_x": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "rmse_delta_u": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
    }


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def adjacent_sign_agreement(values: np.ndarray) -> float:
    signs = np.sign(values)
    horizontal = signs[:, 1:] == signs[:, :-1]
    vertical = signs[1:, :] == signs[:-1, :]
    return float(
        (np.sum(horizontal) + np.sum(vertical))
        / (horizontal.size + vertical.size)
    )


def region_masks(x: np.ndarray, xi: np.ndarray) -> dict[str, np.ndarray]:
    absolute_xi = np.abs(xi)
    return {
        "core": absolute_xi < 0.5,
        "shoulder": (absolute_xi >= 0.5) & (absolute_xi < 0.9),
        "edge": (absolute_xi >= 0.9) & (absolute_xi <= 0.99),
        "central_abs_x_le_8p5": np.abs(x) <= 8.5,
        "outer_abs_x_gt_8p5": np.abs(x) > 8.5,
    }


def summarize_grid(
    arrays: dict[str, np.ndarray], grid: dict[str, np.ndarray]
) -> dict[str, Any]:
    summaries = {}
    masks = region_masks(grid["x"], grid["xi"])
    central = masks["central_abs_x_le_8p5"]
    for treatment in TREATMENTS:
        mean_ex = arrays[f"{treatment}__mean_error_x"]
        mean_eu = arrays[f"{treatment}__mean_error_u"]
        mean_e = arrays[f"{treatment}__mean_combined_error"]
        std_e = arrays[f"{treatment}__std_combined_error"]
        maximum_index = np.unravel_index(np.argmax(mean_e), mean_e.shape)
        summaries[treatment] = {
            "combined_error": statistics(mean_e),
            "maximum_location": {
                "x": float(grid["x"][maximum_index]),
                "xi": float(grid["xi"][maximum_index]),
                "mean_error_x": float(mean_ex[maximum_index]),
                "mean_error_u": float(mean_eu[maximum_index]),
            },
            "regions": {name: statistics(mean_e[mask]) for name, mask in masks.items()},
            "seed_spread_combined_error": statistics(std_e),
            "signed_error_u": {
                "adjacent_sign_agreement": adjacent_sign_agreement(mean_eu),
                "mean_absolute": float(np.mean(np.abs(mean_eu))),
                "rms": float(np.sqrt(np.mean(mean_eu**2))),
                "p99_absolute": float(np.quantile(np.abs(mean_eu), 0.99)),
                "central_mean_absolute": float(np.mean(np.abs(mean_eu[central]))),
                "central_rms": float(np.sqrt(np.mean(mean_eu[central] ** 2))),
                "central_p99_absolute": float(
                    np.quantile(np.abs(mean_eu[central]), 0.99)
                ),
            },
        }
    return summaries


def evaluate_physical_validation(
    normalization: Normalization,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    validation = load_dataset(DATA_DIR / "physical_validation.npz")
    states = np.column_stack((validation["x"], validation["u"]))
    targets = np.column_stack((validation["delta_x"], validation["delta_u"]))
    group_masks = {
        "core": validation["stratum"] == 0,
        "shoulder": validation["stratum"] == 1,
        "edge": validation["stratum"] == 2,
        "central_abs_x_le_8p5": np.abs(validation["x"]) <= 8.5,
        "outer_abs_x_gt_8p5": np.abs(validation["x"]) > 8.5,
    }
    rows = []
    arrays = {}
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            run_dir = NEW_RUN_DIR / treatment / f"seed_{seed}"
            metadata = json.loads(
                (run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            model = load_trained_model(run_dir / "best_checkpoint.pt")
            prediction = predict_increments(model, states, normalization)
            error = prediction - targets
            prefix = f"{treatment}__seed_{seed}"
            arrays[f"{prefix}__prediction"] = prediction
            arrays[f"{prefix}__error"] = error
            overall = regression_metrics(error, normalization.target_std)
            groups = {
                name: regression_metrics(error[mask], normalization.target_std)
                for name, mask in group_masks.items()
            }
            rows.append(
                {
                    "treatment": treatment,
                    "seed": seed,
                    "best_epoch": metadata["best_epoch"],
                    "stopping_epoch": metadata["stopping_epoch"],
                    "early_stopping_triggered": metadata[
                        "early_stopping_triggered"
                    ],
                    "physical_one_step": {"overall": overall, "groups": groups},
                }
            )
    return rows, arrays


def evaluate_new_grid(
    normalization: Normalization,
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    shape = grid["x"].shape
    states = np.column_stack((grid["x"].ravel(), grid["u"].ravel()))
    target = np.column_stack(
        (
            grid["delta_x_reference"].ravel(),
            grid["delta_u_reference"].ravel(),
        )
    )
    arrays: dict[str, np.ndarray] = {}
    by_treatment: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        treatment: [] for treatment in TREATMENTS
    }
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            model = load_trained_model(
                NEW_RUN_DIR / treatment / f"seed_{seed}" / "best_checkpoint.pt"
            )
            prediction = predict_increments(model, states, normalization)
            error = prediction - target
            combined = np.sqrt(
                (error[:, 0] / normalization.target_std[0]) ** 2
                + (error[:, 1] / normalization.target_std[1]) ** 2
            )
            prefix = f"{treatment}__seed_{seed}"
            arrays[f"{prefix}__delta_x_prediction"] = prediction[:, 0].reshape(shape)
            arrays[f"{prefix}__delta_u_prediction"] = prediction[:, 1].reshape(shape)
            arrays[f"{prefix}__error_x"] = error[:, 0].reshape(shape)
            arrays[f"{prefix}__error_u"] = error[:, 1].reshape(shape)
            arrays[f"{prefix}__combined_error"] = combined.reshape(shape)
            by_treatment[treatment].append(
                (
                    error[:, 0].reshape(shape),
                    error[:, 1].reshape(shape),
                    combined.reshape(shape),
                )
            )
    for treatment, seed_values in by_treatment.items():
        for key, index in (
            ("error_x", 0),
            ("error_u", 1),
            ("combined_error", 2),
        ):
            stack = np.stack([value[index] for value in seed_values])
            arrays[f"{treatment}__mean_{key}"] = np.mean(stack, axis=0)
            arrays[f"{treatment}__std_{key}"] = np.std(stack, axis=0, ddof=1)
    for key, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"nonfinite extended-grid array: {key}")
    return arrays


def aggregate_physical(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for treatment in TREATMENTS:
        selected = [row for row in rows if row["treatment"] == treatment]
        treatment_summary: dict[str, Any] = {}
        for scope, group in (("overall", None), *((name, name) for name in GROUPS)):
            metrics = {}
            for metric in PHYSICAL_METRICS:
                values = []
                for row in selected:
                    source = row["physical_one_step"]["overall"]
                    if group is not None:
                        source = row["physical_one_step"]["groups"][group]
                    values.append(float(source[metric]))
                metrics[metric] = {
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "values_by_seed": {
                        str(row["seed"]): value
                        for row, value in zip(selected, values, strict=True)
                    },
                }
            treatment_summary[scope] = metrics
        output[treatment] = treatment_summary
    return output


def old_physical_rows() -> list[dict[str, Any]]:
    payload = json.loads(OLD_EVALUATION.read_text(encoding="utf-8"))
    return [
        {
            "treatment": row["treatment"],
            "seed": row["seed"],
            "physical_one_step": row["physical_one_step"],
        }
        for row in payload["runs"]
    ]


def relative_percent(new: float, old: float) -> float:
    return float(100.0 * (new - old) / old)


def make_training_figures() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row_index, treatment in enumerate(TREATMENTS):
        old_histories = []
        new_histories = []
        for seed in TRAINING_SEEDS:
            old_history = json.loads(
                (OLD_RUN_DIR / treatment / f"seed_{seed}" / "history.json").read_text(
                    encoding="utf-8"
                )
            )
            new_history = json.loads(
                (NEW_RUN_DIR / treatment / f"seed_{seed}" / "history.json").read_text(
                    encoding="utf-8"
                )
            )
            old_histories.append(old_history)
            new_histories.append(new_history)
        for column, key in enumerate(
            (
                "training_physical_standardized_mse",
                "physical_validation_standardized_mse",
            )
        ):
            ax = axes[row_index, column]
            for seed, history in zip(TRAINING_SEEDS, new_histories, strict=True):
                ax.plot(
                    [item["epoch"] for item in history],
                    [item[key] for item in history],
                    color=COLORS[treatment],
                    alpha=0.24,
                    lw=0.8,
                    label=f"1000-cap seed {seed}" if row_index == 0 else None,
                )
            old_values = np.asarray([[item[key] for item in history] for history in old_histories])
            new_values = np.asarray([[item[key] for item in history] for history in new_histories])
            ax.plot(
                np.arange(1, 501),
                np.mean(old_values, axis=0),
                color="black",
                ls="--",
                lw=1.3,
                label="500-cap seed mean" if row_index == 0 else None,
            )
            ax.plot(
                np.arange(1, new_values.shape[1] + 1),
                np.mean(new_values, axis=0),
                color=COLORS[treatment],
                lw=1.4,
                label="1000-cap seed mean" if row_index == 0 else None,
            )
            ax.axvline(500, color="0.45", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel(f"{LABELS[treatment]}\nstandardized MSE")
    axes[0, 0].set_title("physical training")
    axes[0, 1].set_title("physical validation")
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(
        "Controlled training extension: old prefix and fresh 1000-cap rerun",
        y=1.02,
    )
    fig.savefig(FIGURE_DIR / "training_curve_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row_index, treatment in enumerate(TREATMENTS):
        for column, key in enumerate(
            (
                "physical_validation_rmse_delta_x",
                "physical_validation_rmse_delta_u",
            )
        ):
            ax = axes[row_index, column]
            for seed in TRAINING_SEEDS:
                old_history = json.loads(
                    (OLD_RUN_DIR / treatment / f"seed_{seed}" / "history.json").read_text(
                        encoding="utf-8"
                    )
                )
                new_history = json.loads(
                    (NEW_RUN_DIR / treatment / f"seed_{seed}" / "history.json").read_text(
                        encoding="utf-8"
                    )
                )
                ax.plot(
                    [item["epoch"] for item in old_history],
                    [item[key] for item in old_history],
                    color="black",
                    ls="--",
                    alpha=0.22,
                )
                ax.plot(
                    [item["epoch"] for item in new_history],
                    [item[key] for item in new_history],
                    color=COLORS[treatment],
                    alpha=0.55,
                    label=f"seed {seed}" if row_index == 0 else None,
                )
            ax.axvline(500, color="0.45", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel(f"{LABELS[treatment]}\nphysical-unit RMSE")
    axes[0, 0].set_title(r"validation RMSE $\Delta x$")
    axes[0, 1].set_title(r"validation RMSE $\Delta u$")
    axes[0, 1].legend(fontsize=7)
    fig.suptitle("Componentwise validation histories", y=1.02)
    fig.savefig(FIGURE_DIR / "validation_component_curves.png")
    plt.close(fig)


def make_grid_figures(
    old_arrays: dict[str, np.ndarray], new_arrays: dict[str, np.ndarray]
) -> None:
    extent = (-17.0, 17.0, -0.99, 0.99)
    all_e = np.stack(
        [
            arrays[f"{treatment}__mean_combined_error"]
            for arrays in (old_arrays, new_arrays)
            for treatment in TREATMENTS
        ]
    )
    e_limit = float(np.quantile(all_e, 0.995))
    differences = np.stack(
        [
            new_arrays[f"{treatment}__mean_combined_error"]
            - old_arrays[f"{treatment}__mean_combined_error"]
            for treatment in TREATMENTS
        ]
    )
    difference_limit = float(np.quantile(np.abs(differences), 0.995))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    column_images = [None, None, None]
    for row, treatment in enumerate(TREATMENTS):
        column_images[0] = axes[row, 0].imshow(
            old_arrays[f"{treatment}__mean_combined_error"],
            origin="lower", extent=extent, aspect="auto", cmap="magma",
            vmin=0, vmax=e_limit,
        )
        column_images[1] = axes[row, 1].imshow(
            new_arrays[f"{treatment}__mean_combined_error"],
            origin="lower", extent=extent, aspect="auto", cmap="magma",
            vmin=0, vmax=e_limit,
        )
        column_images[2] = axes[row, 2].imshow(
            differences[row], origin="lower", extent=extent, aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-difference_limit, vmax=difference_limit),
        )
        axes[row, 0].set_ylabel(f"{LABELS[treatment]}\n$\\xi$")
        for column in range(3):
            axes[row, column].set_xlabel("$x$")
    for column, title in enumerate(("500-epoch mean $E$", "extended mean $E$", "extended - 500")):
        axes[0, column].set_title(title)
        fig.colorbar(column_images[column], ax=axes[:, column], shrink=0.84)
    fig.suptitle("Dense physical local-flow error on the reused DOP853 grid", y=1.02)
    fig.savefig(FIGURE_DIR / "local_flow_E_comparison.png")
    plt.close(fig)

    all_eu = np.stack(
        [
            arrays[f"{treatment}__mean_error_u"]
            for arrays in (old_arrays, new_arrays)
            for treatment in TREATMENTS
        ]
    )
    eu_limit = float(np.quantile(np.abs(all_eu), 0.995))
    eu_differences = np.stack(
        [
            new_arrays[f"{treatment}__mean_error_u"]
            - old_arrays[f"{treatment}__mean_error_u"]
            for treatment in TREATMENTS
        ]
    )
    eu_difference_limit = float(np.quantile(np.abs(eu_differences), 0.995))
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    column_images = [None, None, None]
    for row, treatment in enumerate(TREATMENTS):
        column_images[0] = axes[row, 0].imshow(
            old_arrays[f"{treatment}__mean_error_u"],
            origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-eu_limit, vmax=eu_limit),
        )
        column_images[1] = axes[row, 1].imshow(
            new_arrays[f"{treatment}__mean_error_u"],
            origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-eu_limit, vmax=eu_limit),
        )
        column_images[2] = axes[row, 2].imshow(
            eu_differences[row],
            origin="lower", extent=extent, aspect="auto", cmap="coolwarm",
            norm=TwoSlopeNorm(
                vcenter=0.0, vmin=-eu_difference_limit, vmax=eu_difference_limit
            ),
        )
        axes[row, 0].set_ylabel(f"{LABELS[treatment]}\n$\\xi$")
        for column in range(3):
            axes[row, column].set_xlabel("$x$")
    for column, title in enumerate((r"500-epoch mean $e_u$", r"extended mean $e_u$", "extended - 500")):
        axes[0, column].set_title(title)
        fig.colorbar(column_images[column], ax=axes[:, column], shrink=0.84)
    fig.suptitle("Signed local $u$-increment error on the reused DOP853 grid", y=1.02)
    fig.savefig(FIGURE_DIR / "local_flow_eu_comparison.png")
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.6g}"


def make_report(manifest: dict[str, Any]) -> None:
    old_physical = manifest["physical_validation"]["old_aggregate"]
    new_physical = manifest["physical_validation"]["new_aggregate"]
    old_grid = manifest["dense_grid"]["old_summary"]
    new_grid = manifest["dense_grid"]["new_summary"]
    lines = [
        "# Model-A controlled 500-to-1000 epoch extension",
        "",
        "This experiment changes only the maximum epoch count from 500 to 1000. The nine models were freshly initialized from seeds 101, 202, and 303; architecture, optimizer, learning rate, batches, losses, collar coefficient, normalization, data order, patience, and checkpoint selection remained unchanged.",
        "",
        "## Deterministic reproduction and training behavior",
        "",
        "All nine fresh histories reproduce epochs 1--500 exactly: the maximum absolute discrepancy is 0 for physical training MSE, physical validation MSE, and both componentwise validation RMSE fields. Initial-state hashes also match the original runs.",
        "",
        "| treatment | seed | best epoch | stopping epoch | early stop | best validation MSE |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in manifest["training_runs"]:
        lines.append(
            f"| {LABELS[row['treatment']]} | {row['seed']} | {row['best_epoch']} | {row['stopping_epoch']} | {'yes' if row['early_stopping_triggered'] else 'no'} | {fmt(row['best_physical_validation_standardized_mse'])} |"
        )
    lines.extend(
        [
            "",
            "Every run reached epoch 1000, and every best epoch lies between 995 and 1000. Validation error continued to improve materially beyond epoch 500; patience-based early stopping did not trigger, and the curves do not show a completed plateau within this horizon.",
            "",
            "![Training-curve comparison](figures/training_curve_comparison.png)",
            "",
            "![Component validation curves](figures/validation_component_curves.png)",
            "",
            "## Physical one-step validation",
            "",
            "Individual restored-checkpoint results:",
            "",
            "| treatment | seed | old MSE | new MSE | change | old RMSE dx | new RMSE dx | old RMSE du | new RMSE du |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    old_rows = {
        (row["treatment"], row["seed"]): row
        for row in manifest["physical_validation"]["old_individual"]
    }
    for row in manifest["physical_validation"]["new_individual"]:
        old = old_rows[(row["treatment"], row["seed"])]["physical_one_step"]["overall"]
        new = row["physical_one_step"]["overall"]
        lines.append(
            f"| {LABELS[row['treatment']]} | {row['seed']} | {fmt(old['standardized_combined_mse'])} | {fmt(new['standardized_combined_mse'])} | {fmt(relative_percent(new['standardized_combined_mse'], old['standardized_combined_mse']))}% | {fmt(old['rmse_delta_x'])} | {fmt(new['rmse_delta_x'])} | {fmt(old['rmse_delta_u'])} | {fmt(new['rmse_delta_u'])} |"
        )
    lines.extend(
        [
            "",
            "Overall three-seed mean $\\pm$ sample standard deviation:",
            "",
            "| treatment | old MSE | new MSE | old RMSE dx | new RMSE dx | old RMSE du | new RMSE du |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        cells = []
        for metric in PHYSICAL_METRICS:
            old_metric = old_physical[treatment]["overall"][metric]
            new_metric = new_physical[treatment]["overall"][metric]
            cells.extend(
                (
                    f"{fmt(old_metric['mean'])} $\\pm$ {fmt(old_metric['standard_deviation'])}",
                    f"{fmt(new_metric['mean'])} $\\pm$ {fmt(new_metric['standard_deviation'])}",
                )
            )
        lines.append(f"| {LABELS[treatment]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Three-seed means (parentheses give the relative new-minus-old change):",
            "",
            "| treatment | overall MSE old -> new | core | shoulder | edge | central | outer |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        cells = []
        for scope in ("overall", *GROUPS):
            old = old_physical[treatment][scope]["standardized_combined_mse"]["mean"]
            new = new_physical[treatment][scope]["standardized_combined_mse"]["mean"]
            cells.append(f"{fmt(old)} -> {fmt(new)} ({fmt(relative_percent(new, old))}%)")
        lines.append(f"| {LABELS[treatment]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "All reported physical-validation MSE means decrease after the controlled extension. One component-level exception is explicit in the individual table: 0.20-collar seed 101 has a modest increase in $\\Delta x$ RMSE even though its combined MSE and $\\Delta u$ RMSE decrease. Individual-seed values and seed spreads for every component and region are retained in the evaluation manifest.",
            "",
            "## Reused dense physical local-flow map",
            "",
            f"The existing 201 x 201 DOP853 reference was reused without recomputation. Its verified SHA-256 is `{manifest['dense_grid']['reference_sha256']}`.",
            "",
            "| treatment | mean E old -> new | median | p90 | p99 | maximum | new maximum location | mean seed spread old -> new |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for treatment in TREATMENTS:
        old = old_grid[treatment]
        new = new_grid[treatment]
        lines.append(
            f"| {LABELS[treatment]} | {fmt(old['combined_error']['mean'])} -> {fmt(new['combined_error']['mean'])} ({fmt(relative_percent(new['combined_error']['mean'], old['combined_error']['mean']))}%) | {fmt(old['combined_error']['median'])} -> {fmt(new['combined_error']['median'])} | {fmt(old['combined_error']['p90'])} -> {fmt(new['combined_error']['p90'])} | {fmt(old['combined_error']['p99'])} -> {fmt(new['combined_error']['p99'])} | {fmt(old['combined_error']['maximum'])} -> {fmt(new['combined_error']['maximum'])} | ({fmt(new['maximum_location']['x'])}, {fmt(new['maximum_location']['xi'])}) | {fmt(old['seed_spread_combined_error']['mean'])} -> {fmt(new['seed_spread_combined_error']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "| treatment | core mean E | shoulder | edge | central | outer | central RMS signed eu |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for treatment in TREATMENTS:
        old = old_grid[treatment]
        new = new_grid[treatment]
        cells = []
        for region in GROUPS:
            old_value = old["regions"][region]["mean"]
            new_value = new["regions"][region]["mean"]
            cells.append(
                f"{fmt(old_value)} -> {fmt(new_value)} ({fmt(relative_percent(new_value, old_value))}%)"
            )
        old_eu = old["signed_error_u"]["central_rms"]
        new_eu = new["signed_error_u"]["central_rms"]
        cells.append(f"{fmt(old_eu)} -> {fmt(new_eu)} ({fmt(relative_percent(new_eu, old_eu))}%)")
        lines.append(f"| {LABELS[treatment]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The structured central $E$ and signed $e_u$ patterns visibly contract under the common color scales. They are not eliminated: alternating coherent signed regions remain around the central dynamics, but their amplitude and the central mean $E$ decrease for all three treatments. Exact changes are tabulated above rather than used to select a treatment.",
            "",
            "![Combined local-map comparison](figures/local_flow_E_comparison.png)",
            "",
            "![Signed local-u comparison](figures/local_flow_eu_comparison.png)",
            "",
            "No recursive rollout, restart, energy, admissibility, exterior, or restricted-data evaluation was performed. No constraints were introduced, and no treatment or architecture was selected.",
        ]
    )
    (REPORT_DIR / "TRAINING_EXTENSION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    training_manifest_path = NEW_RUN_DIR / "training_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if training_manifest["status"] != "nine_fresh_runs_completed":
        raise RuntimeError("the controlled training extension is incomplete")
    if not training_manifest["history_prefix_gate"]["all_runs_passed"]:
        raise RuntimeError("the deterministic epoch-500 history gate did not pass")

    grid_manifest = json.loads(GRID_MANIFEST.read_text(encoding="utf-8"))
    expected_grid_hash = grid_manifest["artifacts"]["dense_grid_reference"]["sha256"]
    actual_grid_hash = sha256(GRID_REFERENCE)
    if actual_grid_hash != expected_grid_hash:
        raise RuntimeError("the existing dense-grid reference failed its integrity check")
    with np.load(GRID_REFERENCE) as loaded:
        grid = {key: loaded[key] for key in loaded.files}
    if grid["x"].shape != (201, 201) or not np.all(grid["C"] > 0.0):
        raise RuntimeError("the reused dense-grid reference is inconsistent")
    for key, values in grid.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"nonfinite reused grid array: {key}")

    normalization = Normalization.from_stage1(DATA_DIR / "normalization.json")
    new_physical_rows, physical_arrays = evaluate_physical_validation(normalization)
    physical_array_path = NEW_RUN_DIR / "physical_validation_predictions.npz"
    np.savez_compressed(physical_array_path, **physical_arrays)
    old_physical = old_physical_rows()
    old_physical_aggregate = aggregate_physical(old_physical)
    new_physical_aggregate = aggregate_physical(new_physical_rows)

    with np.load(OLD_GRID_ARRAYS) as loaded:
        old_grid_arrays = {key: loaded[key] for key in loaded.files}
    new_grid_arrays = evaluate_new_grid(normalization, grid)
    new_grid_path = NEW_RUN_DIR / "checkpoint_local_grids.npz"
    np.savez_compressed(new_grid_path, **new_grid_arrays)
    old_grid_summary = summarize_grid(old_grid_arrays, grid)
    new_grid_summary = summarize_grid(new_grid_arrays, grid)

    make_training_figures()
    make_grid_figures(old_grid_arrays, new_grid_arrays)

    manifest = {
        "stage": "Round-I controlled 1000-epoch extension evaluation",
        "status": "nine_best_checkpoints_evaluated",
        "scientific_change": {"field": "maximum_epochs", "old": 500, "new": 1000},
        "training_runs": [
            {
                "treatment": row["treatment"],
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "stopping_epoch": row["stopping_epoch"],
                "early_stopping_triggered": row["early_stopping_triggered"],
                "best_physical_validation_standardized_mse": row[
                    "best_physical_validation_standardized_mse"
                ],
                "history_prefix_comparison": row[
                    "deterministic_history_prefix_comparison"
                ],
            }
            for row in training_manifest["runs"]
        ],
        "physical_validation": {
            "state_count": 4000,
            "old_individual": old_physical,
            "new_individual": new_physical_rows,
            "old_aggregate": old_physical_aggregate,
            "new_aggregate": new_physical_aggregate,
            "prediction_artifact": {
                "path": str(physical_array_path),
                "sha256": sha256(physical_array_path),
            },
        },
        "dense_grid": {
            "shape": [201, 201],
            "point_count": 40401,
            "reference_recomputed": False,
            "reference_path": str(GRID_REFERENCE),
            "reference_sha256": actual_grid_hash,
            "old_grid_arrays_path": str(OLD_GRID_ARRAYS),
            "old_grid_arrays_sha256": sha256(OLD_GRID_ARRAYS),
            "new_grid_arrays": {
                "path": str(new_grid_path),
                "sha256": sha256(new_grid_path),
            },
            "old_summary": old_grid_summary,
            "new_summary": new_grid_summary,
        },
        "rollout_evaluation_performed": False,
        "constraints_introduced": False,
        "restricted_data_opened": False,
        "domain_treatment_selected": False,
        "capacity_comparison_started": False,
        "anomalies": [],
    }
    manifest_path = NEW_RUN_DIR / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_report(manifest)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {REPORT_DIR / 'TRAINING_EXTENSION_REPORT.md'}")


if __name__ == "__main__":
    main()
