#!/usr/bin/env python3
"""Prepare and analyze the transformed-coordinate outer-x random datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a_x_xi_outer_data import (
    TRAIN_SEED,
    VALIDATION_SEED,
    X_COMPONENTS,
    integrity_statistics,
    integrate_transformed_targets,
    sample_outer_enriched_states,
    transform_old_control,
)
from wormhole_sciml.stage1_data import (
    STRATUM_NAMES,
    array_content_sha256,
    file_sha256,
    load_dataset,
    save_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "output" / "stage1_model_a"
OLD_PATHS = {
    "train": STAGE1 / "physical_train.npz",
    "validation": STAGE1 / "physical_validation.npz",
}
STAGE1_MANIFEST = STAGE1 / "manifest.json"
EXACT_ARRAYS = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
EXACT_SUMMARY = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_summary.json"
OUTPUT = ROOT / "output" / "model_a_x_xi_outer_sampling"
FIGURES = OUTPUT / "figures"
DATASET_SUMMARY = OUTPUT / "transformed_dataset_summary.json"
RESOLUTION_SUMMARY = OUTPUT / "sampling_resolution_summary.json"
RESOLUTION_ARRAYS = OUTPUT / "sampling_resolution_arrays.npz"
REPORT = OUTPUT / "TRANSFORMED_SAMPLING_REPORT.md"
U_TH = (0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.90)
HARD = U_TH[:3]
K = 5
WINDOW = 0.25
REPRESENTATIVE_X = (-17.0, -15.0, -12.0, -10.0, -8.5, -5.0, -2.0)
X_BANDS = (-17.0, -14.0, -11.0, -8.5, -5.0, 0.0, 5.0, 8.5, 11.0, 14.0, 17.0)


def stem(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=0)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "p99p9": float(np.quantile(values, 0.999)),
    }


def count_summary(data: dict[str, np.ndarray], new_design: bool) -> dict[str, Any]:
    x = data["x"]
    result: dict[str, Any] = {
        "total": int(x.size),
        "central_abs_x_le_8p5": int(np.sum(np.abs(x) <= 8.5)),
        "outer_abs_x_gt_8p5": int(np.sum(np.abs(x) > 8.5)),
        "outer_left_x_lt_minus_8p5": int(np.sum(x < -8.5)),
        "outer_right_x_gt_plus_8p5": int(np.sum(x > 8.5)),
        "x_bands": {},
        "outer_left_strata": {},
    }
    for low, high in zip(X_BANDS[:-1], X_BANDS[1:]):
        mask = (x >= low) & (x < high if high < 17.0 else x <= high)
        result["x_bands"][f"[{low:g},{high:g}{')' if high < 17 else ']'}"] = int(np.sum(mask))
    outer_left = x < -8.5
    result["outer_left_strata"] = {
        name: int(np.sum(outer_left & (data["stratum"] == label)))
        for label, name in enumerate(STRATUM_NAMES)
    }
    if new_design:
        result["x_component_quotas"] = {
            name: int(np.sum(data["x_component"] == component))
            for component, (name, _low, _high) in enumerate(X_COMPONENTS)
        }
        result["strata_by_x_component"] = {
            name: {
                stratum_name: int(np.sum((data["x_component"] == component) & (data["stratum"] == label)))
                for label, stratum_name in enumerate(STRATUM_NAMES)
            }
            for component, (name, _low, _high) in enumerate(X_COMPONENTS)
        }
        result["signs_by_x_component_and_stratum"] = {
            name: {
                stratum_name: {
                    "negative": int(np.sum((data["x_component"] == component) & (data["stratum"] == label) & (data["xi_sign"] == -1))),
                    "positive": int(np.sum((data["x_component"] == component) & (data["stratum"] == label) & (data["xi_sign"] == 1))),
                }
                for label, stratum_name in enumerate(STRATUM_NAMES)
            }
            for component, (name, _low, _high) in enumerate(X_COMPONENTS)
        }
    return result


def normalization(data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "source_row_count": int(data["x"].size),
        "standard_deviation_definition": "population (ddof=0)",
        "columns": {name: stats(data[name]) for name in ("x", "xi", "E0", "delta_x", "delta_xi")},
    }


def delta_xi_structure(data: dict[str, np.ndarray]) -> dict[str, Any]:
    values = np.abs(data["delta_xi"])
    masks = {
        "central_abs_x_le_8p5": np.abs(data["x"]) <= 8.5,
        "outer_abs_x_gt_8p5": np.abs(data["x"]) > 8.5,
        **{name: data["stratum"] == label for label, name in enumerate(STRATUM_NAMES)},
    }
    return {
        "absolute_distribution": stats(values),
        "near_zero_fractions": {
            "abs_lt_1e-6": float(np.mean(values < 1e-6)),
            "abs_lt_1e-5": float(np.mean(values < 1e-5)),
            "abs_lt_1e-4": float(np.mean(values < 1e-4)),
        },
        "regions": {name: stats(values[mask]) for name, mask in masks.items()},
    }


def local_spacing(train_x: np.ndarray, train_xi: np.ndarray, qx: np.ndarray, qxi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(train_x)
    sx, sxi = train_x[order], train_xi[order]
    nearest = np.empty(qx.size)
    mean_k = np.empty(qx.size)
    counts = np.empty(qx.size, dtype=np.int64)
    for index, (x, xi) in enumerate(zip(qx, qxi)):
        left = np.searchsorted(sx, x - WINDOW, side="left")
        right = np.searchsorted(sx, x + WINDOW, side="right")
        differences = np.abs(sxi[left:right] - xi)
        counts[index] = differences.size
        if differences.size == 0:
            nearest[index] = mean_k[index] = np.nan
        else:
            take = min(K, differences.size)
            selected = np.partition(differences, take - 1)[:take]
            nearest[index], mean_k[index] = float(np.min(selected)), float(np.mean(selected))
    return nearest, mean_k, counts


def load_exact() -> dict[float, dict[str, np.ndarray]]:
    result = {}
    with np.load(EXACT_ARRAYS, allow_pickle=False) as stored:
        for value in U_TH:
            name = stem(value)
            state = np.asarray(stored[f"{name}__exact_state"], dtype=np.float64)
            xi = np.asarray(stored[f"{name}__exact_xi"], dtype=np.float64)
            if np.any(np.diff(state[:, 0]) <= 0.0) or state[0, 0] != -17.0 or state[-1, 0] >= 0.0:
                raise RuntimeError("frozen incoming exact reference has unexpected x ordering")
            result[value] = {"x": state[:, 0], "xi": xi}
    return result


def orbit_spacing(exact: dict[float, dict[str, np.ndarray]], family: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xi_values = np.stack([np.interp(x, exact[value]["x"], exact[value]["xi"]) for value in U_TH])
    family_index = U_TH.index(family)
    differences = np.abs(xi_values - xi_values[family_index])
    differences[family_index] = np.inf
    return xi_values[family_index], np.min(differences, axis=0)


def resolution_analysis(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = load_exact()
    families: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for value in U_TH:
        name = stem(value)
        x = exact[value]["x"]
        xi, separation = orbit_spacing(exact, value, x)
        old_nearest, old_k, old_counts = local_spacing(old["x"], old["xi"], x, xi)
        new_nearest, new_k, new_counts = local_spacing(new["x"], new["xi"], x, xi)
        arrays.update({
            f"{name}__x": x, f"{name}__xi": xi, f"{name}__orbit_spacing": separation,
            f"{name}__old_nearest_dxi": old_nearest, f"{name}__new_nearest_dxi": new_nearest,
            f"{name}__old_mean5_dxi": old_k, f"{name}__new_mean5_dxi": new_k,
            f"{name}__old_resolution_ratio": old_nearest / separation,
            f"{name}__new_resolution_ratio": new_nearest / separation,
            f"{name}__old_window_count": old_counts, f"{name}__new_window_count": new_counts,
        })
        family_summary: dict[str, Any] = {
            "point_count": int(x.size),
            "minimum_old_window_count": int(np.min(old_counts)),
            "minimum_new_window_count": int(np.min(new_counts)),
            "old_nearest_dxi": stats(old_nearest), "new_nearest_dxi": stats(new_nearest),
            "old_mean5_dxi": stats(old_k), "new_mean5_dxi": stats(new_k),
            "orbit_spacing": stats(separation),
            "old_resolution_ratio": stats(old_nearest / separation),
            "new_resolution_ratio": stats(new_nearest / separation),
            "representative_positions": [],
        }
        for position in REPRESENTATIVE_X:
            qx = np.asarray([position])
            qxi, qsep = orbit_spacing(exact, value, qx)
            on, ok, oc = local_spacing(old["x"], old["xi"], qx, qxi)
            nn, nk, nc = local_spacing(new["x"], new["xi"], qx, qxi)
            family_summary["representative_positions"].append({
                "x": position, "exact_xi": float(qxi[0]), "nearest_orbit_delta_xi": float(qsep[0]),
                "old_nearest_training_delta_xi": float(on[0]), "new_nearest_training_delta_xi": float(nn[0]),
                "old_mean5_training_delta_xi": float(ok[0]), "new_mean5_training_delta_xi": float(nk[0]),
                "old_resolution_ratio": float(on[0] / qsep[0]), "new_resolution_ratio": float(nn[0] / qsep[0]),
                "old_window_count": int(oc[0]), "new_window_count": int(nc[0]),
                "window_note": "one-sided at domain endpoint" if position == -17.0 else None,
            })
        families[f"{value:.2f}"] = family_summary
    hard_mask = lambda name: np.concatenate([arrays[f"{stem(v)}__{name}"] for v in HARD])
    hard_outer = lambda name: np.concatenate([
        arrays[f"{stem(v)}__{name}"][arrays[f"{stem(v)}__x"] <= -8.5] for v in HARD
    ])
    old_ratio, new_ratio = hard_mask("old_resolution_ratio"), hard_mask("new_resolution_ratio")
    old_outer_ratio = hard_outer("old_resolution_ratio")
    new_outer_ratio = hard_outer("new_resolution_ratio")
    summary = {
        "local_x_window_half_width": WINDOW, "small_k": K,
        "exact_reference": {"path": str(EXACT_ARRAYS), "sha256": file_sha256(EXACT_ARRAYS), "NN_rollouts_used": False},
        "families": families,
        "hard_three_aggregate": {
            "old_nearest_dxi": stats(hard_mask("old_nearest_dxi")),
            "new_nearest_dxi": stats(hard_mask("new_nearest_dxi")),
            "old_mean5_dxi": stats(hard_mask("old_mean5_dxi")),
            "new_mean5_dxi": stats(hard_mask("new_mean5_dxi")),
            "old_resolution_ratio": stats(old_ratio), "new_resolution_ratio": stats(new_ratio),
            "median_nearest_dxi_reduction_percent": float(100 * (1 - np.median(hard_mask("new_nearest_dxi")) / np.median(hard_mask("old_nearest_dxi")))),
            "median_resolution_ratio_reduction_percent": float(100 * (1 - np.median(new_ratio) / np.median(old_ratio))),
        },
        "hard_three_outer_left_aggregate": {
            "x_range": "-17<=x<=-8.5",
            "old_nearest_dxi": stats(hard_outer("old_nearest_dxi")),
            "new_nearest_dxi": stats(hard_outer("new_nearest_dxi")),
            "old_mean5_dxi": stats(hard_outer("old_mean5_dxi")),
            "new_mean5_dxi": stats(hard_outer("new_mean5_dxi")),
            "old_resolution_ratio": stats(old_outer_ratio),
            "new_resolution_ratio": stats(new_outer_ratio),
            "old_fraction_resolution_ratio_lt_1": float(np.mean(old_outer_ratio < 1.0)),
            "new_fraction_resolution_ratio_lt_1": float(np.mean(new_outer_ratio < 1.0)),
            "median_nearest_dxi_reduction_percent": float(100 * (1 - np.median(hard_outer("new_nearest_dxi")) / np.median(hard_outer("old_nearest_dxi")))),
            "median_resolution_ratio_reduction_percent": float(100 * (1 - np.median(new_outer_ratio) / np.median(old_outer_ratio))),
        },
    }
    return summary, arrays


def plot_distributions(old: dict[str, np.ndarray], new: dict[str, np.ndarray]) -> list[Path]:
    paths = []
    bins_x = np.linspace(-17, 17, 69)
    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    ax.hist(old["x"], bins=bins_x, histtype="step", lw=2, label="old 20k")
    ax.hist(new["x"], bins=bins_x, histtype="step", lw=2, label="new outer-enriched 40k")
    ax.axvline(-8.5, color="0.35", lw=1); ax.axvline(8.5, color="0.35", lw=1)
    ax.set(xlabel="$x$", ylabel="training count per 0.5-wide bin", title="Random x-sampling density")
    ax.legend(); ax.grid(alpha=.2)
    path = FIGURES / "x_distribution_comparison.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    bins_xi = np.linspace(-0.99, 0.99, 81)
    ax.hist(old["xi"], bins=bins_xi, density=True, histtype="step", lw=2, label="old 20k")
    ax.hist(new["xi"], bins=bins_xi, density=True, histtype="step", lw=2, label="new 40k")
    for boundary in (-.9, -.5, .5, .9): ax.axvline(boundary, color="0.6", lw=.8)
    ax.set(xlabel=r"$\xi$", ylabel="probability density", title=r"Unchanged random $\xi$ stratification")
    ax.legend(); ax.grid(alpha=.2)
    path = FIGURES / "xi_distribution_comparison.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True, constrained_layout=True)
    for ax, data, title in zip(axes, (old, new), ("old random 20k", "new outer-enriched random 40k")):
        mask = data["x"] < -8.5
        ax.scatter(data["x"][mask], data["xi"][mask], s=2.5, alpha=.28, linewidths=0)
        ax.set(title=f"{title} · n={np.sum(mask):,}", xlabel="$x$")
        ax.grid(alpha=.15)
    axes[0].set_ylabel(r"$\xi$")
    path = FIGURES / "outer_left_sampling_comparison.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def plot_resolution(arrays: dict[str, np.ndarray]) -> list[Path]:
    paths = []
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 9.2), sharex=True, constrained_layout=True)
    for ax, value in zip(axes, HARD):
        name = stem(value); x = arrays[f"{name}__x"]
        ax.plot(x, arrays[f"{name}__old_resolution_ratio"], lw=1.5, label="old 20k")
        ax.plot(x, arrays[f"{name}__new_resolution_ratio"], lw=1.5, label="new 40k")
        ax.axhline(1, color="black", lw=.8); ax.set_yscale("log")
        ax.set(ylabel=r"$d_\xi/\Delta\xi_{orbit}$", title=rf"$u_{{th}}={value:.2f}$")
        ax.grid(alpha=.2, which="both"); ax.legend()
    axes[-1].set_xlabel("exact incoming $x$")
    path = FIGURES / "hard_trajectory_resolution_ratio.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=True)
    for value in U_TH:
        name = stem(value)
        ax.plot(arrays[f"{name}__x"], arrays[f"{name}__orbit_spacing"], lw=1.25, label=rf"$u_{{th}}={value:.2f}$")
    ax.set_yscale("log"); ax.set(xlabel="exact incoming $x$", ylabel=r"nearest-family $\Delta\xi_{orbit}$", title="Compression of the seven controlled orbit families")
    ax.legend(ncol=2); ax.grid(alpha=.2, which="both")
    path = FIGURES / "orbit_spacing.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def plot_delta_xi(data: dict[str, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    abs_values = np.abs(data["delta_xi"])
    positive = abs_values[abs_values > 0]
    axes[0].hist(positive, bins=np.geomspace(np.min(positive), np.max(positive), 90), histtype="step", lw=1.8)
    axes[0].set_xscale("log"); axes[0].set(xlabel=r"$|\Delta\xi|$", ylabel="count", title=r"New-data $|\Delta\xi|$ distribution")
    bins = np.linspace(-17, 17, 35); centers = .5 * (bins[:-1] + bins[1:])
    med, p99 = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        values = abs_values[(data["x"] >= low) & (data["x"] < high)]
        med.append(np.median(values)); p99.append(np.quantile(values, .99))
    axes[1].plot(centers, med, label="median")
    axes[1].plot(centers, p99, label="p99")
    axes[1].set_yscale("log"); axes[1].set(xlabel="$x$", ylabel=r"$|\Delta\xi|$", title=r"Target scale versus $x$")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=.2, which="both")
    path = FIGURES / "delta_xi_structure.png"; fig.savefig(path, dpi=180); plt.close(fig)
    return path


def report_text(dataset: dict[str, Any], resolution: dict[str, Any]) -> str:
    old = dataset["sampling_counts"]["old20k_train"]
    new = dataset["sampling_counts"]["outer40k_train"]
    hard = resolution["hard_three_aggregate"]
    hard_outer = resolution["hard_three_outer_left_aggregate"]
    rows = []
    for family in HARD:
        for point in resolution["families"][f"{family:.2f}"]["representative_positions"]:
            if point["x"] <= -8.5:
                rows.append(
                    f"| {family:.2f} | {point['x']:.1f} | {point['exact_xi']:.7f} | "
                    f"{point['nearest_orbit_delta_xi']:.3e} | {point['old_nearest_training_delta_xi']:.3e} | "
                    f"{point['new_nearest_training_delta_xi']:.3e} | {point['old_resolution_ratio']:.2f} | {point['new_resolution_ratio']:.2f} |"
                )
    dx = dataset["delta_xi_structure"]["outer40k_train"]
    outer_factor = new["outer_abs_x_gt_8p5"] / old["outer_abs_x_gt_8p5"]
    left_factor = new["outer_left_x_lt_minus_8p5"] / old["outer_left_x_lt_minus_8p5"]
    x_rows = "\n".join(
        f"| `{band}` | {old['x_bands'][band]:,} | {new['x_bands'][band]:,} |"
        for band in old["x_bands"]
    )
    stratum_rows = "\n".join(
        f"| {name} | {old['outer_left_strata'][name]:,} | {new['outer_left_strata'][name]:,} |"
        for name in STRATUM_NAMES
    )
    return f"""# Transformed-coordinate outer-sampling analysis

## Scope and integrity

This task generated data and diagnostics only. No model was trained. The old control rows were transformed row-for-row without reintegration; the new rows were independently and continuously sampled, then integrated with the canonical DOP853 physical flow for `h=0.2`. No sealed/test artifact, collar, diagnostic-trajectory sample, clipping, projection, or loss modification was used.

All four datasets pass the physical, finite-array, reconstruction, and established relative-energy gate (`1e-9`). Source Stage-1 train/validation artifacts and frozen incoming references retained identical hashes.

## Sampling change

| training dataset | total | `|x|<=8.5` | `|x|>8.5` | `x<-8.5` | `x>8.5` |
|---|---:|---:|---:|---:|---:|
| old control | {old['total']:,} | {old['central_abs_x_le_8p5']:,} | {old['outer_abs_x_gt_8p5']:,} | {old['outer_left_x_lt_minus_8p5']:,} | {old['outer_right_x_gt_plus_8p5']:,} |
| new outer-enriched | {new['total']:,} | {new['central_abs_x_le_8p5']:,} | {new['outer_abs_x_gt_8p5']:,} | {new['outer_left_x_lt_minus_8p5']:,} | {new['outer_right_x_gt_plus_8p5']:,} |

Outer coverage increased by `{outer_factor:.2f}x`; specifically, outer-left coverage increased by `{left_factor:.2f}x`. Every new training component contains exactly 3,000/3,500/3,500 core/shoulder/edge rows and exactly balanced signs; validation contains 600/700/700 per component. Because the same stratum fractions and continuous uniform-within-stratum construction were retained, the normalized xi distribution remains comparable to the old design.

| x band | old 20k | new 40k |
|---|---:|---:|
{x_rows}

| outer-left xi stratum | old 20k | new 40k |
|---|---:|---:|
{stratum_rows}

## Incoming-family resolution

Using `|x_train-x_q|<=0.25` and `k=5`, the hard-family median nearest-xi spacing changes from `{hard['old_nearest_dxi']['median']:.3e}` to `{hard['new_nearest_dxi']['median']:.3e}` (`{hard['median_nearest_dxi_reduction_percent']:.1f}%` reduction); mean-5 spacing changes from `{hard['old_mean5_dxi']['median']:.3e}` to `{hard['new_mean5_dxi']['median']:.3e}`. The median controlled-family resolution ratio changes from `{hard['old_resolution_ratio']['median']:.3f}` to `{hard['new_resolution_ratio']['median']:.3f}` (`{hard['median_resolution_ratio_reduction_percent']:.1f}%` reduction).

Specifically over `-17<=x<=-8.5`, median nearest spacing changes from `{hard_outer['old_nearest_dxi']['median']:.3e}` to `{hard_outer['new_nearest_dxi']['median']:.3e}` (`{hard_outer['median_nearest_dxi_reduction_percent']:.1f}%` reduction), and median resolution ratio changes from `{hard_outer['old_resolution_ratio']['median']:.3f}` to `{hard_outer['new_resolution_ratio']['median']:.3f}`. The fraction of hard-family states with `R<1` rises from `{hard_outer['old_fraction_resolution_ratio_lt_1']:.1%}` to `{hard_outer['new_fraction_resolution_ratio_lt_1']:.1%}`. Individual representative points can worsen under independent random sampling, so the improvement is substantial statistically, not pointwise guaranteed.

| u_th | x | exact xi | neighboring-orbit Δxi | old nearest dxi | new nearest dxi | old R | new R |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

At `x=-17`, the fixed window is naturally one-sided. No window had fewer than five training points; exact minimum counts are retained in the JSON. The seven-family spacing is interpreted only as a controlled diagnostic, not a universal orbit-continuum spacing.

## Delta-xi target structure

For new training data, `|Delta xi|` has median `{dx['absolute_distribution']['median']:.3e}`, p99 `{dx['absolute_distribution']['p99']:.3e}`, and maximum `{dx['absolute_distribution']['maximum']:.3e}`. Fractions below `1e-6`, `1e-5`, and `1e-4` are `{dx['near_zero_fractions']['abs_lt_1e-6']:.3%}`, `{dx['near_zero_fractions']['abs_lt_1e-5']:.3%}`, and `{dx['near_zero_fractions']['abs_lt_1e-4']:.3%}`. The central and outer absolute-target medians are `{dx['regions']['central_abs_x_le_8p5']['median']:.3e}` and `{dx['regions']['outer_abs_x_gt_8p5']['median']:.3e}`; core/shoulder/edge medians are `{dx['regions']['core']['median']:.3e}`, `{dx['regions']['shoulder']['median']:.3e}`, and `{dx['regions']['edge']['median']:.3e}`. Thus target scale depends strongly on x and moderately on xi stratum. The target is finite and learnable with ordinary supervised regression; its rare near-zero values make naive relative error ill-defined, but no modified loss is proposed here.

## Future controlled study (record only)

The intended 2x2 study is old-20k versus new-40k sampling, crossed with `(x,xi)->(Delta x,Delta xi)` (`2->32->32->2`) versus `(x,xi,E0)->(Delta x,Delta xi)` (`3->32->32->2`). No one of these four models was trained in this task.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    protected = [*OLD_PATHS.values(), STAGE1_MANIFEST, EXACT_ARRAYS, EXACT_SUMMARY]
    before = {str(path): file_sha256(path) for path in protected}
    old_sources = {name: load_dataset(path) for name, path in OLD_PATHS.items()}
    old = {name: transform_old_control(source) for name, source in old_sources.items()}
    sampled_train = sample_outer_enriched_states(40_000, TRAIN_SEED)
    sampled_validation = sample_outer_enriched_states(8_000, VALIDATION_SEED)
    new_train = integrate_transformed_targets(sampled_train, print)
    new_validation = integrate_transformed_targets(sampled_validation, print)

    datasets = {
        "old20k_train": old["train"], "old4k_validation": old["validation"],
        "outer40k_train": new_train, "outer8k_validation": new_validation,
    }
    integrity = {name: integrity_statistics(data) for name, data in datasets.items()}
    if not np.array_equal(old["train"]["x"], old_sources["train"]["x"]) or not np.array_equal(old["validation"]["x"], old_sources["validation"]["x"]):
        raise RuntimeError("old-control row identity failed")

    OUTPUT.mkdir(parents=True); FIGURES.mkdir()
    filenames = {
        "old20k_train": "old20k_train_x_xi.npz", "old4k_validation": "old4k_validation_x_xi.npz",
        "outer40k_train": "outer40k_train_x_xi.npz", "outer8k_validation": "outer8k_validation_x_xi.npz",
    }
    artifacts = {name: save_dataset(OUTPUT / filenames[name], data) for name, data in datasets.items()}
    resolution, resolution_arrays = resolution_analysis(old["train"], new_train)
    np.savez_compressed(RESOLUTION_ARRAYS, **resolution_arrays)
    distribution_figures = plot_distributions(old["train"], new_train)
    resolution_figures = plot_resolution(resolution_arrays)
    delta_figure = plot_delta_xi(new_train)
    figures = distribution_figures + resolution_figures + [delta_figure]

    after = {path: file_sha256(Path(path)) for path in before}
    if before != after:
        raise RuntimeError("a protected Stage-1 or exact-reference artifact changed")
    summary = {
        "stage": "transformed-coordinate data generation and outer-x resolution analysis only",
        "seeds": {"outer40k_train": TRAIN_SEED, "outer8k_validation": VALIDATION_SEED},
        "physics": {"b0": 1, "m": 2, "A": -2, "W": 1, "S": 0.25, "h": 0.2, "X": 17, "Xc": 8.5, "solver": "DOP853 production settings"},
        "artifacts": artifacts,
        "sampling_counts": {name: count_summary(data, name.startswith("outer")) for name, data in datasets.items()},
        "integrity": integrity,
        "normalization_statistics_train_only": {"old20k_train": normalization(old["train"]), "outer40k_train": normalization(new_train)},
        "delta_xi_structure": {"old20k_train": delta_xi_structure(old["train"]), "outer40k_train": delta_xi_structure(new_train)},
        "old_control": {"reintegrated": False, "row_identity_exact": True, "source_content_sha256": {name: array_content_sha256(data) for name, data in old_sources.items()}},
        "future_models_recorded_not_trained": {
            "old20k": ["2->32->32->2: (x,xi)->(delta_x,delta_xi)", "3->32->32->2: (x,xi,E0)->(delta_x,delta_xi)"],
            "outer40k": ["2->32->32->2: (x,xi)->(delta_x,delta_xi)", "3->32->32->2: (x,xi,E0)->(delta_x,delta_xi)"],
        },
        "figures": [{"path": str(path), "sha256": file_sha256(path)} for path in figures],
        "protected_hashes_before": before, "protected_hashes_after": after,
        "protocol": {"training_performed": False, "sealed_or_test_data_accessed": False, "collars_used": False, "trajectory_specific_samples_used": False, "loss_changed": False, "clipping_or_projection_used": False},
    }
    DATASET_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resolution["arrays"] = {"path": str(RESOLUTION_ARRAYS), "sha256": file_sha256(RESOLUTION_ARRAYS)}
    RESOLUTION_SUMMARY.write_text(json.dumps(resolution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(report_text(summary, resolution), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
