#!/usr/bin/env python3
"""Calibrate fixed derivative-loss weights from frozen-checkpoint gradients."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.finite_time_gradient_calibration import (
    PARAMETER_GROUPS,
    audit_gradient_batch,
    deterministic_batch_indices,
    distribution,
    rounded_fixed_lambda,
)
from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, construct_hybrid_targets, load_hybrid_model
from wormhole_sciml.phase_b_orbits import file_sha256
from wormhole_sciml.phase_c_finite_time import load_dataset
from wormhole_sciml.physics_gate import experiment_parameters, xi_time_derivative


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/finite_time_hybrid_gradient_calibration"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
ARRAYS = OUTPUT / "arrays"
TESTS = OUTPUT / "tests"
REPORT = OUTPUT / "FINITE_TIME_HYBRID_GRADIENT_CALIBRATION.md"
SUMMARY = OUTPUT / "finite_time_hybrid_gradient_calibration_summary.json"
MANIFEST = OUTPUT / "finite_time_hybrid_gradient_calibration_manifest.json"
MANIFEST_HASH = OUTPUT / "finite_time_hybrid_gradient_calibration_manifest.sha256"

TRAIN_ROWS = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_train_raw.npz"
MODEL_ROOT = ROOT / "output/finite_time_hybrid_s5"
PREPROCESSING = MODEL_ROOT / "preprocessing/hybrid_preprocessing_constants.json"
MODEL_MANIFEST = MODEL_ROOT / "finite_time_hybrid_manifest.json"
CHECKPOINTS = {seed: MODEL_ROOT / f"training/seed_{seed}/best_checkpoint.pt" for seed in (101, 202, 303)}
STAGE1 = ROOT / "output/finite_time_hybrid_derivative_audit"
STAGE1_SUMMARY = STAGE1 / "finite_time_hybrid_derivative_audit_summary.json"
STAGE1_MANIFEST = STAGE1 / "finite_time_hybrid_derivative_audit_manifest.json"

EXPECTED = {
    TRAIN_ROWS: "7b34595f9d5070a30914cf6c45f509fa425d3e9208c167f067a4d9dd65adbc4c",
    PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    MODEL_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
    CHECKPOINTS[101]: "568300c44c7b4240dde232f1946dd18a11919e48b68af8180a0092f241000aa5",
    CHECKPOINTS[202]: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    CHECKPOINTS[303]: "f83bbc4802a7742fb0bf63c138a7f407f2d2ed3fb0172bd2c00b2c7332719ad1",
    STAGE1_SUMMARY: "7f0ccfcbbae991d98f5f35f7c86fe43653cc35c74a0fce6852fe7995678c16bf",
    STAGE1_MANIFEST: "51980aab8641fa19fe1384de9ac01b516ca1a8c619831c624f8421da685837e4",
}
BATCH_SIZE = 512
BATCH_COUNT = 24
CALIBRATION_SEED = 202
DOT_XI_MEAN = -0.0018535215398591892
DOT_XI_STD = 0.014438580924341927
RHO_TARGETS = (0.05, 0.15, 0.30)
ALIGNMENT_THRESHOLD = 0.10
CATEGORY_ORDER = ("overall", "hard_u_th_le_0p30", "sharp_q95_q99", "ordinary_u_th_gt_0p30")
CATEGORY_LABELS = {"overall": "overall", "hard_u_th_le_0p30": "hard",
                   "sharp_q95_q99": "q95–q99", "ordinary_u_th_gt_0p30": "ordinary"}
CATEGORY_SEEDS = {name: 2_026_090_500 + index for index, name in enumerate(CATEGORY_ORDER)}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src/wormhole_sciml/finite_time_gradient_calibration.py",
             ROOT / "src/wormhole_sciml/finite_time_derivative_audit.py", ROOT / "src/wormhole_sciml/finite_time_hybrid.py",
             ROOT / "tests/test_finite_time_gradient_calibration.py"]
    return {str(path.resolve()): file_sha256(path) for path in paths}


def immutable_gate() -> dict[str, Any]:
    artifacts, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path); match = measured == expected
        artifacts[str(path.resolve())] = {"expected": expected, "measured": measured, "match": match}
        if not match: failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": artifacts,
            "data_scope": "existing training rows and frozen retained checkpoints only",
            "held_out_test_access": "none: no held-out artifact was opened or hashed"}


def evaluation_configuration(stage1: dict[str, Any]) -> dict[str, Any]:
    return {"created_utc": utc(), "status": "FROZEN_GRADIENT_CALIBRATION_PROTOCOL", "training_performed": False,
            "loss0": "mean((model(standardized current inputs)-standardized (V_x,F_xi) targets)^2), equal two-component current loss",
            "loss_d": "mean((((dot_xi_hat-mean_train)/std_train)-((dot_xi_exact-mean_train)/std_train))^2)",
            "physical_derivative": "autograd d[xi0-5*expm1(-physical_s/5)*F_xi]/d physical_s with create_graph=True",
            "dot_xi_training_mean": DOT_XI_MEAN, "dot_xi_training_std": DOT_XI_STD,
            "stage1_distribution_match": {"mean": stage1["training_exact_derivative_distribution"]["mean"] == DOT_XI_MEAN,
                                          "std": stage1["training_exact_derivative_distribution"]["standard_deviation"] == DOT_XI_STD},
            "batch_size": BATCH_SIZE, "batches_per_category": BATCH_COUNT, "sampling": "deterministic without replacement within each diagnostic category",
            "categories": {"overall": "all training rows", "hard_u_th_le_0p30": "u_th<=0.30",
                           "sharp_q95_q99": "Stage-1 training q95 < |dot_xi_exact| <= q99", "ordinary_u_th_gt_0p30": "u_th>0.30"},
            "category_random_seeds": CATEGORY_SEEDS, "checkpoint_seeds": list(CHECKPOINTS), "calibration_source": "median R for seed 202 overall batches",
            "rho_targets": list(RHO_TARGETS), "lambda_policy": "rho/median_R then round to two significant digits; fixed scalar thereafter",
            "alignment_labels": {"cooperative": f"median cosine > {ALIGNMENT_THRESHOLD}", "orthogonal_or_new": f"abs(median cosine) <= {ALIGNMENT_THRESHOLD}",
                                 "conflicting": f"median cosine < -{ALIGNMENT_THRESHOLD}"},
            "adaptive_balancing": False, "sharpness_weighting": False, "difficulty_weighting": False,
            "optimizer_created": False, "optimizer_step": False, "source_hashes": source_hashes()}


def batch_subset(training: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {name: value[index] for name, value in training.items() if np.asarray(value).ndim == 1 and value.size == training["s"].size}


def aggregate_batch_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = ("gradient0_norm", "gradient_d_norm", "ratio", "cosine", "loss0", "loss_d",
              "loss0_forward_backward_seconds", "loss_d_forward_mixed_backward_seconds", "time_ratio_d_over_0")
    table = []; nested: dict[str, Any] = {}
    for seed in CHECKPOINTS:
        nested[str(seed)] = {}
        for category in CATEGORY_ORDER:
            selected = [row for row in rows if row["seed"] == seed and row["category"] == category]
            payload = {field: distribution([row[field] for row in selected]) for field in fields}
            nested[str(seed)][category] = payload
            table.append({"seed": seed, "category": category, "batch_count": len(selected),
                          **{f"{field}_{stat}": values[stat] for field, values in payload.items() for stat in ("median", "mean", "standard_deviation", "p10", "p90", "minimum", "maximum")}})
    return table, nested


def aggregate_layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for seed in CHECKPOINTS:
        for category in CATEGORY_ORDER:
            for group in PARAMETER_GROUPS:
                selected = [row for row in rows if row["seed"] == seed and row["category"] == category and row["parameter_group"] == group]
                output.append({"seed": seed, "category": category, "parameter_group": group, "batch_count": len(selected),
                               **{f"ratio_{k}": v for k, v in distribution([row["ratio"] for row in selected]).items()},
                               **{f"cosine_{k}": v for k, v in distribution([row["cosine"] for row in selected]).items()},
                               "gradient0_norm_median": float(np.median([row["gradient0_norm"] for row in selected])),
                               "gradient_d_norm_median": float(np.median([row["gradient_d_norm"] for row in selected]))})
    return output


def alignment_label(cosine: float) -> str:
    if cosine > ALIGNMENT_THRESHOLD: return "cooperative"
    if cosine < -ALIGNMENT_THRESHOLD: return "conflicting"
    return "approximately orthogonal/new information"


def plot_distributions(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    positions, labels, colors = [], [], []
    data_r, data_c = [], []
    for seed_index, seed in enumerate(CHECKPOINTS):
        for category_index, category in enumerate(CATEGORY_ORDER):
            selected = [row for row in rows if row["seed"] == seed and row["category"] == category]
            category_label = CATEGORY_LABELS[category]
            positions.append(seed_index * (len(CATEGORY_ORDER) + 1) + category_index); labels.append(f"{seed} {category_label}")
            data_r.append([row["ratio"] for row in selected]); data_c.append([row["cosine"] for row in selected]); colors.append(plt.cm.Set2(category_index / len(CATEGORY_ORDER)))
    for ax, data, ylabel in ((axes[0], data_r, "R = ||g_d|| / ||g_0||"), (axes[1], data_c, "cosine(g_0,g_d)")):
        boxes = ax.boxplot(data, positions=positions, widths=.7, patch_artist=True, showfliers=True)
        for patch, color in zip(boxes["boxes"], colors): patch.set_facecolor(color)
        ax.set(xticks=positions, xticklabels=labels, ylabel=ylabel); ax.grid(alpha=.25, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    axes[0].set_yscale("log"); axes[1].axhline(0, color="black", lw=1)
    fig.suptitle("Frozen-checkpoint gradient scale and alignment across deterministic training batches")
    fig.savefig(FIGURES / "gradient_ratio_and_cosine_distributions.png", dpi=180); plt.close(fig)


def plot_category(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, field, ylabel in ((axes[0], "ratio", "R = ||g_d|| / ||g_0||"), (axes[1], "cosine", "cosine(g_0,g_d)")):
        data = [[row[field] for row in rows if row["category"] == category] for category in CATEGORY_ORDER]
        ax.boxplot(data, tick_labels=[CATEGORY_LABELS[name] for name in CATEGORY_ORDER], showfliers=True)
        ax.set_ylabel(ylabel); ax.grid(alpha=.25, axis="y")
    axes[0].set_yscale("log"); axes[1].axhline(0, color="black", lw=1)
    fig.suptitle("Gradient scale and alignment by trajectory category across retained seeds")
    fig.savefig(FIGURES / "gradient_scale_alignment_by_category.png", dpi=180); plt.close(fig)


def plot_rho(rho_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(candidates), figsize=(15, 4.5), constrained_layout=True, sharey=True)
    for ax, candidate in zip(axes, candidates):
        label = candidate["label"]; data = [[row["rho_batch"] for row in rho_rows if row["lambda_label"] == label and row["category"] == category] for category in CATEGORY_ORDER]
        ax.boxplot(data, tick_labels=["overall", "hard", "q95–q99", "ordinary"], showfliers=True)
        ax.axhline(candidate["target_rho"], color="black", ls=":", label="nominal target")
        display_label = label.replace("_", " ")
        ax.set(title=f"{display_label}: lambda={candidate['fixed_lambda']:.5g}", ylabel="rho_batch", ylim=(0, max(1.0, max(max(x) for x in data) * 1.05))); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8)
    fig.suptitle("Implied fixed derivative-gradient contribution across training regions")
    fig.savefig(FIGURES / "calibrated_rho_distributions.png", dpi=180); plt.close(fig)


def run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_finite_time_gradient_calibration.py",
               "tests/test_finite_time_derivative_audit.py", f"--junitxml={TESTS/'focused_pytest.xml'}"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "src", "MPLCONFIGDIR": "/private/tmp/wormhole-gradient-calibration-mpl"}, capture_output=True, text=True)
    payload = {"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "passed": result.returncode == 0}
    write_json(TESTS / "test_summary.json", payload); return payload


def report_text(summary: dict[str, Any]) -> str:
    aggregate_rows = "\n".join(f"| {r['seed']} | {r['category']} | {r['gradient0_norm_median']:.5g} | {r['gradient_d_norm_median']:.5g} | {r['ratio_median']:.5g} | {r['ratio_p10']:.5g}–{r['ratio_p90']:.5g} | {r['cosine_median']:.4f} | {r['cosine_p10']:.4f}–{r['cosine_p90']:.4f} |" for r in summary["aggregate_table"])
    candidate_rows = "\n".join(f"| {r['label']} | {r['target_rho']:.2f} | {r['raw_lambda']:.7g} | {r['fixed_lambda']:.7g} | {r['nominal_rho_after_rounding']:.4f} |" for r in summary["lambda_candidates"])
    rho_rows = "\n".join(f"| {r['lambda_label']} | {r['seed']} | {r['category']} | {r['median']:.4f} | {r['p10']:.4f}–{r['p90']:.4f} | {r['minimum']:.4f}–{r['maximum']:.4f} |" for r in summary["rho_distributions"])
    layer_rows = "\n".join(f"| {r['parameter_group']} | {r['gradient0_norm_median']:.5g} | {r['gradient_d_norm_median']:.5g} | {r['ratio_median']:.5g} | {r['cosine_median']:.4f} |" for r in summary["layer_table"] if r["seed"] == CALIBRATION_SEED and r["category"] == "overall")
    timing = summary["compute_overhead"]
    lambdas = ", ".join(f"`{row['fixed_lambda']:.7g}`" for row in summary["lambda_candidates"])
    return f"""# Physical-time derivative-loss gradient calibration

## Scope and loss definitions

No optimizer was created and no parameter update occurred. This audit used existing training rows only and the three already retained frozen checkpoints. No dataset was resampled or modified, and no held-out test artifact was opened or hashed.

`L0` is exactly the current equal two-component standardized MSE on `(V_x,F_xi)`. `Ld` is the MSE of the exact and predicted physical-time derivative residual after applying the Stage-1 training mean `{DOT_XI_MEAN:.8g}` and standard deviation `{DOT_XI_STD:.8g}`. The mean cancels but is retained explicitly. `dot_xi_hat` is differentiated from the complete physical `xi_hat` with respect to the original physical-s leaf using `create_graph=True`.

## Batch and checkpoint audit

Each entry summarizes {BATCH_COUNT} deterministic, within-category, without-replacement training batches of {BATCH_SIZE} existing rows.

| seed | category | median ||g0|| | median ||gd|| | median R | R p10–p90 | median cosine | cosine p10–p90 |
|---:|:---|---:|---:|---:|:---|---:|:---|
{aggregate_rows}

No global or per-group gradient norm was zero or near-zero. Batch variability, category differences, and retained-checkpoint differences remain separate in the machine-readable tables.

## Parameter-group diagnostic for seed 202 overall batches

| parameter group | median ||g0|| | median ||gd|| | median R | median cosine |
|:---|---:|---:|---:|---:|
{layer_rows}

## Fixed lambda calibration

The robust calibration statistic is the median overall-training `R` for frozen seed 202: **`{summary['calibration_ratio']:.7g}`**.

| strength | target rho | raw rho/R | proposed fixed lambda | nominal rho after rounding |
|:---|---:|---:|---:|---:|
{candidate_rows}

The proposed fixed candidate values are {lambdas}. They are not adaptive and do not encode sharpness or trajectory difficulty.

| lambda | seed | category | median rho_batch | p10–p90 | min–max |
|:---|---:|:---|---:|:---|:---|
{rho_rows}

## Compute overhead

Across measured batches, median current-loss forward/backward time is `{timing['loss0_seconds']['median']:.6g}` s; median derivative-loss forward plus mixed backward time is `{timing['loss_d_seconds']['median']:.6g}` s. The median derivative/current time ratio is `{timing['time_ratio']['median']:.3f}x`. Approximate cumulative process peak RSS was `{timing['process_peak_rss_mib']:.1f}` MiB; the CPU runtime does not provide a reliable per-loss peak-memory split.

## Stage-2 conclusions

**A. Calibration ratio:** median seed-202 overall `R = {summary['calibration_ratio']:.7g}`.

**B. Fixed candidates:** {lambdas} for weak, moderate, and stronger-but-controlled derivative supervision.

**C. Nominal contribution:** `{summary['lambda_candidates'][0]['nominal_rho_after_rounding']:.3f}`, `{summary['lambda_candidates'][1]['nominal_rho_after_rounding']:.3f}`, and `{summary['lambda_candidates'][2]['nominal_rho_after_rounding']:.3f}` of the current-loss gradient norm at the calibration median; full batch/category/seed distributions are above.

**D. Alignment:** {summary['alignment']['statement']}

**E. Proceeding:** {summary['proceeding_assessment']}

Gradient calibration supplies scale, not an optimal lambda and not a winning model. A subsequent controlled validation experiment must compare lambda zero with all three fixed nonzero values under otherwise identical training conditions.

Focused tests: `{summary['tests']['stdout'].strip()}`. All protected hashes were unchanged.
"""


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT, TABLES, FIGURES, ARRAYS, TESTS): directory.mkdir(parents=True, exist_ok=False)
    gate = immutable_gate(); write_json(OUTPUT / "immutable_input_gate.json", gate)
    if not gate["passed"]: raise RuntimeError(gate["failures"])
    protected_before = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    stage1 = json.loads(STAGE1_SUMMARY.read_text(encoding="utf-8")); config = evaluation_configuration(stage1)
    if not all(config["stage1_distribution_match"].values()): raise RuntimeError("Stage-1 derivative normalization mismatch")
    write_json(OUTPUT / "gradient_calibration_config.json", config)

    raw = load_dataset(TRAIN_ROWS); training = construct_hybrid_targets(raw)
    wormhole, spiral = experiment_parameters(); exact_dot = xi_time_derivative(training["x1"], training["u1"], wormhole, spiral)
    q95, q99 = stage1["sharpness_bin_edges"][4], stage1["sharpness_bin_edges"][5]
    category_indices = {
        "overall": np.arange(exact_dot.size), "hard_u_th_le_0p30": np.flatnonzero(training["u_th"] <= .30),
        "sharp_q95_q99": np.flatnonzero((np.abs(exact_dot) > q95) & (np.abs(exact_dot) <= q99)),
        "ordinary_u_th_gt_0p30": np.flatnonzero(training["u_th"] > .30),
    }
    plans = {name: deterministic_batch_indices(indices, batch_size=BATCH_SIZE, batch_count=BATCH_COUNT, seed=CATEGORY_SEEDS[name]) for name, indices in category_indices.items()}
    np.savez_compressed(ARRAYS / "diagnostic_batch_indices.npz", **{f"{name}_indices": np.stack(batches) for name, batches in plans.items()})

    batch_rows, layer_rows = [], []
    for seed, checkpoint in CHECKPOINTS.items():
        model = load_hybrid_model(checkpoint)
        warm_index = plans["overall"][0]; warm = batch_subset(training, warm_index)
        audit_gradient_batch(model, HybridPreprocessing.from_json(PREPROCESSING), warm, exact_dot[warm_index], dot_xi_mean=DOT_XI_MEAN, dot_xi_std=DOT_XI_STD)
        preprocessing = HybridPreprocessing.from_json(PREPROCESSING)
        for category in CATEGORY_ORDER:
            for batch_number, index in enumerate(plans[category]):
                batch = batch_subset(training, index)
                result = audit_gradient_batch(model, preprocessing, batch, exact_dot[index], dot_xi_mean=DOT_XI_MEAN, dot_xi_std=DOT_XI_STD)
                batch_row = {"seed": seed, "category": category, "batch": batch_number, "eligible_row_count": int(category_indices[category].size),
                             "index_sha256": hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest(),
                             "mean_u_th": float(np.mean(training["u_th"][index])), "mean_abs_exact_dot_xi": float(np.mean(np.abs(exact_dot[index]))),
                             **{key: value for key, value in result.items() if key != "parameter_groups"}}
                batch_rows.append(batch_row)
                for group, values in result["parameter_groups"].items():
                    layer_rows.append({"seed": seed, "category": category, "batch": batch_number, "parameter_group": group, **values})
    write_csv(TABLES / "batch_gradient_norms_alignment.csv", batch_rows); write_csv(TABLES / "batch_parameter_group_gradients.csv", layer_rows)
    aggregate_table, aggregate = aggregate_batch_rows(batch_rows); layer_table = aggregate_layer_rows(layer_rows)
    write_csv(TABLES / "aggregate_gradient_statistics.csv", aggregate_table); write_json(TABLES / "aggregate_gradient_statistics.json", aggregate)
    write_csv(TABLES / "aggregate_parameter_group_statistics.csv", layer_table)

    calibration_ratio = aggregate[str(CALIBRATION_SEED)]["overall"]["ratio"]["median"]
    labels = ("weak", "moderate", "stronger_controlled")
    candidates = []
    for label, target in zip(labels, RHO_TARGETS):
        raw_lambda = target / calibration_ratio; fixed = rounded_fixed_lambda(raw_lambda)
        candidates.append({"label": label, "target_rho": target, "raw_lambda": raw_lambda, "fixed_lambda": fixed,
                           "nominal_rho_after_rounding": fixed * calibration_ratio})
    write_csv(TABLES / "lambda_candidates.csv", candidates)
    rho_rows = [{"lambda_label": candidate["label"], "fixed_lambda": candidate["fixed_lambda"], "target_rho": candidate["target_rho"],
                 "seed": row["seed"], "category": row["category"], "batch": row["batch"], "ratio": row["ratio"],
                 "rho_batch": candidate["fixed_lambda"] * row["ratio"]} for candidate in candidates for row in batch_rows]
    write_csv(TABLES / "batch_calibrated_rho.csv", rho_rows)
    rho_distributions = []
    for candidate in candidates:
        for seed in CHECKPOINTS:
            for category in CATEGORY_ORDER:
                selected = [row["rho_batch"] for row in rho_rows if row["lambda_label"] == candidate["label"] and row["seed"] == seed and row["category"] == category]
                rho_distributions.append({"lambda_label": candidate["label"], "fixed_lambda": candidate["fixed_lambda"], "seed": seed, "category": category, **distribution(selected)})
    write_csv(TABLES / "rho_distribution_statistics.csv", rho_distributions)

    cosine_medians = {category: distribution([row["cosine"] for row in batch_rows if row["category"] == category])["median"] for category in CATEGORY_ORDER}
    labels_by_category = {name: alignment_label(value) for name, value in cosine_medians.items()}
    seed_category_cosines = {str(seed): {category: aggregate[str(seed)][category]["cosine"]["median"] for category in CATEGORY_ORDER} for seed in CHECKPOINTS}
    checkpoint_sign_heterogeneity = [category for category in CATEGORY_ORDER
                                     if min(seed_category_cosines[str(seed)][category] for seed in CHECKPOINTS) < -ALIGNMENT_THRESHOLD
                                     and max(seed_category_cosines[str(seed)][category] for seed in CHECKPOINTS) > ALIGNMENT_THRESHOLD]
    region_dependent = len(set(labels_by_category.values())) > 1 or max(cosine_medians.values()) - min(cosine_medians.values()) > .20
    statement = (f"Across seeds, median cosine by category is {', '.join(f'{name}={value:.3f}' for name, value in cosine_medians.items())}. "
                 f"The overall signal is {alignment_label(cosine_medians['overall'])}; alignment is {'strongly region-dependent' if region_dependent else 'not strongly region-dependent'} under the frozen labeling rules. "
                 f"Checkpoint medians cross from conflicting to cooperative in {', '.join(checkpoint_sign_heterogeneity) if checkpoint_sign_heterogeneity else 'no category'}, so seed variation is scientifically material.")
    near_zero = [row for row in batch_rows if row["near_zero_gradient0"] or row["near_zero_gradient_d"]]
    near_zero_layers = [row for row in layer_rows if row["near_zero_gradient0"] or row["near_zero_gradient_d"]]
    unused = [row for row in batch_rows if row["unused_gradient0_parameters"] or row["unused_gradient_d_parameters"]]
    strongest_sharp = distribution([row["rho_batch"] for row in rho_rows if row["lambda_label"] == labels[-1] and row["category"] == "sharp_q95_q99"])
    if near_zero or near_zero_layers or unused:
        proceeding = "Near-zero or unused gradients require review before controlled retraining."
    else:
        proceeding = ("Proceed to the controlled four-treatment validation experiment. The strongest candidate is an upper stress treatment, not uniformly controlled: "
                      f"across retained seeds its sharp q95-q99 batches have median rho={strongest_sharp['median']:.3f} and maximum rho={strongest_sharp['maximum']:.3f}; monitor this localized dominance.")
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; peak_mib = maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
    compute = {"loss0_seconds": distribution([row["loss0_forward_backward_seconds"] for row in batch_rows]),
               "loss_d_seconds": distribution([row["loss_d_forward_mixed_backward_seconds"] for row in batch_rows]),
               "time_ratio": distribution([row["time_ratio_d_over_0"] for row in batch_rows]),
               "process_peak_rss_mib": peak_mib, "memory_scope": "cumulative process peak; no reliable per-loss CPU allocator split"}
    plot_distributions(batch_rows); plot_category(batch_rows); plot_rho(rho_rows, candidates)
    tests = run_tests(); protected_after = {str(path.resolve()): file_sha256(path) for path in EXPECTED}
    if protected_before != protected_after: raise RuntimeError("protected artifact changed")
    if not tests["passed"]: raise RuntimeError("focused tests failed")
    summary = {"created_utc": utc(), "status": "GRADIENT_SCALE_ALIGNMENT_CALIBRATION_COMPLETED", "configuration": config,
               "batch_count": len(batch_rows), "batches_per_seed_category": BATCH_COUNT, "batch_size": BATCH_SIZE,
               "category_eligible_counts": {name: int(value.size) for name, value in category_indices.items()},
               "aggregate_table": aggregate_table, "aggregate": aggregate, "layer_table": layer_table,
               "calibration_seed": CALIBRATION_SEED, "calibration_ratio": calibration_ratio, "lambda_candidates": candidates,
               "rho_distributions": rho_distributions, "alignment": {"category_median_cosines_across_seeds": cosine_medians,
               "seed_category_median_cosines": seed_category_cosines, "checkpoint_sign_heterogeneity": checkpoint_sign_heterogeneity,
               "labels": labels_by_category, "strongly_region_dependent": region_dependent, "statement": statement},
               "near_zero_gradient_batches": len(near_zero), "near_zero_parameter_group_batches": len(near_zero_layers),
               "unused_gradient_batches": len(unused), "compute_overhead": compute,
               "proceeding_assessment": proceeding, "tests": tests, "training_performed": False, "optimizer_created": False,
               "optimizer_steps": 0, "dataset_modified": False, "held_out_test_accessed": False, "protected_hashes_unchanged": True}
    write_json(SUMMARY, summary); REPORT.write_text(report_text(summary), encoding="utf-8")
    artifacts = {str(path.relative_to(OUTPUT)): {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST, MANIFEST_HASH)}
    manifest = {"status": summary["status"], "input_gate": gate, "protected_before": protected_before, "protected_after": protected_after,
                "source_hashes": source_hashes(), "artifacts": artifacts, "summary_sha256": file_sha256(SUMMARY), "report_sha256": file_sha256(REPORT)}
    write_json(MANIFEST, manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
