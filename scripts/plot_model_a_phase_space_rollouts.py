#!/usr/bin/env python3
"""Plot saved Model-A architecture rollouts as physical ``(x, u)`` paths."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.stage1_data import file_sha256
from evaluate_round1_1000_rollouts import prefix


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_DIR = ROOT / "output" / "model_a_architecture_rollouts"
ROLLOUT_MANIFEST = ROLLOUT_DIR / "architecture_rollout_manifest.json"
ROLLOUT_ARRAYS = ROLLOUT_DIR / "architecture_validation_rollouts.npz"
REFERENCE_DIR = ROOT / "output" / "round1_model_a_1000_rollouts"
REFERENCE_MANIFEST = REFERENCE_DIR / "rollout_evaluation_manifest.json"
REFERENCE_ARRAYS = REFERENCE_DIR / "full_validation_rollouts.npz"
OUTPUT_DIR = ROOT / "output" / "model_a_architecture_phase_space"
FIGURE_DIR = OUTPUT_DIR / "trajectories"
MODELS = ("a64", "a32x32", "c16x16", "c32x32")
LABELS = {"a64": "A64", "a32x32": "A32×32", "c16x16": "C16×16", "c32x32": "C32×32"}
SEED_COLORS = {101: "#277da1", 202: "#f8961e", 303: "#9b5de5"}


def trajectory_metadata(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest["models"]["a64"][0]["rollouts"]["trajectories"]


def row_lookup(manifest: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {
        (model, int(run["seed"]), row["id"]): row
        for model in MODELS for run in manifest["models"][model]
        for row in run["rollouts"]["trajectories"]
    }


def limits(reference: np.ndarray, arrays: Any, key: str) -> tuple[tuple[float, float], tuple[float, float]]:
    states = [reference]
    states.extend(
        arrays[f"{key}__{model}__seed_{seed}__predicted_state"]
        for model in MODELS for seed in (101, 202, 303)
    )
    joined = np.concatenate(states)
    bounds = []
    for column in (0, 1):
        low, high = float(np.min(joined[:, column])), float(np.max(joined[:, column]))
        padding = 0.06 * max(high - low, 1e-9)
        bounds.append((low - padding, high + padding))
    return bounds[0], bounds[1]


def plot_trajectory(member: dict[str, Any], arrays: Any, reference_arrays: Any,
                    lookup: dict[tuple[str, int, str], dict[str, Any]],
                    output: Path, highlight: bool = False) -> dict[str, Any]:
    identifier, key = member["id"], prefix(member["id"])
    reference = reference_arrays[f"{key}__reference_state"]
    time = reference_arrays[f"{key}__time"]
    xlim, ylim = limits(reference, arrays, key)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10) if highlight else (11, 8.2),
                             constrained_layout=True, sharex=True, sharey=True)
    summary: dict[str, Any] = {
        "trajectory_id": identifier, "category": member["category"],
        "reference_class": member["reference_class"],
    }
    for axis, model in zip(axes.ravel(), MODELS):
        axis.plot(reference[:, 0], reference[:, 1], color="black", lw=2.6,
                  marker=">", markevery=max(1, len(reference) // 10), ms=4, label="DOP853")
        axis.scatter(reference[0, 0], reference[0, 1], marker="*", s=90, color="black", zorder=6)
        axis.scatter(reference[-1, 0], reference[-1, 1], marker="s", s=45,
                     facecolor="white", edgecolor="black", zorder=6)
        time_errors, final_errors, exit_labels = [], [], []
        for seed in (101, 202, 303):
            stem = f"{key}__{model}__seed_{seed}"
            state, error = arrays[f"{stem}__predicted_state"], arrays[f"{stem}__combined_error"]
            axis.plot(state[:, 0], state[:, 1], color=SEED_COLORS[seed], lw=1.35,
                      marker=".", markevery=max(1, len(state) // 12), ms=3, label=f"seed {seed}")
            axis.scatter(state[-1, 0], state[-1, 1], marker="x", s=35,
                         color=SEED_COLORS[seed], zorder=5)
            rollout = lookup[(model, seed, identifier)]
            if rollout["physical_exit"]:
                step = int(rollout["first_exit_step"])
                axis.scatter(state[step, 0], state[step, 1], marker="X", s=75,
                             color="red", edgecolor="white", linewidth=0.5, zorder=7)
                exit_labels.append(f"{seed}: step {step}, s={time[step]:.1f}")
            time_errors.append(float(np.mean(error)))
            final_errors.append(float(error[-1]))
        mean_time, mean_final = float(np.mean(time_errors)), float(np.mean(final_errors))
        summary[f"{model}_mean_time_mean_ez"] = mean_time
        summary[f"{model}_mean_final_ez"] = mean_final
        summary[f"{model}_exit_count"] = len(exit_labels)
        annotation = (f"{LABELS[model]} · seeds=3\n"
                      f"mean time $e_z$={mean_time:.4g}\nmean final $e_z$={mean_final:.4g}\n"
                      f"exits={len(exit_labels)}/3")
        if exit_labels:
            annotation += "\n" + "\n".join(exit_labels)
        axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top", fontsize=8,
                  bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"})
        axis.set(title=LABELS[model], xlabel="$x$", ylabel="$u$", xlim=xlim, ylim=ylim)
        axis.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{identifier} — {member['category']} / {member['reference_class']}\n"
                 "full phase-space rollout comparison")
    fig.savefig(output, dpi=220 if highlight else 170)
    plt.close(fig)
    summary["worst_mean_time_mean_ez"] = max(summary[f"{model}_mean_time_mean_ez"] for model in MODELS)
    summary["worst_architecture"] = max(MODELS, key=lambda model: summary[f"{model}_mean_time_mean_ez"])
    return summary


def write_csv(rows: list[dict[str, Any]]) -> Path:
    path = OUTPUT_DIR / "worst_trajectory_summary.csv"
    fields = ["rank", "trajectory_id", "category", "reference_class", "worst_architecture",
              "worst_mean_time_mean_ez"]
    for model in MODELS:
        fields.extend((f"{model}_mean_time_mean_ez", f"{model}_mean_final_ez", f"{model}_exit_count"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **row})
    return path


def contact_sheet(rows: list[dict[str, Any]], figures: dict[str, Path]) -> Path:
    fig, axes = plt.subplots(6, 4, figsize=(16, 21), constrained_layout=True)
    for rank, (axis, row) in enumerate(zip(axes.ravel(), rows), 1):
        axis.imshow(mpimg.imread(figures[row["trajectory_id"]]))
        axis.set_title(f"{rank}. {row['trajectory_id'].removeprefix('validation-')}\n"
                       f"worst={LABELS[row['worst_architecture']]} {row['worst_mean_time_mean_ez']:.3g}",
                       fontsize=8)
        axis.axis("off")
    path = OUTPUT_DIR / "phase_space_contact_sheet.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    manifest = json.loads(ROLLOUT_MANIFEST.read_text(encoding="utf-8"))
    reference_manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest["frozen_validation_identity_sha256"] != reference_manifest["frozen_validation_identity_sha256"]:
        raise RuntimeError("frozen validation identities disagree")
    protected = (ROLLOUT_MANIFEST, ROLLOUT_ARRAYS, REFERENCE_MANIFEST, REFERENCE_ARRAYS)
    before = {str(path): file_sha256(path) for path in protected}
    OUTPUT_DIR.mkdir(parents=True)
    FIGURE_DIR.mkdir()
    lookup, rows, figures = row_lookup(manifest), [], {}
    with np.load(ROLLOUT_ARRAYS) as arrays, np.load(REFERENCE_ARRAYS) as references:
        for member in trajectory_metadata(manifest):
            figure = FIGURE_DIR / f"{prefix(member['id'])}_phase_space.png"
            rows.append(plot_trajectory(member, arrays, references, lookup, figure))
            figures[member["id"]] = figure
        anchor = next(member for member in trajectory_metadata(manifest)
                      if member["id"] == "validation-crossing-anchor")
        highlight = OUTPUT_DIR / "validation_crossing_anchor_phase_space_highlight.png"
        plot_trajectory(anchor, arrays, references, lookup, highlight, highlight=True)
    rows.sort(key=lambda row: row["worst_mean_time_mean_ez"], reverse=True)
    csv_path, sheet = write_csv(rows), contact_sheet(rows, figures)
    after = {str(path): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("a reused rollout or reference artifact changed")
    artifacts = {
        "trajectory_figures": [{"trajectory_id": row["trajectory_id"],
                                "path": str(figures[row["trajectory_id"]]),
                                "sha256": file_sha256(figures[row["trajectory_id"]])} for row in rows],
        "crossing_anchor_highlight": {"path": str(highlight), "sha256": file_sha256(highlight)},
        "summary_csv": {"path": str(csv_path), "sha256": file_sha256(csv_path)},
        "contact_sheet": {"path": str(sheet), "sha256": file_sha256(sheet)},
    }
    output_manifest = {
        "stage": "evaluation-only full physical phase-space rollout plots",
        "status": "24_trajectory_figures_plus_highlight_and_contact_sheet_completed",
        "models": list(MODELS), "seed_count_per_model": 3,
        "frozen_validation_identity_sha256": manifest["frozen_validation_identity_sha256"],
        "reused_inputs": {"rollout_manifest": str(ROLLOUT_MANIFEST),
                          "rollout_arrays": str(ROLLOUT_ARRAYS),
                          "reference_manifest": str(REFERENCE_MANIFEST),
                          "reference_arrays": str(REFERENCE_ARRAYS)},
        "protected_hashes_before": before, "protected_hashes_after": after,
        "ranking_definition": "descending maximum across architectures of per-trajectory three-seed mean time-mean e_z",
        "ranked_trajectories": rows, "artifacts": artifacts,
        "protocol": {"training_performed": False, "checkpoint_loaded": False,
                     "rollout_recomputed": False, "dense_grid_evaluated": False,
                     "restart_evaluated": False, "clipping_or_projection_performed": False,
                     "restricted_data_accessed": False},
    }
    path = OUTPUT_DIR / "phase_space_manifest.json"
    path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
