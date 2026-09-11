#!/usr/bin/env python3
"""Fresh nine-run Model-A rerun with only the epoch cap extended to 1000."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wormhole_sciml.model_a import (
    COLLAR_BATCH_SIZE,
    COLLAR_COEFFICIENT,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    PHYSICAL_BATCH_SIZE,
    TRAINING_SEEDS,
    TREATMENTS,
    train_round1_run,
)


ORIGINAL_EPOCH_CAP = 500
EXTENDED_EPOCH_CAP = 1000
HISTORY_TOLERANCE = 0.0
ORIGINAL_TREE = "round1_model_a"
EXTENDED_TREE = "round1_model_a_1000"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_identity(path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        file_hash = sha256(candidate)
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(file_hash.encode("ascii") + b"\0")
    return {
        "path": str(path),
        "file_count": len(files),
        "tree_sha256": digest.hexdigest(),
    }


def audited_original_runs(project_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_path = (
        project_root / "output" / ORIGINAL_TREE / "training_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["run_count"] != 9:
        raise RuntimeError("the original Round-I manifest does not contain nine runs")
    runs = {(row["treatment"], int(row["seed"])): row for row in manifest["runs"]}
    expected = {
        (treatment, seed) for treatment in TREATMENTS for seed in TRAINING_SEEDS
    }
    if set(runs) != expected:
        raise RuntimeError("the original treatment/seed matrix is inconsistent")
    for row in runs.values():
        if row["maximum_epochs"] != ORIGINAL_EPOCH_CAP:
            raise RuntimeError("an original run does not have the 500-epoch cap")
        if row["architecture"] != "2->32->2, one tanh hidden layer":
            raise RuntimeError("the original architecture metadata is inconsistent")
        if row["learning_rate"] != LEARNING_RATE or row["optimizer"] != "Adam":
            raise RuntimeError("the original optimizer metadata is inconsistent")
        if row["early_stopping_patience"] != PATIENCE:
            raise RuntimeError("the original patience is inconsistent")
        if row["physical_batch_size"] != PHYSICAL_BATCH_SIZE:
            raise RuntimeError("the original physical batch size is inconsistent")
        expected_collar_batch = None if row["collar_dataset"] is None else COLLAR_BATCH_SIZE
        expected_collar_coefficient = (
            None if row["collar_dataset"] is None else COLLAR_COEFFICIENT
        )
        if row["collar_batch_size"] != expected_collar_batch:
            raise RuntimeError("the original collar batch size is inconsistent")
        if row["collar_coefficient"] != expected_collar_coefficient:
            raise RuntimeError("the original collar coefficient is inconsistent")
    if MAX_EPOCHS != ORIGINAL_EPOCH_CAP:
        raise RuntimeError("the frozen default Model-A epoch cap is no longer 500")
    return runs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "output" / EXTENDED_TREE
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing extended-run tree: {output_root}"
        )

    original_runs = audited_original_runs(project_root)
    preserved_paths = (
        project_root / "output" / ORIGINAL_TREE,
        project_root / "reports" / ORIGINAL_TREE,
        project_root / "output" / "round1_local_restart",
        project_root / "reports" / "round1_local_restart",
    )
    preserved_before = {str(path): tree_identity(path) for path in preserved_paths}

    completed = []
    for treatment in TREATMENTS:
        for seed in TRAINING_SEEDS:
            original = original_runs[(treatment, seed)]
            result = train_round1_run(
                project_root,
                treatment,
                seed,
                maximum_epochs=EXTENDED_EPOCH_CAP,
                output_tree=EXTENDED_TREE,
                reference_history_path=Path(original["history"]),
                reference_prefix_epochs=ORIGINAL_EPOCH_CAP,
                deterministic_history_atol=HISTORY_TOLERANCE,
                stage_label="Round-I controlled 1000-epoch extension",
            )
            if result["initial_state_sha256"] != original["initial_state_sha256"]:
                raise RuntimeError("fresh-run initialization hash differs from Round I")
            comparison = result["deterministic_history_prefix_comparison"]
            if comparison is None or not comparison["passed"]:
                raise RuntimeError("the deterministic 500-epoch history gate failed")
            completed.append(result)

    preserved_after = {str(path): tree_identity(path) for path in preserved_paths}
    if preserved_before != preserved_after:
        raise RuntimeError("an existing Round-I output or diagnostic tree changed")

    manifest = {
        "stage": "Round-I controlled 1000-epoch extension",
        "status": "nine_fresh_runs_completed",
        "scientific_change": {
            "field": "maximum_epochs",
            "old": ORIGINAL_EPOCH_CAP,
            "new": EXTENDED_EPOCH_CAP,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "treatments": list(TREATMENTS),
        "run_count": len(completed),
        "runs": completed,
        "history_prefix_gate": {
            "epochs": ORIGINAL_EPOCH_CAP,
            "absolute_tolerance": HISTORY_TOLERANCE,
            "all_runs_passed": True,
        },
        "preserved_trees_before": preserved_before,
        "preserved_trees_after": preserved_after,
        "previous_outputs_preserved": True,
        "constraints_introduced": False,
        "restricted_data_opened": False,
        "domain_treatment_selected": False,
        "capacity_comparison_started": False,
        "rollout_evaluation_performed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
