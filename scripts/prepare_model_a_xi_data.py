#!/usr/bin/env python3
"""Prepare unsealed physical Model-A data in ``(x, xi)`` coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wormhole_sciml.model_a_xi_data import (
    IDENTITY_FIELDS,
    transform_state_pairs_to_x_xi,
    x_xi_normalization,
)
from wormhole_sciml.physics_gate import experiment_parameters, state_from_xi
from wormhole_sciml.stage1_data import (
    H,
    array_content_sha256,
    file_sha256,
    load_dataset,
    save_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "output" / "stage1_model_a"
OUTPUT_DIR = PROJECT_ROOT / "output" / "stage1_model_a_x_xi"
SOURCE_NAMES = ("physical_train", "physical_validation")


def verify_transformation(
    source: dict[str, np.ndarray],
    transformed: dict[str, np.ndarray],
) -> dict[str, Any]:
    wormhole, spiral = experiment_parameters()
    _, reconstructed_u = state_from_xi(
        transformed["x"], transformed["xi"], wormhole, spiral
    )
    next_x = transformed["x"] + transformed["delta_x"]
    next_xi = transformed["xi"] + transformed["delta_xi"]
    _, reconstructed_next_u = state_from_xi(next_x, next_xi, wormhole, spiral)
    expected_next_u = source["u"] + source["delta_u"]
    ordering_preserved = np.array_equal(transformed["x"], source["x"]) and all(
        np.array_equal(transformed[name], source[name]) for name in IDENTITY_FIELDS
    )
    if not ordering_preserved or transformed["x"].size != source["x"].size:
        raise RuntimeError("sample count or ordering changed during transformation")
    maximum_abs_xi = float(np.max(np.abs(transformed["xi"])))
    if maximum_abs_xi > 0.99 + 32.0 * np.finfo(np.float64).eps:
        raise RuntimeError(f"unexpected physical xi-range violation: {maximum_abs_xi}")
    return {
        "source_row_count": int(source["x"].size),
        "transformed_row_count": int(transformed["x"].size),
        "sample_count_and_ordering_preserved": ordering_preserved,
        "maximum_abs_stored_vs_recomputed_xi_error": float(
            np.max(np.abs(source["xi"] - transformed["xi"]))
        ),
        "maximum_abs_current_u_reconstruction_error": float(
            np.max(np.abs(source["u"] - reconstructed_u))
        ),
        "maximum_abs_next_u_reconstruction_error": float(
            np.max(np.abs(expected_next_u - reconstructed_next_u))
        ),
        "xi_minimum": float(np.min(transformed["xi"])),
        "xi_maximum": float(np.max(transformed["xi"])),
        "maximum_abs_xi": maximum_abs_xi,
        "physical_xi_range_check_passed": True,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    source_paths = {name: SOURCE_DIR / f"{name}.npz" for name in SOURCE_NAMES}
    source_hashes_before = {name: file_sha256(path) for name, path in source_paths.items()}
    sources = {name: load_dataset(path) for name, path in source_paths.items()}
    wormhole, spiral = experiment_parameters()
    transformed = {
        name: transform_state_pairs_to_x_xi(data, wormhole, spiral)
        for name, data in sources.items()
    }
    verification = {
        name: verify_transformation(sources[name], transformed[name])
        for name in SOURCE_NAMES
    }
    normalization = x_xi_normalization(transformed["physical_train"])

    OUTPUT_DIR.mkdir(parents=True)
    artifacts = {
        name: save_dataset(OUTPUT_DIR / f"{name}_x_xi.npz", transformed[name])
        for name in SOURCE_NAMES
    }
    normalization_path = OUTPUT_DIR / "normalization.json"
    normalization_path.write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_hashes_after = {name: file_sha256(path) for name, path in source_paths.items()}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("a canonical Stage-1 source artifact changed")
    manifest = {
        "stage": "Model-A deterministic (x,u) to (x,xi) data transformation",
        "status": "transformed_and_verified",
        "learning_problem_old": "(x,u) -> (delta_x,delta_u)",
        "learning_problem_new": "(x,xi) -> (delta_x,delta_xi)",
        "transformation": {
            "xi_n": "xi_from_state(x_n,u_n)=(u_n-c(x_n))/d(x_n)",
            "xi_next": "xi_from_state(x_n+delta_x,u_n+delta_u)",
            "delta_xi": "xi_next-xi_n",
            "inverse": "state_from_xi(x,xi): u=c(x)+d(x)*xi",
            "finite_step_h": H,
            "canonical_functions": [
                "wormhole_sciml.physics_gate.xi_from_state",
                "wormhole_sciml.physics_gate.state_from_xi",
            ],
        },
        "source_artifacts": {
            name: {
                "path": str(source_paths[name]),
                "file_sha256": source_hashes_before[name],
                "content_sha256": array_content_sha256(sources[name]),
            }
            for name in SOURCE_NAMES
        },
        "source_hashes_before_and_after_identical": True,
        "verification": verification,
        "artifacts": artifacts,
        "normalization": {
            "path": str(normalization_path),
            "file_sha256": file_sha256(normalization_path),
            "source_dataset_only": "physical_train_x_xi",
            "statistics": normalization,
        },
        "protocol": {
            "DOP853_integration_performed": False,
            "training_performed": False,
            "constraint_introduced": False,
            "clipping_or_projection_used": False,
            "sealed_data_accessed": False,
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Prepared deterministic (x,xi) data without integration or training.")
    for name in SOURCE_NAMES:
        check = verification[name]
        print(
            f"{name}: {check['transformed_row_count']} rows; "
            f"max |u-u_rec|={check['maximum_abs_current_u_reconstruction_error']:.3e}; "
            f"max |u_next-u_next_rec|={check['maximum_abs_next_u_reconstruction_error']:.3e}; "
            f"xi=[{check['xi_minimum']:.9f},{check['xi_maximum']:.9f}]"
        )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
