"""Deterministic fixed-step data generation for the Stage-1 Model-A task.

This module only samples states and calls the already validated reference
integrators.  It contains no duplicate dynamics and no machine-learning code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .dynamics import timelike_margin
from .integrate import integrate_trajectory, integrate_vector_field_continuation
from .physics_gate import (
    PRODUCTION_SOLVER,
    TIGHT_SOLVER,
    experiment_parameters,
    source_manifest,
    state_from_xi,
    xi_from_state,
)


H = 0.2
X = 17.0
X_C = 8.5
S_ESC = 39.4

SEEDS = {
    "physical_train": 2_026_081_601,
    "physical_validation": 2_026_081_602,
    "physical_sealed_test": 2_026_081_603,
    "exterior_paired_train": 2_026_081_611,
    "exterior_validation": 2_026_081_612,
    "exterior_sealed_stress": 2_026_081_613,
}

DATASET_SPECS = {
    "physical_train": {"count": 20_000, "domain": "physical"},
    "physical_validation": {"count": 4_000, "domain": "physical"},
    "physical_sealed_test": {"count": 4_000, "domain": "physical"},
    "exterior_train_0p20": {
        "count": 5_000,
        "domain": "nonphysical_exterior",
        "abs_xi": [1.01, 1.20],
    },
    "exterior_train_0p25": {
        "count": 5_000,
        "domain": "nonphysical_exterior",
        "abs_xi": [1.01, 1.25],
    },
    "exterior_validation": {
        "count": 2_000,
        "domain": "nonphysical_exterior",
        "abs_xi": [1.01, 1.25],
    },
    "exterior_sealed_stress": {
        "count": 2_000,
        "domain": "nonphysical_exterior",
        "abs_xi": [1.01, 1.25],
    },
}

STRATUM_NAMES = ("core", "shoulder", "edge")
STRATUM_BOUNDS = ((0.0, 0.5), (0.5, 0.9), (0.9, 0.99))
STRATUM_FRACTIONS = (0.30, 0.35, 0.35)


class Stage1BoundaryError(RuntimeError):
    """Raised before serialization when a requested target cannot be made."""

    def __init__(self, diagnostic: dict[str, Any]):
        super().__init__(diagnostic["summary"])
        self.diagnostic = diagnostic


def _sample_x(count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the exactly balanced central/broad x mixture."""

    if count % 2:
        raise ValueError("x-mixture count must be even")
    half = count // 2
    x = np.concatenate((rng.uniform(-X_C, X_C, half), rng.uniform(-X, X, half)))
    component = np.concatenate(
        (np.zeros(half, dtype=np.int8), np.ones(half, dtype=np.int8))
    )
    # Randomize the mixture assignment before it is paired with independently
    # constructed xi strata/signs; exact component counts are preserved.
    permutation = rng.permutation(count)
    return x[permutation].astype(np.float64, copy=False), component[permutation]


def sample_physical_initial_states(count: int, seed: int) -> dict[str, np.ndarray]:
    """Return exact-stratum, exact-sign physical initial states."""

    counts = [int(count * fraction) for fraction in STRATUM_FRACTIONS]
    if sum(counts) != count or any(value % 2 for value in counts):
        raise ValueError("count does not permit exact requested strata/sign balance")
    rng = np.random.default_rng(seed)
    x, component = _sample_x(count, rng)
    xi_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sign_parts: list[np.ndarray] = []
    for label, ((low, high), stratum_count) in enumerate(zip(STRATUM_BOUNDS, counts)):
        magnitude = rng.uniform(low, high, stratum_count).astype(np.float64)
        signs = np.concatenate(
            (
                -np.ones(stratum_count // 2, dtype=np.int8),
                np.ones(stratum_count // 2, dtype=np.int8),
            )
        )
        rng.shuffle(signs)
        xi_parts.append(magnitude * signs)
        label_parts.append(np.full(stratum_count, label, dtype=np.uint8))
        sign_parts.append(signs)
    xi = np.concatenate(xi_parts)
    stratum = np.concatenate(label_parts)
    xi_sign = np.concatenate(sign_parts)
    permutation = rng.permutation(count)
    x = x[permutation]
    component = component[permutation]
    xi = xi[permutation]
    stratum = stratum[permutation]
    xi_sign = xi_sign[permutation]
    wormhole, spiral = experiment_parameters()
    _, u = state_from_xi(x, xi, wormhole, spiral)
    return {
        "x": np.asarray(x, dtype=np.float64),
        "u": np.asarray(u, dtype=np.float64),
        "xi": np.asarray(xi, dtype=np.float64),
        "x_component": component,
        "xi_sign": xi_sign,
        "stratum": stratum,
        "is_physical": np.ones(count, dtype=np.bool_),
    }


def _sample_exterior_base(count: int, seed: int) -> dict[str, np.ndarray]:
    if count % 2:
        raise ValueError("exterior count must be even")
    rng = np.random.default_rng(seed)
    x, component = _sample_x(count, rng)
    signs = np.concatenate(
        (-np.ones(count // 2, dtype=np.int8), np.ones(count // 2, dtype=np.int8))
    )
    rng.shuffle(signs)
    r = rng.random(count, dtype=np.float64)
    permutation = rng.permutation(count)
    return {
        "x": x[permutation],
        "x_component": component[permutation],
        "xi_sign": signs[permutation],
        "r": r[permutation],
        "pair_id": np.arange(count, dtype=np.int64)[permutation],
    }


def sample_exterior_initial_states(
    count: int, seed: int, upper: float
) -> dict[str, np.ndarray]:
    """Return exterior states using a saved uniform variate r."""

    base = _sample_exterior_base(count, seed)
    magnitude = 1.01 + base["r"] * (upper - 1.01)
    xi = magnitude * base["xi_sign"]
    wormhole, spiral = experiment_parameters()
    _, u = state_from_xi(base["x"], xi, wormhole, spiral)
    return {
        **base,
        "u": np.asarray(u, dtype=np.float64),
        "xi": np.asarray(xi, dtype=np.float64),
        "stratum": np.full(count, 255, dtype=np.uint8),
        "is_physical": np.zeros(count, dtype=np.bool_),
    }


def sample_paired_exterior_training() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the genuinely paired 0.20 and 0.25 exterior collars."""

    base = _sample_exterior_base(5_000, SEEDS["exterior_paired_train"])
    wormhole, spiral = experiment_parameters()
    outputs = []
    for upper in (1.20, 1.25):
        xi = base["xi_sign"] * (1.01 + base["r"] * (upper - 1.01))
        _, u = state_from_xi(base["x"], xi, wormhole, spiral)
        outputs.append(
            {
                **{key: value.copy() for key, value in base.items()},
                "u": np.asarray(u, dtype=np.float64),
                "xi": np.asarray(xi, dtype=np.float64),
                "stratum": np.full(5_000, 255, dtype=np.uint8),
                "is_physical": np.zeros(5_000, dtype=np.bool_),
            }
        )
    return outputs[0], outputs[1]


def _target_for_state(
    x: float,
    u: float,
    physical: bool,
    *,
    tight: bool = False,
) -> np.ndarray:
    wormhole, spiral = experiment_parameters()
    solver = TIGHT_SOLVER if tight else PRODUCTION_SOLVER
    integrator = integrate_trajectory if physical else integrate_vector_field_continuation
    kwargs: dict[str, Any] = solver.kwargs()
    if physical:
        kwargs["stop_at_null_boundary"] = True
    solution = integrator(
        (x, u),
        (0.0, H),
        wormhole,
        spiral,
        t_eval=(H,),
        **kwargs,
    )
    if solution.y.shape != (2, 1) or float(solution.t[-1]) != H:
        raise RuntimeError("reference integration did not reach the requested h=0.2")
    final = np.asarray(solution.y[:, -1], dtype=np.float64)
    if not np.all(np.isfinite(final)):
        raise FloatingPointError("reference integration returned nonfinite state")
    return final - np.asarray((x, u), dtype=np.float64)


def add_targets(
    name: str,
    sampled: dict[str, np.ndarray],
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Integrate every sampled row, stopping on any boundary ambiguity."""

    physical = bool(np.all(sampled["is_physical"]))
    count = sampled["x"].size
    delta = np.empty((count, 2), dtype=np.float64)
    failures: list[dict[str, Any]] = []
    wormhole, spiral = experiment_parameters()
    for index, (x, u, xi) in enumerate(zip(sampled["x"], sampled["u"], sampled["xi"])):
        try:
            delta[index] = _target_for_state(float(x), float(u), physical)
        except Exception as exc:  # diagnostic must preserve the originally sampled row
            failures.append(
                {
                    "index": index,
                    "x": float(x),
                    "u": float(u),
                    "xi": float(xi),
                    "C": float(timelike_margin(x, u, wormhole, spiral)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if len(failures) >= 10:
                break
        if progress is not None and (index + 1) % 5_000 == 0:
            progress(f"{name}: integrated {index + 1}/{count}")
    if failures:
        raise Stage1BoundaryError(
            {
                "summary": f"{name}: target generation failed; no rows were serialized",
                "affected_rows_observed": len(failures),
                "representative_rows": failures,
                "policy": "No resampling, clipping, extrapolation, or definition change was applied.",
            }
        )
    return {
        **sampled,
        "delta_x": delta[:, 0],
        "delta_u": delta[:, 1],
    }


def array_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, shapes, dtypes, and C-order contents deterministically."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dataset(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "count": int(arrays["x"].size),
        "schema": {name: str(value.dtype) for name, value in arrays.items()},
        "content_sha256": array_content_sha256(arrays),
        "file_sha256": file_sha256(path),
    }


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _initial_state_keys(data: dict[str, np.ndarray]) -> set[bytes]:
    states = np.ascontiguousarray(np.column_stack((data["x"], data["u"])))
    return {row.tobytes() for row in states}


def physical_summary(data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "count": int(data["x"].size),
        "strata": {
            name: int(np.sum(data["stratum"] == index))
            for index, name in enumerate(STRATUM_NAMES)
        },
        "signs": {
            "negative": int(np.sum(data["xi_sign"] == -1)),
            "positive": int(np.sum(data["xi_sign"] == 1)),
        },
        "x_components": {
            "central": int(np.sum(data["x_component"] == 0)),
            "broad": int(np.sum(data["x_component"] == 1)),
        },
    }


def normalization_statistics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    columns = ("x", "u", "delta_x", "delta_u")
    return {
        "source_dataset": "physical_train",
        "source_row_count": int(data["x"].size),
        "dtype": "float64",
        "standard_deviation_definition": "population (ddof=0)",
        "columns": {
            name: {
                "mean": float(np.mean(data[name], dtype=np.float64)),
                "standard_deviation": float(np.std(data[name], ddof=0, dtype=np.float64)),
            }
            for name in columns
        },
    }


def explicit_target_checks(datasets: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Tight-solver recomputations for unsealed train/validation examples only."""

    requests: list[tuple[str, int, str]] = []
    for dataset_name in ("physical_train", "physical_validation"):
        data = datasets[dataset_name]
        for label, stratum_name in enumerate(STRATUM_NAMES):
            index = int(np.flatnonzero(data["stratum"] == label)[0])
            requests.append((dataset_name, index, stratum_name))
    for dataset_name, label in (
        ("exterior_train_0p20", "exterior-0.20"),
        ("exterior_train_0p25", "exterior-0.25"),
        ("exterior_validation", "exterior-validation"),
    ):
        data = datasets[dataset_name]
        for sign in (-1, 1):
            index = int(np.flatnonzero(data["xi_sign"] == sign)[0])
            requests.append((dataset_name, index, f"{label}, sign={sign:+d}"))

    checks: list[dict[str, Any]] = []
    for dataset_name, index, case in requests:
        data = datasets[dataset_name]
        physical = bool(data["is_physical"][index])
        stored = np.asarray((data["delta_x"][index], data["delta_u"][index]))
        recomputed = _target_for_state(
            float(data["x"][index]), float(data["u"][index]), physical, tight=True
        )
        error = stored - recomputed
        checks.append(
            {
                "dataset": dataset_name,
                "row_index": index,
                "case": case,
                "x": float(data["x"][index]),
                "u": float(data["u"][index]),
                "xi": float(data["xi"][index]),
                "stored_delta_x": float(stored[0]),
                "stored_delta_u": float(stored[1]),
                "tight_delta_x": float(recomputed[0]),
                "tight_delta_u": float(recomputed[1]),
                "abs_error_delta_x": float(abs(error[0])),
                "abs_error_delta_u": float(abs(error[1])),
            }
        )
    return checks


def generate_stage1(project_root: Path, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Generate, verify, and serialize all requested Stage-1 data artifacts."""

    output_dir = project_root / "output" / "stage1_model_a"
    report_dir = project_root / "reports" / "stage1_model_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sampled = {
        "physical_train": sample_physical_initial_states(20_000, SEEDS["physical_train"]),
        "physical_validation": sample_physical_initial_states(
            4_000, SEEDS["physical_validation"]
        ),
        "physical_sealed_test": sample_physical_initial_states(
            4_000, SEEDS["physical_sealed_test"]
        ),
    }
    collar_020, collar_025 = sample_paired_exterior_training()
    sampled.update(
        {
            "exterior_train_0p20": collar_020,
            "exterior_train_0p25": collar_025,
            "exterior_validation": sample_exterior_initial_states(
                2_000, SEEDS["exterior_validation"], 1.25
            ),
            "exterior_sealed_stress": sample_exterior_initial_states(
                2_000, SEEDS["exterior_sealed_stress"], 1.25
            ),
        }
    )

    # Exact initial-state overlap checks happen before the expensive integrations.
    physical_names = ("physical_train", "physical_validation", "physical_sealed_test")
    physical_keys = {name: _initial_state_keys(sampled[name]) for name in physical_names}
    overlaps = {
        f"{left}__{right}": len(physical_keys[left] & physical_keys[right])
        for i, left in enumerate(physical_names)
        for right in physical_names[i + 1 :]
    }
    if any(overlaps.values()):
        raise Stage1BoundaryError(
            {"summary": "sampled physical splits overlap", "overlap_counts": overlaps}
        )

    datasets: dict[str, dict[str, np.ndarray]] = {}
    try:
        for name, values in sampled.items():
            progress(f"Generating {name} ({values['x'].size} rows)")
            datasets[name] = add_targets(name, values, progress)
    except Stage1BoundaryError as exc:
        diagnostic_path = report_dir / "BOUNDARY_DIAGNOSTIC.json"
        diagnostic_path.write_text(
            json.dumps(exc.diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    artifacts: dict[str, Any] = {}
    for name, values in datasets.items():
        artifacts[name] = save_dataset(output_dir / f"{name}.npz", values)
        artifacts[name]["domain"] = DATASET_SPECS[name]["domain"]
        artifacts[name]["sealed"] = name in (
            "physical_sealed_test",
            "exterior_sealed_stress",
        )

    normalization = normalization_statistics(datasets["physical_train"])
    normalization_path = output_dir / "normalization.json"
    normalization_path.write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    target_checks = explicit_target_checks(datasets)
    target_checks_path = report_dir / "explicit_target_checks.json"
    target_checks_path.write_text(
        json.dumps(target_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "stage": "Stage 1: fixed-step Model-A datasets only",
        "status": "generated_and_verified",
        "parameters": {
            "b0": 1.0,
            "m": 2,
            "A": -2.0,
            "W": 1.0,
            "S": 0.25,
            "h": H,
            "X": X,
            "X_c": X_C,
            "S_esc": S_ESC,
        },
        "numeric_dtype": "float64",
        "solver": PRODUCTION_SOLVER.metadata(),
        "reference_integrator": "DOP853 via validated wormhole_sciml integration APIs",
        "physics_source_manifest": source_manifest(project_root),
        "physics_gate": {
            "path": "reports/physics_gate/gate_results.json",
            "exterior_contract": (
                "The passed gate certifies h=0.2 finite-step continuation from "
                "x in [-17,17] and initial |xi| through 1.25."
            ),
        },
        "seeds": SEEDS,
        "sampling": {
            "x_mixture": "exactly 50% U[-8.5,8.5] and 50% U[-17,17]",
            "physical_strata": {
                "core": {"abs_xi": "[0,0.5)", "fraction": 0.30},
                "shoulder": {"abs_xi": "[0.5,0.9)", "fraction": 0.35},
                "edge": {"abs_xi": "[0.9,0.99]", "fraction": 0.35},
            },
            "sign_balance": "exactly 50% negative and 50% positive xi",
            "paired_collars": {
                "seed": SEEDS["exterior_paired_train"],
                "same_fields": ["x", "x_component", "xi_sign", "r", "pair_id"],
                "mapping_0p20": "abs(xi)=1.01+r*(1.20-1.01)",
                "mapping_0p25": "abs(xi)=1.01+r*(1.25-1.01)",
            },
        },
        "artifacts": artifacts,
        "physical_summaries": {
            name: physical_summary(datasets[name]) for name in physical_names
        },
        "physical_initial_state_overlap_counts": overlaps,
        "normalization": {
            "path": str(normalization_path),
            "file_sha256": file_sha256(normalization_path),
            "source_dataset_only": "physical_train",
        },
        "explicit_target_checks": {
            "path": str(target_checks_path),
            "count": len(target_checks),
            "max_abs_error_delta_x": max(row["abs_error_delta_x"] for row in target_checks),
            "max_abs_error_delta_u": max(row["abs_error_delta_u"] for row in target_checks),
            "sealed_sets_used": False,
        },
        "boundary_and_integration_anomalies": {
            "count": 0,
            "resampling_or_clipping_used": False,
        },
        "sealed_handling": {
            "scientifically_inspected": False,
            "permitted_checks_only": [
                "row count",
                "schema/dtype",
                "finite values",
                "initial-domain membership",
                "initial-state non-overlap",
                "content/file hashes",
            ],
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress(f"Wrote manifest: {manifest_path}")
    return manifest
