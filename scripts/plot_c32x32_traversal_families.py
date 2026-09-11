#!/usr/bin/env python3
"""Evaluation-only C32x32 phase-space traversal-family diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from wormhole_sciml.dynamics import (
    conserved_energy,
    radial_system,
    timelike_margin,
    velocity_bounds,
    velocity_from_energy,
)
from wormhole_sciml.model_a import (
    Normalization,
    load_trained_model,
    parameter_count,
    predict_increments,
)
from wormhole_sciml.physics_gate import experiment_parameters, xi_from_state
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "c32x32_traversal_families"
MODEL_DIR = ROOT / "output" / "model_a_architecture_comparison" / "c32x32" / "physical_only"
CHECKPOINTS = {seed: MODEL_DIR / f"seed_{seed}" / "best_checkpoint.pt" for seed in (101, 202, 303)}
METADATA = {seed: MODEL_DIR / f"seed_{seed}" / "metadata.json" for seed in (101, 202, 303)}
THROAT_VELOCITIES = (0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.90)
CROSS_CHECKS = {
    0.05: 0.498297776,
    0.15: 0.498435325,
    0.30: 0.498869321,
    0.50: 0.500000000,
    0.65: 0.501707114,
    0.80: 0.506169002,
    0.90: 0.527163032,
}
SEED_COLORS = {101: "#277da1", 202: "#f8961e", 303: "#9b5de5"}
X_ENDPOINT = 17.0
H = 0.2
MAX_MODEL_STEPS = 10_000
MAX_EXACT_TIME = 2_000.0
EXACT_SAMPLE_STEP = 0.05
THROAT_TOLERANCE = 1e-9


def exact_left_velocity(u_th: float) -> tuple[float, float]:
    wormhole, spiral = experiment_parameters()
    energy = float(conserved_energy(0.0, u_th, wormhole, spiral))
    u_left = float(velocity_from_energy(-X_ENDPOINT, energy, wormhole, spiral, branch=1))
    return u_left, energy


def _endpoint_event(_time: float, state: np.ndarray, *_args: Any) -> float:
    return float(state[0] - X_ENDPOINT)


_endpoint_event.terminal = True  # type: ignore[attr-defined]
_endpoint_event.direction = 1.0  # type: ignore[attr-defined]


def _null_event(_time: float, state: np.ndarray, wormhole: Any, spiral: Any) -> float:
    return float(timelike_margin(state[0], state[1], wormhole, spiral))


_null_event.terminal = True  # type: ignore[attr-defined]
_null_event.direction = -1.0  # type: ignore[attr-defined]


def _throat_event(_time: float, state: np.ndarray, *_args: Any) -> float:
    return float(state[0])


_throat_event.terminal = False  # type: ignore[attr-defined]
_throat_event.direction = 1.0  # type: ignore[attr-defined]


def exact_rollout(initial: np.ndarray, *, verify_throat: bool) -> tuple[np.ndarray, dict[str, Any]]:
    wormhole, spiral = experiment_parameters()
    events = (_endpoint_event, _null_event, _throat_event) if verify_throat else (
        _endpoint_event,
        _null_event,
    )
    result = solve_ivp(
        radial_system,
        (0.0, MAX_EXACT_TIME),
        np.asarray(initial, dtype=np.float64),
        args=(wormhole, spiral),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        max_step=H,
        dense_output=True,
        events=events,
    )
    if not result.success:
        raise RuntimeError(f"DOP853 traversal failed: {result.message}")
    endpoint_event, null_event = result.t_events[:2]
    reached = bool(endpoint_event.size)
    exited = bool(null_event.size)
    status = "reached_x_plus_17" if reached else "physical_exit" if exited else "maximum_time_guard"
    terminal_time = float(result.t[-1])
    sample_times = np.arange(0.0, terminal_time, EXACT_SAMPLE_STEP, dtype=np.float64)
    sample_times = np.append(sample_times, terminal_time)
    path = np.asarray(result.sol(sample_times).T, dtype=np.float64)
    summary: dict[str, Any] = {
        "method": "DOP853",
        "status": status,
        "reached_x_plus_17": reached,
        "physical_exit": exited,
        "final_x": float(path[-1, 0]),
        "final_u": float(path[-1, 1]),
        "elapsed_time": terminal_time,
        "solver_steps": int(result.t.size - 1),
        "plotted_points": int(path.shape[0]),
    }
    if verify_throat:
        throat_events = result.t_events[2]
        throat_states = result.y_events[2]
        if throat_events.size != 1:
            raise RuntimeError(f"expected one forward throat crossing, found {throat_events.size}")
        summary["throat_crossing_time"] = float(throat_events[0])
        summary["throat_crossing_u"] = float(throat_states[0, 1])
    return path, summary


def learned_rollout(model: Any, normalization: Normalization, initial: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    wormhole, spiral = experiment_parameters()
    states = [np.asarray(initial, dtype=np.float64)]
    status = "maximum_step_guard"
    for _step in range(1, MAX_MODEL_STEPS + 1):
        current = states[-1]
        xi = float(xi_from_state(current[0], current[1], wormhole, spiral))
        inputs = np.array([[current[0], current[1], xi]], dtype=np.float64)
        following = current + predict_increments(model, inputs, normalization)[0]
        states.append(following)
        if not np.all(np.isfinite(following)):
            status = "nonfinite"
            break
        if float(timelike_margin(following[0], following[1], wormhole, spiral)) <= 0.0:
            status = "physical_exit"
            break
        if following[0] >= X_ENDPOINT:
            status = "reached_x_plus_17"
            break
    path = np.asarray(states, dtype=np.float64)
    summary = {
        "status": status,
        "reached_x_plus_17": status == "reached_x_plus_17",
        "physical_exit": status == "physical_exit",
        "final_x": float(path[-1, 0]),
        "final_u": float(path[-1, 1]),
        "rollout_steps": int(path.shape[0] - 1),
        "elapsed_time": float((path.shape[0] - 1) * H),
    }
    crossing = np.flatnonzero((path[:-1, 0] < 0.0) & (path[1:, 0] >= 0.0))
    if crossing.size:
        index = int(crossing[0])
        fraction = float(-path[index, 0] / (path[index + 1, 0] - path[index, 0]))
        summary["throat_crossing_step"] = index + 1
        summary["interpolated_throat_u"] = float(
            path[index, 1] + fraction * (path[index + 1, 1] - path[index, 1])
        )
    else:
        summary["throat_crossing_step"] = None
        summary["interpolated_throat_u"] = None
    return path, summary


def _panel_limits(paths: list[np.ndarray], x_range: tuple[float, float]) -> tuple[float, float]:
    wormhole, spiral = experiment_parameters()
    x = np.linspace(x_range[0], x_range[1], 1601)
    lower, upper = velocity_bounds(x, wormhole, spiral)
    values = [lower, upper, *(path[:, 1] for path in paths if np.all(np.isfinite(path[:, 1])))]
    low, high = min(float(np.min(value)) for value in values), max(float(np.max(value)) for value in values)
    padding = 0.06 * (high - low)
    return low - padding, high + padding


def plot_case(u_th: float, u_left: float, panels: dict[str, Any], path: Path) -> None:
    wormhole, spiral = experiment_parameters()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    specifications = (
        ("full", (-X_ENDPOINT, X_ENDPOINT), "start at $(-17,u_{left})$, evolve forward"),
        ("throat", (0.0, X_ENDPOINT), "start at $(0,u_{th})$, evolve forward"),
    )
    for axis, (name, x_range, title) in zip(axes, specifications):
        exact = panels[name]["exact_path"]
        learned = panels[name]["learned_paths"]
        grid = np.linspace(x_range[0], x_range[1], 1601)
        lower, upper = velocity_bounds(grid, wormhole, spiral)
        axis.fill_between(grid, lower, upper, color="#c9d7e3", alpha=0.28,
                          label="exact admissible region", zorder=0)
        axis.plot(exact[:, 0], exact[:, 1], color="black", lw=2.8, label="DOP853", zorder=2)
        all_paths = [exact]
        for seed, learned_path in learned.items():
            terminal = panels[name]["learned_summaries"][seed]
            label = f"seed {seed} · {terminal['status'].replace('_', ' ')}"
            axis.plot(learned_path[:, 0], learned_path[:, 1], color=SEED_COLORS[seed],
                      lw=1.45, label=label, zorder=4)
            all_paths.append(learned_path)
        for rollout, color in [(exact, "black"), *[(learned[s], SEED_COLORS[s]) for s in learned]]:
            axis.scatter(rollout[0, 0], rollout[0, 1], marker="o", s=35,
                         facecolor="white", edgecolor=color, linewidth=1.2, zorder=6)
            axis.scatter(rollout[-1, 0], rollout[-1, 1], marker="s", s=34,
                         facecolor=color, edgecolor="white", linewidth=0.6, zorder=6)
        for seed, learned_path in learned.items():
            if panels[name]["learned_summaries"][seed]["physical_exit"]:
                axis.scatter(learned_path[-1, 0], learned_path[-1, 1], marker="X", s=95,
                             color="red", edgecolor="white", linewidth=0.7, zorder=8)
        axis.set(
            title=title,
            xlabel="$x$",
            ylabel="radial velocity $u$",
            xlim=(x_range[0] - 0.35, x_range[1] + 0.35),
            ylim=_panel_limits(all_paths, x_range),
        )
        axis.grid(alpha=0.16)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle(
        rf"C32×32 traversal family: $u_{{th}}={u_th:.2f}$, exact $u(-17)={u_left:.9f}$",
        fontsize=15,
    )
    fig.savefig(path, dpi=190)
    plt.close(fig)


def make_report(cases: list[dict[str, Any]], summary_path: Path) -> str:
    full_failures, throat_failures = [], []
    for case in cases:
        for seed, row in case["full_traversal"]["learned"].items():
            if not row["reached_x_plus_17"]:
                full_failures.append((case["u_th"], seed, row["status"]))
        for seed, row in case["throat_started_outgoing"]["learned"].items():
            if not row["reached_x_plus_17"]:
                throat_failures.append((case["u_th"], seed, row["status"]))
    mismatch = max(abs(case["exact_throat_verification_mismatch"]) for case in cases)
    table_rows = []
    for case in cases:
        full = case["full_traversal"]["learned"]
        outgoing = case["throat_started_outgoing"]["learned"]
        crossing_errors = [
            abs(row["interpolated_throat_u_error"])
            for row in full.values()
            if row.get("interpolated_throat_u_error") is not None
        ]
        table_rows.append(
            f"| {case['u_th']:.2f} | {sum(row['reached_x_plus_17'] for row in full.values())}/3 | "
            f"{sum(row['physical_exit'] for row in full.values())}/3 | "
            f"{max(crossing_errors):.4f} | "
            f"{sum(row['reached_x_plus_17'] for row in outgoing.values())}/3 | "
            f"{sum(row['physical_exit'] for row in outgoing.values())}/3 |"
        )
    def describe(rows: list[tuple[float, str, str]]) -> str:
        return "none" if not rows else ", ".join(
            f"$u_{{th}}={u:.2f}$ seed {seed} ({status.replace('_', ' ')})"
            for u, seed, status in rows
        )
    return f"""# C32×32 traversal-family diagnostic

This evaluation-only diagnostic compares the exact DOP853 trajectories with forward recursive Model-A C32×32 rollouts for seeds 101, 202, and 303. No training, clipping, projection, or backward-time learned construction was used.

## Exact initialization verification

All seven left-end states were obtained from the validated conserved-energy branch. Direct DOP853 integration from $x=-17$ crossed $x=0$ with a maximum absolute throat-velocity mismatch of `{mismatch:.3e}`, passing the `{THROAT_TOLERANCE:.1e}` tolerance.

## Learned traversal outcomes

- Full left-to-right failures: {describe(full_failures)}.
- Throat-started outgoing failures: {describe(throat_failures)}.

| $u_{{th}}$ | Full reached | Full exits | Largest full $|\\Delta u_{{th}}|$ among crossings | Outgoing reached | Outgoing exits |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The hardest families are the low-throat-velocity cases $u_{{th}}=0.05$ and $0.15$: seed 303 exits before reaching the throat, while the successful full traversals can cross the throat with large velocity errors. The $u_{{th}}=0.30$ family still has strong seed-dependent throat under/overshoot. The discrepancy contracts through $u_{{th}}=0.50$ and $0.65$; the $0.80$ and $0.90$ families closely reproduce the exact full phase-space branch.

All 21 throat-started learned rollouts reach $x=+17$ without a physical exit and visually track DOP853 closely. Therefore the severe low-$u_{{th}}$ failures are primarily generated while recursively traversing the incoming branch, rather than being an intrinsic inability to evolve the exact outgoing throat state. Seed 202 tends to overshoot the low/intermediate throat velocities, while seed 303 undershoots and is the only seed to exit in this diagnostic; seed 101 shows smaller positive throat-velocity bias. Seed-specific terminal states and interpolated learned throat velocities are retained in [{summary_path.name}]({summary_path.name}).

## Figures

Seven two-panel figures are stored alongside this report. The left panel starts at the exact energy-matched state $(-17,u_{{left}})$; the right panel starts directly at $(0,u_{{th}})$. Black is DOP853, and the three solid colored curves are the stored C32×32 seeds. Red X markers indicate first detected physical exits.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace this script's output files")
    args = parser.parse_args()
    if OUTPUT_DIR.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    metadata = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in METADATA.items()}
    normalization_paths = {Path(row["normalization"]) for row in metadata.values()}
    if len(normalization_paths) != 1:
        raise RuntimeError("C32x32 seeds do not share one normalization artifact")
    normalization_path = normalization_paths.pop()
    normalization = Normalization.from_stage1(
        normalization_path,
        input_columns=("x", "u", "xi"),
        expected_source_dataset="physical_train_x_u_xi",
    )
    protected = [*CHECKPOINTS.values(), *METADATA.values(), normalization_path]
    before = {str(path): file_sha256(path) for path in protected}
    models = {seed: load_trained_model(checkpoint) for seed, checkpoint in CHECKPOINTS.items()}
    if any(parameter_count(model) != 1250 for model in models.values()):
        raise RuntimeError("a checkpoint is not the prescribed 1250-parameter C32x32 model")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=args.overwrite)
    cases = []
    for u_th in THROAT_VELOCITIES:
        u_left, energy = exact_left_velocity(u_th)
        if abs(u_left - CROSS_CHECKS[u_th]) > 5e-10:
            raise RuntimeError(f"u_left cross-check failed for u_th={u_th}")
        full_exact, full_exact_summary = exact_rollout(
            np.array([-X_ENDPOINT, u_left]), verify_throat=True
        )
        mismatch = full_exact_summary["throat_crossing_u"] - u_th
        if abs(mismatch) > THROAT_TOLERANCE:
            raise RuntimeError(f"exact throat verification failed for u_th={u_th}: {mismatch}")
        throat_exact, throat_exact_summary = exact_rollout(
            np.array([0.0, u_th]), verify_throat=False
        )
        panels: dict[str, Any] = {}
        case: dict[str, Any] = {
            "u_th": u_th,
            "energy": energy,
            "u_left": u_left,
            "u_left_cross_check": CROSS_CHECKS[u_th],
            "u_left_cross_check_difference": u_left - CROSS_CHECKS[u_th],
            "exact_throat_verification_mismatch": mismatch,
        }
        for name, initial, exact_path, exact_summary in (
            ("full", np.array([-X_ENDPOINT, u_left]), full_exact, full_exact_summary),
            ("throat", np.array([0.0, u_th]), throat_exact, throat_exact_summary),
        ):
            learned_paths, learned_summaries = {}, {}
            for seed, model in models.items():
                learned_paths[seed], learned_summaries[seed] = learned_rollout(
                    model, normalization, initial
                )
                if name == "full" and learned_summaries[seed]["interpolated_throat_u"] is not None:
                    learned_summaries[seed]["interpolated_throat_u_error"] = (
                        learned_summaries[seed]["interpolated_throat_u"] - u_th
                    )
            panels[name] = {
                "exact_path": exact_path,
                "learned_paths": learned_paths,
                "learned_summaries": learned_summaries,
            }
            case["full_traversal" if name == "full" else "throat_started_outgoing"] = {
                "initial_x": float(initial[0]),
                "initial_u": float(initial[1]),
                "exact": exact_summary,
                "learned": {str(seed): row for seed, row in learned_summaries.items()},
            }
        filename = f"c32x32_traversal_u_th_{u_th:.2f}".replace(".", "p") + ".png"
        figure_path = OUTPUT_DIR / filename
        plot_case(u_th, u_left, panels, figure_path)
        case["figure"] = {"path": str(figure_path), "sha256": file_sha256(figure_path)}
        cases.append(case)
        print(f"completed u_th={u_th:.2f}", flush=True)
    after = {str(path): file_sha256(path) for path in protected}
    if before != after:
        raise RuntimeError("a checkpoint, metadata file, or normalization artifact changed")
    summary_path = OUTPUT_DIR / "traversal_family_summary.json"
    summary = {
        "stage": "evaluation-only C32x32 traversal-family diagnostic",
        "architecture": "3->32->32->2 with tanh hidden layers",
        "parameter_count": 1250,
        "seeds": list(CHECKPOINTS),
        "step_size": H,
        "x_endpoint": X_ENDPOINT,
        "maximum_model_steps": MAX_MODEL_STEPS,
        "exact_solver": {"method": "DOP853", "rtol": 1e-11, "atol": 1e-13, "max_step": H},
        "throat_verification_tolerance": THROAT_TOLERANCE,
        "checkpoints": {str(seed): {"path": str(path), "sha256": before[str(path)]}
                        for seed, path in CHECKPOINTS.items()},
        "normalization": {"path": str(normalization_path), "sha256": before[str(normalization_path)]},
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "cases": cases,
        "protocol": {
            "training_performed": False,
            "checkpoint_modified": False,
            "backward_time_learned_rollout": False,
            "clipping_or_projection_performed": False,
            "sealed_or_test_data_accessed": False,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = OUTPUT_DIR / "TRAVERSAL_FAMILY_REPORT.md"
    report_path.write_text(make_report(cases, summary_path), encoding="utf-8")
    print(f"Wrote {summary_path} and {report_path}")


if __name__ == "__main__":
    main()
