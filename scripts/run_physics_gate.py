#!/usr/bin/env python3
"""Run and serialize the revised Simple GEB physics-only gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wormhole_sciml.physics_gate import run_physics_gate


def write_markdown(result: dict, path: Path) -> None:
    force = result["compact_domain_force_envelope"]
    reconnaissance = result["trajectory_reconnaissance"]
    step = result["learning_step_calibration"]
    mapping = result["package_mapping"]
    parameters = mapping["parameters"]
    design = result["learning_step_design"]
    lines = [
        f"# Physics-only gate: {result['status']}",
        "",
        "## Outcome",
        "",
    ]
    if result["status"] == "BLOCKED":
        lines.extend(
            (
                f"Blocked at `{result['blocker']['stage']}`: "
                f"{result['blocker']['condition']}.",
                "",
                "No exterior gate certification, rollout suites, final datasets, "
                "or ML work was performed after this stop condition.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "All physics-only gate conditions passed. Final dataset generation "
                "and ML work remain deliberately out of scope.",
                "",
            )
        )
    lines.extend(
        (
            "## Package mapping and conventions",
            "",
            f"- Package version: `{result['package_version']}`",
            f"- Source-manifest SHA-256: `{result['physics_source_manifest']['sha256']}`",
            f"- Coordinates: {mapping['coordinates']}",
            f"- Fixed parameters: throat_radius={parameters['throat_radius']}, "
            f"m={parameters['m']}, alpha={parameters['alpha']}, "
            f"omega={parameters['omega']}, theta={parameters['theta']}",
            f"- R_m: `{mapping['R_m']}`",
            f"- F: `{mapping['F']}`",
            f"- Physical flow: `{mapping['physical_flow']}`",
            f"- Exterior continuation: `{mapping['exterior_flow']}`",
            f"- C: `{mapping['C']}`",
            f"- xi: `{mapping['xi']}`",
            f"- Energy convention: `{mapping['energy']}`",
            "",
            "## Solver settings",
            "",
            f"- Production: {result['solvers']['production']}",
            f"- Validation: {result['solvers']['validation']}",
            f"- Tight: {result['solvers']['tight']}",
            "",
            "## Physics and compact domain",
            "",
            f"- L: {force['L']}",
            f"- Certification extent: {force['certification_extent']}",
            f"- F*: {force['F_star']:.17g}",
            f"- Terminal acceleration threshold 0.05 F*: {0.05 * force['F_star']:.17g}",
            f"- X_acc: {force['X_acc']:.17g}",
            f"- X_term: {reconnaissance['X_term']:.17g}",
            f"- X: {reconnaissance['X']}",
            f"- X_c: {reconnaissance['X_c']:.17g}",
            "",
            "## Reference-class reconnaissance",
            "",
            f"- Escaping/force-free: {reconnaissance['counts']['escaping_force_free']}",
            f"- Null-boundary-asymptotic: {reconnaissance['counts']['null_boundary_asymptotic']}",
            f"- Unresolved: {reconnaissance['counts']['unresolved']}",
            "",
            "## Learning-step candidates",
            "",
            f"- Disposable design: count={design['count']}, seed={design['seed']}, SHA-256=`{design['state_sha256']}`",
            "",
            "| h | q_h | rho_99,h | prod-tight dx | prod-tight du | tight-two-half dx | tight-two-half du | epsilon_ref | nontrivial | local | abs conv | rel conv | all |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|",
        )
    )
    for row in step["candidates"]:
        components = row["componentwise_max_abs"]
        lines.append(
            f"| {row['h']} | {row['q_h']:.12g} | {row['rho_99_h']:.12g} | "
            f"{components['production_vs_tight']['x']:.12g} | "
            f"{components['production_vs_tight']['u']:.12g} | "
            f"{components['tight_vs_two_half']['x']:.12g} | "
            f"{components['tight_vs_two_half']['u']:.12g} | "
            f"{row['epsilon_ref']:.12g} | {row['passes_nontriviality']} | "
            f"{row['passes_locality']} | {row['passes_convergence_absolute']} | "
            f"{row['passes_convergence_relative']} | {row['passes_all']} |"
        )
    lines.extend(
        (
            "",
            f"Selected h: {step['selected_h']}",
            "",
        )
    )
    if result["status"] == "PASSED":
        exterior = result["exterior_certification"]
        regression = result["integrator_contract_regression"]
        suites = result["rollout_suites"]
        lines.extend(
            (
                "## Exterior-flow certification",
                "",
                "- API: `integrate_vector_field_continuation`",
                f"- Status: {exterior['status']}",
                f"- State count: {exterior['design_state_count']}",
                f"- xi range: [{exterior['grid']['xi_range'][0]}, {exterior['grid']['xi_range'][1]}]",
                f"- Maximum |F| on certification grid: {exterior['grid']['max_abs_F']:.17g}",
                f"- RMS increment norm: {exterior['rms_increment_norm']:.17g}",
                f"- epsilon_ref: {exterior['epsilon_ref']:.17g}",
                f"- Absolute convergence: {exterior['passes_absolute']}",
                f"- Relative convergence: {exterior['passes_relative']}",
                "- Exterior continuations are mathematical vector-field continuations, not physical particle trajectories.",
                "",
                "## Integrator contract regression",
                "",
                f"- Status: {regression['status']}",
                f"- Physical-region equivalence component maxima: {regression['physical_vs_continuation_max_abs']}",
                f"- Exterior probe C0: {regression['exterior_probe']['C0']:.17g}",
                f"- Physical API rejection: `{regression['exterior_probe']['physical_rejection']}`",
                f"- Continuation endpoint: {regression['exterior_probe']['continuation_final']}",
                "",
                "## Rollout suites",
                "",
                f"- S_esc: {suites['S_esc']:.17g}",
                f"- Global minimum reference C: {suites['global_minimum_reference_C']:.17g}",
                f"- Reference invariant-error floor: {suites['reference_invariant_error_floor']:.17g}",
                f"- Validation categories: {suites['validation']['category_counts']}",
                f"- Validation reference classes: {suites['validation']['reference_class_counts']}",
                f"- Validation exclusive escaping statuses through T_i: {suites['validation']['exclusive_escaping_rollout_status_counts']}",
                f"- Validation exclusive boundary statuses through T_i: {suites['validation']['exclusive_boundary_rollout_status_counts']}",
                f"- Validation near-edge classes: {suites['validation']['near_edge_reference_class_counts']}",
                f"- Sealed-test categories: {suites['sealed_test']['category_counts']}",
                f"- Sealed-test reference classes: {suites['sealed_test']['reference_class_counts']}",
                f"- Sealed-test exclusive escaping statuses through T_i: {suites['sealed_test']['exclusive_escaping_rollout_status_counts']}",
                f"- Sealed-test exclusive boundary statuses through T_i: {suites['sealed_test']['exclusive_boundary_rollout_status_counts']}",
                f"- Sealed-test near-edge classes: {suites['sealed_test']['near_edge_reference_class_counts']}",
                f"- Combined categories: {suites['combined_counts']['category_counts']}",
                f"- Combined reference classes: {suites['combined_counts']['reference_class_counts']}",
                f"- Combined exclusive escaping statuses through T_i: {suites['combined_counts']['exclusive_escaping_rollout_status_counts']}",
                f"- Combined exclusive boundary statuses through T_i: {suites['combined_counts']['exclusive_boundary_rollout_status_counts']}",
                "- `terminal_entry_time_by_T_i` searches only [0,T_i]; `right_censored_at_T_i` means no sustained terminal entry was observed on that frozen interval.",
                "- `classification_*_terminal_entry_time` is separate class-reconnaissance evidence evaluated through its recorded survey cap (normally 100, at most 200), with spatial survey boundary |x|=40.",
                "",
                "| split | id | category | class | x0 | xi0 | T_i | boundary event | event time | terminal entry by T_i | min C | exclusive rollout status |",
                "|:--|:--|:--|:--|--:|--:|--:|:--|--:|--:|--:|:--|",
            )
        )
        for split_name in ("validation", "sealed_test"):
            for member in suites[split_name]["members"]:
                event_time = (
                    ""
                    if member["tau_data_ref"] is None
                    else f"{member['tau_data_ref']:.12g}"
                )
                terminal_entry = (
                    ""
                    if member["terminal_entry_time_by_T_i"] is None
                    else f"{member['terminal_entry_time_by_T_i']:.12g}"
                )
                rollout_status = (
                    member["exclusive_escaping_rollout_status"]
                    or member["exclusive_boundary_rollout_status"]
                )
                lines.append(
                    f"| {split_name} | {member['id']} | {member['category']} | "
                    f"{member['reference_class']} | {member['x0']:.12g} | "
                    f"{member['xi0']:.12g} | {member['T_i']:.12g} | "
                    f"{member['tau_data_event'] or ''} | {event_time} | "
                    f"{terminal_entry} | {member['minimum_reference_C']:.12g} | "
                    f"{rollout_status} |"
                )
        lines.extend(("", "## Required follow-up", "", "None.", ""))
    else:
        lines.extend(
            (
                "## Required follow-up",
                "",
                result["blocker"]["smallest_follow_up"],
                "",
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_tables(result: dict, output_dir: Path) -> None:
    with (output_dir / "h_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "h",
            "q_h",
            "rho_99_h",
            "rms_increment_norm",
            "epsilon_ref",
            "production_vs_tight_x_max_abs",
            "production_vs_tight_u_max_abs",
            "tight_vs_two_half_x_max_abs",
            "tight_vs_two_half_u_max_abs",
            "production_vs_tight_normalized_max",
            "tight_vs_two_half_normalized_max",
            "passes_nontriviality",
            "passes_locality",
            "passes_convergence_absolute",
            "passes_convergence_relative",
            "passes_all",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["learning_step_calibration"]["candidates"]:
            components = row["componentwise_max_abs"]
            normalized = row["normalized_max"]
            output = {name: row[name] for name in fieldnames if name in row}
            output.update(
                {
                    "production_vs_tight_x_max_abs": components[
                        "production_vs_tight"
                    ]["x"],
                    "production_vs_tight_u_max_abs": components[
                        "production_vs_tight"
                    ]["u"],
                    "tight_vs_two_half_x_max_abs": components[
                        "tight_vs_two_half"
                    ]["x"],
                    "tight_vs_two_half_u_max_abs": components[
                        "tight_vs_two_half"
                    ]["u"],
                    "production_vs_tight_normalized_max": normalized[
                        "production_vs_tight"
                    ],
                    "tight_vs_two_half_normalized_max": normalized[
                        "tight_vs_two_half"
                    ],
                }
            )
            writer.writerow(output)

    with (output_dir / "trajectory_reconnaissance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "id",
            "source",
            "x0",
            "xi0",
            "u0",
            "C0",
            "class",
            "tolerance_class_consistent",
            "tight_event",
            "tight_event_time",
            "tight_endpoint_x",
            "tight_endpoint_u",
            "tight_endpoint_C",
            "tight_minimum_C",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["trajectory_reconnaissance"]["table"]:
            writer.writerow(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "x0": row["x0"],
                    "xi0": row["xi0"],
                    "u0": row["u0"],
                    "C0": row["C0"],
                    "class": row["class"],
                    "tolerance_class_consistent": row["tolerance_class_consistent"],
                    "tight_event": row["tight"]["event"],
                    "tight_event_time": row["tight"]["event_time"],
                    "tight_endpoint_x": row["tight"]["endpoint"]["x"],
                    "tight_endpoint_u": row["tight"]["endpoint"]["u"],
                    "tight_endpoint_C": row["tight"]["endpoint"]["C"],
                    "tight_minimum_C": row["tight"]["minimum_C"],
                }
            )

    if result["status"] == "PASSED":
        with (output_dir / "rollout_suites.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = [
                "split",
                "id",
                "category",
                "presentation_role",
                "origin",
                "reference_class",
                "x0",
                "xi0",
                "u0",
                "C0",
                "T_i",
                "step_count",
                "minimum_horizon_required",
                "tau_data_ref",
                "tau_data_event",
                "minimum_reference_C",
                "terminal_entry_time_by_T_i",
                "terminal_entry_search_end_T_i",
                "exclusive_escaping_rollout_status",
                "exclusive_boundary_rollout_status",
                "classification_spatial_boundary_abs_x",
                "classification_production_time_domain_end",
                "classification_tight_time_domain_end",
                "classification_production_terminal_entry_time",
                "classification_tight_terminal_entry_time",
                "reference_integration_success_through_T_i",
                "reference_states_finite_through_T_i",
                "reference_physical_through_T_i",
                "throat_crossing_time",
                "reference_energy_relative_error_max",
                "classification_tolerance_consistent",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            suites = result["rollout_suites"]
            for split_name in ("validation", "sealed_test"):
                for member in suites[split_name]["members"]:
                    output = {
                        name: member[name]
                        for name in fieldnames
                        if name in member
                    }
                    output["terminal_entry_search_end_T_i"] = member[
                        "terminal_entry_search_interval"
                    ][1]
                    output["classification_production_time_domain_end"] = member[
                        "classification_production_time_domain"
                    ][1]
                    output["classification_tight_time_domain_end"] = member[
                        "classification_tight_time_domain"
                    ][1]
                    writer.writerow(output)

        with (output_dir / "rollout_suite_counts.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = ["scope", "dimension", "label", "count"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for scope in ("validation", "sealed_test", "combined"):
                counts = (
                    suites["combined_counts"]
                    if scope == "combined"
                    else suites[scope]
                )
                for dimension in (
                    "category_counts",
                    "reference_class_counts",
                    "exclusive_escaping_rollout_status_counts",
                    "exclusive_boundary_rollout_status_counts",
                ):
                    for label, count in counts[dimension].items():
                        writer.writerow(
                            {
                                "scope": scope,
                                "dimension": dimension,
                                "label": label,
                                "count": count,
                            }
                        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/physics_gate"))
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    result = run_physics_gate(project_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "rollout_suites.json").write_text(
        json.dumps(result["rollout_suites"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv_tables(result, args.output_dir)
    write_markdown(result, args.output_dir / "PHYSICS_GATE_REPORT.md")
    print(json.dumps({
        "status": result["status"],
        "blocker": result.get("blocker"),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
