# Reproducibility scripts

Run scripts from the repository root with `PYTHONPATH=src`.  The names below
follow the retained experiment record; public terminology is defined in
`docs/model_definitions.md`.

## Physics and short-step data

- `validate_baseline.py`, `run_physics_gate.py`, `make_figures.py`: validate
  equations, integration, admissibility, and reference physics.
- `generate_stage1_data.py`, `prepare_model_a_xi_data.py`: generate the local
  short-step data and transform it to \((x,\xi)\).

## Local-model progression

- `run_round1_1000.py`, `evaluate_round1_1000.py`,
  `evaluate_round1_1000_rollouts.py`: reproduce the controlled early local
  models and recursive evaluations.
- `run_model_a_architectures.py`, `evaluate_model_a_xi_rollouts.py`,
  `evaluate_model_a_architecture_rollouts.py`, and
  `plot_model_a_phase_space_rollouts.py`: reproduce the A64/C32x32 capacity and
  coordinate comparison.
- `diagnose_round1_local_restart.py`: build the signed-error map and exact-state
  restart/error-growth evidence.
- `plot_c32x32_traversal_families.py`: inspect the strongest pre-energy local
  architecture across trajectory families.
- `prepare_model_a_x_xi_outer_sampling.py`,
  `prepare_model_a_x_xi_outer_microcore_sampling.py`,
  `train_model_a_x_xi_sampling_comparison.py`, and
  `evaluate_model_a_x_xi_sampling_rollouts.py`: reproduce the retained targeted
  sampling transition.
- `train_model_a_x_xi_energy_microcore40k.py` and
  `evaluate_model_a_x_xi_energy_microcore40k.py`: reproduce the selected
  fixed-conserved-energy local model and its recursive rollouts.
- `evaluate_energy_gradient_alignment.py`,
  `train_model_a_fixed_E0_energy_normal.py`,
  `evaluate_model_a_fixed_E0_energy_normal.py`,
  `evaluate_energy_normal_trajectory_postmortem.py`,
  `evaluate_sensitivity_map_physics_gate.py`,
  `train_model_a_fixed_E0_sensitivity_aware.py`, and
  `evaluate_model_a_fixed_E0_sensitivity_aware.py`: reproduce the two retained
  physically motivated local-loss tests and their diagnostics.

## Direct finite-time model

- `generate_phase_b_orbit_banks.py`: generate complete, independently split
  orbit banks.
- `generate_phase_c_finite_time_dataset.py`: sample finite-time transitions
  from the saved orbits without reintegration.
- `run_finite_time_baseline.py`, `run_finite_time_rate.py`, and
  `run_finite_time_hybrid.py`: train the accumulated-residual, average-rate,
  and time-rescaled target formulations.  The last filename retains the old
  internal name.
- `evaluate_finite_time_trajectories.py` and
  `evaluate_finite_time_hybrid_dense.py`: run trajectory-level and dense direct
  evaluation.
- `audit_finite_time_xi_gate.py`: audit the continuous time-rescaling function.
- `evaluate_direct_vs_recursive_rollouts.py`: compare intended direct use with
  recursive local prediction and the diagnostic self-composition control.

## Validation-only follow-ups and sealed test

- `audit_finite_time_hybrid_derivatives.py`,
  `calibrate_finite_time_derivative_gradients.py`,
  `run_finite_time_derivative_loss_experiment.py`, and
  `audit_finite_time_phase_space.py`: reproduce derivative-aware and
  tangent/orbit-normal diagnostics on validation/reference data only.
- `run_finite_time_split_head_experiment.py` and
  `analyze_finite_time_split_head_experiment.py`: reproduce the late split-head
  output-specialization control on validation/reference data only.
- `evaluate_final_sealed_test.py`: one-time evaluation of the frozen selected
  shared-head time-rescaled model on 1,024 held-out parent trajectories.
- `build_public_figures.py`: convert retained outputs into the compact README
  figure set.

## Intentionally omitted development scripts

The public whitelist excludes the superseded 500-epoch round, the generic raw
\((x,u,E_0)\) energy-input trial, redundant coordinate comparisons, contact
sheets, experimental calibration branches, and narrow visualization-only
postmortems.  They remain part of the private development record but are not
needed to reproduce the report's claims.
