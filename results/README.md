# Compact result record

These files are the numerical bridge between the experiment archive and the
scientific report.  They contain aggregate or seed-level values only; no
training data, checkpoint tensors, or row-level predictions are committed.

| File | Claim supported | Authoritative generated source |
|---|---|---|
| `local/selected_model.json` | Exact architecture, protocol, selected seed, and checkpoint identity of the strongest local model | `output/model_a_x_xi_energy_microcore40k_comparison/energy_xi_training_manifest.json` |
| `local/energy_input_comparison.csv` | Fixed conserved energy changes little locally but strongly improves difficult recursive rollouts | `output/model_a_x_xi_energy_microcore40k_comparison/energy_xi_rollout_summary.json` and associated report |
| `local/loss_comparison.csv` | Energy-normal and sensitivity-aware local objectives improve their targeted diagnostics without improving the decisive hard rollout | The two retained fixed-\(E_0\) loss reports and summaries |
| `dataset/orbit_and_transition_counts.csv` | Whole-orbit separation and exact train/validation/test row counts | Complete-orbit-bank and finite-time-dataset manifests |
| `finite_time/selected_model.json` | Selected time-rescaled model definition and checkpoint identity | `output/finite_time_hybrid_s5/finite_time_hybrid_summary.json` and training metadata |
| `finite_time/three_way_comparison.csv` | Accumulated-residual, average-rate, and time-rescaled target comparison | `output/finite_time_hybrid_s5/validation/three_way_comparison.csv` |
| `direct_vs_recursive/crossover_summary.json` | Horizon- and region-dependent direct/recursive crossover behavior | `output/direct_vs_recursive_rollouts/tables/crossover_summary.json` |
| `derivative_loss/aggregate_treatment_comparison.csv` | Validation-only derivative-loss comparison | `output/finite_time_hybrid_derivative_loss_experiment/tables/aggregate_treatment_comparison.csv` |
| `split_head/paired_architecture_differences.csv` | Validation-only shared-versus-late-split control | `output/finite_time_split_head_architecture_comparison/tables/paired_architecture_differences.csv` |
| `sealed_test/validation_vs_test.csv` | Agreement between validation and the held-out 1,024-orbit test | `output/final_sealed_test/tables/validation_vs_test.csv` |

The identifier `hybrid_s5` inside retained tables is the historical artifact
name for the public **time-rescaled finite-time model** with \(s_\star=5\).
