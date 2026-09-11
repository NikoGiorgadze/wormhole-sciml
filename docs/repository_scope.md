# Public-release scope

The repository is the reproducibility companion to the scientific report.  It
contains enough material to inspect and rerun the evidence chain, but it is not
an archive of every development experiment.

| Repository item | Release classification | Reason |
|---|---|---|
| `src/wormhole_sciml/` physics and coordinate modules | Essential for reproducibility | Defines the equations, admissibility region, conserved energy, integration, and \(u\leftrightarrow\xi\) map. |
| `src/wormhole_sciml/` local and finite-time model modules | Essential for reproducibility | Defines model interfaces, target construction, training, prediction, exact identity, derivative loss, and phase-space diagnostics. |
| Selected scripts in `scripts/` | Essential for reproducibility | Reconstruct the retained progression from physics through local and finite-time models to the sealed test. |
| `tests/` | Essential for reproducibility | Checks the independent physics identities, coordinate round trip, numerical integration, model parameter counts and interfaces, exact identity, differentiation, and tangent/normal decomposition. |
| `results/` compact CSV/JSON files | Essential for claim traceability | Supports numerical claims without publishing row-level predictions. |
| `figures/` | Useful supporting material | Gives a small visual account of trajectory geometry, finite-time prediction, and horizon-dependent direct-versus-recursive behavior. |
| `docs/` | Essential for reproducibility | Fixes conventions, definitions, terminology, generation order, selection rules, and the release boundary. |
| Final scientific-report PDF | Essential scientific narrative | Provides the complete study design, results, qualifications, and interpretation linked from the top of the README. |
| Development-only scripts omitted by `.gitignore` | Useful only historically | Earlier 500-epoch runs, redundant raw-coordinate/energy trials, contact sheets, and narrow postmortems helped development but do not define retained claims. |
| Generated datasets, checkpoints, histories, dense arrays, and per-row tables | Unnecessary for the public release | They are large, reproducible, and less useful than the generators plus compact summaries. |
| Draft reports and private references | Unnecessary for the public release | The versioned final PDF carries the scientific narrative; drafts and source archives are not release artifacts. |

## Figure policy

The README displays three figures: transformed trajectory-family geometry, an
actual difficult-family finite-time prediction, and direct-versus-recursive
error growth.  Ranking bars and multi-metric dashboards are excluded.  Extra
report figures should be committed only when they carry spatial, temporal, or
geometric structure that a short table cannot replace.

The signed local-error map, exact-state restart diagnostic, final local
trajectory/energy panels, finite-time trajectory comparisons, and
tangent/orbit-normal diagnostic are therefore appropriate report companions.
Architecture rankings, target-treatment comparisons, derivative-loss numbers,
and shared-versus-split metrics remain compact tables unless a later plot adds
qualitative information.
