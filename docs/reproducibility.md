# Reproducibility guide

## What this release reproduces

The public repository contains the physical equations, coordinate maps,
dataset generators, model definitions, training programs, diagnostic
evaluators, compact results, and README-figure builder.  It intentionally does
not contain generated trajectory banks, finite-time row arrays, checkpoints,
or row-level predictions.

The fast test suite is self-contained:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The heavier experiment scripts write to `output/`, which is ignored by Git.
Run them from the repository root with `PYTHONPATH=src`.  All scripts resolve
their input and output paths relative to the repository rather than the shell's
current directory.

## Reproduction order

1. Validate the equations and numerical solver with `validate_baseline.py` and
   `run_physics_gate.py`.
2. Generate the short-step data, prepare the transformed coordinate, and run
   the local-model progression listed in `scripts/README.md`.
3. Generate the complete orbit banks and then the finite-time transition data.
4. Train the accumulated-residual, average-rate, and time-rescaled finite-time
   formulations in that order.
5. Run dense trajectory evaluation and the direct-versus-recursive study.
6. Run derivative-aware and split-head studies only as validation-only
   follow-ups.
7. Run the sealed-test script only after all selections are frozen.
8. Build the public figures with `python scripts/build_public_figures.py`.

Several scripts check hashes of prerequisite artifacts.  A failure is usually
evidence that the requested experiment does not match the retained protocol,
not an instruction to bypass the check.

## Determinism and selection

The principal neural-network seeds are 101, 202, and 303.  Python, NumPy, and
PyTorch are seeded; deterministic PyTorch algorithms and one CPU thread are
used by the retained training functions.  Dataset generators have their own
fixed seeds recorded in their manifests.

Model selection is validation-only.  The local fixed-energy primary model is
seed 101.  The selected time-rescaled finite-time model is seed 202 with
checkpoint SHA-256
`a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323`.
The one-time sealed test covers only this selected shared-head finite-time
model; later derivative-loss and split-head experiments were not independently
confirmed on that test split.

## Regenerating public figures

`scripts/build_public_figures.py` reads retained experiment outputs and writes
only the compact images committed under `figures/`.  The difficult
finite-time trajectory comes from the selected seed-202 checkpoint, and the
horizon plot compares that direct model with the selected seed-101
fixed-energy local model on the common-survivor cohort.

The strongest local-rollout report figure is deliberately not frozen here yet.
Once the scientific report is written, its existing trajectory-and-energy
graphic will be regenerated as two separately titled panels, with a tighter
energy-axis range and no internal sampling terminology in the titles.

## Storage policy

Keep compact CSV and JSON summaries, final publication figures, and manifests
needed to identify selected runs.  Exclude generated `.npy`/`.npz` datasets,
checkpoints, training histories, dense prediction arrays, per-row tables,
cache directories, and the complete private experiment archive.  These files
can be recreated from the public code and are too large or too detailed to
clarify the scientific narrative.
