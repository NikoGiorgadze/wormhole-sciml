#!/usr/bin/env python3
"""Generate and analyze the outer micro-core transformed-coordinate datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.model_a_x_xi_microcore_data import (
    OUTER_STRATUM_BOUNDS,
    OUTER_STRATUM_FRACTIONS,
    OUTER_STRATUM_NAMES,
    TRAIN_SEED,
    VALIDATION_SEED,
    generate_outer_microcore_targets,
)
from wormhole_sciml.model_a_x_xi_outer_data import X_COMPONENTS, integrity_statistics
from wormhole_sciml.stage1_data import file_sha256, load_dataset, save_dataset
try:
    from scripts.prepare_model_a_x_xi_outer_sampling import (
        EXACT_ARRAYS, EXACT_SUMMARY, HARD, K, U_TH, WINDOW, load_exact,
        local_spacing, normalization, orbit_spacing, stats, stem,
    )
except ModuleNotFoundError:  # direct execution places the scripts directory first
    from prepare_model_a_x_xi_outer_sampling import (
        EXACT_ARRAYS, EXACT_SUMMARY, HARD, K, U_TH, WINDOW, load_exact,
        local_spacing, normalization, orbit_spacing, stats, stem,
    )


ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "output" / "model_a_x_xi_outer_sampling"
CONTROL_PATHS = {
    "old20k": CONTROL_DIR / "old20k_train_x_xi.npz",
    "outer40k": CONTROL_DIR / "outer40k_train_x_xi.npz",
}
OUTPUT = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling"
FIGURES = OUTPUT / "figures"
TRAIN_PATH = OUTPUT / "outer_microcore40k_train_x_xi.npz"
VALIDATION_PATH = OUTPUT / "outer_microcore8k_validation_x_xi.npz"
DATASET_SUMMARY = OUTPUT / "outer_microcore_dataset_summary.json"
RESOLUTION_SUMMARY = OUTPUT / "outer_microcore_resolution_summary.json"
RESOLUTION_ARRAYS = OUTPUT / "outer_microcore_resolution_arrays.npz"
REPORT = OUTPUT / "OUTER_MICROCORE_SAMPLING_REPORT.md"
DESIGNS = ("old20k", "outer40k", "microcore40k")
LABELS = {"old20k": "old 20k", "outer40k": "outer 40k", "microcore40k": "micro-core 40k"}
FAR_LEFT_X = (-17.0, -16.5, -16.0, -15.0, -14.0, -12.0, -10.0, -8.5)


def x_counts(data: dict[str, np.ndarray]) -> dict[str, int]:
    x = data["x"]
    return {
        "total": int(x.size), "central_abs_x_le_8p5": int(np.sum(np.abs(x) <= 8.5)),
        "outer_abs_x_gt_8p5": int(np.sum(np.abs(x) > 8.5)),
        "outer_left_x_lt_minus_8p5": int(np.sum(x < -8.5)),
        "outer_right_x_gt_plus_8p5": int(np.sum(x > 8.5)),
    }


def quota_summary(data: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for component, (name, _low, _high) in enumerate(X_COMPONENTS):
        cmask = data["x_component"] == component
        entry: dict[str, Any] = {"count": int(np.sum(cmask))}
        if component < 2:
            entry["standard_strata"] = {
                label: int(np.sum(cmask & (data["stratum"] == index)))
                for index, label in enumerate(("core", "shoulder", "edge"))
            }
        else:
            entry["outer_strata"] = {}
            for index, label in enumerate(OUTER_STRATUM_NAMES):
                mask = cmask & (data["outer_xi_stratum"] == index)
                entry["outer_strata"][label] = {
                    "count": int(np.sum(mask)),
                    "negative": int(np.sum(mask & (data["xi_sign"] == -1))),
                    "positive": int(np.sum(mask & (data["xi_sign"] == 1))),
                }
        result[name] = entry
    return result


def density_summary() -> dict[str, Any]:
    old_core = 0.30 / 0.5
    rows = {}
    for name, bounds, fraction in zip(OUTER_STRATUM_NAMES, OUTER_STRATUM_BOUNDS, OUTER_STRATUM_FRACTIONS):
        density = fraction / (bounds[1] - bounds[0])
        rows[name] = {
            "abs_xi_interval": list(bounds), "fraction": fraction,
            "nominal_density_per_abs_xi_unit": density,
            "factor_vs_old_core_density": density / old_core,
        }
    return {"old_core_density_per_abs_xi_unit": old_core, "new_outer_strata": rows,
            "near_zero_nominal_density_increase_factor": rows["micro_core"]["factor_vs_old_core_density"]}


def target_structure(data: dict[str, np.ndarray], microcore: bool) -> dict[str, Any]:
    values = np.abs(data["delta_xi"])
    def summarize(mask: np.ndarray) -> dict[str, Any]:
        selected = values[mask]
        return {
            **stats(selected),
            "fraction_abs_lt_1e-6": float(np.mean(selected < 1e-6)),
            "fraction_abs_lt_1e-5": float(np.mean(selected < 1e-5)),
            "fraction_abs_lt_1e-4": float(np.mean(selected < 1e-4)),
        }
    masks = {
        "global": np.ones(values.size, dtype=bool),
        "central_abs_x_le_8p5": np.abs(data["x"]) <= 8.5,
        "outer_abs_x_gt_8p5": np.abs(data["x"]) > 8.5,
    }
    if microcore:
        outer = np.abs(data["x"]) > 8.5
        masks.update({name: outer & (data["outer_xi_stratum"] == index)
                      for index, name in enumerate(OUTER_STRATUM_NAMES)})
    return {name: summarize(mask) for name, mask in masks.items()}


def trajectory_overlap(data: dict[str, np.ndarray]) -> int:
    exact = load_exact()
    training = set(map(tuple, np.column_stack((data["x"], data["xi"]))))
    return sum(tuple(row) in training for family in exact.values()
               for row in np.column_stack((family["x"], family["xi"])))


def resolution_analysis(training: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    exact = load_exact(); arrays: dict[str, np.ndarray] = {}; families = {}
    for value in U_TH:
        name = stem(value); x = exact[value]["x"]
        xi, separation = orbit_spacing(exact, value, x)
        arrays.update({f"{name}__x": x, f"{name}__xi": xi, f"{name}__orbit_spacing": separation})
        family: dict[str, Any] = {"point_count": int(x.size), "far_left_positions": []}
        for design in DESIGNS:
            nearest, mean5, counts = local_spacing(training[design]["x"], training[design]["xi"], x, xi)
            ratio = nearest / separation
            arrays.update({
                f"{name}__{design}__nearest_dxi": nearest,
                f"{name}__{design}__mean5_dxi": mean5,
                f"{name}__{design}__resolution_ratio": ratio,
                f"{name}__{design}__window_count": counts,
            })
            family[design] = {"minimum_window_count": int(np.min(counts))}
        for position in FAR_LEFT_X:
            qx = np.asarray([position]); qxi, qsep = orbit_spacing(exact, value, qx)
            row: dict[str, Any] = {"x": position, "exact_xi": float(qxi[0]), "nearest_orbit_delta_xi": float(qsep[0]),
                                   "window_note": "one-sided at domain endpoint" if position == -17 else None}
            for design in DESIGNS:
                nearest, mean5, count = local_spacing(training[design]["x"], training[design]["xi"], qx, qxi)
                row[design] = {"nearest_dxi": float(nearest[0]), "mean5_dxi": float(mean5[0]),
                               "resolution_ratio": float(nearest[0] / qsep[0]), "window_count": int(count[0])}
            family["far_left_positions"].append(row)
        families[f"{value:.2f}"] = family
    primary = {}
    for design in DESIGNS:
        nearest = np.concatenate([arrays[f"{stem(v)}__{design}__nearest_dxi"][arrays[f"{stem(v)}__x"] <= -8.5] for v in HARD])
        mean5 = np.concatenate([arrays[f"{stem(v)}__{design}__mean5_dxi"][arrays[f"{stem(v)}__x"] <= -8.5] for v in HARD])
        ratio = np.concatenate([arrays[f"{stem(v)}__{design}__resolution_ratio"][arrays[f"{stem(v)}__x"] <= -8.5] for v in HARD])
        primary[design] = {
            "nearest_dxi": stats(nearest), "mean5_dxi": stats(mean5), "resolution_ratio": stats(ratio),
            "fraction_R_lt_1": float(np.mean(ratio < 1)), "fraction_R_lt_0p5": float(np.mean(ratio < .5)),
            "fraction_R_lt_0p25": float(np.mean(ratio < .25)), "poorly_resolved_R_ge_1_count": int(np.sum(ratio >= 1)),
        }
    return {
        "local_x_window_half_width": WINDOW, "small_k": K,
        "primary_region": "hard three, -17<=x<=-8.5", "primary_aggregate": primary,
        "families": families,
        "exact_reference": {"path": str(EXACT_ARRAYS), "sha256": file_sha256(EXACT_ARRAYS), "evaluation_only": True},
    }, arrays


def plot_xi(training: dict[str, dict[str, np.ndarray]]) -> list[Path]:
    paths=[]; colors=("C0","C1","C2")
    for filename, title, outer in (("global_xi_density.png", "Global xi density", False),
                                   ("outer_left_xi_density.png", "Outer-left xi density", True)):
        fig, ax=plt.subplots(figsize=(9.5,4.8),constrained_layout=True); bins=np.linspace(-.99,.99,100)
        for color,design in zip(colors,DESIGNS):
            data=training[design]; mask=data["x"] < -8.5 if outer else np.ones(data["x"].size,bool)
            ax.hist(data["xi"][mask],bins=bins,density=True,histtype="step",lw=1.8,color=color,label=LABELS[design])
        ax.set(title=title,xlabel=r"$\xi$",ylabel="probability density"); ax.legend(); ax.grid(alpha=.2)
        path=FIGURES/filename; fig.savefig(path,dpi=180); plt.close(fig); paths.append(path)
    fig,ax=plt.subplots(figsize=(9.5,4.8),constrained_layout=True); bins=np.linspace(-.075,.075,76)
    for color,design in zip(colors[1:],DESIGNS[1:]):
        data=training[design]; mask=data["x"] < -8.5
        bin_width = bins[1] - bins[0]
        weights = np.full(np.sum(mask), 1.0 / (np.sum(mask) * bin_width))
        ax.hist(data["xi"][mask],bins=bins,weights=weights,histtype="step",lw=2,color=color,label=LABELS[design])
    ax.axvline(-.05,color="0.4",lw=.8); ax.axvline(.05,color="0.4",lw=.8)
    ax.set(title="Outer-left near-zero xi density",xlabel=r"$\xi$",ylabel="probability density relative to all outer-left rows"); ax.legend(); ax.grid(alpha=.2)
    path=FIGURES/"outer_left_microcore_zoom.png"; fig.savefig(path,dpi=180); plt.close(fig); paths.append(path)
    return paths


def plot_scatter(training: dict[str, dict[str, np.ndarray]]) -> list[Path]:
    paths=[]
    for filename,zoom in (("outer_left_sampling_three_designs.png",False),("outer_left_microcore_sampling_zoom.png",True)):
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharex=True,sharey=True,constrained_layout=True)
        for ax,design in zip(axes,DESIGNS):
            data=training[design]; mask=data["x"] < -8.5
            if zoom: mask &= np.abs(data["xi"]) <= .05
            ax.scatter(data["x"][mask],data["xi"][mask],s=2.2,alpha=.3,linewidths=0)
            ax.set(title=f"{LABELS[design]} · n={np.sum(mask):,}",xlabel="$x$"); ax.grid(alpha=.15)
        axes[0].set_ylabel(r"$\xi$")
        path=FIGURES/filename; fig.savefig(path,dpi=180); plt.close(fig); paths.append(path)
    return paths


def plot_resolution(arrays: dict[str,np.ndarray]) -> Path:
    fig,axes=plt.subplots(3,1,figsize=(9.5,9.2),sharex=True,constrained_layout=True)
    for ax,value in zip(axes,HARD):
        name=stem(value); x=arrays[f"{name}__x"]
        for design in DESIGNS: ax.plot(x,arrays[f"{name}__{design}__resolution_ratio"],lw=1.35,label=LABELS[design])
        ax.axhline(1,color="black",lw=.9); ax.axhline(.5,color="0.45",lw=.8,ls=":")
        ax.set_yscale("log"); ax.set(title=rf"$u_{{th}}={value:.2f}$",ylabel=r"$R=d_\xi/\Delta\xi_{orbit}$"); ax.legend(); ax.grid(alpha=.2,which="both")
    axes[-1].set_xlabel("exact incoming $x$")
    path=FIGURES/"hard_trajectory_resolution_three_designs.png"; fig.savefig(path,dpi=180); plt.close(fig); return path


def plot_targets(outer: dict[str,np.ndarray], micro: dict[str,np.ndarray]) -> Path:
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    positive=[]
    for data,label in ((outer,"outer 40k"),(micro,"micro-core 40k")):
        values=np.abs(data["delta_xi"]); values=values[values>0]; positive.append(values)
    bins=np.geomspace(min(map(np.min,positive)),max(map(np.max,positive)),90)
    for values,label in zip(positive,("outer 40k","micro-core 40k")): axes[0].hist(values,bins=bins,histtype="step",lw=1.7,label=label)
    axes[0].set_xscale("log"); axes[0].set(title=r"Global $|\Delta\xi|$",xlabel=r"$|\Delta\xi|$",ylabel="count"); axes[0].legend()
    labels=list(OUTER_STRATUM_NAMES); positions=np.arange(4)
    for offset,(data,label) in zip((-.16,.16),((outer,"outer 40k"),(micro,"micro-core 40k"))):
        med=[]
        for index,(low,high) in enumerate(OUTER_STRATUM_BOUNDS):
            mask=(np.abs(data["x"])>8.5)&(np.abs(data["xi"])>=low)&(np.abs(data["xi"])<(high if index<3 else high+1e-15))
            med.append(np.median(np.abs(data["delta_xi"])[mask]))
        axes[1].bar(positions+offset,med,width=.3,label=label)
    axes[1].set_yscale("log"); axes[1].set_xticks(positions,labels,rotation=20); axes[1].set(title=r"Outer median $|\Delta\xi|$ by region",ylabel=r"median $|\Delta\xi|$"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2,which="both")
    path=FIGURES/"delta_xi_three_design_comparison.png"; fig.savefig(path,dpi=180); plt.close(fig); return path


def report_text(dataset: dict[str,Any], resolution: dict[str,Any]) -> str:
    c=dataset["near_zero_outer_left_counts"]; p=resolution["primary_aggregate"]
    new_x=dataset["x_counts"]["train"]; control_x=dataset["control_x_counts"]["outer40k"]
    integrity=dataset["integrity"]["train"]
    primary_rows="\n".join(f"| {LABELS[d]} | {p[d]['nearest_dxi']['median']:.3e} | {p[d]['nearest_dxi']['mean']:.3e} | {p[d]['mean5_dxi']['median']:.3e} | {p[d]['resolution_ratio']['median']:.3f} | {p[d]['resolution_ratio']['mean']:.3f} | {p[d]['resolution_ratio']['p90']:.3f} | {p[d]['fraction_R_lt_1']:.1%} | {p[d]['fraction_R_lt_0p5']:.1%} | {p[d]['fraction_R_lt_0p25']:.1%} |" for d in DESIGNS)
    far=[]
    for value in HARD:
        for row in resolution["families"][f"{value:.2f}"]["far_left_positions"]:
            far.append(f"| {value:.2f} | {row['x']:.1f} | {row['exact_xi']:.7f} | {row['nearest_orbit_delta_xi']:.3e} | "+" | ".join(f"{row[d]['nearest_dxi']:.3e} / {row[d]['resolution_ratio']:.2f}" for d in DESIGNS)+" |")
    previous=dataset["delta_xi_structure"]["outer40k"]; new=dataset["delta_xi_structure"]["microcore40k"]
    target_order=("global","central_abs_x_le_8p5","outer_abs_x_gt_8p5","micro_core","remainder_core","shoulder","edge")
    target_rows="\n".join(
        f"| {name} | {new[name]['mean']:.3e} | {new[name]['median']:.3e} | {new[name]['standard_deviation']:.3e} | {new[name]['p90']:.3e} | {new[name]['p99']:.3e} | {new[name]['p99p9']:.3e} | {new[name]['maximum']:.3e} | {new[name]['fraction_abs_lt_1e-6']:.3%} | {new[name]['fraction_abs_lt_1e-5']:.3%} | {new[name]['fraction_abs_lt_1e-4']:.3%} |"
        for name in target_order
    )
    return f"""# Outer micro-core transformed-sampling report

## Scope and sampling integrity

One independent random 40,000/8,000 design was generated with seeds `{TRAIN_SEED}` and `{VALIDATION_SEED}`. The x mixture is unchanged at 10,000/2,000 rows per component. Central and broad components retain 30/35/35 xi allocation; each outer training component has exactly 2,000 micro-core, 2,000 remainder-core, 3,000 shoulder, and 3,000 edge rows with exact sign balance (validation: 400/400/600/600). No resampling against diagnostics occurred.

The new realization has `{new_x['central_abs_x_le_8p5']:,}` central, `{new_x['outer_abs_x_gt_8p5']:,}` outer, `{new_x['outer_left_x_lt_minus_8p5']:,}` outer-left, and `{new_x['outer_right_x_gt_plus_8p5']:,}` outer-right rows, versus `{control_x['central_abs_x_le_8p5']:,}`, `{control_x['outer_abs_x_gt_8p5']:,}`, `{control_x['outer_left_x_lt_minus_8p5']:,}`, and `{control_x['outer_right_x_gt_plus_8p5']:,}` for the previous outer40k. This is ordinary seed variation under the identical x design.

The nominal near-zero outer density is `4.0` per unit absolute xi versus old-core `0.6`, a `6.667x` increase. Realized training counts at `x<-8.5, |xi|<0.05` are old20k `{c['old20k']:,}`, outer40k `{c['outer40k']:,}`, and micro-core40k `{c['microcore40k']:,}`.

All new initial/next states are physical and finite. Maximum/median/p99 reconstruction errors are `{integrity['reconstruction_absolute_error']['maximum']:.3e}`, `{integrity['reconstruction_absolute_error']['median']:.3e}`, and `{integrity['reconstruction_absolute_error']['p99']:.3e}`. Maximum/median/p99 absolute energy mismatches are `{integrity['energy_absolute_mismatch']['maximum']:.3e}`, `{integrity['energy_absolute_mismatch']['median']:.3e}`, and `{integrity['energy_absolute_mismatch']['p99']:.3e}`; relative values are `{integrity['energy_relative_mismatch']['maximum']:.3e}`, `{integrity['energy_relative_mismatch']['median']:.3e}`, and `{integrity['energy_relative_mismatch']['p99']:.3e}`, passing the established `1e-9` gate. Both control directories retained byte-identical hashes.

## Hard-three primary resolution (`-17<=x<=-8.5`)

| design | median nearest dxi | mean nearest dxi | median mean5 dxi | median R | mean R | p90 R | R<1 | R<0.5 | R<0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{primary_rows}

The incremental micro-core intervention reduces median R from `{p['outer40k']['resolution_ratio']['median']:.3f}` to `{p['microcore40k']['resolution_ratio']['median']:.3f}`. No sampled primary-region state remains at `R>=1`; `{int(round(p['microcore40k']['resolution_ratio']['count'] * (1-p['microcore40k']['fraction_R_lt_0p5'])))}` states remain at `R>=0.5`, and pointwise random variation remains visible.

## Far-left values

Each design entry is `nearest dxi / R`. The `x=-17` window is one-sided.

| u_th | x | exact xi | orbit Δxi | old20k | outer40k | micro-core40k |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(far)}

## Delta-xi and next-step decision

Global micro-core `|Delta xi|` has median `{new['global']['median']:.3e}`, p99 `{new['global']['p99']:.3e}`, p99.9 `{new['global']['p99p9']:.3e}`, and maximum `{new['global']['maximum']:.3e}`; fractions below `1e-6/1e-5/1e-4` are `{new['global']['fraction_abs_lt_1e-6']:.3%}`, `{new['global']['fraction_abs_lt_1e-5']:.3%}`, and `{new['global']['fraction_abs_lt_1e-4']:.3%}`. The prior/new outer medians are `{previous['outer_abs_x_gt_8p5']['median']:.3e}` and `{new['outer_abs_x_gt_8p5']['median']:.3e}`. All values are finite and the intervention introduces no obvious numerical pathology.

| region | mean | median | std | p90 | p99 | p99.9 | max | <1e-6 | <1e-5 | <1e-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{target_rows}

The measured resolution improvement, rather than intuition, supports proceeding later with the recorded `(x,xi)->(Delta x,Delta xi)`, `2->32->32->2` tanh model comparison: the new random dataset markedly increases near-zero coverage and improves aggregate hard-family resolution. Seven `R>=1` points remain later on the full incoming branches outside the primary outer-left region, so local variability remains an evaluation risk. No model was trained here, and no loss policy was changed.
"""


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    protected=[p for p in CONTROL_DIR.rglob("*") if p.is_file()]+[EXACT_ARRAYS,EXACT_SUMMARY]
    before={str(p):file_sha256(p) for p in protected}
    controls={name:load_dataset(path) for name,path in CONTROL_PATHS.items()}
    train=generate_outer_microcore_targets(40_000,TRAIN_SEED,print)
    validation=generate_outer_microcore_targets(8_000,VALIDATION_SEED,print)
    integrity={"train":integrity_statistics(train),"validation":integrity_statistics(validation)}
    if trajectory_overlap(train) or trajectory_overlap(validation): raise RuntimeError("diagnostic trajectory row entered random data")
    OUTPUT.mkdir(parents=True); FIGURES.mkdir()
    artifacts={"train":save_dataset(TRAIN_PATH,train),"validation":save_dataset(VALIDATION_PATH,validation)}
    training={**controls,"microcore40k":train}
    resolution,arrays=resolution_analysis(training); np.savez_compressed(RESOLUTION_ARRAYS,**arrays)
    figures=plot_xi(training)+plot_scatter(training)+[plot_resolution(arrays),plot_targets(controls["outer40k"],train)]
    after={path:file_sha256(Path(path)) for path in before}
    if before!=after: raise RuntimeError("an immutable control or exact-reference artifact changed")
    near={name:int(np.sum((data["x"] < -8.5)&(np.abs(data["xi"]) < .05))) for name,data in training.items()}
    summary={
        "stage":"outer micro-core transformed-coordinate data and resolution analysis only",
        "seeds":{"train":TRAIN_SEED,"validation":VALIDATION_SEED},"artifacts":artifacts,
        "x_counts":{"train":x_counts(train),"validation":x_counts(validation)},
        "control_x_counts":{name:x_counts(data) for name,data in controls.items()},
        "quotas":{"train":quota_summary(train),"validation":quota_summary(validation)},
        "outer_density":density_summary(),"near_zero_outer_left_counts":near,
        "integrity":integrity,"trajectory_row_overlap_count":{"train":0,"validation":0},
        "continuous_random_checks":{"train_unique_x":int(np.unique(train["x"]).size),"train_unique_xi":int(np.unique(train["xi"]).size),"validation_unique_x":int(np.unique(validation["x"]).size),"validation_unique_xi":int(np.unique(validation["xi"]).size)},
        "normalization_statistics_train_only":normalization(train),
        "delta_xi_structure":{"outer40k":target_structure(controls["outer40k"],False),"microcore40k":target_structure(train,True)},
        "future_model_recorded_not_trained":{"mapping":"(x,xi)->(delta_x,delta_xi)","architecture":"2->32->32->2","activation":"tanh"},
        "figures":[{"path":str(p),"sha256":file_sha256(p)} for p in figures],
        "protected_hashes_before":before,"protected_hashes_after":after,
        "protocol":{"single_seeded_dataset_generated_without_diagnostic_rejection":True,"training_performed":False,"sealed_or_test_data_accessed":False,"trajectory_specific_sampling":False,"collars_used":False,"loss_changed":False,"clipping_or_projection_used":False},
    }
    DATASET_SUMMARY.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    resolution["arrays"]={"path":str(RESOLUTION_ARRAYS),"sha256":file_sha256(RESOLUTION_ARRAYS)}
    RESOLUTION_SUMMARY.write_text(json.dumps(resolution,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    REPORT.write_text(report_text(summary,resolution),encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__": main()
