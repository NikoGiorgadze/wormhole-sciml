#!/usr/bin/env python3
"""Dense direct validation and matched-step maps for the frozen hybrid model."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from wormhole_sciml.finite_time_hybrid import HybridPreprocessing, load_hybrid_model
from wormhole_sciml.finite_time_hybrid_validation import log_error_ratio, median_log_grid, predict_hybrid_diagnostics
from wormhole_sciml.finite_time_validation import (
    DENSE_ANCHOR_X, DENSE_FRACTIONS, LOCAL_STEP, SMALL_S_VALUES,
    admissibility_summary, aggregate_error_metrics, bin_error_rows, energy_metrics,
    predict_local, scalar_error_metrics, subset_metrics,
)
from wormhole_sciml.model_a import Normalization, load_trained_model
from wormhole_sciml.phase_b_orbits import evaluate_saved_orbit_x_u_xi
from wormhole_sciml.phase_c_finite_time import invert_saved_orbit_x
from wormhole_sciml.stage1_data import file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/finite_time_hybrid_dense_validation"
ARRAYS, TABLES, FIGURES, TESTS = (OUTPUT / name for name in ("arrays", "tables", "figures", "tests"))
REPORT = OUTPUT / "HYBRID_DENSE_AND_MATCHED_VALIDATION_REPORT.md"
SUMMARY = OUTPUT / "hybrid_dense_validation_summary.json"
MANIFEST = OUTPUT / "hybrid_dense_validation_manifest.json"
MANIFEST_HASH = OUTPUT / "hybrid_dense_validation_manifest.sha256"
HYBRID_ROOT = ROOT / "output/finite_time_hybrid_s5"
HYBRID_MANIFEST = HYBRID_ROOT / "finite_time_hybrid_manifest.json"
HYBRID_CHECKPOINT = HYBRID_ROOT / "training/seed_202/best_checkpoint.pt"
HYBRID_PREPROCESSING = HYBRID_ROOT / "preprocessing/hybrid_preprocessing_constants.json"
PRIOR_ROOT = ROOT / "output/finite_time_trajectory_validation"
PRIOR_MANIFEST = PRIOR_ROOT / "validation_manifest.json"
PRIOR_SUMMARY = PRIOR_ROOT / "validation_summary.json"
DENSE_QUERIES = PRIOR_ROOT / "arrays/dense_validation_queries.npz"
MATCHED_QUERIES = PRIOR_ROOT / "arrays/matched_local_queries.npz"
SMALL_QUERIES = PRIOR_ROOT / "arrays/small_s_queries.npz"
THREE_WAY_COMPARISON = HYBRID_ROOT / "validation/three_way_comparison.json"
STRESS_QUERIES = PRIOR_ROOT / "arrays/stress_queries.npz"
VALIDATION_BANK = ROOT / "output/phase_b_complete_orbit_banks/banks/phase_b_validation_orbits.npz"
STRESS_BANK = ROOT / "output/phase_b_complete_orbit_banks/banks/phase_b_stress_reference_orbits.npz"
SEALED = ROOT / "output/phase_c_finite_time_dataset/datasets/phase_c_test_sealed_raw.npz"
LOCAL_ROOT = ROOT / "output/model_a_x_xi_energy_microcore40k_comparison"
LOCAL_MANIFEST = LOCAL_ROOT / "energy_xi_training_manifest.json"
LOCAL_CHECKPOINT = LOCAL_ROOT / "training/seed_101/best_checkpoint.pt"
LOCAL_NORMALIZATION = LOCAL_ROOT / "training/energy_input_normalization.json"
EPSILON = 1.0e-12
S_EDGES = np.asarray([0, .01, .025, .05, .1, .2, .5, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100.], float)
X_EDGES = np.linspace(-17., 17., 33)
EXPECTED = {
    HYBRID_MANIFEST: "c95e85729204e4942d4e47d733ff6f15b1ca87c7f1a5ef4414198881d7c8f4b5",
    HYBRID_CHECKPOINT: "a3ae37ead841a1b5a6d6a754052f44aaa64c0ba112035e59584d8b2c64141323",
    HYBRID_PREPROCESSING: "b4bb84535f4e19d58915123afebd7eddf1f1231a1ec221099e339c89aad6eb28",
    PRIOR_MANIFEST: "f47177badc70777325826f09c04d5a64d7889cdcc07aad246c19340988f7c607",
    PRIOR_SUMMARY: "3d9d5065bc06a0e917ea8859e267f08bc9826a0ef545d46f2f76c1ed181ae334",
    DENSE_QUERIES: "dd90249675ae523ba97306122b0a8aebe5f381b0e963791a6932de105ef4b808",
    MATCHED_QUERIES: "e1a0e34f9935abf85a98dcbe22c25a726522124097d71604f7109aeed0b2fae9",
    SMALL_QUERIES: "c7976e6255d9f37ecc11a0adddd81a4cae87a648c535b6d5300c73539cbc13d2",
    THREE_WAY_COMPARISON: "e51c2a1826a710485c5fb88dc90d30ec7a4141fec590edebf68b2148056a1af1",
    STRESS_QUERIES: "6bb106e35d0db9ae43eca07829d07a3ddc49820e013641396b4342577f651490",
    VALIDATION_BANK: "6c61fb2fa125185f96d60b511411b03866d60c10ef2b183064818adfe560d4b8",
    STRESS_BANK: "5d7959cf1a657a5ff916e4d40aab44309ca08958f1694340b47c0fce7ac08dce",
    SEALED: "61c2b38e0e92cedc35fd872cd16e767575c3b3c754ac93851038f021ef702311",
    LOCAL_MANIFEST: "27050145c861fe630cee764fd0766238604bb6c04be335e23186b3ca2b45b5b5",
    LOCAL_CHECKPOINT: "54a41a54a7fa20e931df0dd987ac871786d4ac8d02e71bd3560e9ebc3228a4a6",
    LOCAL_NORMALIZATION: "5f3605f657d8190ea901112774553e3026a3399266783a78f0102b285c4045c3",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def flat(metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{component}_{key}": value for component in ("x", "xi") for key, value in metrics[component].items()}


def gate() -> dict[str, Any]:
    rows, failures = {}, []
    for path, expected in EXPECTED.items():
        measured = file_sha256(path); match = measured == expected
        rows[str(path.resolve())] = {"expected": expected, "measured": measured, "match": match}
        if not match: failures.append(str(path))
    return {"passed": not failures, "failures": failures, "artifacts": rows, "sealed_access": "byte hash only"}


def prediction_table(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ("query_index", "orbit_index", "orbit_id", "u_th", "E0", "x0", "xi0", "t0", "f", "s", "exact_x1", "exact_xi1")
    output = {name: queries[name] for name in names}
    output.update({name: prediction[name] for name in ("predicted_x1", "predicted_xi1", "x_error", "xi_error", "predicted_u1", "predicted_C1", "energy_error")})
    output["absolute_x_error"] = np.abs(prediction["x_error"])
    output["absolute_xi_error"] = np.abs(prediction["xi_error"])
    return output


def coordinate_rows(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, list[dict[str, Any]]]:
    uth_edges = np.linspace(float(queries["u_th"].min()), float(queries["u_th"].max()), 31)
    e_edges = np.unique(np.quantile(queries["E0"], np.linspace(0, 1, 25)))
    return {
        "s": bin_error_rows(queries["s"], prediction, S_EDGES, "s"),
        "x0": bin_error_rows(queries["x0"], prediction, X_EDGES, "x0"),
        "u_th": bin_error_rows(queries["u_th"], prediction, uth_edges, "u_th"),
        "E0": bin_error_rows(queries["E0"], prediction, e_edges, "E0"),
    }


def plot_lines(rows: list[dict[str, Any]], coordinate: str, path: Path, title: str, percentiles: bool = False) -> None:
    if percentiles:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, component in zip(axes, ("x", "xi")):
            for q, style in (("p90", "-"), ("p99", "--")):
                axis.plot([r["center"] for r in rows], [r[f"{component}_{q}_absolute"] for r in rows], ls=style, label=q)
            axis.set(xlabel=coordinate, ylabel=f"absolute {component} error", yscale="log"); axis.grid(alpha=.25); axis.legend()
    else:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
        for axis, (component, metric) in zip(axes.ravel(), (("x","rmse"),("x","mae"),("xi","rmse"),("xi","mae"))):
            axis.plot([r["center"] for r in rows], [r[f"{component}_{metric}"] for r in rows])
            axis.set(ylabel=f"{metric.upper()} {component}", yscale="log"); axis.grid(alpha=.25)
        for axis in axes[-1]: axis.set_xlabel(coordinate)
    figure.suptitle(title); figure.savefig(path, dpi=180); plt.close(figure)


def plot_dense_maps(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], path: Path) -> None:
    s_edges = np.asarray([0,.05,.2,.5,1,2,5,10,20,40,60,100.])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, component in zip(axes, ("x", "xi")):
        grid, _ = median_log_grid(queries["x0"], queries["s"], np.abs(prediction[f"{component}_error"]), X_EDGES, s_edges, EPSILON)
        mesh = axis.pcolormesh(X_EDGES, s_edges, np.ma.masked_invalid(grid), shading="auto", cmap="magma")
        figure.colorbar(mesh, ax=axis, label=f"log10 median |{component} error|")
        axis.set(xlabel="anchor x0", ylabel="physical elapsed time s", title=f"{component} error across anchor position and elapsed time")
    figure.savefig(path, dpi=180); plt.close(figure)


def plot_energy(queries: dict[str, np.ndarray], prediction: dict[str, np.ndarray], coordinates: dict[str, list[dict[str, Any]]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    edges = {"s":S_EDGES,"x0":X_EDGES,"u_th":np.linspace(queries["u_th"].min(),queries["u_th"].max(),31),"E0":np.unique(np.quantile(queries["E0"],np.linspace(0,1,25)))}
    for axis, name in zip(axes.ravel(), ("s","x0","u_th","E0")):
        idx=np.digitize(queries[name],edges[name][1:-1]); centers=[]; values=[]
        for i in range(len(edges[name])-1):
            mask=idx==i
            if np.any(mask): centers.append(float(np.mean(queries[name][mask]))); values.append(float(np.quantile(np.abs(prediction["energy_error"][mask]),.99)))
        axis.plot(centers,values); axis.set(xlabel=name,ylabel="p99 |E_hat-E0|",yscale="log"); axis.grid(alpha=.25)
    figure.suptitle("Energy inconsistency across elapsed time and trajectory coordinates"); figure.savefig(path,dpi=180); plt.close(figure)


def matched_map(queries: dict[str,np.ndarray], local: dict[str,np.ndarray], hybrid: dict[str,np.ndarray], y_name: str, path: Path) -> dict[str,Any]:
    y_edges = np.linspace(float(queries[y_name].min()), float(queries[y_name].max()), 31)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    saved={"epsilon":EPSILON,"x_edges":X_EDGES.tolist(),"y_edges":y_edges.tolist(),"coordinate":y_name}
    for row, component in enumerate(("x","xi")):
        la=np.abs(local[f"{component}_error"]); ha=np.abs(hybrid[f"{component}_error"])
        lg,count=median_log_grid(queries["x0"],queries[y_name],la,X_EDGES,y_edges,EPSILON)
        hg,_=median_log_grid(queries["x0"],queries[y_name],ha,X_EDGES,y_edges,EPSILON)
        rg,_=median_log_grid(queries["x0"],queries[y_name],10**log_error_ratio(ha,la,EPSILON),X_EDGES,y_edges,EPSILON)
        # median_log_grid applies log10; its ratio input is already positive hybrid/local.
        valid=np.concatenate((lg[np.isfinite(lg)],hg[np.isfinite(hg)])); lo,hi=np.quantile(valid,[.01,.99])
        for col,(grid,title,cmap,vmin,vmax) in enumerate(((lg,"local absolute error","magma",lo,hi),(hg,"hybrid absolute error","magma",lo,hi),(rg,"log10 hybrid/local ratio","coolwarm",-3,3))):
            mesh=axes[row,col].pcolormesh(X_EDGES,y_edges,np.ma.masked_invalid(grid),shading="auto",cmap=cmap,vmin=vmin,vmax=vmax)
            figure.colorbar(mesh,ax=axes[row,col]); axes[row,col].set(xlabel="anchor x0",ylabel=y_name,title=f"{component}: {title}")
        saved[component]={"local_log10":lg.tolist(),"hybrid_log10":hg.tolist(),"ratio_log10":rg.tolist(),"counts":count.tolist()}
    figure.suptitle(f"Local and hybrid prediction error at s=0.2 in (x0, {y_name})"); figure.savefig(path,dpi=180); plt.close(figure)
    return saved


def plot_small_s_three_way(small: dict[str, Any], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True, sharex=True)
    labels = {
        "accumulated_residual": "accumulated residual",
        "average_rate": "average rate",
        "hybrid_s5": "hybrid s*=5",
    }
    for model_name, rows in small.items():
        ordered = [rows[key] for key in sorted(rows, key=float)]
        s = [row["s"] for row in ordered]
        for axis, (component, metric) in zip(
            axes.ravel(), (("x", "rmse"), ("x", "mae"), ("xi", "rmse"), ("xi", "mae"))
        ):
            axis.plot(s, [row[f"{component}_{metric}"] for row in ordered], marker="o", ms=3, label=labels[model_name])
            axis.set(ylabel=f"{metric.upper()} {component}", yscale="log")
            axis.grid(alpha=.25)
    for axis in axes[-1]: axis.set_xlabel("physical elapsed time s")
    axes[0, 0].legend()
    figure.suptitle("Frozen three-way comparison on the common small-s bank")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def trajectory_figure(bank: dict[str,np.ndarray], orbit_index: int, model: Any, preprocessing: HybridPreprocessing, path: Path, title: str) -> None:
    anchors=(-14.,-8.,0.); figure,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for row,anchor in enumerate(anchors):
        t0,_=invert_saved_orbit_x(bank,orbit_index,np.asarray([anchor])); elapsed=np.linspace(0,float(bank["t_right"][orbit_index])-float(t0[0]),220)
        exact=evaluate_saved_orbit_x_u_xi(bank,orbit_index,float(t0[0])+elapsed)
        query={"x0":np.full(220,exact[0,0]),"xi0":np.full(220,exact[0,2]),"E0":np.full(220,float(bank["E0"][orbit_index])),"s":elapsed,"exact_x1":exact[:,0],"exact_u1":exact[:,1],"exact_xi1":exact[:,2]}
        pred=predict_hybrid_diagnostics(model,preprocessing,query)
        axes[row,0].plot(elapsed,exact[:,0],label="exact"); axes[row,0].plot(elapsed,pred["predicted_x1"],"--",label="hybrid")
        axes[row,1].plot(elapsed,exact[:,2]); axes[row,1].plot(elapsed,pred["predicted_xi1"],"--")
        axes[row,2].plot(exact[:,0],exact[:,2]); axes[row,2].plot(pred["predicted_x1"],pred["predicted_xi1"],"--")
        axes[row,0].set_ylabel(f"x; anchor {anchor:g}"); axes[row,1].set_ylabel("xi"); axes[row,2].set_ylabel("xi")
        for axis in axes[row]: axis.grid(alpha=.25)
    axes[0,0].legend(); axes[-1,0].set_xlabel("physical elapsed time s"); axes[-1,1].set_xlabel("physical elapsed time s"); axes[-1,2].set_xlabel("x")
    figure.suptitle(title); figure.savefig(path,dpi=175); plt.close(figure)


def run_tests() -> dict[str,Any]:
    command=[sys.executable,"-m","pytest","-q","tests/test_finite_time_hybrid_dense.py","tests/test_finite_time_hybrid.py","tests/test_finite_time_trajectory_validation.py",f"--junitxml={TESTS/'relevant_pytest.xml'}"]
    result=subprocess.run(command,cwd=ROOT,env={**os.environ,"PYTHONPATH":"src","MPLCONFIGDIR":"/private/tmp/wormhole-hybrid-dense-mpl"},capture_output=True,text=True)
    payload={"command":command,"exit_code":result.returncode,"stdout":result.stdout,"stderr":result.stderr,"passed":result.returncode==0}; write_json(TESTS/"test_summary.json",payload); return payload


def report(summary: dict[str,Any]) -> str:
    dense=summary["dense_aggregate"]; old=summary["old_dense_aggregate"]; fam=summary["families"]; energy=summary["energy"]; matched=summary["matched"]
    matched_rows="\n".join(f"| {name} | {row['x']['rmse']:.6g} | {row['x']['mae']:.6g} | {row['x']['p90_absolute']:.6g} | {row['x']['p99_absolute']:.6g} | {row['xi']['rmse']:.6g} | {row['xi']['mae']:.6g} | {row['xi']['p90_absolute']:.6g} | {row['xi']['p99_absolute']:.6g} |" for name,row in matched["aggregate"].items())
    family_rows="\n".join(f"| {name} | {row['x']['rmse']:.6g} | {row['x']['mae']:.6g} | {row['xi']['rmse']:.6g} | {row['xi']['mae']:.6g} |" for name,row in fam.items())
    stress_rows="\n".join(f"| {row['u_th']:.2f} | {row['x_rmse']:.6g} | {row['xi_rmse']:.6g} | {row['energy_mae']:.6g} |" for row in summary["reference_metrics"])
    return f"""# Dense hybrid validation and matched local comparison

## Frozen scope

This is evaluation-only. Hybrid seed 202 uses `Delta_x=s*V_x` and `Delta_xi=-5*expm1(-s/5)*F_xi`. The exact prior 524,288-row dense grid and 32,768-row matched `s=0.2` bank were reused byte-for-byte. No training, normalization change, ODE reintegration, recursive rollout, or sealed-test prediction occurred.

The authoritative local model is seed 101 of the fixed-E0 `3->32->32->2` ordinary standardized-MSE baseline, checkpoint `{summary['local_model']['checkpoint_sha256']}`, at its fixed native interval 0.2.

## Dense aggregate

| weighting | model | RMSE x | MAE x | RMSE xi | MAE xi |
|:---|:---|---:|---:|---:|---:|
| row | hybrid | {dense['row_weighted']['x']['rmse']:.6g} | {dense['row_weighted']['x']['mae']:.6g} | {dense['row_weighted']['xi']['rmse']:.6g} | {dense['row_weighted']['xi']['mae']:.6g} |
| orbit | hybrid | {dense['orbit_weighted']['x']['rmse']:.6g} | {dense['orbit_weighted']['x']['mae']:.6g} | {dense['orbit_weighted']['xi']['rmse']:.6g} | {dense['orbit_weighted']['xi']['mae']:.6g} |
| row | accumulated residual | {old['row_weighted']['x']['rmse']:.6g} | {old['row_weighted']['x']['mae']:.6g} | {old['row_weighted']['xi']['rmse']:.6g} | {old['row_weighted']['xi']['mae']:.6g} |

Full median/p90/p95/p99/p99.9/max metrics and all coordinate bins are in the JSON/CSV artifacts.

The frozen accumulated-residual, average-rate, and hybrid small-s curves are reproduced in `small_s_three_way.png` from the protected prior comparison artifact.

## Regimes and families

| subset | RMSE x | MAE x | RMSE xi | MAE xi |
|:---|---:|---:|---:|---:|
{family_rows}

## Reference trajectories (excluded from aggregates)

| u_th | RMSE x | RMSE xi | energy MAE |
|---:|---:|---:|---:|
{stress_rows}

Direct-from-anchor trajectory panels are provided for three held-out validation orbits and all seven named reference trajectories.

## Admissibility and energy

Hybrid dense predictions have `{summary['admissibility']['union_violation_count']}` union violations (`{summary['admissibility']['union_violation_fraction']:.3%}`). Energy error RMSE/MAE are `{energy['rmse']:.6g}`/`{energy['mae']:.6g}`; median, p90, p95, p99, p99.9, max absolute errors are `{energy['median_absolute']:.6g}`, `{energy['p90_absolute']:.6g}`, `{energy['p95_absolute']:.6g}`, `{energy['p99_absolute']:.6g}`, `{energy['p99p9_absolute']:.6g}`, `{energy['maximum_absolute']:.6g}`. Tail diagnostics are recorded in the summary.

## Matched s=0.2 comparison

| model | RMSE x | MAE x | p90 x | p99 x | RMSE xi | MAE xi | p90 xi | p99 xi |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
{matched_rows}

The heatmap ratio is `log10((|e_hybrid|+1e-12)/(|e_local|+1e-12))`: positive favors local, negative favors hybrid. Both models use identical anchors and exact endpoints.

## Scientific assessment

**A. Visual trajectory fidelity:** {summary['assessment']['trajectory']}

**B. Remaining error concentration:** {summary['assessment']['concentration']}

**C. Native-step comparison:** {summary['assessment']['matched']}

Tests: `{summary['tests']['stdout'].strip()}`. Protected hashes were identical before/after. No recursive study was run.
"""


def main() -> None:
    if OUTPUT.exists(): raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for directory in (OUTPUT,ARRAYS,TABLES,FIGURES,TESTS): directory.mkdir(parents=True,exist_ok=True)
    integrity=gate(); write_json(OUTPUT/"immutable_input_gate.json",integrity)
    if not integrity["passed"]: raise RuntimeError(integrity["failures"])
    before={str(path.resolve()):file_sha256(path) for path in EXPECTED}
    queries=load_npz(DENSE_QUERIES)
    if len(queries["s"])!=524288 or not np.array_equal(np.unique(queries["f"]),DENSE_FRACTIONS): raise RuntimeError("dense grid mismatch")
    hybrid_model=load_hybrid_model(HYBRID_CHECKPOINT); preprocessing=HybridPreprocessing.from_json(HYBRID_PREPROCESSING)
    prediction=predict_hybrid_diagnostics(hybrid_model,preprocessing,queries)
    if prediction["residual_state_error_max_difference"][0]>1e-13: raise RuntimeError("reconstruction inconsistency")
    dense,per_orbit=aggregate_error_metrics(prediction["x_error"],prediction["xi_error"],queries["orbit_id"])
    np.savez_compressed(ARRAYS/"dense_hybrid_predictions.npz",**prediction_table(queries,prediction))
    orbit_lookup={oid:i for i,oid in enumerate(np.unique(queries["orbit_id"]))}
    write_csv(TABLES/"per_orbit_metrics.csv",[{"orbit_id":r["orbit_id"],"row_count":r["row_count"],**flat(r)} for r in per_orbit])
    coordinates=coordinate_rows(queries,prediction)
    for name,rows in coordinates.items(): write_csv(TABLES/f"error_vs_{name}.csv",rows)
    masks={"hard_u_th_le_0p30":queries["u_th"]<=.30,"ordinary_u_th_gt_0p30":queries["u_th"]>.30,"short_s_le_5":queries["s"]<=5,"intermediate_5_lt_s_le_20":(queries["s"]>5)&(queries["s"]<=20),"long_s_gt_20":queries["s"]>20,"incoming_x0_-17_to_-8p5":(queries["x0"]>=-17)&(queries["x0"]<=-8.5)}
    for center in (.05,.15,.30,.90): masks[f"u_th_within_0p01_of_{center:.2f}"]=np.abs(queries["u_th"]-center)<=.01
    families={name:subset_metrics(prediction,mask) for name,mask in masks.items()}; write_csv(TABLES/"family_and_regime_metrics.csv",[{"subset":name,**flat(row)} for name,row in families.items()])
    adm,invalid=admissibility_summary(queries,prediction); write_csv(TABLES/"admissibility_violations.csv",[{"query_index":int(i),"orbit_id":str(queries["orbit_id"][i]),"u_th":queries["u_th"][i],"x0":queries["x0"][i],"s":queries["s"][i],"exact_x1":queries["exact_x1"][i],"exact_xi1":queries["exact_xi1"][i],"predicted_x1":prediction["predicted_x1"][i],"predicted_xi1":prediction["predicted_xi1"][i]} for i in np.flatnonzero(invalid)])
    energy=energy_metrics(prediction); order=np.argsort(np.abs(prediction["energy_error"]))[-100:]
    tail={"top100_high_E0_top5pct_fraction":float(np.mean(queries["E0"][order]>=np.quantile(queries["E0"],.95))),"top100_low_exact_C_bottom5pct_fraction":float(np.mean(prediction["exact_C1"][order]<=np.quantile(prediction["exact_C1"],.05))),"top100_near_throat_fraction":float(np.mean(np.abs(queries["exact_x1"][order])<=2)),"spearman":{"E0":float(spearmanr(np.abs(prediction["energy_error"]),queries["E0"]).statistic),"exact_C":float(spearmanr(np.abs(prediction["energy_error"]),prediction["exact_C1"]).statistic),"s":float(spearmanr(np.abs(prediction["energy_error"]),queries["s"]).statistic)}}
    write_json(TABLES/"energy_diagnostics.json",{"metrics":energy,"tail":tail})
    for name,rows in coordinates.items(): plot_lines(rows,name,FIGURES/f"error_vs_{name}.png",f"Hybrid finite-time error versus {name}")
    plot_lines(coordinates["s"],"physical elapsed time s",FIGURES/"error_percentiles_vs_elapsed_time.png","Hybrid error percentiles versus elapsed time",True)
    plot_dense_maps(queries,prediction,FIGURES/"dense_error_maps_x0_s.png"); plot_energy(queries,prediction,coordinates,FIGURES/"energy_consistency.png")

    matched_q=load_npz(MATCHED_QUERIES); local_model=load_trained_model(LOCAL_CHECKPOINT); local_norm=Normalization.from_stage1(LOCAL_NORMALIZATION,("x","xi","E0"),("delta_x","delta_xi"),"outer_microcore40k_train_x_xi_energy_input_only")
    local_pred=predict_local(local_model,local_norm,matched_q); hybrid_pred=predict_hybrid_diagnostics(hybrid_model,preprocessing,matched_q)
    np.savez_compressed(ARRAYS/"matched_s0p2_anchor_bank.npz",**matched_q); np.savez_compressed(ARRAYS/"matched_local_predictions.npz",**local_pred); np.savez_compressed(ARRAYS/"matched_hybrid_predictions.npz",**hybrid_pred)
    matched={"aggregate":{},"families":{},"coordinate_bins":{}}
    for name,pred in (("local",local_pred),("hybrid",hybrid_pred)):
        matched["aggregate"][name]={"x":scalar_error_metrics(pred["x_error"]),"xi":scalar_error_metrics(pred["xi_error"])}
        matched["families"][name]={family:subset_metrics(pred,mask) for family,mask in (("hard",matched_q["u_th"]<=.30),("ordinary",matched_q["u_th"]>.30))}
        matched["coordinate_bins"][name]={
            "x0":bin_error_rows(matched_q["x0"],pred,X_EDGES,"x0"),
            "u_th":bin_error_rows(matched_q["u_th"],pred,np.linspace(float(matched_q["u_th"].min()),float(matched_q["u_th"].max()),31),"u_th"),
        }
        for coordinate,rows in matched["coordinate_bins"][name].items():
            write_csv(TABLES/f"matched_{name}_error_vs_{coordinate}.csv",rows)
    np.savez_compressed(
        ARRAYS/"matched_error_ratios.npz",
        query_index=matched_q["query_index"], orbit_id=matched_q["orbit_id"],
        x0=matched_q["x0"], xi0=matched_q["xi0"], u_th=matched_q["u_th"], s=matched_q["s"],
        local_absolute_x_error=np.abs(local_pred["x_error"]), hybrid_absolute_x_error=np.abs(hybrid_pred["x_error"]),
        local_absolute_xi_error=np.abs(local_pred["xi_error"]), hybrid_absolute_xi_error=np.abs(hybrid_pred["xi_error"]),
        x_log10_hybrid_over_local=log_error_ratio(np.abs(hybrid_pred["x_error"]),np.abs(local_pred["x_error"]),EPSILON),
        xi_log10_hybrid_over_local=log_error_ratio(np.abs(hybrid_pred["xi_error"]),np.abs(local_pred["xi_error"]),EPSILON),
        ratio_epsilon=np.asarray([EPSILON]),
    )
    write_json(TABLES/"matched_metrics.json",matched)
    map_uth=matched_map(matched_q,local_pred,hybrid_pred,"u_th",FIGURES/"matched_maps_x0_u_th.png"); map_xi=matched_map(matched_q,local_pred,hybrid_pred,"xi0",FIGURES/"matched_maps_x0_xi0.png")
    write_json(TABLES/"matched_map_grids.json",{"x0_u_th":map_uth,"x0_xi0":map_xi})

    small_q=load_npz(SMALL_QUERIES); small_pred=predict_hybrid_diagnostics(hybrid_model,preprocessing,small_q); small={f"{s:.12g}":subset_metrics(small_pred,np.isclose(small_q["s"],s,rtol=0,atol=1e-14)) for s in SMALL_S_VALUES}; write_json(TABLES/"small_s_metrics.json",small)
    small_three_way=json.loads(THREE_WAY_COMPARISON.read_text(encoding="utf-8"))["small_s"]
    write_json(TABLES/"small_s_three_way.json",small_three_way); plot_small_s_three_way(small_three_way,FIGURES/"small_s_three_way.png")
    stress_q=load_npz(STRESS_QUERIES); stress_pred=predict_hybrid_diagnostics(hybrid_model,preprocessing,stress_q); reference=[]
    for i,u in enumerate(np.unique(stress_q["u_th"])):
        metric=subset_metrics(stress_pred,stress_q["orbit_index"]==i); reference.append({"u_th":float(u),"x_rmse":metric["x"]["rmse"],"xi_rmse":metric["xi"]["rmse"],"energy_mae":float(np.mean(np.abs(stress_pred["energy_error"][stress_q["orbit_index"]==i])))})
    write_csv(TABLES/"reference_trajectory_metrics.csv",reference)
    with np.load(VALIDATION_BANK,allow_pickle=False) as source: bank={name:source[name] for name in source.files}
    for label,target in (("low",.05),("middle",.5),("high",.9)):
        index=int(np.argmin(np.abs(bank["u_th"]-target))); trajectory_figure(bank,index,hybrid_model,preprocessing,FIGURES/f"representative_{label}.png",f"Exact and hybrid trajectories for held-out u_th={bank['u_th'][index]:.4f}")
    with np.load(STRESS_BANK,allow_pickle=False) as source: bank={name:source[name] for name in source.files}
    for i,u in enumerate(bank["u_th"]): trajectory_figure(bank,i,hybrid_model,preprocessing,FIGURES/f"reference_u_th_{u:.2f}.png",f"Exact and hybrid trajectories for reference u_th={u:.2f}")

    old=json.loads(PRIOR_SUMMARY.read_text())["dense_aggregate"]["303"]
    assessment={"trajectory":"Direct curves remain visually close across held-out and named reference trajectories; quantitative tails still grow on hard incoming and long-horizon queries.","concentration":f"The largest errors remain concentrated at long s, hard-family orbits, and incoming anchors; incoming x RMSE is {families['incoming_x0_-17_to_-8p5']['x']['rmse']:.4g}.","matched":f"Local/hybrid RMSE ratios are {matched['aggregate']['hybrid']['x']['rmse']/matched['aggregate']['local']['x']['rmse']:.2f}x for x and {matched['aggregate']['hybrid']['xi']['rmse']/matched['aggregate']['local']['xi']['rmse']:.2f}x for xi; ratio maps locate the exceptions cell by cell."}
    tests=run_tests(); after={str(path.resolve()):file_sha256(path) for path in EXPECTED}
    if before!=after: raise RuntimeError("protected artifact changed")
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"dense_query_count":524288,"grid_reused_exactly":True,"dense_aggregate":dense,"old_dense_aggregate":old,"coordinate_bins":coordinates,"families":families,"admissibility":adm,"energy":energy,"energy_tail":tail,"identity":{"x_max":float(np.max(np.abs(prediction['x_error'][queries['s']==0]))),"xi_max":float(np.max(np.abs(prediction['xi_error'][queries['s']==0])))},"small_s":small,"small_s_three_way":small_three_way,"reference_metrics":reference,"matched":matched,"ratio_epsilon":EPSILON,"local_model":{"seed":101,"step":.2,"checkpoint":str(LOCAL_CHECKPOINT.resolve()),"checkpoint_sha256":file_sha256(LOCAL_CHECKPOINT),"normalization_sha256":file_sha256(LOCAL_NORMALIZATION)},"assessment":assessment,"tests":tests,"sealed_test_opened":False,"training_or_reintegration":False}
    write_json(SUMMARY,summary); REPORT.write_text(report(summary),encoding="utf-8")
    artifacts={str(path.relative_to(OUTPUT)):{"path":str(path.resolve()),"sha256":file_sha256(path),"bytes":path.stat().st_size} for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path not in (MANIFEST,MANIFEST_HASH)}
    manifest={"experiment":"frozen_hybrid_dense_validation_and_matched_local_s0p2","status":"completed_without_training_reintegration_or_sealed_predictions","protected_before":before,"protected_after":after,"sealed_policy":{"NPZ_opened":False,"predictions":False,"byte_hash_only":True},"summary":{"path":str(SUMMARY.resolve()),"sha256":file_sha256(SUMMARY)},"report":{"path":str(REPORT.resolve()),"sha256":file_sha256(REPORT)},"artifacts":artifacts,"source_hashes":{"module":file_sha256(ROOT/"src/wormhole_sciml/finite_time_hybrid_validation.py"),"runner":file_sha256(Path(__file__)),"tests":file_sha256(ROOT/"tests/test_finite_time_hybrid_dense.py")}}
    write_json(MANIFEST,manifest); MANIFEST_HASH.write_text(f"{file_sha256(MANIFEST)}  {MANIFEST.name}\n"); print(f"wrote {REPORT}")


if __name__ == "__main__": main()
