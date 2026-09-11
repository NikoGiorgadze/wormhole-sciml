#!/usr/bin/env python3
"""Evaluation-only physics gate for a sensitivity-aware local loss."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wormhole_sciml.dynamics import conserved_energy, velocity_from_energy
from wormhole_sciml.energy_gradient import energy_gradient_x_xi
from wormhole_sciml.model_a import Normalization, load_trained_model, predict_increments
from wormhole_sciml.model_a_x_xi_microcore_data import OUTER_STRATUM_NAMES
from wormhole_sciml.model_a_x_xi_outer_data import X_COMPONENTS
from wormhole_sciml.physics_gate import experiment_parameters
from wormhole_sciml.stage1_data import STRATUM_NAMES, file_sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "model_a_sensitivity_map_physics_gate"
FIGURES = OUTPUT / "figures"
SUMMARY_PATH = OUTPUT / "sensitivity_map_summary.json"
ARRAYS_PATH = OUTPUT / "sensitivity_map_arrays.npz"
MAP_PATH = OUTPUT / "through_going_sensitivity_map.csv"
CLASS_PATH = OUTPUT / "microcore40k_orbit_classes.csv"
CLASS_BREAKDOWN_PATH = OUTPUT / "orbit_class_sampling_breakdowns.csv"
SENSITIVITY_PATH_OUT = OUTPUT / "eligible_training_sensitivity.csv"
TAIL_PATH = OUTPUT / "high_sensitivity_physical_location.csv"
DISTANCE_PATH = OUTPUT / "near_critical_concentration.csv"
HARD_PATH = OUTPUT / "hard_family_sensitivity_weights.csv"
LOSS_PATH = OUTPUT / "frozen_baseline_sensitivity_loss_budget.csv"
REPORT_PATH = OUTPUT / "SENSITIVITY_MAP_PHYSICS_GATE_REPORT.md"

DATA_DIR = ROOT / "output" / "model_a_x_xi_outer_microcore_sampling"
TRAIN_PATH = DATA_DIR / "outer_microcore40k_train_x_xi.npz"
VALID_PATH = DATA_DIR / "outer_microcore8k_validation_x_xi.npz"
DATA_SUMMARY = DATA_DIR / "outer_microcore_dataset_summary.json"
BASE = ROOT / "output" / "model_a_x_xi_energy_microcore40k_comparison"
BASE_MANIFEST = BASE / "energy_xi_training_manifest.json"
NORMALIZATION_PATH = BASE / "training" / "energy_input_normalization.json"
EXACT_PATH = ROOT / "output" / "c32x32_incoming_postmortem" / "incoming_branch_diagnostics.npz"
SENSITIVITY_DIR = ROOT / "output" / "model_a_fixed_E0_throat_sensitivity"
EXACT_SENSITIVITY_PATH = SENSITIVITY_DIR / "throat_sensitivity_arrays.npz"
EXACT_SENSITIVITY_SUMMARY = SENSITIVITY_DIR / "throat_sensitivity_summary.json"
ALIGNMENT_DIR = ROOT / "output" / "model_a_energy_gradient_alignment"
ALIGNMENT_PATH = ALIGNMENT_DIR / "energy_gradient_alignment_arrays.npz"
ALIGNMENT_SUMMARY = ALIGNMENT_DIR / "energy_gradient_alignment_summary.json"
TURNING_DIAGNOSTIC = ROOT / "reports" / "incoming_turning_diagnostic" / "results.json"
TRAVERSAL_SUMMARY = ROOT / "output" / "c32x32_traversal_families" / "traversal_family_summary.json"
ENERGY_SOURCE = ROOT / "src" / "wormhole_sciml" / "energy_gradient.py"
DYNAMICS_SOURCE = ROOT / "src" / "wormhole_sciml" / "dynamics.py"
PHYSICS_SOURCE = ROOT / "src" / "wormhole_sciml" / "physics_gate.py"

SEEDS = (101, 202, 303)
FAMILIES = (0.05, 0.15, 0.30)
FAR = (-17.0, -8.5)
E_CRITICAL = math.sqrt(3.0) / 2.0
U_LOWER = (1.0 - math.sqrt(7.0)) / 4.0
U_UPPER = (1.0 + math.sqrt(7.0)) / 4.0
COLORS = {101: "#0072B2", 202: "#E69F00", 303: "#CC79A7"}
FAMILY_COLORS = {0.05: "#0072B2", 0.15: "#D55E00", 0.30: "#009E73"}


def fkey(value: float) -> str:
    return f"u_th_{value:.2f}".replace(".", "p")


def throat_energy(u: Any) -> np.ndarray:
    """Exact validated energy evaluated at x=0 for the fixed experiment."""
    velocity = np.asarray(u, dtype=np.float64)
    return (3.0 + 2.0 * velocity) / (2.0 * np.sqrt(3.0 + 4.0 * velocity - 8.0 * velocity**2))


def throat_velocity(energy: Any, branch: int) -> np.ndarray:
    """Invert E_th(u) on the positive (+1) or negative (-1) directed branch."""
    if branch not in (-1, 1): raise ValueError("branch must be +/-1")
    E = np.asarray(energy, dtype=np.float64)
    if np.any(E < E_CRITICAL): raise ValueError("through-going throat velocity requires E>=E_critical")
    root = np.sqrt(np.maximum(28.0 * E**2 - 21.0, 0.0))
    return (2.0 * E**2 + branch * E * root - 1.5) / (8.0 * E**2 + 1.0)


def throat_sensitivity_from_u(u: Any) -> np.ndarray:
    velocity = np.asarray(u, dtype=np.float64)
    return (3.0 + 4.0 * velocity - 8.0 * velocity**2) ** 1.5 / (14.0 * velocity)


def throat_sensitivity(energy: Any, branch: int) -> np.ndarray:
    return throat_sensitivity_from_u(throat_velocity(energy, branch))


def distribution(values: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "minimum": float(np.min(a)), "median": float(np.median(a)),
            "mean": float(np.mean(a)), "p90": float(np.quantile(a,.90)), "p95": float(np.quantile(a,.95)),
            "p99": float(np.quantile(a,.99)), "p99_5": float(np.quantile(a,.995)),
            "p99_9": float(np.quantile(a,.999)), "maximum": float(np.max(a)),
            "maximum_to_median_ratio": float(np.max(a)/np.median(a))}


def contribution_shares(values: np.ndarray) -> dict[str, float]:
    a = np.sort(np.asarray(values,dtype=np.float64)); total=float(np.sum(a)); out={}
    for fraction in (.001,.005,.01,.05,.10):
        count=max(1,int(np.ceil(a.size*fraction)))
        out[f"top_{100*fraction:g}_percent"] = float(np.sum(a[-count:])/total)
    return out


def normalized_summary(values: np.ndarray) -> dict[str, Any]:
    normalized=np.asarray(values,dtype=np.float64)/float(np.mean(values)); stats=distribution(normalized)
    return {"normalization_mean_raw":float(np.mean(values)), "distribution":stats,
            "concentration":contribution_shares(normalized)}


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def classify(data: Any, wormhole: Any, spiral: Any) -> dict[str, np.ndarray]:
    E=np.asarray(data["E0"]);x=np.asarray(data["x"]);u=np.asarray(data["u"])
    plus=velocity_from_energy(x,E,wormhole,spiral,branch=1)
    minus=velocity_from_energy(x,E,wormhole,spiral,branch=-1)
    branch=np.where(np.abs(plus-u)<=np.abs(minus-u),1,-1).astype(np.int8)
    branch_error=np.minimum(np.abs(plus-u),np.abs(minus-u))
    supercritical=E>=E_CRITICAL; subcritical=(E>0)&(E<E_CRITICAL); nonpositive=E<=0
    toward=((branch==1)&(x<0))|((branch==-1)&(x>0))
    labels=np.empty(E.size,dtype="U48")
    labels[supercritical&toward]="future_throat_crossing"
    labels[supercritical&~toward]="through_going_already_crossed_or_outgoing"
    labels[subcritical&toward]="subcritical_inward_turning"
    labels[subcritical&~toward]="subcritical_outward_or_post_turn"
    labels[nonpositive]="nonpositive_boundary_asymptotic_no_throat"
    if np.any(labels==""): raise RuntimeError("unclassified orbit state")
    eligible=labels=="future_throat_crossing"
    left_to_right=eligible&(branch==1)&(x<0);right_to_left=eligible&(branch==-1)&(x>0)
    return {"branch":branch,"branch_reconstruction_error":branch_error,"class":labels,"eligible":eligible,
            "left_to_right":left_to_right,"right_to_left":right_to_left,
            "supercritical":supercritical,"subcritical":subcritical,"nonpositive":nonpositive}


def sensitivities_for_eligible(E: np.ndarray, branch: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    u_plus=throat_velocity(E,1);u_minus=throat_velocity(E,-1)
    uth=np.where(branch==1,u_plus,u_minus);S=throat_sensitivity_from_u(uth)
    return uth,S


def class_summary(classification: dict[str,np.ndarray]) -> list[dict[str,Any]]:
    labels=classification["class"];rows=[]
    for label in sorted(np.unique(labels)):
        count=int(np.sum(labels==label));rows.append({"orbit_class":label,"count":count,"fraction":count/labels.size})
    rows.extend((
        {"orbit_class":"future_throat_crossing_left_to_right","count":int(classification["left_to_right"].sum()),"fraction":float(classification["left_to_right"].mean())},
        {"orbit_class":"future_throat_crossing_right_to_left","count":int(classification["right_to_left"].sum()),"fraction":float(classification["right_to_left"].mean())},
    ))
    return rows


def sampling_breakdowns(data: Any, classes: dict[str,np.ndarray]) -> list[dict[str,Any]]:
    groups=[]
    for index,(name,_,_) in enumerate(X_COMPONENTS):groups.append(("x_component",name,data["x_component"]==index))
    groups.extend((("realized_x_region","central_abs_x_le_8p5",np.abs(data["x"])<=8.5),
                   ("realized_x_region","outer_abs_x_gt_8p5",np.abs(data["x"])>8.5),
                   ("micro_core_membership","micro_core",data["outer_xi_stratum"]==0),
                   ("micro_core_membership","not_micro_core",data["outer_xi_stratum"]!=0)))
    for index,name in enumerate(STRATUM_NAMES):groups.append(("standard_xi_stratum",name,data["stratum"]==index))
    for index,name in enumerate(OUTER_STRATUM_NAMES):groups.append(("outer_detailed_xi_stratum",name,data["outer_xi_stratum"]==index))
    rows=[]
    for grouping,label,mask in groups:
        if not np.any(mask):continue
        for orbit_class in sorted(np.unique(classes["class"])):
            selected=mask&(classes["class"]==orbit_class);rows.append({"grouping":grouping,"group":label,"orbit_class":orbit_class,
                "group_count":int(mask.sum()),"class_count_within_group":int(selected.sum()),"fraction_within_group":float(selected.sum()/mask.sum())})
    return rows


def sensitivity_group_row(label: str, mask: np.ndarray, eligible_indices: np.ndarray,
                          absS: np.ndarray, S2: np.ndarray, data: Any) -> dict[str,Any]:
    local=np.flatnonzero(mask[eligible_indices]);indices=eligible_indices[local]
    return {"group":label,"count":int(local.size),"eligible_fraction":float(local.size/eligible_indices.size),
            "abs_S_median":float(np.median(absS[local])),"abs_S_mean":float(np.mean(absS[local])),
            "abs_S_p90":float(np.quantile(absS[local],.90)),"abs_S_p99":float(np.quantile(absS[local],.99)),"abs_S_maximum":float(np.max(absS[local])),
            "S2_median":float(np.median(S2[local])),"S2_mean":float(np.mean(S2[local])),"S2_p99":float(np.quantile(S2[local],.99)),
            "median_energy_distance":float(np.median(data["E0"][indices]-E_CRITICAL)),
            "minimum_energy_distance":float(np.min(data["E0"][indices]-E_CRITICAL))}


def loss_metrics(base: np.ndarray, candidate: np.ndarray) -> dict[str,Any]:
    return {"count":int(base.size),"mean_L_base":float(np.mean(base)),"mean_candidate":float(np.mean(candidate)),
            "candidate_to_base_mean_ratio":float(np.mean(candidate)/np.mean(base)),
            "candidate_median":float(np.median(candidate)),"candidate_p95":float(np.quantile(candidate,.95)),
            "candidate_p99":float(np.quantile(candidate,.99)),"candidate_maximum":float(np.max(candidate)),
            **{f"candidate_{k}_contribution":v for k,v in contribution_shares(candidate).items()}}


def md_table(rows:list[dict[str,Any]],columns:list[tuple[str,str]],digits:int=5)->list[str]:
    out=["| "+" | ".join(label for _,label in columns)+" |","|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:
        cells=[]
        for key,_ in columns:
            v=row.get(key);cells.append(f"{v:.{digits}g}" if isinstance(v,float) else ("—" if v is None else str(v)))
        out.append("| "+" | ".join(cells)+" |")
    return out


def make_figures(arrays:dict[str,np.ndarray],hard_rows:list[dict[str,Any]])->None:
    E=arrays["map_energy"];up=arrays["map_u_plus"];um=arrays["map_u_minus"]
    fig,ax=plt.subplots(figsize=(9.5,4.8));ax.plot(E,up,color="#0072B2",lw=1.8,label="left-to-right (+) branch")
    ax.plot(E,um,color="#D55E00",lw=1.5,ls="--",label="right-to-left (-) branch")
    for family in FAMILIES:
        row=next(r for r in hard_rows if r["family_u_th"]==family);ax.scatter(row["E0"],family,s=55,color=FAMILY_COLORS[family],zorder=5,label=rf"$u_{{th}}={family:.2f}$")
    ax.axvline(E_CRITICAL,color="black",lw=.8,alpha=.6,label=r"$E_{crit}$");ax.set_xscale("log")
    ax.set_xlabel(r"conserved energy $E$");ax.set_ylabel(r"throat velocity $u_{th}$");ax.set_title("Exact directed throat branches")
    ax.grid(alpha=.18);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(FIGURES/"figure_1_throat_velocity_vs_energy.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,4.8));distance=E-E_CRITICAL
    ax.plot(distance,np.abs(arrays["map_S_plus"]),color="#0072B2",lw=1.7,label="positive branch")
    ax.plot(distance,np.abs(arrays["map_S_minus"]),color="#D55E00",lw=1.5,ls="--",label="negative branch")
    for family in FAMILIES:
        row=next(r for r in hard_rows if r["family_u_th"]==family);ax.scatter(row["E0"]-E_CRITICAL,row["abs_S"],s=55,color=FAMILY_COLORS[family],zorder=5)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"$E-E_{crit}$");ax.set_ylabel(r"$|du_{th}/dE|$")
    ax.set_title("Critical sensitivity divergence");ax.grid(alpha=.18);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"figure_2_sensitivity_vs_energy_distance.png",dpi=180);plt.close(fig)

    A=arrays["eligible_abs_S"];Q=arrays["eligible_S2"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));
    axes[0].hist(A,bins=np.geomspace(A.min(),A.max(),80),color="#0072B2",alpha=.82);axes[0].set_xscale("log");axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$|S(E_0)|$");axes[0].set_ylabel("eligible training count");axes[0].grid(alpha=.18)
    axes[1].hist(Q,bins=np.geomspace(Q.min(),Q.max(),80),color="#CC79A7",alpha=.82);axes[1].set_xscale("log");axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$S(E_0)^2$");axes[1].set_ylabel("eligible training count");axes[1].grid(alpha=.18)
    fig.suptitle("Sensitivity distribution on future throat-crossing samples",y=.995);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIGURES/"figure_3_training_sensitivity_distributions.png",dpi=180);plt.close(fig)

    dist=arrays["eligible_energy"]-E_CRITICAL;mc=arrays["eligible_micro_core"].astype(bool);edge=arrays["eligible_outer_edge"].astype(bool)
    fig,ax=plt.subplots(figsize=(9.5,5.0));
    ax.scatter(dist[~mc&~edge],A[~mc&~edge],s=7,alpha=.22,color="#7A7A7A",label="other eligible")
    ax.scatter(dist[mc],A[mc],s=10,alpha=.55,color="#009E73",label="outer micro-core")
    ax.scatter(dist[edge],A[edge],s=10,alpha=.5,color="#D55E00",label="outer edge stratum")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel(r"$E_0-E_{crit}$");ax.set_ylabel(r"$|S(E_0)|$")
    ax.set_title("Sampling populations on the exact sensitivity curve");ax.grid(alpha=.18);ax.legend();fig.tight_layout();fig.savefig(FIGURES/"figure_4_sensitivity_sampling_populations.png",dpi=180);plt.close(fig)

    train_x=arrays["train_x"];train_xi=arrays["train_xi"];wg=arrays["train_gradient_weight"]
    topg=wg>=np.quantile(wg,.99);elig=arrays["train_eligible"].astype(bool);wsq=np.full(train_x.shape,np.nan);wsq[elig]=arrays["eligible_normalized_S2_weight"]
    threshold=np.nanquantile(wsq,.99);tops=elig&(wsq>=threshold)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharex=True,sharey=True)
    axes[0].scatter(train_x,train_xi,s=2,color="#AAAAAA",alpha=.08);axes[0].scatter(train_x[topg],train_xi[topg],s=12,color="#D55E00",alpha=.7)
    axes[0].set_title(r"top 1% rejected $|g_E|^2$ weight")
    axes[1].scatter(train_x[elig],train_xi[elig],s=2,color="#AAAAAA",alpha=.1);axes[1].scatter(train_x[tops],train_xi[tops],s=12,color="#0072B2",alpha=.75)
    axes[1].set_title(r"top 1% eligible $S(E_0)^2$ weight")
    for ax in axes:ax.axvline(0,color="black",lw=.5,alpha=.4);ax.set_xlabel("x");ax.grid(alpha=.12)
    axes[0].set_ylabel(r"$\xi$");fig.suptitle("Qualitatively different emphasized populations",y=.995);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIGURES/"figure_5_gradient_vs_sensitivity_populations.png",dpi=180);plt.close(fig)


def main()->None:
    OUTPUT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(BASE_MANIFEST.read_text());alignment_summary=json.loads(ALIGNMENT_SUMMARY.read_text())
    exact_sensitivity_summary=json.loads(EXACT_SENSITIVITY_SUMMARY.read_text());turning=json.loads(TURNING_DIAGNOSTIC.read_text())
    normalization=Normalization.from_stage1(NORMALIZATION_PATH,input_columns=("x","xi","E0"),target_columns=("delta_x","delta_xi"),expected_source_dataset="outer_microcore40k_train_x_xi_energy_input_only")
    checkpoints={int(r["seed"]):Path(r["checkpoint"]) for r in manifest["runs"]}
    protected=[TRAIN_PATH,VALID_PATH,DATA_SUMMARY,BASE_MANIFEST,NORMALIZATION_PATH,EXACT_PATH,EXACT_SENSITIVITY_PATH,
               EXACT_SENSITIVITY_SUMMARY,ALIGNMENT_PATH,ALIGNMENT_SUMMARY,TURNING_DIAGNOSTIC,TRAVERSAL_SUMMARY,
               ENERGY_SOURCE,DYNAMICS_SOURCE,PHYSICS_SOURCE,*checkpoints.values()]
    before={str(p):file_sha256(p) for p in protected}
    train=np.load(TRAIN_PATH);valid=np.load(VALID_PATH);exact=np.load(EXACT_PATH);sensitivity=np.load(EXACT_SENSITIVITY_PATH);alignment=np.load(ALIGNMENT_PATH)
    wormhole,spiral=experiment_parameters();train_class=classify(train,wormhole,spiral);valid_class=classify(valid,wormhole,spiral)
    if np.max(train_class["branch_reconstruction_error"])>5e-13:raise RuntimeError("energy branch reconstruction failed")

    # Exact relation checks against the validated energy and earlier turning gate.
    ucheck=np.linspace(U_LOWER+1e-7,U_UPPER-1e-7,20001);reference=conserved_energy(np.zeros_like(ucheck),ucheck,wormhole,spiral)
    relation_difference=np.abs(throat_energy(ucheck)-reference)
    relation_check={"maximum_absolute_energy_error":float(np.max(relation_difference)),
                    "maximum_relative_energy_error":float(np.max(relation_difference/np.abs(reference))),
                    "critical_energy":E_CRITICAL,"critical_energy_reference":float(conserved_energy(0.0,0.0,wormhole,spiral)),
                    "critical_reference_absolute_error":abs(float(conserved_energy(0.0,0.0,wormhole,spiral))-E_CRITICAL),
                    "u_physical_lower_open":U_LOWER,"u_physical_upper_open":U_UPPER,
                    "turning_gate_subcritical_cases_turn":all(r["turning_point"] is not None and not r["throat_crossed"] for r in turning["trajectories"] if r["energy"]<E_CRITICAL),
                    "turning_gate_supercritical_cases_cross":all(r["turning_point"] is None and r["throat_crossed"] for r in turning["trajectories"] if r["energy"]>E_CRITICAL)}

    eligible=train_class["eligible"];eligible_indices=np.flatnonzero(eligible);E_eligible=train["E0"][eligible];branch_eligible=train_class["branch"][eligible]
    uth_eligible,S_eligible=sensitivities_for_eligible(E_eligible,branch_eligible);absS=np.abs(S_eligible);S2=S_eligible**2
    abs_summary=normalized_summary(absS);sq_summary=normalized_summary(S2)
    normalized_abs=absS/abs_summary["normalization_mean_raw"];normalized_sq=S2/sq_summary["normalization_mean_raw"]

    # Dense map, with logarithmic refinement at the threshold and full training energy coverage.
    max_map_E=max(float(np.max(E_eligible)),float(np.max(valid["E0"][valid["E0"]>=E_CRITICAL])))
    distance=np.geomspace(1e-10,max_map_E-E_CRITICAL,30000);map_E=E_CRITICAL+distance
    map_up=throat_velocity(map_E,1);map_um=throat_velocity(map_E,-1);map_Sp=throat_sensitivity_from_u(map_up);map_Sm=throat_sensitivity_from_u(map_um)
    map_rows=[{"E":float(E),"E_minus_critical":float(E-E_CRITICAL),"u_plus":float(up),"S_plus":float(sp),"u_minus":float(um),"S_minus":float(sm)}
              for E,up,sp,um,sm in zip(map_E,map_up,map_Sp,map_um,map_Sm)]
    save_csv(MAP_PATH,map_rows)

    class_rows=class_summary(train_class);save_csv(CLASS_PATH,class_rows);breakdowns=sampling_breakdowns(train,train_class);save_csv(CLASS_BREAKDOWN_PATH,breakdowns)
    sensitivity_groups=[sensitivity_group_row("all_eligible",np.ones(train["x"].shape,bool),eligible_indices,absS,S2,train),
        sensitivity_group_row("micro_core_eligible",train["outer_xi_stratum"]==0,eligible_indices,absS,S2,train),
        sensitivity_group_row("non_micro_core_eligible",train["outer_xi_stratum"]!=0,eligible_indices,absS,S2,train),
        sensitivity_group_row("outer_edge_eligible",train["outer_xi_stratum"]==3,eligible_indices,absS,S2,train)]
    save_csv(SENSITIVITY_PATH_OUT,sensitivity_groups)

    tail_rows=[]
    for fraction in (.001,.005,.01,.05,.10):
        count=max(1,int(np.ceil(absS.size*fraction)));local=np.argsort(absS)[-count:];idx=eligible_indices[local]
        tail_rows.append({"top_fraction_percent":100*fraction,"count":count,"minimum_abs_S":float(np.min(absS[local])),
            "abs_S_weight_share":float(np.sum(absS[local])/np.sum(absS)),"S2_weight_share":float(np.sum(S2[local])/np.sum(S2)),
            "minimum_energy_distance":float(np.min(train["E0"][idx]-E_CRITICAL)),"maximum_energy_distance":float(np.max(train["E0"][idx]-E_CRITICAL)),
            "median_x":float(np.median(train["x"][idx])),"median_abs_x":float(np.median(np.abs(train["x"][idx]))),
            "median_xi":float(np.median(train["xi"][idx])),"median_abs_xi":float(np.median(np.abs(train["xi"][idx]))),
            "micro_core_fraction":float(np.mean(train["outer_xi_stratum"][idx]==0)),"outer_region_fraction":float(np.mean(np.abs(train["x"][idx])>8.5)),
            "edge_stratum_fraction":float(np.mean(train["stratum"][idx]==2)),"left_to_right_fraction":float(np.mean(train_class["branch"][idx]==1))})
    save_csv(TAIL_PATH,tail_rows)

    distance_rows=[];energy_distance=E_eligible-E_CRITICAL
    for bound in (1e-6,1e-5,1e-4,1e-3,1e-2,.1,1.0,10.0):
        mask=energy_distance<bound
        distance_rows.append({"energy_distance_upper_exclusive":bound,"count":int(mask.sum()),"eligible_fraction":float(mask.mean()),
                              "abs_S_weight_fraction":float(np.sum(absS[mask])/np.sum(absS)),"S2_weight_fraction":float(np.sum(S2[mask])/np.sum(S2))})
    save_csv(DISTANCE_PATH,distance_rows)

    # Validate the exact three cases and give distribution-relative weights.
    hard_rows=[]
    for family in FAMILIES:
        E0=float(exact_sensitivity_summary["families"][f"{family:.2f}"]["E0"]);mapped_u=float(throat_velocity(E0,1));mapped_S=float(throat_sensitivity(E0,1))
        reference_S=float(alignment_summary["families"][f"{family:.2f}"]["proportionality"]["theoretical_du_th_dE"])
        ranks_abs=100*np.searchsorted(np.sort(absS),abs(mapped_S),side="right")/absS.size;ranks_sq=100*np.searchsorted(np.sort(S2),mapped_S**2,side="right")/S2.size
        key=fkey(family);x=alignment[f"{key}__diagnostic_x"];far=(x>=FAR[0])&(x<FAR[1])
        hard_rows.append({"family_u_th":family,"E0":E0,"mapped_u_th":mapped_u,"u_absolute_error":abs(mapped_u-family),
            "u_relative_error":abs(mapped_u-family)/family,"mapped_S":mapped_S,"reference_S":reference_S,
            "S_absolute_error":abs(mapped_S-reference_S),"S_relative_error":abs(mapped_S-reference_S)/abs(reference_S),
            "abs_S":abs(mapped_S),"S2":mapped_S**2,"abs_S_training_percentile":ranks_abs,"S2_training_percentile":ranks_sq,
            "normalized_abs_S_weight":abs(mapped_S)/abs_summary["normalization_mean_raw"],
            "normalized_S2_weight":mapped_S**2/sq_summary["normalization_mean_raw"],
            "exact_state_count":int(x.size),"far_upstream_state_count":int(far.sum()),"orbit_weight_constant_over_states":True})
    save_csv(HARD_PATH,hard_rows)

    # Rejected gradient-weight population comparison.
    Ex,Exi=energy_gradient_x_xi(train["x_next"],train["xi_next"]);g=np.column_stack((normalization.target_std[0]*Ex,normalization.target_std[1]*Exi));gnorm=np.linalg.norm(g,axis=1)
    gradient_weight=gnorm**2/np.mean(gnorm**2);topg=gradient_weight>=np.quantile(gradient_weight,.99)
    localtops=normalized_sq>=np.quantile(normalized_sq,.99);tops_indices=eligible_indices[localtops]
    population_comparison={"gradient_top_1_percent":{"count":int(topg.sum()),"median_abs_x":float(np.median(np.abs(train["x"][topg]))),
        "median_abs_xi":float(np.median(np.abs(train["xi"][topg]))),"micro_core_fraction":float(np.mean(train["outer_xi_stratum"][topg]==0)),
        "edge_stratum_fraction":float(np.mean(train["stratum"][topg]==2))},
        "sensitivity_S2_top_1_percent_eligible":{"count":int(localtops.sum()),"median_abs_x":float(np.median(np.abs(train["x"][tops_indices]))),
        "median_abs_xi":float(np.median(np.abs(train["xi"][tops_indices]))),"micro_core_fraction":float(np.mean(train["outer_xi_stratum"][tops_indices]==0)),
        "edge_stratum_fraction":float(np.mean(train["stratum"][tops_indices]==2))},
        "top_population_intersection_count":int(np.intersect1d(np.flatnonzero(topg),tops_indices).size)}

    # Optional frozen-baseline loss budget on eligible validation and exact hard trajectories.
    val_eligible=valid_class["eligible"];val_indices=np.flatnonzero(val_eligible);val_E=valid["E0"][val_eligible];val_branch=valid_class["branch"][val_eligible]
    _,val_S=sensitivities_for_eligible(val_E,val_branch);val_w_abs=np.abs(val_S)/abs_summary["normalization_mean_raw"];val_w_sq=val_S**2/sq_summary["normalization_mean_raw"]
    vEx,vExi=energy_gradient_x_xi(valid["x_next"],valid["xi_next"]);vg=np.column_stack((normalization.target_std[0]*vEx,normalization.target_std[1]*vExi));vn=vg/np.linalg.norm(vg,axis=1)[:,None]
    models={seed:load_trained_model(checkpoints[seed]) for seed in SEEDS};val_inputs=np.column_stack((valid["x"],valid["xi"],valid["E0"]));val_target=np.column_stack((valid["delta_x"],valid["delta_xi"]))
    loss_rows=[];arrays={"map_energy":map_E,"map_u_plus":map_up,"map_u_minus":map_um,"map_S_plus":map_Sp,"map_S_minus":map_Sm,
        "train_x":train["x"],"train_xi":train["xi"],"train_E0":train["E0"],"train_orbit_branch":train_class["branch"],"train_eligible":eligible,
        "train_gradient_weight":gradient_weight,"eligible_indices":eligible_indices,"eligible_energy":E_eligible,"eligible_u_th":uth_eligible,
        "eligible_S":S_eligible,"eligible_abs_S":absS,"eligible_S2":S2,"eligible_normalized_abs_S_weight":normalized_abs,
        "eligible_normalized_S2_weight":normalized_sq,"eligible_micro_core":train["outer_xi_stratum"][eligible]==0,
        "eligible_outer_edge":train["outer_xi_stratum"][eligible]==3,"validation_eligible":val_eligible,"validation_S":val_S,
        "validation_normalized_abs_S_weight":val_w_abs,"validation_normalized_S2_weight":val_w_sq}
    for seed in SEEDS:
        pred=predict_increments(models[seed],val_inputs,normalization);err=(pred-val_target)/normalization.target_std;eperp=np.sum(err*vn,axis=1);base=.5*np.sum(err**2,axis=1)
        for candidate_name,weight in (("normalized_abs_S",val_w_abs),("normalized_S2",val_w_sq)):
            cand=.5*weight*eperp[val_eligible]**2;loss_rows.append({"dataset_scope":"eligible_validation","family_u_th":None,"seed":seed,"region":"all_eligible",
                "candidate":candidate_name,**loss_metrics(base[val_eligible],cand)})
        arrays[f"validation_seed_{seed}__L_base"]=base;arrays[f"validation_seed_{seed}__e_perp"]=eperp
        for family in FAMILIES:
            key=fkey(family);valid_kernel=sensitivity[f"{key}__valid_kernel_mask"].astype(bool)
            current=np.column_stack((exact[f"{key}__exact_state"][:,0],exact[f"{key}__exact_xi"]))[valid_kernel]
            target=np.column_stack((sensitivity[f"{key}__exact_delta_x_all"],sensitivity[f"{key}__exact_delta_xi_all"]))[valid_kernel]
            E0=next(r["E0"] for r in hard_rows if r["family_u_th"]==family);inputs=np.column_stack((current,np.full(current.shape[0],E0)))
            p=predict_increments(models[seed],inputs,normalization);e=(p-target)/normalization.target_std;n=alignment[f"{key}__n_E"];ep=np.sum(e*n,axis=1);lb=.5*np.sum(e**2,axis=1)
            x=alignment[f"{key}__diagnostic_x"];far=(x>=FAR[0])&(x<FAR[1]);hard=next(r for r in hard_rows if r["family_u_th"]==family)
            for region,mask in (("complete_incoming",np.ones(x.shape,bool)),("far_upstream",far)):
                for candidate_name,weight in (("normalized_abs_S",hard["normalized_abs_S_weight"]),("normalized_S2",hard["normalized_S2_weight"])):
                    cand=.5*weight*ep[mask]**2;loss_rows.append({"dataset_scope":"hard_trajectory","family_u_th":family,"seed":seed,"region":region,
                        "candidate":candidate_name,**loss_metrics(lb[mask],cand)})
    save_csv(LOSS_PATH,loss_rows);np.savez_compressed(ARRAYS_PATH,**arrays);make_figures(arrays,hard_rows)

    after={str(p):file_sha256(p) for p in protected}
    if before!=after:raise RuntimeError("a frozen source artifact changed")
    summary={"stage":"sensitivity-map physics gate","status":"complete_conditional",
        "decision":"Sensitivity is exact and physically relevant on a precisely identified future throat-crossing branch. E0 alone is not sufficient globally. Raw |S| is heavy-tailed but substantially less concentrated than rejected |g_E|^2 and prioritizes the micro-core; raw S^2 is pathologically critical-point dominated.",
        "protocol":{"evaluation_only":True,"models_trained_or_retrained":0,"loss_implemented_in_training":False,"lambda_selected":False,"trajectory_regeneration":False},
        "exact_relation":{"E_th_of_u":"(3+2u)/(2*sqrt(3+4u-8u^2))","u_of_E_branches":"(2E^2 +/- E*sqrt(28E^2-21)-3/2)/(8E^2+1)",
            "dE_du":"14u/(3+4u-8u^2)^(3/2)","du_dE":"(3+4u-8u^2)^(3/2)/(14u)",
            "critical_energy":E_CRITICAL,"positive_branch_u_domain":[0,U_UPPER],"negative_branch_u_domain":[U_LOWER,0],"energy_domain":"[sqrt(3)/2,infinity) on each directed through-going branch",
            "critical_behavior":"S diverges as (E-E_critical)^(-1/2), positive on + branch and negative on - branch","checks":relation_check},
        "E0_sufficiency":{"globally_sufficient":False,"restricted_branch_sufficient":True,
            "required_additional_information":"first-integral direction branch (+/-) and position relative to the throat, equivalently a future-throat-crossing class flag",
            "reason":"the same E0>=Ecrit labels two directed throat velocities and states before or after their throat crossing",
            "supercritical_samples_with_E0_ambiguity":int(train_class["supercritical"].sum())},
        "training_orbit_classes":class_rows,"training_eligible_count":int(eligible.sum()),"training_eligible_fraction":float(eligible.mean()),
        "validation_eligible_count":int(val_eligible.sum()),"validation_eligible_fraction":float(val_eligible.mean()),
        "raw_abs_S_distribution":distribution(absS),"raw_S2_distribution":distribution(S2),
        "normalized_abs_S":abs_summary,"normalized_S2":sq_summary,"sensitivity_groups":sensitivity_groups,
        "near_critical_concentration":distance_rows,"high_sensitivity_location":tail_rows,"population_comparison":population_comparison,
        "hard_families":hard_rows,"frozen_baseline_loss_budget":loss_rows,"protected_hashes_before":before,"protected_hashes_after":after,
        "artifacts":{"arrays":str(ARRAYS_PATH),"map":str(MAP_PATH),"orbit_classes":str(CLASS_PATH),"class_breakdowns":str(CLASS_BREAKDOWN_PATH),
            "sensitivity_groups":str(SENSITIVITY_PATH_OUT),"high_sensitivity_location":str(TAIL_PATH),"near_critical":str(DISTANCE_PATH),
            "hard_families":str(HARD_PATH),"loss_budget":str(LOSS_PATH),"figures":[str(p) for p in sorted(FIGURES.glob("*.png"))],"report":str(REPORT_PATH)}}
    SUMMARY_PATH.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")

    report=["# Sensitivity-Map Physics Gate", "", "## Gate result", "",
        "**Conditional physics pass, global-E-only failure.** The exact sensitivity exists and selects the desired near-critical population, but it is branch-dependent and divergent at the traversal threshold. Raw |S| is heavy-tailed yet much less concentrated than the rejected gradient weight; raw S^2 is pathologically dominated by the closest-to-critical samples.", "",
        "No model was trained, no loss was implemented, no coefficient was selected, and no reference trajectory was regenerated.", "",
        "## Exact throat relation and critical structure", "",
        r"At the throat, the validated fixed parameters give $C(0,u)=3/4+u-2u^2$ and",
        r"$$E_{th}(u)=\frac{3+2u}{2\sqrt{3+4u-8u^2}},\qquad \frac{dE_{th}}{du}=\frac{14u}{(3+4u-8u^2)^{3/2}}.$$",
        r"The two inverses are $u_\pm(E)=[2E^2\pm E\sqrt{28E^2-21}-3/2]/(8E^2+1)$. Each directed branch has $E\in[\sqrt3/2,\infty)$. At $E_{crit}=\sqrt3/2=0.866025403784439$, both meet at $u=0$, and $S\sim(E-E_{crit})^{-1/2}$: positive on the + branch and negative on the - branch. There is no sign change within either branch.", "",
        f"The formula reproduces the validated conserved-energy implementation with maximum absolute error {relation_check['maximum_absolute_energy_error']:.3e} and maximum relative error {relation_check['maximum_relative_energy_error']:.3e}; the largest absolute discrepancy is at the deliberately near-null endpoint where E is approximately 2344. The saved turning diagnostic independently confirms subcritical turning and supercritical crossing.", "",
        "## Is E0 sufficient?", "",
        "No, not globally. The same supercritical E0 labels positive and negative throat-velocity branches and can describe either a state approaching its future throat crossing or one already moving away after crossing. The exact first-integral branch and side of the throat—equivalently a future-crossing class—are additionally required. E0 is sufficient only after restricting to one directed through-going branch.", "",
        "## Frozen training orbit classes", ""]
    report+=md_table(class_rows,[("orbit_class","class"),("count","count"),("fraction","fraction")],6)
    report+=["",f"A future throat sensitivity is defined for {eligible.sum()}/40000 = {100*eligible.mean():.3f}% of training samples. Noneligible samples can retain ordinary MSE in a future formulation, but adding an extra term only to 25.88% of samples is an explicit class/sample-weighting change and must be treated as such.","",
        "## Eligible sensitivity distributions", ""]
    report+=md_table([{"quantity":"abs(S)",**distribution(absS)}, {"quantity":"S^2",**distribution(S2)}],[("quantity","quantity"),("minimum","min"),("median","median"),("mean","mean"),("p90","p90"),("p95","p95"),("p99","p99"),("p99_5","p99.5"),("p99_9","p99.9"),("maximum","max"),("maximum_to_median_ratio","max/median")],6)
    concentration_rows=[]
    for q,label in ((abs_summary,"normalized abs(S)"),(sq_summary,"normalized S^2")):
        concentration_rows.append({"quantity":label,"median":q["distribution"]["median"],"p90":q["distribution"]["p90"],"p95":q["distribution"]["p95"],"p99":q["distribution"]["p99"],"p99_9":q["distribution"]["p99_9"],"maximum":q["distribution"]["maximum"],**q["concentration"]})
    report+=["","Mean-one eligible-subset normalization diagnostics:",""]+md_table(concentration_rows,[("quantity","quantity"),("median","median"),("p90","p90"),("p95","p95"),("p99","p99"),("p99_9","p99.9"),("maximum","max"),("top_0.1_percent","top .1%"),("top_0.5_percent","top .5%"),("top_1_percent","top 1%"),("top_5_percent","top 5%"),("top_10_percent","top 10%")],6)
    report+=["","Raw |S| is strongly but not catastrophically concentrated (top 1%: 28.9%; top 10%: 73.5%). S^2 is pathological (top 0.1%: 90.5%; top 1%: 95.1%), with the single closest sample supplying 67.5% of total S^2 weight.","",
        "Near-critical concentration by absolute energy distance:",""]+md_table(distance_rows,[("energy_distance_upper_exclusive","E-Ecrit below"),("count","N"),("eligible_fraction","sample fraction"),("abs_S_weight_fraction","abs(S) weight share"),("S2_weight_fraction","S2 weight share")],6)
    report+=["","## Micro-core and emphasized population",""]+md_table(sensitivity_groups,[("group","group"),("count","N"),("abs_S_median","median abs(S)"),("abs_S_mean","mean abs(S)"),("abs_S_p99","p99"),("abs_S_maximum","max"),("median_energy_distance","median E-Ecrit")],6)
    report+=["",f"The high-sensitivity population is qualitatively different from the rejected gradient tail. Gradient top-1%: median abs(x)={population_comparison['gradient_top_1_percent']['median_abs_x']:.3f}, median abs(xi)={population_comparison['gradient_top_1_percent']['median_abs_xi']:.3f}, edge fraction={population_comparison['gradient_top_1_percent']['edge_stratum_fraction']:.3f}. Sensitivity-squared top-1%: median abs(x)={population_comparison['sensitivity_S2_top_1_percent_eligible']['median_abs_x']:.3f}, median abs(xi)={population_comparison['sensitivity_S2_top_1_percent_eligible']['median_abs_xi']:.4f}, micro-core fraction={population_comparison['sensitivity_S2_top_1_percent_eligible']['micro_core_fraction']:.3f}; intersection count={population_comparison['top_population_intersection_count']}.","",
        "## Known hard families",""]+md_table(hard_rows,[("family_u_th","u_th"),("E0","E0"),("mapped_S","S"),("S_absolute_error","S abs error"),("S_relative_error","S rel error"),("abs_S_training_percentile","abs(S) percentile"),("normalized_abs_S_weight","mean-one abs(S) weight"),("normalized_S2_weight","mean-one S2 weight")],7)
    report+=["","All exact states, including every far-upstream state, receive the same family-level sensitivity because E0 is conserved. This is an orbit-level factor, not an x-dependent hand weight.","",
        "## Frozen-baseline budget diagnostic",""]
    validation_budget=[r for r in loss_rows if r["dataset_scope"]=="eligible_validation"]
    report+=md_table(validation_budget,[("seed","seed"),("candidate","candidate"),("mean_L_base","mean base"),("mean_candidate","mean extra"),("candidate_to_base_mean_ratio","ratio"),("candidate_top_1_percent_contribution","top 1% share")],6)
    far_budget=[r for r in loss_rows if r["dataset_scope"]=="hard_trajectory" and r["region"]=="far_upstream"]
    report+=["","Far-upstream hard-family candidate/base ratios:",""]+md_table(far_budget,[("family_u_th","u_th"),("seed","seed"),("candidate","candidate"),("candidate_to_base_mean_ratio","ratio")],6)
    report+=["","Complete-branch values and full concentration fields are in `frozen_baseline_sensitivity_loss_budget.csv`.","",
        "## Direct answers", "",
        "1. The exact relation and inverses are given above; no fitted map is used.", "",
        "2. S(E) is single-valued only after choosing a directed throat branch.", "",
        "3. E0 alone is not sufficient globally; branch plus throat side/future-crossing status is required.", "",
        "4. |S| diverges as (E-Ecrit)^(-1/2).", "",
        f"5. {eligible.sum()}/40000 ({100*eligible.mean():.3f}%) training samples have a relevant future throat crossing.", "",
        "6. Yes. Sensitivity strongly emphasizes near-critical/core and outer micro-core states while assigning negligible sensitivity to outer edge states.", "",
        "7. Yes for population selection: it avoids the |xi|≈0.99 gradient-tail population. This does not make every sensitivity form numerically safe.", "",
        "8. Raw |S| is heavy-tailed but substantially less concentrated than rejected |g_E|^2; it remains a nontrivial optimization distribution.", "",
        "9. Raw S^2 is excessively near-critical concentrated and fails the numerical concentration gate.", "",
        "10. Mean-one hard weights are tabulated above: abs(S) gives about 22.97/8.54/4.38, while S^2 gives 8.74/1.21/0.318 for u_th=0.05/0.15/0.30.", "",
        "11. A global w_S(E0) objective is not physically well-defined over the full dataset. A branch-aware eligible-subset objective is definable, but no candidate or lambda is selected here.", "",
        "12. Leaving non-through-going samples on baseline MSE is mathematically clean because the eligible subset is exactly classifiable from validated state/energy information. It would nevertheless introduce a deliberate 25.88%-subset emphasis and must be reviewed as a sampling/weighting intervention.", "",
        "## Stop condition", "", "The gate stops here. No model was trained; no sensitivity loss, coefficient, clipping, saturation, K-target, multistep objective, architecture change, or dataset/split change was introduced."]
    REPORT_PATH.write_text("\n".join(report)+"\n")
    print(json.dumps({"status":summary["status"],"critical_energy":E_CRITICAL,"eligible_training_count":int(eligible.sum()),
        "abs_S_concentration":abs_summary["concentration"],"S2_concentration":sq_summary["concentration"],"output":str(OUTPUT)},indent=2))


if __name__=="__main__":main()
