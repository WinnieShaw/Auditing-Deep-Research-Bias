#!/usr/bin/env python3
"""CPU-only controlled robustness matrix and zero-support identification for v6."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_causal_search_scm import (
    categorical_from_uniform, fit_nuisance, generate_tasks, model_features,
    nuisance_predictions, policy_probabilities, split_ranges, stable_seed,
)

GRID = np.round(np.arange(0, 1.0001, .05), 2)


def softmax(x):
    x=np.asarray(x,float); x=x-x.max(); e=np.exp(np.clip(x,-700,0)); return e/e.sum()


def transform_propensity(p, kind, rng=None):
    p=np.asarray(p,float)
    if kind=="exact": return p
    if kind.startswith("temperature_"):
        t=float(kind.split("_")[1]); q=np.power(np.maximum(p,1e-300),1/t); return q/q.sum()
    if kind.startswith("logit_noise_"):
        s=float(kind.split("_")[2]); return softmax(np.log(np.maximum(p,1e-300))+rng.normal(0,s,len(p)))
    if kind.startswith("clip_"):
        return p
    if kind=="poor_overlap":
        q=np.power(np.maximum(p,1e-300),2.5); return q/q.sum()
    raise KeyError(kind)


def fit_visible_propensity(dev_tasks, scenario, cfg):
    rows=[]; targets=[]
    for task in dev_tasks:
        for z in (-1,1):
            p=policy_probabilities(task,0.,0.,z,scenario,float(cfg["query_feedback"]))
            f=np.column_stack([np.ones(len(task.y)),task.x,task.proxy,np.full(len(task.y),z)])
            rows.append(f); targets.append(np.log(np.maximum(p,1e-12)))
    return np.linalg.lstsq(np.vstack(rows),np.concatenate(targets),rcond=None)[0]


def visible_propensity(task,z,coef):
    f=np.column_stack([np.ones(len(task.y)),task.x,task.proxy,np.full(len(task.y),z)])
    return softmax(f@coef)


def fit_omitted(tasks):
    y=np.concatenate([t.y for t in tasks]); x=np.vstack([np.column_stack([np.ones(len(t.y)),np.full(len(t.y),t.mu),t.x[:,1],t.x[:,2]]) for t in tasks])
    return np.linalg.lstsq(x,y,rcond=None)[0]


def predict_omitted(task,coef):
    x=np.column_stack([np.ones(len(task.y)),np.full(len(task.y),task.mu),task.x[:,1],task.x[:,2]])
    return np.clip(x@coef,-1,1)


def task_estimate(task, obs, pred, propensity_kind, visible_coef, noise_seed, cap=None):
    n=len(task.y); anchor=float(np.mean(pred)); corr=0.; weights=[]
    rng=np.random.default_rng(noise_seed)
    for r in obs:
        if propensity_kind=="estimated_visible_linear": ph=visible_propensity(task,r["z"],visible_coef)
        else: ph=transform_propensity(r["p_full"],propensity_kind,rng)
        p=float(ph[r["selected"]]); w=1/(n*max(p,1e-300));
        if cap is not None: w=min(w,cap)
        weights.append(w); corr+=w*(r["y"]-pred[r["selected"]])
    raw=anchor+corr/len(obs); ess=sum(weights)**2/max(sum(w*w for w in weights),1e-12)
    return anchor,raw,ess,max(weights),min(float((visible_propensity(task,r["z"],visible_coef) if propensity_kind=="estimated_visible_linear" else transform_propensity(r["p_full"],propensity_kind,np.random.default_rng(noise_seed)))[r["selected"]]) for r in obs)


def simulate(task,z,scenario,cfg,seed):
    rounds=4; rng=np.random.default_rng(stable_seed(cfg["base_seed"],seed,task.task_id,"v6_robust_policy")); us=rng.random(rounds); qnoise=rng.normal(0,.08,rounds)
    query=0.; ys=[]; obs=[]
    for r in range(rounds):
        opened=float(np.mean(ys)) if ys else 0.; p=policy_probabilities(task,query,opened,z,scenario,float(cfg["query_feedback"])); j=categorical_from_uniform(p,float(us[r])); y=float(task.y[j])
        obs.append({"round":r+1,"selected":j,"y":y,"p_full":p,"z":z}); ys.append(y); query=.55*query+.35*y+qnoise[r]
    return obs


def choose_lambda(dev_rows):
    losses=[]
    for lam in GRID:
        e=np.clip(dev_rows.anchor+lam*(np.clip(dev_rows.raw,-1,1)-dev_rows.anchor),-1,1)
        losses.append(float(np.mean(abs(e-dev_rows.target))))
    i=int(np.argmin(losses)); return float(GRID[i]),losses[i]


def aggregate(rows):
    d=pd.DataFrame(rows); d["estimate"]=np.clip(d.anchor+d.lam*(np.clip(d.raw,-1,1)-d.anchor),-1,1); d["error"]=d.estimate-d.target
    pair=d.pivot_table(index=["seed","task_id"],columns="z",values="estimate",aggfunc="mean")
    return {"task_clusters":int(d.groupby(["seed","task_id"]).ngroups),"mae":float(d.groupby(["seed","task_id"]).error.apply(lambda x:abs(x).mean()).mean()),"bias":float(d.groupby(["seed","task_id"]).error.mean().mean()),"wrong_direction_rate":float((np.sign(d.estimate)!=np.sign(d.target)).groupby([d.seed,d.task_id]).mean().mean()),"ranking_sensitivity":float((pair[1]-pair[-1]).abs().mean()),"mean_ess":float(d.ess.mean()),"p95_max_weight":float(d.max_weight.quantile(.95)),"minimum_estimated_propensity":float(d.min_p.min())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True); (args.output_dir/"figures").mkdir()
    cfg=yaml.safe_load(args.config.read_text()); scenario={"id":"robust","kappa":4.,"rounds":4,"rho":.20,"stop_strength":0.,"force_full_horizon":True,"outcome_model":"correct"}
    propensity=["exact","temperature_0.5","temperature_1.5","temperature_2.0","logit_noise_0.1","logit_noise_0.25","logit_noise_0.5","logit_noise_1.0","estimated_visible_linear","poor_overlap"]
    omodels=["oracle","linear","constant","misspecified_omit_key","cross_domain_shift"]
    raw=[]; zero_rows=[]; pi_diag=[]
    for seed in cfg["seeds"]:
        ranges=split_ranges(cfg); tasks=generate_tasks(seed,max(x.stop for x in ranges.values()),int(cfg["candidate_documents"]),cfg); dev=[tasks[i] for i in ranges["development"]]; cal=[tasks[i] for i in ranges["calibration"]]; test=[tasks[i] for i in ranges["test"]]
        models={"oracle":{"kind":"correct"},"linear":fit_nuisance(dev,"misspecified"),"constant":fit_nuisance(dev,"constant")}; omit=fit_omitted(dev)
        shifted=generate_tasks(seed+100000,len(dev),int(cfg["candidate_documents"]),{**cfg,"base_seed":int(cfg["base_seed"])+777}); models["cross_domain_shift"]=fit_nuisance(shifted,"misspecified")
        vcoef=fit_visible_propensity(dev,scenario,cfg)
        for split,tasks0 in (("development",dev),("test",test)):
            for task in tasks0:
                predictions={k:nuisance_predictions(task,v)[0] for k,v in models.items()}; predictions["misspecified_omit_key"]=predict_omitted(task,omit)
                for z in (-1,1):
                    obs=simulate(task,z,scenario,cfg,seed)
                    for pk in propensity:
                        for ok,pred in predictions.items():
                            anchor,est,ess,mw,mp=task_estimate(task,obs,pred,pk,vcoef,stable_seed(seed,task.task_id,z,pk))
                            raw.append({"seed":seed,"split":split,"task_id":task.task_id,"z":z,"target":task.target,"propensity":pk,"outcome_model":ok,"anchor":anchor,"raw":est,"ess":ess,"max_weight":mw,"min_p":mp})
                        if split=="test" and pk=="exact":
                            for cap in (5.,10.,20.):
                                pred=predictions["linear"]; a,e,ess,mw,mp=task_estimate(task,obs,pred,pk,vcoef,stable_seed(seed,task.task_id,z,pk),cap)
                                raw.append({"seed":seed,"split":split,"task_id":task.task_id,"z":z,"target":task.target,"propensity":f"exact_weight_clip_{int(cap)}","outcome_model":"linear","anchor":a,"raw":e,"ess":ess,"max_weight":mw,"min_p":mp})
                    if split=="test":
                        # deterministic zero-support replay and sharp bounded-outcome interval
                        det={**scenario,"rho":0.,"deterministic":True}; dobs=simulate(task,z,det,cfg,seed); support=sorted({r["selected"] for r in dobs}); lo=(float(task.y[support].sum())-(len(task.y)-len(support)))/len(task.y); hi=(float(task.y[support].sum())+(len(task.y)-len(support)))/len(task.y)
                        zero_rows.append({"seed":seed,"task_id":task.task_id,"z":z,"target":task.target,"support_size":len(support),"inaccessible_fraction":1-len(support)/len(task.y),"lower_bound":lo,"upper_bound":hi,"true_target_covered":lo<=task.target<=hi,"sign_identifiable":(lo>0) or (hi<0),"interval_width":hi-lo})
                        if seed==cfg["seeds"][0]:
                            with gzip.open(args.output_dir/"epsilon0_candidate_audit.jsonl.gz","at",encoding="utf-8") as f:
                                for j,y in enumerate(task.y): f.write(json.dumps({"seed":seed,"task_id":task.task_id,"z":z,"candidate_id":j,"support":j in support,"candidate_outcome":float(y),"target_mean":task.target,"observed_support_outcome":float(y) if j in support else None})+"\n")
        print(json.dumps({"seed":seed,"rows":len(raw)}),flush=True)
    raw=pd.DataFrame(raw); raw.to_csv(args.output_dir/"controlled_robustness_task_rows.csv.gz",index=False,compression="gzip")
    tuning=[]; metrics=[]
    for (pk,ok),g in raw.groupby(["propensity","outcome_model"]):
        dev=g[g.split=="development"]
        if dev.empty: lam=.5; dev_mae=np.nan
        else: lam,dev_mae=choose_lambda(dev)
        te=g[g.split=="test"].copy(); te["lam"]=lam
        tuning.append({"propensity":pk,"outcome_model":ok,"lambda":lam,"development_mae":dev_mae})
        metrics.append({"propensity":pk,"outcome_model":ok,"lambda":lam,**aggregate(te)})
    m=pd.DataFrame(metrics); pd.DataFrame(tuning).to_csv(args.output_dir/"controlled_tuning.csv",index=False)
    m.to_csv(args.output_dir/"propensity_x_or_matrix.csv",index=False)
    m[m.outcome_model.eq("linear")].to_csv(args.output_dir/"propensity_perturbation.csv",index=False)
    m[m.propensity.eq("exact")].to_csv(args.output_dir/"outcome_model_robustness.csv",index=False)
    pd.DataFrame([{"status":"not_applicable_fixed_horizon","continuation_probability":1.0,"reason":"Controlled robustness fixes four rounds; real PERSPECTRUM lacks candidate probability vectors needed for counterfactual replay."}]).to_csv(args.output_dir/"continuation_perturbation.csv",index=False)
    z=pd.DataFrame(zero_rows); z.to_csv(args.output_dir/"zero_support_partial_identification.csv",index=False)
    pd.DataFrame([{"simulator_parameter_epsilon":e,"minimum_selection_probability":e/30 if e>0 else 0.,"mapping":"p=(1-epsilon)*softmax(score)+epsilon/N; epsilon=0 uses deterministic top-1"} for e in (0,.01,.02,.05,.10,.20)]).to_csv(args.output_dir/"epsilon_probability_mapping.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11,"pdf.fonttype":42})
    q=m[m.outcome_model.eq("linear") & ~m.propensity.str.contains("weight_clip")].sort_values("mae")
    fig,ax=plt.subplots(figsize=(7.2,4.4)); ax.scatter(q.minimum_estimated_propensity,q.mae,c="#7CB9A8",edgecolor="black");
    for _,r in q.iterrows(): ax.annotate(r.propensity.replace("temperature_","T=").replace("logit_noise_","noise="),(r.minimum_estimated_propensity,r.mae),xytext=(4,3),textcoords="offset points",fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Minimum estimated selection probability"); ax.set_ylabel("Task-macro MAE"); fig.tight_layout(); fig.savefig(args.output_dir/"figures/propensity_error_vs_mae.pdf",bbox_inches="tight"); fig.savefig(args.output_dir/"figures/propensity_error_vs_mae.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    piv=m[~m.propensity.str.contains("weight_clip")].pivot(index="outcome_model",columns="propensity",values="mae")
    fig,ax=plt.subplots(figsize=(11,4.8)); im=ax.imshow(piv,cmap="YlGnBu",aspect="auto"); ax.set_xticks(range(len(piv.columns)),piv.columns,rotation=42,ha="right"); ax.set_yticks(range(len(piv.index)),piv.index); [ax.text(j,i,f"{piv.iloc[i,j]:.3f}",ha="center",va="center",fontsize=7,color="black") for i in range(len(piv)) for j in range(len(piv.columns))]; fig.colorbar(im,ax=ax,label="MAE"); fig.tight_layout(); fig.savefig(args.output_dir/"figures/propensity_x_or_heatmap.pdf",bbox_inches="tight"); fig.savefig(args.output_dir/"figures/propensity_x_or_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    summary={"seeds":len(cfg["seeds"]),"test_task_clusters":int(z.groupby(["seed","task_id"]).ngroups),"zero_support":{"mean_interval_width":float(z.interval_width.mean()),"coverage":float(z.true_target_covered.mean()),"sign_identifiable_rate":float(z.sign_identifiable.mean()),"mean_inaccessible_fraction":float(z.inaccessible_fraction.mean())},"real_trajectory_limit":"PERSPECTRUM lacks full candidate probability vectors; propensity perturbations are controlled-simulator results only."}
    (args.output_dir/"controlled_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (args.output_dir/"controlled_manifest.json").write_text(json.dumps({"config":str(args.config),"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"cpu_only":True,"model_calls":0,"propensity_variants":propensity,"outcome_models":omodels},indent=2)+"\n")

if __name__=="__main__": main()
