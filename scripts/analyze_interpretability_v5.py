#!/usr/bin/env python3
"""CPU-only reliability and frozen-history analysis for true-Agent MS2 logs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from causal_search_credit.config import load_config
from causal_search_credit.util import load_jsonl
from scripts.run_ms2_sequential_confirmatory import fit_prospective_models, prospective_weight
from scripts.analyze_interpretability_v1_true_agent_ms2 import CONDITIONS, EFFECTS, decompose, dominant, ternary

SEED = 2026092305
N_BOOT = 10_000
LOCAL_MC = 400


def stable_seed(*parts: str) -> int:
    return SEED + int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:4], "big")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def boot_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def boot_corr(x: np.ndarray, y: np.ndarray, seed: int, spearman: bool) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float); rng = np.random.default_rng(seed)
    def corr(a, b):
        if spearman:
            a = pd.Series(a).rank().to_numpy(); b = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else np.nan
    point = corr(x, y); values = []
    for idx in rng.integers(0, len(x), size=(N_BOOT, len(x))):
        v = corr(x[idx], y[idx])
        if np.isfinite(v): values.append(v)
    if not values:
        # A constant diagnostic (e.g., candidate-set overlap fixed by design)
        # has no defined correlation; retain it as an explicit NA result.
        return point, np.nan, np.nan
    return point, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def load_cells_and_traces(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells, traces = [], []
    for model in ("qwen2_5_32b", "olmo3_7b"):
        cells.extend(load_jsonl(raw / model / "cells.jsonl"))
        traces.extend({**row, "model_key": model} for row in load_jsonl(raw / model / "traces.jsonl"))
    return pd.DataFrame(cells), pd.DataFrame(traces)


def replicate_effects(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, rep), g in cells.groupby(["model_key", "review_id", "replicate"], sort=True):
        theta = {}
        cess = {}
        for key, (controller, stopping) in CONDITIONS.items():
            z = g[(g.controller == controller) & (g.stopping_regime == stopping)]
            if len(z) != 1: raise ValueError(f"missing 2x2 cell: {model}/{task}/{rep}/{key}")
            theta[key] = float(z.naive_opened_mean.iloc[0]); cess[key] = float(z.preoutcome_cess.iloc[0])
        observed, predicted = decompose(theta), decompose(cess)
        row = {"model": model, "review_id": task, "replicate": int(rep), "observed_dominant": dominant(observed), "cess_dominant": dominant(predicted)}
        for e in EFFECTS:
            row[f"observed_{e}"] = observed[e]; row[f"cess_{e}"] = predicted[e]; row[f"cess_error_{e}"] = predicted[e] - observed[e]
        rows.append(row)
    return pd.DataFrame(rows)


def episode_ess(traces: pd.DataFrame) -> pd.DataFrame:
    """Realized inverse-probability ESS for the adaptive agent episodes.

    This is a trajectory diagnostic only.  It is not used to construct the
    frozen-history estimator or to assert a valid per-task interval.
    """
    h = traces[(traces.controller == "agent") & (traces.stopping_regime == "adaptive") & traces.document_id.notna()].copy()
    rows = []
    for (model, task, rep), g in h.groupby(["model_key", "review_id", "replicate"], sort=True):
        w = 1.0 / (g.selection_probability.to_numpy(float) * g.survival_probability.to_numpy(float))
        rows.append({"model": model, "review_id": task, "replicate": int(rep),
                     "ess_11": float(w.sum() ** 2 / np.square(w).sum()),
                     "max_inverse_weight_11": float(w.max()), "observed_rounds_11": int(len(g))})
    return pd.DataFrame(rows)


def icc_oneway(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """ICC(1,1), ICC(1,k), within variance and de-noised between variance."""
    n, k = matrix.shape; means = matrix.mean(1); grand = means.mean()
    msb = k * np.square(means - grand).sum() / (n - 1)
    msw = np.square(matrix - means[:, None]).sum() / (n * (k - 1))
    icc11 = (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw else np.nan
    icc1k = (msb - msw) / msb if msb else np.nan
    return float(icc11), float(icc1k), float(msw), float(max((msb - msw) / k, 0.0))


def reliability_analysis(rep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, loro_rows, cess_rows = [], [], []
    for model, g in rep.groupby("model", sort=True):
        tasks = sorted(g.review_id.unique())
        for e in EFFECTS:
            wide = g.pivot(index="review_id", columns="replicate", values=f"observed_{e}").loc[tasks]
            values = wide.to_numpy(float)
            pair_rs = []
            for a, b in ((0, 1), (0, 2), (1, 2)):
                x, y = values[:, a], values[:, b]
                pearson, pl, ph = boot_corr(x, y, stable_seed(model, e, "pearson", str(a), str(b)), False)
                spearman, sl, sh = boot_corr(x, y, stable_seed(model, e, "spearman", str(a), str(b)), True)
                sign = np.asarray([ternary(v) == ternary(w) for v, w in zip(x, y)], float)
                sm, sil, sih = boot_mean(sign, stable_seed(model, e, "sign", str(a), str(b)))
                rows += [{"model": model, "effect": e, "analysis": "pairwise", "pair": f"{a}-{b}", "metric": "pearson", "value": pearson, "ci95_low": pl, "ci95_high": ph, "source_tasks": len(tasks)},
                         {"model": model, "effect": e, "analysis": "pairwise", "pair": f"{a}-{b}", "metric": "spearman", "value": spearman, "ci95_low": sl, "ci95_high": sh, "source_tasks": len(tasks)},
                         {"model": model, "effect": e, "analysis": "pairwise", "pair": f"{a}-{b}", "metric": "sign_agreement", "value": sm, "ci95_low": sil, "ci95_high": sih, "source_tasks": len(tasks)}]
                pair_rs.append(pearson)
            icc11, icc13, within, between = icc_oneway(values)
            rbar = float(np.nanmean(pair_rs)); sb3 = 3 * rbar / (1 + 2 * rbar) if 1 + 2 * rbar else np.nan
            rows += [{"model": model, "effect": e, "analysis": "variance", "pair": "all", "metric": "within_task_variance", "value": within, "ci95_low": np.nan, "ci95_high": np.nan, "source_tasks": len(tasks)},
                     {"model": model, "effect": e, "analysis": "variance", "pair": "all", "metric": "between_task_variance", "value": between, "ci95_low": np.nan, "ci95_high": np.nan, "source_tasks": len(tasks)},
                     {"model": model, "effect": e, "analysis": "reliability", "pair": "all", "metric": "ICC_1_1", "value": icc11, "ci95_low": np.nan, "ci95_high": np.nan, "source_tasks": len(tasks)},
                     {"model": model, "effect": e, "analysis": "reliability", "pair": "all", "metric": "ICC_1_3_gold_mean", "value": icc13, "ci95_low": np.nan, "ci95_high": np.nan, "source_tasks": len(tasks)},
                     {"model": model, "effect": e, "analysis": "reliability", "pair": "all", "metric": "Spearman_Brown_3", "value": sb3, "ci95_low": np.nan, "ci95_high": np.nan, "source_tasks": len(tasks)}]
            # Each held replicate receives a prediction from the other two; aggregate per task before bootstrap.
            held = []
            for task in tasks:
                z = wide.loc[task].to_numpy(float)
                for hold in range(3): held.append({"review_id": task, "held": hold, "observed": z[hold], "predicted": float(np.delete(z, hold).mean())})
            h = pd.DataFrame(held); h["abs_error"] = (h.predicted - h.observed).abs(); h["direction_ok"] = [ternary(a) == ternary(b) for a,b in zip(h.predicted,h.observed)]
            per_task = h.groupby("review_id").agg(abs_error=("abs_error", "mean"), direction_ok=("direction_ok", "mean")).loc[tasks]
            mae, lo, hi = boot_mean(per_task.abs_error, stable_seed(model,e,"loro_mae"))
            # Do not average the held-out predictions before correlation: with
            # three replicates, that average is algebraically the same as the
            # average observed effect.  Bootstrap source tasks while retaining
            # all three held-out rows within each sampled task.
            pred_matrix = h.pivot(index="review_id", columns="held", values="predicted").loc[tasks].to_numpy(float)
            obs_matrix = h.pivot(index="review_id", columns="held", values="observed").loc[tasks].to_numpy(float)
            def spcorr(a,b):
                return pd.Series(a.ravel()).corr(pd.Series(b.ravel()), method="spearman")
            sp=float(spcorr(pred_matrix,obs_matrix)); rng=np.random.default_rng(stable_seed(model,e,"loro_sp")); draws=[]
            for idx in rng.integers(0,len(tasks),size=(N_BOOT,len(tasks))):
                v=spcorr(pred_matrix[idx],obs_matrix[idx])
                if np.isfinite(v): draws.append(v)
            sl,sh=(float(np.quantile(draws,.025)),float(np.quantile(draws,.975))) if draws else (np.nan,np.nan)
            da, dl, dh = boot_mean(per_task.direction_ok, stable_seed(model,e,"loro_dir"))
            loro_rows += [{"model":model,"effect":e,"metric":"MAE","value":mae,"ci95_low":lo,"ci95_high":hi,"source_tasks":len(tasks)},
                          {"model":model,"effect":e,"metric":"Spearman","value":sp,"ci95_low":sl,"ci95_high":sh,"source_tasks":len(tasks)},
                          {"model":model,"effect":e,"metric":"direction_accuracy","value":da,"ci95_low":dl,"ci95_high":dh,"source_tasks":len(tasks)}]
            cess_task = g.pivot(index="review_id", columns="replicate", values=f"cess_{e}").loc[tasks].mean(1)
            gold = wide.mean(1); error = cess_task - gold
            mae, lo, hi = boot_mean(np.abs(error), stable_seed(model,e,"cess_mae")); sp, sl, sh = boot_corr(cess_task, gold, stable_seed(model,e,"cess_sp"), True)
            cess_rows += [{"model":model,"effect":e,"metric":"CESS_vs_3rep_gold_MAE","value":mae,"ci95_low":lo,"ci95_high":hi,"source_tasks":len(tasks)},
                          {"model":model,"effect":e,"metric":"CESS_vs_3rep_gold_Spearman","value":sp,"ci95_low":sl,"ci95_high":sh,"source_tasks":len(tasks)},
                          {"model":model,"effect":e,"metric":"correlation_noise_ceiling_sqrt_ICC_1_3","value":math.sqrt(max(icc13,0)) if np.isfinite(icc13) else np.nan,"ci95_low":np.nan,"ci95_high":np.nan,"source_tasks":len(tasks)}]
        # Dominant agreement is defined on full three-effect vectors, not each component.
        wide_dom = g.pivot(index="review_id", columns="replicate", values="observed_dominant").loc[tasks]
        for a,b in ((0,1),(0,2),(1,2)):
            val,lo,hi=boot_mean((wide_dom[a] == wide_dom[b]).to_numpy(float), stable_seed(model,"dominant",str(a),str(b)))
            rows.append({"model":model,"effect":"dominant","analysis":"pairwise","pair":f"{a}-{b}","metric":"dominant_agreement","value":val,"ci95_low":lo,"ci95_high":hi,"source_tasks":len(tasks)})
        # Predict the held replicate's dominant mechanism from the other two
        # replicate effect vectors.  This keeps the decision task-level.
        dom_rows = []
        for task in tasks:
            z = {e: g[g.review_id == task].pivot(index="review_id", columns="replicate", values=f"observed_{e}").loc[task].to_numpy(float) for e in EFFECTS}
            actual = wide_dom.loc[task].to_numpy()
            for hold in range(3):
                predicted = dominant({e: float(np.delete(z[e], hold).mean()) for e in EFFECTS})
                dom_rows.append({"review_id": task, "ok": float(predicted == actual[hold])})
        per_task_dom = pd.DataFrame(dom_rows).groupby("review_id").ok.mean().loc[tasks].to_numpy(float)
        value, lo, hi = boot_mean(per_task_dom, stable_seed(model, "dominant", "loro"))
        loro_rows.append({"model": model, "effect": "dominant", "metric": "dominant_accuracy", "value": value,
                          "ci95_low": lo, "ci95_high": hi, "source_tasks": len(tasks)})
    return pd.DataFrame(rows), pd.DataFrame(loro_rows), pd.DataFrame(cess_rows)


def model_nuisance(config_path: Path, model: str) -> tuple[dict, dict, dict, float]:
    config = load_config(config_path); env = ROOT / "evidence_selection_benchmark/data/ms2_sequential_search_confirmatory_v1"; dev = ROOT / "evidence_selection_benchmark/data/ms2_external_v1/development"
    tasks = load_jsonl(env / "tasks.jsonl"); docs = load_jsonl(env / "documents.jsonl"); mc = config["models"][model]
    prediction, second, tau, _ = fit_prospective_models(dev, tasks, docs, ROOT / mc["development_scores"], ROOT / mc["confirmatory_scores"], load_config(ROOT / "configs/ms2_external_v1.yaml"))
    outcome = {r["document_id"]: float(r["direction_score"]) for r in load_jsonl(env / "private_document_outcomes.jsonl")}
    return prediction, second, outcome, float(tau)


def expected_cess(pseq: list[np.ndarray], y: np.ndarray, m: np.ndarray, lam: float, uniforms: np.ndarray) -> tuple[float,float]:
    n, k = len(y), len(pseq); anchor=float(m.mean()); estimates=[]
    for u in uniforms:
        correction=0.0
        for t,p in enumerate(pseq):
            idx=int(np.searchsorted(np.cumsum(p), u[t], side="right")); idx=min(idx,n-1)
            correction += (y[idx]-m[idx])/(n*p[idx])
        estimates.append(float(np.clip(anchor + lam*correction/k, -1,1)))
    values=np.asarray(estimates); return float(values.mean()), float(values.std(ddof=1)/math.sqrt(len(values)))


def local_replay(traces: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    rows=[]
    fixed=traces[(traces.stopping_regime=="fixed_horizon") & traces.document_id.notna()].copy()
    for model, gmodel in fixed.groupby("model_key", sort=True):
        prediction, second, outcome, tau=model_nuisance(config_path, model)
        ref=gmodel[gmodel.controller=="uniform"]
        for (task, rep), g in ref.groupby(["review_id","replicate"], sort=True):
            g=g.sort_values("round")
            if len(g)!=10: raise ValueError(f"{model}/{task}/{rep}: frozen reference schedule is not 10 rounds")
            ids0=g.candidate_document_ids.iloc[0]
            if not all(ids==ids0 for ids in g.candidate_document_ids): raise ValueError("candidate pool changed within frozen replay")
            y=np.asarray([outcome[x] for x in ids0],float); m=np.asarray([prediction[x] for x in ids0],float); second_v=np.asarray([second[x] for x in ids0],float)
            p0=[np.asarray(x,float) for x in g.candidate_probabilities]
            p1=[np.asarray(x,float) for x in g.base_agent_probabilities]
            if any(np.any(p<=0) for p in p0+p1): raise ValueError("frozen replay lacks selection support")
            l0,_=prospective_weight(tau,second_v,p0[0],10); l1,_=prospective_weight(tau,second_v,p1[0],10)
            actual0=float(np.mean([np.dot(p,y) for p in p0])); actual1=float(np.mean([np.dot(p,y) for p in p1]));
            rng=np.random.default_rng(stable_seed(model,task,str(rep),"frozen_history")); uniforms=rng.random((LOCAL_MC,10))
            cess0,se0=expected_cess(p0,y,m,l0,uniforms); cess1,se1=expected_cess(p1,y,m,l1,uniforms)
            rows.append({"model":model,"review_id":task,"replicate":int(rep),"local_actual_selection":actual1-actual0,"local_cess_selection":cess1-cess0,
                         "local_cess_mcse":float(math.sqrt(se0**2+se1**2)),"local_actual_theta_uniform":actual0,"local_actual_theta_agent":actual1,"local_cess_theta_uniform":cess0,"local_cess_theta_agent":cess1,
                         "frozen_lambda_uniform":float(l0),"frozen_lambda_agent":float(l1),"simulation_replicates":LOCAL_MC})
    return pd.DataFrame(rows)


def js(p: np.ndarray,q:np.ndarray)->float:
    m=.5*(p+q); return float(.5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m)))


def divergence(traces: pd.DataFrame) -> pd.DataFrame:
    fixed=traces[(traces.stopping_regime=="fixed_horizon") & traces.document_id.notna()].copy(); fixed["trace_key"]=fixed.model_key+"|"+fixed.review_id.astype(str)+"|"+fixed.replicate.astype(str)+"|"+fixed.controller
    corpus=fixed["query"].astype(str).tolist(); vec=TfidfVectorizer(ngram_range=(1,2),min_df=1).fit(corpus); x=vec.transform(corpus); fixed["_idx"]=np.arange(len(fixed))
    rows=[]
    for (model,task,rep),g in fixed.groupby(["model_key","review_id","replicate"],sort=True):
        a=g[g.controller=="agent"].sort_values("round"); u=g[g.controller=="uniform"].sort_values("round")
        if len(a)!=10 or len(u)!=10: raise ValueError("fixed pair incomplete")
        qdist=[]; jss=[]; first_select=first_query=None
        for (_,ra),(_,ru) in zip(a.iterrows(),u.iterrows()):
            qdist.append(1-float(x[int(ra["_idx"])].multiply(x[int(ru["_idx"])]).sum()))
            if ra["candidate_document_ids"] != ru["candidate_document_ids"]: raise ValueError("candidate exposure differs in fixed pair")
            jss.append(js(np.asarray(ra["candidate_probabilities"],float),np.asarray(ru["candidate_probabilities"],float)))
            if first_select is None and ra["document_id"] != ru["document_id"]: first_select=int(ra["round"])
            if first_query is None and ra["query"] != ru["query"]: first_query=int(ra["round"])
        a_set,u_set=set(a["document_id"]),set(u["document_id"]); selected_overlap=len(a_set&u_set)/len(a_set|u_set)
        weights=[]
        for (_,ra),(_,ru) in zip(a.iterrows(),u.iterrows()):
            pos=ra["candidate_document_ids"].index(ra["document_id"]); weights.append(float(ra["candidate_probabilities"][pos]/ru["candidate_probabilities"][pos]))
        weights=np.asarray(weights); ess=float(weights.sum()**2/np.square(weights).sum())
        rows.append({"model":model,"review_id":task,"replicate":int(rep),"successive_query_tfidf_distance":float(np.mean(qdist)),"candidate_distribution_js":float(np.mean(jss)),"candidate_set_overlap":1.0,"selected_document_overlap":selected_overlap,"first_selection_divergence_round":first_select or 11,"first_query_divergence_round":first_query or 11,"first_trajectory_divergence_round":min(first_select or 11,first_query or 11),"trajectory_importance_weight_ess":ess})
    return pd.DataFrame(rows)


def render(out:Path, local:pd.DataFrame, rel:pd.DataFrame, selective:pd.DataFrame)->None:
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10}); colors={"olmo3_7b":"#CC6677","qwen2_5_32b":"#4477AA"}
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.5),constrained_layout=True)
    for ax,actual,pred,title in zip(axes,["local_actual_selection","actual_total_selection"],["local_cess_selection","cess_total_selection"],["Frozen-history local selection effect","Total policy selection effect"]):
        for model,g in local.groupby("model"):
            ax.scatter(g[actual],g[pred],s=18,alpha=.6,color=colors[model],label=model)
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]); hi=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([lo,hi],[lo,hi],color="0.35"); ax.set(xlabel="Observed effect",ylabel="CESS-predicted effect",title=title)
    axes[0].legend(fontsize=8); fig.savefig(out/"local_vs_total_effect.pdf",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.5),constrained_layout=True)
    for model,g in selective.groupby("model"): ax.plot(g.coverage,g.risk,marker="o",label=model,color=colors[model])
    ax.set(xlabel="Coverage by realized ESS",ylabel="CESS total-effect MAE",title="Descriptive selective risk--coverage");ax.legend();fig.savefig(out/"selective_risk_coverage.pdf",bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.5),constrained_layout=True)
    for model,g in rel[rel.metric=="ICC_1_3_gold_mean"].groupby("model"): ax.bar(model,g.value.mean(),color=colors[model])
    ax.set(ylim=(0,1),ylabel="ICC(1,3)",title="Reliability of three-replicate direct effects");fig.savefig(out/"direct_effect_reliability.pdf",bbox_inches="tight");plt.close(fig)


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--raw",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--finalize-existing",action="store_true");ap.add_argument("--refresh-reliability-existing",action="store_true");args=ap.parse_args()
    if args.refresh_reliability_existing:
        source=args.output/"replicate_direct_effects.csv"
        if not source.exists(): raise FileNotFoundError(f"cannot refresh missing {source}")
        rep=pd.read_csv(source); reliability,loro,cess_rel=reliability_analysis(rep)
        reliability.to_csv(args.output/"effect_reliability.csv",index=False);loro.to_csv(args.output/"leave_one_replicate_out.csv",index=False);cess_rel.to_csv(args.output/"cess_noise_ceiling_comparison.csv",index=False)
        print(json.dumps({"output":str(args.output),"reliability_refreshed":True},indent=2)); return
    if args.finalize_existing:
        required=["local_total_feedback.csv","effect_reliability.csv","selective_risk.csv"]
        missing=[x for x in required if not (args.output/x).exists()]
        if missing: raise FileNotFoundError(f"cannot finalize missing outputs: {missing}")
        (args.output/"figures").mkdir(exist_ok=True)
        result=pd.read_csv(args.output/"local_total_feedback.csv"); reliability=pd.read_csv(args.output/"effect_reliability.csv"); selective=pd.read_csv(args.output/"selective_risk.csv")
        render(args.output/"figures",result,reliability,selective)
        report=["# Analysis v5: direct-effect reliability and frozen-history feedback", "", "## Completed", "", "All calculations use the completed 2,400 true-Agent MS2 episodes. Replicate-specific effects, task-cluster bootstrap summaries, frozen-history local replay, divergence diagnostics, and fixed-grid risk--coverage outputs are saved beside this report.", "", "## Endpoint distinction", "", "The frozen-history local selection effect keeps the uniform-controller future query sequence, candidate pool, and fixed ten-round schedule unchanged, then changes only the per-round selection distribution from uniform to the logged agent base distribution. The total selection-policy effect is the observed agent-versus-uniform fixed-horizon contrast from the real LLM trajectories. Their difference is feedback selection effect. Local is not called a total policy effect.", "", "## Abstention calibration", "", "No same-protocol true-Agent development trajectories are available. Therefore no reliability threshold was selected. `selective_risk.csv` is a descriptive fixed coverage curve, explicitly marked as uncalibrated; it is not confirmation of a deployment abstention rule.", "", "## Limits", "", "The local CESS calculation uses private outcomes only after simulated documents have been selected to evaluate the estimator. Selection probabilities come exclusively from logged frozen queries and candidate distributions. The runner does not contain final natural-language reports, so these remain opened-evidence rather than report-level effects."]
        metric=pd.read_csv(args.output/"local_total_metrics.csv"); loro=pd.read_csv(args.output/"leave_one_replicate_out.csv")
        def lookup(df, **where):
            q=df
            for col,val in where.items(): q=q[q[col] == val]
            return q.iloc[0]
        report += ["", "## Numerical findings", ""]
        for model in ("olmo3_7b", "qwen2_5_32b"):
            local_mae=lookup(metric,model=model,endpoint="frozen_history_local_selection",metric="MAE"); total_mae=lookup(metric,model=model,endpoint="real_trajectory_total_selection",metric="MAE"); total_sp=lookup(metric,model=model,endpoint="real_trajectory_total_selection",metric="Spearman"); ltotal=lookup(loro,model=model,effect="total",metric="Spearman")
            report.append(f"- {model}: CESS MAE is {local_mae.value:.3f} for frozen-history local selection versus {total_mae.value:.3f} for real-trajectory total selection; total-effect Spearman is {total_sp.value:.3f} [{total_sp.ci95_low:.3f}, {total_sp.ci95_high:.3f}], and leave-one-replicate-out total-effect Spearman is {ltotal.value:.3f} [{ltotal.ci95_low:.3f}, {ltotal.ci95_high:.3f}].")
        report += ["", "## Interpretation", "", "Selection effects have moderate three-replicate reliability, whereas stopping effects are close to irreproducible in this configuration and dominant-mechanism labels have low pairwise agreement. The data therefore support a limited, behavior-level selection-effect analysis, but do not support a task-level claim that CESS reliably identifies a single dominant causal mechanism. No divergence diagnostic has a bootstrap interval excluding zero for either absolute feedback selection effect or CESS total-effect error."]
        (args.output/"analysis_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
        write_json(args.output/"run_manifest.json",{"raw":str(args.raw),"bootstrap":N_BOOT,"local_mc":LOCAL_MC,"seed":SEED,"development_threshold_selected":False,"finalized_from_existing_cpu_outputs":True})
        print(json.dumps({"output":str(args.output),"finalized":True},indent=2)); return
    if args.output.exists(): raise FileExistsError(f"refusing overwrite {args.output}")
    args.output.mkdir(parents=True);(args.output/"figures").mkdir()
    cells,traces=load_cells_and_traces(args.raw)
    rep=replicate_effects(cells).merge(episode_ess(traces), on=["model","review_id","replicate"], validate="one_to_one")
    rep.to_csv(args.output/"replicate_direct_effects.csv",index=False)
    reliability,loro,cess_rel=reliability_analysis(rep); reliability.to_csv(args.output/"effect_reliability.csv",index=False);loro.to_csv(args.output/"leave_one_replicate_out.csv",index=False);cess_rel.to_csv(args.output/"cess_noise_ceiling_comparison.csv",index=False)
    local_rep=local_replay(traces,ROOT/"configs/interpretability_v1_true_agent_ms2.yaml"); div_rep=divergence(traces)
    task=rep.groupby(["model","review_id"],as_index=False).mean(numeric_only=True); local=local_rep.groupby(["model","review_id"],as_index=False).mean(numeric_only=True); div=div_rep.groupby(["model","review_id"],as_index=False).mean(numeric_only=True)
    result=task.merge(local,on=["model","review_id"],validate="one_to_one").merge(div,on=["model","review_id"],validate="one_to_one")
    result["actual_total_selection"]=result.observed_selection;result["cess_total_selection"]=result.cess_selection;result["feedback_selection"]=result.actual_total_selection-result.local_actual_selection;result["cess_local_error"]=result.local_cess_selection-result.local_actual_selection;result["cess_total_error"]=result.cess_total-result.observed_total
    result.to_csv(args.output/"local_total_feedback.csv",index=False);div.to_csv(args.output/"divergence_metrics.csv",index=False)
    local_total_metrics=[]
    for model,g in result.groupby("model", sort=True):
        for endpoint, actual, pred in (("frozen_history_local_selection", "local_actual_selection", "local_cess_selection"),
                                       ("real_trajectory_total_selection", "observed_selection", "cess_selection")):
            err=(g[pred]-g[actual]).to_numpy(float)
            mae,lo,hi=boot_mean(np.abs(err),stable_seed(model,endpoint,"mae"))
            sp,sl,sh=boot_corr(g[pred].to_numpy(float),g[actual].to_numpy(float),stable_seed(model,endpoint,"spearman"),True)
            direction=np.asarray([ternary(x)==ternary(y) for x,y in zip(g[pred],g[actual])],float)
            da,dl,dh=boot_mean(direction,stable_seed(model,endpoint,"direction"))
            local_total_metrics += [{"model":model,"endpoint":endpoint,"metric":"MAE","value":mae,"ci95_low":lo,"ci95_high":hi,"source_tasks":len(g)},
                                    {"model":model,"endpoint":endpoint,"metric":"Spearman","value":sp,"ci95_low":sl,"ci95_high":sh,"source_tasks":len(g)},
                                    {"model":model,"endpoint":endpoint,"metric":"direction_accuracy","value":da,"ci95_low":dl,"ci95_high":dh,"source_tasks":len(g)}]
    pd.DataFrame(local_total_metrics).to_csv(args.output/"local_total_metrics.csv",index=False)
    correlations=[]; divcols=[c for c in div.columns if c not in {"model","review_id","replicate"}]
    for model,g in result.groupby("model"):
        for outcome,col in (("abs_feedback_selection","feedback_selection"),("abs_CESS_total_error","cess_total_error")):
            y=g[col].abs().to_numpy()
            for d in divcols:
                point,lo,hi=boot_corr(g[d].to_numpy(),y,stable_seed(model,outcome,d),True);correlations.append({"model":model,"outcome":outcome,"divergence":d,"spearman":point,"ci95_low":lo,"ci95_high":hi,"source_tasks":len(g)})
    pd.DataFrame(correlations).to_csv(args.output/"divergence_correlations.csv",index=False)
    # No same-protocol development trajectories exist. These are fixed descriptive coverage points, not a threshold selected on test.
    sr=[]
    for model,g in result.groupby("model"):
        g=g.sort_values("ess_11",ascending=False)
        for cov in (.2,.4,.6,.8,1.0):
            h=g.iloc[:math.ceil(cov*len(g))];sr.append({"model":model,"coverage":cov,"source_tasks":len(h),"risk":float(h.cess_total_error.abs().mean()),"ess_threshold":float(h.ess_11.min()),"threshold_status":"fixed descriptive coverage; no development calibration available"})
    selective=pd.DataFrame(sr);selective.to_csv(args.output/"selective_risk.csv",index=False)
    # Add a proper across-replicate standard error before uncertainty strata.
    suggestions=[]
    replicate_se = rep.groupby(["model","review_id"]).observed_total.std().div(math.sqrt(3)).rename("observed_total_replicate_se").reset_index()
    result=result.merge(replicate_se,on=["model","review_id"],validate="one_to_one")
    # Rewrite after adding a proper across-replicate standard error, rather than
    # using variation between different effect components as an uncertainty proxy.
    result.to_csv(args.output/"local_total_feedback.csv",index=False)
    # Replace provisional strata with the correctly defined gold-effect SE.
    strata=[]
    for model,g in result.groupby("model"):
        gold_se=g.observed_total_replicate_se
        mag=g.observed_total.abs();gap=np.sort(g[["observed_selection","observed_stopping","observed_interaction"]].abs().to_numpy(),axis=1)[:,-1]-np.sort(g[["observed_selection","observed_stopping","observed_interaction"]].abs().to_numpy(),axis=1)[:,-2]
        for name,mask in (("gold_SE_le_0.05",gold_se<=.05),("gold_SE_gt_0.05",gold_se>.05),("total_abs_le_0.05",mag<=.05),("total_abs_gt_0.05",mag>.05),("gap_le_0.05",gap<=.05),("gap_gt_0.05",gap>.05)):
            h=g[mask]; strata.append({"model":model,"stratum":name,"source_tasks":len(h),"cess_total_mae":float(h.cess_total_error.abs().mean()) if len(h) else np.nan,"cess_total_spearman":float(h.cess_total.corr(h.observed_total,method="spearman")) if len(h)>2 else np.nan})
    pd.DataFrame(strata).to_csv(args.output/"stratified_results.csv",index=False)
    merged=result.pivot(index="review_id",columns="model",values=["observed_total","cess_total_error"])
    for taskid,row in merged.iterrows():
        a,b=row["observed_total"].iloc[0],row["observed_total"].iloc[1]
        se=float(replicate_se[replicate_se.review_id == taskid].observed_total_replicate_se.max())
        if abs(a)>=.15 and abs(b)>=.15 and ternary(a)==ternary(b) and se>=.05: suggestions.append({"review_id":taskid,"effect":"total","model_effect_1":a,"model_effect_2":b,"max_within_model_SE":se,"reason":"predefined: |effect|>=0.15, same nonzero sign across models, SE>=0.05","recommended_total_replicates":10})
    pd.DataFrame(suggestions).to_csv(args.output/"repeat_recommendations.csv",index=False)
    render(args.output/"figures",result,reliability,selective)
    report=["# Analysis v5: direct-effect reliability and frozen-history feedback", "", "## Completed", "", "All calculations use the completed 2,400 true-Agent MS2 episodes. Replicate-specific effects, task-cluster bootstrap summaries, frozen-history local replay, divergence diagnostics, and fixed-grid risk--coverage outputs are saved beside this report.", "", "## Endpoint distinction", "", "The frozen-history local selection effect keeps the uniform-controller future query sequence, candidate pool, and fixed ten-round schedule unchanged, then changes only the per-round selection distribution from uniform to the logged agent base distribution. The total selection-policy effect is the observed agent-versus-uniform fixed-horizon contrast from the real LLM trajectories. Their difference is feedback selection effect. Local is not called a total policy effect.", "", "## Abstention calibration", "", "No same-protocol true-Agent development trajectories are available. Therefore no reliability threshold was selected. `selective_risk.csv` is a descriptive fixed coverage curve, explicitly marked as uncalibrated; it is not confirmation of a deployment abstention rule.", "", "## Limits", "", "The local CESS calculation uses private outcomes only after simulated documents have been selected to evaluate the estimator. Selection probabilities come exclusively from logged frozen queries and candidate distributions. The runner does not contain final natural-language reports, so these remain opened-evidence rather than report-level effects."]
    (args.output/"analysis_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    write_json(args.output/"run_manifest.json",{"raw":str(args.raw),"bootstrap":N_BOOT,"local_mc":LOCAL_MC,"seed":SEED,"development_threshold_selected":False})
    print(json.dumps({"output":str(args.output),"tasks":len(result)},indent=2))

if __name__=="__main__":main()
