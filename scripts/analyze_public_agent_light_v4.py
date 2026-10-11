#!/usr/bin/env python3
"""CPU-only post-test analysis for the preregistered public-agent-light v4 run.

The script deliberately does not make model calls, alter trajectories, or choose
gamma/lambda from holdout data.  It consumes only v4's frozen configuration and
test JSONL and writes a new ``analysis`` directory once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOT = 10_000
RANKS = ("supporting_first", "opposing_first")
K_VALUES = (1, 2, 4)
GRID = np.round(np.arange(0.0, 1.0001, .05), 2)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def clip(x): return float(np.clip(x, -1., 1.))


def estimate_episode(ep, k: int, global_mean: float):
    """All estimators use the first k logged selections and their logged p's."""
    ss = ep["searches"][:k]
    y = np.asarray([s["selected_outcome"] for s in ss], float)
    p = np.asarray([s["selected_probability"] for s in ss], float)
    n = np.asarray([len(s["candidates"]) for s in ss], float)
    # OR is available before any selection: the full frozen pool prediction mean.
    a = float(np.mean([c["outcome_prediction"] for c in ss[0]["candidates"]]))
    w = 1. / (n * p)
    residual = y - np.asarray([s["selected_outcome_prediction"] for s in ss], float)
    opened = float(y.mean())
    ipw = clip(np.mean(w * y))
    snipw = clip(np.sum(w * y) / np.sum(w))
    dr_raw = a + float(np.mean(w * residual))
    snder = clip(a + float(np.sum(w * residual) / np.sum(w)))
    row = {
        "Opened Mean": opened, "Outcome Regression": a,
        "Sequential IPW": ipw, "Self-normalized IPW": snipw,
        "Sequential DR": clip(dr_raw), "Self-normalized DR": snder,
        "Weight-clipped DR (5)": clip(a + np.mean(np.minimum(w, 5.) * residual)),
        "Weight-clipped DR (10)": clip(a + np.mean(np.minimum(w, 10.) * residual)),
        "Weight-clipped DR (20)": clip(a + np.mean(np.minimum(w, 20.) * residual)),
        "CESS without stopping correction": clip(a + .5 * (clip(dr_raw) - a)),
        "Full CESS": clip(a + .5 * (clip(dr_raw) - a)),
    }
    for lam in GRID:
        suffix = f"{lam:.2f}"
        row[f"Opened Mean -> Global Mean [lambda={suffix}]"] = clip(opened + lam * (global_mean - opened))
        row[f"Opened Mean -> OR [lambda={suffix}]"] = clip(opened + lam * (a - opened))
    diag = {"opened_unique_documents": len(set(s["selected_document_id"] for s in ss)),
            "repeat_selection_rate": 1 - len(set(s["selected_document_id"] for s in ss)) / k,
            "weight_ess": float(w.sum() ** 2 / np.square(w).sum()),
            "weight_max": float(w.max()), "weight_min": float(w.min()),
            "raw_dr": float(dr_raw), "raw_dr_outside_unit": bool(abs(dr_raw) > 1.)}
    return row, diag


def long_estimates(episodes, config):
    rows, diagnostics = [], []
    for ep in episodes:
        base = {"review_id": ep.get("review_id"), "seed": ep.get("seed"),
                "rank_condition": ep.get("ranking"), "target": ep.get("target"),
                "episode_error": ep.get("error"), "search_count": ep.get("search_count")}
        ok = not ep.get("error") and ep.get("search_count") == 4
        for k in K_VALUES:
            if ok:
                values, diag = estimate_episode(ep, k, float(config["estimators"]["development_global_mean"]))
                for method, estimate in values.items():
                    rows.append({**base, "horizon": k, "method": method, "estimate": estimate,
                                 "successful_episode": True})
                diagnostics.append({**base, "horizon": k, **diag, "successful_episode": True})
            else:
                rows.append({**base, "horizon": k, "method": "__failed_episode__", "estimate": np.nan,
                             "successful_episode": False})
                diagnostics.append({**base, "horizon": k, "successful_episode": False})
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def method_family(method: str):
    if method.startswith("Opened Mean -> Global"): return "ordinary shrinkage: global"
    if method.startswith("Opened Mean -> OR"): return "ordinary shrinkage: OR"
    return method


def lambda_of(method: str):
    if "lambda=" not in method: return np.nan
    return float(method.split("lambda=")[1].split("]")[0])


def task_seed_metrics(frame: pd.DataFrame, method: str, horizon: int):
    f = frame[(frame.method == method) & (frame.horizon == horizon) & frame.successful_episode].copy()
    f["abs_error"] = (f.estimate - f.target).abs()
    # Seed MAE averages both rank conditions; seed sensitivity is |rank-means difference|.
    mae = f.groupby(["review_id", "seed"], as_index=False).abs_error.mean().rename(columns={"abs_error":"mae"})
    piv = f.pivot_table(index=["review_id", "seed"], columns="rank_condition", values="estimate", aggfunc="mean")
    piv = piv.reindex(columns=RANKS).dropna().reset_index()
    piv["sensitivity"] = (piv[RANKS[0]] - piv[RANKS[1]]).abs()
    # Direction error is per rank condition, then averaged within task/seed.
    f["wrong_direction"] = (np.sign(f.estimate) != np.sign(f.target)).astype(float)
    direction = f.groupby(["review_id", "seed"], as_index=False).wrong_direction.mean()
    out = mae.merge(piv[["review_id","seed","sensitivity"]], on=["review_id","seed"], how="inner")
    return out.merge(direction, on=["review_id","seed"], how="inner")


def aggregate_metrics(frame: pd.DataFrame):
    rows=[]
    for h in K_VALUES:
        for method in sorted(x for x in frame.method.unique() if x != "__failed_episode__"):
            ts=task_seed_metrics(frame,method,h)
            rows.append({"horizon":h,"method":method,"family":method_family(method),"lambda":lambda_of(method),
                         "source_tasks":ts.review_id.nunique(),"task_seed_pairs":len(ts),
                         "mae":ts.mae.mean(),"ranking_sensitivity":ts.sensitivity.mean(),
                         "wrong_direction_rate":ts.wrong_direction.mean()})
    return pd.DataFrame(rows)


def nearest_intersection(lams, x, y, target):
    hits=[]
    for i in range(len(lams)-1):
        d=x[i+1]-x[i]
        if abs(x[i]-target) <= 1e-12: hits.append((lams[i],y[i]))
        if abs(d)>1e-12:
            w=(target-x[i])/d
            if -1e-12 <= w <= 1+1e-12: hits.append((lams[i]+np.clip(w,0,1)*(lams[i+1]-lams[i]),y[i]+np.clip(w,0,1)*(y[i+1]-y[i])))
    if abs(x[-1]-target)<=1e-12: hits.append((lams[-1],y[-1]))
    if not hits: return None
    near=lams[int(np.argmin(np.abs(x-target)))]
    return min(hits,key=lambda z:(abs(z[0]-near),z[0]))


def bootstrap_match(frame, horizon, family, match, seed):
    cess=task_seed_metrics(frame,"Full CESS",horizon)
    grid=[f"Opened Mean -> {'Global Mean' if family.endswith('global') else 'OR'} [lambda={x:.2f}]" for x in GRID]
    base={m:task_seed_metrics(frame,m,horizon) for m in grid}
    tasks=sorted(set(cess.review_id).intersection(*[set(x.review_id) for x in base.values()]))
    # matrices: task x seed. Keep rank paired inside each seed; sample seed once per task.
    seeds={t:sorted(set(cess[cess.review_id==t].seed).intersection(*[set(x[x.review_id==t].seed) for x in base.values()])) for t in tasks}
    tasks=[t for t in tasks if seeds[t]]
    C=np.asarray([[[cess[(cess.review_id==t)&(cess.seed==s)].iloc[0][q] for q in ("mae","sensitivity")] for s in seeds[t]] for t in tasks])
    B=np.asarray([[[[base[m][(base[m].review_id==t)&(base[m].seed==s)].iloc[0][q] for q in ("mae","sensitivity")] for m in grid] for s in seeds[t]] for t in tasks])
    # uneven seed counts are not expected (three), but make behaviour explicit.
    if len({len(x) for x in seeds.values()}) != 1: raise RuntimeError("incomplete seed structure")
    n, ns=len(tasks),len(seeds[tasks[0]]); rng=np.random.default_rng(seed); effects=[]; roots=[]
    for _ in range(BOOT):
        # Hierarchical paired bootstrap: resample source tasks first, then
        # resample all seeds with replacement inside every sampled task.
        ti=rng.integers(0,n,n); si=rng.integers(0,ns,(n,ns))
        c=np.asarray([C[t,si[j]].mean(axis=0) for j,t in enumerate(ti)]).mean(axis=0)
        b=np.asarray([B[t,si[j]].mean(axis=0) for j,t in enumerate(ti)]).mean(axis=0) # lambda x (mae,sens)
        if match=="mae": root=nearest_intersection(GRID,b[:,0],b[:,1],c[0]); v=c[1]
        else: root=nearest_intersection(GRID,b[:,1],b[:,0],c[1]); v=c[0]
        if root is not None: effects.append(v-root[1]); roots.append(root[0])
    # point estimate uses task and seed mean, no resampling.
    c=C.mean(axis=(0,1)); b=B.mean(axis=(0,1))
    root=nearest_intersection(GRID,b[:,0] if match=="mae" else b[:,1], b[:,1] if match=="mae" else b[:,0], c[0] if match=="mae" else c[1])
    return {"horizon":horizon,"baseline_family":family,"match_metric":match,"source_tasks":n,
            "status":"matched_by_interpolation" if root else "not_matchable","cess_mae":c[0],"cess_ranking_sensitivity":c[1],
            "interpolated_lambda":np.nan if not root else root[0],"baseline_matched_value":np.nan if not root else root[1],
            "cess_minus_baseline":np.nan if not root else (c[1] if match=="mae" else c[0])-root[1],
            "bootstrap_replicates":BOOT,"bootstrap_matchable_replicates":len(effects),"bootstrap_matchable_fraction":len(effects)/BOOT,
            "ci95_low":np.nan if not effects else np.quantile(effects,.025),"ci95_high":np.nan if not effects else np.quantile(effects,.975),
            "significantly_lower":bool(effects and np.quantile(effects,.975)<0),"bootstrap_lambda_median":np.nan if not roots else np.median(roots)}


def paired_contrasts(frame):
    rows=[]; comparisons=["Opened Mean","Outcome Regression","Sequential IPW","Sequential DR","Self-normalized DR","CESS without stopping correction"]
    for h in K_VALUES:
        c=task_seed_metrics(frame,"Full CESS",h).set_index(["review_id","seed"])
        for base in comparisons:
            b=task_seed_metrics(frame,base,h).set_index(["review_id","seed"]); z=c.join(b,lsuffix="_cess",rsuffix="_base",how="inner")
            for metric in ("mae","sensitivity","wrong_direction"):
                dif=(z[f"{metric}_cess"]-z[f"{metric}_base"]).to_numpy(float); rng=np.random.default_rng(2026092404+h+len(base));
                # source-task block plus within-task seed sampling.
                by={t:g for t,g in z.assign(diff=z[f"{metric}_cess"]-z[f"{metric}_base"]).groupby(level=0)}; ids=list(by); vals=[]
                for _ in range(BOOT):
                    samp=rng.integers(0,len(ids),len(ids)); task_means=[]
                    for i in samp:
                        g=by[ids[i]]["diff"].to_numpy(float)
                        task_means.append(g[rng.integers(0,len(g),len(g))].mean())
                    vals.append(float(np.mean(task_means)))
                rows.append({"horizon":h,"comparison":"Full CESS - "+base,"metric":metric,"source_tasks":len(ids),"difference":dif.mean(),"ci95_low":np.quantile(vals,.025),"ci95_high":np.quantile(vals,.975)})
    return pd.DataFrame(rows)


def figures(metrics, mmae, msens, out):
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10})
    for h in K_VALUES:
        f,ax=plt.subplots(figsize=(5.4,4.0)); d=metrics[metrics.horizon==h]
        for fam,col in [("ordinary shrinkage: global","#73a9cf"),("ordinary shrinkage: OR","#e6a0c4")]:
            x=d[d.family==fam].sort_values("lambda"); ax.plot(x.mae,x.ranking_sensitivity,"o-",label=fam.replace("ordinary shrinkage: ","shrink: "),color=col,ms=3)
        for m,mark,col in [("Opened Mean","s","#777777"),("Outcome Regression","^","#777777"),("Full CESS","*","#d55e00")]:
            z=d[d.method==m]; ax.scatter(z.mae,z.ranking_sensitivity,s=75 if m=="Full CESS" else 30,marker=mark,color=col,label=m,zorder=4)
        ax.set(xlabel="Pool-target MAE (lower is better)",ylabel="Ranking sensitivity (lower is better)",title=f"K={h}");ax.legend(fontsize=7);ax.grid(alpha=.2);f.tight_layout();f.savefig(out/f"mae_sensitivity_tradeoff_k{h}.pdf");plt.close(f)
    f,ax=plt.subplots(figsize=(6,3.6)); rows=pd.concat([mmae.assign(kind="matched MAE"),msens.assign(kind="matched sensitivity")]); rows=rows[rows.status=="matched_by_interpolation"].copy();
    rows["label"]=rows.baseline_family.str.replace("ordinary shrinkage: ","")+" / K="+rows.horizon.astype(str)+" / "+rows.kind
    for i,r in enumerate(rows.itertuples()):
        ax.errorbar(r.cess_minus_baseline,i,xerr=[[r.cess_minus_baseline-r.ci95_low],[r.ci95_high-r.cess_minus_baseline]],fmt="o",color="#d55e00")
    ax.axvline(0,color="black",lw=.8);ax.set_yticks(range(len(rows)),rows.label,fontsize=7);ax.set_xlabel("Full CESS minus matched baseline (lower is better)");ax.grid(axis="x",alpha=.2);f.tight_layout();f.savefig(out/"matched_effects.pdf");plt.close(f)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--analysis-name",default="analysis_final");ap.add_argument("--figures-name",default="figures_final");a=ap.parse_args(); root=a.output; out=root/a.analysis_name; figs=root/a.figures_name
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite {out}")
    config=json.loads((root/"frozen_test_config.json").read_text()); episodes=read_jsonl(root/"test/episodes.jsonl")
    out.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    long,diag=long_estimates(episodes,config); long.to_csv(out/"test_estimates_long.csv",index=False);diag.to_csv(out/"trajectory_diagnostics.csv",index=False)
    metrics=aggregate_metrics(long);metrics.to_csv(out/"method_metrics.csv",index=False)
    contrasts=paired_contrasts(long);contrasts.to_csv(out/"paired_contrasts.csv",index=False)
    mmae=pd.DataFrame([bootstrap_match(long,h,f,"mae",2026092500+h+i) for h in K_VALUES for i,f in enumerate(("ordinary shrinkage: global","ordinary shrinkage: OR"))]);msens=pd.DataFrame([bootstrap_match(long,h,f,"sensitivity",2026092600+h+i) for h in K_VALUES for i,f in enumerate(("ordinary shrinkage: global","ordinary shrinkage: OR"))]);mmae.to_csv(out/"matched_mae_interpolated.csv",index=False);msens.to_csv(out/"matched_sensitivity_interpolated.csv",index=False)
    figures(metrics,mmae,msens,figs)
    attempted=len(config["test_task_ids"])*len(config["seeds"])*2; success=int(long[long.successful_episode][["review_id","seed","rank_condition"]].drop_duplicates().shape[0]);
    report=["# Public-agent light v4 test analysis", "", "## Scope", "", "CPU-only post-test analysis of the frozen v4 trajectories. Gamma and the estimator grid were fixed before test execution; this report does not select a method from test results.", "", "## Completion", "", f"- Planned episodes: {attempted}; successful four-search episodes: {success}; failed/missing: {attempted-success}.", f"- Planned source tasks: {len(config['test_task_ids'])}; seeds per paired condition: {len(config['seeds'])}.", "- All-task rows are retained in `test_estimates_long.csv`; estimation metrics use complete four-search episodes and are accompanied by the completion count above.", "", "## Matching", "", "For each shrinkage family and horizon, matching is performed only by piecewise-linear interpolation inside the lambda grid. No extrapolation is allowed. Each bootstrap replicate resamples source tasks and then a seed within each selected task, recomputing its own intersection.", "", "## Interpretation guardrails", "", "No-stop CESS is numerically identical to Full CESS here because the experiment uses a fixed four-search budget and the frozen continuation probability is one. This is an experimental-design fact, not evidence that stopping correction is generally unnecessary.", "", "## Files", "", "- `method_metrics.csv`: task-seed macro averages for all fixed estimators.", "- `paired_contrasts.csv`: hierarchical paired source-task bootstrap contrasts.", "- `matched_*_interpolated.csv`: strict MAE and sensitivity matching results.", "- `trajectory_diagnostics.csv`: per-episode/horizon weight and repeat diagnostics."]
    (out/"analysis_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    dump_json(out/"run_manifest.json",{"cpu_only":True,"model_calls":0,"latex_modified":False,"episodes":len(episodes),"bootstrap_replicates":BOOT,"bootstrap":"source task then seed within task","frozen_configuration_sha256":config.get("configuration_sha256")})

if __name__=="__main__": main()
