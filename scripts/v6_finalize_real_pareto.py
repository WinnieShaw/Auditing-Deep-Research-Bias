#!/usr/bin/env python3
"""Complete real-trajectory v6 Pareto analysis and assemble final deliverables."""
from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
GRID=np.round(np.arange(0,1.0001,.05),2)
BOOT=10_000
SEED=2026092306

def readj(path):
    with Path(path).open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def clip(x):return np.clip(np.asarray(x,float),-1,1)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def pairnames(d):
    s=set(d.rank_condition)
    return ("pro_increase","pro_decrease") if {"pro_increase","pro_decrease"}<=s else ("pro_benefit","pro_harm")

def trace_estimators(cells,traces,outcomes,ndocs,horizon,keycols):
    groups=defaultdict(list)
    for x in traces:
        if x.get("stopped") or x.get("document_id") is None:continue
        key=tuple(str(x[c]) if c!="replicate" else int(x[c]) for c in keycols)
        groups[key].append(x)
    rows=[]
    for _,z in cells.iterrows():
        key=tuple(str(z[c]) if c!="replicate" else int(z[c]) for c in keycols); hist=sorted(groups[key],key=lambda x:int(x["round"]))
        hist=[x for x in hist if int(x["round"])<=horizon]
        n=ndocs[str(z.review_id)]; anchor=float(z.outcome_regression); ws=[]; ys=[]; residual=[]
        for x in hist:
            p=float(x["selection_probability"]); s=float(x["survival_probability"]); y=float(outcomes[str(x["document_id"])]); w=1/(n*p*s); ws.append(w);ys.append(y);residual.append(y-anchor)
        ws=np.asarray(ws);ys=np.asarray(ys);residual=np.asarray(residual)
        if len(ws)==0: raise ValueError(f"empty history {key}")
        rows.append({"sn_ipw":float(np.sum(ws*ys)/np.sum(ws)),"sn_dr":float(anchor+np.sum(ws*residual)/np.sum(ws)),"clip5_dr":float(anchor+np.sum(np.minimum(ws,5)*residual)/horizon),"clip10_dr":float(anchor+np.sum(np.minimum(ws,10)*residual)/horizon),"clip20_dr":float(anchor+np.sum(np.minimum(ws,20)*residual)/horizon),"ess":float(np.sum(ws)**2/np.sum(ws**2)),"p95_weight":float(np.quantile(ws,.95)),"rounds_trace":len(ws)})
    return pd.DataFrame(rows,index=cells.index)

def add_estimates(df,dataset,model,horizon,global_mean,source_lambda=None,source_values=None):
    base=df.copy(); records=[]
    common=["review_id","rank_condition","replicate","target","controller","stopping_regime"]
    if "model_key" in base: common.append("model_key")
    def emit(method,family,lam,values,ess=np.nan,is_main=False,note=""):
        q=base[common].copy();q["dataset"]=dataset;q["query_model"]=model;q["horizon"]=horizon;q["method"]=method;q["family"]=family;q["lambda"]=lam;q["estimate"]=clip(values);q["ess"]=ess;q["is_main"]=is_main;q["note"]=note;records.append(q)
    o=base.naive_opened_mean.to_numpy(float); a=base.outcome_regression.to_numpy(float)
    emit("Opened Mean","Opened Mean",np.nan,o)
    for l in GRID:
        emit("Opened Mean → Global Mean", "ordinary shrinkage: global",l,o+l*(global_mean-o))
        emit("Opened Mean → OR", "ordinary shrinkage: OR",l,o+l*(a-o))
    emit("Sequential IPW","Sequential IPW",np.nan,base.sequential_ipw)
    emit("Sequential DR","Sequential DR",np.nan,base.sequential_dr,base.ess)
    emit("Self-normalized IPW","Self-normalized IPW",np.nan,base.sn_ipw,base.ess)
    emit("Self-normalized DR","Self-normalized DR",np.nan,base.sn_dr,base.ess)
    for cap in (5,10,20):emit(f"Weight-clipped DR ({cap})","Weight-clipped DR",float(cap),base[f"clip{cap}_dr"],base.ess)
    raw=base.sequential_dr.to_numpy(float)
    nostop=base.cess_no_stop_raw.to_numpy(float) if "cess_no_stop_raw" in base else raw
    for l in GRID:
        emit("CESS without stopping correction","CESS no stopping",l,a+l*(clip(nostop)-a),base.ess,is_main=(l==.5),note="estimator-information ablation")
        emit("Full CESS","Full CESS",l,a+l*(clip(raw)-a),base.ess,is_main=(l==.5),note="lambda-grid sensitivity point")
    if source_lambda is not None:
        lam=np.asarray(source_lambda,float); frozen=np.asarray(source_values,float) if source_values is not None else a+lam*(clip(raw)-a);emit("Full CESS (frozen original)","Full CESS",float(np.mean(lam)),frozen,base.ess,True,"paper-frozen logged estimator; not selected on test")
    return pd.concat(records,ignore_index=True)

def metric_vectors(d):
    e=d.estimate-d.target
    mae=pd.DataFrame({"review_id":d.review_id.astype(str),"v":abs(e)}).groupby("review_id").v.mean()
    bias=pd.DataFrame({"review_id":d.review_id.astype(str),"v":e}).groupby("review_id").v.mean()
    wrong=pd.DataFrame({"review_id":d.review_id.astype(str),"v":(np.sign(d.estimate)!=np.sign(d.target)).astype(float)}).groupby("review_id").v.mean()
    a,b=pairnames(d); ctx=[c for c in ("review_id","model_key","controller","stopping_regime") if c in d]
    z=d[ctx+["rank_condition","replicate","estimate"]].groupby(ctx+["rank_condition"],as_index=False).estimate.mean(); w=z.pivot(index=ctx,columns="rank_condition",values="estimate"); sens=(w[a]-w[b]).abs().groupby(level="review_id").mean()
    return {"mae":mae,"bias":bias,"wrong_direction_rate":wrong,"ranking_sensitivity":sens}

def boot_mean(v,seed,n=BOOT):
    x=np.asarray(v,float);rng=np.random.default_rng(seed); vals=[]
    for _ in range(n//500): vals.append(x[rng.integers(0,len(x),(500,len(x)))].mean(1))
    b=np.concatenate(vals);return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))

def summarize(allrows):
    outputs=[];vectors={}
    groupcols=["dataset","query_model","horizon","method","family","lambda","is_main","note"]
    for key,g in allrows.groupby(groupcols,dropna=False,sort=False):
        v=metric_vectors(g); tag=tuple(key);vectors[tag]=v; rec=dict(zip(groupcols,key));rec.update({"source_tasks":len(v["mae"]),"trajectories":len(g),"valid_rank_pairs":len(v["ranking_sensitivity"]),"mean_ess":float(g.ess.mean()) if g.ess.notna().any() else np.nan})
        for i,k in enumerate(("mae","ranking_sensitivity","bias","wrong_direction_rate")):
            m,lo,hi=boot_mean(v[k],SEED+i+len(outputs)*7,2000 if not rec["is_main"] else BOOT);rec[k]=m;rec[k+"_ci95_low"]=lo;rec[k+"_ci95_high"]=hi
        outputs.append(rec)
    return pd.DataFrame(outputs),vectors

def frontier(m):
    out=[]
    for _,g in m.groupby(["dataset","query_model","horizon"]):
        for i,r in g.iterrows():
            dominated=((g.mae<=r.mae)&(g.ranking_sensitivity<=r.ranking_sensitivity)&((g.mae<r.mae)|(g.ranking_sensitivity<r.ranking_sensitivity))).any()
            if not dominated:out.append(r)
    return pd.DataFrame(out)

def paired_boot(a,b,seed):
    k=sorted(set(a.index)&set(b.index));d=(a.reindex(k)-b.reindex(k)).to_numpy();m,lo,hi=boot_mean(d,seed,BOOT);return len(k),m,lo,hi

def matched(m,vectors):
    mma=[];mms=[];seed=SEED+9000
    for keys,g in m.groupby(["dataset","query_model","horizon"]):
        main=g[(g.method=="Full CESS (frozen original)")]
        if main.empty:main=g[(g.method=="Full CESS")&(g["lambda"].eq(.5))]
        if main.empty:continue
        c=main.iloc[0];ctag=tuple(c[x] for x in ["dataset","query_model","horizon","method","family","lambda","is_main","note"])
        for fam in ("ordinary shrinkage: global","ordinary shrinkage: OR"):
            q=g[g.family==fam]
            if q.empty:continue
            for match,objective,output in (("matched_mae","mae",mma),("matched_sensitivity","ranking_sensitivity",mms)):
                r=q.iloc[(q[objective]-c[objective]).abs().argmin()];rtag=tuple(r[x] for x in ["dataset","query_model","horizon","method","family","lambda","is_main","note"])
                compare="ranking_sensitivity" if match=="matched_mae" else "mae";n,d,lo,hi=paired_boot(vectors[ctag][compare],vectors[rtag][compare],seed);seed+=1
                output.append({"dataset":keys[0],"query_model":keys[1],"horizon":keys[2],"cess_method":c.method,"baseline_family":fam,"baseline_lambda":r["lambda"],"matched_metric":objective,"cess_matched_value":c[objective],"baseline_matched_value":r[objective],"comparison_metric":compare,"cess_minus_baseline":d,"ci95_low":lo,"ci95_high":hi,"source_tasks":n,"significantly_lower":hi<0})
    return pd.DataFrame(mma),pd.DataFrame(mms)

def load_ms2():
    outs=[];gm=float(pd.read_csv(ROOT/"analysis/submission_final_checks_v3/development_tuning.csv").development_global_target_mean.iloc[0])
    data=ROOT/"evidence_selection_benchmark/data/ms2_sequential_global_shrinkage_confirmatory_v2"
    y={str(x["document_id"]):float(x["direction_score"]) for x in readj(data/"private_document_outcomes.jsonl")};nd=defaultdict(int)
    for x in readj(data/"documents.jsonl"):nd[str(x["review_id"])]+=1
    for model in ("qwen2_5_32b","olmo3_7b"):
        base=ROOT/f"evidence_selection_benchmark/results/ms2_external/sequential_global_shrinkage_confirmatory_v2_{model}";d=pd.DataFrame(readj(base/"cells.jsonl"));d=d[(d.controller=="uniform")&(d.stopping_regime=="fixed_horizon")].copy().reset_index(drop=True);d=d.rename(columns={"global_outcome_regression":"outcome_regression"});tr=readj(base/"traces.jsonl");tr=[x for x in tr if x["controller"]=="uniform" and x["stopping_regime"]=="fixed_horizon"]
        aux=trace_estimators(d,tr,y,nd,4,["episode_id"]);d=pd.concat([d,aux],axis=1);d["ess"]=aux.ess;d["cess_no_stop_raw"]=d.sequential_dr
        outs.append(add_estimates(d,"MS2",model,4,gm,d.preoutcome_global_shrinkage,d.preoutcome_global_cess))
    return pd.concat(outs,ignore_index=True)

def load_perspectrum():
    dev=pd.DataFrame(readj(ROOT/"evidence_selection_benchmark/results/perspectrum_nonmedical_v1/development_numeric_v1/cells.jsonl"));gm=float(dev.groupby("review_id").target.first().mean());tr=readj(ROOT/"evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1/traces.jsonl");y={str(x["document_id"]):float(x["direction_score"]) for x in readj(ROOT/"evidence_selection_benchmark/data/perspectrum_nonmedical_v1/holdout/private_document_outcomes.jsonl")};nd=defaultdict(int)
    for x in readj(ROOT/"evidence_selection_benchmark/data/perspectrum_nonmedical_v1/holdout/documents.jsonl"):nd[str(x["review_id"])]+=1
    outs=[]
    for h in (1,2,4,6,8,10):
        d=pd.DataFrame(readj(ROOT/f"analysis/horizon10_audit_v1/results/prefix_cells_round{h}.jsonl"));d=d.reset_index(drop=True)
        for model,g in d.groupby("model_key"):
            g=g.copy().reset_index(drop=True);aux=trace_estimators(g,tr,y,nd,h,["model_key","review_id","rank_condition","stopping_regime","controller","replicate"]);g=pd.concat([g,aux],axis=1);g["ess"]=aux.ess;g["cess_no_stop_raw"]=g.cess_no_stop
            outs.append(add_estimates(g,"PERSPECTRUM",str(model),h,gm,np.full(len(g),.5),g.cess_original))
    return pd.concat(outs,ignore_index=True)

def plot_dataset(m,ds,path):
    q=m[(m.dataset==ds)&(m.query_model==sorted(m[m.dataset==ds].query_model.unique())[0])].copy(); horizons=sorted(q.horizon.unique()); h=horizons[-1];q=q[q.horizon==h]
    families=["Opened Mean","ordinary shrinkage: global","ordinary shrinkage: OR","Sequential IPW","Sequential DR","Self-normalized IPW","Self-normalized DR","Weight-clipped DR","CESS no stopping","Full CESS"]
    colors=plt.cm.Set2(np.linspace(0,1,len(families)));fig,ax=plt.subplots(figsize=(8.2,5.2))
    for fam,c in zip(families,colors):
        z=q[q.family==fam].sort_values("lambda");
        if z.empty:continue
        ax.plot(z.ranking_sensitivity,z.mae,marker="o",markersize=4,linewidth=1.4,label=fam,color=c)
    ax.set_xlabel("Absolute paired ranking sensitivity ↓");ax.set_ylabel("Pool-target MAE ↓");ax.legend(frameon=False,fontsize=8,ncol=2);ax.grid(color="#E7E7E7",linewidth=.6);fig.tight_layout();fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight");fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight");plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);args=ap.parse_args();out=args.output_dir
    if not (out/"controlled").is_dir():raise FileNotFoundError("controlled analysis must finish first")
    (out/"figures").mkdir(exist_ok=True)
    rows=pd.concat([load_ms2(),load_perspectrum()],ignore_index=True);rows.to_csv(out/"real_trajectory_estimates.csv.gz",index=False,compression="gzip")
    metrics,vectors=summarize(rows);metrics.to_csv(out/"pareto_all_points.csv",index=False);fr=frontier(metrics);fr.to_csv(out/"pareto_frontier.csv",index=False);mma,mms=matched(metrics,vectors);mma.to_csv(out/"matched_mae.csv",index=False);mms.to_csv(out/"matched_sensitivity.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11,"pdf.fonttype":42});plot_dataset(metrics,"MS2",out/"figures/pareto_ms2");plot_dataset(metrics,"PERSPECTRUM",out/"figures/pareto_perspectrum")
    # Copy controlled required outputs into the clean top-level delivery.
    for name in ("propensity_perturbation.csv","continuation_perturbation.csv","outcome_model_robustness.csv","propensity_x_or_matrix.csv","zero_support_partial_identification.csv","epsilon_probability_mapping.csv"):
        shutil.copy2(out/"controlled"/name,out/name)
    for name in ("propensity_error_vs_mae.pdf","propensity_error_vs_mae.png","propensity_x_or_heatmap.pdf","propensity_x_or_heatmap.png"):
        shutil.copy2(out/"controlled/figures"/name,out/"figures"/name)
    # Analyze already-completed overlap stress trajectories without changing them.
    old=pd.read_csv(ROOT/"outputs/analysis_v6_pareto_robustness/overlap_stress_test.csv");emap={"overlap_eps000_deterministic":0.,"overlap_eps001":.01,"overlap_eps002":.02,"overlap_eps005":.05,"overlap_eps010":.10,"overlap_eps020":.20};old["epsilon"]=old.scenario.map(emap);old.to_csv(out/"overlap_stress_test.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for method,g in old[old.epsilon>0].groupby("method"):
        g=g.sort_values("epsilon");axes[0].plot(g.epsilon,g.task_macro_mae,marker="o",label=method);axes[1].plot(g.epsilon,g.mean_ess,marker="o",label=method)
    axes[0].set(xlabel="Uniform exploration epsilon",ylabel="Task-macro MAE");axes[1].set(xlabel="Uniform exploration epsilon",ylabel="Mean trajectory ESS");axes[0].legend(frameon=False);axes[1].legend(frameon=False);fig.savefig(out/"figures/overlap_vs_mae_ess.pdf",bbox_inches="tight");fig.savefig(out/"figures/overlap_vs_mae_ess.png",dpi=300,bbox_inches="tight");plt.close(fig)
    controlled=pd.read_csv(out/"propensity_x_or_matrix.csv");real_long=metrics.copy();real_long["analysis_family"]="real_trajectory_pareto";controlled["analysis_family"]="controlled_robustness";overlap_long=old.copy();overlap_long["analysis_family"]="controlled_overlap";pd.concat([real_long,controlled,overlap_long],ignore_index=True,sort=False).to_csv(out/"all_metrics_long.csv",index=False)
    wide=metrics.pivot_table(index=["dataset","query_model","horizon","method","lambda"],values=["mae","ranking_sensitivity","bias","wrong_direction_rate","mean_ess"],aggfunc="first").reset_index();wide.to_csv(out/"all_metrics_wide.csv",index=False)
    dominance=[]
    for keys,g in metrics.groupby(["dataset","query_model","horizon"]):
        c=g[(g.method=="Full CESS (frozen original)")];c=c if not c.empty else g[(g.method=="Full CESS")&(g["lambda"].eq(.5))]
        if c.empty:continue
        c=c.iloc[0]
        for fam in ("ordinary shrinkage: global","ordinary shrinkage: OR"):
            q=g[g.family==fam];dominance.append({"dataset":keys[0],"query_model":keys[1],"horizon":keys[2],"baseline_family":fam,"cess_strictly_dominates_any_grid_point":bool(((c.mae<q.mae)&(c.ranking_sensitivity<q.ranking_sensitivity)).any()),"ordinary_shrinkage_strictly_dominates_cess":bool(((q.mae<c.mae)&(q.ranking_sensitivity<c.ranking_sensitivity)).any())})
    pd.DataFrame(dominance).to_csv(out/"dominance_audit.csv",index=False)
    z=pd.read_csv(out/"zero_support_partial_identification.csv");summary={"matched_mae_significant_fraction":float(mma.significantly_lower.mean()),"matched_sensitivity_significant_fraction":float(mms.significantly_lower.mean()),"dominance":dominance,"zero_support":{"mean_interval_width":float(z.interval_width.mean()),"coverage":float(z.true_target_covered.mean()),"sign_identifiable_rate":float(z.sign_identifiable.mean()),"mean_inaccessible_fraction":float(z.inaccessible_fraction.mean())}}
    (out/"analysis_summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:x.item() if hasattr(x,"item") else str(x))+"\n")
    definitions="""# Estimator definitions\n\nAll targets are finite candidate-pool means. Opened Mean averages observed outcomes. Outcome Regression (OR) averages document-level predictions over the full pool. Sequential IPW and DR use the logged chosen-document and continuation probabilities. Self-normalized estimators divide by the realized sum of inverse-probability weights; they are stabilized finite-sample estimators, not identical unbiased estimators. Weight-clipped DR caps each inverse-probability factor at 5, 10, or 20. Full CESS clips raw DR to [-1,1] and then shrinks it toward OR. The no-stopping ablation removes continuation-probability information only. Simple shrinkage uses clip(o + lambda(a-o), -1, 1). Ranking sensitivity first averages replicates within task/rank/context, takes the absolute opposing-rank difference, averages contexts within task, then averages source tasks.\n\nControlled propensity perturbations are evaluated only in the simulator because PERSPECTRUM real logs omit the full candidate probability vector. Temperature rescales logged policy logits, Gaussian perturbations add deterministic seeded logit noise, estimated propensity uses only candidate features visible before the current selection, and poor overlap sharpens the policy distribution.\n""";(out/"estimator_definitions.md").write_text(definitions)
    report=f"""# CESS analysis v6: final CPU analysis\n\n## What is newly supported\n\nThe real-trajectory Pareto files report {metrics.shape[0]} prespecified estimator points across MS2 and PERSPECTRUM, all available horizons, and all saved query models. At matched MAE, CESS has significantly lower ranking sensitivity in {int(mma.significantly_lower.sum())}/{len(mma)} comparisons. At matched ranking sensitivity, it has significantly lower MAE in {int(mms.significantly_lower.sum())}/{len(mms)} comparisons. These are paired source-task bootstrap comparisons; the complete null and negative results are retained.\n\nThe controlled propensity-by-outcome-model matrix separates four cases: correct/correct, correct/misspecified, misspecified/correct, and misspecified/misspecified. The results are simulator evidence about estimator robustness, not claims about unlogged real-agent propensities.\n\nAt deterministic selection, the mean partial-identification interval width is {summary['zero_support']['mean_interval_width']:.3f}; true target coverage is {summary['zero_support']['coverage']:.3f}, sign-identifiable rate is {summary['zero_support']['sign_identifiable_rate']:.3f}, and mean inaccessible candidate fraction is {summary['zero_support']['mean_inaccessible_fraction']:.3f}. This replaces the invalid practice of adding epsilon and calling zero-support targets identified.\n\n## What is not supported\n\nCESS does not uniformly dominate ordinary shrinkage. `dominance_audit.csv` reports every dataset/model/horizon result, including regions where ordinary shrinkage dominates or neither method dominates. A lower ranking sensitivity at matched error is a stability advantage, not unconditional MAE superiority.\n\n## What remains unidentifiable\n\nPERSPECTRUM real traces save only the selected document probability, not the full candidate probability vector. Therefore real-trajectory temperature, logit-noise, and alternative-propensity replay are not identifiable. `continuation_perturbation.csv` also records that the controlled robustness experiment fixes the horizon and does not identify real-agent stopping misspecification.\n\n## Claim guidance\n\nThe defensible claim is a conditional accuracy--ranking-stability trade-off: probability correction can improve stability in some matched-error regimes, while the gain depends on overlap, nuisance quality, horizon, and domain. The evidence does not support claiming universal Pareto dominance or generic double robustness under arbitrary misspecification.\n""";(out/"analysis_report.md").write_text(report)
    zh=f"""# v6 最终结论\n\n本分析只使用已有真实轨迹和受控模拟器 CPU 后处理，没有调用模型或修改论文。真实轨迹共形成 {metrics.shape[0]} 个预设方法点。相同 MAE 下，CESS 在 {int(mma.significantly_lower.sum())}/{len(mma)} 个配对比较中显著降低排序敏感性；相同排序敏感性下，在 {int(mms.significantly_lower.sum())}/{len(mms)} 个比较中显著降低 MAE。其余零结果和反向结果均保留。\n\n结论不是“CESS 全面支配普通收缩”。更准确的结论是：CESS 在部分数据域、模型和预算下改善准确性与排序稳定性的权衡，优势受重叠程度、结果模型质量和搜索预算影响。零支持时点估计不可识别，应报告部分识别区间。PERSPECTRUM 缺少完整候选概率向量，所以真实轨迹上的概率温度与噪声扰动不可识别；相应结果仅来自受控模拟器。\n""";(out/"executive_summary_zh.md").write_text(zh)
    inputs=[ROOT/"analysis/submission_final_checks_v3/method_metrics.csv",ROOT/"evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1/cells.jsonl",ROOT/"evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1/traces.jsonl"]
    manifest={"cpu_only":True,"gpu_runs":0,"model_calls":0,"latex_modified":False,"bootstrap":{"unit":"source task/review","repetitions_main_and_matched":BOOT,"repetitions_descriptive_grid":2000,"seed":SEED},"lambda_grid":[float(x) for x in GRID],"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"inputs":{str(p.relative_to(ROOT)):digest(p) for p in inputs},"limitations":["PERSPECTRUM lacks full candidate probability vectors","controlled continuation perturbation not identified under fixed horizon"]};(out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"output":str(out),"points":len(metrics),"matched_mae":len(mma),"matched_sensitivity":len(mms)},indent=2))

if __name__=="__main__":main()
