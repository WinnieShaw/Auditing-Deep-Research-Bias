#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOOT, SEED, TOL = 5000, 20260921, 0.05
BUDGETS = (1, 2, 4, 6, 8, 10)

def direction(x): return (x > TOL).astype(int) - (x < -TOL).astype(int)
def clip(x): return float(np.clip(x, -1, 1))
def key(r): return (r['model_key'], r['review_id'], r['rank_condition'], r['stopping_regime'], r['controller'], int(r['replicate']))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def tune(dev,a,b):
    y=dev.target.to_numpy(float); x=dev[a].to_numpy(float); z=dev[b].to_numpy(float)
    vals=[np.mean(np.abs(x+l*(z-x)-y)) for l in np.linspace(0,1,101)]
    i=int(np.argmin(vals)); return i/100,float(vals[i])

def load(inp):
    cells=pd.read_json(str(inp/'cells.jsonl'),lines=True)
    traces=[json.loads(x) for x in open(inp/'traces.jsonl') if x.strip()]
    data=ROOT/'evidence_selection_benchmark/data/perspectrum_nonmedical_v1/holdout'
    outcomes={r['document_id']:float(r['direction_score']) for r in map(json.loads,open(data/'private_document_outcomes.jsonl'))}
    targets={r['review_id']:float(r['target']) for r in map(json.loads,open(data/'private_targets.jsonl'))}
    sizes={}
    for r in map(json.loads,open(data/'documents.jsonl')): sizes[r['review_id']]=sizes.get(r['review_id'],0)+1
    return cells,traces,outcomes,targets,sizes

def estimate(obs,n,baseline,budget,mode):
    if not obs: return dict(naive=np.nan,or_=baseline,ipw=np.nan,dr=baseline,correction=np.nan,ess=np.nan,maxw=np.nan,minprob=np.nan,rounds=0,unique=0,repeat=np.nan)
    y=np.array([r['direction_score'] for r in obs],float); m=np.full(len(obs),baseline); p=np.array([r['selection_probability'] for r in obs],float); s=np.array([r['survival_probability'] for r in obs],float)
    surv=np.ones_like(s) if mode in ('no_stop','none') else s
    sel=np.full_like(p,1/n) if mode in ('no_select','none') else p
    w=1/(n*sel*surv); corr=float(np.sum(w*(y-m))/budget); ipw=float(np.sum(w*y)/budget); sw=float(w.sum())
    return dict(naive=float(y.mean()),or_=baseline,ipw=clip(ipw),dr=clip(baseline+corr),correction=corr,ess=sw*sw/float(np.square(w).sum()),maxw=float(w.max()),minprob=float(np.min(sel*surv)),rounds=float(len(obs)),unique=float(len(set(r['document_id'] for r in obs))),repeat=float(1-len(set(r['document_id'] for r in obs))/len(obs)))

def build(cells,traces,outcomes,sizes,lambdas,budget):
    grouped={}
    for r in traces:
        if (not r.get('stopped',False)) and r.get('document_id') is not None and int(r['round'])<=budget:
            q=dict(r); q['direction_score']=outcomes[r['document_id']]; grouped.setdefault(key(r),[]).append(q)
    devmean=float(pd.read_json(ROOT/'evidence_selection_benchmark/results/perspectrum_nonmedical_v1/development_numeric_v1/cells.jsonl',lines=True).target.mean())
    out=[]
    for c in cells.to_dict('records'):
        obs=grouped.get(key(c),[]); base=float(c['outcome_regression']); n=sizes[c['review_id']]
        full=estimate(obs,n,base,budget,'full'); ns=estimate(obs,n,base,budget,'no_stop'); np_=estimate(obs,n,base,budget,'no_select'); nb=estimate(obs,n,base,budget,'none')
        naive,or_=full['naive'],base
        out.append(dict(model_key=c['model_key'],review_id=c['review_id'],rank_condition=c['rank_condition'],stopping_regime=c['stopping_regime'],controller=c['controller'],replicate=int(c['replicate']),target=float(c['target']),budget=budget,naive_opened_mean=naive,outcome_regression=or_,sequential_ipw=full['ipw'],sequential_dr=full['dr'],simple_naive_or=clip(naive+lambdas['naive_or']*(or_-naive)),global_shrink=clip((1-lambdas['global_shrink'])*naive+lambdas['global_shrink']*devmean),cess_original=clip(or_+.5*(full['dr']-or_)),cess_tuned=clip(or_+lambdas['cess']*(full['dr']-or_)),cess_no_stop=clip(or_+.5*(ns['dr']-or_)),cess_no_select=clip(or_+.5*(np_['dr']-or_)),cess_no_both=clip(or_+.5*(nb['dr']-or_),),weight_ess=full['ess'],max_weight=full['maxw'],min_joint_probability=full['minprob'],rounds_observed=full['rounds'],unique_documents=full['unique'],repeat_rate=full['repeat']))
    return pd.DataFrame(out)

def metrics(f,m):
    x=f[m].to_numpy(float); y=f.target.to_numpy(float); ok=np.isfinite(x); x=x[ok]; y=y[ok]; q=f.loc[ok].copy()
    p=q.pivot_table(index=['model_key','review_id','stopping_regime','controller','replicate'],columns='rank_condition',values=m).dropna()
    d=p.pro_benefit-p.pro_harm
    g=q.assign(e=np.abs(q[m].to_numpy(float)-q.target.to_numpy(float))).groupby('review_id').e.mean().mean()
    sd=q.groupby(['model_key','review_id','rank_condition','stopping_regime','controller'])[m].std().dropna()
    return dict(episodes=len(x),mae=float(np.abs(x-y).mean()),rmse=float(np.sqrt(np.square(x-y).mean())),wrong_direction=float(np.mean(direction(x)!=direction(y))),signed_rank=float(d.mean()),abs_rank=float(np.abs(d).mean()),task_bias_mae=float(g),seed_sd=float(sd.mean()) if len(sd) else np.nan)

def boot(f,m,ref):
    z=f.assign(d=np.abs(f[m]-f.target)-np.abs(f[ref]-f.target)).groupby('review_id').d.mean().to_numpy(float); rng=np.random.default_rng(SEED+sum(map(ord,m))); v=z[rng.integers(0,len(z),(BOOT,len(z)))].mean(1)
    return dict(mean=float(z.mean()),ci95=[float(np.quantile(v,.025)),float(np.quantile(v,.975))])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=ROOT/'evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1'); ap.add_argument('--output',type=Path,default=ROOT/'analysis/horizon10_audit_v1/results'); a=ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()): raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True); cells,traces,outcomes,targets,sizes=load(a.input)
    dev=pd.read_json(str(ROOT/'evidence_selection_benchmark/results/perspectrum_nonmedical_v1/development_numeric_v1/cells.jsonl'),lines=True)
    l_or,_=tune(dev,'naive_opened_mean','outcome_regression'); l_c,_=tune(dev,'outcome_regression','sequential_dr'); mean=float(dev.target.mean()); best=min(((l,np.mean(np.abs((1-l)*dev.naive_opened_mean+l*mean-dev.target))) for l in np.linspace(0,1,101)),key=lambda z:z[1])[0]
    lambdas={'naive_or':l_or,'cess':l_c,'global_shrink':float(best),'original_cess':.5}
    frames=[]
    for b in BUDGETS:
        f=build(cells,traces,outcomes,sizes,lambdas,b); f.to_json(a.output/f'prefix_cells_round{b}.jsonl',orient='records',lines=True); frames.append(f)
    allf=pd.concat(frames,ignore_index=True); methods=['naive_opened_mean','outcome_regression','simple_naive_or','global_shrink','sequential_ipw','sequential_dr','cess_original','cess_tuned','cess_no_stop','cess_no_select','cess_no_both']
    rows=[]
    for b in BUDGETS:
        f=allf[allf.budget==b]
        for m in methods: rows.append(dict(budget=b,method=m,**metrics(f,m)))
    met=pd.DataFrame(rows); met.to_csv(a.output/'prefix_metrics.csv',index=False)
    diag=allf.groupby('budget').agg(mean_rounds=('rounds_observed','mean'),mean_unique=('unique_documents','mean'),mean_repeat_rate=('repeat_rate','mean'),median_ess=('weight_ess','median'),p05_ess=('weight_ess',lambda x:float(np.quantile(x,.05))),p95_max_weight=('max_weight',lambda x:float(np.quantile(x,.95))),min_joint_probability=('min_joint_probability','min'),episodes=('review_id','size')).reset_index(); diag.to_csv(a.output/'prefix_diagnostics.csv',index=False)
    bt={str(b):{'cess_minus_opened':boot(allf[allf.budget==b],'cess_original','naive_opened_mean'),'cess_minus_simple_naive_or':boot(allf[allf.budget==b],'cess_original','simple_naive_or')} for b in BUDGETS}
    audit=dict(protocol='perspectrum-query-horizon10-v1',input_hashes={x:sha(a.input/x) for x in ['cells.jsonl','traces.jsonl']},episodes=len(cells),trace_rows=len(traces),topics=int(cells.review_id.nunique()),models=sorted(cells.model_key.unique()),budgets=list(BUDGETS),tuning={**lambdas,'source':'development_numeric_v1','grid':'0.00..1.00 step 0.01'},estimator_definitions={'full':'w=1/(N*p_t*s_t)','no_stop':'w=1/(N*p_t)','no_select':'w=1/(N*s_t)','no_both':'w=1/N','prefix':'retain all episodes; round<=budget; denominator=budget'},metrics=met.to_dict('records'),diagnostics=diag.to_dict('records'),bootstrap=bt)
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2));
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11}); labels={'naive_opened_mean':'Opened mean','outcome_regression':'Outcome regression','simple_naive_or':'Naive–OR shrinkage','cess_original':'Original CESS'}; colors={'naive_opened_mean':'#7aa6c2','outcome_regression':'#e9a66c','simple_naive_or':'#84b98c','cess_original':'#b38acb'}; fig,ax=plt.subplots(1,3,figsize=(14,4.2)); small=met[met.method.isin(labels)]
    for m in labels:
        p=small[small.method==m]; ax[0].plot(p.budget,p.mae,'o-',label=labels[m],color=colors[m]); ax[1].plot(p.budget,p.abs_rank,'o-',label=labels[m],color=colors[m]); ax[2].plot(p.abs_rank,p.mae,'o-',label=labels[m],color=colors[m])
    ax[0].set(xlabel='Evaluation budget (rounds)',ylabel='MAE ↓',title='(a) Prefix accuracy'); ax[1].set(xlabel='Evaluation budget (rounds)',ylabel='Ranking sensitivity ↓',title='(b) Prefix ranking sensitivity'); ax[2].set(xlabel='Ranking sensitivity ↓',ylabel='MAE ↓',title='(c) Accuracy–stability trade-off'); ax[0].set_xticks(BUDGETS); ax[1].set_xticks(BUDGETS); ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig(a.output/'horizon10_prefix_tradeoff.pdf',bbox_inches='tight'); fig.savefig(a.output/'horizon10_prefix_tradeoff.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    report=['# PERSPECTRUM horizon-10 audit','',f'The run contains {len(cells)} episodes over {cells.review_id.nunique()} held-out claims and {len(traces)} trace rows.','', '## Implementation audit','', '- The ranking condition enters `rank_probabilities` on every round; it is not first-round-only.', '- Queries are generated by the local LLM from opened abstracts. Document selection is a separate TF–IDF plus visible-score softmax.', '- `p_t` is over the complete preconstructed document pool for the task, not an external exposure probability.', '- `s_t` is cumulative survival probability under the randomized stopping rule; it is not confidence. The LLM supplies only a stop recommendation, while 0.75/0.10 hazards and a seeded RNG make the stop decision.', '- The frozen query CESS is `clip(OR + 0.5*(DR-OR), -1, 1)`. The lock records a five-round diagnostic; this ten-round run is a frozen stress test.', '- There is no report-level probability or valid single-task confidence interval in these logs; no coverage or high-confidence-error claim is computed.', '- The 0.01 probability floor gives positive p_t inside this closed candidate pool, but does not establish positivity for open-web exposure.', '', '## Files','', '- `prefix_metrics.csv`: task-level aggregate metrics for budgets 1,2,4,6,8,10.', '- `prefix_diagnostics.csv`: observed rounds, unique documents, repeats, ESS, weight tails and support diagnostics.', '- `horizon10_prefix_tradeoff.pdf`: accuracy, ranking sensitivity and their joint curve.']
    (a.output/'audit.md').write_text('\n'.join(report)+'\n'); print('\n'.join(report)); print(met[(met.budget==10)&met.method.isin(['naive_opened_mean','outcome_regression','simple_naive_or','cess_original','cess_no_stop','cess_no_select','cess_no_both'])].to_string(index=False))
if __name__=='__main__': main()
