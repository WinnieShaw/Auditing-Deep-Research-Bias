#!/usr/bin/env python3
"""Export complete epsilon=0 candidate/support audit without model calls."""
from __future__ import annotations
import argparse,gzip,json
from pathlib import Path
import yaml
from run_causal_search_scm import generate_tasks,split_ranges
from v6_controlled_robustness_final import simulate

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=yaml.safe_load(args.config.read_text());det={"id":"epsilon0","kappa":4.,"rounds":4,"rho":0.,"stop_strength":0.,"force_full_horizon":True,"outcome_model":"correct","deterministic":True};count=0
    with gzip.open(args.output,"wt",encoding="utf-8") as f:
        for seed in cfg["seeds"]:
            ranges=split_ranges(cfg);tasks=generate_tasks(seed,max(x.stop for x in ranges.values()),int(cfg["candidate_documents"]),cfg)
            for i in ranges["test"]:
                task=tasks[i]
                for z in (-1,1):
                    obs=simulate(task,z,det,cfg,seed); selected_rounds={j:[] for j in range(len(task.y))}
                    for row in obs:selected_rounds[row["selected"]].append(row["round"])
                    for j,y in enumerate(task.y):
                        support=bool(selected_rounds[j]);f.write(json.dumps({"seed":seed,"task_id":task.task_id,"z":z,"candidate_id":j,"support_indicator":support,"selected_rounds":selected_rounds[j],"candidate_outcome":float(y),"target_mean":task.target,"observed_support_outcome":float(y) if support else None,"selection_probability":1.0 if support else 0.0})+"\n");count+=1
    print(json.dumps({"output":str(args.output),"records":count,"seeds":len(cfg["seeds"])},indent=2))
if __name__=="__main__":main()
