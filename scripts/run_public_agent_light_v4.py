#!/usr/bin/env python3
"""v4 public-agent transfer: development-only gamma choice and frozen testing."""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / "scripts/run_public_agent_light_v3.py"
spec = importlib.util.spec_from_file_location("public_agent_light_v3_adapter", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {V3_PATH}")
v3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v3
spec.loader.exec_module(v3)

PROTOCOL = "public-agent-light-v4"
GAMMAS = (0.40, 0.45, 0.50, 0.55, 0.60)
STAGE_A_DRAWS = 10_000
K_VALUES = (1, 2, 4)
TEST_SEEDS = (2026092401, 2026092402, 2026092403)
BOOTSTRAP_REPLICATES = 10_000
LAMBDA_GRID = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 2))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class V4TieredCache(v3.PersistentExactInputCache):
    """Read v4 first, then v3 read-only; write newly generated responses to v4."""

    def __init__(self, directory: Path, legacy_directory: Path):
        super().__init__(directory)
        self.legacy_directory = legacy_directory
        self.legacy_hits = 0

    def _load_record(self, directory: Path, key: str, material: dict[str, Any]):
        path = directory / f"{key}.json"
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["cache_key"] != key or record["key_material"] != material:
            raise AssertionError(f"cache key material mismatch: {path}")
        payload = (directory / record["raw_generation_binary"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["raw_generation_sha256"]:
            raise AssertionError(f"cache digest mismatch: {path}")
        return pickle.loads(payload)

    def lookup(self, prompt: str, llm_string: str):
        material, key = self._material_and_key(prompt, llm_string)
        value = self._load_record(self.directory, key, material)
        if value is not None:
            self.hits += 1
            self._record_event(key, True, material)
            return value
        value = self._load_record(self.legacy_directory, key, material)
        if value is not None:
            self.hits += 1
            self.legacy_hits += 1
            self._record_event(key, True, material)
            return value
        self.misses += 1
        self._record_event(key, False, material)
        return None

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary.update({"legacy_read_only_directory": str(self.legacy_directory), "legacy_hits": self.legacy_hits})
        return summary


def load_split(split: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, float], dict[str, float]]:
    base = ROOT / "evidence_selection_benchmark/data/perspectrum_nonmedical_v1" / split
    # Some abstracts contain Unicode line separators.  ``str.splitlines``
    # mistakes those for JSONL delimiters; iterating the stream splits only on
    # physical file newlines, as the original v2 loader did.
    def source_read(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    tasks = {row["review_id"]: row for row in source_read(base / "tasks.jsonl")}
    documents: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in source_read(base / "documents.jsonl"):
        documents[row["review_id"]].append(row)
    outcomes = {row["document_id"]: float(row["direction_score"]) for row in source_read(base / "private_document_outcomes.jsonl")}
    targets = {row["review_id"]: float(row["target"]) for row in source_read(base / "private_targets.jsonl")}
    return tasks, documents, outcomes, targets


def legal_ids(dataset: tuple[dict, dict, dict, dict]) -> list[str]:
    tasks, documents, outcomes, targets = dataset
    return [task_id for task_id in sorted(tasks) if task_id in documents and task_id in targets and all(row["document_id"] in outcomes and row["document_id"] in v3.VISIBLE_SCORES for row in documents[task_id])]


def build_manifest(output: Path, development_ids: list[str], holdout_ids: list[str]) -> dict[str, Any]:
    stage_a = development_ids[24:48]
    stage_b = development_ids[48:60]
    prior = development_ids[:24]
    sets = [set(prior), set(stage_a), set(stage_b), set(holdout_ids)]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise AssertionError("development/test task overlap")
    manifest = {
        "v3_screening_task_ids": prior,
        "stage_a_task_ids": stage_a,
        "stage_b_task_ids": stage_b,
        "predetermined_test_task_ids": holdout_ids,
        "stage_a_count": len(stage_a), "stage_b_count": len(stage_b), "test_count": len(holdout_ids),
        "overlap_check": "pass",
    }
    atomic_json(output / "development_task_manifest.json", manifest)
    return manifest


def reset_pair_state(task_id: str, seed: int) -> None:
    for ranking in v3.RANKINGS:
        key = f"{task_id}__{ranking}__{seed}"
        v3.CHECKPOINTS.pop(key, None)
        v3.CACHE_EVENTS.pop(key, None)
        v3.SUPERVISOR_PLANS.pop(key, None)


async def run_pair(task_id: str, dataset: tuple, gamma: float, seed: int, base_url: str) -> list[dict[str, Any]]:
    tasks, documents, outcomes, targets = dataset
    v3.v2.SEED = seed
    reset_pair_state(task_id, seed)
    episodes: list[dict[str, Any]] = []
    for ranking in v3.RANKINGS:
        print(json.dumps({"event": "episode_start", "task": task_id, "ranking": ranking, "gamma": gamma, "seed": seed}), flush=True)
        episode = await v3.v2.run_episode(tasks[task_id], documents[task_id], outcomes, targets[task_id], ranking, gamma, base_url)
        checkpoint = dict(v3.CHECKPOINTS.get(episode["episode_id"], {}))
        checkpoint["research_brief"] = episode.get("research_brief", "")
        checkpoint["checkpoint_with_brief_sha256"] = sha256_json(checkpoint)
        episode["pre_intervention_checkpoint"] = checkpoint
        episode["cache_events"] = list(v3.CACHE_EVENTS.get(episode["episode_id"], []))
        episode["run_id"] = f"{episode['episode_id']}__gamma{gamma:.2f}"
        episodes.append(episode)
        print(json.dumps({"event": "episode_end", "task": task_id, "ranking": ranking, "gamma": gamma, "seed": seed, "searches": episode["search_count"], "error": episode["error"]}), flush=True)
    return episodes


def core_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key not in {"checkpoint_sha256", "checkpoint_with_brief_sha256"}}


def replay_selection(episode: dict[str, Any]) -> bool:
    if episode.get("error") or episode.get("search_count") != v3.v2.ROUNDS:
        return False
    for search in episode["searches"]:
        candidates = search["candidates"]
        relevance = np.asarray([row["relevance_standardized_clipped"] for row in candidates], float)
        evidence = np.asarray([row["private_evidence_score"] for row in candidates], float)
        counts = np.asarray([row["previous_open_count"] for row in candidates], float)
        sign = 1.0 if episode["ranking"] == "supporting_first" else -1.0
        logits = v3.v2.BETA * relevance + sign * episode["gamma"] * evidence - v3.v2.REPETITION_PENALTY * counts
        probabilities = (1.0 - v3.v2.EPSILON) * v3.v2.softmax(logits / v3.v2.TEMPERATURE) + v3.v2.EPSILON / len(logits)
        logged = np.asarray([row["selection_probability"] for row in candidates], float)
        if not np.array_equal(probabilities, logged):
            return False
        rng = np.random.default_rng(int(search["gumbel_seed"]))
        uniform = np.clip(rng.random(len(candidates)), 1e-15, 1 - 1e-15)
        selected = int(np.argmax(np.log(probabilities) - np.log(-np.log(uniform))))
        if candidates[selected]["document_id"] != search["selected_document_id"]:
            return False
    return True


def pair_checks(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[str, float, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for episode in episodes:
        pairs[(episode["review_id"], float(episode["gamma"]), int(episode["seed"]))][episode["ranking"]] = episode
    checkpoint_ok = True; gumbel_ok = True; propensity_ok = True; replay_ok = True
    for pair in pairs.values():
        if set(pair) != set(v3.RANKINGS):
            checkpoint_ok = gumbel_ok = propensity_ok = replay_ok = False
            continue
        left, right = pair[v3.RANKINGS[0]], pair[v3.RANKINGS[1]]
        checkpoint_ok &= core_checkpoint(left.get("pre_intervention_checkpoint", {})) == core_checkpoint(right.get("pre_intervention_checkpoint", {}))
        for lsearch, rsearch in zip(left.get("searches", []), right.get("searches", [])):
            gumbel_ok &= lsearch["gumbel_sha256"] == rsearch["gumbel_sha256"]
        for episode in (left, right):
            for search in episode.get("searches", []):
                propensity_ok &= abs(float(search["probability_sum"]) - 1.0) < 1e-12 and float(search["minimum_probability"]) > 0
                selected = next(row["selection_probability"] for row in search["candidates"] if row["document_id"] == search["selected_document_id"])
                propensity_ok &= abs(float(selected) - float(search["selected_probability"])) < 1e-12
            replay_ok &= replay_selection(episode)
    return {"paired_state": bool(checkpoint_ok), "common_gumbel": bool(gumbel_ok), "propensity": bool(propensity_ok), "computational_replay": bool(replay_ok), "pairs": len(pairs)}


def write_episodes(directory: Path, episodes: list[dict[str, Any]]) -> None:
    write_jsonl(directory / "episodes.jsonl", episodes)
    write_jsonl(directory / "queries.jsonl", [{"run_id": e["run_id"], "review_id": e["review_id"], "ranking": e["ranking"], "seed": e["seed"], "gamma": e["gamma"], "round": s["round"], "query": s["query"], "selected_document_id": s["selected_document_id"]} for e in episodes for s in e.get("searches", [])])
    write_jsonl(directory / "propensities.jsonl", [{**s, "run_id": e["run_id"]} for e in episodes for s in e.get("searches", [])])
    for episode in episodes:
        atomic_json(directory / "raw" / f"{episode['run_id']}.json", episode)


async def stage_a(output: Path, manifest: dict[str, Any], development: tuple, cache: V4TieredCache, base_url: str) -> list[float]:
    directory = output / "stage_a_probability_screening"
    if (directory / "summary.json").exists():
        raise FileExistsError("Stage A output already exists")
    captures = []
    v3.CAPTURE_FIRST_QUERY_ONLY = False
    for index, task_id in enumerate(manifest["stage_a_task_ids"], 1):
        v3.FIRST_QUERY_CAPTURES.pop(task_id, None)
        print(json.dumps({"event": "stage_a_first_query", "index": index, "total": len(manifest["stage_a_task_ids"]), "task": task_id}), flush=True)
        captures.append(await v3.capture_shared_query(task_id, development, TEST_SEEDS[0], base_url))
    write_jsonl(directory / "shared_first_queries.jsonl", captures)
    rows = []
    for capture in captures:
        y = np.asarray(capture["candidate_outcomes"], float); n = len(y)
        rng = np.random.default_rng(v3.v2.stable_seed(PROTOCOL, "stage_a", capture["review_id"], TEST_SEEDS[0]))
        uniform = np.clip(rng.random((STAGE_A_DRAWS, n)), 1e-15, 1 - 1e-15)
        gumbel = -np.log(-np.log(uniform))
        for gamma in GAMMAS:
            support = stage_a_probs(capture, gamma, "supporting_first")
            oppose = stage_a_probs(capture, gamma, "opposing_first")
            ia = np.argmax(np.log(support)[None, :] + gumbel, axis=1)
            ib = np.argmax(np.log(oppose)[None, :] + gumbel, axis=1)
            d1 = float(np.mean(ia != ib))
            for ranking, probability in (("supporting_first", support), ("opposing_first", oppose)):
                rows.append({
                    "review_id": capture["review_id"], "gamma": gamma, "ranking": ranking,
                    "expected_selected_stance": float(np.dot(probability, y)),
                    "expected_stance_gap": float(np.dot(support, y) - np.dot(oppose, y)),
                    "monte_carlo_stance_gap": float(np.mean(y[ia] - y[ib])),
                    "first_selection_document_divergence_d1": d1,
                    "k2_proxy_divergence_d2": float(1 - (1 - d1) ** 2),
                    "k4_proxy_divergence_d4": float(1 - (1 - d1) ** 4),
                    "minimum_propensity": float(probability.min()),
                    "maximum_uniform_target_importance_weight": float(np.max(1 / (n * probability))),
                    "effective_candidate_support": float(1 / np.square(probability).sum()),
                    "uniform_target_theoretical_is_ess_ratio": float(n * n / np.sum(1 / probability)),
                    "probability_sum_error": float(abs(probability.sum() - 1.0)), "draws": STAGE_A_DRAWS,
                })
    write_jsonl(directory / "task_gamma_rows.jsonl", rows)
    summaries=[]
    for gamma in GAMMAS:
        one=[row for row in rows if row["gamma"] == gamma and row["ranking"] == "supporting_first"]
        both=[row for row in rows if row["gamma"] == gamma]
        def desc(key: str) -> dict[str, float]:
            x=np.asarray([row[key] for row in one], float)
            return {"mean": float(x.mean()), "median": float(np.median(x)), "p05": float(np.quantile(x,.05)), "p95": float(np.quantile(x,.95))}
        item={"gamma": gamma, "tasks": len(one), **{key: desc(key) for key in ("expected_stance_gap","first_selection_document_divergence_d1","k2_proxy_divergence_d2","k4_proxy_divergence_d4")}}
        for key in ("minimum_propensity","maximum_uniform_target_importance_weight","effective_candidate_support","uniform_target_theoretical_is_ess_ratio","probability_sum_error"):
            x=np.asarray([row[key] for row in both],float); item[key]={"mean":float(x.mean()),"median":float(np.median(x)),"p05":float(np.quantile(x,.05)),"p95":float(np.quantile(x,.95))}
        item["legal"] = bool(0.40 <= item["expected_stance_gap"]["mean"] <= 0.70 and item["k2_proxy_divergence_d2"]["mean"] >= 0.40 and item["k4_proxy_divergence_d4"]["mean"] >= 0.65 and item["uniform_target_theoretical_is_ess_ratio"]["p05"] >= 0.35 and item["maximum_uniform_target_importance_weight"]["p95"] <= 5.0 + 1e-8 and item["minimum_propensity"]["p05"] > 0 and item["probability_sum_error"]["p95"] < 1e-12)
        item["selection_score"] = abs(item["expected_stance_gap"]["mean"] - .55) / .15 + abs(item["k2_proxy_divergence_d2"]["mean"] - .50) / .10
        summaries.append(item)
    selected=[row["gamma"] for row in sorted((row for row in summaries if row["legal"]), key=lambda row:(row["selection_score"],row["gamma"]))[:2]]
    atomic_json(directory / "summary.json", {"protocol": PROTOCOL, "draws": STAGE_A_DRAWS, "summaries": summaries, "selected_for_stage_b": selected, "cache": cache.summary()})
    if not selected:
        raise RuntimeError("Stage A has no legal gamma")
    return selected


def stage_a_probs(capture: dict[str, Any], gamma: float, ranking: str) -> np.ndarray:
    relevance=np.asarray(capture["relevance_standardized_clipped"],float); evidence=np.asarray(capture["candidate_outcomes"],float)
    sign=1.0 if ranking=="supporting_first" else -1.0
    logits=v3.v2.BETA*relevance+sign*gamma*evidence
    return (1-v3.v2.EPSILON)*v3.v2.softmax(logits/v3.v2.TEMPERATURE)+v3.v2.EPSILON/len(logits)


def selection_only_metrics(episodes: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    pairs: dict[tuple[str,int],dict[str,dict[str,Any]]] = collections.defaultdict(dict)
    for episode in episodes:
        if float(episode["gamma"]) == gamma: pairs[(episode["review_id"],int(episode["seed"]))][episode["ranking"]]=episode
    k2=[]; seq=[]; ess=[]; successes=[]
    for pair in pairs.values():
        if set(pair)!=set(v3.RANKINGS):
            successes.append(False); continue
        a,b=pair[v3.RANKINGS[0]],pair[v3.RANKINGS[1]]; ok=not a["error"] and not b["error"] and a["search_count"]==4 and b["search_count"]==4;successes.append(ok)
        if not ok: continue
        oa=np.mean([s["selected_outcome"] for s in a["searches"][:2]]); ob=np.mean([s["selected_outcome"] for s in b["searches"][:2]]); k2.append(abs(oa-ob));seq.append(a["selected_document_ids"]!=b["selected_document_ids"])
        for episode in (a,b):
            n=len(episode["searches"][0]["candidates"]);w=np.asarray([1/(n*s["selected_probability"]) for s in episode["searches"]],float);ess.append(float(w.sum()**2/np.square(w).sum()))
    checks=pair_checks(episodes)
    return {"gamma":gamma,"pairs":len(pairs),"episode_success_rate":float(np.mean(successes)) if successes else 0.0,"k2_opened_mean_ranking_sensitivity":float(np.mean(k2)) if k2 else np.nan,"k4_document_sequence_different_rate":float(np.mean(seq)) if seq else 0.0,"k4_median_realized_weight_ess":float(np.median(ess)) if ess else np.nan,"checks":checks}


async def stage_b(output: Path, manifest: dict[str, Any], development: tuple, cache: V4TieredCache, gammas: list[float], base_url: str) -> float:
    directory=output/"stage_b_agent_calibration"
    if (directory/"summary.json").exists(): raise FileExistsError("Stage B output already exists")
    episodes=[]
    for gamma in gammas:
        for index,task_id in enumerate(manifest["stage_b_task_ids"],1):
            print(json.dumps({"event":"stage_b_pair","gamma":gamma,"index":index,"total":len(manifest["stage_b_task_ids"]),"task":task_id}),flush=True)
            episodes.extend(await run_pair(task_id,development,gamma,TEST_SEEDS[0],base_url))
    write_episodes(directory,episodes)
    diagnostics=[selection_only_metrics(episodes,gamma) for gamma in gammas]
    for item in diagnostics:
        item["legal"]=bool(item["episode_success_rate"]==1.0 and .30<=item["k2_opened_mean_ranking_sensitivity"]<=.80 and item["k4_document_sequence_different_rate"]>=.60 and item["k4_median_realized_weight_ess"]>=2.0 and all(item["checks"][key] for key in ("paired_state","common_gumbel","propensity","computational_replay")))
    eligible=[x for x in diagnostics if x["legal"]]
    selected=None if not eligible else min(eligible,key=lambda x:(abs(x["k2_opened_mean_ranking_sensitivity"]-.50),-x["k4_median_realized_weight_ess"],x["gamma"]))["gamma"]
    atomic_json(directory/"summary.json",{"diagnostics":diagnostics,"selected_gamma":selected,"cache":cache.summary()})
    if selected is None: raise RuntimeError("Stage B has no legal gamma")
    return float(selected)


def freeze_config(output: Path, manifest: dict[str, Any], selected_gamma: float, development: tuple) -> dict[str, Any]:
    amendment=output/"protocol_amendment_v4.md"; amendment_hash=sha256_file(amendment)
    saved=(output/"protocol_amendment_v4.sha256").read_text(encoding="utf-8").split()[0]
    if amendment_hash != saved: raise RuntimeError("protocol amendment changed after freezing")
    _,_,_,targets=development
    global_mean=float(np.mean([targets[x] for x in manifest["stage_a_task_ids"]+manifest["stage_b_task_ids"]]))
    config={"protocol":PROTOCOL,"agent_commit":"1b7d2e80db9faa586165c60e09096dbbfd483a64","model":"Qwen2.5-32B-Instruct","model_path":v3.v2.MODEL_PATH,"model_config_sha256":v3.MODEL_REVISION,"dataset":"PERSPECTRUM frozen candidate pool","development_task_ids":manifest["stage_a_task_ids"]+manifest["stage_b_task_ids"],"test_task_ids":manifest["predetermined_test_task_ids"],"selected_gamma":selected_gamma,"ranking":{"beta":v3.v2.BETA,"epsilon":v3.v2.EPSILON,"temperature":v3.v2.TEMPERATURE,"repetition_penalty":v3.v2.REPETITION_PENALTY},"seeds":list(TEST_SEEDS),"k_values":list(K_VALUES),"primary_k":2,"estimators":{"or":"mean document-level Qwen visible-direction predictions over full candidate pool","development_global_mean":global_mean,"lambda_grid":list(map(float,LAMBDA_GRID)),"full_cess_lambda":.5,"cess_order":"clip raw sequential DR to [-1,1], then shrink toward OR","weight_clipped_dr_caps":[5,10,20],"continuation_probability":1.0},"bootstrap":{"type":"hierarchical paired task then within-task seed","replicates":BOOTSTRAP_REPLICATES,"seed":2026092404},"failure_policy":"All-task summaries retain every planned task/seed/ranking. Missing estimator values are reported as failures; successful-episode-only summaries are supplied separately without deleting failed rows.","protocol_amendment_sha256":amendment_hash,"code_sha256":sha256_file(Path(__file__)),"frozen_utc":datetime.now(timezone.utc).isoformat(),"test_launched":False}
    config["configuration_sha256"]=sha256_json(config);atomic_json(output/"frozen_test_config.json",config);return config


async def stage_test(output: Path, config: dict[str, Any], holdout: tuple, cache: V4TieredCache, base_url: str) -> None:
    directory=output/"test"
    if (directory/"episodes.jsonl").exists(): raise FileExistsError("test output already exists")
    amendment_hash=sha256_file(output/"protocol_amendment_v4.md")
    if amendment_hash != config["protocol_amendment_sha256"] or sha256_json({key:value for key,value in config.items() if key!="configuration_sha256"}) != config["configuration_sha256"]: raise RuntimeError("frozen config changed before test")
    # A partial log is an execution-resilience record, never an analysis input.
    # It makes an interrupted long test restart only missing pre-specified pairs
    # without changing the frozen task list, seeds, gamma, or model settings.
    partial=directory/"episodes.partial.jsonl"
    episodes=read_jsonl(partial) if partial.exists() else []
    completed={(e["review_id"],int(e["seed"])) for e in episodes if e.get("ranking") in v3.RANKINGS}
    # A completed unit must contain the two pre-specified rank branches.
    completed={key for key in completed if {e["ranking"] for e in episodes if (e["review_id"],int(e["seed"]))==key}==set(v3.RANKINGS)}
    episodes=[e for e in episodes if (e["review_id"],int(e["seed"])) in completed]
    total=len(config["test_task_ids"])*len(TEST_SEEDS)
    counter=0
    for seed in TEST_SEEDS:
        for task_id in config["test_task_ids"]:
            counter+=1
            if (task_id,seed) in completed:
                print(json.dumps({"event":"test_pair_reused_partial","index":counter,"total":total,"task":task_id,"seed":seed}),flush=True)
                continue
            print(json.dumps({"event":"test_pair","index":counter,"total":total,"task":task_id,"seed":seed}),flush=True)
            episodes.extend(await run_pair(task_id,holdout,float(config["selected_gamma"]),seed,base_url))
            write_jsonl(partial,episodes)
    write_episodes(directory,episodes);atomic_json(directory/"checks.json",pair_checks(episodes));atomic_json(directory/"cache_summary.json",cache.summary())


async def main_async(args: argparse.Namespace) -> None:
    output=args.output.resolve(); legacy=ROOT/"outputs/public_agent_light_v3/llm_cache"
    if not (output/"protocol_amendment_v4.md").exists(): raise FileNotFoundError("protocol amendment must be created and frozen before execution")
    v3.PROTOCOL=PROTOCOL; v3.VISIBLE_SCORES=v3.load_visible_scores(); cache=V4TieredCache(output/"llm_cache",legacy);v3.install_adapter(cache)
    development=load_split("development");holdout=load_split("holdout"); manifest_path=output/"development_task_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else build_manifest(output,legal_ids(development),legal_ids(holdout))
    if args.stage=="a":
        selected=await stage_a(output,manifest,development,cache,args.base_url);print(json.dumps({"selected_for_stage_b":selected},indent=2))
    elif args.stage=="b":
        summary=json.loads((output/"stage_a_probability_screening/summary.json").read_text());selected=await stage_b(output,manifest,development,cache,summary["selected_for_stage_b"],args.base_url);config=freeze_config(output,manifest,selected,development);print(json.dumps({"selected_gamma":selected,"frozen_config_sha256":config["configuration_sha256"]},indent=2))
    elif args.stage=="test":
        config=json.loads((output/"frozen_test_config.json").read_text());await stage_test(output,config,holdout,cache,args.base_url);print(json.dumps({"test_trajectories":216},indent=2))
    else: raise ValueError(args.stage)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("a","b","test"),required=True);parser.add_argument("--output",type=Path,default=ROOT/"outputs/public_agent_light_v4");parser.add_argument("--base-url",default="http://127.0.0.1:8000/v1");args=parser.parse_args();asyncio.run(main_async(args))


if __name__=="__main__": main()
