#!/usr/bin/env python3
"""Public-agent v3 paired-intervention pilot and gamma calibration.

This adapter imports the immutable v2 public-agent bridge, adds a persistent
exact-input cache, serializes the pre-intervention state, and supports the
revised pilot plus development-only ranking-strength calibration.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "scripts/run_public_agent_light_v2.py"
spec = importlib.util.spec_from_file_location("public_agent_light_v2_adapter", V2_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {V2_PATH}")
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.types import Command
from langchain_openai.chat_models.base import _convert_message_to_dict


PROTOCOL = "public-agent-light-v3"
MODEL_REVISION = hashlib.sha256((Path(v2.MODEL_PATH) / "config.json").read_bytes()).hexdigest()
GAMMAS = (1.0, 1.5, 2.0, 3.0, 4.0)
CALIBRATION_TASKS = 24
AGENT_CALIBRATION_TASKS = 8
MONTE_CARLO_DRAWS = 2000
SHRINKAGE = 0.5
K_VALUES = (1, 2, 4)
RANKINGS = v2.RANKINGS

CHECKPOINTS: dict[str, dict[str, Any]] = {}
SUPERVISOR_PLANS: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
CACHE_EVENTS: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
FIRST_QUERY_CAPTURES: dict[str, dict[str, Any]] = {}
CAPTURE_FIRST_QUERY_ONLY = False
VISIBLE_SCORES: dict[str, float] = {}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_id(context: Any) -> str:
    return f"{context.task['review_id']}__{context.ranking}__{context.seed}"


def parse_llm_string(llm_string: str) -> tuple[dict[str, Any], dict[str, Any]]:
    serialized: dict[str, Any] = {}
    invocation: dict[str, Any] = {}
    try:
        prefix, suffix = llm_string.split("---", 1)
        serialized = json.loads(prefix)
        invocation = dict(ast.literal_eval(suffix))
    except Exception:
        invocation = {"raw_llm_string": llm_string}
    return serialized, invocation


def normalized_request(prompt: str, llm_string: str) -> dict[str, Any]:
    serialized_model, invocation = parse_llm_string(llm_string)
    try:
        loaded = loads(prompt)
        messages = [_convert_message_to_dict(message) for message in loaded]
    except Exception:
        messages = [{"serialized_prompt": prompt}]
    system_prompt = [row.get("content") for row in messages if row.get("role") in {"system", "developer"}]
    tools = invocation.get("tools", [])
    request = {
        "model_path": v2.MODEL_PATH,
        "model_revision": MODEL_REVISION,
        "system_prompt": system_prompt,
        "messages": messages,
        "tool_schemas": tools,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": v2.MAX_OUTPUT_TOKENS,
        "request_seed": v2.SEED,
        "serialized_model_configuration": serialized_model,
        "remaining_invocation_parameters": {key: value for key, value in invocation.items() if key != "tools"},
    }
    # This token hash is used only to verify equality of complete canonical
    # requests across the paired branches; the server remains responsible for
    # applying the model's chat template.
    try:
        from transformers import AutoTokenizer
        tokenizer = PersistentExactInputCache.tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(v2.MODEL_PATH, local_files_only=True)
            PersistentExactInputCache.tokenizer = tokenizer
        request_text = canonical_json({"messages": messages, "tools": tools})
        token_ids = tokenizer.encode(request_text, add_special_tokens=False)
        request["canonical_request_token_ids_sha256"] = hashlib.sha256(
            np.asarray(token_ids, dtype=np.int64).tobytes()
        ).hexdigest()
        request["canonical_request_token_count"] = len(token_ids)
    except Exception as exc:
        request["canonical_request_tokenization_error"] = f"{type(exc).__name__}: {exc}"
    return request


class PersistentExactInputCache(BaseCache):
    """Disk cache keyed only by normalized model inputs and generation settings."""

    tokenizer: Any = None

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _material_and_key(self, prompt: str, llm_string: str) -> tuple[dict[str, Any], str]:
        material = normalized_request(prompt, llm_string)
        return material, sha256_json(material)

    def _record_event(self, key: str, hit: bool, material: dict[str, Any]) -> None:
        context = v2.ACTIVE
        if context is None:
            return
        CACHE_EVENTS[episode_id(context)].append({
            "cache_key": key,
            "hit": hit,
            "request_sha256": sha256_json(material),
            "canonical_request_token_ids_sha256": material.get("canonical_request_token_ids_sha256"),
            "tool_configuration_sha256": sha256_json(material.get("tool_schemas", [])),
        })

    def lookup(self, prompt: str, llm_string: str):
        material, key = self._material_and_key(prompt, llm_string)
        path = self.directory / f"{key}.json"
        if not path.exists():
            self.misses += 1
            self._record_event(key, False, material)
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["key_material"] != material or record["cache_key"] != key:
            raise AssertionError(f"Persistent cache key mismatch: {path}")
        self.hits += 1
        self._record_event(key, True, material)
        binary_path = self.directory / record["raw_generation_binary"]
        binary = binary_path.read_bytes()
        if hashlib.sha256(binary).hexdigest() != record["raw_generation_sha256"]:
            raise AssertionError(f"Cached generation digest mismatch: {binary_path}")
        return pickle.loads(binary)

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        material, key = self._material_and_key(prompt, llm_string)
        path = self.directory / f"{key}.json"
        binary = pickle.dumps(return_val, protocol=pickle.HIGHEST_PROTOCOL)
        binary_sha256 = hashlib.sha256(binary).hexdigest()
        parsed = []
        for generation in return_val:
            if hasattr(generation, "model_dump"):
                parsed.append(generation.model_dump(mode="json"))
            else:
                parsed.append({"repr": repr(generation)})
        record = {
            "cache_key": key,
            "key_material": material,
            "raw_generation_binary": f"{key}.pkl",
            "raw_generation_sha256": binary_sha256,
            "parsed_generation_structure": parsed,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_binary = (self.directory / existing["raw_generation_binary"]).read_bytes()
            if hashlib.sha256(existing_binary).hexdigest() != binary_sha256:
                raise AssertionError(f"Same exact model input produced a conflicting output: {key}")
            return
        binary_tmp = (self.directory / f"{key}.pkl.tmp")
        binary_tmp.write_bytes(binary)
        binary_tmp.replace(self.directory / f"{key}.pkl")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        self.writes += 1

    def clear(self, **kwargs: Any) -> None:
        raise RuntimeError("The v3 persistent cache is an immutable experiment artifact")

    def summary(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "entries": len(list(self.directory.glob("*.json"))),
        }


def current_checkpoint(context: Any, query: str, round_index: int, gumbel_seed: int, gumbels: np.ndarray) -> dict[str, Any]:
    eid = episode_id(context)
    last_request = CACHE_EVENTS[eid][-1] if CACHE_EVENTS[eid] else {}
    latest_history = context.pre_model_history[-1] if context.pre_model_history else {}
    plan = SUPERVISOR_PLANS[eid][-1] if SUPERVISOR_PLANS[eid] else {}
    state = {
        "review_id": context.task["review_id"],
        "seed": context.seed,
        "research_question": context.task["research_question"],
        "messages_before_first_selection": latest_history,
        "supervisor_plan": plan,
        "first_query": query,
        "model_request_cache_key": last_request.get("cache_key"),
        "canonical_request_token_ids_sha256": last_request.get("canonical_request_token_ids_sha256"),
        "tool_configuration_sha256": last_request.get("tool_configuration_sha256"),
        "candidate_ids": [row["document_id"] for row in context.documents],
        "candidate_order_sha256": hashlib.sha256(
            "\n".join(row["document_id"] for row in context.documents).encode("utf-8")
        ).hexdigest(),
        "gumbel_seed": int(gumbel_seed),
        "gumbel_sha256": hashlib.sha256(gumbels.tobytes()).hexdigest(),
        "search_tool_configuration": {
            "rounds": v2.ROUNDS,
            "beta": v2.BETA,
            "temperature": v2.TEMPERATURE,
            "epsilon": v2.EPSILON,
            "repetition_penalty": v2.REPETITION_PENALTY,
        },
    }
    state["checkpoint_sha256"] = sha256_json(state)
    return state


class FirstQueryCaptured(RuntimeError):
    pass


@tool("tavily_search", description="Search the frozen PERSPECTRUM candidate pool. Supply one adaptive query; one complete evidence passage is returned.")
async def v3_frozen_perspectrum_search(queries: list[str]) -> str:
    context = v2.ACTIVE
    if context is None:
        raise RuntimeError("No active episode context")
    if len(context.searches) >= v2.ROUNDS:
        raise RuntimeError("The fixed four-search budget has already been reached")
    normalized = [str(query).strip() for query in queries if str(query).strip()]
    if not normalized:
        raise ValueError("queries must contain a non-empty query")
    if len(normalized) != 1:
        context.routing_repairs.append({"kind": "multi_query_join", "search_round": len(context.searches) + 1, "original_query_count": len(normalized)})
    query = " ; ".join(normalized)
    round_index = len(context.searches) + 1
    relevance = v2.bm25_scores(context, query)
    evidence = np.asarray([context.outcomes[row["document_id"]] for row in context.documents], dtype=float)
    predictions = np.asarray([VISIBLE_SCORES[row["document_id"]] for row in context.documents], dtype=float)
    sign = 1.0 if context.ranking == "supporting_first" else -1.0
    logits = v2.BETA * relevance + sign * context.gamma * evidence - v2.REPETITION_PENALTY * context.opened_counts
    probabilities = (1.0 - v2.EPSILON) * v2.softmax(logits / v2.TEMPERATURE) + v2.EPSILON / len(logits)
    if abs(float(probabilities.sum()) - 1.0) >= 1e-12 or np.any(probabilities <= 0):
        raise AssertionError("invalid propensity vector")
    gumbel_seed = v2.stable_seed(PROTOCOL, context.task["review_id"], round_index, context.seed)
    generator = np.random.default_rng(gumbel_seed)
    uniform = np.clip(generator.random(len(logits)), 1e-15, 1.0 - 1e-15)
    gumbels = -np.log(-np.log(uniform))
    if round_index == 1:
        checkpoint = current_checkpoint(context, query, round_index, gumbel_seed, gumbels)
        CHECKPOINTS[episode_id(context)] = checkpoint
        if CAPTURE_FIRST_QUERY_ONLY:
            FIRST_QUERY_CAPTURES[context.task["review_id"]] = {
                "review_id": context.task["review_id"],
                "seed": context.seed,
                "query": query,
                "checkpoint": checkpoint,
                "relevance_standardized_clipped": relevance.tolist(),
                "candidate_ids": [row["document_id"] for row in context.documents],
                "candidate_outcomes": evidence.tolist(),
                "candidate_predictions": predictions.tolist(),
            }
            raise FirstQueryCaptured(context.task["review_id"])
    selected_index = int(np.argmax(np.log(probabilities) + gumbels))
    document = context.documents[selected_index]
    candidates = []
    for index, candidate in enumerate(context.documents):
        candidates.append({
            "document_id": candidate["document_id"],
            "relevance_standardized_clipped": float(relevance[index]),
            "private_evidence_score": float(evidence[index]),
            "outcome_prediction": float(predictions[index]),
            "previous_open_count": int(context.opened_counts[index]),
            "policy_logit": float(logits[index]),
            "selection_probability": float(probabilities[index]),
        })
    record = {
        "episode_id": episode_id(context),
        "review_id": context.task["review_id"],
        "ranking": context.ranking,
        "seed": context.seed,
        "gamma": context.gamma,
        "round": round_index,
        "query": query,
        "query_parts": normalized,
        "selected_document_id": document["document_id"],
        "selected_outcome": float(evidence[selected_index]),
        "selected_outcome_prediction": float(predictions[selected_index]),
        "selected_probability": float(probabilities[selected_index]),
        "probability_sum": float(probabilities.sum()),
        "minimum_probability": float(probabilities.min()),
        "gumbel_seed": int(gumbel_seed),
        "gumbel_sha256": hashlib.sha256(gumbels.tobytes()).hexdigest(),
        "candidate_order_sha256": hashlib.sha256("\n".join(row["document_id"] for row in context.documents).encode("utf-8")).hexdigest(),
        "candidates": candidates,
    }
    context.searches.append(record)
    context.opened_counts[selected_index] += 1
    return (
        f"--- SOURCE {round_index}: {document['title']} ---\n"
        f"URL: perspectrum://{document['document_id']}\n"
        f"DOCUMENT ID: {document['document_id']}\n\n"
        f"FULL EVIDENCE PASSAGE:\n{document['full_abstract']}"
    )


ORIGINAL_SUPERVISOR_TOOLS = v2.adapter_supervisor_tools


async def logged_supervisor_tools(state: Any, config: RunnableConfig):
    context = v2.ACTIVE
    if context is not None:
        messages = state.get("supervisor_messages", [])
        recent = messages[-1] if messages else None
        calls = list(getattr(recent, "tool_calls", []) or []) if recent is not None else []
        for call in calls:
            args = call.get("args", {})
            if call.get("name") == "ConductResearch" or (isinstance(args, dict) and "research_topic" in args):
                SUPERVISOR_PLANS[episode_id(context)].append({
                    "research_brief": state.get("research_brief", ""),
                    "research_topic": str(args.get("research_topic", state.get("research_brief", ""))),
                    "supervisor_messages": [v2.message_json(message) for message in messages],
                    "tool_call": call,
                })
    return await ORIGINAL_SUPERVISOR_TOOLS(state, config)


def install_adapter(cache: PersistentExactInputCache) -> None:
    v2.set_llm_cache(cache)
    v2.frozen_perspectrum_search = v3_frozen_perspectrum_search
    v2.adapter_supervisor_tools = logged_supervisor_tools


def load_visible_scores() -> dict[str, float]:
    path = ROOT / "evidence_selection_benchmark/features/perspectrum_nonmedical_v1/qwen2_5_32b_visible_scores.jsonl"
    return {str(row["document_id"]): float(row["visible_direction_score"]) for row in read_jsonl(path)}


def legal_development_tasks() -> tuple[list[str], Any]:
    tasks, documents, outcomes, targets = v2.load_dataset()
    legal = [task_id for task_id in sorted(tasks) if task_id in documents and task_id in targets and all(row["document_id"] in outcomes and row["document_id"] in VISIBLE_SCORES for row in documents[task_id])]
    return legal, (tasks, documents, outcomes, targets)


def episode_statistics(episode: dict[str, Any]) -> dict[str, Any]:
    searches = episode["searches"]
    pool_predictions = [candidate["outcome_prediction"] for candidate in searches[0]["candidates"]]
    anchor = float(np.mean(pool_predictions))
    n_pool = len(pool_predictions)
    output: dict[str, Any] = {"or_anchor": anchor, "pool_size": n_pool, "prefixes": {}}
    for k in K_VALUES:
        prefix = searches[:k]
        y = np.asarray([row["selected_outcome"] for row in prefix], dtype=float)
        m = np.asarray([row["selected_outcome_prediction"] for row in prefix], dtype=float)
        p = np.asarray([row["selected_probability"] for row in prefix], dtype=float)
        weights = 1.0 / (n_pool * p)
        correction = float(np.sum(weights * (y - m)) / k)
        raw_dr = float(anchor + correction)
        clipped_dr = float(np.clip(raw_dr, -1.0, 1.0))
        cess = float(np.clip(anchor + SHRINKAGE * (clipped_dr - anchor), -1.0, 1.0))
        ess = float(weights.sum() ** 2 / np.square(weights).sum())
        output["prefixes"][str(k)] = {
            "opened_mean": float(y.mean()),
            "raw_dr": raw_dr,
            "sequential_dr": clipped_dr,
            "cess": cess,
            "ess": ess,
            "maximum_weight": float(weights.max()),
        }
    return output


async def run_full_pair(task_id: str, dataset: Any, gamma: float, seed: int, base_url: str) -> list[dict[str, Any]]:
    tasks, documents, outcomes, targets = dataset
    results = []
    v2.SEED = seed
    for ranking in RANKINGS:
        print(json.dumps({"event": "episode_start", "task": task_id, "ranking": ranking, "gamma": gamma, "seed": seed}), flush=True)
        episode = await v2.run_episode(tasks[task_id], documents[task_id], outcomes, targets[task_id], ranking, gamma, base_url)
        checkpoint = CHECKPOINTS.get(episode["episode_id"], {})
        checkpoint["research_brief"] = episode.get("research_brief", "")
        checkpoint["checkpoint_with_brief_sha256"] = sha256_json(checkpoint)
        episode["pre_intervention_checkpoint"] = checkpoint
        episode["cache_events"] = CACHE_EVENTS.get(episode["episode_id"], [])
        if episode["search_count"] == v2.ROUNDS and not episode["error"]:
            episode["statistics"] = episode_statistics(episode)
        results.append(episode)
        print(json.dumps({"event": "episode_end", "task": task_id, "ranking": ranking, "gamma": gamma, "searches": episode["search_count"], "error": episode["error"]}), flush=True)
    return results


def core_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key not in {"checkpoint_sha256", "checkpoint_with_brief_sha256"}}


def offline_replay(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for episode in episodes:
        selection_ok = True
        propensity_ok = True
        for search in episode["searches"]:
            candidates = search["candidates"]
            relevance = np.asarray([row["relevance_standardized_clipped"] for row in candidates])
            evidence = np.asarray([row["private_evidence_score"] for row in candidates])
            counts = np.asarray([row["previous_open_count"] for row in candidates])
            sign = 1.0 if episode["ranking"] == "supporting_first" else -1.0
            logits = v2.BETA * relevance + sign * episode["gamma"] * evidence - v2.REPETITION_PENALTY * counts
            probabilities = (1.0 - v2.EPSILON) * v2.softmax(logits / v2.TEMPERATURE) + v2.EPSILON / len(logits)
            logged = np.asarray([row["selection_probability"] for row in candidates])
            propensity_ok = propensity_ok and np.array_equal(probabilities, logged)
            generator = np.random.default_rng(int(search["gumbel_seed"]))
            uniform = np.clip(generator.random(len(logits)), 1e-15, 1.0 - 1e-15)
            gumbels = -np.log(-np.log(uniform))
            chosen = int(np.argmax(np.log(probabilities) + gumbels))
            selection_ok = selection_ok and candidates[chosen]["document_id"] == search["selected_document_id"]
        if len(episode.get("searches", [])) == v2.ROUNDS and "statistics" in episode:
            statistics = episode_statistics(episode)
            statistics_ok = canonical_json(statistics) == canonical_json(episode["statistics"])
        else:
            statistics_ok = False
        rows.append({
            "episode_id": episode["episode_id"],
            "propensity_bitwise_replay": propensity_ok,
            "selection_bitwise_replay": selection_ok,
            "statistics_bitwise_replay": statistics_ok,
        })
    return {"episodes": rows, "passed": all(all(value for key, value in row.items() if key != "episode_id") for row in rows)}


def pilot_gate(episodes: list[dict[str, Any]], cache: PersistentExactInputCache) -> dict[str, Any]:
    base = v2.pilot_gate(episodes)["checks"]
    pairs: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for episode in episodes:
        pairs[episode["review_id"]][episode["ranking"]] = episode
    pre_pair = True
    for pair in pairs.values():
        if set(pair) != set(RANKINGS):
            pre_pair = False
            continue
        left = pair[RANKINGS[0]]["pre_intervention_checkpoint"]
        right = pair[RANKINGS[1]]["pre_intervention_checkpoint"]
        pre_pair = pre_pair and core_checkpoint(left) == core_checkpoint(right)
    replay = offline_replay(episodes)
    checks = {
        "public_agent_execution": all(base[key] for key in (
            "four_successful_searches_each", "queries_are_agent_generated_and_adaptive",
            "evidence_written_to_history", "vendor_final_synthesis_used", "same_pool_and_target",
        )),
        "pre_intervention_pairing": pre_pair,
        "propensity_normalization": base["probability_sums"],
        "strictly_positive_propensities": base["strictly_positive_probabilities"],
        "selected_probability_matches_vector": base["selected_probability_matches_vector"],
        "common_gumbel_noise": base["common_random_numbers"],
        "persistent_exact_input_cache_used": cache.writes > 0 and cache.hits > 0,
        "computational_replay": replay["passed"],
        "episodes_jsonl_self_contained": all("searches" in row and "statistics" in row and "pre_intervention_checkpoint" in row for row in episodes),
    }
    return {"checks": checks, "passed": all(checks.values()), "offline_replay": replay, "legacy_structural_checks": base}


def write_episode_bundle(directory: Path, episodes: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "episodes.jsonl", episodes)
    searches = [search for episode in episodes for search in episode["searches"]]
    write_jsonl(directory / "propensities.jsonl", searches)
    write_jsonl(directory / "queries.jsonl", [{key: search[key] for key in ("episode_id", "review_id", "ranking", "seed", "round", "query", "selected_document_id")} for search in searches])
    for episode in episodes:
        atomic_json(directory / "raw" / f"{episode['episode_id']}.json", episode)


async def run_pilot(args: argparse.Namespace, cache: PersistentExactInputCache, dataset: Any, legal: list[str]) -> None:
    directory = args.output / "pilot"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    episodes: list[dict[str, Any]] = []
    for task_id in legal[:2]:
        episodes.extend(await run_full_pair(task_id, dataset, args.gamma, args.seed, args.base_url))
    gate = pilot_gate(episodes, cache)
    write_episode_bundle(directory, episodes)
    atomic_json(directory / "pilot_gate.json", gate)
    atomic_json(directory / "cache_summary.json", cache.summary())
    report = [
        "# Public-agent v3 pilot gate", "", f"Overall: **{'PASS' if gate['passed'] else 'FAIL'}**", "",
        "The v3 protocol treats cross-process language-generation nondeterminism as a runtime diagnostic. The hard reproducibility target is exact cached-policy and computational replay.", "",
    ]
    report.extend(f"- {'PASS' if value else 'FAIL'} — `{key}`" for key, value in gate["checks"].items())
    report += ["", "- `PUBLIC_AGENT_EXECUTION_VALID`: " + ("YES" if gate["checks"]["public_agent_execution"] else "NO"), "- `BITWISE_CROSS_PROCESS_REPLAY`: NO (carried forward as a diagnostic from v2)", ""]
    (directory / "pilot_gate_report.md").write_text("\n".join(report), encoding="utf-8")
    if not gate["passed"]:
        raise RuntimeError("v3 pilot hard gate failed")


def screening_probabilities(capture: dict[str, Any], gamma: float, ranking: str) -> np.ndarray:
    relevance = np.asarray(capture["relevance_standardized_clipped"], dtype=float)
    evidence = np.asarray(capture["candidate_outcomes"], dtype=float)
    sign = 1.0 if ranking == "supporting_first" else -1.0
    logits = v2.BETA * relevance + sign * gamma * evidence
    return (1.0 - v2.EPSILON) * v2.softmax(logits / v2.TEMPERATURE) + v2.EPSILON / len(logits)


async def capture_shared_query(task_id: str, dataset: Any, seed: int, base_url: str) -> dict[str, Any]:
    global CAPTURE_FIRST_QUERY_ONLY
    tasks, documents, outcomes, targets = dataset
    v2.SEED = seed
    CAPTURE_FIRST_QUERY_ONLY = True
    try:
        await v2.run_episode(tasks[task_id], documents[task_id], outcomes, targets[task_id], "supporting_first", 1.0, base_url)
    finally:
        CAPTURE_FIRST_QUERY_ONLY = False
    if task_id not in FIRST_QUERY_CAPTURES:
        raise RuntimeError(f"Failed to capture first query for {task_id}")
    return FIRST_QUERY_CAPTURES[task_id]


async def run_probability_screening(args: argparse.Namespace, cache: PersistentExactInputCache, dataset: Any, legal: list[str]) -> list[float]:
    directory = args.output / "gamma_screening"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    captures = []
    for index, task_id in enumerate(legal[:CALIBRATION_TASKS], 1):
        print(json.dumps({"event": "first_query_capture", "task": task_id, "index": index, "total": CALIBRATION_TASKS}), flush=True)
        captures.append(await capture_shared_query(task_id, dataset, args.seed, args.base_url))
    write_jsonl(directory / "shared_first_queries.jsonl", captures)
    rows = []
    for capture in captures:
        y = np.asarray(capture["candidate_outcomes"], dtype=float)
        n = len(y)
        for gamma in GAMMAS:
            p_support = screening_probabilities(capture, gamma, "supporting_first")
            p_oppose = screening_probabilities(capture, gamma, "opposing_first")
            rng = np.random.default_rng(v2.stable_seed(PROTOCOL, "gamma-screen", capture["review_id"], args.seed, gamma))
            different = 0
            gaps = []
            for _ in range(MONTE_CARLO_DRAWS):
                uniform = np.clip(rng.random(n), 1e-15, 1.0 - 1e-15)
                gumbel = -np.log(-np.log(uniform))
                a = int(np.argmax(np.log(p_support) + gumbel))
                b = int(np.argmax(np.log(p_oppose) + gumbel))
                different += int(a != b)
                gaps.append(float(y[a] - y[b]))
            for ranking, probability in (("supporting_first", p_support), ("opposing_first", p_oppose)):
                rows.append({
                    "review_id": capture["review_id"], "gamma": gamma, "ranking": ranking,
                    "expected_selected_stance": float(np.dot(probability, y)),
                    "probability_distribution_ess": float(1.0 / np.square(probability).sum()),
                    "maximum_inverse_probability_weight": float(np.max(1.0 / (n * probability))),
                    "paired_expected_opened_mean_gap": float(np.dot(p_support, y) - np.dot(p_oppose, y)),
                    "monte_carlo_selected_stance_gap": float(np.mean(gaps)),
                    "different_document_probability": different / MONTE_CARLO_DRAWS,
                    "finite_weights": bool(np.all(np.isfinite(1.0 / (n * probability)))),
                    "draws": MONTE_CARLO_DRAWS,
                })
    write_jsonl(directory / "probability_screening_task_rows.jsonl", rows)
    summaries = []
    for gamma in GAMMAS:
        subset = [row for row in rows if row["gamma"] == gamma]
        per_task = {}
        for row in subset:
            per_task.setdefault(row["review_id"], row)
        ess = np.asarray([row["probability_distribution_ess"] for row in subset])
        gaps = np.asarray([row["paired_expected_opened_mean_gap"] for row in per_task.values()])
        diffs = np.asarray([row["different_document_probability"] for row in per_task.values()])
        summary = {
            "gamma": gamma,
            "tasks": len(per_task),
            "expected_opened_mean_gap": float(gaps.mean()),
            "different_document_probability": float(diffs.mean()),
            "median_ess": float(np.median(ess)),
            "fifth_percentile_ess": float(np.quantile(ess, 0.05)),
            "maximum_weight": float(max(row["maximum_inverse_probability_weight"] for row in subset)),
            "finite_weights_all_tasks": all(row["finite_weights"] for row in subset),
        }
        summary["legal"] = bool(
            0.35 <= summary["expected_opened_mean_gap"] <= 0.70
            and summary["different_document_probability"] >= 0.60
            and summary["median_ess"] >= 2.0
            and summary["fifth_percentile_ess"] >= 1.2
            and summary["finite_weights_all_tasks"]
        )
        summaries.append(summary)
    selected = [row["gamma"] for row in sorted((row for row in summaries if row["legal"]), key=lambda row: (abs(row["expected_opened_mean_gap"] - 0.50), row["gamma"]))[:2]]
    atomic_json(directory / "probability_screening_summary.json", {"summaries": summaries, "selected_for_agent_calibration": selected})
    atomic_json(directory / "cache_summary.json", cache.summary())
    return selected


def gamma_calibration_metrics(episodes: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    subset = [row for row in episodes if row["gamma"] == gamma]
    pairs: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in subset:
        pairs[row["review_id"]][row["ranking"]] = row
    sensitivities = []
    sequence_different = []
    ess = []
    for pair in pairs.values():
        if set(pair) != set(RANKINGS):
            continue
        left, right = pair[RANKINGS[0]], pair[RANKINGS[1]]
        sensitivities.append(abs(left["statistics"]["prefixes"]["4"]["opened_mean"] - right["statistics"]["prefixes"]["4"]["opened_mean"]))
        sequence_different.append(left["selected_document_ids"] != right["selected_document_ids"])
        ess.extend([left["statistics"]["prefixes"]["4"]["ess"], right["statistics"]["prefixes"]["4"]["ess"]])
    return {
        "gamma": gamma,
        "task_seed_pairs": len(sensitivities),
        "median_ess": float(np.median(ess)),
        "different_document_sequence_fraction": float(np.mean(sequence_different)),
        "opened_mean_ranking_sensitivity": float(np.mean(sensitivities)),
        "eligible": bool(np.median(ess) >= 2.0 and np.mean(sequence_different) >= 0.60),
    }


async def run_agent_calibration(args: argparse.Namespace, cache: PersistentExactInputCache, dataset: Any, legal: list[str], gammas: list[float]) -> float:
    directory = args.output / "gamma_calibration"
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    episodes: list[dict[str, Any]] = []
    for gamma in gammas:
        for task_id in legal[:AGENT_CALIBRATION_TASKS]:
            episodes.extend(await run_full_pair(task_id, dataset, gamma, args.seed, args.base_url))
    write_episode_bundle(directory, episodes)
    metrics = [gamma_calibration_metrics(episodes, gamma) for gamma in gammas]
    eligible = [row for row in metrics if row["eligible"]]
    if not eligible:
        atomic_json(directory / "gamma_calibration_summary.json", {"metrics": metrics, "selected_gamma": None})
        raise RuntimeError("No gamma passed the real-agent calibration requirements")
    selected = min(eligible, key=lambda row: (abs(row["opened_mean_ranking_sensitivity"] - 0.50), row["gamma"]))["gamma"]
    summary = {"metrics": metrics, "selected_gamma": selected, "selection_rule": "eligible median ESS >= 2 and sequence divergence >= 0.60; then Opened Mean ranking sensitivity nearest 0.50"}
    atomic_json(directory / "gamma_calibration_summary.json", summary)
    frozen = {
        "protocol": PROTOCOL,
        "selected_gamma": selected,
        "selection_data": "PERSPECTRUM development split only",
        "model": v2.MODEL_NAME,
        "model_path": v2.MODEL_PATH,
        "model_revision": MODEL_REVISION,
        "temperature": 0.0,
        "top_p": 1.0,
        "request_seed": args.seed,
        "search_rounds": 4,
        "primary_k": 2,
        "ranking": {"beta": v2.BETA, "gamma": selected, "temperature": v2.TEMPERATURE, "epsilon": v2.EPSILON, "repetition_penalty": v2.REPETITION_PENALTY},
        "test_trajectories_planned": 216,
        "test_launched": False,
    }
    frozen["configuration_sha256"] = sha256_json(frozen)
    atomic_json(args.output / "frozen_public_agent_config.json", frozen)
    atomic_json(directory / "cache_summary.json", cache.summary())
    return float(selected)


def write_protocol_revision(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    text = """# Public-agent v3 protocol revision

The v2 structural checks established that the public Agent executes four searches, consumes evidence through Agent state, logs normalized positive propensities, shares common Gumbel noise, and uses the vendor synthesis path. v3 therefore separates two reproducibility targets:

- `PUBLIC_AGENT_EXECUTION_VALID: YES` when the structural and paired-state gates pass.
- `BITWISE_CROSS_PROCESS_REPLAY: NO` records the observed vLLM runtime diagnostic and is not a stopping condition.

CESS requires correct logged probabilities conditional on each realized history. It does not require an independently restarted neural inference process to regenerate the identical natural-language history. v3 fixes the realized LLM policy through a persistent exact-input cache, requires identical intervention-preceding state, and makes cached-policy selection and statistics exactly replayable.

The public-agent experiment remains a frozen-candidate-pool evaluation rather than unrestricted-web validation.
"""
    (output / "protocol_revision.md").write_text(text, encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    global VISIBLE_SCORES
    args.output = args.output.resolve()
    write_protocol_revision(args.output)
    VISIBLE_SCORES = load_visible_scores()
    cache = PersistentExactInputCache(args.output / "llm_cache")
    install_adapter(cache)
    legal, dataset = legal_development_tasks()
    if len(legal) < CALIBRATION_TASKS:
        raise RuntimeError(f"Need {CALIBRATION_TASKS} legal development tasks, found {len(legal)}")
    if args.stage == "pilot":
        await run_pilot(args, cache, dataset, legal)
    elif args.stage == "gamma-screen":
        selected = await run_probability_screening(args, cache, dataset, legal)
        print(json.dumps({"selected_for_agent_calibration": selected}, indent=2), flush=True)
    elif args.stage == "gamma-calibrate":
        if not args.gammas:
            raise ValueError("--gammas is required for gamma-calibrate")
        selected = await run_agent_calibration(args, cache, dataset, legal, args.gammas)
        print(json.dumps({"selected_gamma": selected}, indent=2), flush=True)
    else:
        raise ValueError(args.stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/public_agent_light_v3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--stage", choices=("pilot", "gamma-screen", "gamma-calibrate"), required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gammas", type=float, nargs="*")
    parser.add_argument("--seed", type=int, default=v2.SEED)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
