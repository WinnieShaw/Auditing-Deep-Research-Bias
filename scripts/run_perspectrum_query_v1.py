#!/usr/bin/env python3
"""Run strict-visibility LLM-controlled trajectories on PERSPECTRUM claims."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl, write_jsonl
from scripts.build_ms2_external_manifest import sha256_file
from scripts.run_realeffect_llama_development_numeric_v2 import (
    draw_index, estimate_episode, rank_probabilities, seeded_rng,
)
from scripts.run_realeffect_query_development_llama_v2 import generate, parse


SYSTEM_PROMPT = """You control evidence search for a deep-research agent.
Given a claim and only the evidence passages already opened, propose the next
concise search query. Seek evidence that could verify or overturn the current
conclusion. Return exactly:
STOP: yes or no
QUERY: <search query>"""


def compact(text: str, maximum: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= maximum else text[:maximum].rstrip() + " [...]"


def prompt(task: dict[str, Any], history: list[dict[str, Any]], config: dict[str, Any]) -> str:
    limit = int(config["query_agent"]["maximum_history_items"])
    maximum = int(config["query_agent"]["maximum_document_characters"])
    if history:
        evidence = "\n\n".join(
            f"Opened passage {index}: {compact(row['full_abstract'], maximum)}"
            for index, row in enumerate(history[-limit:], start=1)
        )
    else:
        evidence = "No evidence passage has been opened yet."
    return f"Claim: {task['claim']}\n\n{evidence}\n\nGive the next decision."


def task_text(task: dict[str, Any]) -> str:
    return " ".join(
        [task["claim"], " ".join(task.get("topics", [])), task.get("source", "")]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perspectrum_query_v1.yaml")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    if args.model_key not in config["models"]:
        raise ValueError(args.model_key)
    development = ROOT / config["development_data"]
    holdout = ROOT / config["holdout_data"]
    lock_path = ROOT / config["method_lock"]
    if sha256_file(lock_path) != config["method_lock_sha256"]:
        raise ValueError("method lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "frozen_before_query_evaluation":
        raise ValueError("method not frozen")

    development_tasks = {row["review_id"]: row for row in load_jsonl(development / "tasks.jsonl")}
    holdout_tasks = {row["review_id"]: row for row in load_jsonl(holdout / "tasks.jsonl")}
    if len(development_tasks) != int(config["split"]["expected_development_topics"]):
        raise ValueError("development topic count mismatch")
    if len(holdout_tasks) != int(config["split"]["expected_holdout_topics"]):
        raise ValueError("holdout topic count mismatch")
    development_documents = load_jsonl(development / "documents.jsonl")
    holdout_documents = load_jsonl(holdout / "documents.jsonl")
    all_ids = {row["document_id"] for row in development_documents + holdout_documents}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key, relative in config["visible_scores"].items():
        scores[model_key] = {
            row["document_id"]: row for row in load_jsonl(ROOT / relative)
            if row["document_id"] in all_ids
        }
        if set(scores[model_key]) != all_ids:
            raise ValueError(f"{model_key} score coverage mismatch")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=int(config["nuisance_model"]["tfidf_max_features"]),
        sublinear_tf=True,
    )
    development_task_rows = [development_tasks[key] for key in sorted(development_tasks)]
    holdout_task_rows = [holdout_tasks[key] for key in sorted(holdout_tasks)]
    development_design = vectorizer.fit_transform(
        [task_text(row) for row in development_task_rows]
    )
    holdout_design = vectorizer.transform([task_text(row) for row in holdout_task_rows])
    development_targets = {
        row["review_id"]: float(row["target"])
        for row in load_jsonl(development / "private_targets.jsonl")
    }
    nuisance = Ridge(alpha=float(lock["ridge_alpha"]))
    nuisance.fit(
        development_design,
        np.asarray([development_targets[row["review_id"]] for row in development_task_rows]),
    )
    task_predictions = np.clip(nuisance.predict(holdout_design), -1.0, 1.0)
    prediction_by_topic = {
        row["review_id"]: float(value)
        for row, value in zip(holdout_task_rows, task_predictions)
    }
    prediction_by_id = {
        row["document_id"]: prediction_by_topic[row["review_id"]]
        for row in holdout_documents
    }

    private_outcomes = {
        row["document_id"]: float(row["direction_score"])
        for row in load_jsonl(holdout / "private_document_outcomes.jsonl")
    }
    targets = {
        row["review_id"]: float(row["target"])
        for row in load_jsonl(holdout / "private_targets.jsonl")
    }
    development_summary = json.loads((ROOT / config["development_summary"]).read_text(encoding="utf-8"))
    tau_squared = float(development_summary["tau_squared"])
    residual_variance = float(development_summary["residual_variance"])

    selected_topics = [
        topic for index, topic in enumerate(sorted(holdout_tasks))
        if index % args.shard_count == args.shard_index
    ]
    documents_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in holdout_documents:
        if row["review_id"] in selected_topics:
            documents_by_topic[row["review_id"]].append(row)
    for rows in documents_by_topic.values():
        rows.sort(key=lambda row: row["document_id"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(config["models"][args.model_key]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    query_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    query_model.eval()

    retrieval = {}
    visible = {}
    for topic, rows in documents_by_topic.items():
        local_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = local_vectorizer.fit_transform([f"{row['title']} {row['full_abstract']}" for row in rows])
        retrieval[topic] = (local_vectorizer, matrix)
        visible[topic] = np.asarray([
            float(scores[args.model_key][row["document_id"]]["visible_direction_score"])
            for row in rows
        ])

    search = config["search"]
    episodes = []
    for topic in selected_topics:
        for rank_condition in search["rank_conditions"]:
            for stopping_regime in search["stopping_regimes"]:
                for controller in search["controllers"]:
                    for replicate in range(int(search["replicates"])):
                        episodes.append({
                            "review_id": topic, "rank_condition": rank_condition,
                            "stopping_regime": stopping_regime, "controller": controller,
                            "replicate": replicate, "history": [],
                            "opened_counts": np.zeros(len(documents_by_topic[topic])),
                            "survival_probability": 1.0, "finished": False, "parser_failures": 0,
                        })

    traces = []
    for round_index in range(1, int(search["maximum_rounds"]) + 1):
        active = [row for row in episodes if not row["finished"]]
        rendered = []
        for episode in active:
            content = prompt(holdout_tasks[episode["review_id"]], episode["history"], config)
            rendered.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        generated = generate(rendered, tokenizer, query_model, args.batch_size, config)
        for episode, raw in zip(active, generated):
            topic = episode["review_id"]
            query, recommend_stop, parsed = parse(raw, holdout_tasks[topic]["research_question"])
            episode["parser_failures"] += int(not parsed)
            stop_p = 0.0
            if episode["stopping_regime"] == "adaptive" and round_index > int(search["minimum_rounds_before_stop"]):
                stop_p = float(
                    search["stop_probability_if_recommend_stop"]
                    if recommend_stop else search["stop_probability_if_recommend_continue"]
                )
                stop_rng = seeded_rng(
                    config["seed"], "stop", args.model_key, topic,
                    episode["rank_condition"], episode["controller"], episode["replicate"], round_index,
                )
                if stop_rng.random() < stop_p:
                    episode["finished"] = True
                    traces.append({
                        "model_key": args.model_key, "review_id": topic,
                        "rank_condition": episode["rank_condition"], "stopping_regime": episode["stopping_regime"],
                        "controller": episode["controller"], "replicate": episode["replicate"],
                        "round": round_index, "query": query, "raw_model_output": raw,
                        "parsed": parsed, "recommend_stop": recommend_stop,
                        "stop_probability": stop_p, "stopped": True, "document_id": None,
                    })
                    continue
                episode["survival_probability"] *= 1.0 - stop_p
            rows = documents_by_topic[topic]
            local_vectorizer, matrix = retrieval[topic]
            probability = rank_probabilities(
                query, local_vectorizer, matrix, visible[topic], episode["opened_counts"],
                episode["rank_condition"], episode["controller"], config,
            )
            select_rng = seeded_rng(
                config["seed"], "select", args.model_key, topic,
                episode["rank_condition"], episode["stopping_regime"], episode["controller"],
                episode["replicate"], round_index,
            )
            index = draw_index(select_rng, probability)
            document = rows[index]
            observation = {
                "document_id": document["document_id"], "title": document["title"],
                "full_abstract": document["full_abstract"],
                "direction_score": private_outcomes[document["document_id"]],
                "prediction": prediction_by_id[document["document_id"]],
                "selection_probability": float(probability[index]),
                "survival_probability": float(episode["survival_probability"]),
            }
            episode["history"].append(observation)
            episode["opened_counts"][index] += 1.0
            traces.append({
                "model_key": args.model_key, "review_id": topic,
                "rank_condition": episode["rank_condition"], "stopping_regime": episode["stopping_regime"],
                "controller": episode["controller"], "replicate": episode["replicate"],
                "round": round_index, "query": query, "raw_model_output": raw,
                "parsed": parsed, "recommend_stop": recommend_stop,
                "stop_probability": stop_p, "stopped": False,
                "document_id": document["document_id"],
                "selection_probability": float(probability[index]),
                "survival_probability": float(episode["survival_probability"]),
            })
        print(json.dumps({"round": round_index, "active": len(active), "unique_prompts": len(set(rendered))}), flush=True)

    cells = []
    for episode in episodes:
        rows = documents_by_topic[episode["review_id"]]
        result = estimate_episode(
            episode["history"],
            np.asarray([prediction_by_id[row["document_id"]] for row in rows]),
            targets[episode["review_id"]], int(search["maximum_rounds"]),
            tau_squared, residual_variance, config,
        )
        result["risk_adaptive_cess"] = float(np.clip(
            result[lock["selected_anchor"]]
            + float(lock["anchor_shrinkage_lambda"])
            * (result["sequential_dr"] - result[lock["selected_anchor"]]),
            -1.0, 1.0,
        ))
        cells.append({
            "model_key": args.model_key, "review_id": episode["review_id"],
            "rank_condition": episode["rank_condition"], "stopping_regime": episode["stopping_regime"],
            "controller": episode["controller"], "replicate": episode["replicate"],
            "parser_failures": episode["parser_failures"], **result,
        })

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "cells.jsonl", cells)
    write_jsonl(args.output / "traces.jsonl", traces)
    summary = {
        "protocol_version": config["protocol_version"], "scope": config["scope"],
        "model_key": args.model_key, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "topics": len(selected_topics), "episodes": len(cells), "trace_rows": len(traces),
        "frozen_anchor": lock["selected_anchor"],
        "frozen_anchor_shrinkage": lock["anchor_shrinkage_lambda"],
        "unopened_evidence_text_used_by_nuisance_model": False,
        "query_model_path_read_only": str(model_path),
        "parameter_selection_on_holdout": False,
        "output_is_confirmatory": bool(config["integrity"]["output_is_confirmatory"]),
        "parse_rate": float(1.0 - sum(row["parser_failures"] for row in cells) / (len(traces) or 1)),
        "input_hashes": {
            "config": sha256_file(args.config), "method_lock": sha256_file(lock_path),
            "development_tasks": sha256_file(development / "tasks.jsonl"),
            "holdout_tasks": sha256_file(holdout / "tasks.jsonl"),
            "holdout_documents": sha256_file(holdout / "documents.jsonl"),
            "holdout_private_outcomes": sha256_file(holdout / "private_document_outcomes.jsonl"),
            "holdout_private_targets": sha256_file(holdout / "private_targets.jsonl"),
            "model_config": sha256_file(model_path / "config.json"),
            "cells": sha256_file(args.output / "cells.jsonl"),
            "traces": sha256_file(args.output / "traces.jsonl"),
        },
    }
    atomic_write_json(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
