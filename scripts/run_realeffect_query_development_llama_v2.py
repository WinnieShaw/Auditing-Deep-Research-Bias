#!/usr/bin/env python3
"""Gate Llama-controlled search trajectories on RealEffect development topics."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl, write_jsonl
from scripts.build_ms2_external_manifest import sha256_file
from scripts.run_realeffect_llama_development_numeric_v2 import (
    build_design,
    draw_index,
    estimate_episode,
    leave_topic_out_predictions,
    rank_probabilities,
    seeded_rng,
)

SYSTEM_PROMPT = """You control literature search for a research synthesis agent.
Given the review question and only the studies already opened, decide whether
the evidence is sufficient and propose the next concise search query. Seek
evidence that could verify or overturn the current conclusion. Return exactly:
STOP: yes or no
QUERY: <search query>"""
STOP_RE = re.compile(r"STOP\s*:\s*(yes|no)", re.I)
QUERY_RE = re.compile(r"QUERY\s*:\s*(.+)", re.I)


def compact(text: str, maximum: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= maximum else text[:maximum].rstrip() + " [...]"


def prompt(task: dict[str, Any], history: list[dict[str, Any]], config: dict[str, Any]) -> str:
    limit = int(config["query_agent"]["maximum_history_items"])
    maximum = int(config["query_agent"]["maximum_document_characters"])
    if history:
        evidence = "\n\n".join(
            f"Opened study {index}: {row['title']}\nAbstract: {compact(row['full_abstract'], maximum)}"
            for index, row in enumerate(history[-limit:], start=1)
        )
    else:
        evidence = "No study has been opened yet."
    return f"Review question: {task['research_question']}\n\n{evidence}\n\nGive the next decision."


def parse(text: str, fallback: str) -> tuple[str, bool, bool]:
    stop_match = STOP_RE.search(text)
    query_match = QUERY_RE.search(text)
    query = " ".join(query_match.group(1).split()) if query_match else fallback
    return query or fallback, bool(stop_match and stop_match.group(1).lower() == "yes"), bool(stop_match and query_match)


def generate(prompts: list[str], tokenizer: Any, model: Any, batch_size: int, config: dict[str, Any]) -> list[str]:
    import torch

    unique = list(dict.fromkeys(prompts))
    outputs: dict[str, str] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(config["query_agent"]["max_input_tokens"]),
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(config["query_agent"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[:, encoded.input_ids.shape[1] :]
        outputs.update(zip(batch, tokenizer.batch_decode(continuation, skip_special_tokens=True)))
    return [outputs[value] for value in prompts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/realeffect_query_development_llama_v2.yaml")
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
    data = ROOT / config["data"]
    task_rows = load_jsonl(data / "tasks.jsonl")
    if len(task_rows) != int(config["split"]["expected_topics"]):
        raise ValueError("development topic count mismatch")
    if any(row["development_role"] != "development" for row in task_rows):
        raise ValueError("non-development task")
    all_tasks = {row["review_id"]: row for row in task_rows}
    all_documents = load_jsonl(data / "documents.jsonl")
    if any(row["development_role"] != "development" for row in all_documents):
        raise ValueError("non-development document")
    all_ids = {row["document_id"] for row in all_documents}
    private_outcomes = {row["document_id"]: float(row["direction_score"]) for row in load_jsonl(data / "private_document_outcomes.jsonl")}
    targets = {row["review_id"]: float(row["target"]) for row in load_jsonl(data / "private_targets.jsonl")}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key, relative in config["visible_scores"].items():
        scores[model_key] = {
            row["document_id"]: row
            for row in load_jsonl(ROOT / relative)
            if row["document_id"] in all_ids
        }
        if set(scores[model_key]) != all_ids:
            raise ValueError(f"{model_key} score coverage mismatch")

    design, _ = build_design(
        all_tasks, all_documents, scores, int(config["nuisance_model"]["tfidf_max_features"])
    )
    outcomes = np.asarray([private_outcomes[row["document_id"]] for row in all_documents])
    topic_ids = np.asarray([row["review_id"] for row in all_documents])
    predictions, selected_alpha, _ = leave_topic_out_predictions(
        design, outcomes, topic_ids, [float(value) for value in config["nuisance_model"]["ridge_alphas"]]
    )
    residual = outcomes - predictions
    residual_variance = float(np.mean(np.square(residual)))
    residual_means = [float(residual[topic_ids == topic].mean()) for topic in sorted(all_tasks)]
    tau_squared = float(np.var(residual_means, ddof=1))
    prediction_by_id = {row["document_id"]: float(value) for row, value in zip(all_documents, predictions)}

    selected_topics = [
        topic for index, topic in enumerate(sorted(all_tasks))
        if index % args.shard_count == args.shard_index
    ]
    documents_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_documents:
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    retrieval = {}
    visible = {}
    for topic, rows in documents_by_topic.items():
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = vectorizer.fit_transform([f"{row['title']} {row['full_abstract']}" for row in rows])
        retrieval[topic] = (vectorizer, matrix)
        visible[topic] = np.asarray(
            [float(scores[args.model_key][row["document_id"]]["visible_direction_score"]) for row in rows]
        )

    search = config["search"]
    episodes = []
    for topic in selected_topics:
        for rank_condition in search["rank_conditions"]:
            for stopping_regime in search["stopping_regimes"]:
                for controller in search["controllers"]:
                    for replicate in range(int(search["replicates"])):
                        episodes.append(
                            {
                                "review_id": topic,
                                "rank_condition": rank_condition,
                                "stopping_regime": stopping_regime,
                                "controller": controller,
                                "replicate": replicate,
                                "history": [],
                                "opened_counts": np.zeros(len(documents_by_topic[topic])),
                                "survival_probability": 1.0,
                                "finished": False,
                                "parser_failures": 0,
                            }
                        )

    traces = []
    for round_index in range(1, int(search["maximum_rounds"]) + 1):
        active = [row for row in episodes if not row["finished"]]
        rendered = []
        for episode in active:
            content = prompt(all_tasks[episode["review_id"]], episode["history"], config)
            rendered.append(
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            )
        generated = generate(rendered, tokenizer, model, args.batch_size, config)
        for episode, raw in zip(active, generated):
            topic = episode["review_id"]
            fallback = all_tasks[topic]["research_question"]
            query, recommend_stop, parsed = parse(raw, fallback)
            episode["parser_failures"] += int(not parsed)
            stop_p = 0.0
            if (
                episode["stopping_regime"] == "adaptive"
                and round_index > int(search["minimum_rounds_before_stop"])
            ):
                stop_p = float(
                    search["stop_probability_if_recommend_stop"]
                    if recommend_stop else search["stop_probability_if_recommend_continue"]
                )
                rng = seeded_rng(
                    config["seed"], "stop", args.model_key, topic,
                    episode["rank_condition"], episode["controller"], episode["replicate"], round_index,
                )
                if rng.random() < stop_p:
                    episode["finished"] = True
                    traces.append({
                        "review_id": topic, "rank_condition": episode["rank_condition"],
                        "stopping_regime": episode["stopping_regime"], "controller": episode["controller"],
                        "replicate": episode["replicate"], "round": round_index,
                        "query": query, "raw_model_output": raw, "parsed": parsed,
                        "recommend_stop": recommend_stop, "stop_probability": stop_p,
                        "stopped": True, "document_id": None,
                    })
                    continue
                episode["survival_probability"] *= 1.0 - stop_p
            rows = documents_by_topic[topic]
            vectorizer, matrix = retrieval[topic]
            probability = rank_probabilities(
                query, vectorizer, matrix, visible[topic], episode["opened_counts"],
                episode["rank_condition"], episode["controller"], config,
            )
            rng = seeded_rng(
                config["seed"], "select", args.model_key, topic,
                episode["rank_condition"], episode["stopping_regime"], episode["controller"],
                episode["replicate"], round_index,
            )
            selected_index = draw_index(rng, probability)
            document = rows[selected_index]
            observation = {
                "document_id": document["document_id"],
                "title": document["title"],
                "full_abstract": document["full_abstract"],
                "direction_score": private_outcomes[document["document_id"]],
                "prediction": prediction_by_id[document["document_id"]],
                "selection_probability": float(probability[selected_index]),
                "survival_probability": float(episode["survival_probability"]),
            }
            episode["history"].append(observation)
            episode["opened_counts"][selected_index] += 1.0
            traces.append({
                "review_id": topic, "rank_condition": episode["rank_condition"],
                "stopping_regime": episode["stopping_regime"], "controller": episode["controller"],
                "replicate": episode["replicate"], "round": round_index,
                "query": query, "raw_model_output": raw, "parsed": parsed,
                "recommend_stop": recommend_stop, "stop_probability": stop_p,
                "stopped": False, "document_id": document["document_id"],
                "selection_probability": float(probability[selected_index]),
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
        cells.append({
            "model_key": args.model_key, "review_id": episode["review_id"],
            "rank_condition": episode["rank_condition"],
            "stopping_regime": episode["stopping_regime"],
            "controller": episode["controller"], "replicate": episode["replicate"],
            "parser_failures": episode["parser_failures"], **result,
        })

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "traces.jsonl", traces)
    write_jsonl(args.output / "cells.jsonl", cells)
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": config["scope"], "model_key": args.model_key,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "topics": len(selected_topics), "episodes": len(cells), "trace_rows": len(traces),
        "selected_ridge_alpha": selected_alpha,
        "query_model_path_read_only": str(model_path),
        "query_prompt_sees_unopened_documents": False,
        "holdout_targets_loaded": False, "output_is_confirmatory": False,
        "parse_rate": float(1.0 - sum(row["parser_failures"] for row in cells) / (len(traces) or 1)),
        "input_hashes": {
            "config": sha256_file(args.config), "tasks": sha256_file(data / "tasks.jsonl"),
            "documents": sha256_file(data / "documents.jsonl"),
            "private_document_outcomes": sha256_file(data / "private_document_outcomes.jsonl"),
            "private_targets": sha256_file(data / "private_targets.jsonl"),
            "model_config": sha256_file(model_path / "config.json"),
            "cells": sha256_file(args.output / "cells.jsonl"),
            "traces": sha256_file(args.output / "traces.jsonl"),
        },
    }
    atomic_write_json(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
