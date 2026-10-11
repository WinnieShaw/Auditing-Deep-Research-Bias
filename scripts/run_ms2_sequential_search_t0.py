#!/usr/bin/env python3
"""Run actual LLM query, randomized ranking, reading, and stopping trajectories."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl, stable_hash, write_jsonl
from scripts.build_ms2_external_manifest import sha256_file
from scripts.run_ms2_program_development import fit_design, seeded_rng, standardized

DEFAULT_CONFIG = ROOT / "configs/ms2_sequential_search_t0_v1.yaml"
DEFAULT_ENVIRONMENT = ROOT / "evidence_selection_benchmark/data/ms2_sequential_search_t0_v1"
DEFAULT_DEVELOPMENT = ROOT / "evidence_selection_benchmark/data/ms2_external_v1/development"

SYSTEM_PROMPT = """You control literature search for a research synthesis agent.
Given the research question and only the evidence already opened, decide whether
the current evidence is sufficient and propose the next concise search query.
Seek evidence that could verify or overturn the current conclusion. Do not
invent study findings. Return exactly two lines:
QUERY: <search query>
STOP: yes or no"""

QUERY_PATTERN = re.compile(r"QUERY\s*:\s*(.+)", re.IGNORECASE)
STOP_PATTERN = re.compile(r"STOP\s*:\s*(yes|no)", re.IGNORECASE)


def compact_text(text: str, maximum: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= maximum:
        return normalized
    left = maximum // 2
    right = maximum - left
    return normalized[:left].rstrip() + " [...] " + normalized[-right:].lstrip()


def build_query_prompt(task: dict[str, Any], history: list[dict[str, Any]], config: dict[str, Any]) -> str:
    maximum_items = int(config["query_agent"]["maximum_history_items"])
    maximum_characters = int(config["query_agent"]["maximum_evidence_characters"])
    if history:
        evidence = []
        for index, item in enumerate(history[-maximum_items:], start=1):
            evidence.append(
                f"Opened study {index}: {item['title']}\n"
                f"Observed direction score: {item['direction_score']:+.3f}\n"
                f"Evidence: {compact_text(item['evidence_sentence'], maximum_characters)}"
            )
        history_text = "\n\n".join(evidence)
    else:
        history_text = "No evidence has been opened yet."
    return (
        f"Review question: {task['review_title']}\n"
        f"Intervention: {task['target_intervention']}\n"
        f"Outcome: {task['target_outcome']}\n\n"
        f"{history_text}\n\n"
        "Produce the next search decision."
    )


def parse_query_decision(text: str, fallback_query: str) -> tuple[str, bool, bool]:
    query_match = QUERY_PATTERN.search(text)
    stop_match = STOP_PATTERN.search(text)
    parsed = query_match is not None and stop_match is not None
    query = query_match.group(1).strip() if query_match else fallback_query
    query = " ".join(query.split())
    if not query:
        query = fallback_query
        parsed = False
    stop = stop_match.group(1).lower() == "yes" if stop_match else False
    return query, stop, parsed


def floored_softmax(logits: np.ndarray, temperature: float, floor: float) -> np.ndarray:
    values = np.asarray(logits, dtype=float) / temperature
    values = values - float(np.max(values))
    base = np.exp(values)
    base /= base.sum()
    if floor < 0.0 or floor * len(base) >= 1.0:
        raise ValueError("invalid minimum selection probability")
    probabilities = floor + (1.0 - floor * len(base)) * base
    probabilities /= probabilities.sum()
    return probabilities


def sample_categorical(rng: Any, probabilities: np.ndarray) -> int:
    """Draw an index using the stdlib ``random.Random`` API."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(values.sum())
    if not np.isclose(total, 1.0):
        raise ValueError("probabilities must sum to one")
    return int(rng.choices(range(len(values)), weights=values.tolist(), k=1)[0])


def rank_probabilities(
    query: str,
    vectorizer: TfidfVectorizer,
    document_matrix: Any,
    visible_scores: np.ndarray,
    opened_counts: np.ndarray,
    rank_condition: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    query_vector = vectorizer.transform([query])
    relevance = np.asarray((document_matrix @ query_vector.T).toarray()).ravel()
    relevance_z = standardized(relevance)
    direction_z = standardized(visible_scores)
    if rank_condition == "neutral":
        sign = 0.0
    elif rank_condition == "pro_increase":
        sign = 1.0
    elif rank_condition == "pro_decrease":
        sign = -1.0
    else:
        raise ValueError(f"unknown rank condition: {rank_condition}")
    search = config["search"]
    logits = (
        float(search["relevance_weight"]) * relevance_z
        + sign * float(search["direction_weight"]) * direction_z
        - float(search["duplicate_logit_penalty"]) * opened_counts
    )
    probabilities = floored_softmax(
        logits,
        float(search["selection_temperature"]),
        float(search["minimum_selection_probability"]),
    )
    return relevance, probabilities


def fit_development_only_nuisance(
    development_path: Path,
    environment_tasks: list[dict[str, Any]],
    environment_documents: list[dict[str, Any]],
    development_scores_path: Path,
    holdout_scores_path: Path,
    base_config: dict[str, Any],
) -> tuple[dict[str, float], float]:
    development_tasks = load_jsonl(development_path / "tasks.jsonl")
    development_documents_source = load_jsonl(development_path / "documents.jsonl")
    development_documents = []
    for source in development_documents_source:
        row = dict(source)
        row["development_role"] = "train"
        development_documents.append(row)
    environment_design_documents = []
    for source in environment_documents:
        row = dict(source)
        row["development_role"] = "holdout"
        environment_design_documents.append(row)
    tasks = {
        row["review_id"]: row
        for row in development_tasks + environment_tasks
    }
    development_scores = load_jsonl(development_scores_path)
    selected_ids = {row["document_id"] for row in environment_documents}
    environment_scores = [
        row for row in load_jsonl(holdout_scores_path)
        if row["document_id"] in selected_ids
    ]
    if {row["document_id"] for row in environment_scores} != selected_ids:
        raise ValueError("missing environment visible scores")
    scores = {
        row["document_id"]: row
        for row in development_scores + environment_scores
    }
    documents = development_documents + environment_design_documents
    _, design, train_mask, _ = fit_design(
        tasks,
        documents,
        scores,
        int(base_config["development_experiment"]["tfidf_max_features"]),
    )
    if int(train_mask.sum()) != len(development_documents):
        raise ValueError("nuisance design leaked environment rows into training")
    development_outcomes = np.asarray(
        [row["outcome"]["direction_score"] for row in development_documents],
        dtype=float,
    )
    outcomes_for_fit = np.concatenate(
        [development_outcomes, np.zeros(len(environment_design_documents))]
    )
    alpha = float(base_config["holdout_lock"]["point_estimator"]["ridge_alpha"])
    model = Ridge(alpha=alpha)
    model.fit(design[train_mask], outcomes_for_fit[train_mask])
    predictions = np.clip(model.predict(design), -1.0, 1.0)
    start = len(development_documents)
    prediction_by_id = {
        row["document_id"]: float(value)
        for row, value in zip(environment_documents, predictions[start:])
    }

    folds = int(base_config["development_experiment"]["outcome_crossfit_folds"])
    task_ids = sorted({row["review_id"] for row in development_documents})
    fold_by_task = {
        review_id: int(stable_hash(["sequential-t0-nuisance", review_id])[:8], 16)
        % folds
        for review_id in task_ids
    }
    crossfit = np.full(len(development_documents), np.nan)
    for fold in range(folds):
        fit_mask = np.asarray(
            [fold_by_task[row["review_id"]] != fold for row in development_documents]
        )
        valid_mask = ~fit_mask
        fold_model = Ridge(alpha=alpha)
        fold_model.fit(design[:start][fit_mask], development_outcomes[fit_mask])
        crossfit[valid_mask] = np.clip(
            fold_model.predict(design[:start][valid_mask]), -1.0, 1.0
        )
    if np.isnan(crossfit).any():
        raise ValueError("missing cross-fitted development nuisance predictions")
    residual_means = []
    for review_id in task_ids:
        indices = [
            index
            for index, row in enumerate(development_documents)
            if row["review_id"] == review_id
        ]
        residual_means.append(
            float(np.mean(development_outcomes[indices] - crossfit[indices]))
        )
    return prediction_by_id, float(np.var(residual_means, ddof=1))


def episode_estimators(
    observations: list[dict[str, Any]],
    predictions: np.ndarray,
    target: float,
    maximum_rounds: int,
    tau_squared: float,
) -> dict[str, Any]:
    n = len(predictions)
    global_prediction = float(np.mean(predictions))
    y = np.asarray([row["direction_score"] for row in observations], dtype=float)
    selected_predictions = np.asarray(
        [row["nuisance_prediction"] for row in observations], dtype=float
    )
    probability = np.asarray(
        [row["selection_probability"] for row in observations], dtype=float
    )
    survival = np.asarray(
        [row["survival_probability"] for row in observations], dtype=float
    )
    residual = y - selected_predictions
    correction_terms = residual / (n * probability * survival)
    outcome_terms = y / (n * probability * survival)
    sequential_dr_unclipped = global_prediction + float(
        correction_terms.sum() / maximum_rounds
    )
    sequential_ipw_unclipped = float(outcome_terms.sum() / maximum_rounds)
    uncorrected_dr = global_prediction + float(
        np.mean(residual / (n * probability))
    )
    sampling_variance = float(
        np.sum(np.square(correction_terms))
        / max(maximum_rounds * maximum_rounds, 1)
    )
    shrinkage = (
        tau_squared / (tau_squared + sampling_variance)
        if tau_squared + sampling_variance > 0.0
        else 0.0
    )
    cess_unclipped = global_prediction + shrinkage * (
        sequential_dr_unclipped - global_prediction
    )
    return {
        "target": target,
        "naive_opened_mean": float(np.mean(y)),
        "global_outcome_regression": global_prediction,
        "uncorrected_dr": float(np.clip(uncorrected_dr, -1.0, 1.0)),
        "sequential_ipw_unclipped": sequential_ipw_unclipped,
        "sequential_ipw": float(np.clip(sequential_ipw_unclipped, -1.0, 1.0)),
        "sequential_dr_unclipped": sequential_dr_unclipped,
        "sequential_dr": float(np.clip(sequential_dr_unclipped, -1.0, 1.0)),
        "cess_unclipped": cess_unclipped,
        "cess": float(np.clip(cess_unclipped, -1.0, 1.0)),
        "cess_shrinkage": shrinkage,
        "cess_sampling_variance": sampling_variance,
        "rounds_observed": len(observations),
        "unique_documents_opened": len(
            {row["document_id"] for row in observations}
        ),
        "minimum_selection_probability": float(np.min(probability)),
        "minimum_survival_probability": float(np.min(survival)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/ms2_external_v1.yaml")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"sequential-search T0 output exists: {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    base_config = load_config(args.base_config)
    model_config = config["models"][args.model_key]
    model_path = Path(model_config["model_path"])
    tasks = load_jsonl(args.environment / "tasks.jsonl")
    documents = load_jsonl(args.environment / "documents.jsonl")
    environment_summary = json.loads((args.environment / "summary.json").read_text())
    if environment_summary["protocol_version"] != config["protocol_version"]:
        raise ValueError("environment/config protocol mismatch")
    if any("outcome" in row for row in documents):
        raise ValueError("search environment contains outcome leakage")
    task_by_id = {row["review_id"]: row for row in tasks}
    documents_by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in documents:
        documents_by_review[row["review_id"]].append(row)
    for rows in documents_by_review.values():
        rows.sort(key=lambda row: row["document_id"])

    development_scores_path = ROOT / model_config["development_scores"]
    holdout_scores_path = ROOT / model_config["holdout_scores"]
    predictions, tau_squared = fit_development_only_nuisance(
        args.development,
        tasks,
        documents,
        development_scores_path,
        holdout_scores_path,
        base_config,
    )
    selected_ids = {row["document_id"] for row in documents}
    visible_by_id = {
        row["document_id"]: float(row["visible_direction_score"])
        for row in load_jsonl(holdout_scores_path)
        if row["document_id"] in selected_ids
    }
    if set(visible_by_id) != selected_ids or set(predictions) != selected_ids:
        raise ValueError("score/prediction coverage mismatch")

    private_outcomes = {
        row["document_id"]: row
        for row in load_jsonl(args.environment / "private_document_outcomes.jsonl")
    }
    targets = {
        row["review_id"]: float(row["target"])
        for row in load_jsonl(args.environment / "private_targets.jsonl")
    }
    if set(private_outcomes) != selected_ids:
        raise ValueError("private outcome/document mismatch")

    retrieval = {}
    for review_id, rows in documents_by_review.items():
        texts = [
            f"{row['title']} {row['full_abstract']}" for row in rows
        ]
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        retrieval[review_id] = (vectorizer, matrix)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    search = config["search"]
    episodes = []
    for review_id in sorted(task_by_id):
        for rank_condition in search["rank_conditions"]:
            for stopping_regime in search["stopping_regimes"]:
                for replicate in range(int(search["replicates"])):
                    episodes.append(
                        {
                            "episode_id": (
                                f"{args.model_key}:{review_id}:{rank_condition}:"
                                f"{stopping_regime}:{replicate}"
                            ),
                            "review_id": review_id,
                            "rank_condition": rank_condition,
                            "stopping_regime": stopping_regime,
                            "replicate": replicate,
                            "history": [],
                            "opened_counts": np.zeros(
                                len(documents_by_review[review_id]), dtype=float
                            ),
                            "survival_probability": 1.0,
                            "finished": False,
                            "parser_failures": 0,
                            "stop_recommendations": 0,
                            "stop_decisions": 0,
                        }
                    )

    traces = []
    maximum_rounds = int(search["maximum_rounds"])
    for round_index in range(1, maximum_rounds + 1):
        active = [episode for episode in episodes if not episode["finished"]]
        prompts = []
        for episode in active:
            task = task_by_id[episode["review_id"]]
            content = build_query_prompt(task, episode["history"], config)
            prompts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        generated_texts = []
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_prompts,
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
            generated_texts.extend(
                tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
        for episode, raw_text in zip(active, generated_texts):
            review_id = episode["review_id"]
            task = task_by_id[review_id]
            fallback = (
                f"{task['target_intervention']} {task['target_outcome']} "
                "systematic evidence"
            )
            query, recommend_stop, parsed = parse_query_decision(raw_text, fallback)
            episode["parser_failures"] += int(not parsed)
            episode["stop_recommendations"] += int(recommend_stop)
            stop_probability = 0.0
            stopped = False
            if (
                episode["stopping_regime"] == "adaptive"
                and round_index >= int(search["minimum_rounds_before_stop"])
            ):
                stop_probability = float(
                    search[
                        "stop_probability_if_recommend_stop"
                        if recommend_stop
                        else "stop_probability_if_recommend_continue"
                    ]
                )
                stop_rng = seeded_rng(
                    config["seed"],
                    "sequential-search-stop",
                    args.model_key,
                    review_id,
                    episode["replicate"],
                    round_index,
                )
                stopped = bool(stop_rng.random() < stop_probability)
                episode["stop_decisions"] += int(stopped)
                if not stopped:
                    episode["survival_probability"] *= 1.0 - stop_probability
            if stopped:
                episode["finished"] = True
                traces.append(
                    {
                        "episode_id": episode["episode_id"],
                        "review_id": review_id,
                        "rank_condition": episode["rank_condition"],
                        "stopping_regime": episode["stopping_regime"],
                        "replicate": episode["replicate"],
                        "round": round_index,
                        "query": query,
                        "raw_model_output": raw_text,
                        "parsed": parsed,
                        "recommend_stop": recommend_stop,
                        "stop_probability": stop_probability,
                        "stopped": True,
                        "survival_probability_if_continued": (
                            episode["survival_probability"] * (1.0 - stop_probability)
                        ),
                        "document_id": None,
                    }
                )
                continue

            rows = documents_by_review[review_id]
            vectorizer, document_matrix = retrieval[review_id]
            visible = np.asarray(
                [visible_by_id[row["document_id"]] for row in rows], dtype=float
            )
            relevance, probabilities = rank_probabilities(
                query,
                vectorizer,
                document_matrix,
                visible,
                episode["opened_counts"],
                episode["rank_condition"],
                config,
            )
            selection_rng = seeded_rng(
                config["seed"],
                "sequential-search-selection",
                args.model_key,
                review_id,
                episode["stopping_regime"],
                episode["replicate"],
                round_index,
            )
            selected_index = sample_categorical(selection_rng, probabilities)
            selected = rows[selected_index]
            document_id = selected["document_id"]
            outcome = private_outcomes[document_id]
            observation = {
                "round": round_index,
                "document_id": document_id,
                "title": selected["title"],
                "direction_score": float(outcome["direction_score"]),
                "evidence_sentence": outcome["evidence_sentence"],
                "nuisance_prediction": predictions[document_id],
                "selection_probability": float(probabilities[selected_index]),
                "survival_probability": float(episode["survival_probability"]),
            }
            episode["history"].append(observation)
            episode["opened_counts"][selected_index] += 1.0
            traces.append(
                {
                    "episode_id": episode["episode_id"],
                    "review_id": review_id,
                    "rank_condition": episode["rank_condition"],
                    "stopping_regime": episode["stopping_regime"],
                    "replicate": episode["replicate"],
                    "round": round_index,
                    "query": query,
                    "raw_model_output": raw_text,
                    "parsed": parsed,
                    "recommend_stop": recommend_stop,
                    "stop_probability": stop_probability,
                    "stopped": False,
                    "survival_probability": episode["survival_probability"],
                    "document_id": document_id,
                    "selection_probability": float(probabilities[selected_index]),
                    "candidate_document_ids": [row["document_id"] for row in rows],
                    "candidate_probabilities": list(map(float, probabilities)),
                    "candidate_relevance": list(map(float, relevance)),
                }
            )
        print(
            json.dumps(
                {
                    "round": round_index,
                    "active_before_decision": len(active),
                    "episodes_with_evidence": sum(
                        len(episode["history"]) >= round_index for episode in episodes
                    ),
                }
            ),
            flush=True,
        )

    cells = []
    for episode in episodes:
        review_id = episode["review_id"]
        rows = documents_by_review[review_id]
        prediction_values = np.asarray(
            [predictions[row["document_id"]] for row in rows], dtype=float
        )
        result = episode_estimators(
            episode["history"],
            prediction_values,
            targets[review_id],
            maximum_rounds,
            tau_squared,
        )
        result.update(
            {
                "episode_id": episode["episode_id"],
                "review_id": review_id,
                "model_key": args.model_key,
                "rank_condition": episode["rank_condition"],
                "stopping_regime": episode["stopping_regime"],
                "replicate": episode["replicate"],
                "parser_failures": episode["parser_failures"],
                "stop_recommendations": episode["stop_recommendations"],
                "stop_decisions": episode["stop_decisions"],
            }
        )
        cells.append(result)

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "traces.jsonl", traces)
    write_jsonl(args.output / "cells.jsonl", cells)
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": "postlock_sequential_search_t0",
        "model_key": args.model_key,
        "tasks": len(task_by_id),
        "episodes": len(cells),
        "trace_rows": len(traces),
        "tau_squared": tau_squared,
        "query_model_path_read_only": str(model_path),
        "environment_label_free": True,
        "retrieval_used_true_outcomes": False,
        "query_prompt_received_unopened_outcomes": False,
        "input_hashes": {
            "config": sha256_file(args.config),
            "environment_summary": sha256_file(args.environment / "summary.json"),
            "environment_tasks": sha256_file(args.environment / "tasks.jsonl"),
            "environment_documents": sha256_file(args.environment / "documents.jsonl"),
            "development_scores": sha256_file(development_scores_path),
            "holdout_scores": sha256_file(holdout_scores_path),
            "model_config": sha256_file(model_path / "config.json"),
            "traces": sha256_file(args.output / "traces.jsonl"),
            "cells": sha256_file(args.output / "cells.jsonl"),
        },
    }
    atomic_write_json(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
