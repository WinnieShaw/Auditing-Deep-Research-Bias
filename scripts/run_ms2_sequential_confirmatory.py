#!/usr/bin/env python3
"""Run the prospective sequential-search benchmark with frozen active CESS."""
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
from causal_search_credit.util import atomic_write_json, load_jsonl, stable_hash, write_jsonl
from scripts.build_ms2_external_manifest import sha256_file
from scripts.run_ms2_program_development import fit_design, seeded_rng
from scripts.run_ms2_sequential_search_t0 import (
    build_query_prompt,
    parse_query_decision,
    rank_probabilities,
    sample_categorical,
)

DEFAULT_CONFIG = ROOT / "configs/ms2_sequential_search_confirmatory_v1.yaml"
DEFAULT_ENVIRONMENT = (
    ROOT / "evidence_selection_benchmark/data/ms2_sequential_search_confirmatory_v1"
)
DEFAULT_DEVELOPMENT = ROOT / "evidence_selection_benchmark/data/ms2_external_v1/development"

SYSTEM_PROMPT = """You control literature search for a research synthesis agent.
Given the question and only evidence already opened, first decide whether the
evidence is sufficient, then propose the next concise search query. Seek
evidence that could verify or overturn the current conclusion. Do not invent
findings. Return exactly two lines in this order:
STOP: yes or no
QUERY: <search query>"""


def fit_prospective_models(
    development_path: Path,
    environment_tasks: list[dict[str, Any]],
    environment_documents: list[dict[str, Any]],
    development_scores_path: Path,
    environment_scores_path: Path,
    base_config: dict[str, Any],
    development_private_outcomes_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Fit nuisance and residual second moments without confirmatory outcomes."""
    development_tasks = load_jsonl(development_path / "tasks.jsonl")
    development_documents = load_jsonl(development_path / "documents.jsonl")
    if development_private_outcomes_path is not None:
        private_outcomes = {
            row["document_id"]: row
            for row in load_jsonl(development_private_outcomes_path)
        }
        development_documents = [
            {**row, "outcome": private_outcomes[row["document_id"]]}
            for row in development_documents
            if row.get("development_role") == "train"
        ]
        development_review_ids = {
            row["review_id"] for row in development_documents
        }
        development_tasks = [
            row for row in development_tasks
            if row["review_id"] in development_review_ids
        ]
    development_scores = load_jsonl(development_scores_path)
    environment_scores = load_jsonl(environment_scores_path)
    tasks = {
        row["review_id"]: row
        for row in development_tasks + environment_tasks
    }
    environment_design_documents = [
        {**row, "development_role": "confirmatory"}
        for row in environment_documents
    ]
    documents = development_documents + environment_design_documents
    scores = {
        row["document_id"]: row
        for row in development_scores + environment_scores
    }
    _, design, train_mask, _ = fit_design(
        tasks,
        documents,
        scores,
        int(base_config["development_experiment"]["tfidf_max_features"]),
    )
    start = len(development_documents)
    outcomes = np.asarray(
        [row["outcome"]["direction_score"] for row in development_documents],
        dtype=float,
    )
    padded_outcomes = np.concatenate(
        [outcomes, np.zeros(len(environment_documents), dtype=float)]
    )
    alpha = float(base_config["holdout_lock"]["point_estimator"]["ridge_alpha"])
    model = Ridge(alpha=alpha)
    model.fit(design[train_mask], padded_outcomes[train_mask])
    predictions = np.clip(model.predict(design), -1.0, 1.0)

    train_indices = np.flatnonzero(train_mask[:start])
    folds = int(base_config["development_experiment"]["outcome_crossfit_folds"])
    train_task_ids = sorted(
        {development_documents[index]["review_id"] for index in train_indices}
    )
    fold_by_task = {
        review_id: int(stable_hash(["confirmatory-cess-nuisance", review_id])[:8], 16)
        % folds
        for review_id in train_task_ids
    }
    crossfit = np.full(start, np.nan)
    for fold in range(folds):
        fit_rows = np.asarray(
            [
                index
                for index in train_indices
                if fold_by_task[development_documents[index]["review_id"]] != fold
            ],
            dtype=int,
        )
        held_rows = np.asarray(
            [
                index
                for index in train_indices
                if fold_by_task[development_documents[index]["review_id"]] == fold
            ],
            dtype=int,
        )
        fold_model = Ridge(alpha=alpha)
        fold_model.fit(design[fit_rows], outcomes[fit_rows])
        crossfit[held_rows] = np.clip(
            fold_model.predict(design[held_rows]), -1.0, 1.0
        )
    if np.isnan(crossfit[train_indices]).any():
        raise ValueError("missing cross-fitted nuisance predictions")
    residual = outcomes[train_indices] - crossfit[train_indices]
    second_model = Ridge(alpha=alpha)
    second_model.fit(
        design[train_indices], np.log(np.square(residual) + 1e-4)
    )
    raw_second = np.clip(np.exp(second_model.predict(design)), 1e-6, 4.0)
    scale = float(
        np.mean(np.square(residual)) / np.mean(raw_second[train_indices])
    )
    second = np.clip(scale * raw_second, 1e-4, 4.0)
    residual_means = []
    for review_id in train_task_ids:
        indices = np.asarray(
            [
                index
                for index in train_indices
                if development_documents[index]["review_id"] == review_id
            ],
            dtype=int,
        )
        residual_means.append(
            float(np.mean(outcomes[indices] - crossfit[indices]))
        )
    tau_squared = float(np.mean(np.square(residual_means)))
    development_index_by_id = {
        row["document_id"]: index
        for index, row in enumerate(development_documents)
    }
    prediction_by_id = {}
    second_by_id = {}
    for offset, row in enumerate(environment_documents):
        document_id = row["document_id"]
        if document_id in development_index_by_id:
            index = development_index_by_id[document_id]
            prediction_by_id[document_id] = float(crossfit[index])
            second_by_id[document_id] = float(second[index])
        else:
            prediction_by_id[document_id] = float(predictions[start + offset])
            second_by_id[document_id] = float(second[start + offset])
    return prediction_by_id, second_by_id, tau_squared, scale


def controller_probabilities(
    base: np.ndarray,
    second_moments: np.ndarray,
    controller: str,
    mixture: float,
    power: float,
) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    if controller == "agent":
        result = base
    elif controller == "uniform":
        result = (1.0 - mixture) * base + mixture / len(base)
    elif controller == "active_cess":
        influence = np.power(
            np.clip(np.asarray(second_moments, dtype=float), 1e-8, None),
            power,
        )
        proposal = influence / influence.sum()
        result = (1.0 - mixture) * base + mixture * proposal
    else:
        raise ValueError(f"unknown controller: {controller}")
    result = result / result.sum()
    if np.any(result <= 0.0) or not np.isclose(result.sum(), 1.0):
        raise ValueError("invalid controller probabilities")
    return result


def prospective_weight(
    tau_squared: float,
    second_moments: np.ndarray,
    initial_probability: np.ndarray,
    maximum_rounds: int,
) -> tuple[float, float]:
    n = len(second_moments)
    predicted_variance = float(
        np.sum(second_moments / (n * n * initial_probability))
        / maximum_rounds
    )
    denominator = tau_squared + predicted_variance
    shrinkage = float(tau_squared / denominator) if denominator else 0.0
    return shrinkage, predicted_variance


def episode_estimators(
    observations: list[dict[str, Any]],
    predictions: np.ndarray,
    target: float,
    maximum_rounds: int,
    shrinkage: float,
    predicted_variance: float,
    global_shrinkage: float | None = None,
) -> dict[str, Any]:
    n = len(predictions)
    baseline = float(np.mean(predictions))
    y = np.asarray([row["direction_score"] for row in observations], dtype=float)
    m = np.asarray(
        [row["nuisance_prediction"] for row in observations], dtype=float
    )
    probability = np.asarray(
        [row["selection_probability"] for row in observations], dtype=float
    )
    survival = np.asarray(
        [row["survival_probability"] for row in observations], dtype=float
    )
    correction = float(
        np.sum((y - m) / (n * probability * survival)) / maximum_rounds
    )
    ipw = float(
        np.sum(y / (n * probability * survival)) / maximum_rounds
    )
    return {
        "target": target,
        "naive_opened_mean": float(np.mean(y)),
        "global_outcome_regression": baseline,
        "sequential_ipw": float(np.clip(ipw, -1.0, 1.0)),
        "sequential_dr": float(np.clip(baseline + correction, -1.0, 1.0)),
        "preoutcome_cess": float(
            np.clip(baseline + shrinkage * correction, -1.0, 1.0)
        ),
        "preoutcome_global_cess": (
            float(np.clip(baseline + global_shrinkage * correction, -1.0, 1.0))
            if global_shrinkage is not None
            else None
        ),
        "unclipped_sequential_correction": correction,
        "preoutcome_shrinkage": shrinkage,
        "preoutcome_global_shrinkage": global_shrinkage,
        "predicted_correction_variance": predicted_variance,
        "rounds_observed": len(observations),
        "unique_documents_opened": len(
            {row["document_id"] for row in observations}
        ),
        "minimum_selection_probability": float(np.min(probability)),
        "minimum_survival_probability": float(np.min(survival)),
    }


def generate_deduplicated(
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    config: dict[str, Any],
) -> list[str]:
    import torch

    unique_prompts = list(dict.fromkeys(prompts))
    generated_by_prompt: dict[str, str] = {}
    for start in range(0, len(unique_prompts), batch_size):
        batch_prompts = unique_prompts[start : start + batch_size]
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
        texts = tokenizer.batch_decode(continuation, skip_special_tokens=True)
        generated_by_prompt.update(zip(batch_prompts, texts))
    return [generated_by_prompt[prompt] for prompt in prompts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument(
        "--base-config", type=Path, default=ROOT / "configs/ms2_external_v1.yaml"
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--environment-role", choices=["train", "validation"]
    )
    parser.add_argument(
        "--development-private-outcomes", type=Path
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists: {args.output}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(args.config)
    base_config = load_config(args.base_config)
    model_config = config["models"][args.model_key]
    model_path = Path(model_config["model_path"])
    tasks = load_jsonl(args.environment / "tasks.jsonl")
    documents = load_jsonl(args.environment / "documents.jsonl")
    if args.environment_role is not None:
        tasks = [
            row for row in tasks
            if row.get("development_role") == args.environment_role
        ]
        selected_reviews = {row["review_id"] for row in tasks}
        documents = [
            row for row in documents if row["review_id"] in selected_reviews
        ]
    environment_summary = json.loads((args.environment / "summary.json").read_text())
    if environment_summary["protocol_version"] != config["protocol_version"]:
        raise ValueError("environment/config protocol mismatch")
    if any("outcome" in row or "claim_matched" in row for row in documents):
        raise ValueError("environment contains outcome leakage")
    task_by_id = {row["review_id"]: row for row in tasks}
    documents_by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in documents:
        documents_by_review[row["review_id"]].append(row)
    for rows in documents_by_review.values():
        rows.sort(key=lambda row: row["document_id"])

    development_scores_path = ROOT / model_config["development_scores"]
    environment_scores_path = ROOT / model_config["confirmatory_scores"]
    predictions, second_moments, tau_squared, second_scale = fit_prospective_models(
        args.development,
        tasks,
        documents,
        development_scores_path,
        environment_scores_path,
        base_config,
        args.development_private_outcomes,
    )
    selected_ids = {row["document_id"] for row in documents}
    visible_by_id = {
        row["document_id"]: float(row["visible_direction_score"])
        for row in load_jsonl(environment_scores_path)
        if row["document_id"] in selected_ids
    }
    if (
        set(predictions) != selected_ids
        or set(second_moments) != selected_ids
        or set(visible_by_id) != selected_ids
    ):
        raise ValueError("model score coverage mismatch")

    private_outcomes = {
        row["document_id"]: row
        for row in load_jsonl(args.environment / "private_document_outcomes.jsonl")
        if row["document_id"] in selected_ids
    }
    selected_reviews = set(task_by_id)
    targets = {
        row["review_id"]: float(row["target"])
        for row in load_jsonl(args.environment / "private_targets.jsonl")
        if row["review_id"] in selected_reviews
    }
    if set(private_outcomes) != selected_ids:
        raise ValueError("private outcome/document mismatch")

    retrieval = {}
    for review_id, rows in documents_by_review.items():
        texts = [f"{row['title']} {row['full_abstract']}" for row in rows]
        vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        retrieval[review_id] = (vectorizer, vectorizer.fit_transform(texts))

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
                for controller in search["controllers"]:
                    for replicate in range(int(search["replicates"])):
                        episodes.append(
                            {
                                "episode_id": (
                                    f"{args.model_key}:{review_id}:{rank_condition}:"
                                    f"{stopping_regime}:{controller}:{replicate}"
                                ),
                                "review_id": review_id,
                                "rank_condition": rank_condition,
                                "stopping_regime": stopping_regime,
                                "controller": controller,
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
                                "preoutcome_shrinkage": None,
                                "predicted_correction_variance": None,
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
        generated_texts = generate_deduplicated(
            prompts, tokenizer, model, args.batch_size, config
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
                    "confirmatory-stop",
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
                        "controller": episode["controller"],
                        "replicate": episode["replicate"],
                        "round": round_index,
                        "query": query,
                        "raw_model_output": raw_text,
                        "parsed": parsed,
                        "recommend_stop": recommend_stop,
                        "stop_probability": stop_probability,
                        "stopped": True,
                        "document_id": None,
                    }
                )
                continue

            rows = documents_by_review[review_id]
            vectorizer, document_matrix = retrieval[review_id]
            visible = np.asarray(
                [visible_by_id[row["document_id"]] for row in rows], dtype=float
            )
            _, base = rank_probabilities(
                query,
                vectorizer,
                document_matrix,
                visible,
                episode["opened_counts"],
                episode["rank_condition"],
                config,
            )
            task_second = np.asarray(
                [second_moments[row["document_id"]] for row in rows], dtype=float
            )
            probability = controller_probabilities(
                base,
                task_second,
                episode["controller"],
                float(search["protected_exploration_mixture"]),
                float(search["active_influence_power"]),
            )
            if episode["preoutcome_shrinkage"] is None:
                shrinkage, predicted_variance = prospective_weight(
                    tau_squared, task_second, probability, maximum_rounds
                )
                episode["preoutcome_shrinkage"] = shrinkage
                episode["predicted_correction_variance"] = predicted_variance
            selection_rng = seeded_rng(
                config["seed"],
                "confirmatory-selection",
                args.model_key,
                review_id,
                episode["rank_condition"],
                episode["stopping_regime"],
                episode["replicate"],
                round_index,
            )
            selected_index = sample_categorical(selection_rng, probability)
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
                "selection_probability": float(probability[selected_index]),
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
                    "controller": episode["controller"],
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
                    "selection_probability": float(probability[selected_index]),
                    "candidate_document_ids": [
                        row["document_id"] for row in rows
                    ],
                    "candidate_probabilities": list(map(float, probability)),
                    "base_agent_probabilities": list(map(float, base)),
                }
            )
        print(
            json.dumps(
                {
                    "round": round_index,
                    "active_before_decision": len(active),
                    "episodes_with_evidence": sum(
                        len(episode["history"]) >= round_index
                        for episode in episodes
                    ),
                    "unique_prompts": len(set(prompts)),
                }
            ),
            flush=True,
        )

    cells = []
    for episode in episodes:
        rows = documents_by_review[episode["review_id"]]
        result = episode_estimators(
            episode["history"],
            np.asarray(
                [predictions[row["document_id"]] for row in rows], dtype=float
            ),
            targets[episode["review_id"]],
            maximum_rounds,
            float(episode["preoutcome_shrinkage"]),
            float(episode["predicted_correction_variance"]),
            (
                float(model_config["frozen_global_shrinkage"])
                if "frozen_global_shrinkage" in model_config
                else None
            ),
        )
        cells.append(
            {
                "episode_id": episode["episode_id"],
                "model_key": args.model_key,
                "review_id": episode["review_id"],
                "rank_condition": episode["rank_condition"],
                "stopping_regime": episode["stopping_regime"],
                "controller": episode["controller"],
                "replicate": episode["replicate"],
                "parser_failures": episode["parser_failures"],
                "stop_recommendations": episode["stop_recommendations"],
                "stop_decisions": episode["stop_decisions"],
                **result,
            }
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "traces.jsonl", traces)
    write_jsonl(args.output / "cells.jsonl", cells)
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": (
            "clean_development_validation"
            if args.environment_role == "validation"
            else "prospective_sequential_confirmatory"
        ),
        "environment_role": args.environment_role,
        "model_key": args.model_key,
        "tasks": len(tasks),
        "episodes": len(cells),
        "trace_rows": len(traces),
        "tau_squared": tau_squared,
        "second_moment_scale": second_scale,
        "query_model_path_read_only": str(model_path),
        "environment_label_free": True,
        "retrieval_used_true_outcomes": False,
        "query_prompt_received_unopened_outcomes": False,
        "shrinkage_frozen_before_trajectory_outcomes": True,
        "input_hashes": {
            "config": sha256_file(args.config),
            "environment_summary": sha256_file(args.environment / "summary.json"),
            "environment_tasks": sha256_file(args.environment / "tasks.jsonl"),
            "environment_documents": sha256_file(args.environment / "documents.jsonl"),
            "development_scores": sha256_file(development_scores_path),
            "confirmatory_scores": sha256_file(environment_scores_path),
            "model_config": sha256_file(model_path / "config.json"),
            "development_private_outcomes": (
                sha256_file(args.development_private_outcomes)
                if args.development_private_outcomes is not None
                else None
            ),
            "traces": sha256_file(args.output / "traces.jsonl"),
            "cells": sha256_file(args.output / "cells.jsonl"),
        },
    }
    atomic_write_json(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
