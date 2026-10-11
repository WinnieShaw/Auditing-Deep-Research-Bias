#!/usr/bin/env python3
"""Develop selection-bias corrections on the RealEffect Qwen+Llama development topics only."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl, stable_hash, write_jsonl
from scripts.build_ms2_external_manifest import sha256_file

DEFAULT_CONFIG = ROOT / "configs/realeffect_llama_development_numeric_v2.yaml"
DEFAULT_OUTPUT = (
    ROOT / "evidence_selection_benchmark/results/realeffect_aligned_v1/llama_development_numeric_v2"
)


def seeded_rng(*parts: Any) -> random.Random:
    return random.Random(int(stable_hash(list(parts))[:16], 16))


def standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    return np.zeros_like(values) if scale < 1e-12 else (values - values.mean()) / scale


def floored_softmax(logits: np.ndarray, temperature: float, floor: float) -> np.ndarray:
    if floor < 0.0 or floor * len(logits) >= 1.0:
        raise ValueError("invalid selection probability floor")
    values = np.asarray(logits, dtype=float) / temperature
    values -= float(values.max())
    values = np.exp(values)
    values /= values.sum()
    values = floor + (1.0 - floor * len(values)) * values
    return values / values.sum()


def draw_index(rng: random.Random, probabilities: np.ndarray) -> int:
    return int(rng.choices(range(len(probabilities)), weights=probabilities.tolist(), k=1)[0])


def direction(value: float, tolerance: float) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def build_design(
    tasks: dict[str, dict[str, Any]],
    documents: list[dict[str, Any]],
    scores: dict[str, dict[str, dict[str, Any]]],
    max_features: int,
) -> tuple[csr_matrix, list[str]]:
    texts = [
        " ".join(
            [
                tasks[row["review_id"]]["research_question"],
                row["title"],
                row["full_abstract"],
            ]
        )
        for row in documents
    ]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    text_features = vectorizer.fit_transform(texts)
    dense = []
    for row in documents:
        document_id = row["document_id"]
        features = []
        for model_key in ("qwen2_5_32b", "llama3_3_70b"):
            score = scores[model_key][document_id]
            features.extend(
                [
                    float(score["probability_A"]),
                    float(score["probability_B"]),
                    float(score["probability_C"]),
                    float(score["probability_D"]),
                    float(score["visible_direction_score"]),
                    float(score["permutation_consensus_fraction"]),
                ]
            )
        year = row.get("year")
        features.append((float(year) - 2005.0) / 20.0 if year else 0.0)
        dense.append(features)
    return hstack([text_features, csr_matrix(np.asarray(dense))], format="csr"), texts


def leave_topic_out_predictions(
    design: csr_matrix,
    outcomes: np.ndarray,
    topic_ids: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    unique_topics = sorted(set(topic_ids.tolist()))
    candidates: dict[float, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for alpha in alphas:
        predictions = np.full(len(outcomes), np.nan)
        for topic_id in unique_topics:
            test = topic_ids == topic_id
            train = ~test
            model = Ridge(alpha=alpha)
            model.fit(design[train], outcomes[train])
            predictions[test] = np.clip(model.predict(design[test]), -1.0, 1.0)
        if np.isnan(predictions).any():
            raise ValueError("missing leave-topic-out predictions")
        topic_errors = []
        for topic_id in unique_topics:
            mask = topic_ids == topic_id
            topic_errors.append(abs(float(predictions[mask].mean() - outcomes[mask].mean())))
        diagnostics[str(alpha)] = {
            "leave_topic_out_macro_mae": float(np.mean(topic_errors)),
            "document_mae": float(np.mean(np.abs(predictions - outcomes))),
        }
        candidates[alpha] = predictions
    selected = min(alphas, key=lambda value: (diagnostics[str(value)]["leave_topic_out_macro_mae"], value))
    return candidates[selected], selected, diagnostics


def rank_probabilities(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    visible: np.ndarray,
    opened_counts: np.ndarray,
    rank_condition: str,
    controller: str,
    config: dict[str, Any],
) -> np.ndarray:
    query_vector = vectorizer.transform([query])
    relevance = np.asarray((matrix @ query_vector.T).toarray()).ravel()
    sign = {"neutral": 0.0, "pro_benefit": 1.0, "pro_harm": -1.0}[rank_condition]
    search = config["search"]
    if controller == "mmr":
        opened = np.flatnonzero(opened_counts > 0)
        if len(opened):
            similarity = (matrix @ matrix[opened].T).max(axis=1).toarray().ravel()
        else:
            similarity = np.zeros(len(relevance), dtype=float)
        mixing = float(search["mmr_relevance_tradeoff"])
        retrieval_score = (
            mixing * standardize(relevance)
            - (1.0 - mixing) * standardize(similarity)
        )
    else:
        retrieval_score = float(search["relevance_weight"]) * standardize(relevance)
    logits = (
        retrieval_score
        + sign * float(search["direction_weight"]) * standardize(visible)
        - float(search["duplicate_logit_penalty"]) * opened_counts
    )
    base = floored_softmax(
        logits,
        float(search["selection_temperature"]),
        float(search["minimum_selection_probability"]),
    )
    if controller == "agent":
        return base
    if controller == "mmr":
        return base
    if controller == "protected":
        mixture = float(search["protected_uniform_mixture"])
        return (1.0 - mixture) * base + mixture / len(base)
    raise ValueError(controller)


def stop_probability(history: list[dict[str, Any]], config: dict[str, Any]) -> float:
    values = np.asarray([row["direction_score"] for row in history], dtype=float)
    signs = np.sign(values)
    majority = max(float(np.mean(signs >= 0.0)), float(np.mean(signs <= 0.0)))
    rule = config["search"]["adaptive_stop"]
    logit = (
        float(rule["intercept"])
        + float(rule["magnitude_coefficient"]) * abs(float(values.mean()))
        + float(rule["agreement_coefficient"]) * majority
    )
    probability = 1.0 / (1.0 + math.exp(-logit))
    return min(probability, float(rule["maximum_probability"]))


def estimate_episode(
    history: list[dict[str, Any]],
    predictions: np.ndarray,
    target: float,
    maximum_rounds: int,
    tau_squared: float,
    residual_variance: float,
    config: dict[str, Any],
) -> dict[str, float]:
    n = len(predictions)
    y = np.asarray([row["direction_score"] for row in history], dtype=float)
    m = np.asarray([row["prediction"] for row in history], dtype=float)
    p = np.asarray([row["selection_probability"] for row in history], dtype=float)
    survival = np.asarray([row["survival_probability"] for row in history], dtype=float)
    weights = 1.0 / (n * p * survival)
    baseline = float(predictions.mean())
    correction = float(np.sum(weights * (y - m)) / maximum_rounds)
    ipw = float(np.sum(weights * y) / maximum_rounds)
    result = {
        "target": target,
        "naive_opened_mean": float(y.mean()),
        "outcome_regression": baseline,
        "sequential_ipw": float(np.clip(ipw, -1.0, 1.0)),
        "sequential_dr": float(np.clip(baseline + correction, -1.0, 1.0)),
    }
    predicted_variance = residual_variance * float(np.sum(np.square(weights))) / (maximum_rounds**2)
    analytic = tau_squared / (tau_squared + predicted_variance) if tau_squared + predicted_variance > 0 else 0.0
    result["cess_analytic"] = float(np.clip(baseline + analytic * correction, -1.0, 1.0))
    result["cess_analytic_shrinkage"] = float(analytic)
    result["predicted_correction_variance"] = float(predicted_variance)
    for shrinkage in config["estimators"]["candidate_cess_shrinkages"]:
        key = f"cess_lambda_{str(float(shrinkage)).replace('.', '_')}"
        result[key] = float(np.clip(baseline + float(shrinkage) * correction, -1.0, 1.0))
    residual = y - m
    for cap in config["estimators"]["importance_weight_caps"]:
        capped = np.minimum(weights, float(cap))
        robust_correction = float(np.sum(capped * residual) / maximum_rounds)
        key = f"robust_anchor_cap_{str(float(cap)).replace('.', '_')}"
        result[key] = float(np.clip(baseline + analytic * robust_correction, -1.0, 1.0))
    result["rounds_observed"] = float(len(history))
    result["unique_documents_opened"] = float(len({row["document_id"] for row in history}))
    result["minimum_joint_probability"] = float(np.min(p * survival))
    return result


def summarize(cells: list[dict[str, Any]], methods: list[str], tolerance: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in methods:
        errors = np.asarray([abs(float(row[method]) - float(row["target"])) for row in cells])
        wrong = np.asarray(
            [
                direction(float(row[method]), tolerance) != direction(float(row["target"]), tolerance)
                for row in cells
            ],
            dtype=float,
        )
        result[method] = {
            "episode_mae": float(errors.mean()),
            "episode_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "wrong_direction_rate": float(wrong.mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    config = load_config(args.config)
    data = ROOT / config["data"]
    task_rows = load_jsonl(data / "tasks.jsonl")
    if len(task_rows) != int(config["split"]["expected_topics"]):
        raise ValueError("development topic count mismatch")
    selected_topics = {row["review_id"] for row in task_rows}
    if any(row["development_role"] == "holdout" for row in task_rows):
        raise ValueError("holdout task entered development runner")
    tasks = {row["review_id"]: row for row in task_rows}
    documents = load_jsonl(data / "documents.jsonl")
    if any(row["development_role"] != "development" for row in documents):
        raise ValueError("holdout document entered development runner")
    selected_ids = {row["document_id"] for row in documents}
    private_rows = load_jsonl(data / "private_document_outcomes.jsonl")
    private_targets = load_jsonl(data / "private_targets.jsonl")
    if len(private_rows) != len(documents) or len(private_targets) != len(tasks):
        raise ValueError("development private data coverage mismatch")
    outcomes_by_id = {row["document_id"]: float(row["direction_score"]) for row in private_rows}
    targets = {row["review_id"]: float(row["target"]) for row in private_targets}

    scores: dict[str, dict[str, dict[str, Any]]] = {}
    score_hashes = {}
    for model_key, relative_path in config["visible_scores"].items():
        path = ROOT / relative_path
        score_hashes[model_key] = sha256_file(path)
        rows = [row for row in load_jsonl(path) if row["document_id"] in selected_ids]
        scores[model_key] = {row["document_id"]: row for row in rows}
        if set(scores[model_key]) != selected_ids:
            raise ValueError(f"{model_key} score coverage mismatch")

    design, _ = build_design(
        tasks, documents, scores, int(config["nuisance_model"]["tfidf_max_features"])
    )
    outcomes = np.asarray([outcomes_by_id[row["document_id"]] for row in documents], dtype=float)
    topic_ids = np.asarray([row["review_id"] for row in documents])
    predictions, selected_alpha, alpha_diagnostics = leave_topic_out_predictions(
        design,
        outcomes,
        topic_ids,
        [float(value) for value in config["nuisance_model"]["ridge_alphas"]],
    )
    residuals = outcomes - predictions
    residual_variance = float(np.mean(np.square(residuals)))
    topic_residual_means = [float(residuals[topic_ids == topic].mean()) for topic in sorted(selected_topics)]
    tau_squared = float(np.var(topic_residual_means, ddof=1))
    prediction_by_id = {
        row["document_id"]: float(value) for row, value in zip(documents, predictions)
    }

    docs_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in documents:
        docs_by_topic[row["review_id"]].append(row)
    for rows in docs_by_topic.values():
        rows.sort(key=lambda row: row["document_id"])

    search = config["search"]
    cells = []
    traces = []
    for review_id in sorted(selected_topics):
        rows = docs_by_topic[review_id]
        corpus = [f"{row['title']} {row['full_abstract']}" for row in rows]
        retrieval = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = retrieval.fit_transform(corpus)
        for score_source in search["policy_score_sources"]:
            if score_source == "ensemble":
                visible = np.asarray(
                    [
                        0.5 * (
                            float(scores["qwen2_5_32b"][row["document_id"]]["visible_direction_score"])
                            + float(scores["llama3_3_70b"][row["document_id"]]["visible_direction_score"])
                        )
                        for row in rows
                    ]
                )
            else:
                visible = np.asarray(
                    [float(scores[score_source][row["document_id"]]["visible_direction_score"]) for row in rows]
                )
            for rank_condition in search["rank_conditions"]:
                for stopping_regime in search["stopping_regimes"]:
                    for controller in search["controllers"]:
                        for replicate in range(int(search["replicates"])):
                            rng = seeded_rng(
                                config["seed"], review_id, score_source, rank_condition,
                                stopping_regime, controller, replicate,
                            )
                            history: list[dict[str, Any]] = []
                            opened_counts = np.zeros(len(rows), dtype=float)
                            survival = 1.0
                            for round_index in range(1, int(search["maximum_rounds"]) + 1):
                                if (
                                    stopping_regime == "adaptive"
                                    and round_index > int(search["minimum_rounds_before_stop"])
                                ):
                                    probability_stop = stop_probability(history, config)
                                    if rng.random() < probability_stop:
                                        traces.append(
                                            {
                                                "review_id": review_id,
                                                "score_source": score_source,
                                                "rank_condition": rank_condition,
                                                "stopping_regime": stopping_regime,
                                                "controller": controller,
                                                "replicate": replicate,
                                                "round": round_index,
                                                "stopped": True,
                                                "stop_probability": probability_stop,
                                            }
                                        )
                                        break
                                    survival *= 1.0 - probability_stop
                                query = tasks[review_id]["research_question"]
                                if history:
                                    query = f"{query} {history[-1]['title']}"
                                probability = rank_probabilities(
                                    query, retrieval, matrix, visible, opened_counts,
                                    rank_condition, controller, config,
                                )
                                index = draw_index(rng, probability)
                                row = rows[index]
                                observation = {
                                    "document_id": row["document_id"],
                                    "title": row["title"],
                                    "direction_score": outcomes_by_id[row["document_id"]],
                                    "prediction": prediction_by_id[row["document_id"]],
                                    "selection_probability": float(probability[index]),
                                    "survival_probability": float(survival),
                                }
                                history.append(observation)
                                opened_counts[index] += 1.0
                                traces.append(
                                    {
                                        "review_id": review_id,
                                        "score_source": score_source,
                                        "rank_condition": rank_condition,
                                        "stopping_regime": stopping_regime,
                                        "controller": controller,
                                        "replicate": replicate,
                                        "round": round_index,
                                        "stopped": False,
                                        "document_id": row["document_id"],
                                        "selection_probability": float(probability[index]),
                                        "survival_probability": float(survival),
                                    }
                                )
                            estimates = estimate_episode(
                                history,
                                np.asarray([prediction_by_id[row["document_id"]] for row in rows]),
                                targets[review_id],
                                int(search["maximum_rounds"]),
                                tau_squared,
                                residual_variance,
                                config,
                            )
                            cells.append(
                                {
                                    "review_id": review_id,
                                    "score_source": score_source,
                                    "rank_condition": rank_condition,
                                    "stopping_regime": stopping_regime,
                                    "controller": controller,
                                    "replicate": replicate,
                                    **estimates,
                                }
                            )

    method_prefixes = (
        "naive_opened_mean", "outcome_regression", "sequential_ipw", "sequential_dr",
        "cess_analytic", "cess_lambda_", "robust_anchor_cap_",
    )
    methods = [key for key in cells[0] if any(key == prefix or key.startswith(prefix) for prefix in method_prefixes)]
    methods = [key for key in methods if key not in {"cess_analytic_shrinkage"}]
    overall = summarize(cells, methods, float(config["estimators"]["direction_tolerance"]))
    grouped = {}
    group_keys = ("score_source", "rank_condition", "stopping_regime", "controller")
    for values in sorted({tuple(row[key] for key in group_keys) for row in cells}):
        subset = [row for row in cells if tuple(row[key] for key in group_keys) == values]
        grouped["|".join(map(str, values))] = summarize(
            subset, methods, float(config["estimators"]["direction_tolerance"])
        )

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "cells.jsonl", cells)
    write_jsonl(args.output / "traces.jsonl", traces)
    atomic_write_json(args.output / "metrics.json", {"overall": overall, "by_design_cell": grouped})
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": "development_only_method_selection",
        "topics": len(tasks),
        "documents": len(documents),
        "episodes": len(cells),
        "trace_rows": len(traces),
        "selected_ridge_alpha": selected_alpha,
        "alpha_diagnostics": alpha_diagnostics,
        "residual_variance": residual_variance,
        "tau_squared": tau_squared,
        "holdout_targets_loaded": False,
        "holdout_metrics_computed": False,
        "output_is_confirmatory": False,
        "input_hashes": {
            "config": sha256_file(args.config),
            "tasks": sha256_file(data / "tasks.jsonl"),
            "documents": sha256_file(data / "documents.jsonl"),
            "private_document_outcomes": sha256_file(data / "private_document_outcomes.jsonl"),
            "private_targets": sha256_file(data / "private_targets.jsonl"),
            **{f"visible_scores_{key}": value for key, value in score_hashes.items()},
            "cells": sha256_file(args.output / "cells.jsonl"),
            "traces": sha256_file(args.output / "traces.jsonl"),
        },
    }
    atomic_write_json(args.output / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
