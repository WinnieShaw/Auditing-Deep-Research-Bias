#!/usr/bin/env python3
"""Develop CLOC on MS² using only the frozen development cohort."""
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
from causal_search_credit.selection_estimators import inclusion_probabilities
from causal_search_credit.util import (
    atomic_write_json,
    load_jsonl,
    stable_hash,
    write_jsonl,
)
from scripts.build_ms2_external_manifest import sha256_file

DEFAULT_DATA = ROOT / "evidence_selection_benchmark/data/ms2_external_v1/development"
DEFAULT_VISIBLE_SCORES = (
    ROOT
    / "evidence_selection_benchmark/features/ms2_external_v1"
    / "qwen3_8b_edge60_visible_scores.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "evidence_selection_benchmark/results/ms2_external/development_program_v4"
)


def document_text(task: dict[str, Any], document: dict[str, Any]) -> str:
    return " ".join(
        [
            task["review_title"],
            task["target_intervention"],
            task["target_outcome"],
            document["title"],
            document["full_abstract"],
        ]
    )


def direction(value: float, tolerance: float) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def standardized(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(values.std())
    if standard_deviation < 1e-8:
        return np.zeros_like(values)
    return (values - values.mean()) / standard_deviation


def effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    return float(weights.sum() ** 2 / denominator) if denominator else 0.0


def cloc_estimate(
    outcome_predictions: np.ndarray,
    outcomes: np.ndarray,
    selected: np.ndarray,
    propensities: np.ndarray,
    tau_squared: float,
    variance_scale: float,
    weighting: str,
) -> tuple[float, float, float]:
    """Empirical-Bayes shrinkage of a task-local residual calibration."""
    baseline = float(outcome_predictions.mean())
    residuals = outcomes - outcome_predictions
    if weighting == "logged_propensity":
        weights = 1.0 / propensities[selected]
    elif weighting == "uniform":
        weights = np.ones(int(selected.sum()), dtype=float)
    else:
        raise ValueError(f"unknown CLOC weighting: {weighting}")
    selected_residuals = residuals[selected]
    local_shift = float(np.average(selected_residuals, weights=weights))
    ess = effective_sample_size(weights)
    residual_variance = float(
        np.average(np.square(selected_residuals - local_shift), weights=weights)
    )
    standard_error_squared = residual_variance / max(ess, 1.0)
    denominator = tau_squared + variance_scale * standard_error_squared
    shrinkage = tau_squared / denominator if denominator > 0 else 0.0
    estimate = baseline + shrinkage * local_shift
    return float(np.clip(estimate, -1.0, 1.0)), shrinkage, ess


def seeded_rng(*parts: Any) -> random.Random:
    seed = int(stable_hash(list(parts))[:16], 16)
    return random.Random(seed)


def draw_selection(
    propensities: np.ndarray,
    rng: random.Random,
    minimum_selected: int = 2,
    attempts: int = 100,
) -> np.ndarray | None:
    for _ in range(attempts):
        selected = np.array(
            [rng.random() < probability for probability in propensities],
            dtype=bool,
        )
        if int(selected.sum()) >= minimum_selected:
            return selected
    return None


def metric_summary(
    cells: list[dict[str, Any]],
    methods: list[str],
    tolerance: float,
) -> dict[str, dict[str, float]]:
    result = {}
    for method in methods:
        estimates = [float(row[method]) for row in cells]
        targets = [float(row["target"]) for row in cells]
        result[method] = {
            "mae": float(
                np.mean([abs(estimate - target) for estimate, target in zip(estimates, targets)])
            ),
            "wrong_direction_rate": float(
                np.mean(
                    [
                        direction(estimate, tolerance) != direction(target, tolerance)
                        for estimate, target in zip(estimates, targets)
                    ]
                )
            ),
        }
    return result


def fit_design(
    tasks: dict[str, dict[str, Any]],
    documents: list[dict[str, Any]],
    visible_scores: dict[str, dict[str, Any]],
    max_features: int,
) -> tuple[TfidfVectorizer, csr_matrix, np.ndarray, list[str]]:
    texts = [document_text(tasks[row["review_id"]], row) for row in documents]
    train_mask = np.array(
        [row["development_role"] == "train" for row in documents], dtype=bool
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    vectorizer.fit([text for text, keep in zip(texts, train_mask) if keep])
    text_features = vectorizer.transform(texts)
    dense = []
    for document in documents:
        score = visible_scores[document["document_id"]]
        year = document.get("year")
        try:
            year_value = (float(year) - 2005.0) / 20.0
        except (TypeError, ValueError):
            year_value = 0.0
        dense.append(
            [
                score["probability_decreased"],
                score["probability_no_difference"],
                score["probability_increased"],
                float(score.get("probability_unclear", 0.0)),
                score["visible_direction_score"],
                year_value,
            ]
        )
    design = hstack([text_features, csr_matrix(np.asarray(dense))], format="csr")
    return vectorizer, design, train_mask, texts


def task_macro_mae(
    documents: list[dict[str, Any]],
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (document, keep) in enumerate(zip(documents, mask)):
        if keep:
            grouped[document["review_id"]].append(index)
    errors = []
    for indices in grouped.values():
        target = float(
            np.mean(
                [
                    documents[index]["outcome"]["direction_score"]
                    for index in indices
                ]
            )
        )
        estimate = float(np.mean(predictions[indices]))
        errors.append(abs(estimate - target))
    return float(np.mean(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ms2_external_v1.yaml"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--visible-scores", type=Path, default=DEFAULT_VISIBLE_SCORES
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--policy-score-source",
        choices=["nuisance_prediction", "visible_direction"],
        default="nuisance_prediction",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing overwrite: {args.output}")

    config = load_config(args.config)
    experiment = config["development_experiment"]
    tasks_rows = load_jsonl(args.data / "tasks.jsonl")
    tasks = {row["review_id"]: row for row in tasks_rows}
    documents = load_jsonl(args.data / "documents.jsonl")
    visible_rows = load_jsonl(args.visible_scores)
    visible_scores = {row["document_id"]: row for row in visible_rows}
    visible_direction_scores = np.asarray(
        [visible_scores[row["document_id"]]["visible_direction_score"] for row in documents],
        dtype=float,
    )
    if set(visible_scores) != {row["document_id"] for row in documents}:
        raise ValueError("visible-score/document IDs do not match")
    if any(row["development_role"] not in {"train", "validation"} for row in documents):
        raise ValueError("development runner refuses non-development documents")

    _, design, train_mask, _ = fit_design(
        tasks,
        documents,
        visible_scores,
        int(experiment["tfidf_max_features"]),
    )
    validation_mask = ~train_mask
    outcomes = np.array(
        [row["outcome"]["direction_score"] for row in documents], dtype=float
    )
    alphas = [float(value) for value in experiment["outcome_ridge_alphas"]]
    alpha_results = {}
    candidate_predictions = {}
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(design[train_mask], outcomes[train_mask])
        predictions = np.clip(model.predict(design), -1.0, 1.0)
        candidate_predictions[alpha] = predictions
        alpha_results[str(alpha)] = {
            "validation_task_macro_mae": task_macro_mae(
                documents, predictions, validation_mask
            )
        }
    selected_alpha = min(
        alphas,
        key=lambda alpha: (
            alpha_results[str(alpha)]["validation_task_macro_mae"],
            alpha,
        ),
    )
    predictions = candidate_predictions[selected_alpha]

    train_task_ids = sorted(
        {
            row["review_id"]
            for row, keep in zip(documents, train_mask)
            if keep
        }
    )
    fold_by_task = {
        review_id: int(stable_hash(["outcome-crossfit", review_id])[:8], 16)
        % int(experiment["outcome_crossfit_folds"])
        for review_id in train_task_ids
    }
    crossfit_predictions = np.full(len(documents), np.nan)
    for fold in range(int(experiment["outcome_crossfit_folds"])):
        fit_mask = np.array(
            [
                keep and fold_by_task[row["review_id"]] != fold
                for row, keep in zip(documents, train_mask)
            ]
        )
        predict_mask = np.array(
            [
                keep and fold_by_task[row["review_id"]] == fold
                for row, keep in zip(documents, train_mask)
            ]
        )
        model = Ridge(alpha=selected_alpha)
        model.fit(design[fit_mask], outcomes[fit_mask])
        crossfit_predictions[predict_mask] = np.clip(
            model.predict(design[predict_mask]), -1.0, 1.0
        )
    if np.isnan(crossfit_predictions[train_mask]).any():
        raise ValueError("missing cross-fitted outcome predictions")

    task_residual_means = []
    for review_id in train_task_ids:
        indices = np.array(
            [
                index
                for index, document in enumerate(documents)
                if document["review_id"] == review_id
                and document["development_role"] == "train"
            ]
        )
        task_residual_means.append(
            float(np.mean(outcomes[indices] - crossfit_predictions[indices]))
        )
    tau_squared = float(np.var(task_residual_means, ddof=1))

    validation_indices: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        if document["development_role"] == "validation":
            validation_indices[document["review_id"]].append(index)

    policies = list(config["selection"]["policies"])
    strengths = [float(value) for value in experiment["policy_strengths"]]
    budgets = [int(value) for value in config["selection"]["budgets"]]
    replicates = int(experiment["replicates"])
    exploration = float(config["selection"]["minimum_exploration_probability"])
    minimum_propensity = float(experiment["minimum_propensity"])
    maximum_propensity = float(experiment["maximum_propensity"])
    variance_scales = [float(value) for value in experiment["cloc_variance_scales"]]
    weightings = list(experiment["cloc_weightings"])

    fixed_methods = [
        "naive_opened_mean",
        "global_outcome_regression",
        "logged_hajek_ipw",
        "horvitz_aipw",
        "hajek_dr",
    ]
    cells = []
    invalid_draws = 0
    for review_id, indices_list in sorted(validation_indices.items()):
        indices = np.asarray(indices_list)
        task_outcomes = outcomes[indices]
        task_predictions = predictions[indices]
        target = float(task_outcomes.mean())
        task_policy_values = (
            task_predictions
            if args.policy_score_source == "nuisance_prediction"
            else visible_direction_scores[indices]
        )
        global_estimate = float(task_predictions.mean())
        for policy in policies:
            policy_strengths = [0.0] if policy == "neutral" else strengths
            if policy == "neutral":
                raw_policy_score = np.zeros(len(indices))
            elif policy == "pro_increase":
                raw_policy_score = task_policy_values
            elif policy == "pro_decrease":
                raw_policy_score = -task_policy_values
            elif policy == "pro_no_difference":
                raw_policy_score = -np.abs(task_policy_values)
            else:
                raise ValueError(f"unknown policy: {policy}")
            policy_score = standardized(raw_policy_score)
            for strength in policy_strengths:
                for budget in budgets:
                    probabilities, _ = inclusion_probabilities(
                        policy_score,
                        "neutral" if policy == "neutral" else "pro_positive",
                        strength,
                        budget,
                        exploration,
                        minimum_propensity,
                        maximum_propensity,
                    )
                    propensities = np.asarray(probabilities)
                    for replicate in range(replicates):
                        selected = draw_selection(
                            propensities,
                            seeded_rng(
                                config["seed"],
                                review_id,
                                policy,
                                strength,
                                budget,
                                replicate,
                            ),
                        )
                        if selected is None:
                            invalid_draws += 1
                            continue
                        selected_outcomes = task_outcomes[selected]
                        inverse = 1.0 / propensities[selected]
                        naive = float(selected_outcomes.mean())
                        logged_ipw = float(
                            np.average(selected_outcomes, weights=inverse)
                        )
                        residuals = task_outcomes - task_predictions
                        horvitz_aipw = float(
                            global_estimate
                            + np.sum(
                                selected.astype(float)
                                * residuals
                                / propensities
                            )
                            / len(indices)
                        )
                        hajek_dr = float(
                            global_estimate
                            + np.average(residuals[selected], weights=inverse)
                        )
                        row = {
                            "review_id": review_id,
                            "policy": policy,
                            "policy_strength": strength,
                            "budget": budget,
                            "replicate": replicate,
                            "selected_documents": int(selected.sum()),
                            "target": target,
                            "naive_opened_mean": naive,
                            "global_outcome_regression": global_estimate,
                            "logged_hajek_ipw": logged_ipw,
                            "horvitz_aipw": float(
                                np.clip(horvitz_aipw, -1.0, 1.0)
                            ),
                            "hajek_dr": float(np.clip(hajek_dr, -1.0, 1.0)),
                        }
                        for weighting in weightings:
                            for scale in variance_scales:
                                key = f"cloc__{weighting}__{scale:g}"
                                estimate, shrinkage, ess = cloc_estimate(
                                    task_predictions,
                                    task_outcomes,
                                    selected,
                                    propensities,
                                    tau_squared,
                                    scale,
                                    weighting,
                                )
                                row[key] = estimate
                                row[f"{key}__shrinkage"] = shrinkage
                                row[f"{key}__ess"] = ess
                        cells.append(row)

    cloc_methods = [
        f"cloc__{weighting}__{scale:g}"
        for weighting in weightings
        for scale in variance_scales
    ]
    tolerance = float(config["evaluation"]["direction_tolerance"])
    candidate_metrics = metric_summary(cells, cloc_methods, tolerance)
    selected_cloc = min(
        cloc_methods,
        key=lambda method: (candidate_metrics[method]["mae"], method),
    )
    methods = fixed_methods + [selected_cloc]
    biased = [row for row in cells if row["policy"] != "neutral"]
    weak = [
        row
        for row in biased
        if row["policy_strength"] == min(strengths)
    ]
    strong = [
        row
        for row in biased
        if row["policy_strength"] == max(strengths)
    ]
    neutral = [row for row in cells if row["policy"] == "neutral"]
    subsets = {
        "overall": metric_summary(cells, methods, tolerance),
        "neutral": metric_summary(neutral, methods, tolerance),
        "weak": metric_summary(weak, methods, tolerance),
        "strong": metric_summary(strong, methods, tolerance),
    }
    best_fixed = min(
        fixed_methods,
        key=lambda method: (subsets["overall"][method]["mae"], method),
    )
    proposed = subsets["overall"][selected_cloc]
    best = subsets["overall"][best_fixed]
    naive_strong = subsets["strong"]["naive_opened_mean"]["mae"]
    cloc_strong = subsets["strong"][selected_cloc]["mae"]
    naive_weak = subsets["weak"]["naive_opened_mean"]["mae"]
    cloc_weak = subsets["weak"][selected_cloc]["mae"]
    gates = {
        "weak_relative_harm": {
            "value": cloc_weak / naive_weak - 1.0,
            "threshold": config["gate"]["maximum_weak_policy_relative_harm"],
        },
        "strong_relative_gain": {
            "value": 1.0 - cloc_strong / naive_strong,
            "threshold": config["gate"]["minimum_strong_policy_relative_gain"],
        },
        "overall_gain_over_best_fixed": {
            "value": 1.0 - proposed["mae"] / best["mae"],
            "threshold": config["gate"][
                "minimum_overall_gain_over_best_deployable_baseline"
            ],
        },
        "wrong_direction_relative_gain": {
            "value": 1.0
            - proposed["wrong_direction_rate"]
            / max(
                subsets["overall"]["naive_opened_mean"][
                    "wrong_direction_rate"
                ],
                1e-12,
            ),
            "threshold": config["gate"][
                "minimum_wrong_direction_relative_gain"
            ],
        },
    }
    for name, gate in gates.items():
        gate["pass"] = (
            gate["value"] <= gate["threshold"]
            if name == "weak_relative_harm"
            else gate["value"] >= gate["threshold"]
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "cells.jsonl", cells)
    summary = {
        "protocol_version": config["protocol_version"],
        "scope": "development_validation_only",
        "development_tasks": len(validation_indices),
        "cells": len(cells),
        "invalid_draws": invalid_draws,
        "outcome_model": {
            "selection_score": args.policy_score_source,
            "selected_ridge_alpha": selected_alpha,
            "candidate_results": alpha_results,
            "crossfit_folds": int(experiment["outcome_crossfit_folds"]),
        },
        "tau_squared": tau_squared,
        "selected_cloc": selected_cloc,
        "selected_cloc_candidate_metrics": candidate_metrics[selected_cloc],
        "best_fixed_deployable": best_fixed,
        "metrics": subsets,
        "gate_checks": gates,
        "development_gate_pass": all(gate["pass"] for gate in gates.values()),
        "input_hashes": {
            "tasks": sha256_file(args.data / "tasks.jsonl"),
            "documents": sha256_file(args.data / "documents.jsonl"),
            "visible_scores": sha256_file(args.visible_scores),
        },
        "holdout_opened": False,
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
