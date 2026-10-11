#!/usr/bin/env python3
"""Post-lock robustness analyses for the prospective sequential CESS traces."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl
from scripts.run_ms2_sequential_confirmatory import (
    DEFAULT_DEVELOPMENT,
    DEFAULT_ENVIRONMENT,
    fit_prospective_models,
    prospective_weight,
)

DEFAULT_CONFIG = ROOT / "configs/ms2_sequential_search_confirmatory_v1.yaml"
DEFAULT_OUTPUT = ROOT / "reports/ms2_sequential_robustness_statistics.json"
MODEL_KEYS = ("qwen2_5_32b", "olmo3_7b")
CLIP_CAPS = (2.0, 5.0, 10.0)
FIXED_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROPENSITY_TEMPERATURES = (0.75, 1.0, 1.25)


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def mae(rows: list[dict[str, Any]], method: str) -> float:
    return float(np.mean([abs(float(row[method]) - float(row["target"])) for row in rows]))


def relative_gain(rows: list[dict[str, Any]], proposed: str, reference: str) -> float:
    return float(1.0 - mae(rows, proposed) / mae(rows, reference))


def bootstrap_gain(
    rows: list[dict[str, Any]], proposed: str, reference: str, replicates: int, seed: int
) -> dict[str, Any]:
    grouped = group_rows(rows, "review_id")
    review_ids = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(review_ids, len(review_ids), replace=True)
        sample = [row for review_id in sampled for row in grouped[review_id]]
        draws.append(relative_gain(sample, proposed, reference))
    return {
        "relative_gain": relative_gain(rows, proposed, reference),
        "95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def tempered_probability(trace: dict[str, Any], alpha: float) -> float:
    probabilities = np.asarray(trace["candidate_probabilities"], dtype=float)
    probabilities = np.power(probabilities, alpha)
    probabilities /= probabilities.sum()
    selected = trace["candidate_document_ids"].index(trace["document_id"])
    return float(probabilities[selected])


def initial_tempered_probabilities(trace: dict[str, Any], alpha: float) -> np.ndarray:
    probabilities = np.asarray(trace["candidate_probabilities"], dtype=float)
    probabilities = np.power(probabilities, alpha)
    return probabilities / probabilities.sum()


def episode_estimates(
    traces: list[dict[str, Any]],
    predictions: dict[str, float],
    second_moments: dict[str, float],
    outcomes: dict[str, float],
    tau_squared: float,
    horizon: int,
    propensity_alpha: float = 1.0,
) -> dict[str, float]:
    observed = sorted(traces, key=lambda row: int(row["round"]))[:horizon]
    candidate_ids = observed[0]["candidate_document_ids"]
    baseline = float(np.mean([predictions[document_id] for document_id in candidate_ids]))
    initial_probability = initial_tempered_probabilities(observed[0], propensity_alpha)
    shrinkage, predicted_variance = prospective_weight(
        tau_squared,
        np.asarray([second_moments[document_id] for document_id in candidate_ids]),
        initial_probability,
        horizon,
    )
    y = np.asarray([outcomes[row["document_id"]] for row in observed])
    m = np.asarray([predictions[row["document_id"]] for row in observed])
    probability = np.asarray(
        [tempered_probability(row, propensity_alpha) for row in observed]
    )
    survival = np.asarray([float(row["survival_probability"]) for row in observed])
    n = len(candidate_ids)
    design_weight = 1.0 / (n * probability * survival)
    residual = y - m
    correction = float(np.sum(design_weight * residual) / horizon)
    estimates = {
        "naive": float(np.mean(y)),
        "outcome_regression": baseline,
        "sequential_dr": float(np.clip(baseline + correction, -1.0, 1.0)),
        "cess": float(np.clip(baseline + shrinkage * correction, -1.0, 1.0)),
        "shrinkage": shrinkage,
        "predicted_variance": predicted_variance,
        "unclipped_correction": correction,
        "minimum_probability": float(np.min(probability)),
    }
    hajek_denominator = float(np.sum(1.0 / (probability * survival)))
    hajek_residual = float(
        np.sum(residual / (probability * survival)) / hajek_denominator
    )
    estimates["hajek_dr"] = float(np.clip(baseline + hajek_residual, -1.0, 1.0))
    for cap in CLIP_CAPS:
        clipped = np.minimum(design_weight, cap)
        value = baseline + float(np.sum(clipped * residual) / horizon)
        estimates[f"clipped_dr_{cap:g}"] = float(np.clip(value, -1.0, 1.0))
    for value in FIXED_LAMBDAS:
        estimate = baseline + value * correction
        estimates[f"fixed_lambda_{value:g}"] = float(np.clip(estimate, -1.0, 1.0))
    return estimates


def analyze_model(
    model_key: str,
    config: dict[str, Any],
    environment: Path,
    development: Path,
    result_dir: Path,
) -> dict[str, Any]:
    base_config = load_config(ROOT / config["source"]["base_config"])
    tasks = load_jsonl(environment / "tasks.jsonl")
    documents = load_jsonl(environment / "documents.jsonl")
    model_config = config["models"][model_key]
    predictions, second_moments, tau_squared, _ = fit_prospective_models(
        development,
        tasks,
        documents,
        ROOT / model_config["development_scores"],
        ROOT / model_config["confirmatory_scores"],
        base_config,
    )
    outcomes = {
        row["document_id"]: float(row["direction_score"])
        for row in load_jsonl(environment / "private_document_outcomes.jsonl")
    }
    targets = {
        row["review_id"]: float(row["target"])
        for row in load_jsonl(environment / "private_targets.jsonl")
    }
    cells = load_jsonl(result_dir / "cells.jsonl")
    traces = load_jsonl(result_dir / "traces.jsonl")
    fixed_active_ids = {
        row["episode_id"]
        for row in cells
        if row["stopping_regime"] == "fixed_horizon"
        and row["controller"] == "active_cess"
    }
    trace_groups = group_rows(
        [
            row
            for row in traces
            if row["episode_id"] in fixed_active_ids and not row["stopped"]
        ],
        "episode_id",
    )
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    horizon_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    temperature_rows: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for episode_id in sorted(fixed_active_ids):
        episode_traces = trace_groups[episode_id]
        first = episode_traces[0]
        base = {
            "episode_id": episode_id,
            "review_id": first["review_id"],
            "rank_condition": first["rank_condition"],
            "target": targets[first["review_id"]],
        }
        for horizon in range(1, int(config["search"]["maximum_rounds"]) + 1):
            horizon_rows[horizon].append(
                {
                    **base,
                    **episode_estimates(
                        episode_traces,
                        predictions,
                        second_moments,
                        outcomes,
                        tau_squared,
                        horizon,
                    ),
                }
            )
        for alpha in PROPENSITY_TEMPERATURES:
            temperature_rows[alpha].append(
                {
                    **base,
                    **episode_estimates(
                        episode_traces,
                        predictions,
                        second_moments,
                        outcomes,
                        tau_squared,
                        int(config["search"]["maximum_rounds"]),
                        propensity_alpha=alpha,
                    ),
                }
            )
    horizon_output = {}
    for horizon, rows in horizon_rows.items():
        mean_variance = float(
            np.mean([float(row["predicted_variance"]) for row in rows])
        )
        cohort_lambda = float(tau_squared / (tau_squared + mean_variance))
        for row in rows:
            row["cohort_risk_shrinkage"] = float(
                np.clip(
                    float(row["outcome_regression"])
                    + cohort_lambda * float(row["unclipped_correction"]),
                    -1.0,
                    1.0,
                )
            )
        methods = [
            "naive",
            "outcome_regression",
            "hajek_dr",
            "sequential_dr",
            "cess",
            "cohort_risk_shrinkage",
        ]
        methods += [f"clipped_dr_{cap:g}" for cap in CLIP_CAPS]
        methods += [f"fixed_lambda_{value:g}" for value in FIXED_LAMBDAS]
        horizon_output[str(horizon)] = {
            "cells": len(rows),
            "mae": {method: mae(rows, method) for method in methods},
            "cess_vs_naive": bootstrap_gain(rows, "cess", "naive", replicates, seed + horizon),
            "cohort_risk_vs_cess": bootstrap_gain(
                rows,
                "cohort_risk_shrinkage",
                "cess",
                replicates,
                seed + 100 + horizon,
            ),
            "cohort_risk_lambda": cohort_lambda,
            "mean_shrinkage": float(np.mean([row["shrinkage"] for row in rows])),
            "minimum_probability_quantiles": {
                str(q): float(np.quantile([row["minimum_probability"] for row in rows], q))
                for q in (0.01, 0.05, 0.5)
            },
        }
    temperature_output = {}
    for alpha, rows in temperature_rows.items():
        temperature_output[str(alpha)] = {
            "mae": {method: mae(rows, method) for method in ("naive", "sequential_dr", "cess")},
            "cess_vs_naive": bootstrap_gain(rows, "cess", "naive", replicates, seed + int(alpha * 100)),
            "mean_shrinkage": float(np.mean([row["shrinkage"] for row in rows])),
        }
    positivity = {}
    for controller in config["search"]["controllers"]:
        subset = [
            row
            for row in cells
            if row["stopping_regime"] == "fixed_horizon" and row["controller"] == controller
        ]
        values = np.asarray([float(row["minimum_selection_probability"]) for row in subset])
        positivity[controller] = {
            "cells": len(subset),
            "minimum_probability_quantiles": {
                str(q): float(np.quantile(values, q)) for q in (0.01, 0.05, 0.5)
            },
            "mean_predicted_variance": float(
                np.mean([float(row["predicted_correction_variance"]) for row in subset])
            ),
            "mean_shrinkage": float(
                np.mean([float(row["preoutcome_shrinkage"]) for row in subset])
            ),
        }
    return {
        "model_key": model_key,
        "scope": "post_lock_diagnostic_reuse_of_confirmatory_traces",
        "horizon": horizon_output,
        "propensity_temperature": temperature_output,
        "positivity": positivity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists: {args.output}")
    config = load_config(args.config)
    output = {
        "protocol_version": config["protocol_version"],
        "status": "post_lock_diagnostic_not_confirmatory",
        "models": {},
    }
    for model_key in MODEL_KEYS:
        result_dir = ROOT / f"evidence_selection_benchmark/results/ms2_external/sequential_confirmatory_v1_{model_key}"
        output["models"][model_key] = analyze_model(
            model_key, config, args.environment, args.development, result_dir
        )
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
