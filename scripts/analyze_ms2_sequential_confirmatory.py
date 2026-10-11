#!/usr/bin/env python3
"""Analyze prospective sequential CESS with review-cluster inference."""
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
from scripts.build_ms2_external_manifest import sha256_file

DEFAULT_CONFIG = ROOT / "configs/ms2_sequential_search_confirmatory_v1.yaml"
METHODS = (
    "naive_opened_mean",
    "global_outcome_regression",
    "sequential_ipw",
    "sequential_dr",
    "preoutcome_cess",
)


def direction(value: float, tolerance: float) -> int:
    return 1 if value > tolerance else (-1 if value < -tolerance else 0)


def metric_summary(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    output: dict[str, Any] = {
        "cells": len(rows),
        "tasks": len({row["review_id"] for row in rows}),
    }
    for method in METHODS:
        errors = np.asarray(
            [float(row[method]) - float(row["target"]) for row in rows]
        )
        output[method] = {
            "mae": float(np.mean(np.abs(errors))),
            "signed_bias": float(np.mean(errors)),
            "wrong_direction_rate": float(
                np.mean(
                    [
                        direction(float(row[method]), tolerance)
                        != direction(float(row["target"]), tolerance)
                        for row in rows
                    ]
                )
            ),
        }
    output["trajectory"] = {
        "mean_rounds": float(np.mean([row["rounds_observed"] for row in rows])),
        "mean_unique_documents": float(
            np.mean([row["unique_documents_opened"] for row in rows])
        ),
        "mean_shrinkage": float(
            np.mean([row["preoutcome_shrinkage"] for row in rows])
        ),
        "mean_predicted_variance": float(
            np.mean([row["predicted_correction_variance"] for row in rows])
        ),
    }
    return output


def by_review(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["review_id"]].append(row)
    return grouped


def review_bootstrap(
    rows: list[dict[str, Any]], statistic, replicates: int, seed: int
) -> tuple[float, list[float]]:
    grouped = by_review(rows)
    review_ids = sorted(grouped)
    point = float(statistic([row for key in review_ids for row in grouped[key]]))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sample_ids = rng.choice(review_ids, len(review_ids), replace=True)
        sample = [row for key in sample_ids for row in grouped[key]]
        draws.append(float(statistic(sample)))
    return point, [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def relative_gain(rows: list[dict[str, Any]], proposed: str, reference: str) -> float:
    proposed_mae = np.mean(
        [abs(float(row[proposed]) - float(row["target"])) for row in rows]
    )
    reference_mae = np.mean(
        [abs(float(row[reference]) - float(row["target"])) for row in rows]
    )
    return float(1.0 - proposed_mae / reference_mae)


def paired_design_gain(
    rows: list[dict[str, Any]],
    proposed_controller: str,
    reference_controller: str,
    method: str,
) -> float:
    proposed = [
        row for row in rows if row["controller"] == proposed_controller
    ]
    reference = [
        row for row in rows if row["controller"] == reference_controller
    ]
    return relative_gain_between_rows(proposed, reference, method)


def relative_gain_between_rows(
    proposed: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    method: str,
) -> float:
    proposed_mae = np.mean(
        [abs(float(row[method]) - float(row["target"])) for row in proposed]
    )
    reference_mae = np.mean(
        [abs(float(row[method]) - float(row["target"])) for row in reference]
    )
    return float(1.0 - proposed_mae / reference_mae)


def rank_gap(rows: list[dict[str, Any]], method: str) -> float:
    means = {}
    for condition in ("pro_increase", "pro_decrease"):
        subset = [row for row in rows if row["rank_condition"] == condition]
        means[condition] = float(
            np.mean([float(row[method]) - float(row["target"]) for row in subset])
        )
    return means["pro_increase"] - means["pro_decrease"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"analysis output exists: {args.output}")
    config = load_config(args.config)
    cells = load_jsonl(args.cells)
    traces = load_jsonl(args.traces)
    if {row["model_key"] for row in cells} != {args.model_key}:
        raise ValueError("model key mismatch")
    tolerance = float(config["evaluation"]["direction_tolerance"])
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    parsed_rate = float(np.mean([bool(row["parsed"]) for row in traces]))

    fixed = [row for row in cells if row["stopping_regime"] == "fixed_horizon"]
    active = [row for row in fixed if row["controller"] == "active_cess"]
    uniform = [row for row in fixed if row["controller"] == "uniform"]
    agent = [row for row in fixed if row["controller"] == "agent"]
    expected = 200 * 3
    if not all(len(rows) == expected for rows in (active, uniform, agent)):
        raise ValueError("primary fixed-horizon design is incomplete")

    subsets = {
        "overall": cells,
        "primary_fixed": fixed,
        **{
            f"controller_{controller}": [
                row for row in cells if row["controller"] == controller
            ]
            for controller in config["search"]["controllers"]
        },
        **{
            f"stop_{regime}": [
                row for row in cells if row["stopping_regime"] == regime
            ]
            for regime in config["search"]["stopping_regimes"]
        },
    }
    metrics = {
        name: metric_summary(rows, tolerance) for name, rows in subsets.items()
    }

    active_gain, active_gain_ci = review_bootstrap(
        active,
        lambda rows: relative_gain(rows, "preoutcome_cess", "naive_opened_mean"),
        replicates,
        seed,
    )
    active_uniform_gain, active_uniform_ci = review_bootstrap(
        fixed,
        lambda rows: paired_design_gain(
            rows, "active_cess", "uniform", "preoutcome_cess"
        ),
        replicates,
        seed + 1,
    )
    agent_gap, agent_gap_ci = review_bootstrap(
        agent,
        lambda rows: rank_gap(rows, "naive_opened_mean"),
        replicates,
        seed + 2,
    )
    active_naive_gap, active_naive_gap_ci = review_bootstrap(
        active,
        lambda rows: rank_gap(rows, "naive_opened_mean"),
        replicates,
        seed + 3,
    )
    active_cess_gap, active_cess_gap_ci = review_bootstrap(
        active,
        lambda rows: rank_gap(rows, "preoutcome_cess"),
        replicates,
        seed + 4,
    )
    comparisons = {
        "active_cess_vs_active_naive": {
            "relative_mae_gain": active_gain,
            "95_ci": active_gain_ci,
        },
        "active_cess_vs_uniform_cess": {
            "relative_mae_gain": active_uniform_gain,
            "95_ci": active_uniform_ci,
        },
        "agent_naive_rank_gap": {"value": agent_gap, "95_ci": agent_gap_ci},
        "active_naive_rank_gap": {
            "value": active_naive_gap,
            "95_ci": active_naive_gap_ci,
        },
        "active_cess_rank_gap": {
            "value": active_cess_gap,
            "95_ci": active_cess_gap_ci,
        },
    }
    gate = config["gate"]
    gates = {
        "query_parse_rate": {
            "value": parsed_rate,
            "threshold": float(gate["minimum_query_parse_rate"]),
            "pass": parsed_rate >= float(gate["minimum_query_parse_rate"]),
        },
        "agent_naive_rank_gap": {
            "value": agent_gap,
            "threshold": float(gate["minimum_agent_naive_rank_gap"]),
            "pass": agent_gap >= float(gate["minimum_agent_naive_rank_gap"]),
        },
        "active_cess_gain": {
            "value": active_gain,
            "threshold": float(
                gate["minimum_active_cess_gain_over_active_naive"]
            ),
            "pass": active_gain
            >= float(gate["minimum_active_cess_gain_over_active_naive"]),
        },
        "active_vs_uniform_nonharm": {
            "value": active_uniform_gain,
            "threshold": -float(gate["maximum_active_harm_vs_uniform_cess"]),
            "pass": active_uniform_gain
            >= -float(gate["maximum_active_harm_vs_uniform_cess"]),
        },
        "rank_gap_reduction": {
            "naive_absolute": abs(active_naive_gap),
            "cess_absolute": abs(active_cess_gap),
            "pass": abs(active_cess_gap) < abs(active_naive_gap),
        },
    }
    output = {
        "protocol_version": config["protocol_version"],
        "scope": "prospective_sequential_confirmatory_analysis",
        "model_key": args.model_key,
        "query_parse_rate": parsed_rate,
        "metrics": metrics,
        "comparisons": comparisons,
        "gates": gates,
        "model_gate_pass": all(value["pass"] for value in gates.values()),
        "input_hashes": {
            "config": sha256_file(args.config),
            "cells": sha256_file(args.cells),
            "traces": sha256_file(args.traces),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
