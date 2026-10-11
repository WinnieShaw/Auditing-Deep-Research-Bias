#!/usr/bin/env python3
"""Analyze the prospectively frozen global-shrinkage reconfirmation."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, load_jsonl
from scripts.build_ms2_external_manifest import sha256_file

DEFAULT_CONFIG = (
    ROOT / "configs/ms2_sequential_global_shrinkage_confirmatory_v2.yaml"
)
METHODS = (
    "naive_opened_mean",
    "global_outcome_regression",
    "sequential_ipw",
    "sequential_dr",
    "preoutcome_cess",
    "preoutcome_global_cess",
)


def by_review(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["review_id"])].append(row)
    return output


def mae(rows: list[dict[str, Any]], method: str) -> float:
    return float(
        np.mean(
            [
                abs(float(row[method]) - float(row["target"]))
                for row in rows
            ]
        )
    )


def relative_gain(
    rows: list[dict[str, Any]], proposed: str, reference: str
) -> float:
    return float(1.0 - mae(rows, proposed) / mae(rows, reference))


def rank_gap(rows: list[dict[str, Any]], method: str) -> float:
    means = {}
    for condition in ("pro_increase", "pro_decrease"):
        subset = [row for row in rows if row["rank_condition"] == condition]
        means[condition] = float(
            np.mean(
                [
                    float(row[method]) - float(row["target"])
                    for row in subset
                ]
            )
        )
    return means["pro_increase"] - means["pro_decrease"]


def review_bootstrap(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped = by_review(rows)
    review_ids = sorted(grouped)
    point = float(statistic(rows))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = rng.choice(review_ids, len(review_ids), replace=True)
        sample = [row for review_id in sampled for row in grouped[review_id]]
        draws.append(float(statistic(sample)))
    return {
        "value": point,
        "95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


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
    expected = int(config["cohort"]["tasks"]) * len(
        config["search"]["rank_conditions"]
    )
    if len(cells) != expected:
        raise ValueError(f"expected {expected} cells, found {len(cells)}")
    if {row["model_key"] for row in cells} != {args.model_key}:
        raise ValueError("model key mismatch")
    if any(row["preoutcome_global_cess"] is None for row in cells):
        raise ValueError("global CESS is missing")
    if {row["controller"] for row in cells} != {"uniform"}:
        raise ValueError("v2 must use only uniform protected exploration")
    if {row["stopping_regime"] for row in cells} != {"fixed_horizon"}:
        raise ValueError("v2 must use only the fixed horizon")

    replicates = int(config["evaluation"]["bootstrap_replicates"])
    seed = int(config["evaluation"]["bootstrap_seed"])
    parse_rate = float(np.mean([bool(row["parsed"]) for row in traces]))
    metrics = {
        method: {"mae": mae(cells, method)}
        for method in METHODS
    }
    comparisons = {
        "global_vs_naive": review_bootstrap(
            cells,
            lambda rows: relative_gain(
                rows, "preoutcome_global_cess", "naive_opened_mean"
            ),
            replicates,
            seed,
        ),
        "global_vs_adaptive_cess": review_bootstrap(
            cells,
            lambda rows: relative_gain(
                rows, "preoutcome_global_cess", "preoutcome_cess"
            ),
            replicates,
            seed + 1,
        ),
        "naive_rank_gap": review_bootstrap(
            cells,
            lambda rows: rank_gap(rows, "naive_opened_mean"),
            replicates,
            seed + 2,
        ),
        "global_rank_gap": review_bootstrap(
            cells,
            lambda rows: rank_gap(rows, "preoutcome_global_cess"),
            replicates,
            seed + 3,
        ),
    }
    gate = config["gate"]
    gates = {
        "query_parse_rate": {
            "value": parse_rate,
            "threshold": float(gate["minimum_query_parse_rate"]),
            "pass": parse_rate >= float(gate["minimum_query_parse_rate"]),
        },
        "global_gain_over_naive": {
            **comparisons["global_vs_naive"],
            "threshold": float(gate["minimum_global_gain_over_naive"]),
            "pass": comparisons["global_vs_naive"]["value"]
            >= float(gate["minimum_global_gain_over_naive"]),
        },
        "global_gain_over_adaptive_cess": {
            **comparisons["global_vs_adaptive_cess"],
            "threshold": float(
                gate["minimum_global_gain_over_adaptive_cess"]
            ),
            "pass": comparisons["global_vs_adaptive_cess"]["value"]
            >= float(gate["minimum_global_gain_over_adaptive_cess"]),
        },
        "rank_gap_reduction": {
            "naive_absolute": abs(comparisons["naive_rank_gap"]["value"]),
            "global_absolute": abs(comparisons["global_rank_gap"]["value"]),
            "pass": abs(comparisons["global_rank_gap"]["value"])
            < abs(comparisons["naive_rank_gap"]["value"]),
        },
    }
    output = {
        "protocol_version": config["protocol_version"],
        "scope": "prospective_global_shrinkage_reconfirmation",
        "model_key": args.model_key,
        "tasks": len({row["review_id"] for row in cells}),
        "cells": len(cells),
        "query_parse_rate": parse_rate,
        "frozen_global_shrinkage": float(
            config["models"][args.model_key]["frozen_global_shrinkage"]
        ),
        "metrics": metrics,
        "comparisons": comparisons,
        "gates": gates,
        "model_gate_pass": all(item["pass"] for item in gates.values()),
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
