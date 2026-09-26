#!/usr/bin/env python3
"""Run the CPU-only logged-evidence audit on a JSONL episode file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bootstrap import bootstrap_ci, task_mae, task_ranking_sensitivity
from cess_audit import METHODS, evaluate


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument(
        "--global-mean",
        type=float,
        default=None,
        help="Optional development-set anchor supplied by the caller; never inferred from test targets.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    rows, summary = evaluate(
        read_jsonl(args.input),
        horizon=args.horizon,
        shrinkage=args.shrinkage,
        global_mean=args.global_mean,
    )
    methods = [name for name in METHODS if name in rows[0]]
    summary["bootstrap"] = {}
    for method in methods:
        summary["bootstrap"][method] = {
            "mae": bootstrap_ci(
                rows,
                lambda sampled, method=method: task_mae(sampled, method),
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            ),
        }
        try:
            summary["bootstrap"][method]["ranking_sensitivity"] = bootstrap_ci(
                rows,
                lambda sampled, method=method: task_ranking_sensitivity(sampled, method),
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
        except ValueError:
            summary["bootstrap"][method]["ranking_sensitivity"] = None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "estimates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
