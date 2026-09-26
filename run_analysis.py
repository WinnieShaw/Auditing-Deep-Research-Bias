#!/usr/bin/env python3
"""Run the minimal CESS audit on a JSONL episode file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cess_audit import evaluate


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    args = parser.parse_args()

    rows, summary = evaluate(
        read_jsonl(args.input), horizon=args.horizon, shrinkage=args.shrinkage
    )
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
