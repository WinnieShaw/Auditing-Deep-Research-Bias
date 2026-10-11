#!/usr/bin/env python3
"""Summarize paired public-agent-light v4 trajectories without model calls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


RANKS = ("supporting_first", "opposing_first")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_distance(left: str, right: str):
    a, b = tokens(left), tokens(right)
    return 1.0 - len(a & b) / len(a | b) if a | b else 0.0


def set_jaccard(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return index
    return None


def opened_mean(episode, k):
    return mean(float(row["selected_outcome"]) for row in episode["searches"][:k])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    episodes = read_jsonl(args.episodes)
    paired = defaultdict(dict)
    for episode in episodes:
        paired[(episode["review_id"], int(episode["seed"]))][episode["ranking"]] = episode

    rows = []
    for (review_id, seed), pair in sorted(paired.items()):
        if set(pair) != set(RANKS):
            raise RuntimeError(f"incomplete ranking pair: {review_id}, {seed}, {sorted(pair)}")
        supporting, opposing = (pair[name] for name in RANKS)
        q_support, q_oppose = supporting["queries"], opposing["queries"]
        d_support, d_oppose = supporting["selected_document_ids"], opposing["selected_document_ids"]
        if len(q_support) != 4 or len(q_oppose) != 4 or len(d_support) != 4 or len(d_oppose) != 4:
            raise RuntimeError(f"non-four-round trajectory: {review_id}, {seed}")
        first_query = first_difference(q_support, q_oppose)
        first_document = first_difference(d_support, d_oppose)
        row = {
            "review_id": review_id,
            "seed": seed,
            "shared_first_query": q_support[0] == q_oppose[0],
            "query_diverged_by_k4": first_query is not None,
            "first_query_divergence_round": first_query,
            "mean_query_token_jaccard_distance": mean(
                jaccard_distance(a, b) for a, b in zip(q_support, q_oppose)
            ),
            "document_sequence_diverged_by_k4": first_document is not None,
            "first_document_divergence_round": first_document,
            "position_document_overlap_k2": mean(
                d_support[i] == d_oppose[i] for i in range(2)
            ),
            "position_document_overlap_k4": mean(
                d_support[i] == d_oppose[i] for i in range(4)
            ),
            "set_document_jaccard_k2": set_jaccard(d_support[:2], d_oppose[:2]),
            "set_document_jaccard_k4": set_jaccard(d_support, d_oppose),
            "opened_evidence_abs_gap_k1": abs(opened_mean(supporting, 1) - opened_mean(opposing, 1)),
            "opened_evidence_abs_gap_k2": abs(opened_mean(supporting, 2) - opened_mean(opposing, 2)),
            "opened_evidence_abs_gap_k4": abs(opened_mean(supporting, 4) - opened_mean(opposing, 4)),
            "supporting_unique_documents_k4": len(set(d_support)),
            "opposing_unique_documents_k4": len(set(d_oppose)),
            "supporting_final_report_saved": bool(supporting.get("final_report")),
            "opposing_final_report_saved": bool(opposing.get("final_report")),
        }
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [
        "shared_first_query",
        "query_diverged_by_k4",
        "mean_query_token_jaccard_distance",
        "document_sequence_diverged_by_k4",
        "position_document_overlap_k2",
        "position_document_overlap_k4",
        "set_document_jaccard_k2",
        "set_document_jaccard_k4",
        "opened_evidence_abs_gap_k1",
        "opened_evidence_abs_gap_k2",
        "opened_evidence_abs_gap_k4",
        "supporting_unique_documents_k4",
        "opposing_unique_documents_k4",
        "supporting_final_report_saved",
        "opposing_final_report_saved",
    ]
    summary = {
        "paired_task_seed_units": len(rows),
        "definitions": {
            "query_distance": "mean round-wise token-set Jaccard distance; descriptive, not an embedding metric",
            "position_document_overlap": "fraction of aligned rounds selecting the same document",
            "set_document_jaccard": "Jaccard overlap of selected-document ID sets",
            "opened_evidence_abs_gap": "absolute difference between ranking conditions in mean selected private evidence score",
            "missing_first_divergence_round": "no divergence through round four",
            "report_scope": "final reports are saved but not scored; opened-evidence balance is not called report-level balance",
        },
        "means": {key: mean(float(row[key]) for row in rows) for key in numeric_keys},
        "first_query_divergence_round_counts": dict(
            sorted(Counter(str(row["first_query_divergence_round"] or "none") for row in rows).items())
        ),
        "first_document_divergence_round_counts": dict(
            sorted(Counter(str(row["first_document_divergence_round"] or "none") for row in rows).items())
        ),
    }
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
