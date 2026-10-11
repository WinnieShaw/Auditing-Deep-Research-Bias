#!/usr/bin/env python3
"""Build label-value-free MS² development and sealed holdout manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal_search_credit.config import load_config
from causal_search_credit.util import atomic_write_json, stable_hash, write_jsonl

DEFAULT_ARCHIVE = (
    ROOT
    / "evidence_selection_benchmark/data/raw/ms2_original"
    / "ms_data_2021-04-12.zip"
)
DEFAULT_OUTPUT = ROOT / "evidence_selection_benchmark/data/ms2_external_v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_phrase(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def claim_key(significance: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_phrase(significance.get("intervention")),
        normalize_phrase(significance.get("outcome")),
    )


def canonical_references(review: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose one abstract-bearing report per included study without labels."""
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for study in review.get("included_studies") or []:
        candidates = [
            reference
            for reference in (study.get("references") or [])
            if str(reference.get("title") or "").strip()
            and str(reference.get("abstract") or "").strip()
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda reference: (
                str(reference.get("pmid") or ""),
                str(reference.get("s2id") or ""),
                str(reference.get("title") or ""),
            )
        )
        reference = candidates[0]
        identity = str(
            reference.get("pmid")
            or reference.get("s2id")
            or stable_hash(
                [
                    normalize_phrase(reference.get("title")),
                    normalize_phrase(reference.get("abstract")),
                ]
            )
        )
        if identity not in seen:
            chosen.append(reference)
            seen.add(identity)
    return chosen


def choose_claim(
    review_significances: Iterable[dict[str, Any]],
    references: list[dict[str, Any]],
) -> tuple[tuple[str, str] | None, int]:
    """Choose by label availability only, never by classification values."""
    candidate_keys = {
        claim_key(significance)
        for significance in review_significances
        if all(claim_key(significance))
    }
    if not candidate_keys:
        return None, 0
    document_keys = [
        {
            claim_key(significance)
            for significance in (reference.get("significances") or [])
            if all(claim_key(significance))
        }
        for reference in references
    ]
    coverage = {
        key: sum(key in keys for keys in document_keys) for key in candidate_keys
    }
    best = sorted(candidate_keys, key=lambda key: (-coverage[key], key))[0]
    return best, coverage[best]


def iter_reviews(
    archive: zipfile.ZipFile, member: str
) -> Iterable[dict[str, Any]]:
    with archive.open(member) as stream:
        for raw_line in stream:
            if raw_line.strip():
                yield json.loads(raw_line)


def eligible_rows(
    archive: zipfile.ZipFile,
    member: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = config["eligibility"]
    excluded = set(map(str, config["cohorts"]["excluded_review_ids"]))
    rows: list[dict[str, Any]] = []
    audit = Counter()
    document_counts: list[int] = []
    coverage_counts: list[int] = []

    for review in iter_reviews(archive, member):
        audit["source_reviews"] += 1
        review_id = str(review.get("docid") or review.get("s2id") or "")
        if review_id in excluded:
            audit["excluded_inspected"] += 1
            continue
        if rules["require_review_title"] and not str(
            review.get("title") or ""
        ).strip():
            audit["missing_review_title"] += 1
            continue
        references = canonical_references(review)
        claim, coverage = choose_claim(review.get("significances") or [], references)
        document_counts.append(len(references))
        coverage_counts.append(coverage)
        if claim is None:
            audit["missing_review_claim"] += 1
            continue
        if not (
            int(rules["minimum_candidate_documents"])
            <= len(references)
            <= int(rules["maximum_candidate_documents"])
        ):
            audit["outside_document_count"] += 1
            continue
        if coverage < int(rules["minimum_claim_matched_documents"]):
            audit["insufficient_claim_coverage"] += 1
            continue
        rows.append(
            {
                "review_id": review_id,
                "review_title": str(review["title"]).strip(),
                "candidate_documents": len(references),
                "claim_matched_documents": coverage,
                "target_intervention": claim[0],
                "target_outcome": claim[1],
                "source_member": member,
                "outcome_values_materialized": False,
                "expert_summary_materialized": False,
            }
        )
        audit["eligible_reviews"] += 1

    summary = dict(audit)
    summary["candidate_documents_min"] = min(document_counts, default=0)
    summary["candidate_documents_max"] = max(document_counts, default=0)
    summary["claim_coverage_min"] = min(coverage_counts, default=0)
    summary["claim_coverage_max"] = max(coverage_counts, default=0)
    return rows, summary


def select_by_hash(
    rows: list[dict[str, Any]], seed: int, cohort: str, count: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            stable_hash([seed, cohort, row["review_id"]]),
            row["review_id"],
        ),
    )
    if len(ordered) < count:
        raise ValueError(
            f"{cohort} requires {count} tasks but only {len(ordered)} are eligible"
        )
    selected = []
    for row in ordered[:count]:
        copy = dict(row)
        copy["cohort"] = cohort
        copy["selection_hash"] = stable_hash([seed, cohort, row["review_id"]])
        selected.append(copy)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ms2_external_v1.yaml"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)

    source_hash = sha256_file(args.archive)
    expected_hash = str(config["source"]["archive_sha256"])
    if source_hash != expected_hash:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected_hash}, got {source_hash}"
        )

    cohort_specs = (
        (
            "development",
            config["source"]["development_member"],
            int(config["cohorts"]["development_tasks"]),
        ),
        (
            "holdout",
            config["source"]["holdout_member"],
            int(config["cohorts"]["holdout_tasks"]),
        ),
    )
    audit: dict[str, Any] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(args.archive) as archive:
        for cohort, member, count in cohort_specs:
            rows, summary = eligible_rows(archive, member, config)
            manifests[cohort] = select_by_hash(
                rows, int(config["seed"]), cohort, count
            )
            summary["selected_reviews"] = count
            audit[cohort] = summary

    development_ids = {row["review_id"] for row in manifests["development"]}
    holdout_ids = {row["review_id"] for row in manifests["holdout"]}
    overlap = development_ids & holdout_ids
    if overlap:
        raise ValueError(f"development/holdout overlap: {sorted(overlap)[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for cohort, rows in manifests.items():
        path = args.output / f"{cohort}_manifest.jsonl"
        write_jsonl(path, rows)
        paths[cohort] = path

    summary = {
        "protocol_version": config["protocol_version"],
        "status": "covariates_and_claim_keys_only_outcomes_sealed",
        "source_archive": str(args.archive),
        "source_sha256": source_hash,
        "development_holdout_overlap": 0,
        "audits": audit,
        "manifests": {
            cohort: {
                "tasks": len(manifests[cohort]),
                "sha256": sha256_file(path),
                "path": str(path.relative_to(ROOT)),
            }
            for cohort, path in paths.items()
        },
    }
    atomic_write_json(args.output / "manifest_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
