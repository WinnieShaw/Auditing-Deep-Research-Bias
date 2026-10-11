#!/usr/bin/env python3
"""CPU-only submission check for the original MS2 Table 1 trajectories.

This script is intentionally post-hoc and read-only with respect to all input
artifacts.  It reproduces the Table 1 cell-level MAEs, adds development-tuned
simple shrinkage baselines on the same v2 test trajectories, and reports both
the signed aggregate rank gap and the absolute within-review paired rank gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DEFAULT = ROOT / "analysis/submission_final_checks_v3"
MODELS = ("qwen2_5_32b", "olmo3_7b")
RANKS = ("neutral", "pro_increase", "pro_decrease")
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026092211
TOLERANCE = 0.05
TABLE1_FIELDS = {
    "Opened Mean": "naive_opened_mean",
    "Outcome Regression": "global_outcome_regression",
    "Sequential IPW": "sequential_ipw",
    "Sequential DR": "sequential_dr",
    "Task-wise CESS": "preoutcome_cess",
    "CESS": "preoutcome_global_cess",
}
TABLE1_EXPECTED = {
    "qwen2_5_32b": {
        "Opened Mean": 0.1780,
        "Outcome Regression": 0.2529,
        "Sequential IPW": 0.1885,
        "Sequential DR": 0.1909,
        "Task-wise CESS": 0.1724,
        "CESS": 0.1609,
    },
    "olmo3_7b": {
        "Opened Mean": 0.1706,
        "Outcome Regression": 0.2612,
        "Sequential IPW": 0.1881,
        "Sequential DR": 0.1876,
        "Task-wise CESS": 0.1621,
        "CESS": 0.1557,
    },
}
FROZEN_GLOBAL_SHRINKAGE = {
    "qwen2_5_32b": 0.5484175425465773,
    "olmo3_7b": 0.5649384082205048,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), -1.0, 1.0)


def task_values(
    rows: list[dict[str, Any]], estimate: np.ndarray
) -> dict[str, dict[str, float]]:
    """Return equal-weighted source-review values for all requested metrics.

    Replicates are averaged within review/rank before rank sensitivity is
    calculated.  This makes the cluster bootstrap source-review based and
    prevents a future multi-seed run from changing the estimand by replication.
    """
    buckets: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, value in zip(rows, estimate):
        buckets[str(row["review_id"])].append((row, float(value)))
    output: dict[str, dict[str, float]] = {}
    for review_id, entries in buckets.items():
        abs_errors = [abs(value - float(row["target"])) for row, value in entries]
        by_rank: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row, value in entries:
            by_rank[str(row["rank_condition"])].append(
                (value, float(row["target"]))
            )
        if not {"pro_increase", "pro_decrease"}.issubset(by_rank):
            raise ValueError(f"missing opposing ranking pair for review {review_id}")
        rank_estimate = {
            rank: float(np.mean([value for value, _ in values]))
            for rank, values in by_rank.items()
        }
        rank_target = {
            rank: float(np.mean([target for _, target in values]))
            for rank, values in by_rank.items()
        }
        signed_gap = (
            (rank_estimate["pro_increase"] - rank_target["pro_increase"])
            - (rank_estimate["pro_decrease"] - rank_target["pro_decrease"])
        )
        output[review_id] = {
            "mae": float(np.mean(abs_errors)),
            "signed_rank_error_gap": float(signed_gap),
            "absolute_paired_ranking_sensitivity": float(
                abs(rank_estimate["pro_increase"] - rank_estimate["pro_decrease"])
            ),
            "rank_pair_count": 1.0,
            "trajectory_count": float(len(entries)),
        }
    return output


def metric_summary(per_review: dict[str, dict[str, float]]) -> dict[str, float]:
    review_ids = sorted(per_review)
    return {
        "source_tasks": len(review_ids),
        "task_macro_mae": float(np.mean([per_review[key]["mae"] for key in review_ids])),
        "signed_rank_error_gap": float(
            np.mean([per_review[key]["signed_rank_error_gap"] for key in review_ids])
        ),
        "absolute_paired_ranking_sensitivity": float(
            np.mean(
                [
                    per_review[key]["absolute_paired_ranking_sensitivity"]
                    for key in review_ids
                ]
            )
        ),
        "effective_opposing_rank_pairs": int(
            sum(per_review[key]["rank_pair_count"] for key in review_ids)
        ),
        "trajectories": int(sum(per_review[key]["trajectory_count"] for key in review_ids)),
    }


def model_task_overall(
    values_by_model: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Average model values within a source review for a paired overall metric."""
    common = set.intersection(*(set(values_by_model[model]) for model in MODELS))
    if len(common) != 200:
        raise ValueError(f"expected 200 source reviews common to both models, got {len(common)}")
    result: dict[str, dict[str, float]] = {}
    for review_id in sorted(common):
        result[review_id] = {
            metric: float(np.mean([values_by_model[model][review_id][metric] for model in MODELS]))
            for metric in (
                "mae",
                "signed_rank_error_gap",
                "absolute_paired_ranking_sensitivity",
                "rank_pair_count",
                "trajectory_count",
            )
        }
    return result


def tune_alpha(
    dev_rows: list[dict[str, Any]], anchor: Callable[[dict[str, Any]], float]
) -> tuple[float, float, list[dict[str, float]]]:
    opened = np.asarray([float(row["naive_opened_mean"]) for row in dev_rows])
    target = np.asarray([float(row["target"]) for row in dev_rows])
    anchor_values = np.asarray([anchor(row) for row in dev_rows])
    curve: list[dict[str, float]] = []
    for alpha in GRID:
        values = task_values(dev_rows, clip(opened + alpha * (anchor_values - opened)))
        curve.append({"alpha": float(alpha), "development_task_macro_mae": metric_summary(values)["task_macro_mae"]})
    minimum = min(item["development_task_macro_mae"] for item in curve)
    # Fixed tie rule: smallest alpha among numerical ties at double precision.
    selected = next(
        item for item in curve if np.isclose(item["development_task_macro_mae"], minimum, atol=1e-12, rtol=0.0)
    )
    return float(selected["alpha"]), float(selected["development_task_macro_mae"]), curve


def bootstrap_delta(
    proposed: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
    metric: str,
    seed: int,
) -> dict[str, float | int]:
    keys = sorted(set(proposed) & set(reference))
    if not keys:
        raise ValueError("no source-review overlap for paired bootstrap")
    difference = np.asarray(
        [proposed[key][metric] - reference[key][metric] for key in keys], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = difference[rng.integers(0, len(difference), size=(BOOTSTRAP_REPLICATES, len(difference)))].mean(axis=1)
    return {
        "source_tasks": len(difference),
        "delta_cess_minus_reference": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = [",".join(columns)]
    for row in rows:
        value = []
        for column in columns:
            raw = row.get(column, "")
            if isinstance(raw, str):
                value.append('"' + raw.replace('"', '""') + '"')
            else:
                value.append(str(raw))
        lines.append(",".join(value))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_table1_inputs(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row["model_key"] == model
        and row["controller"] == "uniform"
        and row["stopping_regime"] == "fixed_horizon"
    ]
    if len(subset) != 600:
        raise ValueError(f"{model}: expected 600 Table 1 cells, got {len(subset)}")
    if {row["rank_condition"] for row in subset} != set(RANKS):
        raise ValueError(f"{model}: Table 1 rank support is incomplete")
    if {int(row["rounds_observed"]) for row in subset} != {4}:
        raise ValueError(f"{model}: Table 1 must be a fixed four-round setting")
    if {int(row["replicate"]) for row in subset} != {0}:
        raise ValueError(f"{model}: unexpected replicate structure")
    if any(row["preoutcome_global_cess"] is None for row in subset):
        raise ValueError(f"{model}: frozen CESS field missing")
    coefficients = {float(row["preoutcome_global_shrinkage"]) for row in subset}
    if coefficients != {FROZEN_GLOBAL_SHRINKAGE[model]}:
        raise ValueError(
            f"{model}: logged frozen coefficient {coefficients} does not match config"
        )
    return {"rows": subset, "source_tasks": len({str(row["review_id"]) for row in subset})}


def overlap_audit(left_docs: Path, right_docs: Path) -> dict[str, Any]:
    left, right = load_jsonl(left_docs), load_jsonl(right_docs)
    left_review = {str(row["review_id"]) for row in left}
    right_review = {str(row["review_id"]) for row in right}
    left_id = {str(row["document_id"]) for row in left}
    right_id = {str(row["document_id"]) for row in right}
    left_pmid = {str(row["pmid"]) for row in left if row.get("pmid") is not None}
    right_pmid = {str(row["pmid"]) for row in right if row.get("pmid") is not None}
    return {
        "left_documents": len(left),
        "right_documents": len(right),
        "source_review_id_overlap": len(left_review & right_review),
        "namespaced_document_id_overlap": len(left_id & right_id),
        "underlying_pmid_overlap": len(left_pmid & right_pmid),
        "left_review_count": len(left_review),
        "right_review_count": len(right_review),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    expected_outputs = [
        "audit_report.md",
        "analysis_metadata.json",
        "table1_reproduction.csv",
        "development_tuning.csv",
        "method_metrics.csv",
        "cess_paired_contrasts.csv",
        "ranking_sensitivity_explanation.md",
        "implementation_audit.json",
    ]
    if any((out / name).exists() for name in expected_outputs):
        raise FileExistsError(f"refusing to overwrite existing analysis outputs in {out}")
    out.mkdir(parents=True, exist_ok=True)

    v1_cells = {
        model: ROOT / f"evidence_selection_benchmark/results/ms2_external/sequential_confirmatory_v1_{model}/cells.jsonl"
        for model in MODELS
    }
    v2_cells = {
        model: ROOT / f"evidence_selection_benchmark/results/ms2_external/sequential_global_shrinkage_confirmatory_v2_{model}/cells.jsonl"
        for model in MODELS
    }
    v1_docs = ROOT / "evidence_selection_benchmark/data/ms2_sequential_search_confirmatory_v1/documents.jsonl"
    v2_docs = ROOT / "evidence_selection_benchmark/data/ms2_sequential_global_shrinkage_confirmatory_v2/documents.jsonl"
    initial_dev_docs = ROOT / "evidence_selection_benchmark/data/ms2_external_v1/development/documents.jsonl"
    v2_config = ROOT / "configs/ms2_sequential_global_shrinkage_confirmatory_v2.yaml"

    dev_by_model: dict[str, list[dict[str, Any]]] = {}
    test_by_model: dict[str, list[dict[str, Any]]] = {}
    table_rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    tuning_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    observed_frozen_coefficients: dict[str, float] = {}
    formula_diagnostics: dict[str, dict[str, float]] = {}

    for model in MODELS:
        dev_all = load_jsonl(v1_cells[model])
        test_all = load_jsonl(v2_cells[model])
        dev = [
            row
            for row in dev_all
            if row["controller"] == "uniform"
            and row["stopping_regime"] == "fixed_horizon"
        ]
        test_audit = audit_table1_inputs(test_all, model)
        test = test_audit["rows"]
        observed_frozen_coefficients[model] = float(test[0]["preoutcome_global_shrinkage"])
        anchor_check = np.asarray([float(row["global_outcome_regression"]) for row in test])
        correction_check = np.asarray([float(row["unclipped_sequential_correction"]) for row in test])
        task_lambda_check = np.asarray([float(row["preoutcome_shrinkage"]) for row in test])
        global_lambda_check = np.asarray([float(row["preoutcome_global_shrinkage"]) for row in test])
        formula_diagnostics[model] = {
            "max_abs_sequential_dr_reconstruction_error": float(np.max(np.abs(
                clip(anchor_check + correction_check)
                - np.asarray([float(row["sequential_dr"]) for row in test])
            ))),
            "max_abs_task_wise_cess_reconstruction_error": float(np.max(np.abs(
                clip(anchor_check + task_lambda_check * correction_check)
                - np.asarray([float(row["preoutcome_cess"]) for row in test])
            ))),
            "max_abs_main_cess_reconstruction_error": float(np.max(np.abs(
                clip(anchor_check + global_lambda_check * correction_check)
                - np.asarray([float(row["preoutcome_global_cess"]) for row in test])
            ))),
        }
        if any(value > 1e-12 for value in formula_diagnostics[model].values()):
            raise ValueError(f"{model}: logged estimator equations do not reconstruct")
        if len(dev) != 600 or len({str(row["review_id"]) for row in dev}) != 200:
            raise ValueError(f"{model}: v1 development pool does not match the fixed uniform 200-review design")
        overlap = {str(row["review_id"]) for row in dev} & {str(row["review_id"]) for row in test}
        if overlap:
            raise ValueError(f"{model}: v1 development pool overlaps v2 test reviews: {len(overlap)}")
        dev_by_model[model], test_by_model[model] = dev, test

        for label, field in TABLE1_FIELDS.items():
            value = metric_summary(task_values(test, np.asarray([float(row[field]) for row in test])))
            expected = TABLE1_EXPECTED[model][label]
            table_rows.append(
                {
                    "query_model": model,
                    "method": label,
                    "field": field,
                    "recomputed_task_macro_mae": value["task_macro_mae"],
                    "table1_reported_mae": expected,
                    "difference_recomputed_minus_reported": value["task_macro_mae"] - expected,
                    "matches_four_decimal_display": round(value["task_macro_mae"], 4) == expected,
                    "source_tasks": value["source_tasks"],
                    "trajectories": value["trajectories"],
                }
            )

        review_target = {}
        for row in dev:
            old = review_target.setdefault(str(row["review_id"]), float(row["target"]))
            if not np.isclose(old, float(row["target"])):
                raise ValueError(f"{model}: nonconstant target within development review")
        dev_global_mean = float(np.mean(list(review_target.values())))
        opened_dev = np.asarray([float(row["naive_opened_mean"]) for row in dev])
        or_dev = np.asarray([float(row["global_outcome_regression"]) for row in dev])
        global_constant_values = np.full(len(dev), dev_global_mean)
        global_alpha, global_loss, global_curve = tune_alpha(dev, lambda _: dev_global_mean)
        or_alpha, or_loss, or_curve = tune_alpha(dev, lambda row: float(row["global_outcome_regression"]))
        for entry in global_curve:
            grid_rows.append({"query_model": model, "baseline": "Opened Mean -> Development Global Mean", **entry})
        for entry in or_curve:
            grid_rows.append({"query_model": model, "baseline": "Opened Mean -> Outcome Regression", **entry})
        tuning_rows.extend(
            [
                {
                    "query_model": model,
                    "baseline": "Development Global Mean",
                    "anchor": "equal-weighted v1 source-review target mean",
                    "alpha": "not_applicable",
                    "development_task_macro_mae": metric_summary(task_values(dev, global_constant_values))["task_macro_mae"],
                    "development_global_target_mean": dev_global_mean,
                    "development_source_tasks": len(review_target),
                },
                {
                    "query_model": model,
                    "baseline": "Opened Mean -> Development Global Mean",
                    "anchor": "equal-weighted v1 source-review target mean",
                    "alpha": global_alpha,
                    "development_task_macro_mae": global_loss,
                    "development_global_target_mean": dev_global_mean,
                    "development_source_tasks": len(review_target),
                },
                {
                    "query_model": model,
                    "baseline": "Opened Mean -> Outcome Regression",
                    "anchor": "per-trajectory global outcome regression",
                    "alpha": or_alpha,
                    "development_task_macro_mae": or_loss,
                    "development_global_target_mean": dev_global_mean,
                    "development_source_tasks": len(review_target),
                },
            ]
        )
        opened_test = np.asarray([float(row["naive_opened_mean"]) for row in test])
        or_test = np.asarray([float(row["global_outcome_regression"]) for row in test])
        estimates = {
            **{
                label: np.asarray([float(row[field]) for row in test])
                for label, field in TABLE1_FIELDS.items()
            },
            "Development Global Mean": np.full(len(test), dev_global_mean),
            "Opened Mean -> Development Global Mean": clip(
                opened_test + global_alpha * (dev_global_mean - opened_test)
            ),
            "Opened Mean -> Outcome Regression": clip(
                opened_test + or_alpha * (or_test - opened_test)
            ),
        }
        baseline_metrics[model] = {
            label: task_values(test, values) for label, values in estimates.items()
        }

    method_rows: list[dict[str, Any]] = []
    all_methods = list(TABLE1_FIELDS) + [
        "Development Global Mean",
        "Opened Mean -> Development Global Mean",
        "Opened Mean -> Outcome Regression",
    ]
    overall_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        overall_metrics[method] = model_task_overall(
            {model: baseline_metrics[model][method] for model in MODELS}
        )
    for method in all_methods:
        for model in MODELS:
            summary = metric_summary(baseline_metrics[model][method])
            method_rows.append(
                {
                    "query_model": model,
                    "method": method,
                    **summary,
                    "source_review_clusters": 200,
                    "model_task_units": 200,
                    "raw_test_trajectories": 600,
                    "rank_pairs_after_model_aggregation": summary["effective_opposing_rank_pairs"],
                    "effective_model_rank_pairs": 200,
                }
            )
        overall = metric_summary(overall_metrics[method])
        method_rows.append(
            {
                "query_model": "overall_equal_model_average",
                "method": method,
                **overall,
                "trajectories": 1200,
                "source_review_clusters": 200,
                "model_task_units": 400,
                "raw_test_trajectories": 1200,
                "rank_pairs_after_model_aggregation": 200,
                "effective_model_rank_pairs": 400,
                "note": "source-review bootstrap averages the two model-specific values within each of 200 shared reviews",
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    references = [
        "Opened Mean",
        "Development Global Mean",
        "Opened Mean -> Development Global Mean",
        "Opened Mean -> Outcome Regression",
    ]
    for model_index, model in enumerate(MODELS):
        for ref_index, reference in enumerate(references):
            for metric_index, metric in enumerate(
                ("mae", "signed_rank_error_gap", "absolute_paired_ranking_sensitivity")
            ):
                contrast_rows.append(
                    {
                        "query_model": model,
                        "candidate": "CESS",
                        "reference": reference,
                        "metric": metric,
                        "interpretation": "negative favors CESS for MAE and absolute sensitivity; signed gap is directional and can cancel",
                        **bootstrap_delta(
                            baseline_metrics[model]["CESS"],
                            baseline_metrics[model][reference],
                            metric,
                            BOOTSTRAP_SEED + 1000 * model_index + 100 * ref_index + metric_index,
                        ),
                    }
                )
    for ref_index, reference in enumerate(references):
        for metric_index, metric in enumerate(
            ("mae", "signed_rank_error_gap", "absolute_paired_ranking_sensitivity")
        ):
            contrast_rows.append(
                {
                    "query_model": "overall_equal_model_average",
                    "candidate": "CESS",
                    "reference": reference,
                    "metric": metric,
                    "interpretation": "negative favors CESS for MAE and absolute sensitivity; signed gap is directional and can cancel",
                    **bootstrap_delta(
                        overall_metrics["CESS"],
                        overall_metrics[reference],
                        metric,
                        BOOTSTRAP_SEED + 3000 + 100 * ref_index + metric_index,
                    ),
                }
            )

    isolation = {
        "new_simple_baseline_development_pool": {
            "cells": {model: len(dev_by_model[model]) for model in MODELS},
            "source_tasks": {model: len({str(row["review_id"]) for row in dev_by_model[model]}) for model in MODELS},
            "setting": "v1 fixed_horizon + uniform controller, four rounds, three ranking conditions",
            "outcomes_used_for_alpha_tuning": True,
        },
        "table1_test_pool": {
            "cells": {model: len(test_by_model[model]) for model in MODELS},
            "source_tasks": {model: len({str(row["review_id"]) for row in test_by_model[model]}) for model in MODELS},
            "setting": "v2 fixed_horizon + uniform controller, four rounds, one replicate, three ranking conditions",
        },
        "new_baseline_v1_vs_table1_v2": overlap_audit(v1_docs, v2_docs),
        "initial_nuisance_development_vs_table1_v2": overlap_audit(initial_dev_docs, v2_docs),
        "existing_main_cess_development_evidence": {
            "nuisance_outcome_regression_and_residual_second_moments": "initial 400-review ms2_external_v1/development pool; loaded by fit_prospective_models in scripts/run_ms2_sequential_confirmatory.py",
            "global_shrinkage_coefficients": "model-level 0.5484175425465773 (Qwen) and 0.5649384082205048 (OLMo); v2 report states these were derived from confirmation-v1 design information without reading v1 outcomes",
            "design_information_cohort": "sequential_confirmatory_v1; its review/document overlap with v2 is recorded above",
            "silent_retraining_or_replacement": False,
        },
    }

    implementation_audit = {
        "table1_input_cells": {
            "paths": {model: str(v2_cells[model].relative_to(ROOT)) for model in MODELS},
            "retained_fields": [
                "review_id", "rank_condition", "controller", "stopping_regime", "replicate",
                "target", "naive_opened_mean", "global_outcome_regression", "sequential_ipw",
                "sequential_dr", "preoutcome_cess", "preoutcome_global_cess",
                "preoutcome_shrinkage", "preoutcome_global_shrinkage",
                "unclipped_sequential_correction", "rounds_observed",
            ],
            "actual_setting": {
                "source_reviews": 200,
                "models": list(MODELS),
                "raw_trajectories": 1200,
                "ranking_conditions": list(RANKS),
                "maximum_rounds": 4,
                "stopping_rule": "fixed_horizon; every retained cell has four observed rounds",
                "controller": "uniform protected exploration",
                "replicate": 0,
                "config_seed": 2026091617,
            },
        },
        "estimator_implementation": {
            "code": "scripts/run_ms2_sequential_confirmatory.py:229-280",
            "outcome_regression": "b = mean of nuisance predictions over the review candidate pool",
            "sequential_correction": "c = (1/4) sum_t (y_t - m_t)/(N p_t s_t)",
            "sequential_ipw": "clip((1/4) sum_t y_t/(N p_t s_t), -1, 1)",
            "sequential_dr": "clip(b + c, -1, 1)",
            "task_wise_cess": "clip(b + lambda_task*c, -1, 1), where lambda_task is preoutcome_shrinkage from predicted correction variance",
            "main_cess": "clip(b + lambda_model*c, -1, 1), where lambda_model is preoutcome_global_shrinkage",
            "logged_frozen_model_coefficients": observed_frozen_coefficients,
            "config_frozen_model_coefficients": FROZEN_GLOBAL_SHRINKAGE,
            "cell_level_formula_reconstruction_max_abs_error": formula_diagnostics,
            "fixed_horizon_survival": "all retained Table 1 cells are fixed horizon, so the logged survival factor is 1 at every observed round",
        },
    }

    metadata = {
        "analysis": "submission-final-checks-v3",
        "scope": "CPU-only, post-hoc analysis of completed Table 1 trajectories",
        "no_model_calls": True,
        "no_raw_result_overwrite": True,
        "no_latex_modification": True,
        "bootstrap": {"cluster": "source review", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "alpha_grid": [float(value) for value in GRID],
        "alpha_tie_rule": "smallest alpha among equal task-macro MAE values",
        "test_input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in v2_cells.values()},
        "development_input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in v1_cells.values()},
        "config_hash": sha256(v2_config),
        "rank_sensitivity_definitions": {
            "signed_rank_error_gap": "mean_review[(estimate_pro_increase-target) - (estimate_pro_decrease-target)] after averaging replicate(s) within rank",
            "absolute_paired_ranking_sensitivity": "mean_review abs(mean_replicate estimate_pro_increase - mean_replicate estimate_pro_decrease)",
        },
        "overall_aggregation": "average model-specific review values within each of 200 shared source reviews; bootstrap those 200 review clusters",
    }

    write_csv(out / "table1_reproduction.csv", table_rows)
    write_csv(out / "development_tuning.csv", tuning_rows)
    write_csv(out / "development_alpha_grid.csv", grid_rows)
    write_csv(out / "method_metrics.csv", method_rows)
    write_csv(out / "cess_paired_contrasts.csv", contrast_rows)
    (out / "isolation_audit.json").write_text(json.dumps(isolation, indent=2) + "\n", encoding="utf-8")
    (out / "implementation_audit.json").write_text(json.dumps(implementation_audit, indent=2) + "\n", encoding="utf-8")
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    reproduction_failures = [row for row in table_rows if not row["matches_four_decimal_display"]]
    report = [
        "# Submission-final CPU checks: original MS2 Table 1",
        "",
        "This is a post-hoc CPU-only analysis of completed trajectories. It does not modify the paper, model calls, source trajectories, existing CESS, or existing output directories.",
        "",
        "## Table 1 reconstruction",
        "",
        f"The v2 Table 1 input has 1,200 cells: 200 source reviews × 3 ranking conditions × 2 query models. Every retained cell uses the uniform controller, a fixed four-round horizon, and replicate 0.",
        f"All six reported MAEs per model reproduce to the displayed four decimals: {'yes' if not reproduction_failures else 'no'}. Exact deviations are in `table1_reproduction.csv`.",
        "",
        "## Logged estimators",
        "",
        "The Table 1 input fields are `naive_opened_mean`, `global_outcome_regression`, `sequential_ipw`, `sequential_dr`, `preoutcome_cess`, and `preoutcome_global_cess`. In `scripts/run_ms2_sequential_confirmatory.py:238-279`, the outcome-regression anchor is the mean nuisance prediction across the full candidate pool. The correction is c=(1/4) sum_t (y_t-m_t)/(N p_t s_t). Task-wise CESS is clip(anchor + lambda_task c), with its logged task-specific `preoutcome_shrinkage`; main CESS is clip(anchor + lambda_model c), with `preoutcome_global_shrinkage`. The latter is exactly 0.5484175425465773 for Qwen and 0.5649384082205048 for OLMo in every Table 1 cell, matching the v2 configuration. The retained setting is fixed horizon, so s_t=1 on all observed rounds.",
        "",
        "## Added simple shrinkage baselines",
        "",
        "For each query model, the alpha grid {0, 0.05, ..., 1} is selected by equal-weighted source-review MAE on the v1 fixed-horizon, uniform-controller cohort, then applied once to the v2 Table 1 trajectories. The Development Global Mean is the equal-weighted mean of v1 review targets; it is never calculated from test targets. Ties use the smaller alpha.",
        "",
        "## Ranking sensitivity",
        "",
        "The signed rank-error gap is the average within-review difference of (estimate-target) under pro-increase minus pro-decrease ranking. It preserves direction, and opposite review-level shifts can cancel. Absolute paired ranking sensitivity averages abs(estimate_pro-increase-estimate_pro-decrease) within each review after averaging any replicate(s) inside a rank. It measures magnitude without cancellation. Both are reported in `method_metrics.csv`; all CESS-minus-reference source-review bootstrap intervals are in `cess_paired_contrasts.csv`.",
        "",
        "## Development/test isolation",
        "",
        "The isolation counts, including namespaced document IDs and underlying PMIDs when available, are in `isolation_audit.json`. The existing main CESS is left unchanged: Table 1 CESS remains the logged `preoutcome_global_cess` field with its original model-level frozen coefficients and outcome-regression anchor.",
    ]
    (out / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "ranking_sensitivity_explanation.md").write_text(
        "# Two ranking-sensitivity summaries\n\n"
        "**Signed rank-error gap** averages, review by review, the pro-increase minus pro-decrease difference of `(estimate - target)`. It retains direction. If the ranking change moves some reviews upward and others downward, these changes can cancel in the overall mean.\n\n"
        "**Absolute paired ranking sensitivity** first averages replicate(s) within each review/rank, computes `abs(estimate_pro_increase - estimate_pro_decrease)` for each review, then averages reviews. It measures the typical magnitude of a ranking-induced change and cannot cancel across opposite directions.\n\n"
        "The Table 1 design has one replicate per review/rank. The distinction still matters because the signed summary can be near zero while the absolute paired summary is large. Both use the same fixed methods, coefficients, and v2 test trajectories.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "outputs": expected_outputs + ["development_alpha_grid.csv", "isolation_audit.json"]}, indent=2))


if __name__ == "__main__":
    main()
