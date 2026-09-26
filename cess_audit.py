"""Minimal CPU implementation of the sequential evidence audit.

The public release intentionally contains post-processing only. It does not
download models, call search APIs, or include private candidate documents.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def estimate_episode(
    episode: dict[str, Any],
    *,
    horizon: int | None = None,
    shrinkage: float = 0.5,
    correct_stopping: bool = True,
) -> dict[str, float]:
    """Estimate one trajectory using its logged selection probabilities.

    Required search fields are ``selected_outcome``, ``selected_probability``,
    ``candidate_count`` (or ``candidates``), and
    ``selected_outcome_prediction``. ``outcome_regression`` may be supplied at
    the episode level; otherwise it is averaged from the first candidate list.
    """
    searches = episode["searches"][:horizon]
    if not searches:
        raise ValueError("episode has no search observations")

    if "outcome_regression" in episode:
        anchor = float(episode["outcome_regression"])
    else:
        candidates = searches[0].get("candidates", [])
        predictions = [float(row["outcome_prediction"]) for row in candidates]
        if not predictions:
            raise ValueError("provide outcome_regression or first-round candidates")
        anchor = fmean(predictions)

    outcomes = []
    ipw_terms = []
    residual_terms = []
    for search in searches:
        n = float(search.get("candidate_count", len(search.get("candidates", []))))
        p = float(search["selected_probability"])
        continuation = float(search.get("continuation_probability", 1.0))
        if n <= 0 or p <= 0 or (correct_stopping and continuation <= 0):
            raise ValueError("candidate_count and logged probabilities must be positive")
        denominator = n * p * (continuation if correct_stopping else 1.0)
        y = float(search["selected_outcome"])
        m_t = float(search["selected_outcome_prediction"])
        outcomes.append(y)
        ipw_terms.append(y / denominator)
        residual_terms.append((y - m_t) / denominator)

    opened_mean = fmean(outcomes)
    sequential_ipw = clip(fmean(ipw_terms))
    dr_raw = anchor + fmean(residual_terms)
    sequential_dr = clip(dr_raw)
    cess = clip(anchor + shrinkage * (sequential_dr - anchor))
    return {
        "opened_mean": opened_mean,
        "outcome_regression": clip(anchor),
        "sequential_ipw": sequential_ipw,
        "sequential_dr": sequential_dr,
        "cess": cess,
        "dr_raw": dr_raw,
    }


def evaluate(
    episodes: Iterable[dict[str, Any]],
    *,
    horizon: int | None = None,
    shrinkage: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-episode estimates and task-level summary metrics."""
    rows = []
    for episode in episodes:
        estimates = estimate_episode(
            episode, horizon=horizon, shrinkage=shrinkage
        )
        target = float(episode["target"])
        rows.append(
            {
                "task_id": episode["task_id"],
                "ranking": episode.get("ranking", "unspecified"),
                "target": target,
                **estimates,
            }
        )

    methods = ("opened_mean", "outcome_regression", "sequential_ipw", "sequential_dr", "cess")
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "horizon": horizon,
        "shrinkage": shrinkage,
        "metrics": {},
    }
    for method in methods:
        summary["metrics"][method] = {
            "mae": fmean(abs(row[method] - row["target"]) for row in rows),
        }

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for method in methods:
        gaps = []
        for task_rows in by_task.values():
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in task_rows:
                grouped[row["ranking"]].append(row[method])
            if "supporting_first" in grouped and "opposing_first" in grouped:
                gaps.append(abs(fmean(grouped["supporting_first"]) - fmean(grouped["opposing_first"])))
        summary["metrics"][method]["ranking_sensitivity"] = fmean(gaps) if gaps else None
    return rows, summary
