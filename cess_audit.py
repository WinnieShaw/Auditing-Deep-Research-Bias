"""CPU estimators for a logged sequential evidence-selection audit.

The module is deliberately limited to post-processing. It never downloads
documents, calls a model, or infers a probability that was not logged by the
caller. The public functions are small enough to audit line by line.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import fmean
from typing import Any, Iterable, Mapping


METHODS = (
    "opened_mean",
    "outcome_regression",
    "opened_to_global",
    "opened_to_or",
    "sequential_ipw",
    "self_normalized_ipw",
    "sequential_dr",
    "self_normalized_dr",
    "dr_clip_5",
    "dr_clip_10",
    "dr_clip_20",
    "cess",
)


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clip an estimate to the outcome scale used by the public examples."""
    return max(lower, min(upper, float(value)))


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_episode(episode: Mapping[str, Any]) -> None:
    """Validate fields required by the estimators.

    A zero selection or continuation probability violates the support required
    by inverse-probability weighting and is rejected explicitly; no epsilon is
    silently added to make the estimate look identified.
    """
    for field in ("task_id", "target", "searches"):
        if field not in episode:
            raise ValueError(f"episode is missing required field: {field}")
    _finite(episode["target"], "target")
    searches = episode["searches"]
    if not isinstance(searches, list) or not searches:
        raise ValueError("searches must be a non-empty list")
    if "outcome_regression" in episode:
        _finite(episode["outcome_regression"], "outcome_regression")
    for index, search in enumerate(searches):
        prefix = f"searches[{index}]"
        for field in ("selected_outcome", "selected_probability", "selected_outcome_prediction"):
            if field not in search:
                raise ValueError(f"{prefix} is missing required field: {field}")
        n = search.get("candidate_count", len(search.get("candidates", [])))
        n_value = _finite(n, f"{prefix}.candidate_count")
        p_value = _finite(search["selected_probability"], f"{prefix}.selected_probability")
        c_value = _finite(search.get("continuation_probability", 1.0), f"{prefix}.continuation_probability")
        if n_value <= 0 or p_value <= 0 or p_value > 1 or c_value <= 0 or c_value > 1:
            raise ValueError(f"{prefix} has unsupported candidate or probability value")
        _finite(search["selected_outcome"], f"{prefix}.selected_outcome")
        _finite(search["selected_outcome_prediction"], f"{prefix}.selected_outcome_prediction")


def _anchor(episode: Mapping[str, Any], searches: list[Mapping[str, Any]]) -> float:
    if "outcome_regression" in episode:
        return _finite(episode["outcome_regression"], "outcome_regression")
    candidates = searches[0].get("candidates", [])
    predictions = [_finite(row["outcome_prediction"], "candidate.outcome_prediction") for row in candidates]
    if not predictions:
        raise ValueError("provide outcome_regression or first-round candidates")
    return fmean(predictions)


def estimate_episode(
    episode: dict[str, Any],
    *,
    horizon: int | None = None,
    shrinkage: float = 0.5,
    correct_stopping: bool = True,
    global_mean: float | None = None,
    weight_clip: float | None = None,
) -> dict[str, float]:
    """Estimate one trajectory from probabilities logged before the outcome.

    The logged inclusion-and-continuation factor is ``n * p_t * c_t``. The
    sequential DR value is formed first, clipped to the outcome scale, and
    CESS then shrinks that clipped value toward the OR anchor.
    """
    validate_episode(episode)
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    searches = episode["searches"][:horizon]
    if not searches:
        raise ValueError("episode has no search observations in the requested horizon")
    anchor = _anchor(episode, searches)

    outcomes: list[float] = []
    base_weights: list[float] = []
    residuals: list[float] = []
    for search in searches:
        n = float(search.get("candidate_count", len(search.get("candidates", []))))
        p = float(search["selected_probability"])
        continuation = float(search.get("continuation_probability", 1.0))
        factor = n * p * (continuation if correct_stopping else 1.0)
        base_weights.append(1.0 / factor)
        outcomes.append(float(search["selected_outcome"]))
        residuals.append(float(search["selected_outcome"]) - float(search["selected_outcome_prediction"]))

    def weighted_mean(values: list[float], weights: list[float]) -> float:
        return sum(weight * value for weight, value in zip(weights, values)) / sum(weights)

    weights = list(base_weights)
    if weight_clip is not None:
        if weight_clip <= 0:
            raise ValueError("weight_clip must be positive")
        weights = [min(weight, float(weight_clip)) for weight in weights]
    opened_mean = fmean(outcomes)
    ipw_raw = fmean(weight * y for weight, y in zip(weights, outcomes))
    self_normalized_ipw = weighted_mean(outcomes, weights)
    dr_raw = anchor + fmean(weight * residual for weight, residual in zip(weights, residuals))
    self_normalized_dr_raw = anchor + weighted_mean(residuals, weights)
    result: dict[str, float] = {
        "opened_mean": opened_mean,
        "outcome_regression": clip(anchor),
        "sequential_ipw": clip(ipw_raw),
        "self_normalized_ipw": clip(self_normalized_ipw),
        "sequential_dr": clip(dr_raw),
        "self_normalized_dr": clip(self_normalized_dr_raw),
        "dr_clip_5": clip(anchor + fmean(min(weight, 5.0) * residual for weight, residual in zip(base_weights, residuals))),
        "dr_clip_10": clip(anchor + fmean(min(weight, 10.0) * residual for weight, residual in zip(base_weights, residuals))),
        "dr_clip_20": clip(anchor + fmean(min(weight, 20.0) * residual for weight, residual in zip(base_weights, residuals))),
        "dr_raw": dr_raw,
        "cess": clip(anchor + shrinkage * (clip(dr_raw) - anchor)),
    }
    if global_mean is not None:
        global_value = _finite(global_mean, "global_mean")
        result["opened_to_global"] = clip(opened_mean + shrinkage * (global_value - opened_mean))
    result["opened_to_or"] = clip(opened_mean + shrinkage * (anchor - opened_mean))
    return result


def _task_sensitivity(rows: list[dict[str, Any]], method: str) -> float | None:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["task_id"])][str(row["ranking"])].append(float(row[method]))
    gaps = []
    for rankings in grouped.values():
        if "supporting_first" in rankings and "opposing_first" in rankings:
            gaps.append(abs(fmean(rankings["supporting_first"]) - fmean(rankings["opposing_first"])))
    return fmean(gaps) if gaps else None


def evaluate(
    episodes: Iterable[dict[str, Any]],
    *,
    horizon: int | None = None,
    shrinkage: float = 0.5,
    global_mean: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-episode estimates and task-macro summary metrics."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        estimates = estimate_episode(episode, horizon=horizon, shrinkage=shrinkage, global_mean=global_mean)
        rows.append({
            "task_id": episode["task_id"],
            "ranking": episode.get("ranking", "unspecified"),
            "target": float(episode["target"]),
            **estimates,
        })
    if not rows:
        raise ValueError("input contains no episodes")

    methods = tuple(name for name in METHODS if name in rows[0])
    summary: dict[str, Any] = {
        "episodes": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "horizon": horizon,
        "shrinkage": shrinkage,
        "global_mean": global_mean,
        "aggregation": "episode MAE and task-macro ranking sensitivity",
        "metrics": {},
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    for method in methods:
        task_mae = [fmean(abs(row[method] - row["target"]) for row in task_rows) for task_rows in by_task.values()]
        summary["metrics"][method] = {
            "mae": fmean(abs(row[method] - row["target"]) for row in rows),
            "task_macro_mae": fmean(task_mae),
            "ranking_sensitivity": _task_sensitivity(rows, method),
        }
    return rows, summary
