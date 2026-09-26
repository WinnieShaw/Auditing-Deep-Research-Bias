"""Task-cluster bootstrap utilities for public audit outputs.

Rows from the same source task are sampled together. The implementation uses
only the Python standard library so that the public smoke test has no hidden
scientific dependency.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import fmean
from typing import Any, Callable, Iterable


def _groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_id"])].append(row)
    if not groups:
        raise ValueError("cannot bootstrap an empty row set")
    return groups


def task_mae(rows: list[dict[str, Any]], method: str) -> float:
    """Task-macro MAE: each task contributes equally."""
    groups = _groups(rows)
    return fmean(fmean(abs(float(row[method]) - float(row["target"])) for row in task_rows) for task_rows in groups.values())


def task_ranking_sensitivity(rows: list[dict[str, Any]], method: str) -> float:
    """Mean within-task absolute supporting-vs-opposing ranking gap."""
    groups = _groups(rows)
    contributions = []
    for task_rows in groups.values():
        supporting = [float(row[method]) for row in task_rows if row.get("ranking") == "supporting_first"]
        opposing = [float(row[method]) for row in task_rows if row.get("ranking") == "opposing_first"]
        if supporting and opposing:
            contributions.append(abs(fmean(supporting) - fmean(opposing)))
    if not contributions:
        raise ValueError("ranking sensitivity requires both ranking conditions")
    return fmean(contributions)


def bootstrap_ci(
    rows: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    *,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Return a percentile 95% CI using source-task cluster resampling."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups = _groups(rows)
    task_ids = list(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sampled = []
        for _ in task_ids:
            sampled.extend(groups[rng.choice(task_ids)])
        values.append(float(statistic(sampled)))
    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return {
        "estimate": float(statistic(rows)),
        "lower": values[lower_index],
        "upper": values[upper_index],
        "replicates": replicates,
        "cluster_count": len(task_ids),
        "seed": seed,
    }


def paired_difference_ci(
    rows: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    *,
    metric: str = "mae",
    replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """CI for method A minus method B, preserving task-level pairing."""
    def statistic(sampled: list[dict[str, Any]]) -> float:
        if metric == "mae":
            return task_mae(sampled, method_a) - task_mae(sampled, method_b)
        if metric == "sensitivity":
            return task_ranking_sensitivity(sampled, method_a) - task_ranking_sensitivity(sampled, method_b)
        raise ValueError("metric must be 'mae' or 'sensitivity'")

    return bootstrap_ci(rows, statistic, replicates=replicates, seed=seed)

