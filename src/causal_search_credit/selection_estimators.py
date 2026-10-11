from __future__ import annotations

import math
from collections.abc import Sequence


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def calibrate_intercept(
    scores: Sequence[float],
    target_count: float,
    minimum: float,
    maximum: float,
) -> float:
    if not scores:
        raise ValueError("cannot calibrate an empty score vector")
    target = max(minimum * len(scores), min(maximum * len(scores), target_count))
    low, high = -20.0, 20.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        total = sum(
            max(minimum, min(maximum, sigmoid(midpoint + score)))
            for score in scores
        )
        if total < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def inclusion_probabilities(
    proxies: Sequence[float],
    regime: str,
    strength: float,
    target_count: float,
    exploration_probability: float,
    minimum: float,
    maximum: float,
) -> tuple[list[float], float]:
    direction = {"neutral": 0.0, "pro_positive": 1.0, "pro_negative": -1.0}[regime]
    scores = [direction * strength * proxy for proxy in proxies]
    intercept = calibrate_intercept(scores, target_count, minimum, maximum)
    base = [
        max(minimum, min(maximum, sigmoid(intercept + score)))
        for score in scores
    ]
    uniform = target_count / len(proxies)
    mixed = [
        (1.0 - exploration_probability) * probability
        + exploration_probability * uniform
        for probability in base
    ]
    return [max(minimum, min(maximum, value)) for value in mixed], intercept


def weighted_mean(
    outcomes: Sequence[float],
    sample_sizes: Sequence[float],
    selected: Sequence[bool] | None = None,
) -> float:
    mask = list(selected) if selected is not None else [True] * len(outcomes)
    numerator = sum(
        outcome * weight
        for outcome, weight, keep in zip(outcomes, sample_sizes, mask)
        if keep
    )
    denominator = sum(
        weight for weight, keep in zip(sample_sizes, mask) if keep
    )
    return numerator / denominator if denominator else float("nan")


def hajek_ipw(
    outcomes: Sequence[float],
    sample_sizes: Sequence[float],
    selected: Sequence[bool],
    propensities: Sequence[float],
) -> float:
    adjusted = [
        weight / propensity
        for weight, keep, propensity in zip(sample_sizes, selected, propensities)
        if keep
    ]
    denominator = sum(adjusted)
    if not denominator:
        return float("nan")
    numerator = sum(
        outcome * weight / propensity
        for outcome, weight, keep, propensity in zip(
            outcomes, sample_sizes, selected, propensities
        )
        if keep
    )
    return numerator / denominator


def augmented_ipw(
    outcomes: Sequence[float],
    outcome_predictions: Sequence[float],
    sample_sizes: Sequence[float],
    selected: Sequence[bool],
    propensities: Sequence[float],
) -> float:
    denominator = sum(sample_sizes)
    if not denominator:
        return float("nan")
    regression_total = sum(
        weight * prediction
        for weight, prediction in zip(sample_sizes, outcome_predictions)
    )
    correction = sum(
        weight * (outcome - prediction) / propensity
        for outcome, prediction, weight, keep, propensity in zip(
            outcomes, outcome_predictions, sample_sizes, selected, propensities
        )
        if keep
    )
    return (regression_total + correction) / denominator
