#!/usr/bin/env python3
"""Run the preregistered CausalSim-Search structural-causal simulation.

The script keeps the finite candidate population fixed within each task.  Search
then samples documents with replacement under a logged, adaptive policy.  The
full-population mean is never used by an estimator on test tasks; it is retained
only to evaluate the estimators after the trajectory has been generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


METHODS = (
    "opened_mean",
    "outcome_regression",
    "sequential_ipw",
    "hajek_ipw",
    "sequential_dr",
    "clipped_dr",
    "task_cess",
    "global_cess",
    "fixed_lambda",
    "oracle_cess",
)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    values = np.exp(np.clip(shifted, -700.0, 0.0))
    return values / float(values.sum())


def categorical_from_uniform(probabilities: np.ndarray, uniform: float) -> int:
    index = int(np.searchsorted(np.cumsum(probabilities), uniform, side="right"))
    return min(index, len(probabilities) - 1)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


@dataclass
class Task:
    task_id: int
    mu: float
    x: np.ndarray
    proxy: np.ndarray
    y: np.ndarray
    conditional_mean: np.ndarray
    conditional_variance: np.ndarray

    @property
    def target(self) -> float:
        return float(np.mean(self.y))


def conditional_moments(mu: float, x: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite moments for E[tanh(a + sigma Z) | X, mu]."""
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    base = (
        mu
        + 0.55 * x[:, 0]
        + 0.35 * np.sin(1.4 * x[:, 1])
        + 0.25 * x[:, 0] * x[:, 2]
        - 0.18 * (x[:, 2] ** 2 - 1.0)
    )
    draws = np.tanh(base[:, None] + math.sqrt(2.0) * sigma * nodes[None, :])
    mean = draws @ weights / math.sqrt(math.pi)
    second = (draws**2) @ weights / math.sqrt(math.pi)
    return mean, np.maximum(second - mean**2, 1e-6)


def generate_tasks(seed: int, count: int, n_documents: int, config: dict) -> list[Task]:
    rng = np.random.default_rng(stable_seed(config["base_seed"], seed, "task_population"))
    tasks: list[Task] = []
    for task_id in range(count):
        mu = float(rng.normal(0.0, 0.35))
        x = rng.normal(size=(n_documents, 3))
        mean, variance = conditional_moments(mu, x, float(config["outcome_noise"]))
        structural = (
            mu
            + 0.55 * x[:, 0]
            + 0.35 * np.sin(1.4 * x[:, 1])
            + 0.25 * x[:, 0] * x[:, 2]
            - 0.18 * (x[:, 2] ** 2 - 1.0)
        )
        y = np.tanh(structural + rng.normal(0.0, float(config["outcome_noise"]), n_documents))
        proxy = (
            0.70 * x[:, 0]
            + 0.35 * x[:, 1]
            + 0.25 * x[:, 0] * x[:, 2]
            + rng.normal(0.0, float(config["proxy_noise"]), n_documents)
        )
        proxy = (proxy - float(proxy.mean())) / max(float(proxy.std()), 1e-6)
        tasks.append(Task(task_id, mu, x, proxy, y, mean, variance))
    return tasks


def model_features(task: Task) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(task.y)),
            np.full(len(task.y), task.mu),
            task.x[:, 0],
            task.x[:, 1],
            task.x[:, 2],
        ]
    )


def variance_features(task: Task) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(task.y)),
            np.full(len(task.y), abs(task.mu)),
            np.abs(task.x[:, 0]),
            task.x[:, 1] ** 2,
            task.x[:, 2] ** 2,
        ]
    )


def fit_nuisance(tasks: list[Task], model_kind: str) -> dict:
    if model_kind == "correct":
        return {"kind": "correct"}
    y = np.concatenate([task.y for task in tasks])
    if model_kind == "constant":
        mean = float(y.mean())
        residual = y - mean
        return {"kind": "constant", "mean": mean, "variance": float(np.mean(residual**2))}
    design = np.vstack([model_features(task) for task in tasks])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coef
    squared_residual = (y - prediction) ** 2
    variance_design = np.vstack([variance_features(task) for task in tasks])
    variance_coef = np.linalg.lstsq(variance_design, squared_residual, rcond=None)[0]
    return {
        "kind": "misspecified",
        "coef": coef.tolist(),
        "variance_coef": variance_coef.tolist(),
    }


def nuisance_predictions(task: Task, model: dict) -> tuple[np.ndarray, np.ndarray]:
    if model["kind"] == "correct":
        return task.conditional_mean.copy(), task.conditional_variance.copy()
    if model["kind"] == "constant":
        mean = np.full(len(task.y), float(model["mean"]))
        variance = np.full(len(task.y), max(float(model["variance"]), 1e-5))
        return mean, variance
    mean = model_features(task) @ np.asarray(model["coef"], dtype=float)
    variance = variance_features(task) @ np.asarray(model["variance_coef"], dtype=float)
    return np.clip(mean, -1.0, 1.0), np.clip(variance, 1e-5, 2.0)


def policy_probabilities(
    task: Task,
    query: float,
    opened_mean: float,
    z: int,
    scenario: dict,
    feedback: float,
) -> np.ndarray:
    relevance = -0.45 * (task.x[:, 0] - query) ** 2 + 0.15 * task.x[:, 1]
    score = relevance + float(scenario["kappa"]) * z * task.proxy
    score += feedback * opened_mean * task.proxy
    if bool(scenario.get("deterministic", False)):
        probabilities = np.zeros(len(task.y), dtype=float)
        probabilities[int(np.argmax(score))] = 1.0
        return probabilities
    agent = softmax(score)
    rho = float(scenario["rho"])
    return (1.0 - rho) * agent + rho / len(agent)


def initial_predicted_variance(
    task: Task,
    residual_variance: np.ndarray,
    z: int,
    scenario: dict,
    feedback: float,
) -> float:
    probabilities = policy_probabilities(task, 0.0, 0.0, z, scenario, feedback)
    if np.any(probabilities <= 0.0):
        return float("inf")
    n = len(probabilities)
    one_round = float(np.sum(residual_variance / (n * n * probabilities)))
    return one_round / max(int(scenario["rounds"]), 1)


def simulate_trajectory(
    task: Task,
    predictions: np.ndarray,
    z: int,
    scenario: dict,
    config: dict,
    seed: int,
) -> dict:
    rounds = int(scenario["rounds"])
    rng = np.random.default_rng(stable_seed(config["base_seed"], seed, task.task_id, "policy"))
    choice_uniforms = rng.random(rounds)
    stop_uniforms = rng.random(rounds)
    query_noise = rng.normal(0.0, 0.08, rounds)
    query = 0.0
    survival = 1.0
    opened_y: list[float] = []
    opened_indices: list[int] = []
    observations: list[dict] = []
    for round_index in range(rounds):
        opened_mean = float(np.mean(opened_y)) if opened_y else 0.0
        probabilities = policy_probabilities(
            task, query, opened_mean, z, scenario, float(config["query_feedback"])
        )
        selected = categorical_from_uniform(probabilities, float(choice_uniforms[round_index]))
        propensity = float(probabilities[selected])
        y = float(task.y[selected])
        m = float(predictions[selected])
        observations.append(
            {
                "index": selected,
                "y": y,
                "m": m,
                "propensity": propensity,
                "survival": survival,
            }
        )
        opened_y.append(y)
        opened_indices.append(selected)
        query = 0.55 * query + 0.35 * y + float(query_noise[round_index])
        if round_index + 1 >= rounds:
            break
        if bool(scenario.get("force_full_horizon", False)):
            continue
        agreement = abs(float(np.mean(np.sign(opened_y))))
        stop_cfg = config["stopping"]
        logit_continue = (
            float(stop_cfg["intercept"])
            - float(stop_cfg["round_penalty"]) * (round_index + 1)
            - float(scenario["stop_strength"]) * abs(float(np.mean(opened_y)))
            - float(stop_cfg["agreement_penalty"]) * agreement
        )
        continue_probability = sigmoid(logit_continue)
        if float(stop_uniforms[round_index]) >= continue_probability:
            break
        survival *= continue_probability
    return {"observations": observations, "opened_indices": opened_indices}


def trajectory_components(trajectory: dict, predictions: np.ndarray, target: float, rounds: int) -> dict:
    observations = trajectory["observations"]
    factors = np.asarray(
        [1.0 / (len(predictions) * row["propensity"] * row["survival"]) for row in observations]
    )
    outcomes = np.asarray([row["y"] for row in observations])
    residuals = np.asarray([row["y"] - row["m"] for row in observations])
    outcome_regression = float(np.mean(predictions))
    correction = float(np.sum(factors * residuals) / rounds)
    ipw = float(np.sum(factors * outcomes) / rounds)
    hajek = float(np.sum(factors * outcomes) / np.sum(factors))
    opened = float(np.mean(outcomes))
    ess = float(np.sum(factors) ** 2 / max(float(np.sum(factors**2)), 1e-12))
    return {
        "opened_mean": opened,
        "outcome_regression": outcome_regression,
        "sequential_ipw": ipw,
        "hajek_ipw": hajek,
        "sequential_dr": outcome_regression + correction,
        "correction": correction,
        "target": target,
        "factors": factors,
        "residuals": residuals,
        "ess": ess,
        "min_propensity": min(row["propensity"] for row in observations),
        "max_weight": float(np.max(factors)),
        "observed_rounds": len(observations),
    }


def clipped_estimate(component: dict, cap: float, rounds: int) -> float:
    correction = float(np.sum(np.minimum(component["factors"], cap) * component["residuals"]) / rounds)
    return float(component["outcome_regression"] + correction)


def path_predicted_variance(
    trajectory: dict,
    residual_variance: np.ndarray,
    rounds: int,
    deterministic: bool = False,
) -> float:
    """Estimate correction variance from the realized, outcome-free log fields.

    For round t, the second moment of R_t A_t r/(N pi_t G_t) contains
    1/(pi_t G_t).  The selected-document expression below is its Horvitz
    estimate and uses the nuisance second moment, never the selected outcome.
    """
    if deterministic:
        return float("inf")
    n = len(residual_variance)
    total = 0.0
    for row in trajectory["observations"]:
        second = float(residual_variance[int(row["index"])])
        denominator = n * n * row["propensity"] ** 2 * row["survival"] ** 2
        total += second / denominator
    return total / (rounds * rounds)


def optimal_lambda(rows: Iterable[dict], error_key: str = "correction") -> float:
    rows = list(rows)
    correction = np.asarray([row[error_key] for row in rows], dtype=float)
    delta = np.asarray([row["target"] - row["outcome_regression"] for row in rows], dtype=float)
    denominator = float(np.sum(correction**2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(correction * delta) / denominator, 0.0, 1.0))


def choose_by_mae(rows: list[dict], candidates: Iterable[float], estimator) -> float:
    scored = []
    for candidate in candidates:
        errors = [abs(float(estimator(row, candidate)) - row["target"]) for row in rows]
        scored.append((float(np.mean(errors)), float(candidate)))
    return min(scored)[1]


def split_ranges(config: dict, task_limit: int | None = None) -> dict[str, range]:
    counts = {key: int(value) for key, value in config["tasks"].items()}
    if task_limit is not None:
        ratios = {"development": 0.4, "calibration": 0.2, "test": 0.4}
        counts = {key: max(2, int(round(task_limit * ratio))) for key, ratio in ratios.items()}
    start = 0
    ranges = {}
    for split in ("development", "calibration", "test"):
        ranges[split] = range(start, start + counts[split])
        start += counts[split]
    return ranges


def run_seed_scenario(seed: int, scenario: dict, config: dict, task_limit: int | None = None) -> tuple[list[dict], dict]:
    ranges = split_ranges(config, task_limit)
    total_tasks = max(item.stop for item in ranges.values())
    tasks = generate_tasks(seed, total_tasks, int(config["candidate_documents"]), config)
    development_tasks = [tasks[index] for index in ranges["development"]]
    model = fit_nuisance(development_tasks, str(scenario["outcome_model"]))
    raw: dict[str, list[dict]] = {key: [] for key in ranges}
    feedback = float(config["query_feedback"])
    for split, indices in ranges.items():
        for index in indices:
            task = tasks[index]
            prediction, residual_variance = nuisance_predictions(task, model)
            for z in (-1, 1):
                trajectory = simulate_trajectory(task, prediction, z, scenario, config, seed)
                component = trajectory_components(trajectory, prediction, task.target, int(scenario["rounds"]))
                component.update(
                    {
                        "seed": seed,
                        "task_id": index,
                        "split": split,
                        "z": z,
                        "initial_predicted_variance": initial_predicted_variance(
                            task, residual_variance, z, scenario, feedback
                        ),
                        "path_predicted_variance": path_predicted_variance(
                            trajectory,
                            residual_variance,
                            int(scenario["rounds"]),
                            bool(scenario.get("deterministic", False)),
                        ),
                    }
                )
                raw[split].append(component)

    dev = raw["development"]
    calibration = raw["calibration"]
    finite_initial_variances = [
        row["initial_predicted_variance"]
        for row in dev
        if math.isfinite(row["initial_predicted_variance"])
    ]
    finite_path_variances = [
        row["path_predicted_variance"]
        for row in dev
        if math.isfinite(row["path_predicted_variance"])
    ]
    delta2 = [(row["target"] - row["outcome_regression"]) ** 2 for row in dev]
    tau2 = float(np.mean(delta2))
    initial_mean_variance = (
        float(np.mean(finite_initial_variances)) if finite_initial_variances else float("inf")
    )
    path_mean_variance = (
        float(np.mean(finite_path_variances)) if finite_path_variances else float("inf")
    )
    use_survival_aware = bool(config.get("survival_aware_variance", False))
    mean_variance = path_mean_variance if use_survival_aware else initial_mean_variance
    global_lambda = tau2 / (tau2 + mean_variance) if math.isfinite(mean_variance) else 0.0
    initial_global_lambda = (
        tau2 / (tau2 + initial_mean_variance) if math.isfinite(initial_mean_variance) else 0.0
    )
    cap = choose_by_mae(
        calibration,
        config["weight_caps"],
        lambda row, candidate: clipped_estimate(row, candidate, int(scenario["rounds"])),
    )
    fixed_lambda = choose_by_mae(
        calibration,
        config["fixed_lambdas"],
        lambda row, candidate: row["outcome_regression"] + candidate * row["correction"],
    )
    oracle_lambda = optimal_lambda(raw["test"])

    output: list[dict] = []
    for row in raw["test"]:
        initial_variance = row["initial_predicted_variance"]
        path_variance = row["path_predicted_variance"]
        predicted_variance = path_variance if use_survival_aware else initial_variance
        task_lambda = tau2 / (tau2 + predicted_variance) if math.isfinite(predicted_variance) else 0.0
        initial_task_lambda = tau2 / (tau2 + initial_variance) if math.isfinite(initial_variance) else 0.0
        estimates = {
            "opened_mean": row["opened_mean"],
            "outcome_regression": row["outcome_regression"],
            "sequential_ipw": row["sequential_ipw"],
            "hajek_ipw": row["hajek_ipw"],
            "sequential_dr": row["sequential_dr"],
            "clipped_dr": clipped_estimate(row, cap, int(scenario["rounds"])),
            "task_cess": row["outcome_regression"] + task_lambda * row["correction"],
            "global_cess": row["outcome_regression"] + global_lambda * row["correction"],
            "fixed_lambda": row["outcome_regression"] + fixed_lambda * row["correction"],
            "oracle_cess": row["outcome_regression"] + oracle_lambda * row["correction"],
            "initial_task_cess": row["outcome_regression"] + initial_task_lambda * row["correction"],
            "initial_global_cess": row["outcome_regression"] + initial_global_lambda * row["correction"],
        }
        output.append(
            {
                "scenario": scenario["id"],
                "family": scenario["family"],
                "seed": seed,
                "task_id": row["task_id"],
                "z": row["z"],
                "target": row["target"],
                "predicted_variance": predicted_variance,
                "initial_predicted_variance": initial_variance,
                "path_predicted_variance": path_variance,
                "task_lambda": task_lambda,
                "initial_task_lambda": initial_task_lambda,
                "global_lambda": global_lambda,
                "initial_global_lambda": initial_global_lambda,
                "fixed_lambda_value": fixed_lambda,
                "oracle_lambda": oracle_lambda,
                "weight_cap": cap,
                "ess": row["ess"],
                "min_propensity": row["min_propensity"],
                "max_weight": row["max_weight"],
                "observed_rounds": row["observed_rounds"],
                **estimates,
            }
        )
    tuning = {
        "scenario": scenario["id"],
        "seed": seed,
        "tau2": tau2,
        "mean_predicted_variance": mean_variance,
        "initial_mean_predicted_variance": initial_mean_variance,
        "path_mean_predicted_variance": path_mean_variance,
        "survival_aware_variance": use_survival_aware,
        "global_lambda": global_lambda,
        "initial_global_lambda": initial_global_lambda,
        "fixed_lambda": fixed_lambda,
        "oracle_lambda": oracle_lambda,
        "weight_cap": cap,
        "n_test_rows": len(output),
    }
    return output, tuning


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/causal_search_scm_v1.yaml")
    parser.add_argument("--output-dir", default="evidence_selection_benchmark/results/causal_search_scm_v1")
    parser.add_argument("--scenario", action="append", help="Run only the named scenario; repeatable")
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--task-limit", type=int, default=None, help="Smoke-test override; never use for paper results")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scenarios = config["scenarios"]
    if args.scenario:
        requested = set(args.scenario)
        scenarios = [item for item in scenarios if item["id"] in requested]
        missing = requested - {item["id"] for item in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")
    seeds = list(config["seeds"])
    if args.seed_count is not None:
        seeds = seeds[: args.seed_count]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tuning = []
    started = time.time()
    for scenario in scenarios:
        for seed in seeds:
            rows, tuning = run_seed_scenario(seed, scenario, config, args.task_limit)
            suffix = "smoke" if args.task_limit is not None else "full"
            write_rows(output_dir / "task_rows" / f"{scenario['id']}__seed{seed}__{suffix}.csv", rows)
            all_tuning.append(tuning)
            print(
                json.dumps(
                    {
                        "scenario": scenario["id"],
                        "seed": seed,
                        "global_lambda": round(tuning["global_lambda"], 5),
                        "weight_cap": tuning["weight_cap"],
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    metadata = {
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "task_limit": args.task_limit,
        "scenarios": [item["id"] for item in scenarios],
        "seeds": seeds,
        "elapsed_seconds": time.time() - started,
        "tuning": all_tuning,
    }
    suffix = "smoke" if args.task_limit is not None else "full"
    (output_dir / f"run_metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
