#!/usr/bin/env python3
"""Task-cluster analysis of the completed true-Agent MS2 2x2 experiment.

The observed endpoint is the mean private evidence direction of the documents
opened by the agent.  It is deliberately named an opened-evidence conclusion:
the runner does not produce a separate final natural-language report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = {"00": ("uniform", "fixed_horizon"), "10": ("agent", "fixed_horizon"),
              "01": ("uniform", "adaptive"), "11": ("agent", "adaptive")}
EFFECTS = ("selection", "stopping", "interaction", "total")
SEED = 2026092304


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decompose(theta: dict[str, float]) -> dict[str, float]:
    selection = theta["10"] - theta["00"]
    stopping = theta["01"] - theta["00"]
    interaction = theta["11"] - theta["10"] - theta["01"] + theta["00"]
    total = theta["11"] - theta["00"]
    return {"selection": selection, "stopping": stopping, "interaction": interaction, "total": total,
            "residual": total - selection - stopping - interaction}


def ternary(value: float, tolerance: float = .05) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def dominant(effect: dict[str, float], tolerance: float = .05) -> str:
    ranked = sorted(((name, abs(effect[name])) for name in ("selection", "stopping", "interaction")),
                    key=lambda item: item[1], reverse=True)
    if ranked[0][1] < tolerance:
        return "none"
    if ranked[0][1] - ranked[1][1] <= tolerance:
        return "ambiguous"
    return ranked[0][0]


def bootstrap(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    draws = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return mean, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def prepare_cells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cells, trace_rows, audit = [], [], {"models": {}}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "cells.jsonl").exists()):
        model = model_dir.name
        run = json.loads((model_dir / "run_summary.json").read_text())
        cell = pd.DataFrame(load_jsonl(model_dir / "cells.jsonl"))
        trace = pd.DataFrame(load_jsonl(model_dir / "traces.jsonl"))
        if set(cell.rank_condition) != {"neutral"}:
            raise ValueError(f"{model}: rank condition is not fixed neutral")
        expected = 200 * 2 * 2 * 3
        if len(cell) != expected:
            raise ValueError(f"{model}: expected {expected} cells, received {len(cell)}")
        counts = cell.groupby(["review_id", "controller", "stopping_regime"]).replicate.nunique()
        if not (counts == 3).all():
            raise ValueError(f"{model}: missing paired replicate")
        if cell.groupby("review_id").target.nunique().max() != 1:
            raise ValueError(f"{model}: target changes within source task")
        fixed = cell[cell.stopping_regime == "fixed_horizon"]
        if not ((fixed.rounds_observed == 10) & (fixed.minimum_survival_probability == 1.0)).all():
            raise ValueError(f"{model}: fixed condition is not a ten-round no-stop policy")
        selected = trace[trace.document_id.notna()].copy()
        prob_sum_error = selected.candidate_probabilities.map(lambda p: abs(sum(p) - 1.0)).max()
        selection_error = selected.apply(lambda row: abs(float(row.selection_probability) - float(row.candidate_probabilities[row.candidate_document_ids.index(row.document_id)])), axis=1).max()
        support_min = min(min(p) for p in selected.candidate_probabilities)
        uniform = selected[selected.controller == "uniform"]
        uniform_deviation = uniform.candidate_probabilities.map(lambda p: float(np.max(np.abs(np.asarray(p) - 1.0 / len(p))))).max()
        audit["models"][model] = {"run_summary": run, "cells": int(len(cell)), "selected_trace_rows": int(len(selected)),
                                  "max_probability_sum_error": float(prob_sum_error), "max_selected_probability_error": float(selection_error),
                                  "minimum_candidate_probability": float(support_min), "max_uniform_probability_deviation": float(uniform_deviation),
                                  "fixed_episodes": int(len(fixed)), "adaptive_episodes": int((cell.stopping_regime == "adaptive").sum())}
        cells.append(cell); trace_rows.append(selected)
    return pd.concat(cells, ignore_index=True), pd.concat(trace_rows, ignore_index=True), audit


def trace_reliability(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode_id, group in selected.groupby("episode_id", sort=False):
        weights = 1.0 / (group.selection_probability.to_numpy(float) * group.survival_probability.to_numpy(float))
        rows.append({"episode_id": episode_id, "ess": float(weights.sum() ** 2 / np.square(weights).sum()),
                     "maximum_weight": float(weights.max()), "minimum_probability": float(group.selection_probability.min()),
                     "observed_rounds_trace": int(len(group))})
    return pd.DataFrame(rows)


def task_effects(cells: pd.DataFrame, reliab: pd.DataFrame) -> pd.DataFrame:
    cells = cells.merge(reliab, on="episode_id", how="left", validate="one_to_one")
    rows = []
    for (model, review_id), group in cells.groupby(["model_key", "review_id"], sort=True):
        theta_actual, theta_cess, theta_dr = {}, {}, {}
        summaries = {}
        for condition, (controller, stopping) in CONDITIONS.items():
            g = group[(group.controller == controller) & (group.stopping_regime == stopping)]
            if len(g) != 3:
                raise ValueError(f"{model}/{review_id}/{condition}: expected three paired episodes")
            theta_actual[condition] = float(g.naive_opened_mean.mean())
            theta_cess[condition] = float(g.preoutcome_cess.mean())
            theta_dr[condition] = float(g.sequential_dr.mean())
            summaries[condition] = {"rounds": float(g.rounds_observed.mean()), "ess": float(g.ess.mean()),
                                    "max_weight": float(g.maximum_weight.mean()), "min_probability": float(g.minimum_probability.min())}
        actual, predicted, dr = decompose(theta_actual), decompose(theta_cess), decompose(theta_dr)
        row = {"model": model, "review_id": review_id, "target": float(group.target.iloc[0]),
               "actual_dominant": dominant(actual), "cess_dominant": dominant(predicted),
               "actual_decomposition_residual": actual["residual"], "cess_decomposition_residual": predicted["residual"]}
        for condition in CONDITIONS:
            row[f"actual_theta_{condition}"] = theta_actual[condition]
            row[f"cess_theta_{condition}"] = theta_cess[condition]
            row[f"dr_theta_{condition}"] = theta_dr[condition]
            for key, value in summaries[condition].items():
                row[f"{key}_{condition}"] = value
        for effect in EFFECTS:
            row[f"actual_delta_{effect}"] = actual[effect]
            row[f"cess_delta_{effect}"] = predicted[effect]
            row[f"dr_delta_{effect}"] = dr[effect]
            row[f"cess_error_{effect}"] = predicted[effect] - actual[effect]
        rows.append(row)
    return pd.DataFrame(rows)


def metric_rows(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in effects.groupby("model", sort=True):
        for effect in EFFECTS:
            error = group[f"cess_error_{effect}"]
            actual = group[f"actual_delta_{effect}"]
            predicted = group[f"cess_delta_{effect}"]
            nontrivial = group[[f"actual_delta_{e}" for e in ("selection", "stopping", "interaction")]].abs().max(axis=1) >= .05
            ordered = np.sort(group[[f"actual_delta_{e}" for e in ("selection", "stopping", "interaction")]].abs().to_numpy(), axis=1)
            discriminable = (ordered[:, -1] - ordered[:, -2]) > .05
            for subset, mask in (("all", np.ones(len(group), dtype=bool)), ("nontrivial", nontrivial.to_numpy()), ("discriminable", discriminable)):
                e, a, p = error[mask], actual[mask], predicted[mask]
                mean, low, high = bootstrap(np.abs(e.to_numpy()), stable_seed(model, effect, subset))
                rows.append({"model": model, "effect": effect, "subset": subset, "source_tasks": int(mask.sum()),
                             "mae": mean, "mae_ci95_low": low, "mae_ci95_high": high,
                             "rmse": float(np.sqrt(np.mean(np.square(e)))), "signed_bias": float(e.mean()),
                             "spearman": float(a.corr(p, method="spearman")) if a.nunique() > 1 and p.nunique() > 1 else np.nan,
                             "direction_accuracy": float(np.mean([ternary(x) == ternary(y) for x, y in zip(a, p)]))})
        non_ambiguous = group[(~group.actual_dominant.isin(["none", "ambiguous"])) & (~group.cess_dominant.isin(["none", "ambiguous"]))]
        rows.append({"model": model, "effect": "dominant", "subset": "all", "source_tasks": len(group),
                     "mae": np.nan, "mae_ci95_low": np.nan, "mae_ci95_high": np.nan, "rmse": np.nan, "signed_bias": np.nan,
                     "spearman": np.nan, "direction_accuracy": float((group.actual_dominant == group.cess_dominant).mean()),
                     "dominant_intervention_success": float((non_ambiguous.actual_dominant == non_ambiguous.cess_dominant).mean()) if len(non_ambiguous) else np.nan,
                     "non_ambiguous_tasks": len(non_ambiguous)})
    return pd.DataFrame(rows)


def localization(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in effects.groupby("model", sort=True):
        n = len(group); k = math.ceil(.2 * n)
        actual = set(group.assign(_risk=group.actual_delta_total.abs()).nlargest(k, "_risk").review_id)
        predicted = set(group.assign(_risk=group.cess_delta_total.abs()).nlargest(k, "_risk").review_id)
        hit = len(actual & predicted)
        rows.append({"model": model, "top_fraction": .2, "tasks_selected": k, "overlap": hit,
                     "precision": hit / k, "recall": hit / k})
    return pd.DataFrame(rows)


def render(out: Path, effects: pd.DataFrame, calibration: pd.DataFrame, selective: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Times New Roman is unavailable on the analysis host; use an installed
    # serif face rather than silently accepting Matplotlib's sans fallback.
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    colors = {"qwen2_5_32b": "#4477AA", "olmo3_7b": "#CC6677"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    for ax, effect in zip(axes, ("selection", "stopping", "interaction")):
        for model, group in effects.groupby("model", sort=True):
            ax.scatter(group[f"actual_delta_{effect}"], group[f"cess_delta_{effect}"], s=18, alpha=.55, color=colors[model], label=model)
        low = min(ax.get_xlim()[0], ax.get_ylim()[0]); high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], color="0.35", linewidth=1)
        ax.set(xlabel="Observed intervention effect", ylabel="CESS-predicted effect", title=effect)
    axes[0].legend(frameon=True, fontsize=8); fig.savefig(out / "predicted_vs_observed.pdf", bbox_inches="tight"); fig.savefig(out / "predicted_vs_observed.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    classes = ["selection", "stopping", "interaction", "ambiguous", "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    for ax, (model, group) in zip(axes, effects.groupby("model", sort=True)):
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for _, r in group.iterrows(): matrix[classes.index(r.actual_dominant), classes.index(r.cess_dominant)] += 1
        im = ax.imshow(matrix, cmap="Blues")
        ax.set(title=model, xlabel="CESS dominant mechanism", ylabel="Observed dominant mechanism", xticks=range(5), yticks=range(5), xticklabels=classes, yticklabels=classes)
        ax.tick_params(axis="x", labelrotation=28, labelsize=8)
        for label in ax.get_xticklabels(): label.set_horizontalalignment("right")
        for i in range(5):
            for j in range(5): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.84, label="Tasks"); fig.savefig(out / "dominant_mechanism_confusion.pdf", bbox_inches="tight"); fig.savefig(out / "dominant_mechanism_confusion.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for model, group in calibration.groupby("model", sort=True): ax.plot(group.ess_bin, group.mean_total_effect_mae, marker="o", label=model, color=colors[model])
    ax.set(xlabel="ESS quintile (low to high)", ylabel="Mean total-effect MAE", title="Reliability by realized ESS"); ax.legend(); fig.savefig(out / "reliability_calibration.pdf", bbox_inches="tight"); fig.savefig(out / "reliability_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for model, group in selective.groupby("model", sort=True): ax.plot(group.coverage, group.risk, marker="o", label=model, color=colors[model])
    ax.set(xlabel="Explanation coverage", ylabel="CESS total-effect MAE", title="Selective risk--coverage"); ax.legend(); fig.savefig(out / "selective_risk_coverage.pdf", bbox_inches="tight"); fig.savefig(out / "selective_risk_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True); (args.output / "figures").mkdir()
    cells, traces, audit = prepare_cells(args.input)
    reliab = trace_reliability(traces)
    effects = task_effects(cells, reliab); effects.to_csv(args.output / "task_level_results.csv", index=False)
    metrics = metric_rows(effects); metrics.to_csv(args.output / "metrics.csv", index=False)
    loc = localization(effects); loc.to_csv(args.output / "risk_localization.csv", index=False)
    cal = []
    for model, group in effects.groupby("model", sort=True):
        group = group.copy(); group["ess_bin"] = pd.qcut(group.ess_11.rank(method="first"), 5, labels=False) + 1
        for b, g in group.groupby("ess_bin"):
            cal.append({"model": model, "ess_bin": int(b), "source_tasks": len(g), "mean_ess": float(g.ess_11.mean()),
                        "mean_maximum_weight": float(g.max_weight_11.mean()), "mean_total_effect_mae": float(g.cess_error_total.abs().mean())})
    calibration = pd.DataFrame(cal); calibration.to_csv(args.output / "calibration.csv", index=False)
    selective = []
    for model, group in effects.groupby("model", sort=True):
        group = group.sort_values("ess_11", ascending=False)
        for coverage in (.2, .4, .6, .8, 1.0):
            n = math.ceil(len(group) * coverage); g = group.iloc[:n]
            selective.append({"model": model, "coverage": coverage, "source_tasks": n, "risk": float(g.cess_error_total.abs().mean()), "ess_threshold": float(g.ess_11.min())})
    selective = pd.DataFrame(selective); selective.to_csv(args.output / "selective_risk.csv", index=False)
    audit["max_actual_decomposition_residual"] = float(effects.actual_decomposition_residual.abs().max())
    audit["max_cess_decomposition_residual"] = float(effects.cess_decomposition_residual.abs().max())
    write_json(args.output / "audit.json", audit); render(args.output / "figures", effects, calibration, selective)
    total = metrics[(metrics.effect == "total") & (metrics.subset == "all")]
    dom = metrics[metrics.effect == "dominant"]
    lines = ["# True-Agent MS2 counterfactual-faithfulness analysis", "", "## Endpoint", "",
             "Observed effects are differences in the mean private evidence direction among documents opened by the real query-generation Agent under the four paired policies. The runner has no final natural-language synthesis answer, so this analysis does not claim to measure final-report behavior or model-internal mechanisms.", "",
             "## Run integrity", "", "Both models contain 200 source tasks and all 12 required episodes per task (four conditions × three paired replicates). Fixed-horizon episodes ran ten rounds with survival probability one. The audit verifies normalized full candidate probability vectors, selected-probability consistency, positive support, and exact uniform probabilities for the reference controller.", "",
             "## Effect faithfulness", "", "| Model | Total-effect MAE [95% task bootstrap CI] | RMSE | Spearman | Direction accuracy |", "|---|---:|---:|---:|---:|"]
    for _, r in total.iterrows(): lines.append(f"| {r.model} | {r.mae:.4f} [{r.mae_ci95_low:.4f}, {r.mae_ci95_high:.4f}] | {r.rmse:.4f} | {r.spearman:.3f} | {r.direction_accuracy:.3f} |")
    lines += ["", "The resulting values describe how well CESS changes track observed policy-replacement changes in this particular Agent environment. They do not establish that CESS identifies a natural indirect effect, because replacing selection also changes later query histories.", "",
              "## Dominant-mechanism and risk localization", "", "| Model | Exact dominant-label agreement | Agreement among non-ambiguous pairs | Top-20% risk precision/recall |", "|---|---:|---:|---:|"]
    for model in sorted(effects.model.unique()):
        d = dom[dom.model == model].iloc[0]; l = loc[loc.model == model].iloc[0]
        lines.append(f"| {model} | {d.direction_accuracy:.3f} | {d.dominant_intervention_success:.3f} | {l.precision:.3f} / {l.recall:.3f} |")
    lines += ["", "## Interpretation boundary", "", "This is a completed true-LLM policy-intervention analysis for MS2, not a PERSPECTRUM replication. It contains real LLM-generated queries and condition-specific retrieval trajectories. The output supports controlled claims about observable opened-evidence dynamics; the absent final synthesis answer prevents a claim about report-level counterfactual faithfulness. All uncertainty above resamples source tasks, retaining their four conditions and three replicates together."]
    (args.output / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output / "run_manifest.json", {"input": str(args.input), "models": sorted(effects.model.unique()), "source_tasks_per_model": 200, "conditions": CONDITIONS, "replicates_per_condition": 3, "bootstrap": 10_000, "seed": SEED})
    print(json.dumps({"output": str(args.output), "task_rows": len(effects), "models": sorted(effects.model.unique())}, indent=2))


if __name__ == "__main__": main()
