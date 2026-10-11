#!/usr/bin/env python3
"""Strict continuous matching for the v6 accuracy--sensitivity comparison.

This is CPU-only post-processing of frozen task-level estimates.  It preserves
all settings, refuses extrapolation, and repeats interpolation inside every
source-task cluster bootstrap replicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOT = 10_000
SEED = 2026092317
BASELINE_FAMILIES = ("ordinary shrinkage: global", "ordinary shrinkage: OR")


def task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one MAE and one paired ranking-sensitivity value per source task."""
    frame = frame.copy()
    frame["abs_error"] = (frame["estimate"] - frame["target"]).abs()
    mae = frame.groupby("review_id", sort=True)["abs_error"].mean()
    ranks = set(frame["rank_condition"].astype(str))
    if {"pro_increase", "pro_decrease"} <= ranks:
        positive, negative = "pro_increase", "pro_decrease"
    elif {"pro_benefit", "pro_harm"} <= ranks:
        positive, negative = "pro_benefit", "pro_harm"
    else:
        raise ValueError(f"missing opposing rank conditions: {sorted(ranks)}")
    context = [c for c in ("review_id", "model_key", "controller", "stopping_regime") if c in frame]
    rank_mean = frame.groupby(context + ["rank_condition"], sort=True, as_index=False)["estimate"].mean()
    paired = rank_mean.pivot(index=context, columns="rank_condition", values="estimate")
    sensitivity = (paired[positive] - paired[negative]).abs().groupby(level="review_id", sort=True).mean()
    common = sorted(set(mae.index.astype(str)) & set(sensitivity.index.astype(str)))
    mae.index = mae.index.astype(str); sensitivity.index = sensitivity.index.astype(str)
    return pd.DataFrame({"mae": mae.reindex(common), "sensitivity": sensitivity.reindex(common)}, index=common)


def nearest_branch_intersection(lambdas: np.ndarray, match_curve: np.ndarray,
                                other_curve: np.ndarray, target: float,
                                tolerance: float = 1e-12) -> dict:
    """Return the legal intersection closest to this replicate's nearest grid point."""
    candidates: list[tuple[float, float, int, float]] = []
    for i in range(len(lambdas) - 1):
        x0, x1 = float(lambdas[i]), float(lambdas[i + 1])
        y0, y1 = float(match_curve[i]), float(match_curve[i + 1])
        z0, z1 = float(other_curve[i]), float(other_curve[i + 1])
        if abs(y0 - target) <= tolerance:
            candidates.append((x0, z0, i, 0.0))
        delta = y1 - y0
        if abs(delta) > tolerance:
            weight = (target - y0) / delta
            if -tolerance <= weight <= 1.0 + tolerance:
                weight = float(np.clip(weight, 0.0, 1.0))
                candidates.append((x0 + weight * (x1 - x0), z0 + weight * (z1 - z0), i, weight))
    if abs(float(match_curve[-1]) - target) <= tolerance:
        candidates.append((float(lambdas[-1]), float(other_curve[-1]), len(lambdas) - 2, 1.0))
    unique: dict[float, tuple[float, float, int, float]] = {}
    for item in candidates:
        unique.setdefault(round(item[0], 12), item)
    if not unique:
        return {"matchable": False, "interpolated_lambda": np.nan,
                "interpolated_other": np.nan, "segment_low": np.nan,
                "segment_high": np.nan, "root_count": 0,
                "nearest_grid_lambda": float(lambdas[int(np.argmin(np.abs(match_curve-target)))])}
    roots = list(unique.values())
    nearest_grid_lambda = float(lambdas[int(np.argmin(np.abs(match_curve - target)))])
    lam, other, segment, _ = min(roots, key=lambda x: (abs(x[0] - nearest_grid_lambda), x[0]))
    return {"matchable": True, "interpolated_lambda": lam,
            "interpolated_other": other,
            "segment_low": float(lambdas[segment]),
            "segment_high": float(lambdas[segment + 1]),
            "root_count": len(roots),
            "nearest_grid_lambda": nearest_grid_lambda}


def prepare_setting(frame: pd.DataFrame, dataset: str, model: str, horizon: int,
                    family: str) -> dict:
    setting = frame[(frame["dataset"] == dataset) &
                    (frame["query_model"] == model) &
                    (frame["horizon"] == horizon)]
    cess = setting[setting["method"] == "Full CESS (frozen original)"]
    baseline = setting[setting["family"] == family]
    if cess.empty or baseline.empty:
        raise ValueError(f"missing CESS/baseline for {(dataset, model, horizon, family)}")
    cess_task = task_metrics(cess)
    lambdas = np.asarray(sorted(baseline["lambda"].dropna().unique()), dtype=float)
    if len(lambdas) < 2 or not np.allclose(np.diff(lambdas), 0.05, atol=1e-12):
        raise ValueError(f"unexpected lambda grid for {(dataset, model, horizon, family)}: {lambdas}")
    per_lambda = {float(lam): task_metrics(baseline[np.isclose(baseline["lambda"], lam)]) for lam in lambdas}
    common = set(cess_task.index)
    for table in per_lambda.values():
        common &= set(table.index)
    tasks = sorted(common)
    if not tasks:
        raise ValueError(f"no common source tasks for {(dataset, model, horizon, family)}")
    return {
        "tasks": tasks,
        "lambdas": lambdas,
        "cess_mae": cess_task.reindex(tasks)["mae"].to_numpy(float),
        "cess_sensitivity": cess_task.reindex(tasks)["sensitivity"].to_numpy(float),
        "baseline_mae": np.column_stack([per_lambda[float(l)].reindex(tasks)["mae"] for l in lambdas]),
        "baseline_sensitivity": np.column_stack([per_lambda[float(l)].reindex(tasks)["sensitivity"] for l in lambdas]),
    }


def bootstrap_interpolated(data: dict, match_metric: str, seed: int) -> dict:
    n = len(data["tasks"]); rng = np.random.default_rng(seed)
    effects: list[float] = []; lambdas: list[float] = []; matchable = 0
    for _ in range(BOOT):
        index = rng.integers(0, n, size=n)
        c_mae = float(data["cess_mae"][index].mean())
        c_sens = float(data["cess_sensitivity"][index].mean())
        b_mae = data["baseline_mae"][index].mean(axis=0)
        b_sens = data["baseline_sensitivity"][index].mean(axis=0)
        if match_metric == "mae":
            root = nearest_branch_intersection(data["lambdas"], b_mae, b_sens, c_mae)
            cess_value = c_sens
        else:
            root = nearest_branch_intersection(data["lambdas"], b_sens, b_mae, c_sens)
            cess_value = c_mae
        if root["matchable"]:
            matchable += 1; lambdas.append(float(root["interpolated_lambda"]))
            effects.append(float(cess_value - root["interpolated_other"]))
    if not effects:
        return {"bootstrap_replicates": BOOT, "bootstrap_matchable_replicates": 0,
                "bootstrap_matchable_fraction": 0.0, "ci95_low": np.nan,
                "ci95_high": np.nan, "bootstrap_effect_mean": np.nan,
                "bootstrap_lambda_median": np.nan}
    effects_array = np.asarray(effects); lambda_array = np.asarray(lambdas)
    return {"bootstrap_replicates": BOOT,
            "bootstrap_matchable_replicates": matchable,
            "bootstrap_matchable_fraction": matchable / BOOT,
            "ci95_low": float(np.quantile(effects_array, .025)),
            "ci95_high": float(np.quantile(effects_array, .975)),
            "bootstrap_effect_mean": float(effects_array.mean()),
            "bootstrap_lambda_median": float(np.median(lambda_array))}


def analyze_one(data: dict, keys: tuple, family: str, match_metric: str, seed: int) -> dict:
    c_mae = float(data["cess_mae"].mean()); c_sens = float(data["cess_sensitivity"].mean())
    b_mae = data["baseline_mae"].mean(axis=0); b_sens = data["baseline_sensitivity"].mean(axis=0)
    if match_metric == "mae":
        root = nearest_branch_intersection(data["lambdas"], b_mae, b_sens, c_mae)
        target, cess_comparison, comparison = c_mae, c_sens, "ranking_sensitivity"
    else:
        root = nearest_branch_intersection(data["lambdas"], b_sens, b_mae, c_sens)
        target, cess_comparison, comparison = c_sens, c_mae, "mae"
    result = {"dataset": keys[0], "query_model": keys[1], "horizon": keys[2],
              "baseline_family": family, "source_tasks": len(data["tasks"]),
              "match_metric": match_metric, "target_cess_value": target,
              "comparison_metric": comparison, "cess_comparison_value": cess_comparison,
              "status": "matched_by_interpolation" if root["matchable"] else "not_matchable",
              **root}
    if not root["matchable"]:
        result.update({"baseline_interpolated_value": np.nan,
                       "cess_minus_baseline": np.nan, "ci95_low": np.nan,
                       "ci95_high": np.nan, "significantly_lower": False,
                       "significantly_higher": False,
                       "bootstrap_replicates": BOOT,
                       "bootstrap_matchable_replicates": 0,
                       "bootstrap_matchable_fraction": 0.0,
                       "bootstrap_effect_mean": np.nan,
                       "bootstrap_lambda_median": np.nan})
        return result
    effect = float(cess_comparison - root["interpolated_other"])
    boot = bootstrap_interpolated(data, match_metric, seed)
    result.update({"baseline_interpolated_value": root["interpolated_other"],
                   "cess_minus_baseline": effect, **boot})
    result["significantly_lower"] = bool(result["ci95_high"] < 0)
    result["significantly_higher"] = bool(result["ci95_low"] > 0)
    return result


def comparison_report(mae: pd.DataFrame, sens: pd.DataFrame,
                      nearest_mae: pd.DataFrame, nearest_sens: pd.DataFrame) -> str:
    key = ["dataset", "query_model", "horizon", "baseline_family"]
    sections = []
    for label, current, nearest in (("Matched MAE", mae, nearest_mae),
                                    ("Matched sensitivity", sens, nearest_sens)):
        merged = nearest.merge(current, on=key, suffixes=("_nearest", "_interpolated"), how="outer")
        matched = merged[merged["status"].eq("matched_by_interpolation")].copy()
        nearest_sig = int(nearest["significantly_lower"].fillna(False).sum())
        interpolated_sig = int(current["significantly_lower"].fillna(False).sum())
        direction_change = matched[(np.sign(matched["cess_minus_baseline_nearest"]) !=
                                    np.sign(matched["cess_minus_baseline_interpolated"]))]
        no_match = current[current["status"].eq("not_matchable")]
        sig_change = matched[(matched["significantly_lower_nearest"].fillna(False) !=
                              matched["significantly_lower_interpolated"].fillna(False))]
        multiple_roots = int((current["root_count"] > 1).sum())
        minimum_bootstrap_matchable = float(current["bootstrap_matchable_fraction"].min())
        median_bootstrap_matchable = float(current["bootstrap_matchable_fraction"].median())
        sections += [f"## {label}", "",
                     f"- Nearest-grid significant wins: {nearest_sig}/{len(nearest)}.",
                     f"- Interpolated significant wins: {interpolated_sig}/{len(current)}; "
                     f"{len(no_match)} settings are not matchable without extrapolation.",
                     f"- Effect-direction changes among matchable settings: {len(direction_change)}.",
                     f"- Significant-win status changes: {len(sig_change)}.", ""]
        sections += [f"- Multiple observed interpolation roots: {multiple_roots}/{len(current)}.",
                     f"- Bootstrap matchable fraction: minimum {minimum_bootstrap_matchable:.4f}, "
                     f"median {median_bootstrap_matchable:.4f}.", ""]
        if len(no_match):
            sections += ["Settings no longer matchable:", ""] + [
                f"- {r.dataset} / {r.query_model} / horizon {int(r.horizon)} / {r.baseline_family}"
                for _, r in no_match.iterrows()] + [""]
        if len(direction_change):
            sections += ["Effect-direction changes:", ""] + [
                f"- {r.dataset} / {r.query_model} / horizon {int(r.horizon)} / {r.baseline_family}: "
                f"nearest {r.cess_minus_baseline_nearest:+.6f}, interpolated {r.cess_minus_baseline_interpolated:+.6f}."
                for _, r in direction_change.iterrows()] + [""]
        if len(sig_change):
            sections += ["Significance-status changes:", ""] + [
                f"- {r.dataset} / {r.query_model} / horizon {int(r.horizon)} / {r.baseline_family}: "
                f"nearest significant={bool(r.significantly_lower_nearest)}, "
                f"interpolated significant={bool(r.significantly_lower_interpolated)}."
                for _, r in sig_change.iterrows()] + [""]
    return "\n".join([
        "# Nearest-grid versus continuously interpolated matching", "",
        "The original nearest-grid files are retained unchanged. The corrected analysis treats each "
        "baseline lambda path as a piecewise-linear curve. Matching is allowed only when an adjacent "
        "segment brackets the CESS target metric; no extrapolation is used. If a non-monotone curve "
        "has multiple intersections, the legal intersection closest to that replicate's nearest "
        "grid point is selected; ties use the lower lambda. "
        "Every source-task cluster bootstrap replicate recomputes the CESS target, baseline curve, "
        "intersection, and comparison. Confidence intervals are computed from matchable replicates; "
        "the matchable fraction is reported for every row.", "",
        *sections]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args(); out = args.analysis_dir
    targets = [out / "matched_mae_interpolated.csv",
               out / "matched_sensitivity_interpolated.csv",
               out / "matched_comparison_sensitivity_analysis.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite an existing corrected output")
    source = out / "real_trajectory_estimates.csv.gz"
    frame = pd.read_csv(source, low_memory=False)
    settings = frame[["dataset", "query_model", "horizon"]].drop_duplicates().sort_values(
        ["dataset", "query_model", "horizon"])
    mae_rows=[]; sens_rows=[]; counter=0
    for row in settings.itertuples(index=False):
        keys=(str(row.dataset),str(row.query_model),int(row.horizon))
        for family in BASELINE_FAMILIES:
            data=prepare_setting(frame,*keys,family)
            mae_rows.append(analyze_one(data,keys,family,"mae",SEED+counter));counter+=1
            sens_rows.append(analyze_one(data,keys,family,"sensitivity",SEED+counter));counter+=1
    mae=pd.DataFrame(mae_rows);sens=pd.DataFrame(sens_rows)
    mae.to_csv(targets[0],index=False);sens.to_csv(targets[1],index=False)
    # Only the final interpolated comparisons are part of the paper release.
    # Historical nearest-grid results are not inputs to the computation above.
    targets[2].write_text("# Interpolated matching\n\nSee the two generated CSV files. "
                          "Historical nearest-grid comparisons are excluded from this release.\n", encoding="utf-8")
    metadata={"cpu_only":True,"model_calls":0,"latex_modified":False,"bootstrap_replicates":BOOT,
              "bootstrap_unit":"source task/review","seed":SEED,"root_selection":"intersection closest to replicate-specific nearest grid point; lower-lambda tie break",
              "extrapolation":False,"input":str(source),"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    (out/"matched_interpolation_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"matched_mae_rows":len(mae),"matched_mae_matchable":int(mae.status.eq("matched_by_interpolation").sum()),"matched_mae_significant_lower":int(mae.significantly_lower.sum()),"matched_sensitivity_rows":len(sens),"matched_sensitivity_matchable":int(sens.status.eq("matched_by_interpolation").sum()),"matched_sensitivity_significant_lower":int(sens.significantly_lower.sum())},indent=2))

if __name__ == "__main__":
    main()
