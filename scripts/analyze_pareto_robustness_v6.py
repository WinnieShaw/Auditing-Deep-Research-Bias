#!/usr/bin/env python3
"""Audit and reproduction gate for the v6 CESS Pareto/robustness analysis.

This script is deliberately CPU-only.  It reads immutable completed trajectories
and prior audit tables, writes a new analysis directory, and never changes a
source trajectory, configuration, or manuscript.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 2026092306
OUTPUT_NAME = "outputs/analysis_v6_pareto_robustness"


def jsonl_schema(path: Path) -> tuple[int, dict]:
    count = 0
    first = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            count += 1
            if first is None:
                first = json.loads(line)
    if first is None:
        raise ValueError(f"empty jsonl: {path}")
    return count, first


def csv_schema(path: Path) -> tuple[int, dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        first = next(rows, None)
        if first is None:
            raise ValueError(f"empty csv: {path}")
        return 1 + sum(1 for _ in rows), first


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path, label: str, split: str, kind: str) -> dict:
    if path.suffix == ".jsonl":
        rows, example = jsonl_schema(path)
    elif path.suffix == ".csv":
        rows, example = csv_schema(path)
    else:
        raise ValueError(path)
    keys = sorted(example)
    lower = set(keys)
    return {
        "label": label,
        "path": str(path),
        "file_kind": kind,
        "split": split,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "fields": ";".join(keys),
        "has_full_candidate_probability_vector": bool({"candidate_probabilities", "base_agent_probabilities"} & lower),
        "has_chosen_selection_probability": "selection_probability" in lower or "minimum_selection_probability" in lower,
        "has_continuation_or_survival_probability": bool({"survival_probability", "minimum_survival_probability", "stop_probability"} & lower),
        "has_outcome_prediction": bool({"outcome_regression", "global_outcome_regression"} & lower),
        "has_true_candidate_pool_target": "target" in lower,
    }


def overall_ms2_metrics(path: Path) -> dict[str, float]:
    d = pd.read_csv(path)
    d = d[d.query_model == "overall_equal_model_average"].set_index("method")
    return {
        "opened_mae": float(d.loc["Opened Mean", "task_macro_mae"]),
        "opened_sensitivity": float(d.loc["Opened Mean", "absolute_paired_ranking_sensitivity"]),
        "cess_mae": float(d.loc["CESS", "task_macro_mae"]),
        "cess_sensitivity": float(d.loc["CESS", "absolute_paired_ranking_sensitivity"]),
    }


def perspectrum_metrics(path: Path) -> dict[str, float]:
    d = pd.read_csv(path)
    d = d[d.dataset == "PERSPECTRUM 10-round"].set_index("method")
    return {
        "opened_mae": float(d.loc["Opened Mean", "task_mae"]),
        "cess_mae": float(d.loc["Frozen CESS", "task_mae"]),
        "opened_sensitivity": float(d.loc["Opened Mean", "absolute_paired_ranking_sensitivity"]),
        "cess_sensitivity": float(d.loc["Frozen CESS", "absolute_paired_ranking_sensitivity"]),
    }


def true_agent_selection(path: Path) -> dict[str, float]:
    d = pd.read_csv(path)
    d = d[(d.endpoint == "real_trajectory_total_selection") & (d.metric == "MAE")].set_index("model")
    return {model: float(d.loc[model, "value"]) for model in ("olmo3_7b", "qwen2_5_32b")}


def analyze_overlap(raw: Path, out: Path) -> None:
    """Aggregate the newly re-sampled paired overlap experiment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = sorted((raw / "task_rows").glob("*__full.csv"))
    if len(paths) != 120:
        raise ValueError(f"expected 120 full overlap files, found {len(paths)}")
    d = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    numeric = ["target", "opened_mean", "sequential_dr", "global_cess", "ess", "max_weight", "min_propensity"]
    d[numeric] = d[numeric].astype(float)
    d["epsilon"] = d.scenario.str.extract(r"eps(\d+)")[0].astype(float) / 1000.0
    d.loc[d.scenario.str.contains("deterministic"), "epsilon"] = 0.0
    d["cluster"] = d.seed.astype(str) + ":" + d.task_id.astype(str)
    rows = []
    for scenario, g in d.groupby("scenario", sort=False):
        epsilon = float(g.epsilon.iloc[0])
        pair = g.pivot(index="cluster", columns="z", values="global_cess")
        target = g.groupby("cluster").target.mean()
        est = g.groupby("cluster").global_cess.mean()
        opened = g.groupby("cluster").opened_mean.mean()
        if set([-1, 1]).issubset(pair.columns):
            sensitivity = (pair[1] - pair[-1]).abs()
        else:
            sensitivity = pd.Series(np.nan, index=est.index)
        for method, values in (("Opened Mean", opened), ("Sequential DR", est * 0 + g.groupby("cluster").sequential_dr.mean()), ("Full CESS", est)):
            error = values - target
            rows.append({
                "scenario": scenario, "epsilon": epsilon, "method": method,
                "source_task_clusters": len(values), "bias": float(error.mean()),
                "task_macro_mae": float(error.abs().mean()),
                "ranking_sensitivity": float(sensitivity.mean()),
                "direction_disagreement": float((np.sign(values) != np.sign(target)).mean()),
                "win_rate_vs_opened": float((error.abs() < (opened-target).abs()).mean()) if method != "Opened Mean" else np.nan,
                "mean_ess": float(g.ess.mean()), "median_weight": float(g.max_weight.median()),
                "p95_weight": float(g.max_weight.quantile(.95)), "max_weight": float(g.max_weight.max()),
                "minimum_chosen_propensity": float(g.min_propensity.min()),
                "status": "point estimators not identified at epsilon=0" if epsilon == 0 else "positive exploration",
            })
    table = pd.DataFrame(rows)
    table.to_csv(out / "overlap_stress_test.csv", index=False)
    # The deterministic run logs task-level rows but no document identities.
    zero = {
        "scenario": "overlap_eps000_deterministic",
        "point_estimator_status": "NOT IDENTIFIED: deterministic selection has zero probability for non-maximizing candidates",
        "per_round_inaccessible_mass_from_known_N30": 29 / 30,
        "sharp_support_interval_width": "NOT IDENTIFIABLE FROM SAVED TASK-LEVEL ROWS: selected document IDs and outcome bounds by support are absent",
        "target_coverage": "NOT IDENTIFIABLE FROM SAVED TASK-LEVEL ROWS",
        "sign_identification_rate": "NOT IDENTIFIABLE FROM SAVED TASK-LEVEL ROWS",
    }
    (out / "zero_support_identification.json").write_text(json.dumps(zero, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    nonzero = table[table.epsilon > 0]
    for method, q in nonzero.groupby("method"):
        axes[0].plot(q.epsilon, q.task_macro_mae, marker="o", label=method)
        axes[1].plot(q.epsilon, q.mean_ess, marker="o", label=method)
    axes[0].set(xscale="log", xlabel="Uniform exploration $\\epsilon$", ylabel="Task-macro MAE")
    axes[1].set(xscale="log", xlabel="Uniform exploration $\\epsilon$", ylabel="Mean trajectory ESS")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)
    fig.savefig(out / "figures/overlap_vs_mae_ess.pdf", bbox_inches="tight")
    fig.savefig(out / "figures/overlap_vs_mae_ess.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_NAME)
    parser.add_argument("--analyze-overlap-existing", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if args.analyze_overlap_existing:
        if not out.is_dir():
            raise FileNotFoundError(out)
        (out / "figures").mkdir(exist_ok=True)
        analyze_overlap(out / "controlled_overlap_raw", out)
        print(json.dumps({"output": str(out), "overlap_analysis": "complete"}, indent=2))
        return
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {out}")
    out.mkdir(parents=True)

    inventory_paths = [
        ("MS2 Qwen cells", ROOT / "evidence_selection_benchmark/results/ms2_external/sequential_confirmatory_v1_qwen2_5_32b/cells.jsonl", "test", "trajectory cells"),
        ("MS2 OLMo cells", ROOT / "evidence_selection_benchmark/results/ms2_external/sequential_confirmatory_v1_olmo3_7b/cells.jsonl", "test", "trajectory cells"),
        ("PERSPECTRUM horizon10 cells", ROOT / "evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1/cells.jsonl", "test", "trajectory cells"),
        ("PERSPECTRUM horizon10 traces", ROOT / "evidence_selection_benchmark/results/perspectrum_nonmedical_v1/query_horizon10_v1/traces.jsonl", "test", "trajectory traces"),
        ("CausalSim Search rows", ROOT / "evidence_selection_benchmark/results/causal_search_scm_v1/task_rows/core_k4_t4__seed101__full.csv", "simulated test", "task rows"),
        ("Survival CausalSim rows", ROOT / "evidence_selection_benchmark/results/causal_search_scm_survival_confirmatory_v2/task_rows/survival_core_k4_t4__seed3001__full.csv", "simulated test", "task rows"),
    ]
    inventory = pd.DataFrame([describe(path, label, split, kind) for label, path, split, kind in inventory_paths])
    inventory.to_csv(out / "data_inventory.csv", index=False)

    ms2_source = ROOT / "analysis/submission_final_checks_v3/method_metrics.csv"
    p_source = ROOT / "analysis/fair_shrinkage_selection_order_v1/fair_shrinkage_method_metrics.csv"
    agent_source = ROOT / "outputs/interpretability_v1_true_agent_ms2/analysis_v5/local_total_metrics.csv"
    ms2 = overall_ms2_metrics(ms2_source)
    p = perspectrum_metrics(p_source)
    agent = true_agent_selection(agent_source)
    checks = [
        ("MS2 Opened Mean MAE", 0.1743, ms2["opened_mae"]),
        ("MS2 CESS MAE", 0.1583, ms2["cess_mae"]),
        ("MS2 Opened Mean sensitivity", 0.2445, ms2["opened_sensitivity"]),
        ("MS2 CESS sensitivity", 0.1481, ms2["cess_sensitivity"]),
        ("PERSPECTRUM ten-round Opened Mean MAE", 0.4108, p["opened_mae"]),
        ("PERSPECTRUM ten-round CESS MAE", 0.3103, p["cess_mae"]),
        ("true-agent OLMo selection-effect CESS MAE", 0.151, agent["olmo3_7b"]),
        ("true-agent Qwen selection-effect CESS MAE", 0.146, agent["qwen2_5_32b"]),
    ]
    rows = []
    for name, expected, observed in checks:
        tolerance = 5e-5 if name.startswith("MS2") or name.startswith("PERSPECTRUM") else 5e-3
        rows.append({"quantity": name, "expected_display_or_approximation": expected, "recomputed": observed,
                     "absolute_difference": abs(expected - observed), "tolerance": tolerance,
                     "pass": abs(expected - observed) <= tolerance})
    reproduction = pd.DataFrame(rows)
    reproduction.to_csv(out / "reproduction_check.csv", index=False)

    source_task_notes = [
        "MS2 Table-1 metric source has 200 source reviews and two query models; the saved main table macro-averages models within review.",
        "PERSPECTRUM ten-round source has 36 held-out claims and 2,592 completed episodes; its candidate pool is fixed per task.",
        "PERSPECTRUM traces record chosen-document and survival probabilities but not the full candidate probability vector, so temperature/noise propensity replay from these saved logs is NOT IDENTIFIABLE FROM SAVED DATA.",
        "True-agent trajectories are a separate four-condition mechanism study; their CESS MAEs are used only for the requested anchor reproduction, not pooled into the Deep Research Pareto analysis.",
        "The saved controlled simulator task rows include target, outcome predictions, propensity diagnostics, weights, and estimators. They are eligible for CPU overlap and outcome-model stress analyses, but they are not a substitute for missing real-agent candidate probability vectors.",
        "The pre-existing MS2 development/test audit excluded 18 overlapping review IDs per query model before tuning its fair shrinkage baselines. This script does not silently assert a document-level leakage check where document IDs are absent from the aggregate table.",
    ]
    feasible = [
        ("A / MS2 Pareto from existing estimator fields and development-selected anchors", "YES"),
        ("A / PERSPECTRUM Pareto from ten-round completed cells and frozen five-round development anchors", "YES"),
        ("A / PERSPECTRUM full per-round propensity temperature/noise perturbation", "NOT IDENTIFIABLE FROM SAVED DATA"),
        ("B / controlled finite-population overlap re-sampling", "YES, CPU simulator"),
        ("B / real trajectory estimated-propensity model", "NOT IDENTIFIABLE FROM SAVED DATA: no stored pre-selection candidate features and vectors"),
        ("B / controlled outcome-model robustness", "YES"),
    ]
    lines = [
        "# v6 data inventory and reproduction gate", "",
        "## Inputs", "",
        inventory.to_markdown(index=False), "",
        "## Split and feasibility audit", "",
        *[f"- {x}" for x in source_task_notes], "",
        "| Planned analysis | Existing-data status |", "|---|---|",
        *[f"| {name} | {status} |" for name, status in feasible], "",
        "## Reproduction gate", "",
        reproduction.to_markdown(index=False), "",
        "The gate passes only if every displayed reference is reproduced within the stated stored-display tolerance. Subsequent analyses must use the exact source and aggregation documented above.",
    ]
    (out / "data_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "script": "scripts/analyze_pareto_robustness_v6.py",
        "seed": SEED,
        "cpu_only": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reproduction_gate_passed": bool(reproduction["pass"].all()),
        "inputs": {row["label"]: {"path": row["path"], "sha256": row["sha256"]} for row in inventory.to_dict("records")},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "reproduction_gate_passed": manifest["reproduction_gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
