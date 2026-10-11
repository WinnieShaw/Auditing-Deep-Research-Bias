# Reproducing the paper experiments

This release is scoped to the supplied manuscript, **Search Shapes Conclusions:
Auditing Evidence Selection Bias in Deep Research Agents** (PDF revision 41).
The manuscript itself is not redistributed. Unused experiments, failed pilots,
historical reports and unverified numerical substitutes are excluded.

## Paper-to-artifact map

| Paper item | Released artifacts |
| --- | --- |
| Table 1: MS2, 200 questions, 1,200 trajectories | `experiments/paper_snapshot_20260922/results/ms2_table1_raw_*`; `tables/main_ms2_reproduction` under that same snapshot |
| Tables 2 and 11: interpolated matched comparisons | `experiments/interpolated_matching_v6/*.csv`; `experiments/pareto_robustness_v6/real_trajectory_estimates.csv.gz`; `research_scripts/analyze_interpolated_matching_v6.py` |
| Figure 3 and MS2 budget analysis | `experiments/paper_snapshot_20260922/figures/fig_budget_efficiency.pdf`; `reports/ms2_sequential_robustness_statistics.json`; `scripts/analyze_ms2_sequential_robustness.py`; `evidence_selection_benchmark/results/ms2_external/sequential_confirmatory_v1_*` |
| Table 3, public-agent transfer and Appendix D | `outputs/public_agent_light_v4/test`, `analysis_final`, frozen configuration, development calibration and exact-input cache; `scripts/run_public_agent_light_v4.py`, `analyze_public_agent_light_v4.py` |
| Table 4 and Appendix C: paired interventions | `outputs/interpretability_v1_true_agent_ms2/{qwen2_5_32b,olmo3_7b}` and selected `analysis_v4`/`analysis_v5` tables; `scripts/analyze_interpretability_v5.py`; `configs/interpretability_v1_true_agent_ms2.yaml` |
| Tables 6–7: five-round PERSPECTRUM | `experiments/paper_snapshot_20260922/results/perspectrum_earlier_query`; `scripts/run_perspectrum_query_v1.py`; `configs/perspectrum_query_v1.yaml` |
| Tables 8–10: ten-round PERSPECTRUM, clipping and prefixes | `experiments/paper_snapshot_20260922/results/perspectrum_horizon10_raw`, `analysis/horizon10_audit_v1`, and `tables/fair_shrinkage_and_clipping_order`; `configs/perspectrum_query_horizon10_v1.yaml` |
| Table 12 and limited-support analysis | `experiments/pareto_robustness_v6/controlled`, corresponding CSVs in its parent; `scripts/v6_controlled_robustness_final.py`, `v6_export_zero_support_candidates.py`, `run_causal_search_scm.py` |
| Candidate pools, hidden evaluation labels and model features | `evidence_selection_benchmark/data` and `features`; only the relevant MS2/PERSPECTRUM subsets and model features are included |

The word `private` in outcome filenames means hidden from the search agent
during evidence acquisition. These are research evaluation labels, not account
credentials or private patient records. Do not expose them to query generation.

## Verified CPU-only reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -p 'test_*.py'
python scripts/verify_archived_results.py
python scripts/verify_paper_release.py
```

The first verification recomputes task-macro metrics from frozen estimates and
MS2 cells. The second recomputes public-agent metrics from 216 logged episodes,
checks Table 3 at its reported precision, and checks the 4,800-trajectory count.
They do not call an LLM, tune a method or certify causal identification.

To reproduce the paper's matched comparisons, use a fresh output directory:

```bash
mkdir -p build/matching
cp experiments/pareto_robustness_v6/real_trajectory_estimates.csv.gz build/matching/
python research_scripts/analyze_interpolated_matching_v6.py --analysis-dir build/matching
```

Both 28-row output tables were reproduced against the released CSVs using
10,000 bootstrap replicates. Historical nearest-grid comparisons are excluded;
they are not needed to compute the interpolated comparisons.

To rerun the full public-agent statistical analysis without model calls:

```bash
mkdir -p build/public_agent/test
cp outputs/public_agent_light_v4/frozen_test_config.json build/public_agent/
cp outputs/public_agent_light_v4/test/episodes.jsonl build/public_agent/test/
python scripts/analyze_public_agent_light_v4.py --output build/public_agent
```

The code refuses to overwrite populated analysis directories. It also renders
diagnostic plots; these are generated outputs, not additional published paper
figures. For the paper budget data, run:

```bash
python scripts/analyze_ms2_sequential_robustness.py --output build/ms2_budget.json
```

## Model-execution code

The original experiment runners and their imported local dependencies are in
`scripts/`, `src/`, `configs/` and `protocol/`. Some dependency filenames contain
earlier project names such as RealEffect or public-agent v2/v3: the final runners
import their utility functions. Their obsolete experiment outputs are not included.
Set `<MODEL_DIR>` to your own model-weight location before model execution.
Model weights, GPU runtimes and font binaries are not redistributed.

Public-agent execution also requires the upstream Open Deep Research project
at revision `1b7d2e80db9faa586165c60e09096dbbfd483a64`, placed at
`third_party/open_deep_research`, and its dependencies. Use the upstream project's
own installation instructions and license. Historical path defaults and hash
checks must be reconciled with the release layout before fresh model runs;
only the CPU verification entry points above have been tested in this checkout.

## Estimator configurations

The MS2 primary estimator shrinks the raw correction before clipping, with
model-specific frozen coefficients. The public-agent and ten-round PERSPECTRUM
versions clip before shrinking toward the pool prediction. The five-round
PERSPECTRUM version uses Opened Mean as its anchor. These configurations are
not silently unified. `cess_audit.py` is the portable clip-then-shrink audit
implementation; it is not a replacement for every paper-specific estimator.

## Intentionally withheld / not yet available

- Figure 2(a)'s source numbers reproduce a signed average ranking contrast,
  while the manuscript describes an absolute ranking difference. The disputed
  figure and simulator result package are withheld pending metric clarification.
- Table 5 and Appendix A.5's exact simulation export was not confirmed. The
  available older fixed-shrinkage simulation uses 0.06, whereas the manuscript
  specifies 0.01. That export is not substituted or published.
- The original drawing source for the exact Figure 3 PDF has not been located.
  The matching PDF, numerical summaries, raw trajectories and analysis code are
  included; the original rendering script is not claimed to be available.

These gaps mean this is a scoped artifact release, not a claim that every
manuscript item has been reproduced end to end. No experiment was rerun or
modified to force agreement with a reported number.

## Data provenance and integrity

MS2 and PERSPECTRUM materials retain their upstream rights; this repository
does not grant a new blanket license for third-party text. The release contains
the processed candidate subsets used by the included experiments, derived
labels, trajectory logs, predictions, evaluation tables and necessary caches.
Acquire the original datasets from their official distributions when needed.

Private host paths have been replaced by portable placeholders. Original run
hashes refer to original artifacts and may differ after path redaction.
`docs/artifact_inventory.json` gives SHA-256 values of the actual released files.
The source PDFs/TeX, internal reviews, unused plots, failed runs and credentials
are not uploaded.
