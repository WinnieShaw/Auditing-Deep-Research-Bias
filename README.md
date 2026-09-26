# Auditing Deep Research Bias

This repository contains a small, CPU-only post-processing implementation for
auditing sequential evidence selection in Deep Research agents.

An agent chooses queries, opens documents, and decides when to continue. The
observed documents can therefore be a selected sample of a larger candidate
pool. The public code computes transparent estimators from a trajectory whose
selection and continuation probabilities were logged before the outcome.

## Quick start

```bash
python3 run_analysis.py \
  --input examples/episodes.example.jsonl \
  --output results \
  --bootstrap-replicates 1000
```

The command writes `estimates.csv` and `summary.json`. A development-set anchor
can be supplied explicitly with `--global-mean`; it is never inferred from
test targets by the runner.

## What is implemented

`cess_audit.py` provides Opened Mean, outcome regression, two simple shrinkage
baselines, sequential IPW, self-normalized IPW, sequential DR,
self-normalized DR, weight-clipped DR, and CESS. CESS follows the documented
order: compute the raw sequential DR correction, clip it to the outcome scale,
then shrink it toward the outcome-regression anchor. `bootstrap.py` computes
task-cluster percentile intervals for task-macro MAE and ranking sensitivity.

The input contract and formulas are in [`docs/data_schema.md`](docs/data_schema.md).
Run the standard-library tests with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Scope and reproducibility boundary

This is a public audit package, not the private search-agent runner. It does
not contain model weights, private candidate documents, search caches, API
credentials, internal machine paths, or unreleased result files. It also does
not create probabilities for unlogged actions: zero support is rejected and
must be addressed in the data-collection protocol. The supplied example is an
engineering smoke test, not a paper result.

The full research evaluation uses authorized task data and separately managed
agent configurations. Public users can reproduce the estimator and all
post-processing once they have an equivalent, shareable JSONL export.

## Citation

Citation information will be added with the public paper release.
