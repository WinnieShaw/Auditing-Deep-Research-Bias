# Auditing Deep Research Bias

Code and scoped experiment artifacts for **Search Shapes Conclusions: Auditing
Evidence Selection Bias in Deep Research Agents**.

**Experiment release:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) maps the
available code, trajectory data, corrected analyses and figures. Start there for
paper-result reproduction and the explicit list of withheld or missing items.
[`experiments/`](experiments/) contains versioned exports; research sources are
in [`scripts/`](scripts/) and [`research_scripts/`](research_scripts/).
Only manuscript-related experiments and their necessary dependencies are included.
Unused historical experiments and inconsistent exports are excluded.

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

The portable audit package does not create probabilities for unlogged actions:
zero support is rejected and must be addressed in the data-collection protocol.
The supplied example is an engineering smoke test, not a paper result.
The release includes the relevant processed candidate pools, model features,
216 public-agent episodes and 4,800 paired-intervention trajectories. It does
not include model weights or GPU runtime dependencies. See the reproduction
guide for tested entry points, exact table mappings and withheld items.

Public users can run the estimator on the documented JSONL schema and reproduce
selected numerical analyses directly from the archived data. Complete end-to-end
model reruns require the additional inputs listed in the reproduction guide.

## Citation

Citation information will be added with the public paper release.
