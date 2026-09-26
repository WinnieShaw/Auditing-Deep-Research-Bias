# Public JSONL schema

The runner consumes one JSON object per episode. An episode is one task under
one ranking condition and one logged search trajectory. The release does not
ship the underlying documents or model outputs; users must provide data they
are authorized to share.

Required episode fields:

```json
{
  "task_id": "task-001",
  "ranking": "supporting_first",
  "target": 0.25,
  "outcome_regression": 0.20,
  "searches": [
    {
      "candidate_count": 4,
      "selected_probability": 0.50,
      "continuation_probability": 1.0,
      "selected_outcome": 0.30,
      "selected_outcome_prediction": 0.22
    }
  ]
}
```

`candidate_count` is the candidate-pool size used by the caller. The two
probabilities must be the values logged before observing the selected outcome:
the selection probability for the opened item and the continuation probability
for reaching that round. They must be in `(0, 1]`. A zero probability means the
support condition is absent and is rejected rather than repaired with epsilon.

`outcome_regression` is an optional outcome-model anchor on the same target
scale. If it is omitted, the first search must contain a `candidates` list with
`outcome_prediction` values. `target` is only for evaluation and must not be
used to construct a test-time anchor.

## Estimator order

For round `t`, the base inverse-probability weight is

`w_t = 1 / (N_t p_t c_t)`.

The sequential IPW estimate averages `w_t y_t`; the sequential DR estimate is
`m_0 + mean[w_t (y_t - m_t)]`. CESS is

`clip(m_0 + lambda * (clip(DR_raw) - m_0), -1, 1)`.

Thus clipping occurs before the CESS shrinkage step. Self-normalized variants
normalize by `sum(w_t)`. Weight-clipped variants cap `w_t` only in their own
estimate. The code also reports simple Opened Mean, OR, and two transparent
shrinkage baselines for comparison.

## Interpretation boundary

The package estimates logged-policy quantities. It does not identify an
unlogged probability, recover candidates that were never represented in the
input, or prove an internal model mechanism. Task-cluster bootstrap intervals
describe uncertainty of the reported task-level metric; they are not
single-task confidence intervals for an unobserved population target.

