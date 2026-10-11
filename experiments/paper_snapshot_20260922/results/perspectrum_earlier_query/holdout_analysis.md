# PERSPECTRUM strict-visibility LLM-query diagnostic

The development-selected search correction was evaluated on 36 PERSPECTRUM test claims with Qwen2.5-32B and Llama-3.3-70B query agents. This is a post-holdout mechanism diagnostic; no parameter was selected from query performance.

| Metric | Naive opened mean | Risk-adaptive CESS |
|---|---:|---:|
| Episode MAE | 0.4734 | 0.3398 |
| Episode RMSE | 0.5779 | 0.4239 |
| Wrong-direction rate | 0.4190 | 0.5077 |
| Task-cell bias MAE | 0.4280 | 0.2884 |
| Opposite-ranking instability | 1.0651 | 0.3185 |

Cluster-bootstrap method-minus-naive MAE difference: -0.1336, 95% CI [-0.1932, -0.0705]. Query parse rates are Qwen 1.000 and Llama 1.000.
