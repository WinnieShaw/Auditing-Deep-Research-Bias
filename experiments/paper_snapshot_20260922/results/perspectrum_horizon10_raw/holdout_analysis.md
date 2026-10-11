# PERSPECTRUM strict-visibility LLM-query diagnostic

The development-selected search correction was evaluated on 36 PERSPECTRUM test claims with Qwen2.5-32B and Llama-3.3-70B query agents. This is a post-holdout mechanism diagnostic; no parameter was selected from query performance.

| Metric | Naive opened mean | Risk-adaptive CESS |
|---|---:|---:|
| Episode MAE | 0.4108 | 0.3103 |
| Episode RMSE | 0.5078 | 0.3905 |
| Wrong-direction rate | 0.4186 | 0.5208 |
| Task-cell bias MAE | 0.3764 | 0.2644 |
| Opposite-ranking instability | 0.9686 | 0.2925 |

Cluster-bootstrap method-minus-naive MAE difference: -0.1004, 95% CI [-0.1597, -0.0359]. Query parse rates are Qwen 1.000 and Llama 1.000.
