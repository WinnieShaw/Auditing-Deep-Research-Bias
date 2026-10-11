# PERSPECTRUM LLM-query frozen-transfer audit

The original pre-holdout CESS uses the opened-evidence mean as its anchor and lambda=0.20. It is evaluated here without retuning on the completed Qwen2.5-32B and Llama-3.3-70B query trajectories.

| Metric | Opened mean | Frozen transferred CESS |
|---|---:|---:|
| Episode MAE | 0.4734 | 0.4250 |
| Episode RMSE | 0.5779 | 0.5240 |
| Wrong-direction rate | 0.4190 | 0.4213 |
| Task-cell bias MAE | 0.4280 | 0.3638 |
| Opposite-ranking instability | 1.0651 | 0.8965 |

Topic-cluster bootstrap, frozen CESS minus opened mean:

- MAE: -0.0484, 95% CI [-0.0569, -0.0395].
- Wrong direction: +0.0023, 95% CI [-0.0116, +0.0147].
- Ranking instability: -0.1686, 95% CI [-0.1978, -0.1381].

The estimator reduces continuous error and ranking sensitivity while leaving direction error statistically unchanged. The later outcome-regression-anchored diagnostic has lower continuous error but worse directional transfer and is not the frozen cross-domain headline method.
