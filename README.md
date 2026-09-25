# Auditing Deep Research Bias

This repository accompanies our project **“Search Is a Sampling Policy: Causal Correction of Evidence Selection Bias in Deep Research Agents.”**

Deep Research agents do not passively summarize a fixed collection of documents. They decide what to query, which results to open, and when to stop. As a result, the evidence used in a report can be a biased sample of the available evidence—even when every citation is valid and faithfully represented.

We study this problem as **sequential evidence selection** and introduce **CESS**, an estimator that uses logged document-selection and continuation probabilities to estimate the conclusion supported by the full candidate pool. The method combines sequential probability correction with finite-budget stabilization.

## Evaluation

Our experiments cover:

- controlled evidence-selection settings on **MS2**;
- cross-domain and long-horizon evaluation on **PERSPECTRUM**;
- ranking, overlap, and estimator-robustness analyses;
- transfer to a public Deep Research agent based on **LangChain Open Deep Research**.

We evaluate both pool-target accuracy and sensitivity to document ranking, and report comparisons with Opened Mean, outcome regression, sequential IPW/DR, and tuned shrinkage baselines.

## Repository Status

The code, experiment configurations, and reproducible result artifacts are being organized for public release. Setup and reproduction commands will be added here with the release.

## Citation

Citation information will be added when the paper is publicly available.
