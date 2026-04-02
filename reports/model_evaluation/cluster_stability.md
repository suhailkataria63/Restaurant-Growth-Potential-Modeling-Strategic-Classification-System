# Cluster Stability Evaluation

## Setup

- Number of clusters evaluated: `3`
- K-Means reruns with different seeds: `10` (baseline seed `42`)
- Subsample stability: `20` runs with `80%` random samples
- Stability metrics: **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**.

## Stability Results

| Check | ARI Mean | ARI Std | NMI Mean | NMI Std | Mean Centroid Shift |
|---|---:|---:|---:|---:|---:|
| Seed reruns vs baseline | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 80% subsamples vs baseline | 0.9761 | 0.0101 | 0.9651 | 0.0127 | 0.1174 |

## Robustness Interpretation

- Overall mean ARI: `0.9841`
- Overall mean NMI: `0.9767`
- Stability verdict: **Highly robust: cluster assignments are very stable across reruns/subsamples.**

Higher ARI/NMI values indicate stronger agreement of cluster membership under reruns and perturbations.