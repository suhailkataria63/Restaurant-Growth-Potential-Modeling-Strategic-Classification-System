# Clustering Model Evaluation

## Selected Clustering Solution

- Selected method: `kmeans`
- Number of clusters: `3`

## Quality Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| Silhouette Score | 0.1930 | Higher is better (range: -1 to 1). Measures separation vs cohesion. |
| Calinski-Harabasz Score | 390.2882 | Higher is better. Measures between-cluster dispersion relative to within-cluster dispersion. |
| Davies-Bouldin Score | 1.7151 | Lower is better. Captures average cluster overlap/similarity. |

## Interpretation Guidance

- A stronger solution typically shows **higher Silhouette**, **higher Calinski-Harabasz**, and **lower Davies-Bouldin**.
- Compare these metrics with the k-sweep CSV and plots to validate that the chosen structure is stable and interpretable.