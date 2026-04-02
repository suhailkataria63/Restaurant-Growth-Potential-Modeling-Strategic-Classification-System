from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

matplotlib.use("Agg")

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

RANDOM_STATE = 42


def _ensure_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _load_features_and_labels(
    clustered_data_path: str,
    feature_matrix_path: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, str, int]:
    clustered_df = pd.read_csv(clustered_data_path)
    feature_matrix_df = pd.read_csv(feature_matrix_path)

    if "selected_cluster" not in clustered_df.columns:
        raise ValueError("`selected_cluster` column not found in clustered output.")
    if len(clustered_df) != len(feature_matrix_df):
        raise ValueError("Row count mismatch between clustered output and feature matrix.")

    labels = clustered_df["selected_cluster"].to_numpy(dtype=int)
    features = feature_matrix_df.to_numpy(dtype=float)
    selected_method = (
        str(clustered_df["selected_method"].iloc[0])
        if "selected_method" in clustered_df.columns
        else "unknown"
    )
    n_clusters = int(np.unique(labels).size)
    if n_clusters < 2:
        raise ValueError("Evaluation requires at least 2 unique clusters in the selected labels.")

    return clustered_df, features, labels, selected_method, n_clusters


def _evaluate_labels(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Evaluation requires at least 2 unique clusters in the selected labels.")

    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
    }


def _evaluate_kmeans_range(features: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(features)
        rows.append(
            {
                "k": int(k),
                "silhouette_score": float(silhouette_score(features, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def _plot_metric_by_k(
    scores_df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
    prefer: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(scores_df["k"], scores_df[metric_col], marker="o", linewidth=2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)

    if prefer == "max":
        best_idx = scores_df[metric_col].idxmax()
    else:
        best_idx = scores_df[metric_col].idxmin()

    best_k = int(scores_df.loc[best_idx, "k"])
    best_val = float(scores_df.loc[best_idx, metric_col])
    ax.scatter([best_k], [best_val], color="#dc2626", s=55, zorder=3)
    ax.annotate(
        f"best k={best_k}",
        xy=(best_k, best_val),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        color="#dc2626",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_metrics_markdown(
    metrics: Dict[str, float],
    selected_method: str,
    n_clusters: int,
    output_path: Path,
) -> None:
    lines = [
        "# Clustering Model Evaluation",
        "",
        "## Selected Clustering Solution",
        "",
        f"- Selected method: `{selected_method}`",
        f"- Number of clusters: `{n_clusters}`",
        "",
        "## Quality Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
        f"| Silhouette Score | {metrics['silhouette_score']:.4f} | Higher is better (range: -1 to 1). Measures separation vs cohesion. |",
        f"| Calinski-Harabasz Score | {metrics['calinski_harabasz_score']:.4f} | Higher is better. Measures between-cluster dispersion relative to within-cluster dispersion. |",
        f"| Davies-Bouldin Score | {metrics['davies_bouldin_score']:.4f} | Lower is better. Captures average cluster overlap/similarity. |",
        "",
        "## Interpretation Guidance",
        "",
        "- A stronger solution typically shows **higher Silhouette**, **higher Calinski-Harabasz**, and **lower Davies-Bouldin**.",
        "- Compare these metrics with the k-sweep CSV and plots to validate that the chosen structure is stable and interpretable.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _centroid_shift_against_reference(
    candidate_centers: np.ndarray,
    reference_centers: np.ndarray,
) -> float:
    if linear_sum_assignment is None:
        return float("nan")

    distances = np.linalg.norm(candidate_centers[:, None, :] - reference_centers[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(distances)
    return float(distances[row_idx, col_idx].mean())


def _run_seed_stability(
    features: np.ndarray,
    n_clusters: int,
    seeds: List[int],
) -> Tuple[pd.DataFrame, KMeans, np.ndarray]:
    baseline_seed = seeds[0]
    baseline_model = KMeans(n_clusters=n_clusters, n_init=20, random_state=baseline_seed)
    baseline_labels = baseline_model.fit_predict(features)

    rows = []
    for i, seed in enumerate(seeds):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = model.fit_predict(features)

        ari = float(adjusted_rand_score(baseline_labels, labels))
        nmi = float(normalized_mutual_info_score(baseline_labels, labels))
        rows.append(
            {
                "evaluation_type": "seed_rerun_vs_baseline",
                "iteration": int(i),
                "seed": int(seed),
                "sample_fraction": 1.0,
                "ari": ari,
                "nmi": nmi,
                "inertia": float(model.inertia_),
                "centroid_shift": _centroid_shift_against_reference(
                    model.cluster_centers_, baseline_model.cluster_centers_
                ),
            }
        )

    return pd.DataFrame(rows), baseline_model, baseline_labels


def _run_subsample_stability(
    features: np.ndarray,
    n_clusters: int,
    baseline_model: KMeans,
    baseline_labels: np.ndarray,
    sample_fraction: float,
    n_iterations: int,
    random_state: int,
) -> pd.DataFrame:
    if not (0 < sample_fraction < 1):
        raise ValueError("sample_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    n_rows = features.shape[0]
    sample_size = max(2, int(round(sample_fraction * n_rows)))

    rows = []
    for i in range(n_iterations):
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        sample_features = features[sample_idx]
        seed = int(random_state + 1000 + i)

        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        sample_labels = model.fit_predict(sample_features)
        baseline_sample_labels = baseline_labels[sample_idx]

        rows.append(
            {
                "evaluation_type": "subsample_vs_reference",
                "iteration": int(i + 1),
                "seed": seed,
                "sample_fraction": float(sample_fraction),
                "ari": float(adjusted_rand_score(baseline_sample_labels, sample_labels)),
                "nmi": float(normalized_mutual_info_score(baseline_sample_labels, sample_labels)),
                "inertia": float(model.inertia_),
                "centroid_shift": _centroid_shift_against_reference(
                    model.cluster_centers_, baseline_model.cluster_centers_
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_stability_metric(
    stability_df: pd.DataFrame,
    metric_col: str,
    output_path: Path,
    title: str,
) -> None:
    order = ["seed_rerun_vs_baseline", "subsample_vs_reference"]
    labels = ["Seed Re-runs", "80% Subsamples"]

    distributions = [
        stability_df.loc[stability_df["evaluation_type"] == eval_type, metric_col].dropna().to_numpy()
        for eval_type in order
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(
        distributions,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8"},
        medianprops={"color": "#1e3a8a", "linewidth": 2},
        whiskerprops={"color": "#1d4ed8"},
        capprops={"color": "#1d4ed8"},
    )

    for i, values in enumerate(distributions, start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.08, 0.08, values.size)
        ax.scatter(np.full(values.size, i) + jitter, values, s=18, alpha=0.7, color="#1e40af")

    ax.set_title(title)
    ax.set_ylabel(metric_col.upper())
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _stability_judgement(ari_mean: float, nmi_mean: float) -> str:
    if ari_mean >= 0.90 and nmi_mean >= 0.90:
        return "Highly robust: cluster assignments are very stable across reruns/subsamples."
    if ari_mean >= 0.75 and nmi_mean >= 0.75:
        return "Robust: cluster structure is stable with manageable variation."
    if ari_mean >= 0.60 and nmi_mean >= 0.60:
        return "Moderately stable: usable structure but sensitivity should be monitored."
    return "Low stability: cluster assignments are sensitive to reruns or sampling."


def _write_stability_markdown(
    stability_df: pd.DataFrame,
    output_path: Path,
    n_clusters: int,
    seeds: List[int],
    sample_fraction: float,
    n_subsamples: int,
) -> Dict[str, float]:
    grouped = (
        stability_df.groupby("evaluation_type", as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            centroid_shift_mean=("centroid_shift", "mean"),
        )
        .fillna(0.0)
    )

    seed_row = grouped[grouped["evaluation_type"] == "seed_rerun_vs_baseline"].iloc[0]
    subsample_row = grouped[grouped["evaluation_type"] == "subsample_vs_reference"].iloc[0]

    overall_ari_mean = float(stability_df["ari"].mean())
    overall_nmi_mean = float(stability_df["nmi"].mean())
    judgement = _stability_judgement(overall_ari_mean, overall_nmi_mean)

    lines = [
        "# Cluster Stability Evaluation",
        "",
        "## Setup",
        "",
        f"- Number of clusters evaluated: `{n_clusters}`",
        f"- K-Means reruns with different seeds: `{len(seeds)}` (baseline seed `{seeds[0]}`)",
        f"- Subsample stability: `{n_subsamples}` runs with `{sample_fraction:.0%}` random samples",
        "- Stability metrics: **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**.",
        "",
        "## Stability Results",
        "",
        "| Check | ARI Mean | ARI Std | NMI Mean | NMI Std | Mean Centroid Shift |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Seed reruns vs baseline | {seed_row['ari_mean']:.4f} | {seed_row['ari_std']:.4f} | "
            f"{seed_row['nmi_mean']:.4f} | {seed_row['nmi_std']:.4f} | {seed_row['centroid_shift_mean']:.4f} |"
        ),
        (
            f"| 80% subsamples vs baseline | {subsample_row['ari_mean']:.4f} | {subsample_row['ari_std']:.4f} | "
            f"{subsample_row['nmi_mean']:.4f} | {subsample_row['nmi_std']:.4f} | {subsample_row['centroid_shift_mean']:.4f} |"
        ),
        "",
        "## Robustness Interpretation",
        "",
        f"- Overall mean ARI: `{overall_ari_mean:.4f}`",
        f"- Overall mean NMI: `{overall_nmi_mean:.4f}`",
        f"- Stability verdict: **{judgement}**",
        "",
        "Higher ARI/NMI values indicate stronger agreement of cluster membership under reruns and perturbations.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "seed_ari_mean": float(seed_row["ari_mean"]),
        "seed_nmi_mean": float(seed_row["nmi_mean"]),
        "subsample_ari_mean": float(subsample_row["ari_mean"]),
        "subsample_nmi_mean": float(subsample_row["nmi_mean"]),
        "overall_ari_mean": overall_ari_mean,
        "overall_nmi_mean": overall_nmi_mean,
        "judgement": judgement,
    }


def evaluate_clustering_model(
    clustered_data_path: str = "data/processed/clustered_restaurants.csv",
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
    metrics_md_path: str = "reports/model_evaluation/clustering_metrics.md",
    k_comparison_csv_path: str = "reports/model_evaluation/kmeans_k_comparison.csv",
    silhouette_plot_path: str = "reports/figures/eval_silhouette_by_k.png",
    davies_bouldin_plot_path: str = "reports/figures/eval_davies_bouldin_by_k.png",
    calinski_plot_path: str = "reports/figures/eval_calinski_harabasz_by_k.png",
    k_min: int = 2,
    k_max: int = 8,
) -> Dict[str, object]:
    _, features, labels, selected_method, n_clusters = _load_features_and_labels(
        clustered_data_path=clustered_data_path,
        feature_matrix_path=feature_matrix_path,
    )

    selected_metrics = _evaluate_labels(features, labels)
    k_scores_df = _evaluate_kmeans_range(features, range(k_min, k_max + 1))

    metrics_md = Path(metrics_md_path)
    k_csv = Path(k_comparison_csv_path)
    silhouette_plot = Path(silhouette_plot_path)
    db_plot = Path(davies_bouldin_plot_path)
    ch_plot = Path(calinski_plot_path)
    _ensure_parent_dirs([metrics_md, k_csv, silhouette_plot, db_plot, ch_plot])

    k_scores_df.to_csv(k_csv, index=False)
    _write_metrics_markdown(selected_metrics, selected_method, n_clusters, metrics_md)

    _plot_metric_by_k(
        scores_df=k_scores_df,
        metric_col="silhouette_score",
        title="Silhouette Score by K (K-Means)",
        y_label="Silhouette Score",
        output_path=silhouette_plot,
        color="#2563eb",
        prefer="max",
    )
    _plot_metric_by_k(
        scores_df=k_scores_df,
        metric_col="davies_bouldin_score",
        title="Davies-Bouldin Score by K (K-Means)",
        y_label="Davies-Bouldin Score",
        output_path=db_plot,
        color="#dc2626",
        prefer="min",
    )
    _plot_metric_by_k(
        scores_df=k_scores_df,
        metric_col="calinski_harabasz_score",
        title="Calinski-Harabasz Score by K (K-Means)",
        y_label="Calinski-Harabasz Score",
        output_path=ch_plot,
        color="#16a34a",
        prefer="max",
    )

    best_silhouette_row = k_scores_df.loc[k_scores_df["silhouette_score"].idxmax()]
    best_db_row = k_scores_df.loc[k_scores_df["davies_bouldin_score"].idxmin()]
    best_ch_row = k_scores_df.loc[k_scores_df["calinski_harabasz_score"].idxmax()]

    return {
        "selected_method": selected_method,
        "selected_num_clusters": n_clusters,
        "selected_metrics": selected_metrics,
        "k_comparison_csv_path": str(k_csv),
        "metrics_md_path": str(metrics_md),
        "best_k_by_silhouette": int(best_silhouette_row["k"]),
        "best_k_by_davies_bouldin": int(best_db_row["k"]),
        "best_k_by_calinski_harabasz": int(best_ch_row["k"]),
        "silhouette_plot_path": str(silhouette_plot),
        "davies_bouldin_plot_path": str(db_plot),
        "calinski_harabasz_plot_path": str(ch_plot),
    }


def evaluate_clustering_stability(
    clustered_data_path: str = "data/processed/clustered_restaurants.csv",
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
    stability_md_path: str = "reports/model_evaluation/cluster_stability.md",
    stability_summary_csv_path: str = "reports/model_evaluation/cluster_stability_summary.csv",
    stability_ari_plot_path: str = "reports/figures/cluster_stability_ari.png",
    stability_nmi_plot_path: str = "reports/figures/cluster_stability_nmi.png",
    seeds: List[int] | None = None,
    sample_fraction: float = 0.8,
    n_subsamples: int = 20,
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    if seeds is None:
        seeds = [42, 7, 11, 19, 23, 29, 37, 53, 71, 89]

    _, features, _, _, n_clusters = _load_features_and_labels(
        clustered_data_path=clustered_data_path,
        feature_matrix_path=feature_matrix_path,
    )

    seed_df, baseline_model, baseline_labels = _run_seed_stability(
        features=features,
        n_clusters=n_clusters,
        seeds=seeds,
    )
    subsample_df = _run_subsample_stability(
        features=features,
        n_clusters=n_clusters,
        baseline_model=baseline_model,
        baseline_labels=baseline_labels,
        sample_fraction=sample_fraction,
        n_iterations=n_subsamples,
        random_state=random_state,
    )

    stability_df = pd.concat([seed_df, subsample_df], ignore_index=True)

    stability_md = Path(stability_md_path)
    stability_csv = Path(stability_summary_csv_path)
    ari_plot = Path(stability_ari_plot_path)
    nmi_plot = Path(stability_nmi_plot_path)
    _ensure_parent_dirs([stability_md, stability_csv, ari_plot, nmi_plot])

    stability_df.to_csv(stability_csv, index=False)
    summary_stats = _write_stability_markdown(
        stability_df=stability_df,
        output_path=stability_md,
        n_clusters=n_clusters,
        seeds=seeds,
        sample_fraction=sample_fraction,
        n_subsamples=n_subsamples,
    )

    _plot_stability_metric(
        stability_df=stability_df,
        metric_col="ari",
        output_path=ari_plot,
        title="Cluster Stability (ARI)",
    )
    _plot_stability_metric(
        stability_df=stability_df,
        metric_col="nmi",
        output_path=nmi_plot,
        title="Cluster Stability (NMI)",
    )

    return {
        "stability_md_path": str(stability_md),
        "stability_summary_csv_path": str(stability_csv),
        "stability_ari_plot_path": str(ari_plot),
        "stability_nmi_plot_path": str(nmi_plot),
        "n_clusters": n_clusters,
        "n_seed_runs": len(seeds),
        "n_subsamples": n_subsamples,
        **summary_stats,
    }


if __name__ == "__main__":
    eval_summary = evaluate_clustering_model()
    stability_summary = evaluate_clustering_stability()

    print("Clustering evaluation summary:")
    for key, value in eval_summary.items():
        print(f"- {key}: {value}")

    print("\nCluster stability summary:")
    for key, value in stability_summary.items():
        print(f"- {key}: {value}")
