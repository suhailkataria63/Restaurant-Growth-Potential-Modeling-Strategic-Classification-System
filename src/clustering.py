from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except ImportError:
    dendrogram = None
    linkage = None

try:
    from .dimensionality_reduction import perform_dimensionality_reduction
except ImportError:  # pragma: no cover
    from dimensionality_reduction import perform_dimensionality_reduction

try:
    from .cluster_interpretation import interpret_clusters
except ImportError:  # pragma: no cover
    from cluster_interpretation import interpret_clusters

RANDOM_STATE = 42


def run_clustering_preparation(
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
) -> Dict[str, object]:
    """
    Prepare clustering inputs by generating PCA (and optional UMAP) embeddings.
    """
    outputs = perform_dimensionality_reduction(feature_matrix_path=feature_matrix_path)
    print("Clustering preparation completed.")
    return outputs


def _ensure_parent_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _plot_kmeans_diagnostics(scores_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(scores_df["k"], scores_df["inertia"], marker="o", linewidth=2, color="#2563eb")
    ax.set_title("K-Means Elbow Plot")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_elbow_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(scores_df["k"], scores_df["silhouette_score"], marker="o", linewidth=2, color="#16a34a")
    ax.set_title("K-Means Silhouette Score Comparison")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "kmeans_silhouette_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dendrogram(pca_matrix: np.ndarray, figures_dir: Path, sample_size: int = 500) -> str:
    output_path = figures_dir / "hierarchical_dendrogram.png"
    if linkage is None or dendrogram is None:
        return "SciPy not available; dendrogram skipped."

    n_rows = pca_matrix.shape[0]
    if n_rows > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(n_rows, size=sample_size, replace=False)
        matrix_for_linkage = pca_matrix[sample_idx]
        note = f"Dendrogram generated on random sample of {sample_size}/{n_rows} rows."
    else:
        matrix_for_linkage = pca_matrix
        note = f"Dendrogram generated on full dataset ({n_rows} rows)."

    linkage_matrix = linkage(matrix_for_linkage, method="ward")
    fig, ax = plt.subplots(figsize=(12, 7))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Clustered Leaves")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return note


def _evaluate_kmeans(pca_matrix: np.ndarray, k_range: Iterable[int]) -> Tuple[pd.DataFrame, np.ndarray, int, float]:
    rows = []
    best_k = None
    best_silhouette = -np.inf
    best_labels = None

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(pca_matrix)
        inertia = model.inertia_
        silhouette = silhouette_score(pca_matrix, labels)
        rows.append({"k": k, "inertia": inertia, "silhouette_score": silhouette})

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_labels = labels

    scores_df = pd.DataFrame(rows)
    return scores_df, best_labels, best_k, best_silhouette


def _run_hierarchical(pca_matrix: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(pca_matrix)
    silhouette = silhouette_score(pca_matrix, labels)
    return labels, silhouette


def _run_dbscan(feature_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    # Parameters selected to avoid trivial all-noise clustering on standardized high-dimensional features.
    model = DBSCAN(eps=3.2, min_samples=10)
    labels = model.fit_predict(feature_matrix)

    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]
    unique_clusters = sorted(set(non_noise_labels.tolist()))

    silhouette = None
    if len(unique_clusters) > 1 and non_noise_mask.sum() > len(unique_clusters):
        silhouette = float(silhouette_score(feature_matrix[non_noise_mask], non_noise_labels))

    summary = {
        "dbscan_clusters_excluding_noise": len(unique_clusters),
        "dbscan_noise_points": int((labels == -1).sum()),
        "dbscan_silhouette_non_noise": silhouette,
    }
    return labels, summary


def run_clustering_analysis(
    featured_data_path: str = "data/processed/restaurants_featured.csv",
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
    pca_features_path: str = "data/processed/pca_features.csv",
    clustered_output_path: str = "data/processed/clustered_restaurants.csv",
    cluster_summary_path: str = "reports/cluster_summary.csv",
    cluster_profiles_path: str = "reports/cluster_profiles.md",
    cluster_figure_path: str = "reports/figures/cluster_kpi_comparison.png",
    figures_dir: str = "reports/figures",
    run_dbscan: bool = True,
) -> Dict[str, object]:
    """
    Run multi-method clustering and export final per-restaurant cluster assignments.
    """
    # Ensure PCA is fresh and available for downstream clustering.
    perform_dimensionality_reduction(
        feature_matrix_path=feature_matrix_path,
        pca_output_path=pca_features_path,
        figures_dir=figures_dir,
    )

    featured_df = pd.read_csv(featured_data_path)
    feature_matrix_df = pd.read_csv(feature_matrix_path)
    pca_df = pd.read_csv(pca_features_path)

    if not (len(featured_df) == len(feature_matrix_df) == len(pca_df)):
        raise ValueError("Input row counts do not match between featured, feature matrix, and PCA data.")

    pca_matrix = pca_df.to_numpy()
    full_matrix = feature_matrix_df.to_numpy()
    figures_path = Path(figures_dir)

    # K-Means model sweep (k=2..8) on PCA space.
    k_values = range(2, 9)
    kmeans_scores, kmeans_labels, best_k, kmeans_best_silhouette = _evaluate_kmeans(pca_matrix, k_values)
    _plot_kmeans_diagnostics(kmeans_scores, figures_path)

    # Hierarchical clustering using selected K from K-Means for direct comparison.
    hierarchical_labels, hierarchical_silhouette = _run_hierarchical(pca_matrix, n_clusters=best_k)
    dendrogram_note = _plot_dendrogram(pca_matrix, figures_path)

    # Optional DBSCAN robustness check on full standardized feature matrix.
    dbscan_labels = None
    dbscan_summary = None
    if run_dbscan:
        dbscan_labels, dbscan_summary = _run_dbscan(full_matrix)

    # Select method by silhouette on PCA space between K-Means and hierarchical.
    if hierarchical_silhouette > kmeans_best_silhouette:
        selected_method = "agglomerative"
        selected_labels = hierarchical_labels
        selected_silhouette = hierarchical_silhouette
    else:
        selected_method = "kmeans"
        selected_labels = kmeans_labels
        selected_silhouette = kmeans_best_silhouette

    profile_df = featured_df.copy()
    profile_df["selected_cluster"] = selected_labels.astype(int)
    profile_df["selected_method"] = selected_method
    label_df = interpret_clusters(
        df=profile_df,
        cluster_col="selected_cluster",
        cluster_summary_path=cluster_summary_path,
        cluster_profiles_path=cluster_profiles_path,
        cluster_figure_path=cluster_figure_path,
    )

    identifier_cols = ["restaurantid", "restaurantname", "cuisinetype", "segment", "subregion"]
    strategic_kpis = [
        "scale_score",
        "cost_discipline_score",
        "aggregator_dependence",
        "expansion_headroom",
        "revenue_quality_score",
        "total_revenue",
        "total_net_profit",
        "delivery_revenue_mix",
        "instore_reliance",
    ]

    base_cols = [col for col in identifier_cols + strategic_kpis if col in featured_df.columns]
    clustered_df = featured_df[base_cols].copy()
    clustered_df = pd.concat([clustered_df, pca_df], axis=1)
    clustered_df["kmeans_cluster"] = kmeans_labels.astype(int)
    clustered_df["hierarchical_cluster"] = hierarchical_labels.astype(int)
    if dbscan_labels is not None:
        clustered_df["dbscan_cluster"] = dbscan_labels.astype(int)
    clustered_df["selected_method"] = selected_method
    clustered_df["selected_cluster"] = selected_labels.astype(int)
    clustered_df = clustered_df.merge(label_df, on="selected_cluster", how="left")

    output_path = Path(clustered_output_path)
    _ensure_parent_dirs([output_path])
    clustered_df.to_csv(output_path, index=False)

    return {
        "selected_method": selected_method,
        "selected_num_clusters": int(len(np.unique(selected_labels))),
        "selected_silhouette": float(selected_silhouette),
        "kmeans_best_k": int(best_k),
        "kmeans_best_silhouette": float(kmeans_best_silhouette),
        "hierarchical_silhouette": float(hierarchical_silhouette),
        "dendrogram_note": dendrogram_note,
        "dbscan_summary": dbscan_summary,
        "cluster_summary_path": cluster_summary_path,
        "cluster_profiles_path": cluster_profiles_path,
        "cluster_kpi_comparison_figure": cluster_figure_path,
        "clustered_output_path": str(output_path),
    }


if __name__ == "__main__":
    summary = run_clustering_analysis()
    print("Clustering analysis summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
