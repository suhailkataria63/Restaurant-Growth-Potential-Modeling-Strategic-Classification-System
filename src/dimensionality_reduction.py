from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def _ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _infer_component_theme(component_loadings: pd.Series) -> str:
    keyword_groups = {
        "Cost pressure": ["cogsrate", "opexrate", "commissionrate", "deliverycost", "sd_deliverytotalcost"],
        "Channel leverage": ["instoreshare", "ue_share", "dd_share", "sd_share", "ubereats", "doordash", "selfdelivery"],
        "Growth momentum": ["growthfactor", "monthlyorders", "orders", "revenue", "netprofit", "aov"],
        "Scalability": ["deliveryradiuskm", "selfdelivery", "instore", "monthlyorders"],
    }

    scores: Dict[str, float] = {}
    for group, keywords in keyword_groups.items():
        group_score = 0.0
        for feature, value in component_loadings.items():
            if any(keyword in feature for keyword in keywords):
                group_score += abs(value)
        scores[group] = group_score

    best_group = max(scores, key=scores.get)
    if scores[best_group] <= 0:
        return "Mixed latent signal"
    return best_group


def _top_loadings(component_loadings: pd.Series, n: int = 5) -> Tuple[pd.Series, pd.Series]:
    positive = component_loadings.sort_values(ascending=False).head(n)
    negative = component_loadings.sort_values().head(n)
    return positive, negative


def _write_pca_summary(
    pca: PCA,
    feature_names: List[str],
    pca_summary_path: Path,
    original_shape: Tuple[int, int],
) -> None:
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )

    lines: List[str] = []
    lines.append("# PCA Explained Variance Summary")
    lines.append("")
    lines.append(f"- Observations: {original_shape[0]}")
    lines.append(f"- Original features: {original_shape[1]}")
    lines.append(f"- PCA components retained: {pca.n_components_}")
    lines.append("")
    lines.append("## Explained Variance by Component")
    lines.append("")
    lines.append("| Component | Explained Variance | Cumulative Variance |")
    lines.append("|-----------|-------------------|---------------------|")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance), start=1):
        lines.append(f"| PC{i} | {var:.4f} | {cum_var:.4f} |")

    lines.append("")
    lines.append("## Candidate Latent Factors (Top Loadings)")
    lines.append("")

    n_interpret = min(4, pca.n_components_)
    for i in range(1, n_interpret + 1):
        component_name = f"PC{i}"
        series = loadings[component_name]
        factor_theme = _infer_component_theme(series)
        positive, negative = _top_loadings(series, n=5)
        lines.append(f"### {component_name} - Suggested theme: {factor_theme}")
        lines.append("")
        lines.append("Positive direction (higher component score):")
        for feature, value in positive.items():
            lines.append(f"- `{feature}` ({value:.3f})")
        lines.append("")
        lines.append("Negative direction (lower component score):")
        for feature, value in negative.items():
            lines.append(f"- `{feature}` ({value:.3f})")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- These themes are hypothesis labels to support cluster storytelling, not fixed causal truth.")
    lines.append("- Use the scree plot elbow and cumulative variance to choose the PCA dimensionality for clustering models.")
    lines.append("- Re-run this step whenever preprocessing features change so factor definitions remain stable.")

    pca_summary_path.write_text("\n".join(lines), encoding="utf-8")


def _save_pca_plots(
    pca_scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(1, len(explained_variance_ratio) + 1)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    ax1.plot(x_axis, explained_variance_ratio, marker="o", linewidth=2, color="#2563eb")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Scree Plot")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_axis, cumulative_variance, marker="s", linestyle="--", color="#16a34a")
    ax2.set_ylabel("Cumulative Variance")

    fig.tight_layout()
    fig.savefig(figures_dir / "pca_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if pca_scores.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            pca_scores[:, 0],
            pca_scores[:, 1],
            alpha=0.6,
            s=30,
            c=pca_scores[:, 0],
            cmap="viridis",
            edgecolors="none",
        )
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
        ax.set_title("2D PCA Scatter Plot")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "pca_2d_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _run_umap(features: np.ndarray, umap_output_path: Path, figures_dir: Path) -> bool:
    if not UMAP_AVAILABLE:
        print("UMAP library not found; skipping UMAP embedding.")
        return False

    try:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
        umap_df.to_csv(umap_output_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            alpha=0.6,
            s=30,
            c=embedding[:, 0],
            cmap="plasma",
            edgecolors="none",
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title("UMAP 2D Embedding")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "umap_2d_embedding.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"UMAP execution failed; skipping UMAP outputs. Error: {exc}")
        return False


def perform_dimensionality_reduction(
    feature_matrix_path: str = "data/processed/feature_matrix.csv",
    pca_output_path: str = "data/processed/pca_features.csv",
    umap_output_path: str = "data/processed/umap_features.csv",
    pca_summary_path: str = "reports/pca_summary.md",
    figures_dir: str = "reports/figures",
    n_components: int = 10,
) -> Dict[str, object]:
    """
    Run PCA (always) and UMAP (optional) on clustering-ready features.
    """
    feature_matrix = pd.read_csv(feature_matrix_path)
    numeric_matrix = feature_matrix.select_dtypes(include=[np.number])
    if numeric_matrix.empty:
        raise ValueError("No numeric columns found in feature matrix for dimensionality reduction.")

    features = numeric_matrix.to_numpy()
    max_components = min(n_components, features.shape[0], features.shape[1])
    if max_components < 2:
        raise ValueError("PCA requires at least 2 components to generate requested plots.")

    pca_output = Path(pca_output_path)
    umap_output = Path(umap_output_path)
    summary_output = Path(pca_summary_path)
    figures_output = Path(figures_dir)
    _ensure_dirs([pca_output, umap_output, summary_output])
    figures_output.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=max_components)
    pca_scores = pca.fit_transform(features)
    pca_df = pd.DataFrame(pca_scores, columns=[f"PC{i + 1}" for i in range(max_components)])
    pca_df.to_csv(pca_output, index=False)

    _write_pca_summary(
        pca=pca,
        feature_names=numeric_matrix.columns.tolist(),
        pca_summary_path=summary_output,
        original_shape=features.shape,
    )
    _save_pca_plots(
        pca_scores=pca_scores,
        explained_variance_ratio=pca.explained_variance_ratio_,
        figures_dir=figures_output,
    )
    umap_created = _run_umap(features, umap_output, figures_output)

    return {
        "pca_features_path": str(pca_output),
        "umap_features_path": str(umap_output) if umap_created else None,
        "pca_summary_path": str(summary_output),
        "figures_dir": str(figures_output),
        "n_components": max_components,
        "umap_available": umap_created,
    }


if __name__ == "__main__":
    outputs = perform_dimensionality_reduction()
    print("Dimensionality reduction outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
