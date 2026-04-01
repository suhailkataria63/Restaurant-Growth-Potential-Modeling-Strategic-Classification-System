from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_LABELS = {
    "scale_score": "Scale Score",
    "cost_discipline_score": "Cost Discipline Score",
    "aggregator_dependence": "Aggregator Dependence",
    "expansion_headroom": "Expansion Headroom",
    "revenue_quality_score": "Revenue Quality Score",
    "total_revenue": "Total Revenue",
    "total_net_profit": "Total Net Profit",
    "delivery_revenue_mix": "Delivery Revenue Mix",
    "instore_reliance": "In-Store Reliance",
    "growthfactor": "Growth Factor",
    "monthlyorders": "Monthly Orders",
    "aov": "Average Order Value",
    "cogsrate": "COGS Rate",
    "opexrate": "OPEX Rate",
    "deliveryradiuskm": "Delivery Radius (km)",
    "deliverycostperorder": "Delivery Cost per Order",
    "sd_share": "Self-Delivery Share",
    "ue_share": "UberEats Share",
    "dd_share": "DoorDash Share",
}

PROFILE_METRICS = [
    "scale_score",
    "cost_discipline_score",
    "aggregator_dependence",
    "expansion_headroom",
    "revenue_quality_score",
    "total_revenue",
    "total_net_profit",
    "delivery_revenue_mix",
    "instore_reliance",
    "growthfactor",
    "monthlyorders",
    "aov",
    "cogsrate",
    "opexrate",
    "deliveryradiuskm",
    "deliverycostperorder",
    "sd_share",
    "ue_share",
    "dd_share",
]

MAJOR_KPI_METRICS = [
    "scale_score",
    "cost_discipline_score",
    "aggregator_dependence",
    "expansion_headroom",
    "revenue_quality_score",
    "total_net_profit",
    "delivery_revenue_mix",
    "instore_reliance",
]


def _safe_pct_delta(cluster_mean: float, overall_mean: float) -> float:
    if np.isclose(overall_mean, 0.0, atol=1e-9):
        return np.nan
    return ((cluster_mean - overall_mean) / overall_mean) * 100.0


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _cluster_theme(z_row: pd.Series) -> Tuple[str, str]:
    scale = z_row.get("scale_score", 0.0)
    cost = z_row.get("cost_discipline_score", 0.0)
    agg = z_row.get("aggregator_dependence", 0.0)
    expansion = z_row.get("expansion_headroom", 0.0)
    profit = z_row.get("total_net_profit", 0.0)
    revenue_quality = z_row.get("revenue_quality_score", 0.0)
    delivery_mix = z_row.get("delivery_revenue_mix", 0.0)
    instore = z_row.get("instore_reliance", 0.0)
    sd_share = z_row.get("sd_share", 0.0)

    if scale > 0.4 and (cost < -0.3 or profit < -0.3 or revenue_quality < -0.3):
        return (
            "High-Growth / High-Risk",
            "Strong top-line momentum, but cost and profit pressure suggest execution risk unless margin discipline improves.",
        )
    if expansion > 0.45 and profit < -0.2:
        return (
            "Overextended, Low Return",
            "Operational footprint appears stretched relative to financial return, indicating potential inefficiency in coverage strategy.",
        )
    if agg > 0.25 and delivery_mix > 0.2 and (profit < 0.0 or revenue_quality < 0.0):
        return (
            "Aggregator-Dependent Low Margin",
            "Heavy channel dependence on aggregators is translating into weaker unit economics and constrained margin resilience.",
        )
    if (sd_share > 0.25 or delivery_mix > 0.25) and profit > 0.4 and cost > 0.2 and scale > 0.2:
        return (
            "Scalable Self-Delivery Leaders",
            "This cluster combines healthy economics with delivery-channel scale, signaling strong platform-independent scaling capacity.",
        )
    if instore > 0.4 and agg < -0.2 and profit > -0.1:
        return (
            "Stable Local Performers",
            "More localized channel mix and steadier economics suggest dependable performance with lower platform exposure.",
        )
    if profit > 0.5 and cost > 0.2 and scale > 0.25:
        return (
            "Scalable Profit Leaders",
            "High scale with healthy profitability and cost control indicates strong strategic capacity for expansion.",
        )
    if scale < -0.25 and profit > 0:
        return (
            "Lean Niche Operators",
            "Smaller scale but relatively efficient economics suggest targeted growth opportunities in focused segments.",
        )
    return (
        "Mixed Transition Operators",
        "Signals are mixed across growth, margin, and channel dependence, indicating a transition segment needing tailored strategy.",
    )


def _build_summary_frame(
    df: pd.DataFrame,
    cluster_col: str,
    metrics: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    overall_mean = df[metrics].mean()
    overall_std = df[metrics].std(ddof=0).replace(0, np.nan)
    cluster_means = df.groupby(cluster_col)[metrics].mean().sort_index()

    records = []
    for cluster_id, row in cluster_means.iterrows():
        for metric in metrics:
            cluster_mean = float(row[metric])
            overall_m = float(overall_mean[metric])
            std = overall_std[metric]
            delta_abs = cluster_mean - overall_m
            delta_std = np.nan if pd.isna(std) else delta_abs / float(std)
            records.append(
                {
                    "cluster_id": int(cluster_id),
                    "metric": metric,
                    "metric_label": _metric_label(metric),
                    "cluster_mean": cluster_mean,
                    "overall_mean": overall_m,
                    "delta_absolute": delta_abs,
                    "delta_percent": _safe_pct_delta(cluster_mean, overall_m),
                    "delta_standardized": delta_std,
                }
            )
    summary_df = pd.DataFrame(records)
    return summary_df, cluster_means, overall_mean, overall_std


def _build_cluster_profiles_md(
    summary_df: pd.DataFrame,
    cluster_sizes: pd.Series,
    label_map: Dict[int, Dict[str, str]],
    output_path: Path,
) -> None:
    total = int(cluster_sizes.sum())
    lines: List[str] = []
    lines.append("# Cluster Profiles")
    lines.append("")
    lines.append("Business-level interpretation of clustering results using relative KPI behavior vs overall dataset averages.")
    lines.append("")

    key_metrics = [m for m in MAJOR_KPI_METRICS if m in summary_df["metric"].unique().tolist()]

    for cluster_id in sorted(cluster_sizes.index.tolist()):
        cluster_block = summary_df[summary_df["cluster_id"] == cluster_id].copy()
        cluster_block = cluster_block.sort_values("delta_standardized", ascending=False)
        top_positive = cluster_block.head(3)
        top_negative = cluster_block.tail(3).sort_values("delta_standardized")
        cluster_size = int(cluster_sizes.loc[cluster_id])
        share_pct = (cluster_size / total) * 100.0
        label_name = label_map[cluster_id]["cluster_label_name"]
        description = label_map[cluster_id]["cluster_description"]

        lines.append(f"## Cluster {cluster_id} - {label_name}")
        lines.append("")
        lines.append(f"- Size: {cluster_size} restaurants ({share_pct:.1f}% of dataset)")
        lines.append(f"- Description: {description}")
        lines.append("")
        lines.append("### Distinguishing Characteristics")
        lines.append("")
        lines.append("Above overall:")
        for _, row in top_positive.iterrows():
            lines.append(f"- {_metric_label(row['metric'])}: z-delta {row['delta_standardized']:.2f}")
        lines.append("")
        lines.append("Below overall:")
        for _, row in top_negative.iterrows():
            lines.append(f"- {_metric_label(row['metric'])}: z-delta {row['delta_standardized']:.2f}")
        lines.append("")
        lines.append("### Major KPI Snapshot")
        lines.append("")
        lines.append("| Metric | Cluster Mean | Overall Mean | Delta | Delta % |")
        lines.append("|--------|--------------|--------------|-------|---------|")
        key_block = cluster_block[cluster_block["metric"].isin(key_metrics)].copy()
        key_block = key_block.sort_values("metric")
        for _, row in key_block.iterrows():
            delta_pct = "n/a" if pd.isna(row["delta_percent"]) else f"{row['delta_percent']:.1f}%"
            lines.append(
                f"| {row['metric_label']} | {row['cluster_mean']:.4f} | {row['overall_mean']:.4f} | {row['delta_absolute']:.4f} | {delta_pct} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_cluster_kpi_comparison(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    major_metrics = [m for m in MAJOR_KPI_METRICS if m in summary_df["metric"].unique().tolist()]
    pivot = (
        summary_df[summary_df["metric"].isin(major_metrics)]
        .pivot(index="cluster_id", columns="metric", values="delta_standardized")
        .sort_index()
    )
    pivot = pivot[major_metrics]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-2.5, vmax=2.5)
    ax.set_title("Cluster KPI Comparison (Standardized Delta vs Overall)")
    ax.set_xlabel("KPI")
    ax.set_ylabel("Cluster")
    ax.set_xticks(np.arange(len(major_metrics)))
    ax.set_xticklabels([_metric_label(m) for m in major_metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"Cluster {idx}" for idx in pivot.index])

    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.values[row_idx, col_idx]
            if pd.isna(value):
                text = "n/a"
            else:
                text = f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Standardized Delta (z-score)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def interpret_clusters(
    df: pd.DataFrame,
    cluster_col: str = "selected_cluster",
    cluster_summary_path: str = "reports/cluster_summary.csv",
    cluster_profiles_path: str = "reports/cluster_profiles.md",
    cluster_figure_path: str = "reports/figures/cluster_kpi_comparison.png",
) -> pd.DataFrame:
    """
    Compute cluster interpretation artifacts and return cluster label mapping.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"'{cluster_col}' column is required for cluster interpretation.")

    metrics = [metric for metric in PROFILE_METRICS if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])]
    if not metrics:
        raise ValueError("No numeric profile metrics available for interpretation.")

    summary_df, cluster_means, overall_mean, overall_std = _build_summary_frame(df, cluster_col, metrics)

    summary_path = Path(cluster_summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    z_means = (cluster_means - overall_mean) / overall_std
    cluster_sizes = df[cluster_col].value_counts().sort_index()

    label_records = []
    for cluster_id in sorted(cluster_means.index.tolist()):
        z_row = z_means.loc[cluster_id].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        label_name, description = _cluster_theme(z_row)
        label_records.append(
            {
                "selected_cluster": int(cluster_id),
                "cluster_label_name": label_name,
                "cluster_description": description,
            }
        )

    labels_df = pd.DataFrame(label_records)
    duplicated = labels_df["cluster_label_name"].duplicated(keep=False)
    if duplicated.any():
        labels_df.loc[duplicated, "cluster_label_name"] = labels_df.loc[duplicated].apply(
            lambda row: f"{row['cluster_label_name']} (Cluster {row['selected_cluster']})",
            axis=1,
        )

    label_map = {
        int(row["selected_cluster"]): {
            "cluster_label_name": row["cluster_label_name"],
            "cluster_description": row["cluster_description"],
        }
        for _, row in labels_df.iterrows()
    }

    _build_cluster_profiles_md(
        summary_df=summary_df,
        cluster_sizes=cluster_sizes,
        label_map=label_map,
        output_path=Path(cluster_profiles_path),
    )
    _plot_cluster_kpi_comparison(summary_df=summary_df, output_path=Path(cluster_figure_path))

    return labels_df
