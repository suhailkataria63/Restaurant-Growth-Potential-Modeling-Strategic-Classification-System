from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Weights sum to 1.00 and balance growth, economics, risk, and channel resilience.
GPI_WEIGHTS: Dict[str, float] = {
    "gpi_scale_norm": 0.25,
    "gpi_cost_discipline_norm": 0.20,
    "gpi_aggregator_penalty_norm": 0.15,
    "gpi_expansion_headroom_norm": 0.10,
    "gpi_revenue_quality_norm": 0.20,
    "gpi_delivery_revenue_mix_norm": 0.05,
    "gpi_instore_reliance_norm": 0.05,
}


def _robust_minmax(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95, invert: bool = False) -> pd.Series:
    """
    Robust min-max normalization with percentile clipping to reduce outlier dominance.
    Returns values on [0, 1].
    """
    values = pd.to_numeric(series, errors="coerce")
    lower = values.quantile(lower_q)
    upper = values.quantile(upper_q)

    if pd.isna(lower) or pd.isna(upper) or np.isclose(upper, lower):
        scaled = pd.Series(0.5, index=values.index, dtype=float)
    else:
        clipped = values.clip(lower=lower, upper=upper)
        scaled = (clipped - lower) / (upper - lower)
        scaled = scaled.clip(0.0, 1.0)

    if invert:
        scaled = 1.0 - scaled
    return scaled.fillna(0.5)


def _balanced_expansion_score(series: pd.Series) -> pd.Series:
    """
    Expansion headroom scoring that rewards moderate-to-high headroom but penalizes extremes.
    Score is 1 at the target quantile and approaches 0 toward distribution extremes.
    """
    values = pd.to_numeric(series, errors="coerce")
    lower = values.quantile(0.05)
    upper = values.quantile(0.95)
    target = values.quantile(0.60)

    if pd.isna(lower) or pd.isna(upper) or np.isclose(upper, lower):
        return pd.Series(0.5, index=values.index, dtype=float)

    clipped = values.clip(lower=lower, upper=upper)
    max_distance = max(abs(target - lower), abs(upper - target))
    if np.isclose(max_distance, 0.0):
        return pd.Series(0.5, index=values.index, dtype=float)

    score = 1.0 - (clipped - target).abs() / max_distance
    return score.clip(0.0, 1.0).fillna(0.5)


def _assign_gpi_band(score: float) -> str:
    if score >= 70.0:
        return "High Potential"
    if score >= 45.0:
        return "Moderate Potential"
    return "Caution Zone"


def _build_gpi_methodology_markdown(
    df: pd.DataFrame,
    methodology_path: Path,
) -> None:
    band_order = ["High Potential", "Moderate Potential", "Caution Zone"]
    band_summary = (
        df.groupby("gpi_band")["gpi_score"]
        .agg(["count", "mean", "min", "max"])
        .rename(columns={"count": "restaurants", "mean": "avg_gpi", "min": "min_gpi", "max": "max_gpi"})
        .reindex(band_order)
        .fillna(0)
    )
    band_summary["share_pct"] = (band_summary["restaurants"] / len(df) * 100.0).round(2)

    lines: List[str] = []
    lines.append("# Growth Potential Index (GPI) Methodology")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Create a single composite index (0-100) that ranks restaurants by structural growth potential while accounting for scale, profitability quality, and channel risk.")
    lines.append("")
    lines.append("## Normalization")
    lines.append("")
    lines.append("- All inputs are normalized to 0-1 before weighting.")
    lines.append("- Most features use robust min-max scaling (5th to 95th percentile clipping).")
    lines.append("- `aggregator_dependence` is inverted into a penalty score so high dependence lowers GPI.")
    lines.append("- `expansion_headroom` uses a balanced scoring function (peaks at moderate-high headroom and penalizes extremes).")
    lines.append("")
    lines.append("## Weighted Formula")
    lines.append("")
    lines.append("`GPI_raw = Σ(weight_i × normalized_component_i)`")
    lines.append("")
    lines.append("`GPI_score = GPI_raw × 100`")
    lines.append("")
    lines.append("| Component | Weight | Direction |")
    lines.append("|-----------|--------|-----------|")
    lines.append("| scale_score | 0.25 | Higher is better |")
    lines.append("| cost_discipline_score | 0.20 | Higher is better |")
    lines.append("| aggregator_dependence (penalty) | 0.15 | Lower dependence is better |")
    lines.append("| expansion_headroom (balanced) | 0.10 | Moderate-high preferred |")
    lines.append("| revenue_quality_score | 0.20 | Higher is better |")
    lines.append("| delivery_revenue_mix | 0.05 | Higher contribution to omnichannel growth |")
    lines.append("| instore_reliance | 0.05 | Higher direct-channel resilience |")
    lines.append("")
    lines.append("## Score Bands")
    lines.append("")
    lines.append("- `High Potential`: GPI >= 70")
    lines.append("- `Moderate Potential`: 45 <= GPI < 70")
    lines.append("- `Caution Zone`: GPI < 45")
    lines.append("")
    lines.append("## Current Distribution")
    lines.append("")
    lines.append("| Band | Restaurants | Share % | Avg GPI | Min GPI | Max GPI |")
    lines.append("|------|-------------|---------|---------|---------|---------|")
    for band, row in band_summary.iterrows():
        lines.append(
            f"| {band} | {int(row['restaurants'])} | {row['share_pct']:.2f}% | {row['avg_gpi']:.2f} | {row['min_gpi']:.2f} | {row['max_gpi']:.2f} |"
        )
    lines.append("")
    lines.append("## Interpretation Guidance")
    lines.append("")
    lines.append("- Use GPI with cluster archetype labels for strategy prioritization, not as a standalone decision rule.")
    lines.append("- Recalibrate weights if business strategy changes (for example, margin-first vs growth-first periods).")

    methodology_path.parent.mkdir(parents=True, exist_ok=True)
    methodology_path.write_text("\n".join(lines), encoding="utf-8")


def _build_gpi_summary(df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    band_order = ["High Potential", "Moderate Potential", "Caution Zone"]
    summary = (
        df.groupby("gpi_band", as_index=False)
        .agg(
            restaurant_count=("restaurantid", "count"),
            avg_gpi_score=("gpi_score", "mean"),
            min_gpi_score=("gpi_score", "min"),
            max_gpi_score=("gpi_score", "max"),
            avg_scale_score=("scale_score", "mean"),
            avg_cost_discipline_score=("cost_discipline_score", "mean"),
            avg_aggregator_dependence=("aggregator_dependence", "mean"),
            avg_revenue_quality_score=("revenue_quality_score", "mean"),
            avg_total_net_profit=("total_net_profit", "mean"),
        )
    )
    summary["share_pct"] = (summary["restaurant_count"] / len(df) * 100.0).round(2)
    summary["gpi_band"] = pd.Categorical(summary["gpi_band"], categories=band_order, ordered=True)
    summary = summary.sort_values("gpi_band").reset_index(drop=True)

    # Keep human-friendly rounding for report output.
    round_cols = [
        "avg_gpi_score",
        "min_gpi_score",
        "max_gpi_score",
        "avg_scale_score",
        "avg_cost_discipline_score",
        "avg_aggregator_dependence",
        "avg_revenue_quality_score",
        "avg_total_net_profit",
        "share_pct",
    ]
    summary[round_cols] = summary[round_cols].round(3)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def compute_growth_potential_index(
    clustered_input_path: str = "data/processed/clustered_restaurants.csv",
    clustered_output_path: str = "data/processed/clustered_restaurants.csv",
    methodology_path: str = "reports/gpi_methodology.md",
    summary_path: str = "reports/gpi_summary.csv",
) -> Dict[str, object]:
    """
    Compute weighted Growth Potential Index (GPI), save outputs, and return run metadata.
    """
    df = pd.read_csv(clustered_input_path)

    required_cols = [
        "scale_score",
        "cost_discipline_score",
        "aggregator_dependence",
        "expansion_headroom",
        "revenue_quality_score",
        "delivery_revenue_mix",
        "instore_reliance",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for GPI: {missing}")

    # Component normalization
    df["gpi_scale_norm"] = _robust_minmax(df["scale_score"])
    df["gpi_cost_discipline_norm"] = _robust_minmax(df["cost_discipline_score"])
    df["gpi_aggregator_penalty_norm"] = _robust_minmax(df["aggregator_dependence"], invert=True)
    df["gpi_expansion_headroom_norm"] = _balanced_expansion_score(df["expansion_headroom"])
    df["gpi_revenue_quality_norm"] = _robust_minmax(df["revenue_quality_score"])
    df["gpi_delivery_revenue_mix_norm"] = _robust_minmax(df["delivery_revenue_mix"])
    df["gpi_instore_reliance_norm"] = _robust_minmax(df["instore_reliance"])

    # Weighted composite on 0-100 scale
    weighted_sum = pd.Series(0.0, index=df.index)
    for component, weight in GPI_WEIGHTS.items():
        weighted_sum = weighted_sum + (df[component] * weight)

    df["gpi_score"] = (weighted_sum * 100.0).clip(0.0, 100.0).round(2)
    df["gpi_band"] = df["gpi_score"].apply(_assign_gpi_band)

    # Ranking helper for prioritization
    df["gpi_rank"] = df["gpi_score"].rank(method="dense", ascending=False).astype(int)

    output_path = Path(clustered_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_df = _build_gpi_summary(df, Path(summary_path))
    _build_gpi_methodology_markdown(df, Path(methodology_path))

    return {
        "clustered_output_path": str(output_path),
        "methodology_path": methodology_path,
        "summary_path": summary_path,
        "weights": GPI_WEIGHTS,
        "band_counts": df["gpi_band"].value_counts().to_dict(),
        "gpi_min": float(df["gpi_score"].min()),
        "gpi_max": float(df["gpi_score"].max()),
        "gpi_mean": float(df["gpi_score"].mean()),
        "summary_rows": int(len(summary_df)),
    }


if __name__ == "__main__":
    result = compute_growth_potential_index()
    print("GPI scoring complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")
