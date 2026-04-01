from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

RECOMMENDATION_ORDER = [
    "Scale Aggressively",
    "Expand Carefully",
    "Rebalance Channels",
    "Optimize",
    "Stabilize Operations",
]

PLAYBOOK = {
    "Scale Aggressively": {
        "meaning": "Push expansion through new zones, capacity upgrades, and growth investments while preserving current unit economics.",
        "when": "Use when GPI is high and both cost discipline and revenue quality are strong with manageable channel risk.",
    },
    "Expand Carefully": {
        "meaning": "Pursue growth in staged pilots with strict guardrails on costs, conversion, and payback windows.",
        "when": "Use when growth potential is high or improving but risk indicators (cost pressure, channel dependence, or execution strain) are present.",
    },
    "Rebalance Channels": {
        "meaning": "Shift mix toward healthier channels by reducing aggregator over-dependence and strengthening direct/self-delivery economics.",
        "when": "Use when aggregator dependence and delivery mix are high relative to profitability quality or in-store resilience.",
    },
    "Optimize": {
        "meaning": "Improve efficiency and conversion in the current footprint before major expansion moves.",
        "when": "Use when performance is moderate and the business needs KPI tuning rather than structural turnaround.",
    },
    "Stabilize Operations": {
        "meaning": "Prioritize operational reset: margin recovery, cost control, and service consistency before pursuing growth.",
        "when": "Use when GPI is low and core economics show clear stress.",
    },
}


def _build_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "cost_low": float(df["cost_discipline_score"].quantile(0.35)),
        "cost_high": float(df["cost_discipline_score"].quantile(0.65)),
        "agg_high": float(df["aggregator_dependence"].quantile(0.70)),
        "agg_very_high": float(df["aggregator_dependence"].quantile(0.85)),
        "expansion_high": float(df["expansion_headroom"].quantile(0.70)),
        "revenue_low": float(df["revenue_quality_score"].quantile(0.35)),
        "revenue_high": float(df["revenue_quality_score"].quantile(0.65)),
        "delivery_high": float(df["delivery_revenue_mix"].quantile(0.70)),
        "instore_low": float(df["instore_reliance"].quantile(0.30)),
    }


def _assign_recommendation(row: pd.Series, t: Dict[str, float]) -> Tuple[str, str]:
    cluster = str(row.get("cluster_label_name", ""))
    gpi_band = str(row.get("gpi_band", ""))
    gpi_score = float(row.get("gpi_score", 0.0))
    agg = float(row.get("aggregator_dependence", 0.0))
    cost = float(row.get("cost_discipline_score", 0.0))
    expansion = float(row.get("expansion_headroom", 0.0))
    revenue_quality = float(row.get("revenue_quality_score", 0.0))
    delivery_mix = float(row.get("delivery_revenue_mix", 0.0))
    instore = float(row.get("instore_reliance", 0.0))

    if gpi_band == "High Potential":
        if (
            cluster == "Scalable Profit Leaders"
            and cost >= t["cost_high"]
            and revenue_quality >= t["revenue_high"]
            and agg <= t["agg_high"]
        ):
            return (
                "Scale Aggressively",
                f"High Potential (GPI {gpi_score:.1f}) with strong cost discipline and revenue quality supports accelerated expansion.",
            )
        if cluster == "High-Growth / High-Risk" or cost < t["cost_low"] or agg > t["agg_very_high"]:
            return (
                "Expand Carefully",
                f"High Potential (GPI {gpi_score:.1f}) but risk signals from cost/channel profile indicate phased expansion is safer.",
            )
        if agg > t["agg_high"] and delivery_mix > t["delivery_high"] and instore < t["instore_low"]:
            return (
                "Rebalance Channels",
                f"High Potential (GPI {gpi_score:.1f}) with heavy aggregator-led mix suggests rebalancing channels before full-scale growth.",
            )
        return (
            "Scale Aggressively",
            f"High Potential (GPI {gpi_score:.1f}) with no major risk flags supports growth acceleration.",
        )

    if gpi_band == "Moderate Potential":
        if cluster == "Aggregator-Dependent Low Margin" or (agg > t["agg_high"] and delivery_mix > t["delivery_high"]):
            return (
                "Rebalance Channels",
                f"Moderate Potential (GPI {gpi_score:.1f}) and elevated aggregator dependence indicate channel rebalancing is the highest-leverage action.",
            )
        if cost < t["cost_low"] or revenue_quality < t["revenue_low"]:
            return (
                "Optimize",
                f"Moderate Potential (GPI {gpi_score:.1f}) with margin-quality pressure points to optimization before expansion.",
            )
        if expansion > t["expansion_high"]:
            return (
                "Expand Carefully",
                f"Moderate Potential (GPI {gpi_score:.1f}) with meaningful headroom suggests cautious, milestone-based expansion.",
            )
        return (
            "Optimize",
            f"Moderate Potential (GPI {gpi_score:.1f}) is best improved through operational and commercial optimization.",
        )

    # Caution Zone
    if cost < t["cost_low"] or revenue_quality < t["revenue_low"] or cluster == "High-Growth / High-Risk":
        return (
            "Stabilize Operations",
            f"Caution Zone (GPI {gpi_score:.1f}) with weak economic quality indicates stabilization should precede growth actions.",
        )
    if agg > t["agg_high"]:
        return (
            "Rebalance Channels",
            f"Caution Zone (GPI {gpi_score:.1f}) and high aggregator dependence suggest urgent channel-risk rebalancing.",
        )
    return (
        "Optimize",
        f"Caution Zone (GPI {gpi_score:.1f}) with mixed signals suggests targeted optimization as the immediate priority.",
    )


def _build_recommendation_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    summary = (
        df.groupby("strategy_recommendation", as_index=False)
        .agg(
            restaurant_count=("restaurantid", "count"),
            avg_gpi_score=("gpi_score", "mean"),
            avg_cost_discipline_score=("cost_discipline_score", "mean"),
            avg_aggregator_dependence=("aggregator_dependence", "mean"),
            avg_expansion_headroom=("expansion_headroom", "mean"),
            avg_revenue_quality_score=("revenue_quality_score", "mean"),
        )
    )
    summary["share_pct"] = (summary["restaurant_count"] / len(df) * 100.0).round(2)

    band_ct = pd.crosstab(df["strategy_recommendation"], df["gpi_band"])
    for band in ["High Potential", "Moderate Potential", "Caution Zone"]:
        summary[f"{band.lower().replace(' ', '_')}_count"] = (
            summary["strategy_recommendation"].map(band_ct.get(band, pd.Series(dtype=int))).fillna(0).astype(int)
        )

    cluster_mode = (
        df.groupby("strategy_recommendation")["cluster_label_name"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A")
        .rename("dominant_cluster_archetype")
    )
    summary = summary.merge(cluster_mode, on="strategy_recommendation", how="left")

    summary["strategy_recommendation"] = pd.Categorical(
        summary["strategy_recommendation"], categories=RECOMMENDATION_ORDER, ordered=True
    )
    summary = summary.sort_values("strategy_recommendation").reset_index(drop=True)

    round_cols = [
        "avg_gpi_score",
        "avg_cost_discipline_score",
        "avg_aggregator_dependence",
        "avg_expansion_headroom",
        "avg_revenue_quality_score",
        "share_pct",
    ]
    summary[round_cols] = summary[round_cols].round(3)

    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def _build_strategy_playbook(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
) -> None:
    lines = []
    lines.append("# Strategy Playbook")
    lines.append("")
    lines.append("Recommendation definitions for restaurant-level strategic actioning.")
    lines.append("")

    count_map = df["strategy_recommendation"].value_counts().to_dict()
    share_map = (df["strategy_recommendation"].value_counts(normalize=True) * 100.0).round(1).to_dict()

    for rec in RECOMMENDATION_ORDER:
        if rec not in count_map:
            continue
        meaning = PLAYBOOK[rec]["meaning"]
        when_apply = PLAYBOOK[rec]["when"]
        examples = (
            df.loc[df["strategy_recommendation"] == rec, "recommendation_reason"]
            .dropna()
            .head(2)
            .tolist()
        )
        lines.append(f"## {rec}")
        lines.append("")
        lines.append(f"- Restaurants: {count_map[rec]} ({share_map[rec]}%)")
        lines.append(f"- Business meaning: {meaning}")
        lines.append(f"- Apply when: {when_apply}")
        lines.append("")
        lines.append("Example assignment reasons:")
        for item in examples:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    columns = summary_df.columns.tolist()
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in summary_df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_strategy_recommendations(
    clustered_input_path: str = "data/processed/clustered_restaurants.csv",
    clustered_output_path: str = "data/processed/clustered_restaurants.csv",
    summary_output_path: str = "reports/recommendation_summary.csv",
    playbook_output_path: str = "reports/strategy_playbook.md",
) -> Dict[str, object]:
    """
    Assign primary strategy recommendation per restaurant and generate summary artifacts.
    """
    df = pd.read_csv(clustered_input_path)

    required_cols = [
        "cluster_label_name",
        "gpi_score",
        "gpi_band",
        "cost_discipline_score",
        "aggregator_dependence",
        "expansion_headroom",
        "revenue_quality_score",
        "delivery_revenue_mix",
        "instore_reliance",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for recommendation engine: {missing}")

    thresholds = _build_thresholds(df)
    assigned = df.apply(lambda row: _assign_recommendation(row, thresholds), axis=1, result_type="expand")
    assigned.columns = ["strategy_recommendation", "recommendation_reason"]
    df["strategy_recommendation"] = assigned["strategy_recommendation"]
    df["recommendation_reason"] = assigned["recommendation_reason"]

    output_path = Path(clustered_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_df = _build_recommendation_summary(df, Path(summary_output_path))
    _build_strategy_playbook(df, summary_df, Path(playbook_output_path))

    return {
        "clustered_output_path": clustered_output_path,
        "summary_output_path": summary_output_path,
        "playbook_output_path": playbook_output_path,
        "recommendation_counts": df["strategy_recommendation"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    result = generate_strategy_recommendations()
    print("Recommendation generation complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")
