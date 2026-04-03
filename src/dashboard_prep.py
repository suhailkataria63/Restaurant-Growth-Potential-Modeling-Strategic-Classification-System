from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd


def _safe_mode(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    if mode.empty:
        return "N/A"
    return str(mode.iloc[0])


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, as_index=False)
        .agg(
            restaurant_count=("restaurantid", "count"),
            avg_gpi_score=("gpi_score", "mean"),
            avg_total_revenue=("total_revenue", "mean"),
            avg_total_net_profit=("total_net_profit", "mean"),
            avg_cost_discipline_score=("cost_discipline_score", "mean"),
            avg_aggregator_dependence=("aggregator_dependence", "mean"),
            dominant_recommendation=("strategy_recommendation", _safe_mode),
            dominant_cluster_archetype=("cluster_label_name", _safe_mode),
        )
    )
    grouped["share_pct"] = (grouped["restaurant_count"] / len(df) * 100.0).round(2)
    round_cols = [
        "avg_gpi_score",
        "avg_total_revenue",
        "avg_total_net_profit",
        "avg_cost_discipline_score",
        "avg_aggregator_dependence",
        "share_pct",
    ]
    grouped[round_cols] = grouped[round_cols].round(3)
    grouped = grouped.sort_values("restaurant_count", ascending=False).reset_index(drop=True)
    return grouped


def prepare_dashboard_datasets(
    clustered_input_path: str = "data/processed/clustered_restaurants.csv",
    dashboard_summary_json_path: str = "data/processed/dashboard_summary.json",
    top_restaurants_path: str = "data/processed/top_restaurants.csv",
    cluster_dashboard_summary_path: str = "data/processed/cluster_dashboard_summary.csv",
    filter_summary_dir: str = "data/processed/filter_summary_tables",
    top_n: int = 25,
) -> Dict[str, object]:
    """
    Build dashboard-ready summary datasets from clustered restaurant outputs.
    """
    df = pd.read_csv(clustered_input_path)

    required_cols = [
        "restaurantid",
        "restaurantname",
        "selected_cluster",
        "cluster_label_name",
        "gpi_score",
        "gpi_band",
        "strategy_recommendation",
        "cost_discipline_score",
        "aggregator_dependence",
        "expansion_headroom",
        "revenue_quality_score",
        "delivery_revenue_mix",
        "instore_reliance",
        "total_revenue",
        "total_net_profit",
        "subregion",
        "cuisinetype",
        "segment",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for dashboard prep: {missing}")

    # 1) Overall KPI summary cards
    overall_cards = {
        "total_restaurants": int(len(df)),
        "avg_gpi_score": round(float(df["gpi_score"].mean()), 3),
        "median_gpi_score": round(float(df["gpi_score"].median()), 3),
        "high_potential_count": int((df["gpi_band"] == "High Potential").sum()),
        "high_potential_share_pct": round(float((df["gpi_band"] == "High Potential").mean() * 100.0), 2),
        "avg_total_revenue": round(float(df["total_revenue"].mean()), 3),
        "avg_total_net_profit": round(float(df["total_net_profit"].mean()), 3),
        "avg_cost_discipline_score": round(float(df["cost_discipline_score"].mean()), 3),
        "avg_aggregator_dependence": round(float(df["aggregator_dependence"].mean()), 3),
    }

    # 2) Cluster-wise counts and average GPI
    cluster_summary = (
        df.groupby(["selected_cluster", "cluster_label_name"], as_index=False)
        .agg(
            restaurant_count=("restaurantid", "count"),
            avg_gpi_score=("gpi_score", "mean"),
            avg_total_net_profit=("total_net_profit", "mean"),
            avg_cost_discipline_score=("cost_discipline_score", "mean"),
            dominant_recommendation=("strategy_recommendation", _safe_mode),
        )
    )
    cluster_summary["share_pct"] = (cluster_summary["restaurant_count"] / len(df) * 100.0).round(2)
    cluster_summary = cluster_summary.sort_values("selected_cluster").reset_index(drop=True)
    cluster_round_cols = ["avg_gpi_score", "avg_total_net_profit", "avg_cost_discipline_score", "share_pct"]
    cluster_summary[cluster_round_cols] = cluster_summary[cluster_round_cols].round(3)

    cluster_summary_out = Path(cluster_dashboard_summary_path)
    cluster_summary_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary.to_csv(cluster_summary_out, index=False)

    # 3) Recommendation-wise counts
    rec_counts_df = (
        df["strategy_recommendation"]
        .value_counts()
        .rename_axis("strategy_recommendation")
        .reset_index(name="restaurant_count")
    )
    rec_counts_df["share_pct"] = (rec_counts_df["restaurant_count"] / len(df) * 100.0).round(2)

    # 4) Top restaurants by GPI
    top_cols = [
        "restaurantid",
        "restaurantname",
        "subregion",
        "cuisinetype",
        "segment",
        "gpi_score",
        "gpi_band",
        "cluster_label_name",
        "strategy_recommendation",
        "total_revenue",
        "total_net_profit",
        "cost_discipline_score",
        "aggregator_dependence",
    ]
    top_df = df[top_cols].sort_values("gpi_score", ascending=False).head(top_n).reset_index(drop=True)
    top_df["gpi_rank"] = top_df.index + 1
    top_df = top_df[
        [
            "gpi_rank",
            "restaurantid",
            "restaurantname",
            "subregion",
            "cuisinetype",
            "segment",
            "gpi_score",
            "gpi_band",
            "cluster_label_name",
            "strategy_recommendation",
            "total_revenue",
            "total_net_profit",
            "cost_discipline_score",
            "aggregator_dependence",
        ]
    ]
    top_output = Path(top_restaurants_path)
    top_output.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(top_output, index=False)

    # 5) Drill-down grouped tables
    filter_dir = Path(filter_summary_dir)
    filter_dir.mkdir(parents=True, exist_ok=True)
    filter_tables = {
        "subregion_summary.csv": _group_summary(df, "subregion"),
        "cuisine_summary.csv": _group_summary(df, "cuisinetype"),
        "segment_summary.csv": _group_summary(df, "segment"),
    }
    for filename, table_df in filter_tables.items():
        table_df.to_csv(filter_dir / filename, index=False)

    dashboard_payload = {
        "source_file": clustered_input_path,
        "generated_at_utc": os.getenv("DASHBOARD_GENERATED_AT_UTC", datetime.now(timezone.utc).isoformat()),
        "record_count": int(len(df)),
        "overall_kpi_summary_cards": overall_cards,
        "cluster_summary": cluster_summary.to_dict(orient="records"),
        "recommendation_counts": rec_counts_df.to_dict(orient="records"),
        "gpi_band_counts": (
            df["gpi_band"].value_counts().rename_axis("gpi_band").reset_index(name="restaurant_count").to_dict(orient="records")
        ),
        "output_files": {
            "top_restaurants_csv": top_restaurants_path,
            "cluster_dashboard_summary_csv": cluster_dashboard_summary_path,
            "filter_summary_tables": {
                "subregion": str(filter_dir / "subregion_summary.csv"),
                "cuisine": str(filter_dir / "cuisine_summary.csv"),
                "segment": str(filter_dir / "segment_summary.csv"),
            },
        },
    }

    summary_json_out = Path(dashboard_summary_json_path)
    summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json_out.write_text(json.dumps(dashboard_payload, indent=2), encoding="utf-8")

    return {
        "dashboard_summary_json": dashboard_summary_json_path,
        "top_restaurants_csv": top_restaurants_path,
        "cluster_dashboard_summary_csv": cluster_dashboard_summary_path,
        "filter_summary_dir": filter_summary_dir,
        "records_processed": int(len(df)),
        "top_n_restaurants": int(len(top_df)),
        "recommendation_distribution": rec_counts_df.set_index("strategy_recommendation")["restaurant_count"].to_dict(),
    }


if __name__ == "__main__":
    result = prepare_dashboard_datasets()
    print("Dashboard prep complete:")
    for key, value in result.items():
        print(f"- {key}: {value}")
