from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Restaurant Growth Dashboard",
    page_icon="🍽️",
    layout="wide",
)


THEMES: Dict[str, Dict[str, object]] = {
    "Navy Glass": {
        "bg": "radial-gradient(circle at top left, rgba(99,102,241,0.16), transparent 28%), radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 24%), linear-gradient(180deg, #050b16 0%, #08111f 100%)",
        "card_bg": "linear-gradient(180deg, rgba(14,24,44,0.94), rgba(8,14,28,0.96))",
        "card_border": "rgba(255,255,255,0.08)",
        "text": "#edf2ff",
        "muted": "#9db3d6",
        "chip_bg": "rgba(255,255,255,0.05)",
        "chip_border": "rgba(255,255,255,0.10)",
        "palette": ["#4f46e5", "#14b8a6", "#f59e0b", "#ec4899", "#38bdf8", "#22c55e"],
        "plot_bg": "#0b162b",
        "grid": "#325173",
        "plot_text": "#9db3d6",
        "field_bg": "#1f2435",
        "field_text": "#edf2ff",
    },
    "Slate Light": {
        "bg": "radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 30%), radial-gradient(circle at top right, rgba(20,184,166,0.08), transparent 25%), linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%)",
        "card_bg": "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,252,255,0.98))",
        "card_border": "rgba(148,163,184,0.35)",
        "text": "#0a1325",
        "muted": "#1f334a",
        "chip_bg": "#f1f5f9",
        "chip_border": "rgba(100,116,139,0.35)",
        "palette": ["#2563eb", "#0ea5e9", "#06b6d4", "#14b8a6", "#16a34a", "#f59e0b"],
        "plot_bg": "#ffffff",
        "grid": "#cbd5e1",
        "plot_text": "#111827",
        "field_bg": "#1f2435",
        "field_text": "#f8fafc",
    },
    "Emerald Ops": {
        "bg": "radial-gradient(circle at top left, rgba(16,185,129,0.16), transparent 32%), radial-gradient(circle at top right, rgba(14,165,233,0.12), transparent 28%), linear-gradient(180deg, #031611 0%, #07231c 100%)",
        "card_bg": "linear-gradient(180deg, rgba(4,31,25,0.96), rgba(3,22,18,0.96))",
        "card_border": "rgba(94,234,212,0.20)",
        "text": "#ecfdf5",
        "muted": "#a7f3d0",
        "chip_bg": "rgba(16,185,129,0.12)",
        "chip_border": "rgba(94,234,212,0.28)",
        "palette": ["#10b981", "#14b8a6", "#22c55e", "#84cc16", "#0ea5e9", "#f59e0b"],
        "plot_bg": "#06241e",
        "grid": "#2e6f62",
        "plot_text": "#a7f3d0",
        "field_bg": "#1c2734",
        "field_text": "#ecfdf5",
    },
    "Mono Focus": {
        "bg": "linear-gradient(180deg, #f8fafc 0%, #edf1f5 100%)",
        "card_bg": "#ffffff",
        "card_border": "rgba(17,24,39,0.28)",
        "text": "#0b0f19",
        "muted": "#1f2937",
        "chip_bg": "#f3f4f6",
        "chip_border": "rgba(17,24,39,0.24)",
        "palette": ["#111827", "#1f2937", "#374151", "#4b5563", "#6b7280", "#9ca3af"],
        "plot_bg": "#ffffff",
        "grid": "#d1d5db",
        "plot_text": "#0f172a",
        "field_bg": "#1f2435",
        "field_text": "#f8fafc",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = repo_root() / "data" / "processed" / "clustered_restaurants.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    if "restaurantid" in df.columns:
        df["restaurantid"] = df["restaurantid"].astype(str)

    for col in [
        "gpi_score",
        "total_revenue",
        "total_net_profit",
        "aggregator_dependence",
        "cost_discipline_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def format_percent(ratio: float, decimals: int = 1) -> str:
    if pd.isna(ratio):
        return "-"
    return f"{ratio * 100:.{decimals}f}%"


def unique_values(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    values = sorted(v for v in df[column].dropna().astype(str).unique().tolist() if v.strip())
    return values


def apply_theme(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: {t['bg']};
            color: {t['text']};
          }}
          [data-testid="stHeader"] {{ background: transparent; }}
          .block-container {{
            padding-top: 1rem;
            max-width: 1550px;
          }}
          .card {{
            border: 1px solid {t['card_border']};
            background: {t['card_bg']};
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 12px;
          }}
          .eyebrow {{
            display: inline-block;
            border: 1px solid {t['chip_border']};
            background: {t['chip_bg']};
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            color: {t['muted']};
            margin-bottom: 10px;
          }}
          .hero-title {{
            margin: 0;
            font-size: clamp(2rem, 3vw, 2.9rem);
            line-height: 1.05;
            letter-spacing: -0.02em;
            color: {t['text']};
          }}
          .muted {{ color: {t['muted']}; }}
          .kpi-label {{ font-size: 0.88rem; color: {t['muted']}; margin-bottom: 3px; }}
          .kpi-value {{ font-size: 1.95rem; font-weight: 700; line-height: 1.1; color: {t['text']}; }}
          .kpi-hint {{ font-size: 0.9rem; color: {t['muted']}; margin-top: 2px; }}
          .tag {{
            display: inline-block;
            border: 1px solid {t['chip_border']};
            background: {t['chip_bg']};
            padding: 4px 10px;
            border-radius: 999px;
            margin-right: 6px;
            margin-top: 6px;
            font-size: 12px;
          }}
          .hero-stat-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
          }}
          .hero-stat {{
            border: 1px solid {t['chip_border']};
            background: {t['chip_bg']};
            border-radius: 12px;
            padding: 9px 10px;
          }}
          .hero-stat-label {{
            color: {t['muted']};
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.07em;
          }}
          .hero-stat-value {{
            margin-top: 4px;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.05;
            color: {t['text']};
          }}
          .chart-title {{
            margin: 0;
            font-size: 1.12rem;
            font-weight: 700;
            color: {t['text']};
            line-height: 1.15;
          }}
          .chart-subtitle {{
            margin: 2px 0 8px 0;
            font-size: 0.9rem;
            color: {t['muted']};
          }}
          div[data-testid="stMetric"] {{
            border: 1px solid {t['card_border']};
            background: {t['card_bg']};
            border-radius: 12px;
            padding: 10px 12px;
          }}
          div[data-testid="stMetric"] label,
          div[data-testid="stMetric"] [data-testid="stMetricLabel"],
          div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {t['text']} !important;
          }}
          div[data-testid="stDataFrame"] {{
            border: 1px solid {t['card_border']};
            border-radius: 12px;
            overflow: hidden;
          }}
          div[data-testid="stPlotlyChart"] {{
            border: 1px solid {t['card_border']};
            border-radius: 12px;
            padding: 6px 6px 0 6px;
            background: {t['plot_bg']};
          }}
          .stButton > button {{
            background: {t['field_bg']} !important;
            color: {t['field_text']} !important;
            border: 1px solid {t['card_border']} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
          }}
          .stButton > button:hover {{
            filter: brightness(1.08);
          }}
          .stButton > button:focus,
          .stButton > button:focus-visible {{
            outline: 2px solid {t['grid']} !important;
            outline-offset: 1px !important;
          }}
          label, .stSelectbox label, .stTextInput label {{
            color: {t['text']} !important;
          }}
          .stSelectbox div[data-baseweb="select"] > div,
          .stTextInput input {{
            background: {t['field_bg']} !important;
            color: {t['field_text']} !important;
            border-color: {t['card_border']} !important;
          }}
          .stTextInput input::placeholder {{
            color: rgba(255,255,255,0.62) !important;
          }}
          div[data-baseweb="popover"] *,
          div[data-baseweb="menu"] *,
          div[role="listbox"] * {{
            color: {t['field_text']} !important;
          }}
          div[data-baseweb="popover"],
          div[data-baseweb="menu"],
          div[role="listbox"] {{
            background: {t['field_bg']} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_bar(series: pd.Series, title: str, subtitle: str, theme_name: str) -> None:
    t = THEMES[theme_name]
    palette = list(t["palette"])

    st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)

    bar_df = series.reset_index()
    bar_df.columns = ["category", "count"]
    bar_df["category"] = bar_df["category"].astype(str)

    fig = px.bar(
        bar_df,
        x="category",
        y="count",
        color="category",
        color_discrete_sequence=palette,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        marker_line_width=0,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=22, b=80),
        height=300,
        paper_bgcolor=t["plot_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(color=t["plot_text"], size=12),
        xaxis=dict(
            tickangle=-22,
            tickfont=dict(size=10, color=t["plot_text"]),
            title=None,
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            gridcolor=t["grid"],
            zeroline=False,
            tickfont=dict(color=t["plot_text"]),
        ),
        hoverlabel=dict(
            bgcolor=t["plot_bg"],
            bordercolor=t["grid"],
            font=dict(color=t["text"]),
        ),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ],
        },
    )


def plot_scatter(df: pd.DataFrame, theme_name: str) -> None:
    t = THEMES[theme_name]
    palette = list(t["palette"])

    st.markdown('<div class="chart-title">Revenue vs Net Profit</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-subtitle">Cluster-colored portfolio spread</div>', unsafe_allow_html=True)

    hover_cols = {
        "restaurantname": True,
        "restaurantid": True,
        "gpi_score": ":.1f",
        "gpi_band": True,
        "strategy_recommendation": True,
        "cluster_label_name": False,
    }
    fig = px.scatter(
        df,
        x="total_revenue",
        y="total_net_profit",
        color="cluster_label_name",
        color_discrete_sequence=palette,
        hover_name="restaurantname",
        hover_data=hover_cols,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.72), selector=dict(mode="markers"))
    fig.update_layout(
        margin=dict(l=10, r=10, t=22, b=6),
        height=500,
        paper_bgcolor=t["plot_bg"],
        plot_bgcolor=t["plot_bg"],
        font=dict(color=t["plot_text"], size=12),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=t["plot_text"], size=11),
        ),
        xaxis=dict(
            title="Total Revenue",
            gridcolor=t["grid"],
            zeroline=False,
            tickfont=dict(color=t["plot_text"]),
            title_font=dict(color=t["plot_text"]),
        ),
        yaxis=dict(
            title="Total Net Profit",
            gridcolor=t["grid"],
            zeroline=False,
            tickfont=dict(color=t["plot_text"]),
            title_font=dict(color=t["plot_text"]),
        ),
        hoverlabel=dict(
            bgcolor=t["plot_bg"],
            bordercolor=t["grid"],
            font=dict(color=t["text"]),
        ),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": "hover",
            "modeBarButtonsToRemove": [
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ],
        },
    )


def render_cluster_cards(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby("cluster_label_name", dropna=False)
        .agg(
            count=("restaurantid", "count"),
            avg_gpi=("gpi_score", "mean"),
            avg_profit=("total_net_profit", "mean"),
            description=("cluster_description", "first"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    dominant_action = (
        df.groupby("cluster_label_name")["strategy_recommendation"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "-")
        .rename("dominant_action")
        .reset_index()
    )
    grouped = grouped.merge(dominant_action, on="cluster_label_name", how="left")

    st.markdown("### Archetype Profile Cards")
    cols = st.columns(2)
    for i, row in grouped.iterrows():
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div style="font-weight:700;">{row['cluster_label_name']}</div>
                    <div class="tag">{int(row['count'])}</div>
                  </div>
                  <div class="muted" style="font-size:0.92rem;margin-bottom:8px;">{row['description']}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
                    <div><div class="muted" style="font-size:11px;">Avg GPI</div><div style="font-weight:700;">{format_number(row['avg_gpi'], 1)}</div></div>
                    <div><div class="muted" style="font-size:11px;">Avg Profit</div><div style="font-weight:700;">{format_currency(row['avg_profit'])}</div></div>
                    <div><div class="muted" style="font-size:11px;">Dominant Action</div><div style="font-weight:700;">{row['dominant_action']}</div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Failed to load dashboard data: {exc}")
        st.stop()

    if "theme_selector" not in st.session_state:
        st.session_state["theme_selector"] = "Navy Glass"

    theme_name = st.session_state["theme_selector"]
    apply_theme(theme_name)

    total_high = int((df["gpi_band"] == "High Potential").sum())
    total_agg = int((df["aggregator_dependence"] >= 0.75).sum())
    total_neg = int((df["total_net_profit"] <= 0).sum())

    hero_left, hero_right = st.columns([2.2, 1.4])
    with hero_left:
        st.markdown('<div class="eyebrow">Restaurant Growth Potential System</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Growth Strategy Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Portfolio health, archetypes, and execution priorities in one view.</div>', unsafe_allow_html=True)
    with hero_right:
        st.selectbox("Theme", list(THEMES.keys()), key="theme_selector")
        st.markdown(
            f"""
            <div class="hero-stat-grid">
              <div class="hero-stat"><div class="hero-stat-label">High Potential</div><div class="hero-stat-value">{format_number(float(total_high))}</div></div>
              <div class="hero-stat"><div class="hero-stat-label">Aggregator-Heavy</div><div class="hero-stat-value">{format_number(float(total_agg))}</div></div>
              <div class="hero-stat"><div class="hero-stat-label">Negative Profit</div><div class="hero-stat-value">{format_number(float(total_neg))}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Portfolio Filters")
    search_col, reset_col = st.columns([5, 1])
    with search_col:
        search = st.text_input(
            "Search restaurant or ID",
            value="",
            key="search_query",
            placeholder="Type name or restaurant ID",
        )
    with reset_col:
        if st.button("Reset"):
            for key in [
                "search_query",
                "subregion_filter",
                "cuisine_filter",
                "segment_filter",
                "archetype_filter",
                "gpi_filter",
                "action_filter",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    subregion_filter = c1.selectbox("Subregion", ["All"] + unique_values(df, "subregion"), key="subregion_filter")
    cuisine_filter = c2.selectbox("Cuisine", ["All"] + unique_values(df, "cuisinetype"), key="cuisine_filter")
    segment_filter = c3.selectbox("Segment", ["All"] + unique_values(df, "segment"), key="segment_filter")
    archetype_filter = c4.selectbox("Archetype", ["All"] + unique_values(df, "cluster_label_name"), key="archetype_filter")
    gpi_filter = c5.selectbox("GPI Band", ["All"] + unique_values(df, "gpi_band"), key="gpi_filter")
    action_filter = c6.selectbox("Recommendation", ["All"] + unique_values(df, "strategy_recommendation"), key="action_filter")

    query = search.strip().lower()
    filtered = df[
        (
            (query == "")
            | (df["restaurantname"].astype(str).str.lower().str.contains(query, na=False))
            | (df["restaurantid"].astype(str).str.lower().str.contains(query, na=False))
        )
        & ((subregion_filter == "All") | (df["subregion"].astype(str) == subregion_filter))
        & ((cuisine_filter == "All") | (df["cuisinetype"].astype(str) == cuisine_filter))
        & ((segment_filter == "All") | (df["segment"].astype(str) == segment_filter))
        & ((archetype_filter == "All") | (df["cluster_label_name"].astype(str) == archetype_filter))
        & ((gpi_filter == "All") | (df["gpi_band"].astype(str) == gpi_filter))
        & ((action_filter == "All") | (df["strategy_recommendation"].astype(str) == action_filter))
    ].copy()

    if filtered.empty:
        st.warning("No restaurants matched these filters. Showing full portfolio.")
        filtered = df.copy()

    avg_gpi = float(filtered["gpi_score"].mean())
    avg_revenue = float(filtered["total_revenue"].mean())
    avg_profit = float(filtered["total_net_profit"].mean())
    high_potential = int((filtered["gpi_band"] == "High Potential").sum())
    scale_candidates = int((filtered["strategy_recommendation"] == "Scale Aggressively").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Restaurants", format_number(float(len(filtered))), f"{format_number(float(len(df)))} total")
    with k2:
        render_kpi_card("Avg GPI", format_number(avg_gpi, 1), f"{format_number(float(high_potential))} high potential")
    with k3:
        render_kpi_card("Scale Candidates", format_number(float(scale_candidates)), "scale aggressively")
    with k4:
        render_kpi_card("Avg Revenue", format_currency(avg_revenue), f"{format_currency(avg_profit)} avg net profit")

    d1, d2, d3 = st.columns(3)
    with d1:
        plot_bar(filtered["cluster_label_name"].value_counts(dropna=False), "Archetype Mix", "Cluster distribution", theme_name)
    with d2:
        plot_bar(filtered["gpi_band"].value_counts(dropna=False), "GPI Bands", "Readiness spread", theme_name)
    with d3:
        plot_bar(filtered["strategy_recommendation"].value_counts(dropna=False), "Action Mix", "Recommendation distribution", theme_name)

    left, right = st.columns([1.65, 1.0])
    with left:
        plot_scatter(filtered, theme_name)

    with right:
        st.markdown("### Restaurant Detail")
        options = (
            filtered.sort_values("gpi_score", ascending=False)
            .assign(label=lambda x: x["restaurantname"].astype(str) + " (" + x["restaurantid"].astype(str) + ")")
            [["restaurantid", "label"]]
            .drop_duplicates()
        )
        selected_label = st.selectbox("Select restaurant", options["label"].tolist(), key="restaurant_selector")
        selected_id = options.loc[options["label"] == selected_label, "restaurantid"].iloc[0]
        row = filtered.loc[filtered["restaurantid"].astype(str) == str(selected_id)].iloc[0]

        st.markdown(
            f"""
            <div class="card">
              <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;">{row['restaurantid']}</div>
              <div style="font-size:1.35rem;font-weight:700;margin-top:4px;">{row['restaurantname']}</div>
              <div class="muted" style="margin-top:4px;">{row['cuisinetype']} · {row['segment']} · {row['subregion']}</div>
              <div style="margin-top:8px;font-weight:700;">GPI {format_number(float(row['gpi_score']), 1)}</div>
              <div style="margin-top:6px;">
                <span class="tag">{row['gpi_band']}</span>
                <span class="tag">{row['cluster_label_name']}</span>
                <span class="tag">{row['strategy_recommendation']}</span>
              </div>
              <div class="muted" style="margin-top:10px;">{row['recommendation_reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Revenue", format_currency(float(row["total_revenue"])))
            st.metric("Aggregator", format_percent(float(row["aggregator_dependence"]), 1))
        with m2:
            st.metric("Net Profit", format_currency(float(row["total_net_profit"])))
            st.metric("Cost Discipline", format_percent(float(row["cost_discipline_score"]), 1))

    st.markdown("### Top GPI Restaurants")
    top = (
        filtered.sort_values("gpi_score", ascending=False)
        .head(12)
        .loc[
            :,
            [
                "restaurantid",
                "restaurantname",
                "cuisinetype",
                "subregion",
                "cluster_label_name",
                "gpi_score",
                "strategy_recommendation",
                "total_revenue",
            ],
        ]
        .copy()
    )
    top["gpi_score"] = top["gpi_score"].map(lambda x: format_number(float(x), 1))
    top["total_revenue"] = top["total_revenue"].map(lambda x: format_currency(float(x)))
    top.columns = ["Restaurant ID", "Restaurant", "Cuisine", "Subregion", "Archetype", "GPI", "Action", "Revenue"]
    st.dataframe(top, use_container_width=True, hide_index=True)

    render_cluster_cards(filtered)


if __name__ == "__main__":
    main()
