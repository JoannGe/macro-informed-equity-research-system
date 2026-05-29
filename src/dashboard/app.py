"""Streamlit dashboard for the macro-informed equity research platform."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.run_all import run_pipeline  # noqa: E402
from src.utils.io_utils import load_config, processed_path  # noqa: E402


st.set_page_config(page_title="Macro Equity Research", layout="wide")


@st.cache_data(show_spinner=False)
def load_outputs() -> dict[str, pd.DataFrame]:
    config = load_config()
    names = [
        "metadata",
        "macro_regimes",
        "industry_scores",
        "stock_scores",
        "portfolio",
        "research_notes",
        "performance_summary",
        "cumulative_returns",
        "drawdown",
        "rolling_volatility",
        "sector_allocation",
    ]
    outputs = {}
    for name in names:
        path = processed_path(f"{name}.csv", config)
        if path.exists():
            outputs[name] = pd.read_csv(path)
        else:
            outputs[name] = pd.DataFrame()
    return outputs


def ensure_outputs_available() -> dict[str, pd.DataFrame]:
    outputs = load_outputs()
    if outputs["metadata"].empty:
        st.warning("No processed outputs found. Run demo mode from the sidebar to create them.")
        if st.sidebar.button("Run Demo Pipeline"):
            run_pipeline(mode="demo")
            st.cache_data.clear()
            st.rerun()
    return outputs


def parse_dates(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for frame in outputs.values():
        for column in ["date", "as_of_date", "latest_rebalance_date", "macro_observation_date"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return outputs


def latest_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "as_of_date" not in df.columns:
        return df
    return df[df["as_of_date"].eq(df["as_of_date"].max())].copy()


def overview(outputs: dict[str, pd.DataFrame]) -> None:
    metadata = outputs["metadata"]
    st.title("Macro-Informed Equity Research")
    st.caption("Educational screening and paper-research platform. Not financial advice. No broker connection.")
    if metadata.empty:
        return
    row = metadata.iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Mode", str(row.get("mode", "unknown")))
    col2.metric("Latest Rebalance", _date_text(row.get("latest_rebalance_date")))
    col3.metric("Benchmark", str(row.get("benchmark_code", "399673")))
    warning = row.get("warnings")
    if isinstance(warning, str) and warning.strip():
        st.warning(warning)
    summary = outputs["performance_summary"]
    if not summary.empty:
        st.subheader("Backtest Snapshot")
        metrics = summary.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Annualized Return", _pct(metrics.get("annualized_return")))
        cols[1].metric("Annualized Volatility", _pct(metrics.get("annualized_volatility")))
        cols[2].metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
        cols[3].metric("Max Drawdown", _pct(metrics.get("maximum_drawdown")))


def macro_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Macro Regime Dashboard")
    macro = outputs["macro_regimes"]
    if macro.empty:
        return
    latest = macro.sort_values("as_of_date").iloc[-1]
    cols = st.columns(5)
    cols[0].metric("Growth", latest["growth_regime"])
    cols[1].metric("Inflation", latest["inflation_regime"])
    cols[2].metric("Rates", latest["rates_regime"])
    cols[3].metric("Credit", latest["credit_regime"])
    cols[4].metric("Liquidity", latest["liquidity_regime"])
    score_columns = [
        "growth_score",
        "inflation_score",
        "rates_score",
        "credit_score",
        "liquidity_score",
        "commodity_score",
        "policy_score",
        "risk_appetite_score",
    ]
    chart_data = macro.melt("as_of_date", score_columns, var_name="indicator_group", value_name="rolling_z_score")
    st.plotly_chart(px.line(chart_data, x="as_of_date", y="rolling_z_score", color="indicator_group"), use_container_width=True)


def industry_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Industry Scores")
    industry_scores = outputs["industry_scores"]
    if industry_scores.empty:
        return
    latest = latest_frame(industry_scores).sort_values("industry_score", ascending=False)
    st.plotly_chart(px.bar(latest, x="industry", y="industry_score", color="industry_score"), use_container_width=True)
    st.dataframe(
        latest[["industry", "industry_score", "industry_explanation"]],
        use_container_width=True,
        hide_index=True,
    )
    component_columns = [column for column in latest.columns if column.startswith("component_")]
    st.subheader("Transmission Components")
    st.dataframe(latest[["industry", *component_columns]], use_container_width=True, hide_index=True)


def ranking_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Stock Ranking")
    scores = latest_frame(outputs["stock_scores"])
    if scores.empty:
        return
    industries = ["All"] + sorted(scores["industry"].dropna().unique().tolist())
    selected_industry = st.selectbox("Industry", industries)
    if selected_industry != "All":
        scores = scores[scores["industry"].eq(selected_industry)]
    score_columns = [
        "stock_code",
        "stock_name",
        "industry",
        "total_score",
        "industry_score",
        "quality_score",
        "growth_score",
        "valuation_score",
        "momentum_score",
        "risk_score",
        "excluded_by_risk_filter",
        "risk_filter_reason",
        "data_quality_warning",
    ]
    st.dataframe(scores[score_columns].sort_values("total_score", ascending=False), use_container_width=True, hide_index=True)


def portfolio_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Selected Portfolio")
    portfolio = latest_frame(outputs["portfolio"])
    if portfolio.empty:
        st.warning("No portfolio holdings are available.")
        return
    cols = st.columns(3)
    cols[0].metric("Holdings", len(portfolio))
    cols[1].metric("Largest Position", _pct(portfolio["weight"].max()))
    cols[2].metric("Largest Industry", _pct(portfolio.groupby("industry")["weight"].sum().max()))
    st.dataframe(
        portfolio[
            [
                "selection_rank",
                "stock_code",
                "stock_name",
                "industry",
                "weight",
                "total_score",
                "risk_filter_reason",
                "data_quality_warning",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    allocation = portfolio.groupby("industry", as_index=False)["weight"].sum()
    st.plotly_chart(px.pie(allocation, names="industry", values="weight"), use_container_width=True)


def backtest_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Backtest Results")
    summary = outputs["performance_summary"]
    cumulative = outputs["cumulative_returns"]
    drawdown = outputs["drawdown"]
    rolling_volatility = outputs["rolling_volatility"]
    if summary.empty:
        return
    metrics = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Annualized Return", _pct(metrics.get("annualized_return")))
    cols[1].metric("Annualized Volatility", _pct(metrics.get("annualized_volatility")))
    cols[2].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    cols[3].metric("Benchmark Relative", _pct(metrics.get("benchmark_relative_return")))
    st.dataframe(summary, use_container_width=True, hide_index=True)
    if not cumulative.empty:
        chart = cumulative.melt("date", ["portfolio_cumulative", "benchmark_cumulative"], var_name="series", value_name="value")
        st.plotly_chart(px.line(chart, x="date", y="value", color="series"), use_container_width=True)
    if not drawdown.empty:
        chart = drawdown.melt("date", ["portfolio_drawdown", "benchmark_drawdown"], var_name="series", value_name="drawdown")
        st.plotly_chart(px.line(chart, x="date", y="drawdown", color="series"), use_container_width=True)
    if not rolling_volatility.empty:
        chart = rolling_volatility.melt(
            "date",
            ["portfolio_rolling_volatility", "benchmark_rolling_volatility"],
            var_name="series",
            value_name="rolling_volatility",
        )
        st.plotly_chart(px.line(chart, x="date", y="rolling_volatility", color="series"), use_container_width=True)


def notes_page(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Research Notes")
    notes = latest_frame(outputs["research_notes"])
    if notes.empty:
        return
    stock = st.selectbox("Stock", notes["stock_code"].tolist(), format_func=lambda code: _note_label(notes, code))
    note = notes[notes["stock_code"].eq(stock)].iloc[0]["research_note"]
    st.text(note)


def methodology_page() -> None:
    st.title("Methodology")
    st.markdown(
        """
This platform is a research and paper-trading system for studying macro-industry-company logic.
It does not connect to brokers, place orders, or provide financial advice.

The model combines an interpretable macro regime classifier, manually editable industry transmission
assumptions, cross-sectional firm factors, risk filters, constrained long-only portfolio construction,
and a quarterly backtest with transaction costs.

Selection variables use trailing prices and fundamentals whose announcement dates are on or before
the rebalance date. This reduces look-ahead bias, but does not remove survivorship bias, overfitting,
missing-data risk, API instability, or regime-shift risk.
        """
    )
    st.subheader("Composite Score")
    st.code(
        """
total_score =
    w_macro_industry * industry_score
  + w_quality * quality_score
  + w_growth * growth_score
  + w_valuation * valuation_score
  + w_momentum * momentum_score
  - w_risk * risk_score
        """.strip()
    )


def _pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return str(pd.Timestamp(value).date())


def _note_label(notes: pd.DataFrame, code: str) -> str:
    row = notes[notes["stock_code"].eq(code)].iloc[0]
    return f"{row.get('stock_name', code)} ({code})"


def main() -> None:
    outputs = parse_dates(ensure_outputs_available())
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Macro regime dashboard",
            "Industry scores",
            "Stock ranking",
            "Selected portfolio",
            "Backtest results",
            "Research notes",
            "Methodology",
        ],
    )
    if page == "Overview":
        overview(outputs)
    elif page == "Macro regime dashboard":
        macro_page(outputs)
    elif page == "Industry scores":
        industry_page(outputs)
    elif page == "Stock ranking":
        ranking_page(outputs)
    elif page == "Selected portfolio":
        portfolio_page(outputs)
    elif page == "Backtest results":
        backtest_page(outputs)
    elif page == "Research notes":
        notes_page(outputs)
    else:
        methodology_page()


if __name__ == "__main__":
    main()
