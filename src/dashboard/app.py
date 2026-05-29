"""Streamlit dashboard for the V2 factor-based research platform."""

from __future__ import annotations

import json
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


st.set_page_config(page_title="V2 Factor Equity Research", layout="wide")


@st.cache_data(show_spinner=False)
def load_outputs() -> dict[str, object]:
    config = load_config()
    csv_names = [
        "metadata",
        "stock_scores",
        "selected_portfolio",
        "portfolio",
        "research_notes",
        "performance_summary",
        "cumulative_returns",
        "drawdown",
        "rolling_volatility",
        "sector_allocation",
    ]
    outputs: dict[str, object] = {}
    for name in csv_names:
        path = processed_path(f"{name}.csv", config)
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    for name in ["data_status", "portfolio_diagnostics"]:
        path = processed_path(f"{name}.json", config)
        outputs[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return outputs


def ensure_outputs_available() -> dict[str, object]:
    outputs = load_outputs()
    if _df(outputs, "metadata").empty:
        st.warning("No processed outputs found. Run the demo pipeline from the sidebar to create them.")
        if st.sidebar.button("Run Demo Pipeline"):
            run_pipeline(mode="demo")
            st.cache_data.clear()
            st.rerun()
    return parse_dates(outputs)


def parse_dates(outputs: dict[str, object]) -> dict[str, object]:
    for value in outputs.values():
        if isinstance(value, pd.DataFrame):
            for column in ["date", "as_of_date", "latest_rebalance_date"]:
                if column in value.columns:
                    value[column] = pd.to_datetime(value[column], errors="coerce")
    return outputs


def overview(outputs: dict[str, object]) -> None:
    st.title("Macro-Informed Factor Equity Research")
    st.write(
        "This dashboard is an educational A-share screening tool. V2 ranks stocks with a transparent "
        "multi-factor model inspired by accepted factor-investing concepts, then builds a paper portfolio "
        "with explicit constraints. It does not connect to brokers or place trades."
    )
    status = _status(outputs)
    metadata = _df(outputs, "metadata")
    if status.get("data_source") == "demo" or status.get("fallback_used"):
        st.warning("The current run uses demo or fallback data. Treat results as a pipeline test, not market evidence.")
    cols = st.columns(4)
    cols[0].metric("Mode", status.get("mode", "n/a"))
    cols[1].metric("Data Source", status.get("data_source", "n/a"))
    cols[2].metric("Latest Price Date", status.get("actual_latest_price_date", "n/a"))
    cols[3].metric("Benchmark", f"{status.get('benchmark_name', 'Benchmark')} ({status.get('benchmark_code', 'n/a')})")
    cols = st.columns(4)
    cols[0].metric("Requested Start", status.get("requested_start_date", "n/a"))
    cols[1].metric("Requested End", status.get("requested_end_date", "n/a"))
    cols[2].metric("Resolved End", status.get("resolved_end_date", "n/a"))
    latest_rebalance = metadata.iloc[0].get("latest_rebalance_date") if not metadata.empty else None
    cols[3].metric("Latest Rebalance", _date_text(latest_rebalance))
    warnings = status.get("warnings", [])
    if warnings:
        st.subheader("Main Warnings")
        for warning in warnings:
            st.warning(warning)


def factor_model_page(outputs: dict[str, object]) -> None:
    config = load_config()
    weights = config.get("scoring_weights", {})
    st.title("Factor Model")
    st.write(
        "Factor investing studies broad, persistent characteristics such as value, momentum, quality, "
        "investment, and risk. This app uses those ideas as a screening framework. Factor scores do not "
        "guarantee returns and should not be read as recommendations."
    )
    st.subheader("Supported Factors")
    st.table(
        pd.DataFrame(
            [
                ["Value", "1 / PE, 1 / PB, 1 / PS", "Lower valuation multiples rank higher after cleaning."],
                ["Momentum", "6-month and 12-month excluding recent month", "Uses trailing prices only."],
                ["Quality", "ROE, ROA, gross margin, operating margin", "Higher profitability and margins rank higher."],
                ["Investment", "Asset growth and capex growth", "Lower aggressive investment ranks higher by default."],
                ["Low volatility", "Realized volatility and drawdown", "Lower risk improves risk-adjusted attractiveness."],
                ["Liquidity", "Average trading amount", "Mostly a tradability filter and small score component."],
                ["Industry", "Macro transmission matrix", "Editable macro-industry context, not a guarantee."],
            ],
            columns=["Factor", "Formula / Inputs", "Interpretation"],
        )
    )
    st.subheader("Current Weights")
    st.dataframe(pd.DataFrame([weights]).T.rename(columns={0: "weight"}), use_container_width=True)
    st.subheader("Composite Score")
    st.code(
        """
total_score =
    w_value * value_score
  + w_momentum * momentum_score
  + w_quality * quality_score
  + w_investment * investment_score
  + w_low_volatility * low_volatility_score
  + w_liquidity * liquidity_score
  + w_industry * industry_score
  - w_risk_penalty * risk_penalty_score
        """.strip()
    )


def data_status_page(outputs: dict[str, object]) -> None:
    st.title("Data Status")
    status = _status(outputs)
    st.write("This page shows which data source was used and whether fallback or demo data entered the run.")
    st.json(status)


def stock_ranking_page(outputs: dict[str, object]) -> None:
    st.title("Stock Ranking")
    scores = latest_frame(_df(outputs, "stock_scores"))
    if scores.empty:
        st.warning("No rankings are available.")
        return
    st.write("The table ranks stocks by the V2 composite factor score after data cleaning and risk flags.")
    search = st.text_input("Search ticker or name", "")
    industries = ["All"] + sorted(scores["industry"].dropna().unique().tolist())
    industry = st.selectbox("Industry", industries)
    filtered = scores.copy()
    if industry != "All":
        filtered = filtered[filtered["industry"].eq(industry)]
    if search:
        text = search.lower()
        names = filtered["stock_name"] if "stock_name" in filtered.columns else pd.Series("", index=filtered.index)
        filtered = filtered[
            filtered["ticker"].astype(str).str.lower().str.contains(text)
            | names.astype(str).str.lower().str.contains(text)
        ]
    columns = [
        "ticker",
        "stock_name",
        "industry",
        "total_score",
        "value_score",
        "momentum_score",
        "quality_score",
        "investment_score",
        "low_volatility_score",
        "liquidity_score",
        "industry_score",
        "risk_penalty_score",
        "data_quality_warning",
    ]
    st.download_button("Download ranking CSV", filtered.to_csv(index=False), "stock_ranking.csv", "text/csv")
    st.dataframe(filtered[[column for column in columns if column in filtered.columns]], use_container_width=True, hide_index=True)


def portfolio_page(outputs: dict[str, object]) -> None:
    st.title("Portfolio")
    portfolio = latest_frame(_df(outputs, "selected_portfolio"))
    diagnostics = outputs.get("portfolio_diagnostics", {})
    if portfolio.empty:
        st.warning("No selected portfolio is available.")
        return
    st.write(
        "Portfolio construction starts with the universe, applies data, liquidity, and risk filters, ranks by "
        "total_score, selects the top names, applies equal or score weights, caps positions and industries, "
        "then normalizes final weights."
    )
    cols = st.columns(4)
    cols[0].metric("Selected Stocks", len(portfolio))
    cols[1].metric("Largest Position", _pct(portfolio["final_weight"].max()))
    cols[2].metric("Largest Industry", _pct(portfolio.groupby("industry")["final_weight"].sum().max()))
    cols[3].metric("Method", diagnostics.get("weighting_method", "n/a"))
    if diagnostics.get("constraints_binding"):
        st.warning("Binding constraints: " + ", ".join(diagnostics["constraints_binding"]))
    st.download_button("Download selected portfolio CSV", portfolio.to_csv(index=False), "selected_portfolio.csv", "text/csv")
    columns = [
        "ticker",
        "stock_name",
        "industry",
        "total_score",
        "final_weight",
        "selection_reason",
        "key_risk_warning",
    ]
    st.dataframe(portfolio[[column for column in columns if column in portfolio.columns]], use_container_width=True, hide_index=True)
    allocation = portfolio.groupby("industry", as_index=False)["final_weight"].sum()
    st.plotly_chart(px.bar(allocation, x="industry", y="final_weight", title="Industry Allocation"), use_container_width=True)
    with st.expander("Portfolio Diagnostics"):
        st.json(diagnostics)


def backtest_page(outputs: dict[str, object]) -> None:
    st.title("Backtest")
    summary = _df(outputs, "performance_summary")
    cumulative = _df(outputs, "cumulative_returns")
    drawdown = _df(outputs, "drawdown")
    rolling_volatility = _df(outputs, "rolling_volatility")
    if summary.empty:
        st.warning("No backtest summary is available.")
        return
    metrics = summary.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Annualized Return", _pct(metrics.get("annualized_return")))
    cols[1].metric("Annualized Volatility", _pct(metrics.get("annualized_volatility")))
    cols[2].metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    cols[3].metric("Max Drawdown", _pct(metrics.get("maximum_drawdown")))
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


def notes_page(outputs: dict[str, object]) -> None:
    st.title("Research Notes")
    notes = latest_frame(_df(outputs, "research_notes"))
    if notes.empty:
        st.warning("No notes are available.")
        return
    ticker = st.selectbox("Ticker", notes["ticker"].tolist())
    note = notes[notes["ticker"].eq(ticker)].iloc[0]["research_note"]
    st.text(note)


def methodology_page(outputs: dict[str, object]) -> None:
    st.title("Methodology & Limitations")
    st.write(
        "V2 is a factor-based screening model with macro-industry context. It is for education, research, "
        "and paper trading only. It does not give financial advice and does not execute orders."
    )
    st.subheader("Limitations")
    for item in [
        "Free APIs can be delayed, unstable, incomplete, or schema-changing.",
        "Look-ahead bias is reduced with trailing prices and announcement dates, but new connectors need review.",
        "Survivorship bias can remain if the available universe excludes delisted firms.",
        "Factor overfitting and regime shifts can make historical relationships unreliable.",
        "Transaction costs, slippage, and liquidity limits are simplified.",
        "Missing fundamentals are flagged rather than invented.",
    ]:
        st.warning(item)


def latest_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "as_of_date" not in df.columns:
        return df
    return df[df["as_of_date"].eq(df["as_of_date"].max())].copy()


def _df(outputs: dict[str, object], name: str) -> pd.DataFrame:
    value = outputs.get(name, pd.DataFrame())
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _status(outputs: dict[str, object]) -> dict[str, object]:
    value = outputs.get("data_status", {})
    return value if isinstance(value, dict) else {}


def _pct(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def _date_text(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return str(pd.Timestamp(value).date())


def main() -> None:
    outputs = ensure_outputs_available()
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Factor Model",
            "Data Status",
            "Stock Ranking",
            "Portfolio",
            "Backtest",
            "Research Notes",
            "Methodology & Limitations",
        ],
    )
    pages = {
        "Overview": overview,
        "Factor Model": factor_model_page,
        "Data Status": data_status_page,
        "Stock Ranking": stock_ranking_page,
        "Portfolio": portfolio_page,
        "Backtest": backtest_page,
        "Research Notes": notes_page,
        "Methodology & Limitations": methodology_page,
    }
    pages[page](outputs)


if __name__ == "__main__":
    main()
