"""Quarterly long-only backtester with lagged holdings."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.metrics import summarize_performance
from src.processing.clean_prices import clean_prices


def run_backtest(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Run a daily return backtest using only post-rebalance returns."""

    price_returns = _build_stock_returns(prices)
    benchmark_returns = _build_benchmark_returns(benchmark)
    transaction_cost = float(config.get("portfolio", {}).get("transaction_cost_bps", 10)) / 10_000
    risk_free_rate = float(config.get("backtest", {}).get("risk_free_rate_annual", 0.02))

    if portfolio.empty:
        empty_returns = pd.DataFrame(columns=["date", "portfolio_return", "benchmark_return"])
        return {
            "daily_returns": empty_returns,
            "performance_summary": summarize_performance(empty_returns, pd.DataFrame(), risk_free_rate),
            "turnover": pd.DataFrame(),
            "sector_allocation": pd.DataFrame(),
            "cumulative_returns": pd.DataFrame(),
            "drawdown": pd.DataFrame(),
            "rolling_volatility": pd.DataFrame(),
        }

    rebalance_dates = sorted(pd.to_datetime(portfolio["as_of_date"].unique()))
    daily_rows: list[dict[str, object]] = []
    turnover_rows: list[dict[str, object]] = []
    previous_weights = pd.Series(dtype=float)

    for idx, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else price_returns["date"].max()
        holdings = portfolio[portfolio["as_of_date"].eq(rebalance_date)].copy()
        weights = holdings.set_index("stock_code")["weight"].astype(float)
        turnover = _calculate_turnover(previous_weights, weights)
        turnover_rows.append({"as_of_date": rebalance_date, "turnover": turnover})

        interval = price_returns[
            (price_returns["date"] > rebalance_date) & (price_returns["date"] <= next_rebalance)
        ]
        if interval.empty:
            previous_weights = weights
            continue
        interval_returns = interval[interval["stock_code"].isin(weights.index)].copy()
        interval_returns["weight"] = interval_returns["stock_code"].map(weights)
        portfolio_daily = (
            interval_returns.assign(weighted_return=interval_returns["daily_return"] * interval_returns["weight"])
            .groupby("date", as_index=False)["weighted_return"]
            .sum()
            .rename(columns={"weighted_return": "portfolio_return"})
        )
        first_date = portfolio_daily["date"].min()
        portfolio_daily.loc[portfolio_daily["date"].eq(first_date), "portfolio_return"] -= turnover * transaction_cost
        daily_rows.extend(portfolio_daily.to_dict("records"))
        previous_weights = weights

    daily_returns = pd.DataFrame(daily_rows)
    if daily_returns.empty:
        daily_returns = pd.DataFrame(columns=["date", "portfolio_return"])
    daily_returns = daily_returns.merge(benchmark_returns, on="date", how="left")
    daily_returns["benchmark_return"] = daily_returns["benchmark_return"].fillna(0.0)
    daily_returns = daily_returns.sort_values("date").reset_index(drop=True)

    turnover = pd.DataFrame(turnover_rows)
    cumulative = _build_cumulative_returns(daily_returns)
    drawdown = _build_drawdown(cumulative)
    rolling_volatility = _build_rolling_volatility(daily_returns, config)
    sector_allocation = _build_sector_allocation(portfolio)
    summary = summarize_performance(daily_returns, turnover, risk_free_rate)
    return {
        "daily_returns": daily_returns,
        "performance_summary": summary,
        "turnover": turnover,
        "sector_allocation": sector_allocation,
        "cumulative_returns": cumulative,
        "drawdown": drawdown,
        "rolling_volatility": rolling_volatility,
    }


def _build_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_prices(prices)
    cleaned["daily_return"] = cleaned.groupby("stock_code")["close"].pct_change()
    return cleaned.dropna(subset=["daily_return"])[["date", "stock_code", "daily_return"]]


def _build_benchmark_returns(benchmark: pd.DataFrame) -> pd.DataFrame:
    result = benchmark.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date")
    result["benchmark_return"] = result["close"].pct_change()
    return result.dropna(subset=["benchmark_return"])[["date", "benchmark_return"]]


def _calculate_turnover(previous_weights: pd.Series, current_weights: pd.Series) -> float:
    all_codes = previous_weights.index.union(current_weights.index)
    previous = previous_weights.reindex(all_codes, fill_value=0.0)
    current = current_weights.reindex(all_codes, fill_value=0.0)
    return float((current - previous).abs().sum())


def _build_cumulative_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    result = daily_returns[["date", "portfolio_return", "benchmark_return"]].copy()
    result["portfolio_cumulative"] = (1 + result["portfolio_return"].fillna(0)).cumprod()
    result["benchmark_cumulative"] = (1 + result["benchmark_return"].fillna(0)).cumprod()
    return result[["date", "portfolio_cumulative", "benchmark_cumulative"]]


def _build_drawdown(cumulative: pd.DataFrame) -> pd.DataFrame:
    result = cumulative[["date", "portfolio_cumulative", "benchmark_cumulative"]].copy()
    result["portfolio_drawdown"] = result["portfolio_cumulative"] / result["portfolio_cumulative"].cummax() - 1
    result["benchmark_drawdown"] = result["benchmark_cumulative"] / result["benchmark_cumulative"].cummax() - 1
    return result[["date", "portfolio_drawdown", "benchmark_drawdown"]]


def _build_rolling_volatility(daily_returns: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    window = int(config.get("backtest", {}).get("rolling_volatility_window_days", 63))
    result = daily_returns[["date", "portfolio_return", "benchmark_return"]].copy()
    result["portfolio_rolling_volatility"] = result["portfolio_return"].rolling(window).std() * (252**0.5)
    result["benchmark_rolling_volatility"] = result["benchmark_return"].rolling(window).std() * (252**0.5)
    return result[["date", "portfolio_rolling_volatility", "benchmark_rolling_volatility"]]


def _build_sector_allocation(portfolio: pd.DataFrame) -> pd.DataFrame:
    return (
        portfolio.groupby(["as_of_date", "industry"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "industry_weight"})
    )
