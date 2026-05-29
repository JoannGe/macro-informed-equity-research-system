"""Performance metrics for paper-research backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound daily returns into an annualized return."""

    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    cumulative = (1 + clean).prod()
    years = len(clean) / periods_per_year
    if years <= 0:
        return 0.0
    return float(cumulative ** (1 / years) - 1)


def annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize the standard deviation of daily returns."""

    return float(daily_returns.dropna().std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate_annual: float = 0.02) -> float:
    """Calculate a simple annualized Sharpe ratio."""

    excess_daily = daily_returns.dropna() - risk_free_rate_annual / 252
    volatility = excess_daily.std(ddof=0)
    if volatility == 0 or pd.isna(volatility):
        return 0.0
    return float(excess_daily.mean() / volatility * np.sqrt(252))


def max_drawdown(cumulative_return: pd.Series) -> float:
    """Return the worst peak-to-trough loss."""

    if cumulative_return.empty:
        return 0.0
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return / running_max - 1
    return float(drawdown.min())


def win_rate(daily_returns: pd.Series) -> float:
    """Share of positive daily return observations."""

    clean = daily_returns.dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def summarize_performance(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    risk_free_rate_annual: float = 0.02,
) -> pd.DataFrame:
    """Build a one-row performance summary."""

    portfolio_returns = returns["portfolio_return"].fillna(0)
    benchmark_returns = returns["benchmark_return"].fillna(0)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    total_turnover = float(turnover["turnover"].sum()) if not turnover.empty else 0.0
    relative_return = float(portfolio_cumulative.iloc[-1] / benchmark_cumulative.iloc[-1] - 1)
    return pd.DataFrame(
        [
            {
                "annualized_return": annualized_return(portfolio_returns),
                "annualized_volatility": annualized_volatility(portfolio_returns),
                "sharpe_ratio": sharpe_ratio(portfolio_returns, risk_free_rate_annual),
                "maximum_drawdown": max_drawdown(portfolio_cumulative),
                "turnover": total_turnover,
                "win_rate": win_rate(portfolio_returns),
                "benchmark_relative_return": relative_return,
            }
        ]
    )
