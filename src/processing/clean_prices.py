"""Price cleaning and trailing market-confirmation features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation_utils import coerce_dates, require_columns


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort daily price observations."""

    require_columns(prices, ["date", "stock_code", "close", "turnover"], "prices")
    result = coerce_dates(prices, ["date"]).copy()
    result = result.sort_values(["stock_code", "date"])
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["turnover"] = pd.to_numeric(result["turnover"], errors="coerce")
    return result.dropna(subset=["date", "stock_code", "close"])


def get_rebalance_dates(prices: pd.DataFrame, frequency: str = "Q") -> list[pd.Timestamp]:
    """Use the last available trading date in each rebalance period."""

    cleaned = clean_prices(prices)
    unique_dates = pd.Series(sorted(cleaned["date"].unique()))
    periods = unique_dates.dt.to_period(frequency)
    dates = unique_dates.groupby(periods).max().tolist()
    return [pd.Timestamp(date) for date in dates]


def build_market_features(prices: pd.DataFrame, rebalance_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Calculate trailing momentum, volatility, and liquidity features.

    Momentum and volatility are backward-looking. The 12-month momentum measure
    excludes roughly the most recent month by comparing t-21 trading days to
    t-252 trading days.
    """

    cleaned = clean_prices(prices)
    frames: list[pd.DataFrame] = []
    for stock_code, group in cleaned.groupby("stock_code", sort=False):
        stock = group.sort_values("date").copy()
        stock["daily_return"] = stock["close"].pct_change()
        stock["momentum_6m"] = stock["close"].div(stock["close"].shift(126)).sub(1)
        stock["momentum_12m_ex_1m"] = stock["close"].shift(21).div(stock["close"].shift(252)).sub(1)
        stock["volatility"] = stock["daily_return"].rolling(126, min_periods=60).std() * np.sqrt(252)
        stock["avg_turnover"] = stock["turnover"].rolling(63, min_periods=20).mean()
        stock["stock_code"] = stock_code
        frames.append(stock)

    features = pd.concat(frames, ignore_index=True)
    feature_dates = pd.DataFrame({"as_of_date": pd.to_datetime(rebalance_dates)})
    latest_rows: list[pd.DataFrame] = []
    for as_of_date in feature_dates["as_of_date"]:
        trailing = features[features["date"] <= as_of_date]
        latest = trailing.sort_values("date").groupby("stock_code", as_index=False).tail(1)
        latest = latest.assign(as_of_date=as_of_date)
        latest_rows.append(
            latest[
                [
                    "as_of_date",
                    "stock_code",
                    "date",
                    "close",
                    "momentum_6m",
                    "momentum_12m_ex_1m",
                    "volatility",
                    "avg_turnover",
                ]
            ].rename(columns={"date": "price_observation_date"})
        )
    return pd.concat(latest_rows, ignore_index=True)
