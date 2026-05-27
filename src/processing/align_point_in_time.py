"""Point-in-time data alignment for stock selection panels."""

from __future__ import annotations

import pandas as pd

from src.processing.clean_fundamentals import build_fundamental_features, latest_fundamentals_as_of
from src.processing.clean_prices import build_market_features
from src.utils.validation_utils import coerce_dates, require_columns


def build_rebalance_panel(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    reporting_lag_days: int,
) -> pd.DataFrame:
    """Merge universe, trailing fundamentals, and trailing market features."""

    require_columns(universe, ["stock_code", "stock_name", "industry", "is_st"], "universe")
    universe_clean = coerce_dates(universe, ["listing_date"])
    fundamental_features = build_fundamental_features(fundamentals, reporting_lag_days)
    point_in_time_fundamentals = latest_fundamentals_as_of(fundamental_features, rebalance_dates)
    market_features = build_market_features(prices, rebalance_dates)

    panel = market_features.merge(universe_clean, on="stock_code", how="left")
    panel = panel.merge(
        point_in_time_fundamentals,
        on=["as_of_date", "stock_code"],
        how="left",
        suffixes=("", "_fundamental"),
    )
    panel["data_is_point_in_time"] = panel["announcement_date"].le(panel["as_of_date"])
    panel.loc[panel["announcement_date"].isna(), "data_is_point_in_time"] = False
    return panel.sort_values(["as_of_date", "stock_code"]).reset_index(drop=True)
