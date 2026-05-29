"""Simple, interpretable macro regime classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.validation_utils import coerce_dates, require_columns


GROWTH_COLUMNS = ["gdp_growth", "pmi", "industrial_production"]
INFLATION_COLUMNS = ["cpi", "ppi"]
RATE_COLUMNS = ["policy_rate", "bond_yield_10y"]
CREDIT_COLUMNS = ["credit_growth"]
LIQUIDITY_COLUMNS = ["m2_growth", "turnover_proxy", "liquidity_proxy"]


def classify_macro_regime(
    macro_data: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, object]:
    """Classify the macro state using trailing rolling z-scores.

    A positive composite z-score means the variable group is above its own
    recent history. This is intentionally simple so the regime label can be
    audited by reading the latest indicator values.
    """

    require_columns(macro_data, ["date"], "macro_data")
    macro = coerce_dates(macro_data, ["date"]).sort_values("date")
    trailing = macro[macro["date"] <= pd.Timestamp(as_of_date)].copy()
    if trailing.empty:
        raise ValueError(f"No macro data is available on or before {as_of_date}.")

    window = int(config.get("macro", {}).get("rolling_window_months", 12))
    min_periods = int(config.get("macro", {}).get("minimum_periods", 6))

    growth = _composite_zscore(trailing, GROWTH_COLUMNS, window, min_periods)
    inflation = _composite_zscore(trailing, INFLATION_COLUMNS, window, min_periods)
    rates = _composite_zscore(trailing, RATE_COLUMNS, window, min_periods)
    credit = _composite_zscore(trailing, CREDIT_COLUMNS, window, min_periods)
    liquidity = _composite_zscore(trailing, LIQUIDITY_COLUMNS, window, min_periods)
    commodity = _composite_zscore(trailing, ["commodity_proxy"], window, min_periods)
    policy = _composite_zscore(trailing, ["policy_support_proxy"], window, min_periods)
    risk_appetite = _composite_zscore(trailing, ["turnover_proxy"], window, min_periods)

    latest = trailing.iloc[-1]
    vector = {
        "growth": _direction(growth),
        "inflation": _direction(inflation),
        "interest_rates": _direction(rates),
        "credit": _direction(credit),
        "liquidity": _direction(liquidity),
        "commodity_prices": _direction(commodity),
        "policy_sensitivity": _direction(policy),
        "risk_appetite": _direction(risk_appetite),
    }
    return {
        "as_of_date": pd.Timestamp(as_of_date),
        "macro_observation_date": latest["date"],
        "growth_score": growth,
        "inflation_score": inflation,
        "rates_score": rates,
        "credit_score": credit,
        "liquidity_score": liquidity,
        "commodity_score": commodity,
        "policy_score": policy,
        "risk_appetite_score": risk_appetite,
        "growth_regime": "growth_up" if growth >= 0 else "growth_down",
        "inflation_regime": "inflation_up" if inflation >= 0 else "inflation_down",
        "rates_regime": "rates_tightening" if rates >= 0 else "rates_easing",
        "credit_regime": "credit_expanding" if credit >= 0 else "credit_contracting",
        "liquidity_regime": "liquidity_loose" if liquidity >= 0 else "liquidity_tight",
        **{f"vector_{key}": value for key, value in vector.items()},
    }


def build_macro_regime_history(
    macro_data: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Classify macro regimes for every rebalance date."""

    rows = [classify_macro_regime(macro_data, date, config) for date in rebalance_dates]
    return pd.DataFrame(rows)


def _composite_zscore(
    data: pd.DataFrame,
    columns: list[str],
    window: int,
    min_periods: int,
) -> float:
    available = [column for column in columns if column in data.columns]
    if not available:
        return 0.0
    zscores: list[float] = []
    for column in available:
        series = pd.to_numeric(data[column], errors="coerce")
        rolling_mean = series.rolling(window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window, min_periods=min_periods).std(ddof=0)
        latest_std = rolling_std.iloc[-1]
        if pd.isna(latest_std) or latest_std == 0:
            zscores.append(0.0)
        else:
            zscores.append(float((series.iloc[-1] - rolling_mean.iloc[-1]) / latest_std))
    return float(np.nanmean(zscores)) if zscores else 0.0


def _direction(value: float) -> int:
    """Convert a continuous regime score into +1/-1 for industry scoring."""

    return 1 if value >= 0 else -1
