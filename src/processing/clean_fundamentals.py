"""Fundamental cleaning and trailing ratio calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation_utils import coerce_dates, require_columns


def clean_fundamentals(fundamentals: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    """Validate fundamentals and create announcement dates when missing."""

    require_columns(fundamentals, ["stock_code", "report_date"], "fundamentals")
    result = coerce_dates(fundamentals, ["report_date", "announcement_date"]).copy()
    if "announcement_date" not in result.columns:
        result["announcement_date"] = result["report_date"] + pd.to_timedelta(reporting_lag_days, unit="D")
    result = result.sort_values(["stock_code", "report_date"])
    return result


def build_fundamental_features(fundamentals: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    """Calculate interpretable ratios and year-over-year growth fields."""

    result = clean_fundamentals(fundamentals, reporting_lag_days)
    numeric_columns = [
        "revenue",
        "net_profit",
        "operating_cash_flow",
        "total_assets",
        "total_equity",
        "total_debt",
        "gross_profit",
        "operating_profit",
        "interest_expense",
        "current_assets",
        "current_liabilities",
        "capex",
        "pe",
        "pb",
        "ps",
        "dividend_yield",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result["roe"] = _safe_divide(result.get("net_profit"), result.get("total_equity"))
    result["roa"] = _safe_divide(result.get("net_profit"), result.get("total_assets"))
    result["gross_margin"] = _safe_divide(result.get("gross_profit"), result.get("revenue"))
    result["operating_margin"] = _safe_divide(result.get("operating_profit"), result.get("revenue"))
    result["debt_to_assets"] = _safe_divide(result.get("total_debt"), result.get("total_assets"))
    result["debt_to_equity"] = _safe_divide(result.get("total_debt"), result.get("total_equity"))
    result["interest_coverage"] = _safe_divide(result.get("operating_profit"), result.get("interest_expense"))
    result["current_ratio"] = _safe_divide(result.get("current_assets"), result.get("current_liabilities"))
    result["ocf_to_revenue"] = _safe_divide(result.get("operating_cash_flow"), result.get("revenue"))
    result["free_cash_flow_proxy"] = result.get("operating_cash_flow", np.nan) - result.get("capex", 0)
    result["fcf_to_revenue"] = _safe_divide(result.get("free_cash_flow_proxy"), result.get("revenue"))
    result["cash_conversion_ratio"] = _safe_divide(result.get("operating_cash_flow"), result.get("net_profit"))

    for column, new_column in [
        ("revenue", "revenue_growth"),
        ("net_profit", "net_profit_growth"),
        ("operating_cash_flow", "operating_cash_flow_growth"),
    ]:
        if column in result.columns:
            result[new_column] = result.groupby("stock_code")[column].pct_change(4)
        else:
            result[new_column] = np.nan
    return result


def latest_fundamentals_as_of(
    fundamental_features: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Select the latest announced fundamentals available at each date."""

    require_columns(
        fundamental_features,
        ["stock_code", "report_date", "announcement_date"],
        "fundamental_features",
    )
    rows: list[pd.DataFrame] = []
    for as_of_date in pd.to_datetime(rebalance_dates):
        eligible = fundamental_features[fundamental_features["announcement_date"] <= as_of_date]
        if eligible.empty:
            continue
        latest = eligible.sort_values(["stock_code", "announcement_date", "report_date"]).groupby(
            "stock_code",
            as_index=False,
        ).tail(1)
        latest = latest.assign(as_of_date=as_of_date)
        rows.append(latest)
    if not rows:
        return pd.DataFrame(columns=list(fundamental_features.columns) + ["as_of_date"])
    return pd.concat(rows, ignore_index=True)


def _safe_divide(numerator: object, denominator: object) -> pd.Series:
    """Divide two series while converting zero denominators to missing values."""

    numerator_series = pd.Series(numerator)
    denominator_series = pd.Series(denominator).replace(0, np.nan)
    return numerator_series.div(denominator_series)
