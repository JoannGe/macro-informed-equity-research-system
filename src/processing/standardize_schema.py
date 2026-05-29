"""Normalize external and demo data into the V2 internal schema."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "amount", "return"]
FUNDAMENTAL_COLUMNS = [
    "date",
    "ticker",
    "pe",
    "pb",
    "ps",
    "market_cap",
    "roe",
    "roa",
    "gross_margin",
    "operating_margin",
    "revenue_growth",
    "profit_growth",
    "debt_to_assets",
    "debt_to_equity",
]
INDUSTRY_COLUMNS = ["ticker", "industry"]
BENCHMARK_COLUMNS = ["date", "benchmark_code", "close", "return"]


def normalize_ticker(value: object) -> str:
    """Normalize A-share tickers while preserving exchange suffixes when present."""

    text = str(value).strip()
    if "." in text:
        code, suffix = text.split(".", 1)
        if suffix.upper() in {"SH", "SZ"}:
            return f"{code.zfill(6)}.{suffix.upper()}"
        if suffix.lower() in {"ss", "sh"}:
            return f"{code.zfill(6)}.SH"
        if suffix.lower() in {"sz"}:
            return f"{code.zfill(6)}.SZ"
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 6:
        code = digits[-6:]
        suffix = "SH" if code.startswith(("6", "9")) else "SZ"
        return f"{code}.{suffix}"
    return text


def ticker_to_baostock_code(ticker: str) -> str:
    """Convert internal ticker format to BaoStock's exchange-prefix format."""

    code, suffix = normalize_ticker(ticker).split(".")
    exchange = "sh" if suffix == "SH" else "sz"
    return f"{exchange}.{code}"


def ticker_to_akshare_symbol(ticker: str) -> str:
    """Convert internal ticker format to the six-digit symbol used by many AKShare endpoints."""

    return normalize_ticker(ticker).split(".")[0]


def standardize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return prices with V2 schema columns and compatibility aliases."""

    result = prices.copy()
    result = _rename_first_available(result, ["stock_code", "code", "symbol"], "ticker")
    result = _rename_first_available(result, ["turnover", "成交额"], "amount")
    result = _rename_first_available(result, ["成交量"], "volume")
    result = _rename_first_available(result, ["收盘", "最新价"], "close")
    result = _rename_first_available(result, ["开盘"], "open")
    result = _rename_first_available(result, ["最高"], "high")
    result = _rename_first_available(result, ["最低"], "low")
    result["date"] = pd.to_datetime(result["date"])
    result["ticker"] = result["ticker"].map(normalize_ticker)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["open"] = result["open"].fillna(result["close"])
    result["high"] = result["high"].fillna(result[["open", "close"]].max(axis=1))
    result["low"] = result["low"].fillna(result[["open", "close"]].min(axis=1))
    result["amount"] = result["amount"].fillna(result["volume"].fillna(0) * result["close"].fillna(0))
    result = result.sort_values(["ticker", "date"])
    if "return" not in result.columns:
        result["return"] = result.groupby("ticker")["close"].pct_change()
    result["return"] = pd.to_numeric(result["return"], errors="coerce")
    result["stock_code"] = result["ticker"]
    result["turnover"] = result["amount"]
    return result[[*PRICE_COLUMNS, "stock_code", "turnover"]]


def standardize_fundamentals(fundamentals: pd.DataFrame, reporting_lag_days: int) -> pd.DataFrame:
    """Return fundamentals with V2 schema columns plus point-in-time aliases."""

    result = fundamentals.copy()
    result = _rename_first_available(result, ["stock_code", "code", "symbol"], "ticker")
    result = _rename_first_available(result, ["report_date", "pubDate", "statDate"], "date")
    result = _rename_first_available(result, ["net_profit_growth"], "profit_growth")
    result["date"] = pd.to_datetime(result["date"])
    result["ticker"] = result["ticker"].map(normalize_ticker)
    for column in FUNDAMENTAL_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    numeric_columns = [column for column in FUNDAMENTAL_COLUMNS if column not in {"date", "ticker"}]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if "roe" in result.columns and result["roe"].isna().all() and {"net_profit", "total_equity"}.issubset(result.columns):
        result["roe"] = _safe_ratio(result["net_profit"], result["total_equity"])
    if "roa" in result.columns and result["roa"].isna().all() and {"net_profit", "total_assets"}.issubset(result.columns):
        result["roa"] = _safe_ratio(result["net_profit"], result["total_assets"])
    if "gross_margin" in result.columns and result["gross_margin"].isna().all() and {"gross_profit", "revenue"}.issubset(result.columns):
        result["gross_margin"] = _safe_ratio(result["gross_profit"], result["revenue"])
    if (
        "operating_margin" in result.columns
        and result["operating_margin"].isna().all()
        and {"operating_profit", "revenue"}.issubset(result.columns)
    ):
        result["operating_margin"] = _safe_ratio(result["operating_profit"], result["revenue"])
    if "debt_to_assets" in result.columns and result["debt_to_assets"].isna().all():
        if {"total_debt", "total_assets"}.issubset(result.columns):
            result["debt_to_assets"] = pd.to_numeric(result["total_debt"], errors="coerce") / pd.to_numeric(
                result["total_assets"],
                errors="coerce",
            ).replace(0, np.nan)
    if "debt_to_equity" in result.columns and result["debt_to_equity"].isna().all():
        if {"total_debt", "total_equity"}.issubset(result.columns):
            result["debt_to_equity"] = pd.to_numeric(result["total_debt"], errors="coerce") / pd.to_numeric(
                result["total_equity"],
                errors="coerce",
            ).replace(0, np.nan)
    if "announcement_date" not in result.columns:
        result["announcement_date"] = result["date"] + pd.to_timedelta(reporting_lag_days, unit="D")
    result["announcement_date"] = pd.to_datetime(result["announcement_date"])
    result["report_date"] = result["date"]
    result["stock_code"] = result["ticker"]
    extra_columns = [
        column
        for column in [
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
            "asset_growth",
            "capex_growth",
            "dividend_yield",
        ]
        if column in result.columns
    ]
    return result[[*FUNDAMENTAL_COLUMNS, *extra_columns, "report_date", "announcement_date", "stock_code"]]


def standardize_industry(industry: pd.DataFrame) -> pd.DataFrame:
    """Return ticker-to-industry mapping with V2 schema."""

    result = industry.copy()
    result = _rename_first_available(result, ["stock_code", "code", "symbol"], "ticker")
    if "industry" not in result.columns:
        result["industry"] = "Unknown"
    result["ticker"] = result["ticker"].map(normalize_ticker)
    return result[INDUSTRY_COLUMNS].drop_duplicates("ticker")


def standardize_benchmark(benchmark: pd.DataFrame, benchmark_code: str) -> pd.DataFrame:
    """Return benchmark data with V2 schema."""

    result = benchmark.copy()
    result = _rename_first_available(result, ["收盘"], "close")
    result["date"] = pd.to_datetime(result["date"])
    if "benchmark_code" not in result.columns:
        result["benchmark_code"] = benchmark_code
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.sort_values("date")
    if "return" not in result.columns:
        result["return"] = result["close"].pct_change()
    return result[BENCHMARK_COLUMNS]


def build_universe_from_schema(industry: pd.DataFrame, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create the legacy universe frame expected by point-in-time alignment."""

    result = standardize_industry(industry)
    result["stock_code"] = result["ticker"]
    result["stock_name"] = result["ticker"]
    if names is not None and {"ticker", "stock_name"}.issubset(names.columns):
        name_map = names.assign(ticker=names["ticker"].map(normalize_ticker)).set_index("ticker")["stock_name"]
        result["stock_name"] = result["ticker"].map(name_map).fillna(result["stock_name"])
    result["is_st"] = False
    result["listing_date"] = pd.NaT
    return result[["stock_code", "stock_name", "industry", "is_st", "listing_date", "ticker"]]


def _rename_first_available(df: pd.DataFrame, candidates: list[str], target: str) -> pd.DataFrame:
    result = df.copy()
    if target in result.columns:
        return result
    for candidate in candidates:
        if candidate in result.columns:
            return result.rename(columns={candidate: target})
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
