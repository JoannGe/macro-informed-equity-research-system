"""BaoStock fallback fetcher for standardized A-share data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.processing.standardize_schema import normalize_ticker, ticker_to_baostock_code
from src.utils.date_utils import compact_date


def fetch_baostock_data(config: dict[str, Any], start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Fetch a compact A-share dataset from BaoStock."""

    try:
        import baostock as bs  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(f"BaoStock is not available: {exc}") from exc

    login_result = bs.login()
    if getattr(login_result, "error_code", "0") != "0":
        raise RuntimeError(f"BaoStock login failed: {getattr(login_result, 'error_msg', '')}")
    try:
        limit = int(config.get("data", {}).get("live_universe_limit", 80))
        benchmark_code = str(config.get("project", {}).get("benchmark_code", "399673"))
        universe = _query_all_stocks(bs, end_date).head(limit)
        prices = _fetch_prices(bs, universe["ticker"].tolist(), start_date, end_date)
        benchmark = _fetch_benchmark(bs, benchmark_code, start_date, end_date)
        fundamentals = _latest_fundamentals_from_prices(prices)
        industry = universe[["ticker", "industry"]].copy()
        names = universe[["ticker", "stock_name"]].copy()
        return {
            "prices": prices,
            "benchmark": benchmark,
            "fundamentals": fundamentals,
            "industry": industry,
            "names": names,
            "universe": universe,
        }
    finally:  # pragma: no cover - external session cleanup
        bs.logout()


def _query_all_stocks(bs: object, end_date: pd.Timestamp) -> pd.DataFrame:
    response = bs.query_all_stock(day=end_date.strftime("%Y-%m-%d"))
    rows = []
    while response.error_code == "0" and response.next():
        rows.append(response.get_row_data())
    result = pd.DataFrame(rows, columns=response.fields)
    if result.empty:
        raise RuntimeError("BaoStock returned an empty stock universe.")
    result["ticker"] = result["code"].map(normalize_ticker)
    result["stock_name"] = result.get("code_name", result["ticker"])
    result["industry"] = "Unknown"
    return result[["ticker", "stock_name", "industry"]]


def _fetch_prices(bs: object, tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    fields = "date,code,open,high,low,close,volume,amount,peTTM,pbMRQ,psTTM,isST"
    rows = []
    for ticker in tickers:
        code = ticker_to_baostock_code(ticker)
        response = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",
        )
        stock_rows = []
        while response.error_code == "0" and response.next():
            stock_rows.append(response.get_row_data())
        if not stock_rows:
            continue
        frame = pd.DataFrame(stock_rows, columns=response.fields)
        frame["ticker"] = ticker
        rows.append(frame)
    if not rows:
        raise RuntimeError("BaoStock returned no usable historical prices.")
    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"peTTM": "pe", "pbMRQ": "pb", "psTTM": "ps"})
    result["date"] = pd.to_datetime(result["date"])
    for column in ["open", "high", "low", "close", "volume", "amount", "pe", "pb", "ps"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values(["ticker", "date"])
    result["return"] = result.groupby("ticker")["close"].pct_change()
    return result[["date", "ticker", "open", "high", "low", "close", "volume", "amount", "return", "pe", "pb", "ps"]]


def _fetch_benchmark(bs: object, benchmark_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    code = f"sz.{benchmark_code}"
    response = bs.query_history_k_data_plus(
        code,
        "date,code,close",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        frequency="d",
    )
    rows = []
    while response.error_code == "0" and response.next():
        rows.append(response.get_row_data())
    if not rows:
        raise RuntimeError("BaoStock returned no usable benchmark data.")
    result = pd.DataFrame(rows, columns=response.fields)
    result["date"] = pd.to_datetime(result["date"])
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["benchmark_code"] = benchmark_code
    result["return"] = result["close"].pct_change()
    return result[["date", "benchmark_code", "close", "return"]]


def _latest_fundamentals_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    latest = prices.sort_values("date").groupby("ticker", as_index=False).tail(1)
    result = latest[["date", "ticker", "pe", "pb", "ps"]].copy()
    result["announcement_date"] = result["date"]
    for column in [
        "market_cap",
        "roe",
        "roa",
        "gross_margin",
        "operating_margin",
        "revenue_growth",
        "profit_growth",
        "debt_to_assets",
        "debt_to_equity",
    ]:
        result[column] = pd.NA
    return result
