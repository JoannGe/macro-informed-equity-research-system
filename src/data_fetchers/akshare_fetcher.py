"""AKShare-based A-share data fetcher.

AKShare is a free public-data wrapper. Its endpoint names and returned column
labels can change, so every function raises clear errors and the router decides
whether to try another source or demo fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.processing.standardize_schema import normalize_ticker, ticker_to_akshare_symbol
from src.utils.date_utils import compact_date


def fetch_akshare_data(config: dict[str, Any], start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Fetch a compact A-share research dataset from AKShare."""

    try:
        import akshare as ak  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(f"AKShare is not available: {exc}") from exc

    limit = int(config.get("data", {}).get("live_universe_limit", 80))
    benchmark_code = str(config.get("project", {}).get("benchmark_code", "399673"))
    spot = ak.stock_zh_a_spot_em()
    universe = _standardize_spot(spot).head(limit)
    tickers = universe["ticker"].tolist()
    prices = _fetch_prices(ak, tickers, start_date, end_date)
    benchmark = _fetch_benchmark(ak, benchmark_code, start_date, end_date)
    fundamentals = _fundamentals_from_spot(universe, prices)
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


def _standardize_spot(spot: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "代码": "code",
        "名称": "stock_name",
        "总市值": "market_cap",
        "市盈率-动态": "pe",
        "市净率": "pb",
        "成交额": "amount",
        "成交量": "volume",
        "最新价": "close",
        "所属行业": "industry",
    }
    result = spot.rename(columns={key: value for key, value in column_map.items() if key in spot.columns}).copy()
    if "code" not in result.columns:
        raise RuntimeError("AKShare spot data did not include a stock code column.")
    result["ticker"] = result["code"].map(normalize_ticker)
    if "industry" not in result.columns:
        result["industry"] = "Unknown"
    if "stock_name" not in result.columns:
        result["stock_name"] = result["ticker"]
    for column in ["pe", "pb", "market_cap", "amount", "volume", "close"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["ps"] = pd.NA
    return result.sort_values("amount", ascending=False, na_position="last")


def _fetch_prices(ak: object, tickers: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        symbol = ticker_to_akshare_symbol(ticker)
        try:
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=compact_date(start_date),
                end_date=compact_date(end_date),
                adjust="qfq",
            )
        except Exception:
            continue
        if raw.empty:
            continue
        frame = raw.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
        frame["ticker"] = ticker
        rows.append(frame[["date", "ticker", "open", "high", "low", "close", "volume", "amount"]])
    if not rows:
        raise RuntimeError("AKShare returned no usable historical stock prices.")
    prices = pd.concat(rows, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    return prices


def _fetch_benchmark(
    ak: object,
    benchmark_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    attempts = [
        lambda: ak.index_hist_cni(symbol=benchmark_code, start_date=compact_date(start_date), end_date=compact_date(end_date)),
        lambda: ak.stock_zh_index_daily_em(symbol=f"sz{benchmark_code}"),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            raw = attempt()
            if raw is None or raw.empty:
                continue
            frame = raw.rename(columns={"日期": "date", "收盘": "close"})
            if "date" not in frame.columns:
                frame = frame.rename(columns={"date": "date", "close": "close"})
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()
            frame["benchmark_code"] = benchmark_code
            frame["return"] = pd.to_numeric(frame["close"], errors="coerce").pct_change()
            return frame[["date", "benchmark_code", "close", "return"]]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"AKShare benchmark fetch failed: {last_error}")


def _fundamentals_from_spot(universe: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    latest_date = prices["date"].max()
    result = universe[["ticker", "pe", "pb", "ps", "market_cap"]].copy()
    result["date"] = latest_date
    result["announcement_date"] = latest_date
    for column in [
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
