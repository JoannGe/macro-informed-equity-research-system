"""Generate deterministic demo data for offline testing and learning.

The generated data is intentionally small and synthetic. It is useful for
checking that the full pipeline runs, but it should not be interpreted as real
market evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.date_utils import requested_date_range
from src.utils.io_utils import demo_path, read_dataframe, write_dataframe


DEMO_FILES = {
    "macro": "macro.csv",
    "prices": "prices.csv",
    "benchmark": "benchmark.csv",
    "fundamentals": "fundamentals.csv",
    "universe": "universe.csv",
}


@dataclass(frozen=True)
class DemoPaths:
    """Absolute paths to the generated demo files."""

    macro: Path
    prices: Path
    benchmark: Path
    fundamentals: Path
    universe: Path


def get_demo_paths(config: dict[str, Any]) -> DemoPaths:
    """Return path objects for all demo files."""

    return DemoPaths(**{key: demo_path(filename, config) for key, filename in DEMO_FILES.items()})


def load_demo_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load generated demo data from disk."""

    paths = get_demo_paths(config)
    return {
        "macro": read_dataframe(paths.macro, parse_dates=["date"]),
        "prices": read_dataframe(paths.prices, parse_dates=["date"]),
        "benchmark": read_dataframe(paths.benchmark, parse_dates=["date"]),
        "fundamentals": read_dataframe(
            paths.fundamentals,
            parse_dates=["report_date", "announcement_date"],
        ),
        "universe": read_dataframe(paths.universe, parse_dates=["listing_date"]),
    }


def ensure_demo_data(config: dict[str, Any], force: bool = False) -> dict[str, pd.DataFrame]:
    """Create demo data when missing, then return it as dataframes."""

    paths = get_demo_paths(config)
    if force or not all(getattr(paths, name).exists() for name in DEMO_FILES):
        data = build_demo_data(config)
        for name, frame in data.items():
            write_dataframe(frame, getattr(paths, name))
    return load_demo_data(config)


def build_demo_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Build a reproducible synthetic dataset with realistic column names."""

    seed = int(config.get("data", {}).get("demo_seed", 42))
    rng = np.random.default_rng(seed)
    start_date, end_date, _ = requested_date_range(config, mode="demo")
    reporting_lag_days = int(config.get("data", {}).get("reporting_lag_days", 75))

    industries = [
        "Technology",
        "Healthcare",
        "Consumer",
        "Financials",
        "Industrials",
        "Materials",
        "Energy",
        "Utilities",
        "Real Estate",
        "Communication",
    ]
    universe = _build_universe(industries)
    macro = _build_macro_data(start_date, end_date, rng)
    fundamentals = _build_fundamentals(universe, start_date, end_date, reporting_lag_days, rng)
    prices, benchmark = _build_prices(universe, start_date, end_date, rng)
    return {
        "macro": macro,
        "prices": prices,
        "benchmark": benchmark,
        "fundamentals": fundamentals,
        "universe": universe,
    }


def _build_universe(industries: list[str]) -> pd.DataFrame:
    rows = []
    for idx in range(30):
        industry = industries[idx % len(industries)]
        exchange = "SZ" if idx % 2 == 0 else "SH"
        code = f"{300000 + idx:06d}.{exchange}"
        rows.append(
            {
                "stock_code": code,
                "ticker": code,
                "stock_name": f"Demo {industry[:4]} {idx + 1:02d}",
                "industry": industry,
                "is_st": idx in {7, 23},
                "listing_date": pd.Timestamp("2016-01-01") + pd.DateOffset(days=idx * 20),
            }
        )
    return pd.DataFrame(rows)


def _build_macro_data(start_date: pd.Timestamp, end_date: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="ME")
    t = np.arange(len(dates))
    cycle = np.sin(t / 7.0)
    credit_cycle = np.cos(t / 8.5)
    noise = lambda scale: rng.normal(0, scale, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": 5.0 + 0.45 * cycle + noise(0.10),
            "pmi": 50.0 + 1.2 * cycle + noise(0.35),
            "industrial_production": 4.8 + 0.75 * cycle + noise(0.20),
            "cpi": 1.8 + 0.35 * np.sin(t / 5.0 + 0.7) + noise(0.08),
            "ppi": 0.8 + 0.8 * np.sin(t / 5.5 + 0.4) + noise(0.15),
            "policy_rate": 2.6 + 0.12 * np.cos(t / 9.0) + noise(0.03),
            "bond_yield_10y": 2.9 + 0.25 * np.cos(t / 10.0) + noise(0.05),
            "credit_growth": 10.0 + 1.1 * credit_cycle + noise(0.25),
            "m2_growth": 8.5 + 1.0 * credit_cycle + noise(0.18),
            "turnover_proxy": 100.0 + 14.0 * np.sin(t / 6.0 + 1.3) + noise(2.5),
            "liquidity_proxy": 0.0 + 0.8 * credit_cycle + noise(0.15),
            "commodity_proxy": 100.0 + 10.0 * np.sin(t / 5.2 + 1.0) + noise(2.0),
            "policy_support_proxy": 0.2 + 0.6 * np.sin(t / 12.0 + 0.3) + noise(0.12),
        }
    )


def _build_fundamentals(
    universe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    reporting_lag_days: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    report_dates = pd.date_range(start_date, end_date, freq="QE")
    rows: list[dict[str, object]] = []
    industry_quality = {
        "Technology": 0.04,
        "Healthcare": 0.05,
        "Consumer": 0.03,
        "Financials": 0.02,
        "Industrials": 0.01,
        "Materials": -0.01,
        "Energy": -0.01,
        "Utilities": 0.0,
        "Real Estate": -0.03,
        "Communication": 0.02,
    }
    for _, company in universe.iterrows():
        base_revenue = rng.uniform(800, 2400)
        assets = base_revenue * rng.uniform(1.1, 2.0)
        equity_ratio = rng.uniform(0.35, 0.72)
        quality_shift = industry_quality.get(str(company["industry"]), 0.0)
        for quarter_index, report_date in enumerate(report_dates):
            seasonality = 1 + 0.03 * np.sin(quarter_index / 2)
            growth = rng.normal(0.018 + quality_shift, 0.035)
            revenue = base_revenue * ((1 + growth) ** quarter_index) * seasonality
            gross_margin = np.clip(rng.normal(0.30 + quality_shift, 0.06), 0.08, 0.68)
            operating_margin = np.clip(gross_margin - rng.uniform(0.08, 0.18), -0.05, 0.42)
            net_margin = operating_margin - rng.uniform(0.02, 0.08)
            net_profit = revenue * net_margin
            total_assets = assets * (1 + 0.012 * quarter_index) + rng.normal(0, 20)
            total_equity = total_assets * np.clip(equity_ratio + rng.normal(0, 0.03), 0.20, 0.82)
            total_debt = total_assets - total_equity
            operating_cash_flow = net_profit * rng.uniform(0.75, 1.25)
            capex = revenue * rng.uniform(0.025, 0.07)
            market_cap = max(revenue * rng.uniform(1.2, 4.0), 100)
            if bool(company["is_st"]):
                net_profit *= -0.4
                operating_cash_flow *= -0.5
                total_debt = total_assets * rng.uniform(0.78, 0.9)
                total_equity = max(total_assets - total_debt, 1)
            rows.append(
                {
                    "stock_code": company["stock_code"],
                    "ticker": company["stock_code"],
                    "date": report_date,
                    "report_date": report_date,
                    "announcement_date": report_date + pd.Timedelta(days=reporting_lag_days),
                    "revenue": revenue,
                    "net_profit": net_profit,
                    "profit_growth": np.nan,
                    "operating_cash_flow": operating_cash_flow,
                    "total_assets": total_assets,
                    "total_equity": total_equity,
                    "total_debt": total_debt,
                    "gross_profit": revenue * gross_margin,
                    "operating_profit": revenue * operating_margin,
                    "interest_expense": max(abs(total_debt) * rng.uniform(0.004, 0.018), 0.1),
                    "current_assets": total_assets * rng.uniform(0.30, 0.62),
                    "current_liabilities": total_assets * rng.uniform(0.18, 0.45),
                    "capex": capex,
                    "market_cap": market_cap,
                    "pe": np.nan if net_profit <= 0 else market_cap / max(net_profit * 4, 1),
                    "pb": market_cap / max(total_equity, 1),
                    "ps": market_cap / max(revenue * 4, 1),
                    "dividend_yield": max(rng.normal(0.018, 0.012), 0),
                }
            )
    fundamentals = pd.DataFrame(rows)
    for column in ["pe", "pb", "ps"]:
        fundamentals.loc[rng.choice(fundamentals.index, size=8, replace=False), column] = np.nan
    fundamentals["profit_growth"] = fundamentals.groupby("stock_code")["net_profit"].pct_change(4)
    fundamentals["asset_growth"] = fundamentals.groupby("stock_code")["total_assets"].pct_change(4)
    fundamentals["capex_growth"] = fundamentals.groupby("stock_code")["capex"].pct_change(4)
    return fundamentals


def _build_prices(
    universe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(start_date, end_date, freq="B")
    market_cycle = np.sin(np.arange(len(dates)) / 95.0) * 0.0008
    industry_beta = {
        "Technology": 1.25,
        "Healthcare": 0.85,
        "Consumer": 0.95,
        "Financials": 0.90,
        "Industrials": 1.05,
        "Materials": 1.10,
        "Energy": 1.00,
        "Utilities": 0.55,
        "Real Estate": 1.20,
        "Communication": 1.05,
    }
    price_rows: list[dict[str, object]] = []
    benchmark_returns = 0.00018 + market_cycle + rng.normal(0, 0.012, len(dates))
    benchmark_close = 1000 * np.cumprod(1 + benchmark_returns)
    benchmark = pd.DataFrame({"date": dates, "benchmark_code": "399673", "close": benchmark_close})

    for _, company in universe.iterrows():
        beta = industry_beta.get(str(company["industry"]), 1.0)
        alpha = rng.normal(0.00005, 0.00015)
        specific = rng.normal(0, 0.010 + 0.004 * beta, len(dates))
        daily_returns = alpha + beta * benchmark_returns * 0.55 + specific
        if bool(company["is_st"]):
            daily_returns -= 0.00025
            daily_returns += rng.normal(0, 0.010, len(dates))
        close = rng.uniform(8, 45) * np.cumprod(1 + daily_returns)
        close = np.maximum(close, 0.8)
        base_volume = rng.uniform(2_000_000, 28_000_000)
        volume = base_volume * (1 + rng.normal(0, 0.25, len(dates)))
        volume = np.maximum(volume, 50_000)
        turnover = volume * close
        open_price = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_price, close) * (1 + np.abs(rng.normal(0, 0.006, len(dates))))
        low = np.minimum(open_price, close) * (1 - np.abs(rng.normal(0, 0.006, len(dates))))
        for date, stock_open, stock_high, stock_low, stock_close, stock_volume, stock_turnover, stock_return in zip(
            dates,
            open_price,
            high,
            low,
            close,
            volume,
            turnover,
            daily_returns,
            strict=True,
        ):
            price_rows.append(
                {
                    "date": date,
                    "stock_code": company["stock_code"],
                    "ticker": company["stock_code"],
                    "open": stock_open,
                    "high": stock_high,
                    "low": stock_low,
                    "close": stock_close,
                    "volume": stock_volume,
                    "turnover": stock_turnover,
                    "amount": stock_turnover,
                    "return": stock_return,
                }
            )
    benchmark["return"] = benchmark["close"].pct_change()
    return pd.DataFrame(price_rows), benchmark
