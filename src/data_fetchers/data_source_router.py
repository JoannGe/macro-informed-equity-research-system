"""Route data requests across demo, AKShare, BaoStock, and optional sources."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from src.data_fetchers.akshare_fetcher import fetch_akshare_data
from src.data_fetchers.baostock_fetcher import fetch_baostock_data
from src.processing.build_demo_data import ensure_demo_data
from src.processing.standardize_schema import (
    build_universe_from_schema,
    standardize_benchmark,
    standardize_fundamentals,
    standardize_industry,
    standardize_prices,
)
from src.utils.date_utils import requested_date_range
from src.utils.io_utils import processed_path, write_dataframe


def load_research_data(
    mode: str,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load data and return standardized tables plus a status report."""

    logger = logger or logging.getLogger("investment_system")
    start_date, end_date, requested_end = requested_date_range(config, mode)
    warnings: list[str] = []
    fallback_used = False
    source = "demo"
    raw: dict[str, pd.DataFrame]

    if mode == "demo":
        raw = _load_demo_standardized(config)
        warnings.append("Demo mode uses synthetic data for reproducible tests.")
    else:
        raw = {}
        errors = []
        for source_name, fetcher in [("akshare", fetch_akshare_data), ("baostock", fetch_baostock_data)]:
            try:
                logger.info("Trying %s live data source.", source_name)
                raw = fetcher(config, start_date, end_date)
                source = source_name
                break
            except Exception as exc:
                message = f"{source_name} failed: {exc}"
                errors.append(message)
                logger.warning(message)
        if not raw:
            if not config.get("data", {}).get("allow_demo_fallback", True):
                raise RuntimeError("Live data sources failed and demo fallback is disabled: " + " | ".join(errors))
            fallback_used = True
            source = "demo"
            warnings.extend(errors)
            warnings.append("Live data unavailable; using synthetic demo fallback. Do not treat output as real market data.")
            raw = _load_demo_standardized(config)

    standardized = _standardize_payload(raw, config)
    status = _build_status(
        mode=mode,
        source=source,
        config=config,
        requested_start=start_date,
        requested_end_label=requested_end,
        requested_end=end_date,
        data=standardized,
        fallback_used=fallback_used,
        warnings=warnings,
    )
    _write_standardized_tables(standardized, config, status)
    return standardized, status


def _load_demo_standardized(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    demo = ensure_demo_data(config, force=True)
    return {
        "prices": demo["prices"],
        "benchmark": demo["benchmark"],
        "fundamentals": demo["fundamentals"],
        "industry": demo["universe"][["stock_code", "industry"]],
        "names": demo["universe"][["stock_code", "stock_name"]],
        "universe": demo["universe"],
        "macro": demo["macro"],
    }


def _standardize_payload(raw: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    reporting_lag_days = int(config.get("data", {}).get("reporting_lag_days", 75))
    benchmark_code = str(config.get("project", {}).get("benchmark_code", "399673"))
    prices = standardize_prices(raw["prices"])
    benchmark = standardize_benchmark(raw["benchmark"], benchmark_code)
    fundamentals = standardize_fundamentals(raw["fundamentals"], reporting_lag_days)
    industry = standardize_industry(raw["industry"])
    names = raw.get("names", pd.DataFrame())
    if not names.empty:
        names = names.rename(columns={"stock_code": "ticker", "code": "ticker"})
    universe = build_universe_from_schema(industry, names)
    return {
        "prices": prices,
        "benchmark": benchmark,
        "fundamentals": fundamentals,
        "industry": industry,
        "universe": universe,
        "macro": raw.get("macro", pd.DataFrame()),
    }


def _build_status(
    *,
    mode: str,
    source: str,
    config: dict[str, Any],
    requested_start: pd.Timestamp,
    requested_end_label: str,
    requested_end: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    fallback_used: bool,
    warnings: list[str],
) -> dict[str, Any]:
    prices = data["prices"]
    fundamentals = data["fundamentals"]
    latest_price = prices["date"].max() if not prices.empty else pd.NaT
    latest_fundamental = fundamentals["date"].max() if not fundamentals.empty else pd.NaT
    if latest_price is not pd.NaT and latest_price < requested_end:
        warnings.append("Market data uses the latest available trading day returned by the source.")
    missing_fundamental_columns = [
        column
        for column in ["roe", "roa", "gross_margin", "operating_margin", "debt_to_assets", "debt_to_equity"]
        if column in fundamentals.columns and fundamentals[column].isna().all()
    ]
    if missing_fundamental_columns:
        warnings.append("Missing fundamental fields: " + ", ".join(missing_fundamental_columns))
    data_type = "demo" if source == "demo" else "latest-available"
    return {
        "mode": mode,
        "data_source": source,
        "requested_start_date": str(requested_start.date()),
        "requested_end_date": requested_end_label,
        "resolved_end_date": str(requested_end.date()),
        "actual_latest_price_date": _json_date(latest_price),
        "actual_latest_fundamental_date": _json_date(latest_fundamental),
        "benchmark_code": config.get("project", {}).get("benchmark_code", "399673"),
        "benchmark_name": config.get("project", {}).get("benchmark_name", "ChiNext 50 Index"),
        "fallback_used": fallback_used,
        "data_freshness": data_type,
        "warnings": warnings,
    }


def _write_standardized_tables(data: dict[str, pd.DataFrame], config: dict[str, Any], status: dict[str, Any]) -> None:
    for name in ["prices", "fundamentals", "industry", "benchmark"]:
        write_dataframe(data[name], processed_path(f"{name}.csv", config))
    status_path = processed_path("data_status.json", config)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def _json_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())
