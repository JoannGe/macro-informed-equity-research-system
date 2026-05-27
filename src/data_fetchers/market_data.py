"""A-share market price loading with demo and local fallback support."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.processing.build_demo_data import ensure_demo_data
from src.utils.io_utils import read_dataframe, resolve_path


def load_price_data(mode: str, config: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load stock prices for the selected universe."""

    logger = logger or logging.getLogger("investment_system")
    if mode == "demo":
        return ensure_demo_data(config)["prices"]

    load_dotenv(resolve_path(".env"))
    csv_path = config.get("data", {}).get("local_csv", {}).get("prices")
    if csv_path and resolve_path(csv_path).exists():
        logger.info("Loading price data from local CSV fallback: %s", csv_path)
        return read_dataframe(csv_path, parse_dates=["date"])

    logger.warning(
        "No live A-share price source was available. Falling back to synthetic demo prices."
    )
    return ensure_demo_data(config)["prices"]


def load_benchmark_data(mode: str, config: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load the benchmark series for ChiNext 50 index code 399673."""

    logger = logger or logging.getLogger("investment_system")
    if mode == "demo":
        return ensure_demo_data(config)["benchmark"]

    csv_path = config.get("data", {}).get("local_csv", {}).get("benchmark")
    if csv_path and resolve_path(csv_path).exists():
        logger.info("Loading benchmark data from local CSV fallback: %s", csv_path)
        return read_dataframe(csv_path, parse_dates=["date"])

    logger.warning(
        "No live benchmark source was available for 399673. Falling back to synthetic demo benchmark."
    )
    return ensure_demo_data(config)["benchmark"]
