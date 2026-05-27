"""Macro data loading for demo and live/research modes."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.processing.build_demo_data import ensure_demo_data
from src.utils.io_utils import read_dataframe, resolve_path


def load_macro_data(mode: str, config: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load macro variables.

    Live mode is deliberately conservative: it first uses a local CSV fallback
    because macro APIs change often. Optional API wiring can be extended later
    without changing downstream factor code.
    """

    logger = logger or logging.getLogger("investment_system")
    if mode == "demo":
        return ensure_demo_data(config)["macro"]

    load_dotenv(resolve_path(".env"))
    csv_path = config.get("data", {}).get("local_csv", {}).get("macro")
    if csv_path and resolve_path(csv_path).exists():
        logger.info("Loading macro data from local CSV fallback: %s", csv_path)
        return read_dataframe(csv_path, parse_dates=["date"])

    logger.warning(
        "Live macro APIs are unavailable or not configured. Falling back to synthetic demo macro data."
    )
    return ensure_demo_data(config)["macro"]
