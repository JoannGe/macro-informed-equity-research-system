"""Fundamental data loading with point-in-time fields preserved."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.processing.build_demo_data import ensure_demo_data
from src.utils.io_utils import read_dataframe, resolve_path


def load_fundamentals(mode: str, config: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load firm-level fundamentals.

    Required date fields are ``report_date`` and ``announcement_date``. The
    latter is what downstream modules use to prevent look-ahead bias.
    """

    logger = logger or logging.getLogger("investment_system")
    if mode == "demo":
        return ensure_demo_data(config)["fundamentals"]

    load_dotenv(resolve_path(".env"))
    csv_path = config.get("data", {}).get("local_csv", {}).get("fundamentals")
    if csv_path and resolve_path(csv_path).exists():
        logger.info("Loading fundamentals from local CSV fallback: %s", csv_path)
        return read_dataframe(csv_path, parse_dates=["report_date", "announcement_date"])

    logger.warning(
        "No live fundamentals source was available. Falling back to synthetic demo fundamentals."
    )
    return ensure_demo_data(config)["fundamentals"]
