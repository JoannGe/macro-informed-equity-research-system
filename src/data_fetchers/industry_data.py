"""Industry universe and transmission matrix loaders."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.processing.build_demo_data import ensure_demo_data
from src.utils.io_utils import load_yaml, read_dataframe, resolve_path


def load_universe(mode: str, config: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load stock universe metadata."""

    logger = logger or logging.getLogger("investment_system")
    if mode == "demo":
        return ensure_demo_data(config)["universe"]

    csv_path = config.get("data", {}).get("local_csv", {}).get("universe")
    if csv_path and resolve_path(csv_path).exists():
        logger.info("Loading universe from local CSV fallback: %s", csv_path)
        return read_dataframe(csv_path, parse_dates=["listing_date"])

    logger.warning("No live universe source was available. Falling back to synthetic demo universe.")
    return ensure_demo_data(config)["universe"]


def load_industry_transmission_matrix(
    path: str = "config/industry_transmission_matrix.yaml",
) -> pd.DataFrame:
    """Load editable industry sensitivity assumptions as a dataframe."""

    payload = load_yaml(path)
    industries = payload.get("industries", {})
    matrix = pd.DataFrame.from_dict(industries, orient="index")
    matrix.index.name = "industry"
    return matrix.reset_index()
