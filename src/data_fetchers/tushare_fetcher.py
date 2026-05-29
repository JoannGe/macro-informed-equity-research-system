"""Optional Tushare Pro fetcher placeholder.

Tushare Pro is not required for V2. This module only activates when the user
provides ``TUSHARE_TOKEN`` in ``.env`` and can be extended later.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.utils.io_utils import resolve_path


def fetch_tushare_data(config: dict[str, Any], start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Raise a clear message unless Tushare is configured."""

    load_dotenv(resolve_path(".env"))
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured; Tushare is optional and was skipped.")
    raise RuntimeError("Tushare support is optional and not implemented in this V2 baseline.")
