"""Date helpers for live and demo research modes."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def resolve_config_date(value: str | date | pd.Timestamp, *, demo_default: str | None = None) -> pd.Timestamp:
    """Resolve config dates, including the special value ``today``.

    ``today`` is evaluated at runtime. A deterministic ``demo_default`` can be
    supplied by callers that need reproducible synthetic data.
    """

    if isinstance(value, str) and value.strip().lower() == "today":
        if demo_default is not None:
            return pd.Timestamp(demo_default)
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(value)


def requested_date_range(config: dict[str, Any], mode: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Return requested start/end dates and the original end-date label."""

    data_config = config.get("data", {})
    start_date = pd.Timestamp(data_config.get("start_date", "2020-01-01"))
    requested_end = str(data_config.get("end_date", "today"))
    demo_default = "2025-12-31" if mode == "demo" else None
    end_date = resolve_config_date(requested_end, demo_default=demo_default)
    return start_date, end_date, requested_end


def compact_date(value: pd.Timestamp | str) -> str:
    """Format a date as YYYYMMDD for public data APIs."""

    return pd.Timestamp(value).strftime("%Y%m%d")
