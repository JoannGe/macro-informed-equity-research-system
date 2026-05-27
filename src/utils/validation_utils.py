"""Validation helpers with explicit, beginner-friendly error messages."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    """Raise a clear error when a dataset is missing expected columns."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def coerce_dates(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with selected columns converted to pandas datetimes."""

    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])
    return result


def missing_warning(row: pd.Series, columns: Iterable[str]) -> str:
    """Create a short warning string for missing key fields in a row."""

    missing = [column for column in columns if column in row.index and pd.isna(row[column])]
    if not missing:
        return ""
    return "Missing key data: " + ", ".join(missing)
