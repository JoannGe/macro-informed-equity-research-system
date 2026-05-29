"""Shared factor-cleaning helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def winsorize_and_zscore(df: pd.DataFrame, columns: list[str], config: dict[str, Any]) -> pd.DataFrame:
    """Winsorize raw metrics and convert them to cross-sectional z-scores by date."""

    result = df.copy()
    lower = float(config.get("factor_processing", {}).get("winsorize_lower_quantile", 0.05))
    upper = float(config.get("factor_processing", {}).get("winsorize_upper_quantile", 0.95))
    for column in columns:
        if column not in result.columns:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result.groupby("as_of_date")[column].transform(
            lambda series: series.clip(series.quantile(lower), series.quantile(upper))
        )
        result[f"{column}_z"] = result.groupby("as_of_date")[column].transform(_zscore)
    return result


def row_mean(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average available columns row-wise, returning zero when all are missing."""

    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(0.0, index=df.index)
    return df[available].mean(axis=1, skipna=True).fillna(0.0)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
