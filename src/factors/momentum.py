"""Price momentum factor using only trailing price data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_momentum_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``momentum_score`` from 6-month and 12-month-ex-1-month momentum."""

    result = panel.copy()
    metrics = ["momentum_6m", "momentum_12m_ex_1m"]
    result = winsorize_and_zscore(result, metrics, config)
    result["momentum_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result
