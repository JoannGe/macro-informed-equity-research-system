"""Liquidity factor and tradability support.

Liquidity is mainly a tradability and implementation-risk control. The score
is included with a small default weight, while hard minimum liquidity remains
in portfolio construction and risk filters.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_liquidity_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``liquidity_score`` from average trading amount."""

    result = panel.copy()
    metrics = ["avg_turnover"]
    result = winsorize_and_zscore(result, metrics, config)
    result["liquidity_score"] = row_mean(result, ["avg_turnover_z"])
    return result
