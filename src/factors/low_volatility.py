"""Low-volatility factor used as a risk-aware attractiveness measure."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_low_volatility_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``low_volatility_score`` from realized volatility and drawdown."""

    result = panel.copy()
    result["realized_volatility_low"] = -pd.to_numeric(result.get("volatility"), errors="coerce")
    result["drawdown_low"] = pd.to_numeric(result.get("max_drawdown_6m"), errors="coerce")
    metrics = ["realized_volatility_low", "drawdown_low"]
    result = winsorize_and_zscore(result, metrics, config)
    result["low_volatility_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result
