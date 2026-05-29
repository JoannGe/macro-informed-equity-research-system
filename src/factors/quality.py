"""Quality/profitability factor."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_quality_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``quality_score`` using profitability and margin metrics."""

    result = panel.copy()
    metrics = ["roe", "roa", "gross_margin", "operating_margin"]
    result = winsorize_and_zscore(result, metrics, config)
    result["quality_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result
