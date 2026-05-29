"""Value factor based on accepted valuation ratios.

The value factor uses earnings yield, book-to-market, and sales yield. These
are transparent transforms of PE, PB, and PS where lower valuation multiples
map to higher factor scores. Missing or non-positive multiples remain missing;
the model does not invent valuation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_value_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add raw value metrics and ``value_score``."""

    result = panel.copy()
    result["earnings_yield"] = _safe_inverse(result.get("pe"))
    result["book_to_market"] = _safe_inverse(result.get("pb"))
    result["sales_yield"] = _safe_inverse(result.get("ps"))
    metrics = ["earnings_yield", "book_to_market", "sales_yield"]
    result = winsorize_and_zscore(result, metrics, config)
    result["value_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result


def _safe_inverse(series: object) -> pd.Series:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    values = values.where(values > 0)
    return 1 / values.replace(0, np.nan)
