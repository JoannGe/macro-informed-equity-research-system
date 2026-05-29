"""Investment / asset-growth factor.

Academic factor models often treat aggressive asset growth as a potential
negative signal rather than automatically rewarding expansion. This module
therefore gives higher scores to firms with lower, more disciplined asset and
capex growth, while leaving missing values transparent.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore


def add_investment_factor(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add ``investment_score`` where lower aggressive growth is better."""

    result = panel.copy()
    result["asset_growth_disciplined"] = -pd.to_numeric(result.get("asset_growth"), errors="coerce")
    result["capex_growth_disciplined"] = -pd.to_numeric(result.get("capex_growth"), errors="coerce")
    metrics = ["asset_growth_disciplined", "capex_growth_disciplined"]
    result = winsorize_and_zscore(result, metrics, config)
    result["investment_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result
