"""V2 factor pipeline for transparent multi-factor equity screening."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.factor_utils import row_mean, winsorize_and_zscore
from src.factors.investment import add_investment_factor
from src.factors.liquidity import add_liquidity_factor
from src.factors.low_volatility import add_low_volatility_factor
from src.factors.momentum import add_momentum_factor
from src.factors.quality import add_quality_factor
from src.factors.value import add_value_factor


KEY_FACTOR_INPUTS = ["pe", "pb", "momentum_6m", "roe", "volatility", "avg_turnover"]


def calculate_factor_scores(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Calculate all V2 factor scores and risk-penalty score."""

    result = panel.copy()
    if "ticker" not in result.columns:
        result["ticker"] = result["stock_code"]
    result["missing_key_count"] = result[[column for column in KEY_FACTOR_INPUTS if column in result.columns]].isna().sum(axis=1)
    result = add_value_factor(result, config)
    result = add_momentum_factor(result, config)
    result = add_quality_factor(result, config)
    result = add_investment_factor(result, config)
    result = add_low_volatility_factor(result, config)
    result = add_liquidity_factor(result, config)
    result = _add_risk_penalty(result, config)
    result["data_quality_warning"] = result.apply(_data_quality_warning, axis=1)
    return result


def _add_risk_penalty(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = panel.copy()
    result["leverage_penalty"] = pd.to_numeric(result.get("debt_to_assets"), errors="coerce")
    result["debt_equity_penalty"] = pd.to_numeric(result.get("debt_to_equity"), errors="coerce")
    result["volatility_penalty"] = pd.to_numeric(result.get("volatility"), errors="coerce")
    result["missing_data_penalty"] = pd.to_numeric(result.get("missing_key_count"), errors="coerce")
    result["negative_cash_flow_penalty"] = pd.to_numeric(result.get("operating_cash_flow"), errors="coerce").lt(0).astype(int)
    metrics = [
        "leverage_penalty",
        "debt_equity_penalty",
        "volatility_penalty",
        "missing_data_penalty",
        "negative_cash_flow_penalty",
    ]
    result = winsorize_and_zscore(result, metrics, config)
    result["risk_penalty_score"] = row_mean(result, [f"{column}_z" for column in metrics])
    return result


def _data_quality_warning(row: pd.Series) -> str:
    missing = [column for column in KEY_FACTOR_INPUTS if column in row.index and pd.isna(row[column])]
    if not missing:
        return ""
    return "Missing factor inputs: " + ", ".join(missing)
