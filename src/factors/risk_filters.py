"""Explicit pre-portfolio risk filters."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_risk_filters(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Flag or exclude stocks based on simple interpretable risk rules."""

    thresholds = config.get("risk_filters", {})
    result = scored.copy()
    result["risk_filter_reason"] = ""
    result["excluded_by_risk_filter"] = False

    _append_filter(
        result,
        result.get("debt_to_assets", 0) > float(thresholds.get("max_debt_to_assets", 0.75)),
        "extreme debt-to-assets",
    )
    _append_filter(
        result,
        result.get("debt_to_equity", 0) > float(thresholds.get("max_debt_to_equity", 3.0)),
        "extreme debt-to-equity",
    )
    _append_filter(
        result,
        result.get("volatility", 0) > float(thresholds.get("max_annualized_volatility", 0.65)),
        "extremely high volatility",
    )
    _append_filter(
        result,
        result.get("avg_turnover", 0) < float(thresholds.get("min_avg_turnover", 5_000_000)),
        "poor liquidity",
    )
    if thresholds.get("exclude_negative_operating_cash_flow", True):
        _append_filter(
            result,
            result.get("operating_cash_flow", 0) < 0,
            "negative operating cash flow",
        )
    if thresholds.get("exclude_st", True) and "is_st" in result.columns:
        _append_filter(result, result["is_st"].apply(_as_bool), "ST or *ST status")

    missing_threshold = int(config.get("factor_processing", {}).get("missing_key_threshold", 3))
    _append_filter(
        result,
        result.get("missing_key_count", 0) >= missing_threshold,
        "too many missing key inputs",
    )
    result["risk_filter_reason"] = result["risk_filter_reason"].str.strip("; ")
    return result


def _append_filter(df: pd.DataFrame, mask: pd.Series | bool, reason: str) -> None:
    if isinstance(mask, bool):
        mask = pd.Series(mask, index=df.index)
    mask = mask.fillna(False)
    df.loc[mask, "excluded_by_risk_filter"] = True
    df.loc[mask, "risk_filter_reason"] = df.loc[mask, "risk_filter_reason"] + reason + "; "


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "st", "*st"}
    return bool(value)
