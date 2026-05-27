"""Cross-sectional firm factor scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


QUALITY_COLUMNS = ["roe", "roa", "gross_margin", "operating_margin", "ocf_to_revenue", "fcf_to_revenue"]
GROWTH_COLUMNS = ["revenue_growth", "net_profit_growth", "operating_cash_flow_growth"]
VALUATION_LOW_IS_BETTER = ["pe", "pb", "ps"]
VALUATION_HIGH_IS_BETTER = ["dividend_yield"]
MOMENTUM_COLUMNS = ["momentum_6m", "momentum_12m_ex_1m", "avg_turnover"]
RISK_COLUMNS_HIGH_IS_BAD = ["debt_to_assets", "debt_to_equity", "volatility", "missing_key_count"]
RISK_COLUMNS_LOW_IS_BAD = ["interest_coverage", "current_ratio", "avg_turnover"]
KEY_DATA_COLUMNS = ["roe", "revenue_growth", "pe", "pb", "momentum_6m", "debt_to_assets", "operating_cash_flow"]


def calculate_firm_factor_scores(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Winsorize, z-score, and aggregate firm-level factor groups by date."""

    result = panel.copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    result["missing_key_count"] = result[KEY_DATA_COLUMNS].isna().sum(axis=1)
    result["negative_ocf_flag"] = result["operating_cash_flow"].lt(0).fillna(False).astype(int)

    for column in VALUATION_LOW_IS_BETTER:
        if column in result.columns:
            result.loc[result[column] <= 0, column] = np.nan

    score_inputs: dict[str, str] = {}
    for column in QUALITY_COLUMNS + GROWTH_COLUMNS + VALUATION_HIGH_IS_BETTER + MOMENTUM_COLUMNS:
        if column in result.columns:
            score_inputs[column] = column
    for column in VALUATION_LOW_IS_BETTER:
        if column in result.columns:
            transformed = f"{column}_lower_is_better"
            result[transformed] = -result[column]
            score_inputs[transformed] = transformed
    result["volatility_lower_is_better"] = -result.get("volatility", np.nan)
    result["negative_ocf_risk"] = result["negative_ocf_flag"]
    result["interest_coverage_low_is_bad"] = -result.get("interest_coverage", np.nan)
    result["current_ratio_low_is_bad"] = -result.get("current_ratio", np.nan)
    result["avg_turnover_low_is_bad"] = -result.get("avg_turnover", np.nan)

    all_score_columns = list(score_inputs.values()) + [
        "volatility_lower_is_better",
        "debt_to_assets",
        "debt_to_equity",
        "volatility",
        "missing_key_count",
        "negative_ocf_risk",
        "interest_coverage_low_is_bad",
        "current_ratio_low_is_bad",
        "avg_turnover_low_is_bad",
    ]
    all_score_columns = [column for column in all_score_columns if column in result.columns]
    result = _winsorize_by_date(result, all_score_columns, config)
    result = _zscore_by_date(result, all_score_columns)

    quality_z = [f"{column}_z" for column in QUALITY_COLUMNS if f"{column}_z" in result.columns]
    growth_z = [f"{column}_z" for column in GROWTH_COLUMNS if f"{column}_z" in result.columns]
    valuation_z = [f"{column}_lower_is_better_z" for column in VALUATION_LOW_IS_BETTER]
    valuation_z += [f"{column}_z" for column in VALUATION_HIGH_IS_BETTER]
    valuation_z = [column for column in valuation_z if column in result.columns]
    momentum_z = [f"{column}_z" for column in MOMENTUM_COLUMNS if f"{column}_z" in result.columns]
    if "volatility_lower_is_better_z" in result.columns:
        momentum_z.append("volatility_lower_is_better_z")

    risk_z = [f"{column}_z" for column in RISK_COLUMNS_HIGH_IS_BAD if f"{column}_z" in result.columns]
    risk_z += [
        column
        for column in [
            "negative_ocf_risk_z",
            "interest_coverage_low_is_bad_z",
            "current_ratio_low_is_bad_z",
            "avg_turnover_low_is_bad_z",
        ]
        if column in result.columns
    ]

    result["quality_score"] = result[quality_z].mean(axis=1, skipna=True)
    result["growth_score"] = result[growth_z].mean(axis=1, skipna=True)
    result["valuation_score"] = result[valuation_z].mean(axis=1, skipna=True)
    result["momentum_score"] = result[momentum_z].mean(axis=1, skipna=True)
    result["risk_score"] = result[risk_z].mean(axis=1, skipna=True)
    result["data_quality_warning"] = result.apply(_data_quality_warning, axis=1)
    return result


def _winsorize_by_date(df: pd.DataFrame, columns: list[str], config: dict[str, Any]) -> pd.DataFrame:
    lower = float(config.get("factor_processing", {}).get("winsorize_lower_quantile", 0.05))
    upper = float(config.get("factor_processing", {}).get("winsorize_upper_quantile", 0.95))
    result = df.copy()
    for column in columns:
        result[column] = result.groupby("as_of_date")[column].transform(
            lambda series: series.clip(series.quantile(lower), series.quantile(upper))
        )
    return result


def _zscore_by_date(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[f"{column}_z"] = result.groupby("as_of_date")[column].transform(_zscore)
    return result


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _data_quality_warning(row: pd.Series) -> str:
    missing = [column for column in KEY_DATA_COLUMNS if column in row.index and pd.isna(row[column])]
    if not missing:
        return ""
    return "Missing key inputs: " + ", ".join(missing)
