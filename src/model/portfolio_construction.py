"""Transparent long-only portfolio construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FACTOR_COLUMNS = [
    "value_score",
    "momentum_score",
    "quality_score",
    "investment_score",
    "low_volatility_score",
    "liquidity_score",
    "industry_score",
    "risk_penalty_score",
]


def construct_portfolios(scored: pd.DataFrame, config: dict[str, Any], method: str | None = None) -> pd.DataFrame:
    """Build transparent portfolios for every rebalance date.

    The returned dataframe includes both ``final_weight`` and a backwards-
    compatible ``weight`` alias used by the backtester.
    """

    portfolio_config = config.get("portfolio", {})
    selected_method = method or portfolio_config.get("method", "equal_weight")
    frames = []
    diagnostics = []
    for as_of_date, date_scores in scored.groupby("as_of_date", sort=True):
        portfolio, date_diagnostics = construct_portfolio_for_date(
            date_scores,
            config,
            selected_method,
            return_diagnostics=True,
        )
        portfolio["as_of_date"] = as_of_date
        date_diagnostics["as_of_date"] = str(pd.Timestamp(as_of_date).date())
        frames.append(portfolio)
        diagnostics.append(date_diagnostics)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.attrs["diagnostics"] = diagnostics
    return result


def construct_portfolio_for_date(
    scored_for_date: pd.DataFrame,
    config: dict[str, Any],
    method: str = "equal_weight",
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Apply filters, rank stocks, assign weights, and explain the result."""

    portfolio_config = config.get("portfolio", {})
    top_n = int(portfolio_config.get("top_n", 20))
    max_stock_weight = float(portfolio_config.get("max_single_stock_weight", 0.05))
    max_industry_weight = float(portfolio_config.get("max_industry_weight", 0.25))
    min_liquidity = float(portfolio_config.get("min_liquidity", 5_000_000))
    benchmark = config.get("project", {}).get("benchmark_code", "399673")

    candidates = scored_for_date.copy()
    if "ticker" not in candidates.columns:
        candidates["ticker"] = candidates.get("stock_code")
    universe_size = len(candidates)

    required_scores = [column for column in ["total_score", "value_score", "momentum_score", "quality_score"] if column in candidates.columns]
    data_mask = candidates[required_scores].notna().all(axis=1) if required_scores else pd.Series(True, index=candidates.index)
    after_data = candidates[data_mask].copy()

    liquidity_mask = after_data.get("avg_turnover", 0).fillna(0) >= min_liquidity
    after_liquidity = after_data[liquidity_mask].copy()

    risk_mask = ~after_liquidity.get("excluded_by_risk_filter", False).fillna(False)
    after_risk = after_liquidity[risk_mask].copy()
    ranked = after_risk.sort_values("total_score", ascending=False)
    selected = _select_with_industry_feasibility(ranked, top_n, max_stock_weight, max_industry_weight)

    diagnostics = {
        "universe_size": universe_size,
        "after_data_filter_size": len(after_data),
        "after_liquidity_filter_size": len(after_liquidity),
        "after_risk_filter_size": len(after_risk),
        "selected_stock_count": len(selected),
        "weighting_method": method,
        "max_single_stock_weight": max_stock_weight,
        "max_industry_weight": max_industry_weight,
        "transaction_cost_bps": portfolio_config.get("transaction_cost_bps", 10),
        "benchmark": benchmark,
        "constraints_binding": [],
        "warnings": [],
    }

    if selected.empty:
        diagnostics["warnings"].append("No stocks passed data, liquidity, and risk filters.")
        empty = pd.DataFrame(columns=_portfolio_columns(scored_for_date))
        return (empty, diagnostics) if return_diagnostics else empty

    selected = selected.copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["portfolio_method"] = method
    selected["initial_weight"] = _initial_weights(selected, method)
    selected["final_weight"] = _apply_weight_caps(
        selected,
        selected["initial_weight"],
        max_stock_weight,
        max_industry_weight,
        diagnostics,
    )
    selected["weight"] = selected["final_weight"]
    selected["selection_reason"] = selected.apply(_selection_reason, axis=1)
    selected["key_risk_warning"] = selected.apply(_risk_warning, axis=1)
    selected["stock_code"] = selected.get("stock_code", selected["ticker"])

    output_columns = [column for column in _portfolio_columns(selected) if column in selected.columns]
    result = selected[output_columns].copy()
    return (result, diagnostics) if return_diagnostics else result


def portfolio_diagnostics_for_latest(portfolio: pd.DataFrame) -> dict[str, Any]:
    """Return the latest diagnostics stored on a portfolio dataframe."""

    diagnostics = portfolio.attrs.get("diagnostics", [])
    return diagnostics[-1] if diagnostics else {}


def _select_with_industry_feasibility(
    ranked: pd.DataFrame,
    top_n: int,
    max_stock_weight: float,
    max_industry_weight: float,
) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    industry_max_count = max(1, int(np.floor(max_industry_weight / max_stock_weight)))
    selected_rows = []
    industry_counts: dict[str, int] = {}
    for _, row in ranked.iterrows():
        industry = str(row.get("industry", "Unknown"))
        if industry_counts.get(industry, 0) >= industry_max_count:
            continue
        selected_rows.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_rows) >= top_n:
            break
    return pd.DataFrame(selected_rows)


def _initial_weights(selected: pd.DataFrame, method: str) -> pd.Series:
    if method == "score_weight":
        scores = selected["total_score"].astype(float)
        shifted = scores - scores.min() + 1e-6
        return shifted / shifted.sum()
    return pd.Series(1 / len(selected), index=selected.index)


def _apply_weight_caps(
    selected: pd.DataFrame,
    initial: pd.Series,
    max_stock_weight: float,
    max_industry_weight: float,
    diagnostics: dict[str, Any],
) -> pd.Series:
    weights = _cap_and_redistribute(initial, pd.Series(max_stock_weight, index=selected.index))
    if (initial > max_stock_weight + 1e-12).any():
        diagnostics["constraints_binding"].append("max_single_stock_weight")
    for industry in selected["industry"].dropna().unique():
        mask = selected["industry"].eq(industry)
        total = weights.loc[mask].sum()
        if total > max_industry_weight + 1e-12:
            diagnostics["constraints_binding"].append(f"max_industry_weight:{industry}")
            weights.loc[mask] *= max_industry_weight / total
            remaining = 1 - weights.sum()
            if remaining > 1e-12:
                caps = pd.Series(max_stock_weight, index=selected.index)
                caps.loc[mask] = weights.loc[mask]
                weights = _redistribute_to_capacity(weights, caps, ~mask, remaining)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return weights


def _cap_and_redistribute(weights: pd.Series, caps: pd.Series) -> pd.Series:
    capped = weights.clip(upper=caps)
    return _redistribute_to_capacity(capped, caps, capped < caps - 1e-12, 1 - capped.sum())


def _redistribute_to_capacity(weights: pd.Series, caps: pd.Series, eligible: pd.Series, remaining: float) -> pd.Series:
    result = weights.copy()
    while remaining > 1e-10 and eligible.any():
        capacity = (caps - result).clip(lower=0)
        available = capacity[eligible]
        if available.sum() <= 1e-12:
            break
        result.loc[eligible] += remaining * available / available.sum()
        result = result.clip(upper=caps)
        remaining = 1 - result.sum()
        eligible = result < caps - 1e-12
    if result.sum() > 0:
        result = result / result.sum()
    return result


def _selection_reason(row: pd.Series) -> str:
    factor_values = {column: row.get(column) for column in FACTOR_COLUMNS if column in row.index and pd.notna(row.get(column))}
    strongest = sorted(factor_values.items(), key=lambda item: item[1], reverse=True)[:3]
    parts = [f"{name.replace('_score', '')} {value:+.2f}" for name, value in strongest]
    return "Selected after filters because total_score ranked highly; strongest factors: " + ", ".join(parts) + "."


def _risk_warning(row: pd.Series) -> str:
    warning = row.get("risk_filter_reason")
    if isinstance(warning, str) and warning.strip():
        return warning
    if pd.notna(row.get("risk_penalty_score")) and float(row.get("risk_penalty_score")) > 0.75:
        return f"Elevated risk penalty score {float(row.get('risk_penalty_score')):.2f}."
    if row.get("data_quality_warning"):
        return str(row.get("data_quality_warning"))
    return "No major configured risk warning after filters."


def _portfolio_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "as_of_date",
        "ticker",
        "stock_code",
        "stock_name",
        "industry",
        "selection_rank",
        "total_score",
        *FACTOR_COLUMNS,
        "initial_weight",
        "final_weight",
        "weight",
        "selection_reason",
        "key_risk_warning",
        "portfolio_method",
        "avg_turnover",
        "data_quality_warning",
    ]
    return [column for column in preferred if column in df.columns]


def validate_portfolio_constraints(portfolio: pd.DataFrame, config: dict[str, Any]) -> dict[str, bool]:
    """Return constraint checks used by tests and the dashboard."""

    portfolio_config = config.get("portfolio", {})
    max_stock_weight = float(portfolio_config.get("max_single_stock_weight", 0.05))
    max_industry_weight = float(portfolio_config.get("max_industry_weight", 0.25))
    weight_column = "final_weight" if "final_weight" in portfolio.columns else "weight"
    checks = {}
    for as_of_date, holdings in portfolio.groupby("as_of_date"):
        prefix = str(pd.Timestamp(as_of_date).date())
        checks[f"{prefix}_weights_sum_to_one"] = bool(np.isclose(holdings[weight_column].sum(), 1.0, atol=1e-6))
        checks[f"{prefix}_single_stock_cap"] = bool(holdings[weight_column].max() <= max_stock_weight + 1e-6)
        industry_weights = holdings.groupby("industry")[weight_column].sum()
        checks[f"{prefix}_industry_cap"] = bool(industry_weights.max() <= max_industry_weight + 1e-6)
        checks[f"{prefix}_long_only"] = bool(holdings[weight_column].min() >= -1e-12)
    return checks
