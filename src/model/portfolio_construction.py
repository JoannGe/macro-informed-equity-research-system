"""Long-only portfolio construction with explicit constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def construct_portfolios(scored: pd.DataFrame, config: dict[str, Any], method: str | None = None) -> pd.DataFrame:
    """Build a portfolio for every rebalance date."""

    portfolio_config = config.get("portfolio", {})
    selected_method = method or portfolio_config.get("method", "equal_weight")
    frames = []
    for as_of_date, date_scores in scored.groupby("as_of_date", sort=True):
        portfolio = construct_portfolio_for_date(date_scores, config, selected_method)
        portfolio["as_of_date"] = as_of_date
        frames.append(portfolio)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def construct_portfolio_for_date(
    scored_for_date: pd.DataFrame,
    config: dict[str, Any],
    method: str = "equal_weight",
) -> pd.DataFrame:
    """Select top stocks and assign long-only weights."""

    portfolio_config = config.get("portfolio", {})
    top_n = int(portfolio_config.get("top_n", 20))
    max_stock_weight = float(portfolio_config.get("max_single_stock_weight", 0.05))
    max_industry_weight = float(portfolio_config.get("max_industry_weight", 0.25))
    min_liquidity = float(portfolio_config.get("min_liquidity", 5_000_000))

    candidates = scored_for_date.copy()
    candidates = candidates[~candidates["excluded_by_risk_filter"].fillna(False)]
    if "avg_turnover" in candidates.columns:
        candidates = candidates[candidates["avg_turnover"].fillna(0) >= min_liquidity]
    candidates = candidates.sort_values("total_score", ascending=False)

    industry_max_count = max(1, int(np.floor(max_industry_weight / max_stock_weight)))
    selected_rows = []
    industry_counts: dict[str, int] = {}
    for _, row in candidates.iterrows():
        industry = str(row.get("industry", "Unknown"))
        if industry_counts.get(industry, 0) >= industry_max_count:
            continue
        selected_rows.append(row)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected_rows) >= top_n:
            break

    if not selected_rows:
        return pd.DataFrame(columns=list(scored_for_date.columns) + ["weight", "selection_rank", "portfolio_method"])

    selected = pd.DataFrame(selected_rows).copy()
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["portfolio_method"] = method
    selected["weight"] = _assign_weights(selected, method, max_stock_weight, max_industry_weight)
    return selected


def _assign_weights(
    selected: pd.DataFrame,
    method: str,
    max_stock_weight: float,
    max_industry_weight: float,
) -> pd.Series:
    """Assign weights while keeping constraints interpretable."""

    if method == "score_weight":
        scores = selected["total_score"].astype(float)
        shifted = scores - scores.min() + 1e-6
        weights = shifted / shifted.sum()
    else:
        weights = pd.Series(1 / len(selected), index=selected.index)

    weights = _cap_and_redistribute(weights, pd.Series(max_stock_weight, index=selected.index))
    industry_caps = selected.groupby("industry").ngroup()
    for industry in selected["industry"].dropna().unique():
        mask = selected["industry"].eq(industry)
        total = weights.loc[mask].sum()
        if total > max_industry_weight:
            weights.loc[mask] *= max_industry_weight / total
            remaining = 1 - weights.sum()
            if remaining > 1e-12:
                other_mask = ~mask
                per_stock_caps = pd.Series(max_stock_weight, index=selected.index)
                per_stock_caps.loc[mask] = weights.loc[mask]
                weights = _redistribute_to_capacity(weights, per_stock_caps, other_mask, remaining)
    weights = weights / weights.sum()
    weights = _cap_and_redistribute(weights, pd.Series(max_stock_weight, index=selected.index))
    weights.name = "weight"
    return weights


def _cap_and_redistribute(weights: pd.Series, caps: pd.Series) -> pd.Series:
    capped = weights.clip(upper=caps)
    remaining = 1 - capped.sum()
    eligible = capped < caps - 1e-12
    return _redistribute_to_capacity(capped, caps, eligible, remaining)


def _redistribute_to_capacity(
    weights: pd.Series,
    caps: pd.Series,
    eligible: pd.Series,
    remaining: float,
) -> pd.Series:
    result = weights.copy()
    while remaining > 1e-10 and eligible.any():
        capacity = (caps - result).clip(lower=0)
        available = capacity[eligible]
        if available.sum() <= 1e-12:
            break
        addition = remaining * available / available.sum()
        result.loc[eligible] += addition
        result = result.clip(upper=caps)
        remaining = 1 - result.sum()
        eligible = result < caps - 1e-12
    if result.sum() > 0:
        result = result / result.sum()
    return result


def validate_portfolio_constraints(portfolio: pd.DataFrame, config: dict[str, Any]) -> dict[str, bool]:
    """Return constraint checks used by tests and the dashboard."""

    portfolio_config = config.get("portfolio", {})
    max_stock_weight = float(portfolio_config.get("max_single_stock_weight", 0.05))
    max_industry_weight = float(portfolio_config.get("max_industry_weight", 0.25))
    checks = {}
    for as_of_date, holdings in portfolio.groupby("as_of_date"):
        prefix = str(pd.Timestamp(as_of_date).date())
        checks[f"{prefix}_weights_sum_to_one"] = bool(np.isclose(holdings["weight"].sum(), 1.0, atol=1e-6))
        checks[f"{prefix}_single_stock_cap"] = bool(holdings["weight"].max() <= max_stock_weight + 1e-6)
        industry_weights = holdings.groupby("industry")["weight"].sum()
        checks[f"{prefix}_industry_cap"] = bool(industry_weights.max() <= max_industry_weight + 1e-6)
        checks[f"{prefix}_long_only"] = bool(holdings["weight"].min() >= -1e-12)
    return checks
