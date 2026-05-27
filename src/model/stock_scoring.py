"""Composite stock scoring model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.factors.risk_filters import apply_risk_filters


def score_stocks(
    factor_scores: pd.DataFrame,
    industry_scores: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Combine macro-industry and firm factor scores into a total score."""

    weights = config.get("scoring_weights", {})
    result = factor_scores.merge(
        industry_scores[["as_of_date", "industry", "industry_score"]],
        on=["as_of_date", "industry"],
        how="left",
    )
    for column in [
        "industry_score",
        "quality_score",
        "growth_score",
        "valuation_score",
        "momentum_score",
        "risk_score",
    ]:
        result[column] = result[column].fillna(0.0)

    result["total_score"] = (
        float(weights.get("w_macro_industry", 0.20)) * result["industry_score"]
        + float(weights.get("w_quality", 0.20)) * result["quality_score"]
        + float(weights.get("w_growth", 0.15)) * result["growth_score"]
        + float(weights.get("w_valuation", 0.15)) * result["valuation_score"]
        + float(weights.get("w_momentum", 0.15)) * result["momentum_score"]
        - float(weights.get("w_risk", 0.15)) * result["risk_score"]
    )
    result = apply_risk_filters(result, config)
    return result.sort_values(["as_of_date", "total_score"], ascending=[True, False]).reset_index(drop=True)


def latest_rankings(scored: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent stock ranking table."""

    latest_date = scored["as_of_date"].max()
    latest = scored[scored["as_of_date"] == latest_date].copy()
    latest["rank"] = latest["total_score"].rank(ascending=False, method="first").astype(int)
    return latest.sort_values("rank")
