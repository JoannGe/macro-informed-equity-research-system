"""Tests for interpretable factor calculations."""

from __future__ import annotations

import pandas as pd

from src.factors.firm_factors import calculate_firm_factor_scores
from src.factors.industry_score import calculate_industry_scores


def test_valuation_sign_is_reversed_after_cleaning() -> None:
    panel = pd.DataFrame(
        {
            "as_of_date": [pd.Timestamp("2024-03-31")] * 3,
            "stock_code": ["A", "B", "C"],
            "pe": [10.0, 20.0, 40.0],
            "pb": [1.0, 2.0, 4.0],
            "ps": [1.0, 2.0, 4.0],
            "dividend_yield": [0.02, 0.02, 0.02],
            "roe": [0.1, 0.1, 0.1],
            "roa": [0.05, 0.05, 0.05],
            "gross_margin": [0.3, 0.3, 0.3],
            "operating_margin": [0.1, 0.1, 0.1],
            "ocf_to_revenue": [0.1, 0.1, 0.1],
            "fcf_to_revenue": [0.05, 0.05, 0.05],
            "revenue_growth": [0.1, 0.1, 0.1],
            "net_profit_growth": [0.1, 0.1, 0.1],
            "operating_cash_flow_growth": [0.1, 0.1, 0.1],
            "momentum_6m": [0.1, 0.1, 0.1],
            "momentum_12m_ex_1m": [0.1, 0.1, 0.1],
            "avg_turnover": [10_000_000, 10_000_000, 10_000_000],
            "volatility": [0.2, 0.2, 0.2],
            "debt_to_assets": [0.3, 0.3, 0.3],
            "debt_to_equity": [1.0, 1.0, 1.0],
            "interest_coverage": [5.0, 5.0, 5.0],
            "current_ratio": [1.5, 1.5, 1.5],
            "operating_cash_flow": [100.0, 100.0, 100.0],
        }
    )
    scored = calculate_firm_factor_scores(panel, {"factor_processing": {}})
    cheapest = scored.loc[scored["stock_code"].eq("A"), "valuation_score"].iloc[0]
    most_expensive = scored.loc[scored["stock_code"].eq("C"), "valuation_score"].iloc[0]
    assert cheapest > most_expensive


def test_industry_score_is_dot_product_average() -> None:
    macro = pd.DataFrame(
        [
            {
                "as_of_date": pd.Timestamp("2024-03-31"),
                "vector_growth": 1,
                "vector_inflation": -1,
                "vector_interest_rates": 1,
                "vector_credit": -1,
                "vector_liquidity": 1,
                "vector_commodity_prices": 1,
                "vector_policy_sensitivity": -1,
                "vector_risk_appetite": 1,
            }
        ]
    )
    matrix = pd.DataFrame(
        [
            {
                "industry": "Test",
                "growth": 1,
                "inflation": 1,
                "interest_rates": 1,
                "credit": 1,
                "liquidity": 1,
                "commodity_prices": 1,
                "policy_sensitivity": 1,
                "risk_appetite": 1,
            }
        ]
    )
    scores = calculate_industry_scores(macro, matrix)
    assert scores["industry_score"].iloc[0] == 0.25
