"""Tests for portfolio construction constraints."""

from __future__ import annotations

import pandas as pd

from src.model.portfolio_construction import construct_portfolio_for_date, validate_portfolio_constraints


def test_portfolio_weights_and_caps_are_respected() -> None:
    config = {
        "portfolio": {
            "top_n": 20,
            "max_single_stock_weight": 0.05,
            "max_industry_weight": 0.25,
            "min_liquidity": 0,
            "method": "equal_weight",
        }
    }
    industries = ["A", "B", "C", "D"]
    rows = []
    for idx in range(40):
        rows.append(
            {
                "as_of_date": pd.Timestamp("2024-03-31"),
                "stock_code": f"S{idx:02d}",
                "industry": industries[idx % len(industries)],
                "total_score": 100 - idx,
                "avg_turnover": 10_000_000,
                "excluded_by_risk_filter": False,
            }
        )
    portfolio = construct_portfolio_for_date(pd.DataFrame(rows), config, method="equal_weight")
    checks = validate_portfolio_constraints(portfolio, config)
    assert len(portfolio) == 20
    assert all(checks.values())


def test_risk_excluded_stocks_are_not_selected() -> None:
    config = {
        "portfolio": {
            "top_n": 20,
            "max_single_stock_weight": 0.05,
            "max_industry_weight": 0.25,
            "min_liquidity": 0,
            "method": "equal_weight",
        }
    }
    rows = []
    for idx in range(25):
        rows.append(
            {
                "as_of_date": pd.Timestamp("2024-03-31"),
                "stock_code": f"S{idx:02d}",
                "industry": f"I{idx % 5}",
                "total_score": 100 - idx,
                "avg_turnover": 10_000_000,
                "excluded_by_risk_filter": idx == 0,
            }
        )
    portfolio = construct_portfolio_for_date(pd.DataFrame(rows), config, method="equal_weight")
    assert "S00" not in portfolio["stock_code"].tolist()
