"""Tests for point-in-time discipline."""

from __future__ import annotations

import pandas as pd

from src.pipeline.run_all import run_pipeline


def test_demo_pipeline_runs_end_to_end() -> None:
    outputs = run_pipeline(mode="demo")
    assert not outputs["stock_scores"].empty
    assert not outputs["portfolio"].empty
    assert not outputs["performance_summary"].empty


def test_fundamentals_are_announced_before_selection_date() -> None:
    outputs = run_pipeline(mode="demo")
    panel = outputs["rebalance_panel"].dropna(subset=["announcement_date"])
    assert (pd.to_datetime(panel["announcement_date"]) <= pd.to_datetime(panel["as_of_date"])).all()


def test_market_features_use_only_past_price_observations() -> None:
    outputs = run_pipeline(mode="demo")
    panel = outputs["rebalance_panel"].dropna(subset=["price_observation_date"])
    assert (pd.to_datetime(panel["price_observation_date"]) <= pd.to_datetime(panel["as_of_date"])).all()
