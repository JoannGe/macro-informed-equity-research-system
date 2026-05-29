"""V2 tests for data routing, schema, factors, and safety boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_fetchers import data_source_router
from src.pipeline.run_all import run_pipeline
from src.utils.date_utils import requested_date_range
from src.utils.io_utils import processed_path


def test_end_date_today_is_dynamic_for_live_and_reproducible_for_demo() -> None:
    config = {"data": {"start_date": "2020-01-01", "end_date": "today"}}
    _, live_end, live_label = requested_date_range(config, "live")
    _, demo_end, demo_label = requested_date_range(config, "demo")
    assert live_label == "today"
    assert demo_label == "today"
    assert live_end == pd.Timestamp.today().normalize()
    assert demo_end == pd.Timestamp("2025-12-31")


def test_standardized_schema_and_json_outputs_are_created() -> None:
    outputs = run_pipeline(mode="demo")
    config = {"data": {"processed_dir": "data/processed"}}
    for name, required_columns in {
        "prices": {"date", "ticker", "open", "high", "low", "close", "volume", "amount", "return"},
        "fundamentals": {"date", "ticker", "pe", "pb", "ps", "market_cap"},
        "industry": {"ticker", "industry"},
        "benchmark": {"date", "benchmark_code", "close", "return"},
    }.items():
        frame = pd.read_csv(processed_path(f"{name}.csv", config))
        assert required_columns.issubset(frame.columns)
    assert processed_path("data_status.json", config).exists()
    assert processed_path("portfolio_diagnostics.json", config).exists()
    assert "value_score" in outputs["stock_scores"].columns
    assert "risk_penalty_score" in outputs["stock_scores"].columns


def test_selected_portfolio_contains_explanation_fields() -> None:
    run_pipeline(mode="demo")
    config = {"data": {"processed_dir": "data/processed"}}
    portfolio = pd.read_csv(processed_path("selected_portfolio.csv", config))
    required = {
        "ticker",
        "industry",
        "total_score",
        "value_score",
        "momentum_score",
        "quality_score",
        "investment_score",
        "low_volatility_score",
        "liquidity_score",
        "industry_score",
        "risk_penalty_score",
        "initial_weight",
        "final_weight",
        "selection_reason",
        "key_risk_warning",
    }
    assert required.issubset(portfolio.columns)
    assert portfolio["selection_reason"].str.len().gt(0).all()


def test_live_mode_falls_back_safely_when_apis_fail(monkeypatch) -> None:
    def fail_fetcher(*args, **kwargs):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr(data_source_router, "fetch_akshare_data", fail_fetcher)
    monkeypatch.setattr(data_source_router, "fetch_baostock_data", fail_fetcher)
    outputs = run_pipeline(mode="live")
    status = json.loads(Path("data/processed/data_status.json").read_text(encoding="utf-8"))
    assert status["fallback_used"] is True
    assert status["data_source"] == "demo"
    assert not outputs["stock_scores"].empty


def test_no_broker_or_order_execution_code_exists() -> None:
    repo = Path(__file__).resolve().parents[1]
    banned_terms = [
        "place_order",
        "submit_order",
        "broker_api",
        "brokerage_api",
        "buy_order",
        "sell_order",
        "order_execution",
    ]
    scanned = []
    for path in (repo / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        scanned.append(path)
        for term in banned_terms:
            assert term not in text, f"Found forbidden trading term {term} in {path}"
    assert scanned
