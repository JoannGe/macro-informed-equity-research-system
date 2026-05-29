"""End-to-end pipeline entry point.

Run from the ``investment_system`` directory:

    python -m src.pipeline.run_all --mode demo
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from src.backtest.backtester import run_backtest
from src.data_fetchers.data_source_router import load_research_data
from src.data_fetchers.fundamentals import load_fundamentals
from src.data_fetchers.industry_data import load_industry_transmission_matrix, load_universe
from src.data_fetchers.macro_data import load_macro_data
from src.data_fetchers.market_data import load_benchmark_data, load_price_data
from src.factors.factor_pipeline import calculate_factor_scores
from src.factors.industry_score import calculate_industry_scores, explain_industry_score
from src.factors.macro_regime import build_macro_regime_history
from src.model.portfolio_construction import construct_portfolios, portfolio_diagnostics_for_latest
from src.model.stock_scoring import score_stocks
from src.processing.align_point_in_time import build_rebalance_panel
from src.processing.build_demo_data import ensure_demo_data
from src.processing.clean_prices import get_rebalance_dates
from src.reports.research_note_generator import generate_research_notes
from src.utils.io_utils import ensure_directories, load_config, processed_path, write_dataframe
from src.utils.logging_utils import setup_logging


def run_pipeline(mode: str = "demo", config_path: str = "config/config.yaml") -> dict[str, pd.DataFrame]:
    """Run the full research pipeline and save processed outputs."""

    if mode not in {"demo", "live"}:
        raise ValueError("mode must be either 'demo' or 'live'")

    logger = setup_logging(logging.INFO)
    config = load_config(config_path)
    ensure_directories(config)
    logger.info("Loading research data through V2 data-source router.")
    data, data_status = load_research_data(mode, config, logger)
    macro = data.get("macro", pd.DataFrame())
    prices = data["prices"]
    benchmark = data["benchmark"]
    fundamentals = data["fundamentals"]
    universe = data["universe"]
    industry_matrix = load_industry_transmission_matrix()

    rebalance_dates = get_rebalance_dates(prices, config.get("portfolio", {}).get("rebalance_frequency", "Q"))
    reporting_lag_days = int(config.get("data", {}).get("reporting_lag_days", 75))
    logger.info("Building point-in-time panel for %s rebalance dates.", len(rebalance_dates))
    panel = build_rebalance_panel(universe, fundamentals, prices, rebalance_dates, reporting_lag_days)
    macro_regimes = _build_macro_or_neutral_regimes(macro, rebalance_dates, config)
    industry_scores = calculate_industry_scores(macro_regimes, industry_matrix)
    industry_scores["industry_explanation"] = industry_scores.apply(explain_industry_score, axis=1)
    firm_scores = calculate_factor_scores(panel, config)
    scored = score_stocks(firm_scores, industry_scores, config)
    portfolio = construct_portfolios(scored, config)
    notes = generate_research_notes(portfolio)
    backtest = run_backtest(prices, benchmark, portfolio, config)

    metadata = _build_metadata(mode, config, scored, portfolio, data_status)
    portfolio_diagnostics = portfolio_diagnostics_for_latest(portfolio)
    outputs: dict[str, pd.DataFrame] = {
        "macro_regimes": macro_regimes,
        "industry_scores": industry_scores,
        "rebalance_panel": panel,
        "firm_scores": firm_scores,
        "stock_scores": scored,
        "portfolio": portfolio,
        "selected_portfolio": portfolio,
        "research_notes": notes,
        "metadata": metadata,
        **backtest,
    }
    _write_outputs(outputs, config, logger)
    _write_json_outputs(data_status, portfolio_diagnostics, config, logger)
    logger.info("Pipeline complete. Outputs saved under %s.", config.get("data", {}).get("processed_dir"))
    return outputs


def _build_metadata(
    mode: str,
    config: dict[str, Any],
    scored: pd.DataFrame,
    portfolio: pd.DataFrame,
    data_status: dict[str, Any],
) -> pd.DataFrame:
    latest_date = scored["as_of_date"].max() if not scored.empty else pd.NaT
    warnings = []
    if mode == "demo":
        warnings.append("Demo mode uses synthetic data and is not evidence about real A-share returns.")
    if portfolio.empty:
        warnings.append("No portfolio holdings passed the configured filters.")
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "data_source": data_status.get("data_source"),
                "requested_start_date": data_status.get("requested_start_date"),
                "requested_end_date": data_status.get("requested_end_date"),
                "actual_latest_price_date": data_status.get("actual_latest_price_date"),
                "actual_latest_fundamental_date": data_status.get("actual_latest_fundamental_date"),
                "data_freshness": data_status.get("data_freshness"),
                "latest_rebalance_date": latest_date,
                "benchmark_code": config.get("project", {}).get("benchmark_code", "399673"),
                "benchmark_name": config.get("project", {}).get("benchmark_name", "ChiNext 50 Index"),
                "transaction_cost_bps": config.get("portfolio", {}).get("transaction_cost_bps", 10),
                "warnings": " ".join([*warnings, *data_status.get("warnings", [])]),
            }
        ]
    )


def _write_outputs(outputs: dict[str, pd.DataFrame], config: dict[str, Any], logger: logging.Logger) -> None:
    for name, frame in outputs.items():
        if isinstance(frame, pd.DataFrame):
            path = processed_path(f"{name}.csv", config)
            write_dataframe(frame, path)
            logger.info("Wrote %s rows to %s.", len(frame), path)


def _write_json_outputs(
    data_status: dict[str, Any],
    portfolio_diagnostics: dict[str, Any],
    config: dict[str, Any],
    logger: logging.Logger,
) -> None:
    for filename, payload in {
        "data_status.json": data_status,
        "portfolio_diagnostics.json": portfolio_diagnostics,
    }.items():
        path = processed_path(filename, config)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s.", path)


def _build_macro_or_neutral_regimes(
    macro: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    config: dict[str, Any],
) -> pd.DataFrame:
    if not macro.empty:
        return build_macro_regime_history(macro, rebalance_dates, config)
    rows = []
    for date in rebalance_dates:
        rows.append(
            {
                "as_of_date": pd.Timestamp(date),
                "macro_observation_date": pd.NaT,
                "growth_score": 0.0,
                "inflation_score": 0.0,
                "rates_score": 0.0,
                "credit_score": 0.0,
                "liquidity_score": 0.0,
                "commodity_score": 0.0,
                "policy_score": 0.0,
                "risk_appetite_score": 0.0,
                "growth_regime": "neutral",
                "inflation_regime": "neutral",
                "rates_regime": "neutral",
                "credit_regime": "neutral",
                "liquidity_regime": "neutral",
                "vector_growth": 0,
                "vector_inflation": 0,
                "vector_interest_rates": 0,
                "vector_credit": 0,
                "vector_liquidity": 0,
                "vector_commodity_prices": 0,
                "vector_policy_sensitivity": 0,
                "vector_risk_appetite": 0,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the macro-informed equity research pipeline.")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo", help="Data mode to run.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(mode=args.mode, config_path=args.config)


if __name__ == "__main__":
    main()
