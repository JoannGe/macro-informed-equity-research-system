"""End-to-end pipeline entry point.

Run from the ``investment_system`` directory:

    python -m src.pipeline.run_all --mode demo
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from src.backtest.backtester import run_backtest
from src.data_fetchers.fundamentals import load_fundamentals
from src.data_fetchers.industry_data import load_industry_transmission_matrix, load_universe
from src.data_fetchers.macro_data import load_macro_data
from src.data_fetchers.market_data import load_benchmark_data, load_price_data
from src.factors.firm_factors import calculate_firm_factor_scores
from src.factors.industry_score import calculate_industry_scores, explain_industry_score
from src.factors.macro_regime import build_macro_regime_history
from src.model.portfolio_construction import construct_portfolios
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
    if mode == "demo":
        logger.info("Generating/loading deterministic demo data.")
        ensure_demo_data(config)
    else:
        logger.info("Running live/research mode with conservative local fallbacks.")

    macro = load_macro_data(mode, config, logger)
    prices = load_price_data(mode, config, logger)
    benchmark = load_benchmark_data(mode, config, logger)
    fundamentals = load_fundamentals(mode, config, logger)
    universe = load_universe(mode, config, logger)
    industry_matrix = load_industry_transmission_matrix()

    rebalance_dates = get_rebalance_dates(prices, config.get("portfolio", {}).get("rebalance_frequency", "Q"))
    reporting_lag_days = int(config.get("data", {}).get("reporting_lag_days", 75))
    logger.info("Building point-in-time panel for %s rebalance dates.", len(rebalance_dates))
    panel = build_rebalance_panel(universe, fundamentals, prices, rebalance_dates, reporting_lag_days)
    macro_regimes = build_macro_regime_history(macro, rebalance_dates, config)
    industry_scores = calculate_industry_scores(macro_regimes, industry_matrix)
    industry_scores["industry_explanation"] = industry_scores.apply(explain_industry_score, axis=1)
    firm_scores = calculate_firm_factor_scores(panel, config)
    scored = score_stocks(firm_scores, industry_scores, config)
    portfolio = construct_portfolios(scored, config)
    notes = generate_research_notes(portfolio)
    backtest = run_backtest(prices, benchmark, portfolio, config)

    metadata = _build_metadata(mode, config, scored, portfolio)
    outputs: dict[str, pd.DataFrame] = {
        "macro_regimes": macro_regimes,
        "industry_scores": industry_scores,
        "rebalance_panel": panel,
        "firm_scores": firm_scores,
        "stock_scores": scored,
        "portfolio": portfolio,
        "research_notes": notes,
        "metadata": metadata,
        **backtest,
    }
    _write_outputs(outputs, config, logger)
    logger.info("Pipeline complete. Outputs saved under %s.", config.get("data", {}).get("processed_dir"))
    return outputs


def _build_metadata(
    mode: str,
    config: dict[str, Any],
    scored: pd.DataFrame,
    portfolio: pd.DataFrame,
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
                "latest_rebalance_date": latest_date,
                "benchmark_code": config.get("project", {}).get("benchmark_code", "399673"),
                "transaction_cost_bps": config.get("portfolio", {}).get("transaction_cost_bps", 10),
                "warnings": " ".join(warnings),
            }
        ]
    )


def _write_outputs(outputs: dict[str, pd.DataFrame], config: dict[str, Any], logger: logging.Logger) -> None:
    for name, frame in outputs.items():
        if isinstance(frame, pd.DataFrame):
            path = processed_path(f"{name}.csv", config)
            write_dataframe(frame, path)
            logger.info("Wrote %s rows to %s.", len(frame), path)


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
