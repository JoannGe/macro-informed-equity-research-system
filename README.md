# Macro-Informed Equity Research Platform V2

This project is an educational A-share equity research and paper-trading dashboard. It helps study how macro context, industry exposure, and accepted factor-investing concepts can be combined into a transparent stock-screening workflow.

It is not financial advice. It does not connect to broker APIs, place orders, or create automatic trade execution.

## What The System Does

V2 builds a factor-based screening model, constructs a long-only paper portfolio, runs a quarterly backtest, and displays the results in Streamlit.

The benchmark is configurable and defaults to the ChiNext 50 Index, code `399673`.

## V2 Factor-Investing Basis

The default model is inspired by broadly studied factor-investing frameworks. It does not claim that factors guarantee returns.

Supported factor groups:

- Value: earnings yield, book-to-market, and sales yield from PE, PB, and PS.
- Momentum: 6-month price momentum and 12-month momentum excluding the most recent month when data allows.
- Quality / profitability: ROE, ROA, gross margin, and operating margin.
- Investment / asset growth: lower aggressive asset and capex growth is treated as more disciplined by default.
- Low volatility / risk: realized volatility and recent drawdown.
- Liquidity: average trading amount, mainly as tradability control.
- Industry context: editable macro-to-industry transmission matrix.
- Risk penalty: leverage, volatility, missing data, and cash-flow risk.

Composite score:

```text
total_score =
    w_value * value_score
  + w_momentum * momentum_score
  + w_quality * quality_score
  + w_investment * investment_score
  + w_low_volatility * low_volatility_score
  + w_liquidity * liquidity_score
  + w_industry * industry_score
  - w_risk_penalty * risk_penalty_score
```

All weights live in `config/config.yaml`.

## Data Sources

Modes:

- Demo mode: deterministic synthetic data for testing and learning.
- Live mode: tries free public APIs and falls back safely if allowed.

Live source order:

1. AKShare.
2. BaoStock.
3. Demo fallback if `data.allow_demo_fallback: true`.

Optional:

- Tushare Pro is optional and only considered if `TUSHARE_TOKEN` is configured. It is not required for basic V2.

V2 writes standardized internal tables under `data/processed/`:

- `prices.csv`
- `fundamentals.csv`
- `industry.csv`
- `benchmark.csv`
- `data_status.json`

If free APIs lack a field, the system leaves it missing and creates a warning. It does not invent real financial values.

## How `end_date: "today"` Works

`config/config.yaml` supports:

```yaml
data:
  start_date: "2020-01-01"
  end_date: "today"
  mode: "live"
  allow_demo_fallback: true
```

For live mode, `today` resolves at runtime to the current date. Market data then uses the latest available trading day returned by the API. If today is not a trading day, the dashboard reports the actual latest data date.

For demo mode, the synthetic data remains deterministic for tests.

## Portfolio Construction

The default portfolio construction is intentionally beginner-readable:

1. Start with the stock universe.
2. Apply data-availability filters.
3. Apply liquidity filter.
4. Apply risk filters.
5. Rank remaining stocks by `total_score`.
6. Select top N stocks.
7. Assign initial equal weights by default.
8. Optionally allow score-weighted weights.
9. Apply maximum single-stock weight.
10. Apply maximum industry weight if feasible.
11. Normalize final weights.
12. Output diagnostics.

Outputs:

- `selected_portfolio.csv`
- `portfolio_diagnostics.json`

Each selected stock includes factor scores, initial and final weights, selection reason, and key risk warning.

## Installation

Use Python 3.11 or newer.

```bash
cd investment_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demo Mode

```bash
python -m src.pipeline.run_all --mode demo
```

## Run Live Mode

```bash
python -m src.pipeline.run_all --mode live
```

Live API calls can be unstable in restricted environments. If AKShare and BaoStock fail and demo fallback is enabled, the system logs warnings and writes them to `data/processed/data_status.json`.

## Run The Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard pages:

- Overview
- Factor Model
- Data Status
- Stock Ranking
- Portfolio
- Backtest
- Research Notes
- Methodology & Limitations

## Run Tests

```bash
pytest
```

Tests cover demo mode, safe live fallback behavior, `end_date = "today"`, standardized schema creation, factor scores, portfolio constraints, diagnostics, data status, and absence of broker/order-execution code.

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Optional settings:

```text
AKSHARE_ENABLED=true
BAOSTOCK_ENABLED=true
TUSHARE_TOKEN=
```

Never commit `.env`, API keys, tokens, passwords, private data, raw downloaded datasets, or large local caches.

## Methodological Limitations

- Factor returns are not guaranteed.
- Look-ahead bias can still appear if new data connectors ignore release timing.
- Survivorship bias can remain if the available universe excludes delisted or removed companies.
- Overfitting can occur when factor weights are tuned to history.
- Transaction costs and liquidity constraints are simplified.
- Free APIs can be delayed, incomplete, unstable, or revised.
- Missing fundamentals are flagged rather than filled with invented values.
- Macro regimes and industry transmission assumptions can shift over time.

## Not Financial Advice

This is a research and education system only. It is not an order-execution system, does not connect to brokers, and should not be used as automatic trading advice.
