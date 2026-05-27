# Macro-Informed Equity Research Platform

This project is an educational investment research system and paper-trading dashboard. It is designed to help study macro-industry-company investment logic for medium- to long-term A-share screening.

It is not financial advice. It does not connect to broker APIs, place orders, or create automatic trade execution.

## Project Purpose

The platform combines:

1. Macro regime classification.
2. Editable industry transmission scoring.
3. Firm-level fundamental factor scoring.
4. Risk filtering.
5. Long-only portfolio construction.
6. Quarterly backtesting with transaction costs.
7. Streamlit dashboard.
8. Deterministic research-note generation.

The first supported market universe is A-shares. The benchmark is the ChiNext 50 Index, code `399673`.

## System Architecture

```text
config/
  config.yaml                         Main weights, lags, constraints, costs
  industry_transmission_matrix.yaml   Editable macro-to-industry assumptions
data/
  demo/                               Generated tiny synthetic demo data
  raw/                                Local raw data fallback, ignored by git
  processed/                          Pipeline outputs, ignored by git
src/
  data_fetchers/                      Demo/live/local fallback loaders
  processing/                         Cleaning and point-in-time alignment
  factors/                            Macro, industry, firm, and risk factors
  model/                              Stock scoring and portfolio construction
  backtest/                           Backtester and performance metrics
  reports/                            Rule-based research notes
  dashboard/                          Streamlit app
  pipeline/                           End-to-end runner
tests/                                Bias, factor, and portfolio tests
notebooks/                            Starter research notebooks
```

## Installation

Use Python 3.11 or newer.

```bash
cd investment_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Demo Mode

Demo mode does not require API keys. It creates a small synthetic dataset under `data/demo/` and writes processed outputs under `data/processed/`.

```bash
python -m src.pipeline.run_all --mode demo
```

## Run Live/Research Mode

Live mode is conservative. It looks for local CSV fallbacks first and falls back to demo data if real sources are unavailable. It does not invent missing real observations.

```bash
python -m src.pipeline.run_all --mode live
```

Expected optional CSV files are configured in `config/config.yaml`:

- `data/raw/macro.csv`
- `data/raw/prices.csv`
- `data/raw/benchmark.csv`
- `data/raw/fundamentals.csv`
- `data/raw/universe.csv`

## Run The Dashboard

```bash
streamlit run src/dashboard/app.py
```

If processed outputs do not exist, the dashboard sidebar can run the demo pipeline.

## Run Tests

```bash
pytest
```

Tests check that demo mode runs end to end, fundamentals are announced before use, price features use past observations, valuation signs are handled correctly, and portfolio weights respect long-only position and industry constraints.

## API Keys And Environment Variables

Copy `.env.example` to `.env` and add local settings there. Do not commit `.env`.

```bash
cp .env.example .env
```

Current live mode is intentionally fallback-first because free data APIs can be unstable. Future connectors can extend `src/data_fetchers/` while preserving the same processed schema.

## Model Summary

Macro regimes are classified using trailing rolling z-scores for growth, inflation, rates, credit, liquidity, commodity, policy, and risk-appetite proxies.

Industry scores use:

```text
industry_score = macro_regime_vector x industry_sensitivity_vector
```

The matrix is manually editable in `config/industry_transmission_matrix.yaml` because these economic assumptions require human review.

The stock score is:

```text
total_score =
    w_macro_industry * industry_score
  + w_quality * quality_score
  + w_growth * growth_score
  + w_valuation * valuation_score
  + w_momentum * momentum_score
  - w_risk * risk_score
```

All weights are stored in `config/config.yaml`.

## Data-Source Limitations

Free A-share data sources such as AKShare and BaoStock can change endpoints, rate-limit requests, revise historical data, or have missing fields. This project therefore includes a demo mode and local CSV fallback path.

Demo data is synthetic. It is only for testing the pipeline and dashboard. It is not evidence about real A-share returns, fundamentals, or macro relationships.

Raw financial datasets, cache files, local `.env` files, and large local data files are ignored by git.

## Methodological Risks

- Look-ahead bias: the system uses announcement dates and trailing prices, but every new data connector should be tested.
- Survivorship bias: a current-index universe may exclude delisted or removed companies.
- Overfitting: factor weights and industry assumptions can be tuned too closely to history.
- Transaction costs: default cost is 10 basis points per trade, but real costs vary by market conditions and account type.
- Missing data: the system flags missing key fields and does not silently fill important fundamentals with arbitrary values.
- API instability: free data APIs may fail or change schemas.
- Regime shifts: historical macro-industry relationships may stop working.

## Future Improvements

- Add robust AKShare and BaoStock live connectors with schema validation.
- Add CSI 300 or CSI 800 constituent history with survivorship-bias controls.
- Add richer China macro data sources and explicit release calendars.
- Add company filing availability calendars.
- Add portfolio cash handling when too few stocks pass constraints.
- Add factor attribution and scenario analysis.
- Add more notebook examples for data quality and factor diagnostics.
