"""Industry scoring from macro regime vectors and editable sensitivities."""

from __future__ import annotations

import pandas as pd


VECTOR_COLUMNS = [
    "growth",
    "inflation",
    "interest_rates",
    "credit",
    "liquidity",
    "commodity_prices",
    "policy_sensitivity",
    "risk_appetite",
]


def calculate_industry_scores(
    macro_regime_history: pd.DataFrame,
    industry_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate ``macro_regime_vector x industry_sensitivity_vector``."""

    rows: list[dict[str, object]] = []
    for _, regime in macro_regime_history.iterrows():
        vector = {column: float(regime.get(f"vector_{column}", 0.0)) for column in VECTOR_COLUMNS}
        for _, industry in industry_matrix.iterrows():
            score_components = {
                column: float(industry.get(column, 0.0)) * vector[column] for column in VECTOR_COLUMNS
            }
            rows.append(
                {
                    "as_of_date": regime["as_of_date"],
                    "industry": industry["industry"],
                    "industry_score": sum(score_components.values()) / len(VECTOR_COLUMNS),
                    **{f"component_{key}": value for key, value in score_components.items()},
                }
            )
    return pd.DataFrame(rows)


def explain_industry_score(row: pd.Series) -> str:
    """Create a deterministic explanation based on the largest components."""

    components = {
        column.replace("component_", ""): row[column]
        for column in row.index
        if column.startswith("component_") and pd.notna(row[column])
    }
    if not components:
        return "No industry transmission components were available."
    ranked = sorted(components.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    parts = [f"{name} {value:+.2f}" for name, value in ranked]
    return "Largest macro transmission effects: " + ", ".join(parts) + "."
