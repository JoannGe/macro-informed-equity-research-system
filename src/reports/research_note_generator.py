"""Rule-based research notes grounded in model inputs."""

from __future__ import annotations

import pandas as pd


def generate_research_notes(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Generate one deterministic note for each selected stock."""

    if portfolio.empty:
        return pd.DataFrame(columns=["as_of_date", "stock_code", "stock_name", "research_note"])
    rows = []
    for _, row in portfolio.iterrows():
        rows.append(
            {
                "as_of_date": row["as_of_date"],
                "stock_code": row["stock_code"],
                "stock_name": row.get("stock_name", row["stock_code"]),
                "industry": row.get("industry", "Unknown"),
                "research_note": build_research_note(row),
            }
        )
    return pd.DataFrame(rows)


def build_research_note(row: pd.Series) -> str:
    """Build a cautious screening note from a selected stock row."""

    company = row.get("stock_name", row.get("stock_code", "This firm"))
    code = row.get("stock_code", "")
    return "\n".join(
        [
            f"{company} ({code})",
            f"The model ranks this firm highly because its total score is {row.get('total_score', 0):.2f}.",
            f"Macro reason: {_score_sentence('industry transmission score', row.get('industry_score'))}",
            f"Industry reason: {row.get('industry', 'Unknown')} receives a macro-linked score from the editable transmission matrix.",
            f"Company fundamental reason: {_score_sentence('quality score', row.get('quality_score'))} {_score_sentence('growth score', row.get('growth_score'))}",
            f"Valuation reason: {_score_sentence('valuation score', row.get('valuation_score'))}",
            f"Momentum or market-confirmation reason: {_score_sentence('momentum score', row.get('momentum_score'))}",
            f"Key risks: {_risk_sentence(row)}",
            f"Data limitations: {_data_limitation_sentence(row)}",
            "This should be interpreted as a screening result, not an investment recommendation.",
        ]
    )


def _score_sentence(label: str, value: object) -> str:
    if pd.isna(value):
        return f"No usable {label} was available."
    numeric = float(value)
    if numeric > 0.5:
        direction = "is materially above the cross-sectional average"
    elif numeric > 0:
        direction = "is modestly above the cross-sectional average"
    elif numeric < -0.5:
        direction = "is materially below the cross-sectional average"
    elif numeric < 0:
        direction = "is modestly below the cross-sectional average"
    else:
        direction = "is near the cross-sectional average"
    return f"The {label} is {numeric:.2f}, which {direction}."


def _risk_sentence(row: pd.Series) -> str:
    reasons = []
    if row.get("risk_filter_reason"):
        reasons.append(str(row.get("risk_filter_reason")))
    if pd.notna(row.get("risk_score")):
        reasons.append(f"risk score {float(row.get('risk_score')):.2f}")
    if pd.notna(row.get("volatility")):
        reasons.append(f"annualized volatility {float(row.get('volatility')):.1%}")
    if not reasons:
        return "No explicit risk warning was generated from available inputs."
    return "; ".join(reasons) + "."


def _data_limitation_sentence(row: pd.Series) -> str:
    warning = row.get("data_quality_warning")
    if warning:
        return str(warning)
    if not bool(row.get("data_is_point_in_time", False)):
        return "Point-in-time fundamentals were unavailable for this rebalance date."
    return "No key data-quality limitation was flagged by the demo pipeline."
