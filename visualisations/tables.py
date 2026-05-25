"""
visualisations/tables.py

Formatted table components for the dashboard.

Rationale: tables complement charts by showing exact values that are
hard to read precisely from a visual. A good dashboard uses both —
charts for patterns and trends, tables for precise values and ranking.
Keeping table formatting here means page files stay clean.
"""

import pandas as pd
import numpy as np
import streamlit as st


def style_screener_table(df: pd.DataFrame) -> pd.DataFrame.style:
    """
    Apply conditional formatting to the screener table.

    Rationale: colour coding makes a table scannable at a glance.
    Green YTD returns immediately draw the eye to outperformers,
    red to underperformers — the same principle used in Bloomberg
    Terminal and Excel financial models.

    Parameters:
        df: screener DataFrame from analytics/screener.py

    Returns:
        Styled pandas DataFrame ready for st.dataframe()
    """

    def colour_return(val):
        """
        Colour a return value green if positive, red if negative.
        Handles percentage strings like "5.23%" and "N/A".
        """
        if val == "N/A" or pd.isna(val):
            return "color: #94A3B8"  # grey for missing
        try:
            numeric = float(str(val).replace("%", ""))
            if numeric > 0:
                return "color: #10B981; font-weight: bold"   # green
            elif numeric < 0:
                return "color: #EF4444; font-weight: bold"   # red
            else:
                return "color: #F1F5F9"
        except ValueError:
            return "color: #94A3B8"

    def colour_sharpe(val):
        """
        Colour Sharpe ratio:
            > 1.0  → green  (good risk-adjusted return)
            0-1.0  → amber  (acceptable)
            < 0    → red    (not beating risk-free rate)
        """
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "color: #94A3B8"
        try:
            numeric = float(val)
            if numeric > 1.0:
                return "color: #10B981; font-weight: bold"
            elif numeric >= 0:
                return "color: #F59E0B"
            else:
                return "color: #EF4444"
        except (ValueError, TypeError):
            return "color: #94A3B8"

    return (
        df.style
        .applymap(colour_return, subset=["YTD Return", "Period Return", "Annual Return"])
        .applymap(colour_sharpe, subset=["Sharpe Ratio"])
        .set_properties(**{
            "background-color": "#1E293B",
            "color":            "#F1F5F9",
            "border-color":     "#0F172A",
        })
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#0F172A"),
                ("color",            "#F1F5F9"),
                ("font-weight",      "bold"),
                ("border-bottom",    "1px solid #2563EB"),
            ]
        }])
        .format(na_rep="N/A")
    )


def render_metric_cards(metrics: dict) -> None:
    """
    Render a row of metric cards using Streamlit columns.

    Rationale: metric cards (large number + label) give users the
    headline figures at a glance before they engage with the charts.
    This follows the F-pattern of reading — people scan across the
    top of a page first. Critical numbers should live there.

    Parameters:
        metrics: dict of {label: (value, delta)} where delta is
                 optional change from previous period shown as
                 a small up/down indicator below the main value.

    Example:
        render_metric_cards({
            "Total Return":   ("12.5%",  "+2.1%"),
            "Sharpe Ratio":   ("1.34",   None),
            "Volatility":     ("18.2%",  "-0.5%"),
        })
    """
    cols = st.columns(len(metrics))

    for col, (label, value_delta) in zip(cols, metrics.items()):
        value = value_delta[0] if isinstance(value_delta, tuple) else value_delta
        delta = value_delta[1] if isinstance(value_delta, tuple) else None

        with col:
            st.metric(
                label=label,
                value=value,
                delta=delta,
            )


def render_optimal_weights_table(optimal: dict) -> None:
    """
    Render the optimal portfolio weights as a clean formatted table.

    Rationale: the efficient frontier chart shows WHERE the optimal
    portfolio sits in risk/return space, but doesn't tell you HOW
    to build it. This table answers that — exactly how much to
    allocate to each stock for the maximum Sharpe ratio portfolio.

    Weights below 1% are excluded as they are economically negligible
    and clutter the table.
    """
    weights = optimal.get("weights", {})

    rows = []
    for ticker, weight in weights.items():
        if weight >= 0.01:  # exclude negligible weights
            from data.fetcher import STOCKS
            rows.append({
                "Stock":      STOCKS.get(ticker, {}).get("name", ticker),
                "Ticker":     ticker,
                "Weight":     f"{weight * 100:.1f}%",
                "Weight_num": weight,
            })

    if not rows:
        st.info("No significant weights found in optimal portfolio.")
        return

    df = (
        pd.DataFrame(rows)
        .sort_values("Weight_num", ascending=False)
        .drop(columns="Weight_num")
        .reset_index(drop=True)
    )

    df.insert(0, "Rank", range(1, len(df) + 1))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # Summary metrics below the table
    st.markdown(
        f"""
        <div style='display:flex; gap:2rem; margin-top:0.5rem;'>
            <div>
                <span style='color:#94A3B8; font-size:12px;'>
                    EXPECTED ANNUAL RETURN
                </span><br>
                <span style='color:#10B981; font-size:18px; font-weight:bold;'>
                    {optimal['return']*100:.2f}%
                </span>
            </div>
            <div>
                <span style='color:#94A3B8; font-size:12px;'>
                    EXPECTED VOLATILITY
                </span><br>
                <span style='color:#F59E0B; font-size:18px; font-weight:bold;'>
                    {optimal['volatility']*100:.2f}%
                </span>
            </div>
            <div>
                <span style='color:#94A3B8; font-size:12px;'>
                    SHARPE RATIO
                </span><br>
                <span style='color:#3B82F6; font-size:18px; font-weight:bold;'>
                    {optimal['sharpe']:.2f}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_movers_table(
    gainers: pd.DataFrame,
    losers:  pd.DataFrame
) -> None:
    """
    Render gainers and losers side by side in two columns.

    Rationale: side-by-side layout mirrors how financial dashboards
    (Bloomberg, Reuters Eikon) present daily movers — you scan left
    for opportunities, right to check if you hold any losers.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h4 style='color:#10B981;'>▲ Top Gainers</h4>",
            unsafe_allow_html=True
        )
        for _, row in gainers.iterrows():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between;"
                f"padding:4px 0; border-bottom:1px solid #1E293B;'>"
                f"<span style='color:#F1F5F9;'>{row['Name']}</span>"
                f"<span style='color:#10B981; font-weight:bold;'>"
                f"{row['Return']}</span></div>",
                unsafe_allow_html=True
            )

    with col2:
        st.markdown(
            "<h4 style='color:#EF4444;'>▼ Top Losers</h4>",
            unsafe_allow_html=True
        )
        for _, row in losers.iterrows():
            st.markdown(
                f"<div style='display:flex; justify-content:space-between;"
                f"padding:4px 0; border-bottom:1px solid #1E293B;'>"
                f"<span style='color:#F1F5F9;'>{row['Name']}</span>"
                f"<span style='color:#EF4444; font-weight:bold;'>"
                f"{row['Return']}</span></div>",
                unsafe_allow_html=True
            )
            