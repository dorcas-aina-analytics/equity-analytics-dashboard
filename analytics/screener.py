"""
analytics/screener.py

Filters and ranks stocks based on fundamental and technical criteria.

Rationale: a stock screener is the primary tool equity analysts use to
narrow down a universe of thousands of stocks to a shortlist of candidates
worth researching further. Building one demonstrates you understand how
equity research workflows actually operate — not just how to plot prices.
"""

import pandas as pd
import numpy as np
from data.fetcher import STOCKS, fetch_stock_info
from analytics.returns import (
    calculate_daily_returns,
    calculate_ytd_return,
    calculate_period_return,
    calculate_annualised_return
)
from analytics.volatility import (
    calculate_sharpe_ratio,
    calculate_rolling_volatility
)


def build_screener_table(
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Build the master screener table combining fundamental and
    technical metrics for all stocks.

    Rationale: combining fundamental data (P/E ratio, market cap,
    dividend yield) with technical metrics (returns, volatility,
    Sharpe ratio) gives a complete picture of each stock. Analysts
    rarely look at fundamentals or technicals in isolation — they
    use both together to make investment decisions.

    Parameters:
        prices:        DataFrame of closing prices
        daily_returns: DataFrame of daily percentage returns

    Returns:
        DataFrame with one row per stock and all metrics as columns
    """
    rows = []

    # Calculate metrics that apply to all stocks at once
    ytd_returns      = calculate_ytd_return(prices)
    period_returns   = calculate_period_return(prices)
    annualised_rets  = calculate_annualised_return(daily_returns)
    sharpe_ratios    = calculate_sharpe_ratio(daily_returns)

    # Rolling volatility — take the most recent value (latest 30-day vol)
    rolling_vol = calculate_rolling_volatility(daily_returns)
    latest_vol  = rolling_vol.iloc[-1] if not rolling_vol.empty else pd.Series()

    for ticker in prices.columns:
        # Fetch fundamental data from Yahoo Finance for each stock
        # Rationale: fundamental data requires a separate API call per
        # stock — we fetch it here and combine with calculated metrics
        info = fetch_stock_info(ticker)

        row = {
            "Ticker":           ticker,
            "Name":             info.get("name",           STOCKS.get(ticker, {}).get("name", ticker)),
            "Sector":           info.get("sector",         STOCKS.get(ticker, {}).get("sector", "Unknown")),
            "Current Price":    info.get("current_price",  None),
            "Market Cap (£bn)": _format_market_cap(info.get("market_cap", None)),
            "P/E Ratio":        info.get("pe_ratio",       None),
            "Dividend Yield":   _format_pct(info.get("dividend_yield", None)),
            "52W High":         info.get("week_52_high",   None),
            "52W Low":          info.get("week_52_low",    None),
            "YTD Return":       _format_pct(ytd_returns.get(ticker,     None)),
            "Period Return":    _format_pct(period_returns.get(ticker,  None)),
            "Annual Return":    _format_pct(annualised_rets.get(ticker, None)),
            "Sharpe Ratio":     round(sharpe_ratios.get(ticker, np.nan), 2)
                                if pd.notna(sharpe_ratios.get(ticker, np.nan)) else None,
            "Volatility":       _format_pct(latest_vol.get(ticker,      None)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def filter_screener(
    df: pd.DataFrame,
    sector:           str   = "All",
    min_sharpe:       float = None,
    max_pe:           float = None,
    min_dividend:     float = None,
    min_ytd_return:   float = None,
) -> pd.DataFrame:
    """
    Apply filters to the screener table.

    Rationale: filtering is how analysts narrow down a stock universe.
    A typical workflow: "show me all Financial sector stocks with a
    P/E ratio below 15, dividend yield above 3%, and positive YTD return."
    Each filter independently reduces the candidate list.

    Parameters:
        df:             the full screener DataFrame from build_screener_table
        sector:         filter by sector name, "All" returns all sectors
        min_sharpe:     minimum Sharpe ratio threshold
        max_pe:         maximum P/E ratio threshold
                        Rationale: low P/E suggests a stock may be
                        undervalued relative to its earnings
        min_dividend:   minimum dividend yield as a percentage
                        Rationale: income investors prioritise dividend yield
        min_ytd_return: minimum YTD return as a percentage
                        Rationale: momentum investors prefer stocks
                        already performing well this year

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    if sector != "All":
        filtered = filtered[filtered["Sector"] == sector]

    if min_sharpe is not None:
        filtered = filtered[
            pd.to_numeric(filtered["Sharpe Ratio"], errors="coerce") >= min_sharpe
        ]

    if max_pe is not None:
        filtered = filtered[
            pd.to_numeric(filtered["P/E Ratio"], errors="coerce") <= max_pe
        ]

    if min_dividend is not None:
        # Dividend yield stored as "3.5%" string — convert back to float
        filtered = filtered[
            pd.to_numeric(
                filtered["Dividend Yield"].str.replace("%", ""),
                errors="coerce"
            ) >= min_dividend
        ]

    if min_ytd_return is not None:
        filtered = filtered[
            pd.to_numeric(
                filtered["YTD Return"].str.replace("%", ""),
                errors="coerce"
            ) >= min_ytd_return
        ]

    return filtered


def rank_stocks(
    df: pd.DataFrame,
    metric: str = "Sharpe Ratio",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Rank stocks by a chosen metric.

    Rationale: ranking makes it easy to spot the top and bottom
    performers at a glance. Analysts regularly sort screener results
    by different metrics depending on their investment strategy —
    value investors sort by P/E, income investors by dividend yield,
    quants by Sharpe ratio.

    Parameters:
        df:        screener DataFrame
        metric:    column name to sort by
        ascending: False = highest first (default), True = lowest first

    Returns:
        DataFrame sorted by the chosen metric with a Rank column added
    """
    numeric_col = pd.to_numeric(
        df[metric].astype(str).str.replace("%", ""),
        errors="coerce"
    )
    sorted_df = df.assign(_sort_col=numeric_col).sort_values(
        "_sort_col", ascending=ascending
    ).drop(columns="_sort_col").reset_index(drop=True)

    sorted_df.insert(0, "Rank", range(1, len(sorted_df) + 1))
    return sorted_df


def get_top_movers(
    daily_returns: pd.DataFrame,
    n: int = 5
) -> tuple:
    """
    Get the top N gainers and losers based on the most recent trading day.

    Rationale: top movers are the first thing analysts check every morning.
    Knowing which stocks moved the most — and why — is fundamental to
    market awareness. This appears on the Market Overview page as the
    headline summary.

    Parameters:
        daily_returns: DataFrame of daily percentage returns
        n:             number of top gainers and losers to return

    Returns:
        tuple of (gainers DataFrame, losers DataFrame)
        each with columns: Ticker, Name, Return
    """
    # Most recent day's returns
    latest_returns = daily_returns.iloc[-1].sort_values(ascending=False)

    gainers = latest_returns.head(n).reset_index()
    gainers.columns = ["Ticker", "Return"]
    gainers["Name"]   = gainers["Ticker"].map(
        lambda t: STOCKS.get(t, {}).get("name", t)
    )
    gainers["Return"] = gainers["Return"].apply(_format_pct)

    losers = latest_returns.tail(n).sort_values().reset_index()
    losers.columns = ["Ticker", "Return"]
    losers["Name"]  = losers["Ticker"].map(
        lambda t: STOCKS.get(t, {}).get("name", t)
    )
    losers["Return"] = losers["Return"].apply(_format_pct)

    return gainers, losers


# ── Private helper functions ──────────────────────────────────────────────────

def _format_pct(value) -> str:
    """
    Format a decimal return as a percentage string.
    e.g. 0.0523 → "5.23%"
    Returns "N/A" if value is None or NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_market_cap(value) -> str:
    """
    Format market cap in billions for readability.
    e.g. 2_800_000_000_000 → "2,800.00bn"
    Rationale: raw market cap numbers are unwieldy — converting to
    billions makes them immediately readable in a table.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    billions = value / 1_000_000_000
    return f"{billions:,.2f}bn"

