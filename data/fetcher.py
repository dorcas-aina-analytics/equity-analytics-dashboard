"""
data/fetcher.py

Responsible for all data retrieval from Yahoo Finance via yfinance.
All other modules import from here — no other file touches yfinance directly.

Rationale: centralising data fetching in one module means if Yahoo Finance
changes its API or we switch to a different data provider, we only need to
update one file. This is the 'single source of truth' principle.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st


# ── Stock universe ────────────────────────────────────────────────────────────
# Rationale: a curated multi-sector universe covering US and UK stocks.
# Dictionary maps ticker symbols to human-readable names and sectors.
# Using a dictionary here means we can easily add or remove stocks
# without changing any other code.

STOCKS = {
    # Technology
    "AAPL":  {"name": "Apple",     "sector": "Technology"},
    "MSFT":  {"name": "Microsoft", "sector": "Technology"},
    "NVDA":  {"name": "Nvidia",    "sector": "Technology"},
    "GOOGL": {"name": "Google",    "sector": "Technology"},

    # Financials
    "GS":    {"name": "Goldman Sachs", "sector": "Financials"},
    "JPM":   {"name": "JPMorgan",      "sector": "Financials"},
    "BARC.L":{"name": "Barclays",      "sector": "Financials"},
    "HSBA.L":{"name": "HSBC",          "sector": "Financials"},

    # Energy
    "SHEL.L":{"name": "Shell",     "sector": "Energy"},
    "BP.L":  {"name": "BP",        "sector": "Energy"},
    "XOM":   {"name": "ExxonMobil","sector": "Energy"},

    # Healthcare
    "AZN.L": {"name": "AstraZeneca",        "sector": "Healthcare"},
    "GSK.L": {"name": "GSK",                "sector": "Healthcare"},
    "JNJ":   {"name": "Johnson & Johnson",  "sector": "Healthcare"},

    # Consumer
    "ULVR.L":{"name": "Unilever",           "sector": "Consumer"},
    "PG":    {"name": "Procter & Gamble",   "sector": "Consumer"},
    "TSCO.L":{"name": "Tesco",              "sector": "Consumer"},
}

# S&P 500 as the market benchmark for Beta calculations
MARKET_TICKER = "^GSPC"

# Risk-free rate — using approximate UK 10-year gilt yield
# Rationale: Sharpe ratio requires a risk-free rate. We use the UK gilt
# yield as this is a UK-focused portfolio. In production this would be
# fetched live but a constant is acceptable for a portfolio project.
RISK_FREE_RATE = 0.045  # 4.5% annualised


# ── Data fetching functions ───────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_price_history(tickers: list, period: str = "1y") -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers.

    Rationale for @st.cache_data: fetching data from Yahoo Finance on every
    user interaction would be slow and hit rate limits. Caching stores the
    result for 3600 seconds (1 hour) — fast for the user, respectful to
    the API. ttl=3600 means data refreshes hourly, keeping it near real-time.

    Parameters:
        tickers: list of Yahoo Finance ticker symbols e.g. ['AAPL', 'MSFT']
        period:  time period string — '1d', '5d', '1mo', '3mo', '6mo',
                 '1y', '2y', '5y', '10y', 'ytd', 'max'

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,   # adjusts for splits and dividends automatically
        progress=False      # suppresses the download progress bar
    )

    # yfinance returns a MultiIndex DataFrame when fetching multiple tickers
    # We only need the 'Close' prices — extract that level
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # Drop any columns (tickers) where ALL values are NaN
    # Rationale: occasionally a ticker fails to fetch — drop it cleanly
    # rather than letting NaN values break downstream calculations
    data = data.dropna(axis=1, how="all")

    return data


@st.cache_data(ttl=3600)
def fetch_market_data(period: str = "1y") -> pd.Series:
    """
    Fetch S&P 500 index data for use as market benchmark in Beta calculations.

    Rationale: Beta measures a stock's volatility relative to the overall
    market. The S&P 500 is the standard benchmark for this calculation.
    """
    data = yf.download(
        tickers=MARKET_TICKER,
        period=period,
        auto_adjust=True,
        progress=False
    )
    return data["Close"].squeeze()


@st.cache_data(ttl=3600)
def fetch_stock_info(ticker: str) -> dict:
    """
    Fetch fundamental data for a single stock — P/E ratio, market cap,
    dividend yield, 52-week high/low.

    Rationale: fundamental data is used in the screener page. Fetching
    one stock at a time is slower but more reliable — Yahoo Finance
    occasionally fails bulk info requests.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker":         ticker,
            "name":           STOCKS.get(ticker, {}).get("name", ticker),
            "sector":         STOCKS.get(ticker, {}).get("sector", "Unknown"),
            "current_price":  info.get("currentPrice",    None),
            "market_cap":     info.get("marketCap",       None),
            "pe_ratio":       info.get("trailingPE",      None),
            "dividend_yield": info.get("dividendYield",   None),
            "week_52_high":   info.get("fiftyTwoWeekHigh",None),
            "week_52_low":    info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        # If fetch fails for any reason, return None values rather than
        # crashing the entire app — defensive programming
        return {
            "ticker": ticker,
            "name":   STOCKS.get(ticker, {}).get("name", ticker),
            "sector": STOCKS.get(ticker, {}).get("sector", "Unknown"),
            "current_price": None, "market_cap": None,
            "pe_ratio": None, "dividend_yield": None,
            "week_52_high": None, "week_52_low": None,
        }


def get_all_tickers() -> list:
    """Return all ticker symbols from the stock universe."""
    return list(STOCKS.keys())


def get_tickers_by_sector(sector: str) -> list:
    """Return tickers filtered by sector."""
    return [t for t, v in STOCKS.items() if v["sector"] == sector]


def get_sectors() -> list:
    """Return list of unique sectors."""
    return list(set(v["sector"] for v in STOCKS.values()))


def get_stock_name(ticker: str) -> str:
    """Return human-readable name for a ticker."""
    return STOCKS.get(ticker, {}).get("name", ticker)
