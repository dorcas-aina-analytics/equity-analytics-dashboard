"""
analytics/returns.py

Calculates all return-based metrics: daily returns, cumulative returns,
YTD performance, and period performance.

Rationale: returns are the foundation of every other calculation in this
project. Volatility, Sharpe ratio, Beta, and correlation all start from
daily returns — so getting this right is the most important step.
"""

import pandas as pd
import numpy as np


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns from price data.

    Rationale: we use percentage change (pct_change) rather than absolute
    price change because percentage returns are comparable across stocks
    with different price levels. Apple at $180 and Tesco at £3 can only
    be compared meaningfully as percentage moves.

    Formula: (today's price - yesterday's price) / yesterday's price
    pct_change() does this automatically for every row.

    Parameters:
        prices: DataFrame of closing prices (dates as index, tickers as columns)

    Returns:
        DataFrame of daily percentage returns, first row will be NaN
        (no previous day to compare against) — we drop it with dropna()
    """
    return prices.pct_change().dropna()


def calculate_cumulative_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative returns from daily returns.

    Rationale: cumulative returns answer the question "if I invested £1000
    at the start of this period, what would it be worth today?"
    This is more intuitive than daily returns for visualisation.

    Formula: (1 + r1) × (1 + r2) × ... × (1 + rn) - 1
    cumprod() multiplies each row by the running product.

    Example: +2% then -1% then +3% gives:
    1.02 × 0.99 × 1.03 - 1 = 3.97% cumulative return
    NOT 2% - 1% + 3% = 4% — compounding matters.
    """
    return (1 + daily_returns).cumprod() - 1


def calculate_ytd_return(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate year-to-date return for each stock.

    Rationale: YTD is the most common performance metric quoted by
    analysts and in financial media. It shows how a stock has performed
    from 1st January to today regardless of the selected date range.

    We find the first trading day of the current year by filtering
    the price data to dates >= January 1st of the current year.
    """
    current_year = pd.Timestamp.now().year
    ytd_prices = prices[prices.index >= f"{current_year}-01-01"]

    if ytd_prices.empty:
        return pd.Series(dtype=float)

    # First price of the year and latest price
    start_price = ytd_prices.iloc[0]
    end_price   = ytd_prices.iloc[-1]

    return (end_price - start_price) / start_price


def calculate_period_return(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate total return over the full selected price history.

    Rationale: shows the total gain or loss over whatever date range
    the user has selected — complements YTD by showing longer-term
    performance.
    """
    start_price = prices.iloc[0]
    end_price   = prices.iloc[-1]
    return (end_price - start_price) / start_price


def calculate_annualised_return(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Annualise daily returns to make them comparable regardless of
    the time period selected.

    Rationale: a 5% return over 1 month is very different from a 5%
    return over 1 year. Annualising converts everything to a
    "per year" basis so stocks across different time periods
    can be compared fairly.

    Formula: (1 + mean_daily_return)^252 - 1
    252 is the number of trading days in a year — standard in finance.
    Calendar days (365) are not used because markets are closed
    on weekends and public holidays.
    """
    mean_daily = daily_returns.mean()
    return (1 + mean_daily) ** 252 - 1


def calculate_portfolio_value(
    daily_returns: pd.DataFrame,
    initial_investment: float = 1000.0
) -> pd.DataFrame:
    """
    Calculate portfolio value over time for a given initial investment.

    Rationale: converts abstract percentage returns into concrete
    monetary values — £1000 invested in Apple vs £1000 in Goldman Sachs,
    which grew more? This is immediately understandable to any audience,
    financial or not.

    Parameters:
        daily_returns:      DataFrame of daily percentage returns
        initial_investment: starting value in GBP, default £1000

    Returns:
        DataFrame showing portfolio value in £ for each stock over time
    """
    cumulative = (1 + daily_returns).cumprod()
    return cumulative * initial_investment

