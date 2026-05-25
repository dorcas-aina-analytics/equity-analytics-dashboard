"""
analytics/volatility.py

Calculates all risk metrics: rolling volatility, Sharpe ratio, Beta,
and Value at Risk (VaR).

Rationale: risk metrics are what separate a basic price chart from a
professional analytical tool. Any analyst at Goldman Sachs, JP Morgan,
or a hedge fund uses these metrics daily. Including them signals that
you understand not just what stocks do, but how risky they are.
"""

import pandas as pd
import numpy as np
from data.fetcher import RISK_FREE_RATE


def calculate_rolling_volatility(
    daily_returns: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling annualised volatility for each stock.

    Rationale: volatility measures how much a stock's price fluctuates.
    A stock with high volatility is riskier — its price can swing
    dramatically in either direction. Rolling volatility shows how
    risk has changed over time, not just a single number.

    We annualise by multiplying by sqrt(252) — the square root of
    trading days in a year. This is the industry standard conversion
    from daily volatility to annual volatility.

    Formula: rolling_std(daily_returns, window) × sqrt(252)

    Parameters:
        daily_returns: DataFrame of daily percentage returns
        window:        number of trading days in the rolling window
                       default 30 = approximately one calendar month

    Returns:
        DataFrame of annualised rolling volatility (as decimals,
        e.g. 0.25 = 25% annualised volatility)
    """
    return daily_returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(
    daily_returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE
) -> pd.Series:
    """
    Calculate the Sharpe ratio for each stock.

    Rationale: the Sharpe ratio is the single most important metric in
    portfolio management. It measures return per unit of risk — a stock
    with a high return but very high volatility may actually be worse
    than a stock with a moderate return and low volatility.

    Formula: (annualised_return - risk_free_rate) / annualised_volatility

    Interpretation:
        Sharpe > 1.0  — good, earning more than 1 unit of return per
                        unit of risk
        Sharpe > 2.0  — very good
        Sharpe < 0    — the stock is not even beating the risk-free rate
                        (e.g. UK gilts at 4.5%) after adjusting for risk

    Parameters:
        daily_returns:  DataFrame of daily percentage returns
        risk_free_rate: annualised risk-free rate as a decimal
                        (default 4.5% = 0.045, approximate UK gilt yield)

    Returns:
        Series with Sharpe ratio for each stock
    """
    # Annualise mean daily return
    annualised_return = daily_returns.mean() * 252

    # Annualise volatility
    annualised_vol = daily_returns.std() * np.sqrt(252)

    # Convert annual risk-free rate to match — already annualised
    daily_rf = risk_free_rate / 252
    excess_return = daily_returns.mean() - daily_rf
    annualised_excess = excess_return * 252

    return annualised_excess / annualised_vol


def calculate_beta(
    daily_returns: pd.DataFrame,
    market_returns: pd.Series
) -> pd.Series:
    """
    Calculate Beta for each stock against the S&P 500.

    Rationale: Beta measures how much a stock moves relative to the
    overall market. It answers: "if the S&P 500 drops 10%, how much
    does this stock typically drop?"

    Interpretation:
        Beta = 1.0  — moves exactly with the market
        Beta > 1.0  — more volatile than the market (e.g. growth stocks,
                      tech stocks like Nvidia often have Beta > 1.5)
        Beta < 1.0  — less volatile than the market (e.g. utility
                      companies, consumer staples like Unilever)
        Beta < 0    — moves opposite to the market (rare, e.g. gold)

    Formula: covariance(stock, market) / variance(market)

    This is calculated using linear regression of stock returns
    against market returns — the slope of that regression line is Beta.

    Parameters:
        daily_returns:  DataFrame of daily percentage returns per stock
        market_returns: Series of daily S&P 500 returns (the benchmark)

    Returns:
        Series with Beta value for each stock
    """
    # Align dates — market and stock data must cover same period
    aligned_market = market_returns.reindex(daily_returns.index).dropna()
    aligned_stocks = daily_returns.reindex(aligned_market.index).dropna()

    betas = {}
    for ticker in aligned_stocks.columns:
        stock_series = aligned_stocks[ticker].dropna()
        market_series = aligned_market.reindex(stock_series.index)

        # Covariance between stock and market
        covariance = np.cov(stock_series, market_series)[0][1]

        # Variance of the market
        market_variance = np.var(market_series, ddof=1)

        betas[ticker] = covariance / market_variance if market_variance != 0 else np.nan

    return pd.Series(betas)


def calculate_var(
    daily_returns: pd.DataFrame,
    confidence: float = 0.95,
    initial_investment: float = 1000.0
) -> pd.Series:
    """
    Calculate Value at Risk (VaR) at a given confidence level.

    Rationale: VaR answers the question "what is the maximum amount
    I would expect to lose on a given day, with X% confidence?"
    It is the standard risk metric used by banks, regulators, and
    fund managers globally. Basel III (banking regulation) requires
    banks to calculate VaR for their trading books.

    Example interpretation: VaR of £25 at 95% confidence means
    "on 95% of trading days, I would not expect to lose more than
    £25 on a £1000 investment." On the remaining 5% of days (roughly
    12 trading days per year) losses could exceed this.

    Method: Historical VaR — we use the actual distribution of past
    returns rather than assuming a normal distribution. This is more
    accurate for stocks which tend to have fat tails (extreme events
    happen more often than a normal distribution predicts).

    Formula: percentile(daily_returns, 1 - confidence) × investment

    Parameters:
        daily_returns:      DataFrame of daily percentage returns
        confidence:         confidence level, default 0.95 (95%)
        initial_investment: portfolio value in £, default £1000

    Returns:
        Series with VaR in £ for each stock (shown as positive number
        representing the loss amount)
    """
    # The (1 - confidence) percentile of returns is the worst return
    # we'd expect with the given confidence level
    # e.g. at 95% confidence, we look at the 5th percentile of returns
    var_pct = daily_returns.quantile(1 - confidence)

    # Convert percentage loss to monetary loss
    # Shown as positive number (loss amount, not negative return)
    return abs(var_pct * initial_investment)


def calculate_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Calculate maximum drawdown for each stock.

    Rationale: maximum drawdown measures the largest peak-to-trough
    decline in a stock's price history. It answers: "what is the worst
    loss someone could have experienced if they bought at the peak
    and sold at the trough?"

    This is a key risk metric for investors — a stock might have
    high average returns but if it regularly drops 50% from peak,
    most investors would panic and sell before recovering.

    Formula: (trough_price - peak_price) / peak_price
    where peak is the running maximum up to each point in time.

    Returns:
        Series with maximum drawdown as a negative decimal
        e.g. -0.35 means the stock fell 35% from its peak at worst
    """
    # Running maximum price up to each point
    rolling_max = prices.cummax()

    # Drawdown at each point = how far below the peak are we?
    drawdown = (prices - rolling_max) / rolling_max

    # Maximum (worst) drawdown over the full period
    return drawdown.min()
