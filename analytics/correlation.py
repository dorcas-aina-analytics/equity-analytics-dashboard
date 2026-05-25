"""
analytics/correlation.py

Calculates correlation matrices and efficient frontier data for
portfolio analysis.

Rationale: correlation shows how stocks move together. A well-diversified
portfolio holds assets that are NOT highly correlated — when one falls,
others may hold steady or rise. This is the mathematical foundation of
Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952,
which won the Nobel Prize in Economics and is still the basis of
institutional portfolio construction today.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from data.fetcher import RISK_FREE_RATE


def calculate_correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix between all stocks.

    Rationale: correlation ranges from -1 to +1:
        +1.0  — stocks move perfectly together (no diversification benefit)
         0.0  — stocks are completely independent of each other
        -1.0  — stocks move perfectly opposite (maximum diversification)

    In practice, most stocks have positive correlation (they all tend
    to fall in a market crash). The goal is to find stocks with LOW
    correlation to each other to build a resilient portfolio.

    We use Pearson correlation on daily returns — not on prices.
    Rationale: correlating prices directly gives misleading results
    because two stocks can both trend upward over time and appear
    highly correlated simply due to the trend, even if their daily
    movements are actually independent.

    Parameters:
        daily_returns: DataFrame of daily percentage returns

    Returns:
        DataFrame — square matrix where entry [i][j] is the correlation
        between stock i and stock j. Diagonal is always 1.0 (a stock
        is perfectly correlated with itself).
    """
    return daily_returns.corr(method="pearson")


def calculate_portfolio_metrics(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame
) -> tuple:
    """
    Calculate expected return and volatility for a given portfolio weighting.

    Rationale: this function is the engine of the efficient frontier.
    Given a set of portfolio weights (e.g. 30% Apple, 20% Goldman, 50%
    Shell), it calculates what return and risk that portfolio would have
    historically delivered.

    The key insight of Modern Portfolio Theory is that combining assets
    in different proportions produces DIFFERENT risk/return profiles —
    and some combinations are mathematically optimal (the efficient
    frontier) while others are suboptimal (same return but higher risk).

    Parameters:
        weights:      array of portfolio weights, must sum to 1.0
                      e.g. [0.3, 0.2, 0.5] for three stocks
        mean_returns: Series of mean daily returns per stock
        cov_matrix:   covariance matrix of daily returns

    Returns:
        tuple of (annualised_return, annualised_volatility, sharpe_ratio)
    """
    # Annualise the mean returns (252 trading days)
    portfolio_return = np.sum(mean_returns * weights) * 252

    # Portfolio variance uses the covariance matrix
    # Formula: w^T × Σ × w where w is weights and Σ is covariance matrix
    # This accounts for how stocks move together — not just individual risk
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Sharpe ratio for this portfolio
    sharpe = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility

    return portfolio_return, portfolio_volatility, sharpe


def generate_efficient_frontier(
    daily_returns: pd.DataFrame,
    num_portfolios: int = 3000
) -> pd.DataFrame:
    """
    Generate random portfolios to plot the efficient frontier.

    Rationale: the efficient frontier is the set of portfolios that
    offer the highest expected return for a given level of risk, or
    equivalently, the lowest risk for a given expected return.
    Portfolios below the frontier are suboptimal — you could get the
    same return with less risk by rebalancing.

    Method: Monte Carlo simulation — we generate 3000 random portfolio
    weightings, calculate the risk/return for each, and plot them all.
    The upper-left boundary of the resulting scatter plot IS the
    efficient frontier.

    3000 portfolios is enough to clearly show the frontier shape
    without being computationally expensive.

    Parameters:
        daily_returns:  DataFrame of daily percentage returns
        num_portfolios: number of random portfolios to simulate

    Returns:
        DataFrame with columns: Return, Volatility, Sharpe, and one
        column per stock showing its weight in that portfolio
    """
    mean_returns = daily_returns.mean()
    cov_matrix   = daily_returns.cov()
    num_stocks   = len(daily_returns.columns)
    tickers      = daily_returns.columns.tolist()

    results = []

    for _ in range(num_portfolios):
        # Generate random weights that sum to 1
        # Rationale: Dirichlet distribution gives uniformly random
        # weights that always sum to exactly 1 — cleaner than
        # normalising random numbers
        weights = np.random.dirichlet(np.ones(num_stocks))

        port_return, port_vol, sharpe = calculate_portfolio_metrics(
            weights, mean_returns, cov_matrix
        )

        result = {
            "Return":     port_return,
            "Volatility": port_vol,
            "Sharpe":     sharpe,
        }
        # Store each stock's weight in this portfolio
        for i, ticker in enumerate(tickers):
            result[f"w_{ticker}"] = weights[i]

        results.append(result)

    return pd.DataFrame(results)


def find_optimal_portfolio(daily_returns: pd.DataFrame) -> dict:
    """
    Find the portfolio with the maximum Sharpe ratio using optimisation.

    Rationale: the maximum Sharpe ratio portfolio is called the
    'tangency portfolio' or 'market portfolio' in Modern Portfolio Theory.
    It represents the single best combination of risk and return —
    the point on the efficient frontier that a rational investor
    should hold (before adding or removing risk-free asset exposure).

    This uses scipy's minimize() function with the SLSQP method
    (Sequential Least Squares Programming) — an industry-standard
    numerical optimisation algorithm.

    Constraints:
        1. Weights must sum to 1.0 (fully invested)
        2. No short selling — each weight must be >= 0
           Rationale: short selling requires special permissions and
           adds complexity. Long-only constraint is realistic for
           most portfolio managers.

    Returns:
        dict with optimal weights, expected return, volatility,
        and Sharpe ratio
    """
    mean_returns = daily_returns.mean()
    cov_matrix   = daily_returns.cov()
    num_stocks   = len(daily_returns.columns)
    tickers      = daily_returns.columns.tolist()

    # We minimise NEGATIVE Sharpe ratio (scipy only minimises)
    # Minimising -Sharpe is equivalent to maximising Sharpe
    def negative_sharpe(weights):
        ret, vol, sharpe = calculate_portfolio_metrics(
            weights, mean_returns, cov_matrix
        )
        return -sharpe

    # Constraint: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # Bounds: each weight between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_stocks))

    # Initial guess: equal weights
    initial_weights = np.array([1 / num_stocks] * num_stocks)

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    opt_return, opt_vol, opt_sharpe = calculate_portfolio_metrics(
        optimal_weights, mean_returns, cov_matrix
    )

    return {
        "weights":    dict(zip(tickers, optimal_weights)),
        "return":     opt_return,
        "volatility": opt_vol,
        "sharpe":     opt_sharpe,
    }
