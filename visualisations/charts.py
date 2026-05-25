"""
visualisations/charts.py

All Plotly chart functions for the dashboard.

Rationale: keeping all chart code in one module means the page files
stay clean and readable — they just call a function and get a chart back.
It also means if we want to change the visual style of every chart
(colours, fonts, layout) we only need to update one file.

Every function returns a Plotly figure object which Streamlit renders
with st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data.fetcher import STOCKS


# ── Colour palette ────────────────────────────────────────────────────────────
# Rationale: consistent colours across all charts makes the dashboard
# feel cohesive and professional. These colours complement the dark
# theme defined in config.toml.

COLOURS = {
    "Technology":  "#3B82F6",  # blue
    "Financials":  "#10B981",  # green
    "Energy":      "#F59E0B",  # amber
    "Healthcare":  "#8B5CF6",  # purple
    "Consumer":    "#EF4444",  # red
    "positive":    "#10B981",  # green for gains
    "negative":    "#EF4444",  # red for losses
    "neutral":     "#94A3B8",  # grey for neutral
}

SECTOR_COLOURS = {
    sector: colour for sector, colour in COLOURS.items()
    if sector not in ["positive", "negative", "neutral"]
}

# Dark background matching config.toml
PLOT_BG    = "#0F172A"
PAPER_BG   = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#F1F5F9"


def _base_layout(title: str = "") -> dict:
    """
    Base layout settings applied to every chart.

    Rationale: defining a base layout once and reusing it ensures
    every chart has the same dark background, font, and grid style.
    This is the chart equivalent of a CSS stylesheet.
    """
    return dict(
        title=dict(
            text=title,
            font=dict(color=TEXT_COLOR, size=16),
            x=0.01
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, family="sans-serif"),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COLOR,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        hovermode="x unified",
    )


# ── Page 1: Market Overview charts ───────────────────────────────────────────

def sector_heatmap(sector_returns: pd.DataFrame) -> go.Figure:
    """
    Heatmap showing sector performance across different time periods.

    Rationale: a heatmap lets you see at a glance which sectors are
    performing well (green) and which are underperforming (red) across
    multiple time horizons simultaneously. This is the standard way
    sector performance is displayed on Bloomberg Terminal and financial
    news sites.

    Parameters:
        sector_returns: DataFrame with sectors as rows and time periods
                        (1D, 1W, 1M, 3M, YTD) as columns, values as
                        decimal returns
    """
    # Format values as percentages for display
    text_values = sector_returns.applymap(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
    )

    fig = go.Figure(data=go.Heatmap(
        z=sector_returns.values,
        x=sector_returns.columns.tolist(),
        y=sector_returns.index.tolist(),
        text=text_values.values,
        texttemplate="%{text}",
        textfont=dict(size=13),
        colorscale=[
            [0.0,  "#EF4444"],   # deep red for large losses
            [0.45, "#7F1D1D"],   # dark red
            [0.5,  "#1E293B"],   # neutral dark
            [0.55, "#064E3B"],   # dark green
            [1.0,  "#10B981"],   # bright green for large gains
        ],
        zmid=0,                  # centre the colour scale at zero
        showscale=True,
        colorbar=dict(
            title="Return",
            tickformat=".1%",
            tickfont=dict(color=TEXT_COLOR),
            titlefont=dict(color=TEXT_COLOR),
        )
    ))

    layout = _base_layout("Sector Performance Heatmap")
    layout["hovermode"] = "closest"
    fig.update_layout(**layout)
    return fig


def top_movers_chart(
    gainers: pd.DataFrame,
    losers: pd.DataFrame
) -> go.Figure:
    """
    Horizontal bar chart showing top 5 gainers and losers.

    Rationale: a horizontal bar chart is the clearest way to show
    ranked performance — the bars extend left (losses) or right (gains)
    from zero, making the magnitude immediately clear.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top 5 Gainers", "Top 5 Losers"),
    )

    # Gainers — green bars
    fig.add_trace(
        go.Bar(
            x=gainers["Return"].str.replace("%", "").astype(float),
            y=gainers["Name"],
            orientation="h",
            marker_color=COLOURS["positive"],
            text=gainers["Return"],
            textposition="outside",
            name="Gainers",
        ),
        row=1, col=1
    )

    # Losers — red bars
    fig.add_trace(
        go.Bar(
            x=losers["Return"].str.replace("%", "").astype(float),
            y=losers["Name"],
            orientation="h",
            marker_color=COLOURS["negative"],
            text=losers["Return"],
            textposition="outside",
            name="Losers",
        ),
        row=1, col=2
    )

    layout = _base_layout("Today's Top Movers")
    layout["showlegend"] = False
    fig.update_layout(**layout)
    fig.update_annotations(font_color=TEXT_COLOR)
    return fig


def volume_trend_chart(volume_data: pd.DataFrame) -> go.Figure:
    """
    Line chart showing total market volume over time.

    Rationale: volume is a key indicator of market activity and
    conviction. High volume on a price rise suggests strong buying
    interest. Unusually high volume often precedes significant
    price moves. Volume is the 'weight of money' in the market.
    """
    total_volume = volume_data.sum(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=total_volume.index,
        y=total_volume.values,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.2)",
        line=dict(color=COLOURS["Technology"], width=1.5),
        name="Total Volume",
    ))

    layout = _base_layout("Total Market Volume")
    layout["yaxis"]["title"] = "Shares Traded"
    fig.update_layout(**layout)
    return fig


# ── Page 2: Price & Trend Analysis charts ────────────────────────────────────

def candlestick_chart(
    ohlc_data: pd.DataFrame,
    ticker: str,
    ma_windows: list = [20, 50, 200]
) -> go.Figure:
    """
    Candlestick chart with moving averages and Bollinger Bands.

    Rationale: candlestick charts show open, high, low, and close prices
    for each day — far more informative than a simple line chart which
    only shows closing price. Each candle is green (price rose) or red
    (price fell). This is the standard chart type used by professional
    traders and analysts.

    Moving averages smooth out daily noise and reveal the underlying
    trend direction. The 200-day MA is particularly significant —
    stocks above their 200-day MA are considered to be in an uptrend.

    Bollinger Bands show when a stock is statistically "stretched"
    above or below its normal range — potential reversal signals.

    Parameters:
        ohlc_data:  DataFrame with Open, High, Low, Close columns
        ticker:     stock ticker for labelling
        ma_windows: list of moving average periods to overlay
    """
    stock_name = STOCKS.get(ticker, {}).get("name", ticker)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{stock_name} Price", "Volume")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data["Open"],
            high=ohlc_data["High"],
            low=ohlc_data["Low"],
            close=ohlc_data["Close"],
            name="Price",
            increasing_line_color=COLOURS["positive"],
            decreasing_line_color=COLOURS["negative"],
        ),
        row=1, col=1
    )

    # Moving averages
    ma_colours = ["#F59E0B", "#8B5CF6", "#EF4444"]
    for i, window in enumerate(ma_windows):
        if len(ohlc_data) >= window:
            ma = ohlc_data["Close"].rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=ohlc_data.index,
                    y=ma,
                    mode="lines",
                    line=dict(color=ma_colours[i], width=1.2),
                    name=f"{window}D MA",
                    opacity=0.8,
                ),
                row=1, col=1
            )

    # Bollinger Bands (20-day, 2 standard deviations)
    # Rationale: 20 days and 2 std deviations is the standard setting
    # used by traders worldwide, developed by John Bollinger in the 1980s
    if len(ohlc_data) >= 20:
        close     = ohlc_data["Close"]
        bb_mid    = close.rolling(20).mean()
        bb_std    = close.rolling(20).std()
        bb_upper  = bb_mid + (2 * bb_std)
        bb_lower  = bb_mid - (2 * bb_std)

        fig.add_trace(
            go.Scatter(
                x=ohlc_data.index, y=bb_upper,
                mode="lines",
                line=dict(color="#94A3B8", width=0.8, dash="dash"),
                name="BB Upper", opacity=0.5,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=ohlc_data.index, y=bb_lower,
                mode="lines",
                line=dict(color="#94A3B8", width=0.8, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.05)",
                name="BB Lower", opacity=0.5,
            ),
            row=1, col=1
        )

    # Volume bars
    colours = [
        COLOURS["positive"] if c >= o else COLOURS["negative"]
        for c, o in zip(ohlc_data["Close"], ohlc_data["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=ohlc_data.index,
            y=ohlc_data["Volume"],
            name="Volume",
            marker_color=colours,
            opacity=0.7,
        ),
        row=2, col=1
    )

    layout = _base_layout(f"{stock_name} — Price and Volume")
    layout["xaxis_rangeslider_visible"] = False
    layout["height"] = 600
    fig.update_layout(**layout)
    fig.update_annotations(font_color=TEXT_COLOR)
    return fig


# ── Page 3: Volatility & Risk charts ─────────────────────────────────────────

def rolling_volatility_chart(rolling_vol: pd.DataFrame) -> go.Figure:
    """
    Line chart showing rolling 30-day annualised volatility over time.

    Rationale: plotting volatility over time reveals periods of market
    stress (spikes) and calm (low flat lines). COVID crash in March 2020,
    the 2022 rate hike period — these all show up as volatility spikes.
    This is how risk managers monitor portfolio risk over time.
    """
    fig = go.Figure()

    for ticker in rolling_vol.columns:
        sector = STOCKS.get(ticker, {}).get("sector", "Technology")
        name   = STOCKS.get(ticker, {}).get("name", ticker)
        colour = SECTOR_COLOURS.get(sector, COLOURS["neutral"])

        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[ticker],
            mode="lines",
            name=name,
            line=dict(color=colour, width=1.5),
            opacity=0.8,
        ))

    layout = _base_layout("Rolling 30-Day Annualised Volatility")
    layout["yaxis"]["tickformat"] = ".1%"
    layout["yaxis"]["title"]      = "Annualised Volatility"
    fig.update_layout(**layout)
    return fig


def risk_metrics_bar(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing Sharpe ratio, Beta, and VaR across stocks.

    Rationale: showing multiple risk metrics side by side for each stock
    allows quick comparison — a stock with high Sharpe but high Beta is
    a different risk profile than one with moderate Sharpe and low Beta.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Sharpe Ratio", "Beta (vs S&P 500)", "VaR 95% (£ per £1000)")
    )

    names = metrics_df["Name"].tolist()

    # Sharpe ratio — colour green if > 1, red if negative
    sharpe_colours = [
        COLOURS["positive"] if s > 1
        else COLOURS["negative"] if s < 0
        else COLOURS["neutral"]
        for s in metrics_df["Sharpe"]
    ]

    fig.add_trace(
        go.Bar(
            x=names, y=metrics_df["Sharpe"],
            marker_color=sharpe_colours,
            name="Sharpe", showlegend=False,
        ),
        row=1, col=1
    )

    # Beta — colour blue if < 1 (defensive), red if > 1.5 (aggressive)
    beta_colours = [
        COLOURS["Technology"] if b < 1
        else COLOURS["negative"] if b > 1.5
        else COLOURS["neutral"]
        for b in metrics_df["Beta"]
    ]

    fig.add_trace(
        go.Bar(
            x=names, y=metrics_df["Beta"],
            marker_color=beta_colours,
            name="Beta", showlegend=False,
        ),
        row=1, col=2
    )

    # VaR — always red (it represents potential loss)
    fig.add_trace(
        go.Bar(
            x=names, y=metrics_df["VaR"],
            marker_color=COLOURS["negative"],
            name="VaR", showlegend=False,
        ),
        row=1, col=3
    )

    layout = _base_layout("Risk Metrics Comparison")
    layout["height"] = 400
    fig.update_layout(**layout)
    fig.update_annotations(font_color=TEXT_COLOR)
    fig.update_xaxes(tickangle=45)
    return fig


# ── Page 4: Correlation & Portfolio charts ────────────────────────────────────

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Heatmap of the correlation matrix between all stocks.

    Rationale: the correlation heatmap is the key tool for understanding
    diversification. Dark red = highly correlated (move together, bad for
    diversification). Dark blue = negatively correlated (move opposite,
    good for diversification). The diagonal is always 1.0 (each stock
    is perfectly correlated with itself).

    We replace ticker symbols with company names on the axes for
    readability — a recruiter or client reading this should not need
    to know that AAPL = Apple.
    """
    # Replace tickers with names for readability
    name_map = {t: STOCKS[t]["name"] for t in corr_matrix.columns if t in STOCKS}
    corr_named = corr_matrix.rename(columns=name_map, index=name_map)

    fig = go.Figure(data=go.Heatmap(
        z=corr_named.values,
        x=corr_named.columns.tolist(),
        y=corr_named.index.tolist(),
        text=np.round(corr_named.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(
            title="Correlation",
            tickfont=dict(color=TEXT_COLOR),
            titlefont=dict(color=TEXT_COLOR),
        )
    ))

    layout = _base_layout("Stock Correlation Matrix")
    layout["height"] = 600
    layout["xaxis"]["tickangle"] = 45
    fig.update_layout(**layout)
    return fig


def efficient_frontier_chart(
    frontier_df: pd.DataFrame,
    optimal: dict
) -> go.Figure:
    """
    Scatter plot of the efficient frontier with optimal portfolio highlighted.

    Rationale: the efficient frontier visualises the risk/return tradeoff
    across thousands of possible portfolios. Each dot is one randomly
    generated portfolio. The colour shows the Sharpe ratio — yellow/green
    dots are better risk-adjusted portfolios. The star marks the optimal
    (maximum Sharpe) portfolio.

    This is one of the most recognised charts in finance — showing it
    immediately signals familiarity with Modern Portfolio Theory.
    """
    fig = go.Figure()

    # All simulated portfolios — coloured by Sharpe ratio
    fig.add_trace(go.Scatter(
        x=frontier_df["Volatility"],
        y=frontier_df["Return"],
        mode="markers",
        marker=dict(
            size=4,
            color=frontier_df["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Sharpe Ratio",
                tickfont=dict(color=TEXT_COLOR),
                titlefont=dict(color=TEXT_COLOR),
            ),
            opacity=0.6,
        ),
        name="Simulated Portfolios",
        hovertemplate=(
            "Return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<br>"
            "<extra></extra>"
        )
    ))

    # Optimal portfolio — gold star
    fig.add_trace(go.Scatter(
        x=[optimal["volatility"]],
        y=[optimal["return"]],
        mode="markers",
        marker=dict(
            size=15,
            color="#F59E0B",
            symbol="star",
        ),
        name=f"Optimal Portfolio (Sharpe: {optimal['sharpe']:.2f})",
        hovertemplate=(
            f"Optimal Portfolio<br>"
            f"Return: {optimal['return']:.2%}<br>"
            f"Volatility: {optimal['volatility']:.2%}<br>"
            f"Sharpe: {optimal['sharpe']:.2f}<br>"
            "<extra></extra>"
        )
    ))

    layout = _base_layout("Efficient Frontier — 3,000 Simulated Portfolios")
    layout["xaxis"]["title"]      = "Annualised Volatility (Risk)"
    layout["xaxis"]["tickformat"] = ".1%"
    layout["yaxis"]["title"]      = "Annualised Return"
    layout["yaxis"]["tickformat"] = ".1%"
    layout["height"]              = 500
    fig.update_layout(**layout)
    return fig


def cumulative_returns_chart(portfolio_values: pd.DataFrame) -> go.Figure:
    """
    Line chart showing growth of £1000 invested in each stock.

    Rationale: showing portfolio value in pounds rather than percentages
    makes returns tangible and immediately understandable. "£1000 in
    Nvidia grew to £3,200" is more compelling than "320% return."
    This is how wealth management firms present performance to clients.
    """
    fig = go.Figure()

    for ticker in portfolio_values.columns:
        sector = STOCKS.get(ticker, {}).get("sector", "Technology")
        name   = STOCKS.get(ticker, {}).get("name", ticker)
        colour = SECTOR_COLOURS.get(sector, COLOURS["neutral"])

        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values[ticker],
            mode="lines",
            name=name,
            line=dict(color=colour, width=1.5),
            hovertemplate=f"{name}<br>£%{{y:,.0f}}<extra></extra>",
        ))

    # Reference line at £1000 (breakeven)
    fig.add_hline(
        y=1000,
        line_dash="dash",
        line_color=COLOURS["neutral"],
        opacity=0.5,
        annotation_text="Initial £1,000",
        annotation_font_color=TEXT_COLOR,
    )

    layout = _base_layout("Growth of £1,000 Investment")
    layout["yaxis"]["title"]      = "Portfolio Value (£)"
    layout["yaxis"]["tickprefix"] = "£"
    layout["yaxis"]["tickformat"] = ",.0f"
    fig.update_layout(**layout)
    return fig
