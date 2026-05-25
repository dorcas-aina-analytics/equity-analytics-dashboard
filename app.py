"""
app.py

Main Streamlit application entry point.

Rationale: app.py is intentionally kept thin — it handles page routing
and calls functions from other modules rather than containing any logic
itself. This separation means if we want to add a new page or change
a calculation, we know exactly which file to edit without touching
the others. This is the Single Responsibility Principle in practice.
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page configuration ────────────────────────────────────────────────────────
# Rationale: must be the first Streamlit call in the script.
# wide layout uses the full browser width — essential for financial
# dashboards where charts need horizontal space to be readable.

st.set_page_config(
    page_title="Equity Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────────

from data.fetcher import (
    fetch_price_history,
    fetch_market_data,
    get_all_tickers,
    get_sectors,
    get_stock_name,
    STOCKS,
    RISK_FREE_RATE,
)
from analytics.returns import (
    calculate_daily_returns,
    calculate_cumulative_returns,
    calculate_ytd_return,
    calculate_period_return,
    calculate_annualised_return,
    calculate_portfolio_value,
)
from analytics.volatility import (
    calculate_rolling_volatility,
    calculate_sharpe_ratio,
    calculate_beta,
    calculate_var,
    calculate_max_drawdown,
)
from analytics.correlation import (
    calculate_correlation_matrix,
    generate_efficient_frontier,
    find_optimal_portfolio,
)
from analytics.screener import (
    build_screener_table,
    filter_screener,
    rank_stocks,
    get_top_movers,
)
from visualisations.charts import (
    sector_heatmap,
    top_movers_chart,
    volume_trend_chart,
    candlestick_chart,
    rolling_volatility_chart,
    risk_metrics_bar,
    correlation_heatmap,
    efficient_frontier_chart,
    cumulative_returns_chart,
)
from visualisations.tables import (
    style_screener_table,
    render_metric_cards,
    render_optimal_weights_table,
    render_top_movers_table,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0;'>
            <h2 style='color:#3B82F6; margin:0;'>📈 Equity Analytics</h2>
            <p style='color:#94A3B8; font-size:12px; margin:4px 0 0 0;'>
                Quantitative Research Dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Page navigation
    # Rationale: st.radio for navigation is cleaner than st.selectbox
    # for a small number of pages — all options visible at once
    page = st.radio(
        "Navigation",
        options=[
            "📊 Market Overview",
            "📈 Price & Trend Analysis",
            "⚡ Volatility & Risk",
            "🔗 Correlation & Portfolio",
            "🔍 Stock Screener",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Global date range selector
    # Rationale: a global control in the sidebar affects all pages
    # simultaneously — the user sets their analysis period once and
    # every chart updates. This is how Bloomberg Terminal works.
    st.markdown(
        "<p style='color:#94A3B8; font-size:12px; margin-bottom:4px;'>"
        "ANALYSIS PERIOD</p>",
        unsafe_allow_html=True,
    )
    period = st.selectbox(
        "Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,  # default to 1 year
        format_func=lambda x: {
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "1y":  "1 Year",
            "2y":  "2 Years",
            "5y":  "5 Years",
        }[x],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        """
        <div style='color:#94A3B8; font-size:11px; line-height:1.6;'>
            <p><b style='color:#F1F5F9;'>Data source:</b> Yahoo Finance</p>
            <p><b style='color:#F1F5F9;'>Universe:</b> 17 stocks across
            5 sectors</p>
            <p><b style='color:#F1F5F9;'>Risk-free rate:</b> 4.5%
            (UK 10Y Gilt)</p>
            <p><b style='color:#F1F5F9;'>Built by:</b>
            <a href='https://dorcasainaa-dotcom.github.io'
            style='color:#3B82F6;'>Dorcas Aina</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data loading ──────────────────────────────────────────────────────────────
# Rationale: we load price data once at the top of app.py rather than
# inside each page. Because fetch_price_history uses @st.cache_data,
# the actual API call only happens once per hour — subsequent page
# switches use the cached result instantly.

all_tickers   = get_all_tickers()
prices        = fetch_price_history(all_tickers, period=period)
daily_returns = calculate_daily_returns(prices)
market_data   = fetch_market_data(period=period)
market_returns = calculate_daily_returns(
    market_data.to_frame()
).squeeze()


# ── Page routing ──────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Market Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Market Overview":
    st.title("Market Overview")
    st.markdown(
        "<p style='color:#94A3B8;'>Headline performance snapshot "
        "across all sectors. Data refreshes every hour.</p>",
        unsafe_allow_html=True,
    )

    # Top metric cards
    ytd = calculate_ytd_return(prices)
    period_ret = calculate_period_return(prices)
    vol = calculate_rolling_volatility(daily_returns).iloc[-1]

    avg_ytd    = ytd.mean()
    avg_ret    = period_ret.mean()
    avg_vol    = vol.mean()
    best_stock = ytd.idxmax()
    best_name  = get_stock_name(best_stock)
    best_ytd   = ytd.max()

    render_metric_cards({
        "Avg YTD Return":     (f"{avg_ytd*100:.2f}%",   None),
        "Avg Period Return":  (f"{avg_ret*100:.2f}%",   None),
        "Avg Volatility":     (f"{avg_vol*100:.2f}%",   None),
        "Best Performer YTD": (f"{best_name}",
                               f"{best_ytd*100:.2f}%"),
    })

    st.divider()

    # Sector performance heatmap
    # Build sector returns across multiple time periods
    periods_map = {
        "1D":  "5d",
        "1W":  "1mo",
        "1M":  "3mo",
        "3M":  "6mo",
        "YTD": "ytd",
    }

    sector_data = {}
    for label, p in periods_map.items():
        p_prices = fetch_price_history(all_tickers, period=p)
        p_ret    = calculate_period_return(p_prices)
        # Average return per sector
        for sector in get_sectors():
            tickers_in_sector = [
                t for t in p_ret.index
                if STOCKS.get(t, {}).get("sector") == sector
            ]
            if tickers_in_sector:
                sector_data.setdefault(sector, {})[label] = (
                    p_ret[tickers_in_sector].mean()
                )

    sector_df = pd.DataFrame(sector_data).T[
        ["1D", "1W", "1M", "3M", "YTD"]
    ]
    st.plotly_chart(
        sector_heatmap(sector_df),
        use_container_width=True,
    )

    st.divider()

    # Top movers and volume
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        gainers, losers = get_top_movers(daily_returns, n=5)
        st.plotly_chart(
            top_movers_chart(gainers, losers),
            use_container_width=True,
        )

    with col2:
        import yfinance as yf
        # Fetch volume data for the universe
        raw = yf.download(
            all_tickers, period=period,
            auto_adjust=True, progress=False
        )
        if isinstance(raw.columns, pd.MultiIndex):
            vol_data = raw["Volume"].dropna(axis=1, how="all")
        else:
            vol_data = raw[["Volume"]].dropna()

        st.plotly_chart(
            volume_trend_chart(vol_data),
            use_container_width=True,
        )

    st.divider()
    render_top_movers_table(gainers, losers)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Price & Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Price & Trend Analysis":
    st.title("Price & Trend Analysis")
    st.markdown(
        "<p style='color:#94A3B8;'>Candlestick charts with moving "
        "averages and Bollinger Bands. Select a stock to analyse.</p>",
        unsafe_allow_html=True,
    )

    # Stock selector
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker_options = {
            get_stock_name(t): t for t in all_tickers
            if t in prices.columns
        }
        selected_name   = st.selectbox(
            "Select Stock",
            options=list(ticker_options.keys()),
        )
        selected_ticker = ticker_options[selected_name]

    # Fetch OHLCV data for selected stock
    ohlcv = yf.download(
        selected_ticker, period=period,
        auto_adjust=True, progress=False
    )

    if ohlcv.empty:
        st.warning(f"No data available for {selected_name}.")
    else:
        # Metric cards for selected stock
        stock_prices  = prices[[selected_ticker]].dropna()
        stock_returns = calculate_daily_returns(stock_prices)
        ytd_ret       = calculate_ytd_return(stock_prices)
        period_ret    = calculate_period_return(stock_prices)
        sharpe        = calculate_sharpe_ratio(stock_returns)
        rolling_v     = calculate_rolling_volatility(stock_returns)
        latest_vol    = rolling_v.iloc[-1].values[0] \
                        if not rolling_v.empty else 0

        render_metric_cards({
            "Current Price":  (
                f"${ohlcv['Close'].iloc[-1]:.2f}"
                if selected_ticker not in ["BARC.L", "HSBA.L",
                    "SHEL.L", "BP.L", "AZN.L", "GSK.L",
                    "ULVR.L", "TSCO.L"]
                else f"£{ohlcv['Close'].iloc[-1]:.2f}", None
            ),
            "YTD Return":    (
                f"{ytd_ret.values[0]*100:.2f}%"
                if not ytd_ret.empty else "N/A", None
            ),
            "Period Return": (
                f"{period_ret.values[0]*100:.2f}%", None
            ),
            "Sharpe Ratio":  (
                f"{sharpe.values[0]:.2f}"
                if not sharpe.empty else "N/A", None
            ),
            "30D Volatility":(
                f"{latest_vol*100:.2f}%", None
            ),
        })

        st.plotly_chart(
            candlestick_chart(ohlcv, selected_ticker),
            use_container_width=True,
        )

        # Moving average explanation
        with st.expander("📖 How to read this chart"):
            st.markdown("""
            - **Candlesticks**: green = price rose that day,
              red = price fell
            - **20D MA** (amber): short-term trend direction
            - **50D MA** (purple): medium-term trend
            - **200D MA** (red): long-term trend. Price above 200D MA
              = uptrend
            - **Bollinger Bands** (grey dashed): price touching upper
              band = potentially overbought; lower band =
              potentially oversold
            - **Volume**: conviction behind price moves — high volume
              moves are more significant
            """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Volatility & Risk
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚡ Volatility & Risk":
    st.title("Volatility & Risk Analysis")
    st.markdown(
        "<p style='color:#94A3B8;'>Rolling volatility, Sharpe ratio, "
        "Beta, Value at Risk, and maximum drawdown across all stocks."
        "</p>",
        unsafe_allow_html=True,
    )

    # Calculate all risk metrics
    rolling_vol  = calculate_rolling_volatility(daily_returns)
    sharpe       = calculate_sharpe_ratio(daily_returns)
    beta         = calculate_beta(daily_returns, market_returns)
    var_95       = calculate_var(daily_returns, confidence=0.95)
    max_dd       = calculate_max_drawdown(prices)

    # Summary cards
    render_metric_cards({
        "Avg Sharpe Ratio":  (f"{sharpe.mean():.2f}",       None),
        "Avg Beta":          (f"{beta.mean():.2f}",         None),
        "Avg 30D Volatility":(
            f"{rolling_vol.iloc[-1].mean()*100:.2f}%",      None
        ),
        "Avg VaR (95%)":     (
            f"£{var_95.mean():.2f} per £1,000",             None
        ),
    })

    st.divider()

    # Rolling volatility chart
    st.plotly_chart(
        rolling_volatility_chart(rolling_vol),
        use_container_width=True,
    )

    st.divider()

    # Risk metrics comparison bar chart
    metrics_df = pd.DataFrame({
        "Ticker": sharpe.index,
        "Name":   [get_stock_name(t) for t in sharpe.index],
        "Sharpe": sharpe.values,
        "Beta":   [beta.get(t, np.nan) for t in sharpe.index],
        "VaR":    [var_95.get(t, np.nan) for t in sharpe.index],
    }).dropna()

    st.plotly_chart(
        risk_metrics_bar(metrics_df),
        use_container_width=True,
    )

    st.divider()

    # Max drawdown table
    st.subheader("Maximum Drawdown by Stock")
    st.markdown(
        "<p style='color:#94A3B8; font-size:13px;'>The worst "
        "peak-to-trough decline experienced over the selected "
        "period.</p>",
        unsafe_allow_html=True,
    )

    dd_df = pd.DataFrame({
        "Stock":        [get_stock_name(t) for t in max_dd.index],
        "Ticker":       max_dd.index,
        "Max Drawdown": [f"{v*100:.2f}%" for v in max_dd.values],
    }).sort_values("Max Drawdown").reset_index(drop=True)

    st.dataframe(dd_df, use_container_width=True, hide_index=True)

    with st.expander("📖 Understanding these metrics"):
        st.markdown("""
        - **Sharpe Ratio**: return per unit of risk. Above 1.0 is good,
          above 2.0 is excellent, below 0 means the stock isn't beating
          the risk-free rate (UK gilts at 4.5%) after adjusting for risk
        - **Beta**: sensitivity to market moves. Beta 1.5 means the stock
          typically moves 1.5× the market — up and down
        - **VaR 95%**: on 95% of trading days, you would not expect to
          lose more than this amount on a £1,000 investment
        - **Max Drawdown**: the worst loss from peak to trough.
          Crucial for understanding downside risk
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Correlation & Portfolio
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔗 Correlation & Portfolio":
    st.title("Correlation & Portfolio Analysis")
    st.markdown(
        "<p style='color:#94A3B8;'>Correlation matrix, efficient "
        "frontier, and optimal portfolio construction based on Modern "
        "Portfolio Theory.</p>",
        unsafe_allow_html=True,
    )

    # Correlation matrix
    corr_matrix = calculate_correlation_matrix(daily_returns)

    st.plotly_chart(
        correlation_heatmap(corr_matrix),
        use_container_width=True,
    )

    st.divider()

    # Cumulative returns
    portfolio_vals = calculate_portfolio_value(daily_returns)

    st.plotly_chart(
        cumulative_returns_chart(portfolio_vals),
        use_container_width=True,
    )

    st.divider()

    # Efficient frontier
    st.subheader("Efficient Frontier")
    st.markdown(
        "<p style='color:#94A3B8; font-size:13px;'>3,000 randomly "
        "generated portfolios. Each dot represents a different "
        "combination of stock weights. The ⭐ marks the optimal "
        "portfolio (maximum Sharpe ratio).</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Generating efficient frontier — simulating "
                    "3,000 portfolios..."):
        frontier_df = generate_efficient_frontier(daily_returns)
        optimal     = find_optimal_portfolio(daily_returns)

    st.plotly_chart(
        efficient_frontier_chart(frontier_df, optimal),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Optimal Portfolio Weights")
    st.markdown(
        "<p style='color:#94A3B8; font-size:13px;'>The portfolio "
        "allocation that maximises the Sharpe ratio — the best "
        "risk-adjusted return achievable from this stock universe."
        "</p>",
        unsafe_allow_html=True,
    )
    render_optimal_weights_table(optimal)

    with st.expander("📖 Understanding the efficient frontier"):
        st.markdown("""
        - **Each dot** is a portfolio with randomly assigned weights
          across all 17 stocks
        - **Colour** shows Sharpe ratio — yellow/green dots have better
          risk-adjusted returns
        - **The upper-left boundary** of the dot cloud is the efficient
          frontier — portfolios that give the best return for a given
          level of risk
        - **Portfolios below the frontier** are suboptimal — you could
          get the same return with less risk by rebalancing
        - **The star** is the tangency portfolio — mathematically the
          optimal combination of all 17 stocks
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Stock Screener
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Stock Screener":
    st.title("Stock Screener")
    st.markdown(
        "<p style='color:#94A3B8;'>Filter and rank stocks by "
        "fundamental and technical criteria. Export results to CSV."
        "</p>",
        unsafe_allow_html=True,
    )

    # Filter controls in a row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sector_filter = st.selectbox(
            "Sector",
            options=["All"] + sorted(get_sectors()),
        )
    with col2:
        min_sharpe = st.number_input(
            "Min Sharpe Ratio",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
        )
    with col3:
        max_pe = st.number_input(
            "Max P/E Ratio",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=1.0,
        )
    with col4:
        sort_by = st.selectbox(
            "Sort By",
            options=[
                "Sharpe Ratio",
                "YTD Return",
                "Annual Return",
                "Volatility",
            ],
        )

    # Build and filter the screener table
    with st.spinner("Loading fundamental data..."):
        screener_df = build_screener_table(prices, daily_returns)

    filtered_df = filter_screener(
        screener_df,
        sector=sector_filter,
        min_sharpe=min_sharpe if min_sharpe > -5.0 else None,
        max_pe=max_pe if max_pe < 200.0 else None,
    )

    ranked_df = rank_stocks(filtered_df, metric=sort_by)

    # Result count
    st.markdown(
        f"<p style='color:#94A3B8;'>Showing "
        f"<b style='color:#F1F5F9;'>{len(ranked_df)}</b> stocks</p>",
        unsafe_allow_html=True,
    )

    # Styled table
    st.dataframe(
        style_screener_table(ranked_df),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    # CSV export
    # Rationale: analysts always need to export data for further
    # analysis in Excel or Python. One-click export is a practical
    # feature that shows awareness of real analyst workflows.
    csv = ranked_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export to CSV",
        data=csv,
        file_name="equity_screener_results.csv",
        mime="text/csv",
    )
    