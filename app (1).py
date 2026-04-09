# ============================================================
# BTC MA Strategy – Streamlit App (PWA-ready)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BTC MA Strategy",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, p, label              { color: #e6edf3 !important; }

    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

    .stSlider > div > div > div { background: #58a6ff !important; }

    .stButton > button {
        background: #238636; color: #fff; border: none;
        border-radius: 6px; padding: 8px 20px; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #2ea043; }

    hr { border-color: #30363d; }

    .signal-buy {
        background: #1a4a2e; border: 1px solid #238636;
        color: #3fb950; padding: 10px 20px; border-radius: 8px;
        font-size: 1.1rem; font-weight: 700; text-align: center;
    }
    .signal-sell {
        background: #4a1a1a; border: 1px solid #da3633;
        color: #f85149; padding: 10px 20px; border-radius: 8px;
        font-size: 1.1rem; font-weight: 700; text-align: center;
    }
    .signal-label {
        color: #8b949e; font-size: 0.8rem; text-align: center; margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
TRADING_DAYS = 252

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e"),
)

AXIS_STYLE = dict(gridcolor="#21262d", zerolinecolor="#21262d")

# ── Backtesting function ─────────────────────────────────────
def backtest(df, sma_fast, sma_slow):
    df = df.copy()
    df[f"SMA_{sma_fast}"] = df["Close"].rolling(sma_fast).mean()
    df[f"SMA_{sma_slow}"] = df["Close"].rolling(sma_slow).mean()
    df = df.dropna()

    df["Signal"]          = (df[f"SMA_{sma_fast}"] > df[f"SMA_{sma_slow}"]).astype(int).shift(1)
    df = df.dropna()

    df["Market_Return"]   = df["Close"].pct_change()
    df["Strategy_Return"] = df["Market_Return"] * df["Signal"]
    df["Market_Cum"]      = (1 + df["Market_Return"]).cumprod()
    df["Strategy_Cum"]    = (1 + df["Strategy_Return"]).cumprod()
    df["Cross"]           = df["Signal"].diff()

    n_years      = len(df) / TRADING_DAYS
    rf_daily     = 0.02 / TRADING_DAYS
    total_return = df["Strategy_Cum"].iloc[-1] - 1
    cagr         = df["Strategy_Cum"].iloc[-1] ** (1 / n_years) - 1
    excess       = df["Strategy_Return"] - rf_daily
    sharpe       = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() > 0 else 0
    rolling_max  = df["Strategy_Cum"].cummax()
    drawdown     = (df["Strategy_Cum"] - rolling_max) / rolling_max
    max_dd       = drawdown.min()
    volatility   = df["Strategy_Return"].std() * np.sqrt(TRADING_DAYS)
    trades_in    = df[df["Signal"] == 1]["Strategy_Return"]
    win_rate     = (trades_in > 0).sum() / len(trades_in) if len(trades_in) > 0 else 0
    n_trades     = int((df["Signal"].diff() == 1).sum())
    calmar       = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "df":           df,
        "total_return": total_return,
        "cagr":         cagr,
        "sharpe":       sharpe,
        "calmar":       calmar,
        "max_dd":       max_dd,
        "volatility":   volatility,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
        "golden_cross": df[df["Cross"] ==  1],
        "death_cross":  df[df["Cross"] == -1],
    }

def bh_kpis(df):
    rf_daily = 0.02 / TRADING_DAYS
    n_years  = len(df) / TRADING_DAYS
    total    = df["Market_Cum"].iloc[-1] - 1
    cagr     = df["Market_Cum"].iloc[-1] ** (1 / n_years) - 1
    excess   = df["Market_Return"] - rf_daily
    sharpe   = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)
    roll_max = df["Market_Cum"].cummax()
    dd       = (df["Market_Cum"] - roll_max) / roll_max
    max_dd   = dd.min()
    vol      = df["Market_Return"].std() * np.sqrt(TRADING_DAYS)
    calmar   = cagr / abs(max_dd)
    return {"total": total, "cagr": cagr, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "vol": vol, "dd": dd}

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ₿ BTC MA Strategy")
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Technical Analysis",
        "📈 MA Strategy",
        "⚔️ vs. Buy & Hold",
        "🔬 Backtesting"
    ])

    st.markdown("---")
    ticker   = st.selectbox("Asset", ["BTC-EUR", "BTC-USD", "ETH-EUR", "ETH-USD"], index=0)
    currency = "€" if "EUR" in ticker else "$"

    if page == "📊 Technical Analysis":
        st.markdown("### Time Range")
        ta_range = st.radio("Period", ["1W", "3W", "6W", "3M"], index=3, horizontal=True)
        st.markdown("### Indicators")
        show_ema20  = st.checkbox("EMA 20",          value=True)
        show_ema50  = st.checkbox("EMA 50",          value=True)
        show_bb     = st.checkbox("Bollinger Bands", value=True)
        show_volume = st.checkbox("Volume",          value=True)
        show_rsi    = st.checkbox("RSI (14)",        value=True)
        show_macd   = st.checkbox("MACD",            value=True)

    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=date(2020, 1, 1))
        with col2:
            end_date = st.date_input("End", value=date.today())

        st.markdown("### Moving Averages")
        sma_fast = st.slider("SMA Fast", min_value=5,  max_value=100, value=10,  step=5)
        sma_slow = st.slider("SMA Slow", min_value=20, max_value=400, value=100, step=10)

        if sma_fast >= sma_slow:
            st.error("SMA Fast must be smaller than SMA Slow!")
            st.stop()

        run_multi = False
        if page == "🔬 Backtesting":
            run_multi = st.checkbox("Test multiple combinations", value=False)
            if run_multi:
                st.caption("Combinations: 10/50 · 10/100 · 10/200\n20/50 · 20/100 · 20/200\n50/100 · 50/200")

        run = st.button("🚀 Run Analysis")

# ════════════════════════════════════════════════════════════
# PAGE: TECHNICAL ANALYSIS
# ════════════════════════════════════════════════════════════
if page == "📊 Technical Analysis":

    st.markdown(f"# {ticker.split('-')[0]} — Technical Analysis")

    range_map = {"1W": 7, "3W": 21, "6W": 42, "3M": 90}
    days_back = range_map[ta_range]
    ta_start  = date.today() - timedelta(days=days_back + 200)

    with st.spinner("Loading data..."):
        df_ta_raw = yf.download(ticker, start=ta_start, end=date.today(), auto_adjust=True)
        if df_ta_raw.empty:
            st.error("No data found.")
            st.stop()
        df_ta_raw.columns = df_ta_raw.columns.get_level_values(0)

    df_ta = df_ta_raw.copy()

    # Indicators
    df_ta["EMA_20"]   = df_ta["Close"].ewm(span=20).mean()
    df_ta["EMA_50"]   = df_ta["Close"].ewm(span=50).mean()
    df_ta["BB_mid"]   = df_ta["Close"].rolling(20).mean()
    df_ta["BB_std"]   = df_ta["Close"].rolling(20).std()
    df_ta["BB_upper"] = df_ta["BB_mid"] + 2 * df_ta["BB_std"]
    df_ta["BB_lower"] = df_ta["BB_mid"] - 2 * df_ta["BB_std"]

    delta          = df_ta["Close"].diff()
    gain           = delta.clip(lower=0).rolling(14).mean()
    loss           = (-delta.clip(upper=0)).rolling(14).mean()
    rs             = gain / loss
    df_ta["RSI"]   = 100 - (100 / (1 + rs))

    df_ta["MACD"]        = df_ta["Close"].ewm(span=12).mean() - df_ta["Close"].ewm(span=26).mean()
    df_ta["MACD_signal"] = df_ta["MACD"].ewm(span=9).mean()
    df_ta["MACD_hist"]   = df_ta["MACD"] - df_ta["MACD_signal"]

    cutoff  = date.today() - timedelta(days=days_back)
    df_view = df_ta[df_ta.index.date >= cutoff]

    # Current signal summary
    last = df_ta.iloc[-1]

    ema_signal  = bool(last["EMA_20"] > last["EMA_50"])
    rsi_val     = float(last["RSI"])
    macd_signal = bool(last["MACD"] > last["MACD_signal"])
    bb_pos      = float((last["Close"] - last["BB_lower"]) / (last["BB_upper"] - last["BB_lower"]))

    st.markdown("### Current Signals")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        sig = "BUY" if ema_signal else "SELL"
        cls = "signal-buy" if ema_signal else "signal-sell"
        st.markdown(f'<div class="{cls}">{"📈" if ema_signal else "📉"} {sig}</div>'
                    f'<div class="signal-label">EMA 20 vs EMA 50</div>', unsafe_allow_html=True)
    with c2:
        rsi_buy = rsi_val < 70
        sig = "BUY" if rsi_buy else "OVERBOUGHT"
        cls = "signal-buy" if rsi_buy else "signal-sell"
        st.markdown(f'<div class="{cls}">{"📈" if rsi_buy else "⚠️"} {sig}</div>'
                    f'<div class="signal-label">RSI {rsi_val:.1f}</div>', unsafe_allow_html=True)
    with c3:
        sig = "BUY" if macd_signal else "SELL"
        cls = "signal-buy" if macd_signal else "signal-sell"
        st.markdown(f'<div class="{cls}">{"📈" if macd_signal else "📉"} {sig}</div>'
                    f'<div class="signal-label">MACD vs Signal</div>', unsafe_allow_html=True)
    with c4:
        bb_buy = bb_pos < 0.5
        sig = "LOW" if bb_buy else "HIGH"
        cls = "signal-buy" if bb_buy else "signal-sell"
        st.markdown(f'<div class="{cls}">{"📈" if bb_buy else "📉"} {sig}</div>'
                    f'<div class="signal-label">BB Position {bb_pos*100:.0f}%</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Build subplots
    subplot_specs = [[{"secondary_y": False}]]
    subplot_titles = ["Price"]
    if show_volume: subplot_specs.append([{"secondary_y": False}]); subplot_titles.append("Volume")
    if show_rsi:    subplot_specs.append([{"secondary_y": False}]); subplot_titles.append("RSI (14)")
    if show_macd:   subplot_specs.append([{"secondary_y": False}]); subplot_titles.append("MACD")

    rows = len(subplot_specs)
    row_heights = [0.55] + [0.15] * (rows - 1)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_view.index,
        open=df_view["Open"].squeeze(),
        high=df_view["High"].squeeze(),
        low=df_view["Low"].squeeze(),
        close=df_view["Close"].squeeze(),
        name="Price",
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
        increasing_fillcolor="#1a4a2e",
        decreasing_fillcolor="#4a1a1a",
    ), row=1, col=1)

    if show_ema20:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["EMA_20"].squeeze(),
            name="EMA 20", line=dict(color="#58a6ff", width=1.5),
        ), row=1, col=1)

    if show_ema50:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["EMA_50"].squeeze(),
            name="EMA 50", line=dict(color="#e67e22", width=1.5),
        ), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["BB_upper"].squeeze(),
            name="BB Upper", line=dict(color="#8b949e", width=1, dash="dash"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["BB_lower"].squeeze(),
            name="BB Lower", line=dict(color="#8b949e", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(139,148,158,0.06)",
        ), row=1, col=1)

    cur_row = 2

    if show_volume:
        vol_colors = ["#1a4a2e" if c >= o else "#4a1a1a"
                      for c, o in zip(df_view["Close"].squeeze(), df_view["Open"].squeeze())]
        fig.add_trace(go.Bar(
            x=df_view.index, y=df_view["Volume"].squeeze(),
            name="Volume", marker_color=vol_colors, showlegend=False,
        ), row=cur_row, col=1)
        cur_row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["RSI"].squeeze(),
            name="RSI", line=dict(color="#d2a8ff", width=1.5),
        ), row=cur_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.5, row=cur_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.5, row=cur_row, col=1)
        fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
        cur_row += 1

    if show_macd:
        macd_colors = ["#3fb950" if v >= 0 else "#f85149"
                       for v in df_view["MACD_hist"].squeeze()]
        fig.add_trace(go.Bar(
            x=df_view.index, y=df_view["MACD_hist"].squeeze(),
            name="MACD Hist", marker_color=macd_colors, showlegend=False,
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["MACD"].squeeze(),
            name="MACD", line=dict(color="#58a6ff", width=1.5),
        ), row=cur_row, col=1)
        fig.add_trace(go.Scatter(
            x=df_view.index, y=df_view["MACD_signal"].squeeze(),
            name="Signal Line", line=dict(color="#e67e22", width=1.5),
        ), row=cur_row, col=1)

    fig.update_layout(
        **PLOT_THEME,
        title=f"{ticker} — {ta_range} Chart",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        height=700 + (rows - 1) * 130,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(**AXIS_STYLE, row=i, col=1)
        fig.update_yaxes(**AXIS_STYLE, row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Period Statistics"):
        price_now  = float(df_view["Close"].iloc[-1])
        price_open = float(df_view["Close"].iloc[0])
        high       = float(df_view["High"].max())
        low        = float(df_view["Low"].min())
        chg        = (price_now - price_open) / price_open * 100
        vol_avg    = float(df_view["Volume"].mean())

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Current Price", f"{currency}{price_now:,.0f}")
        s2.metric("Period Change", f"{chg:+.2f}%")
        s3.metric("Period High",   f"{currency}{high:,.0f}")
        s4.metric("Period Low",    f"{currency}{low:,.0f}")
        s5.metric("Avg. Volume",   f"{vol_avg:,.0f}")

# ════════════════════════════════════════════════════════════
# PAGES: MA STRATEGY / BUY & HOLD / BACKTESTING
# ════════════════════════════════════════════════════════════
else:
    st.markdown(f"# {ticker.split('-')[0]} Moving Average Strategy")

    if not run:
        st.info("👈 Set parameters in the sidebar and click **Run Analysis**.")
        st.stop()

    with st.spinner("Loading data..."):
        df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df_raw.empty:
            st.error("No data found. Please adjust the date range or asset.")
            st.stop()
        df_raw = df_raw[["Close"]].dropna()
        df_raw.columns = ["Close"]

    r  = backtest(df_raw, sma_fast, sma_slow)
    df = r["df"]
    bh = bh_kpis(df)

    # ── Current Buy/Sell Signal ───────────────────────────────
    current_fast = float(df[f"SMA_{sma_fast}"].iloc[-1])
    current_slow = float(df[f"SMA_{sma_slow}"].iloc[-1])
    is_buy       = current_fast > current_slow
    prev_fast    = float(df[f"SMA_{sma_fast}"].iloc[-2])
    prev_slow    = float(df[f"SMA_{sma_slow}"].iloc[-2])
    just_crossed = (current_fast > current_slow) != (prev_fast > prev_slow)

    sig_col1, sig_col2, sig_col3 = st.columns([1, 1, 2])
    with sig_col1:
        sig = "BUY" if is_buy else "SELL"
        cls = "signal-buy" if is_buy else "signal-sell"
        st.markdown(
            f'<div class="{cls}">{"📈" if is_buy else "📉"} {sig} SIGNAL</div>'
            f'<div class="signal-label">SMA {sma_fast} vs SMA {sma_slow}</div>',
            unsafe_allow_html=True
        )
    with sig_col2:
        st.metric(f"SMA {sma_fast}", f"{currency}{current_fast:,.0f}")
        st.metric(f"SMA {sma_slow}", f"{currency}{current_slow:,.0f}")
    with sig_col3:
        if just_crossed:
            cross_type = "🟡 Golden Cross just triggered!" if is_buy else "🟡 Death Cross just triggered!"
            st.warning(cross_type)
        else:
            days_in_signal = 0
            for i in range(len(df) - 1, -1, -1):
                if (df[f"SMA_{sma_fast}"].iloc[i] > df[f"SMA_{sma_slow}"].iloc[i]) == is_buy:
                    days_in_signal += 1
                else:
                    break
            trend = "uptrend" if is_buy else "downtrend"
            st.info(f"Current {trend} active for **{days_in_signal} days**")

    st.markdown("---")

    # ── MA STRATEGY ───────────────────────────────────────────
    if page == "📈 MA Strategy":

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return",  f"{r['total_return']*100:.1f}%",
                  delta=f"{(r['total_return']-bh['total'])*100:+.1f}% vs B&H")
        c2.metric("CAGR",          f"{r['cagr']*100:.1f}%")
        c3.metric("Sharpe Ratio",  f"{r['sharpe']:.2f}",
                  delta=f"{r['sharpe']-bh['sharpe']:+.2f} vs B&H")
        c4.metric("Max Drawdown",  f"{r['max_dd']*100:.1f}%",
                  delta=f"{(r['max_dd']-bh['max_dd'])*100:+.1f}% vs B&H", delta_color="inverse")
        c5.metric("No. of Trades", str(r["n_trades"]))

        st.markdown("---")

        shapes = []
        prev_idx    = df.index[0]
        prev_signal = df["Signal"].iloc[0]
        for i in range(1, len(df)):
            curr = df["Signal"].iloc[i]
            if curr != prev_signal or i == len(df) - 1:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=prev_idx, x1=df.index[i], y0=0, y1=1,
                    fillcolor="#238636" if prev_signal == 1 else "#da3633",
                    opacity=0.07, line_width=0, layer="below"
                ))
                prev_idx    = df.index[i]
                prev_signal = curr

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"].squeeze(),
            name="Price", line=dict(color="#8b949e", width=1),
            hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Price: " + currency + "%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"SMA_{sma_fast}"].squeeze(),
            name=f"SMA {sma_fast}", line=dict(color="#58a6ff", width=2),
            hovertemplate=f"SMA {sma_fast}: " + currency + "%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f"SMA_{sma_slow}"].squeeze(),
            name=f"SMA {sma_slow}", line=dict(color="#e67e22", width=2.5),
            hovertemplate=f"SMA {sma_slow}: " + currency + "%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=r["golden_cross"].index, y=r["golden_cross"][f"SMA_{sma_fast}"].squeeze(),
            name="Golden Cross ↑", mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#3fb950"),
            hovertemplate="<b>Golden Cross</b><br>%{x|%d.%m.%Y}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=r["death_cross"].index, y=r["death_cross"][f"SMA_{sma_fast}"].squeeze(),
            name="Death Cross ↓", mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="#f85149"),
            hovertemplate="<b>Death Cross</b><br>%{x|%d.%m.%Y}<extra></extra>"
        ))
        fig.update_layout(
            **PLOT_THEME,
            shapes=shapes,
            title=f"SMA {sma_fast} vs SMA {sma_slow}",
            xaxis=dict(
                **AXIS_STYLE,
                title="Date",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    bgcolor="#161b22", activecolor="#238636",
                    buttons=[
                        dict(count=3,  label="3M", step="month", stepmode="backward"),
                        dict(count=6,  label="6M", step="month", stepmode="backward"),
                        dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            yaxis=dict(
                **AXIS_STYLE,
                title=f"Price ({currency})",
                tickprefix=currency,
                tickformat=","
            ),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Show all KPIs"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Strategy**")
                st.dataframe(pd.DataFrame({
                    "KPI": ["Total Return", "CAGR", "Sharpe Ratio", "Calmar Ratio",
                            "Max Drawdown", "Volatility", "Win Rate", "Trades"],
                    "Value": [
                        f"{r['total_return']*100:.2f}%",
                        f"{r['cagr']*100:.2f}%",
                        f"{r['sharpe']:.3f}",
                        f"{r['calmar']:.3f}",
                        f"{r['max_dd']*100:.2f}%",
                        f"{r['volatility']*100:.2f}%",
                        f"{r['win_rate']*100:.2f}%",
                        str(r["n_trades"]),
                    ]
                }), hide_index=True, use_container_width=True)
            with col_b:
                st.markdown("**Buy & Hold**")
                st.dataframe(pd.DataFrame({
                    "KPI": ["Total Return", "CAGR", "Sharpe Ratio", "Calmar Ratio",
                            "Max Drawdown", "Volatility", "–", "–"],
                    "Value": [
                        f"{bh['total']*100:.2f}%",
                        f"{bh['cagr']*100:.2f}%",
                        f"{bh['sharpe']:.3f}",
                        f"{bh['calmar']:.3f}",
                        f"{bh['max_dd']*100:.2f}%",
                        f"{bh['vol']*100:.2f}%",
                        "–", "–",
                    ]
                }), hide_index=True, use_container_width=True)

    # ── VS BUY & HOLD ─────────────────────────────────────────
    elif page == "⚔️ vs. Buy & Hold":

        strat_dd = (df["Strategy_Cum"] - df["Strategy_Cum"].cummax()) / df["Strategy_Cum"].cummax()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=df.index, y=df["Market_Cum"],
            name="Buy & Hold", line=dict(color="#8b949e", width=2, dash="dot"),
            hovertemplate="<b>Buy & Hold</b><br>%{x|%d.%m.%Y}<br>%{y:.2f}x<extra></extra>"
        ))
        fig_eq.add_trace(go.Scatter(
            x=df.index, y=df["Strategy_Cum"],
            name=f"SMA {sma_fast}/{sma_slow}", line=dict(color="#58a6ff", width=2),
            hovertemplate=f"<b>Strategy</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.2f}}x<extra></extra>"
        ))
        fig_eq.update_layout(
            **PLOT_THEME,
            title="Equity Curves",
            xaxis=dict(**AXIS_STYLE, title="Date"),
            yaxis=dict(**AXIS_STYLE, title="Growth (1x = start)", tickformat=".1f"),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=400,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=bh["dd"] * 100,
            name="Buy & Hold", line=dict(color="#8b949e", width=1.5),
            fill="tozeroy", fillcolor="rgba(139,148,158,0.1)",
            hovertemplate="<b>B&H DD</b><br>%{x|%d.%m.%Y}<br>%{y:.1f}%<extra></extra>"
        ))
        fig_dd.add_trace(go.Scatter(
            x=df.index, y=strat_dd * 100,
            name=f"SMA {sma_fast}/{sma_slow}", line=dict(color="#f85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.1)",
            hovertemplate="<b>Strategy DD</b><br>%{x|%d.%m.%Y}<br>%{y:.1f}%<extra></extra>"
        ))
        fig_dd.update_layout(
            **PLOT_THEME,
            title="Drawdown Comparison",
            xaxis=dict(**AXIS_STYLE, title="Date"),
            yaxis=dict(**AXIS_STYLE, title="Drawdown (%)", ticksuffix="%"),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=300,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        rf_daily = 0.02 / TRADING_DAYS
        roll_s   = df["Strategy_Return"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else 0)
        roll_bh  = df["Market_Return"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean() - rf_daily) / x.std() * np.sqrt(TRADING_DAYS) if x.std() > 0 else 0)

        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=df.index, y=roll_bh,
            name="Buy & Hold", line=dict(color="#8b949e", width=1.5, dash="dot"),
            hovertemplate="<b>B&H Sharpe</b><br>%{x|%d.%m.%Y}<br>%{y:.2f}<extra></extra>"
        ))
        fig_rs.add_trace(go.Scatter(
            x=df.index, y=roll_s,
            name=f"SMA {sma_fast}/{sma_slow}", line=dict(color="#58a6ff", width=1.5),
            hovertemplate="<b>Strategy Sharpe</b><br>%{x|%d.%m.%Y}<br>%{y:.2f}<extra></extra>"
        ))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="#30363d")
        fig_rs.add_hline(y=1, line_dash="dash", line_color="#238636", opacity=0.5,
                         annotation_text="Sharpe = 1", annotation_font_color="#3fb950")
        fig_rs.update_layout(
            **PLOT_THEME,
            title="Rolling Sharpe Ratio (252 days)",
            xaxis=dict(**AXIS_STYLE, title="Date"),
            yaxis=dict(**AXIS_STYLE, title="Sharpe Ratio"),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=300,
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    # ── BACKTESTING ───────────────────────────────────────────
    elif page == "🔬 Backtesting":
        if not run_multi:
            st.info("Enable **'Test multiple combinations'** in the sidebar to use this tab.")
        else:
            combos = [(10,50),(10,100),(10,200),(20,50),(20,100),(20,200),(50,100),(50,200)]
            results = []
            prog = st.progress(0, text="Running backtests...")
            for i, (f, s) in enumerate(combos):
                res = backtest(df_raw, f, s)
                results.append({
                    "Combination":      f"SMA {f}/{s}",
                    "Total Return (%)": round(res["total_return"] * 100, 2),
                    "CAGR (%)":         round(res["cagr"] * 100, 2),
                    "Sharpe Ratio":     round(res["sharpe"], 3),
                    "Calmar Ratio":     round(res["calmar"], 3),
                    "Max Drawdown (%)": round(res["max_dd"] * 100, 2),
                    "Win Rate (%)":     round(res["win_rate"] * 100, 2),
                    "Trades":           res["n_trades"],
                    "_eq":              res["df"]["Strategy_Cum"],
                    "_df":              res["df"],
                })
                prog.progress((i + 1) / len(combos), text=f"SMA {f}/{s} done...")
            prog.empty()

            df_res     = pd.DataFrame(results).sort_values("Sharpe Ratio", ascending=False)
            best_label = df_res.iloc[0]["Combination"]

            st.success(f"🏆 Best strategy: **{best_label}** | "
                       f"Sharpe: {df_res.iloc[0]['Sharpe Ratio']} | "
                       f"Return: {df_res.iloc[0]['Total Return (%)']}% | "
                       f"Max DD: {df_res.iloc[0]['Max Drawdown (%)']}%")

            st.dataframe(
                df_res.drop(columns=["_eq", "_df"]).set_index("Combination"),
                use_container_width=True
            )

            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(
                x=df_res["Sharpe Ratio"],
                y=df_res["Total Return (%)"],
                mode="markers+text",
                text=df_res["Combination"],
                textposition="top center",
                marker=dict(
                    size=-df_res["Max Drawdown (%)"],
                    color=df_res["Sharpe Ratio"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Sharpe"),
                    line=dict(width=1, color="#30363d")
                ),
                hovertemplate="<b>%{text}</b><br>Sharpe: %{x:.3f}<br>Return: %{y:.1f}%<extra></extra>"
            ))
            fig_b.update_layout(
                **PLOT_THEME,
                title="Sharpe vs. Return (bubble size = Max Drawdown)",
                xaxis=dict(**AXIS_STYLE, title="Sharpe Ratio"),
                yaxis=dict(**AXIS_STYLE, title="Total Return (%)"),
                height=450,
            )
            st.plotly_chart(fig_b, use_container_width=True)

            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(
                x=results[0]["_df"].index, y=results[0]["_df"]["Market_Cum"],
                name="Buy & Hold", line=dict(color="#8b949e", width=2, dash="dot"),
            ))
            colors = px.colors.qualitative.Set2
            for i, res in enumerate(results):
                fig_all.add_trace(go.Scatter(
                    x=res["_eq"].index, y=res["_eq"],
                    name=res["Combination"],
                    line=dict(color=colors[i % len(colors)], width=1.8),
                ))
            fig_all.update_layout(
                **PLOT_THEME,
                title="All Equity Curves vs. Buy & Hold",
                xaxis=dict(**AXIS_STYLE, title="Date"),
                yaxis=dict(**AXIS_STYLE, title="Growth (1x = start)", tickformat=".1f"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0),
                height=500,
            )
            st.plotly_chart(fig_all, use_container_width=True)
