# ============================================================
# BTC MA Strategy – Streamlit App (PWA-ready, Mobile-optimized)
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
    initial_sidebar_state="collapsed",
)

# ── Mobile-first CSS ─────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, p, label, div         { color: #e6edf3; }

    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        max-width: 100% !important;
    }

    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px 12px !important;
    }
    [data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 0.72rem !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.7rem !important; }

    .stSlider > div > div > div { background: #58a6ff !important; }

    .stButton > button {
        background: #238636; color: #fff; border: none;
        border-radius: 8px; padding: 10px 20px; font-weight: 600;
        width: 100%; font-size: 1rem;
        touch-action: manipulation;
    }
    .stButton > button:hover  { background: #2ea043; }
    .stButton > button:active { background: #196127; }

    .signal-buy {
        background: #1a4731; border: 1px solid #3fb950;
        border-radius: 10px; padding: 12px 8px;
        text-align: center; color: #3fb950 !important;
        font-size: 0.95rem; font-weight: 700; line-height: 1.4;
    }
    .signal-sell {
        background: #4a1a1a; border: 1px solid #f85149;
        border-radius: 10px; padding: 12px 8px;
        text-align: center; color: #f85149 !important;
        font-size: 0.95rem; font-weight: 700; line-height: 1.4;
    }
    .signal-label {
        font-size: 0.7rem; color: #8b949e !important;
        margin-bottom: 4px; text-align: center;
    }

    .price-banner {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 12px; padding: 14px 16px;
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 12px;
    }
    .price-main      { font-size: 1.6rem; font-weight: 700; color: #e6edf3; }
    .price-delta-pos { font-size: 0.9rem; color: #3fb950; font-weight: 600; }
    .price-delta-neg { font-size: 0.9rem; color: #f85149; font-weight: 600; }

    [data-testid="stExpander"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem !important; padding: 8px 10px !important;
    }

    .main .block-container { padding-bottom: 1.5rem !important; }

    ::-webkit-scrollbar { width: 3px; height: 3px; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

    @media (min-width: 768px) {
        [data-testid="stSidebar"] { min-width: 280px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────
TRADING_DAYS = 252
PLOT_THEME   = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=11))
AXIS_STYLE   = dict(gridcolor="#21262d", zerolinecolor="#21262d")

# ════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════
def flatten_columns(df):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

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
    return {"df": df, "total_return": total_return, "cagr": cagr, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "volatility": volatility,
            "win_rate": win_rate, "n_trades": n_trades,
            "golden_cross": df[df["Cross"] ==  1],
            "death_cross":  df[df["Cross"] == -1]}

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

def compute_indicators(df):
    df = df.copy()
    delta             = df["Close"].diff()
    gain              = delta.clip(lower=0).rolling(14).mean()
    loss              = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss))
    ema12             = df["Close"].ewm(span=12, adjust=False).mean()
    ema26             = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["BB_Mid"]      = df["Close"].rolling(20).mean()
    bb_std            = df["Close"].rolling(20).std()
    df["BB_Upper"]    = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"]    = df["BB_Mid"] - 2 * bb_std
    if "Volume" in df.columns:
        df["Vol_MA"]  = df["Volume"].rolling(20).mean()
    return df

def mobile_layout(fig, height=340, title=""):
    fig.update_layout(
        **PLOT_THEME,
        height=height,
        margin=dict(l=4, r=4, t=36 if title else 8, b=4),
        hovermode="x unified",
        hoverlabel=dict(font_size=10),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    page = st.radio("Page", [
        "📊 Technical Analysis",
        "📈 MA Strategy",
        "⚔️ vs. Buy & Hold",
        "🔬 Backtesting",
    ], index=0)

    st.markdown("---")
    ticker   = st.selectbox("Asset", ["BTC-EUR", "BTC-USD", "ETH-EUR", "ETH-USD"])
    currency = "€" if "EUR" in ticker else "$"

    if page == "📊 Technical Analysis":
        st.markdown("### Chart")
        ta_range    = st.radio("Period", ["1W", "3W", "6W", "3M"], index=3, horizontal=True)
        show_bb     = st.checkbox("Bollinger Bands", value=True)
        show_rsi    = st.checkbox("RSI (14)",        value=True)
        show_macd   = st.checkbox("MACD",            value=True)
        show_vol    = st.checkbox("Volume",          value=False)
        st.markdown("### Signal SMAs")
        ta_sma_fast = st.slider("Fast", 5,  100, 10,  5)
        ta_sma_slow = st.slider("Slow", 20, 400, 100, 10)
    else:
        st.markdown("### Date Range")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start", value=date(2020, 1, 1))
        end_date   = col2.date_input("End",   value=date.today())
        st.markdown("### Moving Averages")
        sma_fast   = st.slider("Fast", 5,  100, 10,  5)
        sma_slow   = st.slider("Slow", 20, 400, 100, 10)
        if sma_fast >= sma_slow:
            st.error("SMA Fast must be smaller than SMA Slow!")
            st.stop()
        run_multi = False
        if page == "🔬 Backtesting":
            run_multi = st.checkbox("Test all combinations", value=False)
        run = st.button("🚀 Run Analysis")

# ════════════════════════════════════════════════════════════
# PAGE: TECHNICAL ANALYSIS
# ════════════════════════════════════════════════════════════
if page == "📊 Technical Analysis":

    st.markdown(f"### ₿ {ticker}  –  Technical Analysis")

    range_map = {"1W": 7, "3W": 21, "6W": 42, "3M": 90}
    days_back = range_map[ta_range]
    ta_start  = date.today() - timedelta(days=days_back)
    interval  = "1h" if days_back <= 21 else "1d"

    with st.spinner("Loading..."):
        df_ta = yf.download(ticker, start=ta_start, end=date.today(),
                            interval=interval, auto_adjust=True)
        if df_ta.empty:
            st.error("No data found.")
            st.stop()
        df_ta = flatten_columns(df_ta).dropna()
        df_ta = compute_indicators(df_ta)

        df_sig = yf.download(ticker, start=date.today() - timedelta(days=400),
                             end=date.today(), interval="1d", auto_adjust=True)
        df_sig = flatten_columns(df_sig)[["Close"]].dropna()
        df_sig.columns = ["Close"]

    # Price banner
    price_now = float(df_ta["Close"].iloc[-1])
    price_old = float(df_ta["Close"].iloc[0])
    pct       = (price_now - price_old) / price_old * 100
    delta_cls = "price-delta-pos" if pct >= 0 else "price-delta-neg"
    arrow     = "▲" if pct >= 0 else "▼"

    st.markdown(f"""
    <div class="price-banner">
      <div>
        <div style="font-size:0.72rem;color:#8b949e;">{ticker} · {ta_range}</div>
        <div class="price-main">{currency}{price_now:,.0f}</div>
      </div>
      <div class="{delta_cls}">{arrow} {pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # MA Signal badges
    sma_combos = list(dict.fromkeys([(ta_sma_fast, ta_sma_slow),(10,50),(20,50),(50,200)]))
    sig_cols   = st.columns(len(sma_combos))
    for col, (f, s) in zip(sig_cols, sma_combos):
        if len(df_sig) >= s:
            sf  = float(df_sig["Close"].rolling(f).mean().iloc[-1])
            ss  = float(df_sig["Close"].rolling(s).mean().iloc[-1])
            buy = sf > ss
            col.markdown(
                f'<div class="signal-label">SMA {f}/{s}</div>'
                f'<div class="{"signal-buy" if buy else "signal-sell"}">'
                f'{"🟢 BUY" if buy else "🔴 SELL"}</div>',
                unsafe_allow_html=True)
        else:
            col.markdown(
                f'<div class="signal-label">SMA {f}/{s}</div>'
                f'<div class="signal-sell">N/A</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick stats
    rsi_now = float(df_ta["RSI"].iloc[-1])
    high    = float(df_ta["High"].max()) if "High" in df_ta.columns else float(df_ta["Close"].max())
    low     = float(df_ta["Low"].min())  if "Low"  in df_ta.columns else float(df_ta["Close"].min())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ta_range} High", f"{currency}{high:,.0f}")
    c2.metric(f"{ta_range} Low",  f"{currency}{low:,.0f}")
    c3.metric("RSI (14)", f"{rsi_now:.1f}",
              delta="Overbought" if rsi_now > 70 else ("Oversold" if rsi_now < 30 else "Neutral"))
    c4.metric("Candles", str(len(df_ta)))

    st.markdown("---")

    # Build subplots
    row_specs   = [[{}]]
    row_heights = [0.55]
    subtitles   = ["Price"]
    if show_rsi:
        row_specs.append([{}]); row_heights.append(0.18); subtitles.append("RSI (14)")
    if show_macd:
        row_specs.append([{}]); row_heights.append(0.18); subtitles.append("MACD")
    if show_vol and "Volume" in df_ta.columns:
        row_specs.append([{}]); row_heights.append(0.14); subtitles.append("Volume")

    n_rows = len(row_specs)
    fig_ta = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                           row_heights=row_heights, vertical_spacing=0.04,
                           subplot_titles=subtitles)

    if all(c in df_ta.columns for c in ["Open","High","Low","Close"]):
        fig_ta.add_trace(go.Candlestick(
            x=df_ta.index,
            open=df_ta["Open"].squeeze(), high=df_ta["High"].squeeze(),
            low=df_ta["Low"].squeeze(),   close=df_ta["Close"].squeeze(),
            name="Price",
            increasing_line_color="#3fb950", decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149",
        ), row=1, col=1)
    else:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["Close"].squeeze(),
            name="Price", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)

    df_ta[f"SMA_{ta_sma_fast}"] = df_ta["Close"].rolling(ta_sma_fast).mean()
    df_ta[f"SMA_{ta_sma_slow}"] = df_ta["Close"].rolling(ta_sma_slow).mean()
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[f"SMA_{ta_sma_fast}"].squeeze(),
        name=f"SMA {ta_sma_fast}", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta[f"SMA_{ta_sma_slow}"].squeeze(),
        name=f"SMA {ta_sma_slow}", line=dict(color="#e67e22", width=1.5)), row=1, col=1)

    if show_bb:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["BB_Upper"].squeeze(),
            name="BB Upper", line=dict(color="#8b949e", width=1, dash="dot")), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["BB_Lower"].squeeze(),
            name="BB Lower", line=dict(color="#8b949e", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,148,158,0.05)"), row=1, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["BB_Mid"].squeeze(),
            name="BB Mid", line=dict(color="#8b949e", width=1, dash="dash")), row=1, col=1)

    cur_row = 2
    if show_rsi:
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["RSI"].squeeze(),
            name="RSI", line=dict(color="#a371f7", width=1.5)), row=cur_row, col=1)
        fig_ta.add_hline(y=70, line_dash="dash", line_color="#f85149", opacity=0.5,
                         row=cur_row, col=1)
        fig_ta.add_hline(y=30, line_dash="dash", line_color="#3fb950", opacity=0.5,
                         row=cur_row, col=1)
        fig_ta.update_yaxes(range=[0, 100], row=cur_row, col=1)
        cur_row += 1

    if show_macd:
        hist_colors = ["#3fb950" if v >= 0 else "#f85149"
                       for v in df_ta["MACD_Hist"].squeeze()]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta["MACD_Hist"].squeeze(),
            name="MACD Hist", marker_color=hist_colors, opacity=0.6), row=cur_row, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["MACD"].squeeze(),
            name="MACD", line=dict(color="#58a6ff", width=1.5)), row=cur_row, col=1)
        fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["MACD_Signal"].squeeze(),
            name="Signal", line=dict(color="#e67e22", width=1.5)), row=cur_row, col=1)
        cur_row += 1

    if show_vol and "Volume" in df_ta.columns:
        vc = ["#3fb950" if float(df_ta["Close"].squeeze().iloc[i]) >= float(df_ta["Close"].squeeze().iloc[i-1])
              else "#f85149" for i in range(len(df_ta))]
        fig_ta.add_trace(go.Bar(x=df_ta.index, y=df_ta["Volume"].squeeze(),
            name="Volume", marker_color=vc, opacity=0.5), row=cur_row, col=1)
        if "Vol_MA" in df_ta.columns:
            fig_ta.add_trace(go.Scatter(x=df_ta.index, y=df_ta["Vol_MA"].squeeze(),
                name="Vol MA", line=dict(color="#8b949e", width=1.2)), row=cur_row, col=1)

    total_h = 360 + (n_rows - 1) * 130
    fig_ta.update_layout(
        **PLOT_THEME, height=total_h,
        margin=dict(l=4, r=4, t=28, b=4),
        hovermode="x unified", hoverlabel=dict(font_size=10),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, n_rows + 1):
        fig_ta.update_xaxes(**AXIS_STYLE, row=i, col=1)
        fig_ta.update_yaxes(**AXIS_STYLE, row=i, col=1)

    st.plotly_chart(fig_ta, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📋 Signal Summary"):
        macd_v = float(df_ta["MACD"].iloc[-1])
        sig_v  = float(df_ta["MACD_Signal"].iloc[-1])
        bb_u   = float(df_ta["BB_Upper"].iloc[-1])
        bb_l   = float(df_ta["BB_Lower"].iloc[-1])
        st.markdown("**RSI**")
        if rsi_now > 70:   st.error(f"RSI {rsi_now:.1f} – Overbought")
        elif rsi_now < 30: st.success(f"RSI {rsi_now:.1f} – Oversold")
        else:              st.info(f"RSI {rsi_now:.1f} – Neutral")
        st.markdown("**MACD**")
        if macd_v > sig_v: st.success("MACD above signal – Bullish momentum")
        else:              st.error("MACD below signal – Bearish momentum")
        st.markdown("**Bollinger Bands**")
        if price_now > bb_u:   st.error("Price above upper band – Overbought")
        elif price_now < bb_l: st.success("Price below lower band – Oversold")
        else:
            pos = (price_now - bb_l) / (bb_u - bb_l) * 100
            st.info(f"Price at {pos:.0f}% of BB range")

# ════════════════════════════════════════════════════════════
# OTHER PAGES
# ════════════════════════════════════════════════════════════
else:
    st.markdown(f"### ₿ {ticker.split('-')[0]} – {page.split(' ',1)[1]}")

    if not run:
        st.info("👈 Open sidebar, set parameters and tap **Run Analysis**.")
        st.stop()

    with st.spinner("Loading data..."):
        df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df_raw.empty:
            st.error("No data found.")
            st.stop()
        df_raw = flatten_columns(df_raw)[["Close"]].dropna()
        df_raw.columns = ["Close"]

    r  = backtest(df_raw, sma_fast, sma_slow)
    df = r["df"]
    bh = bh_kpis(df)

    # Current signal banner
    sf_val  = float(df[f"SMA_{sma_fast}"].iloc[-1])
    ss_val  = float(df[f"SMA_{sma_slow}"].iloc[-1])
    is_buy  = sf_val > ss_val
    gap_pct = (sf_val - ss_val) / ss_val * 100

    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.markdown(
            f'<div class="signal-label">SMA {sma_fast} / {sma_slow}</div>'
            f'<div class="{"signal-buy" if is_buy else "signal-sell"}">'
            f'{"🟢 BUY SIGNAL" if is_buy else "🔴 SELL SIGNAL"}</div>',
            unsafe_allow_html=True)
    with scol2:
        st.metric(f"SMA {sma_fast}", f"{currency}{sf_val:,.0f}")
        st.metric(f"SMA {sma_slow}", f"{currency}{ss_val:,.0f}",
                  delta=f"Gap {gap_pct:+.2f}%")

    st.markdown("---")

    # ── MA STRATEGY ──────────────────────────────────────────
    if page == "📈 MA Strategy":

        c1, c2 = st.columns(2)
        c1.metric("Total Return", f"{r['total_return']*100:.1f}%",
                  delta=f"{(r['total_return']-bh['total'])*100:+.1f}% vs B&H")
        c2.metric("CAGR", f"{r['cagr']*100:.1f}%")
        c3, c4 = st.columns(2)
        c3.metric("Sharpe", f"{r['sharpe']:.2f}",
                  delta=f"{r['sharpe']-bh['sharpe']:+.2f} vs B&H")
        c4.metric("Max DD", f"{r['max_dd']*100:.1f}%",
                  delta=f"{(r['max_dd']-bh['max_dd'])*100:+.1f}%", delta_color="inverse")

        st.markdown("---")

        shapes, prev_idx, prev_sig = [], df.index[0], df["Signal"].iloc[0]
        for i in range(1, len(df)):
            curr = df["Signal"].iloc[i]
            if curr != prev_sig or i == len(df) - 1:
                shapes.append(dict(
                    type="rect", xref="x", yref="paper",
                    x0=prev_idx, x1=df.index[i], y0=0, y1=1,
                    fillcolor="#238636" if prev_sig == 1 else "#da3633",
                    opacity=0.07, line_width=0, layer="below"))
                prev_idx, prev_sig = df.index[i], curr

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(),
            name="Price", line=dict(color="#8b949e", width=1),
            hovertemplate="<b>%{x|%d.%m.%Y}</b><br>" + currency + "%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{sma_fast}"].squeeze(),
            name=f"SMA {sma_fast}", line=dict(color="#58a6ff", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{sma_slow}"].squeeze(),
            name=f"SMA {sma_slow}", line=dict(color="#e67e22", width=2)))
        fig.add_trace(go.Scatter(x=r["golden_cross"].index,
            y=r["golden_cross"][f"SMA_{sma_fast}"].squeeze(),
            name="Golden Cross ↑", mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#3fb950")))
        fig.add_trace(go.Scatter(x=r["death_cross"].index,
            y=r["death_cross"][f"SMA_{sma_fast}"].squeeze(),
            name="Death Cross ↓", mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#f85149")))

        fig.update_layout(
            **PLOT_THEME, shapes=shapes, height=380,
            margin=dict(l=4, r=4, t=8, b=4),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=9),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(**AXIS_STYLE,
                rangeselector=dict(
                    bgcolor="#161b22", activecolor="#238636", font=dict(size=10),
                    buttons=[
                        dict(count=3,  label="3M", step="month", stepmode="backward"),
                        dict(count=6,  label="6M", step="month", stepmode="backward"),
                        dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                        dict(step="all", label="All"),
                    ]),
                rangeslider=dict(visible=True, thickness=0.06),
            ),
            yaxis=dict(**AXIS_STYLE, tickprefix=currency, tickformat=","),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("📊 Full KPI Table"):
            st.dataframe(pd.DataFrame({
                "KPI":        ["Total Return","CAGR","Sharpe","Calmar","Max DD","Vol","Win Rate","Trades"],
                "Strategy":   [f"{r['total_return']*100:.2f}%", f"{r['cagr']*100:.2f}%",
                               f"{r['sharpe']:.3f}", f"{r['calmar']:.3f}",
                               f"{r['max_dd']*100:.2f}%", f"{r['volatility']*100:.2f}%",
                               f"{r['win_rate']*100:.2f}%", str(r["n_trades"])],
                "Buy & Hold": [f"{bh['total']*100:.2f}%", f"{bh['cagr']*100:.2f}%",
                               f"{bh['sharpe']:.3f}", f"{bh['calmar']:.3f}",
                               f"{bh['max_dd']*100:.2f}%", f"{bh['vol']*100:.2f}%","–","–"],
            }), hide_index=True, use_container_width=True)

    # ── BUY & HOLD ────────────────────────────────────────────
    elif page == "⚔️ vs. Buy & Hold":

        strat_dd = (df["Strategy_Cum"] - df["Strategy_Cum"].cummax()) / df["Strategy_Cum"].cummax()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=df.index, y=df["Market_Cum"],
            name="Buy & Hold", line=dict(color="#8b949e", width=2, dash="dot"),
            hovertemplate="<b>B&H</b> %{x|%d.%m.%Y}<br>%{y:.2f}x<extra></extra>"))
        fig_eq.add_trace(go.Scatter(x=df.index, y=df["Strategy_Cum"],
            name=f"SMA {sma_fast}/{sma_slow}", line=dict(color="#58a6ff", width=2),
            hovertemplate="<b>Strategy</b> %{x|%d.%m.%Y}<br>%{y:.2f}x<extra></extra>"))
        mobile_layout(fig_eq, height=320, title="Equity Curves")
        fig_eq.update_yaxes(tickformat=".1f")
        st.plotly_chart(fig_eq, use_container_width=True, config={"displayModeBar": False})

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df.index, y=bh["dd"]*100,
            name="Buy & Hold", line=dict(color="#8b949e", width=1.5),
            fill="tozeroy", fillcolor="rgba(139,148,158,0.1)"))
        fig_dd.add_trace(go.Scatter(x=df.index, y=strat_dd*100,
            name="Strategy", line=dict(color="#f85149", width=1.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.1)"))
        mobile_layout(fig_dd, height=240, title="Drawdown")
        fig_dd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

        rf_d    = 0.02 / TRADING_DAYS
        roll_s  = df["Strategy_Return"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean()-rf_d)/x.std()*np.sqrt(TRADING_DAYS) if x.std()>0 else 0)
        roll_bh = df["Market_Return"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean()-rf_d)/x.std()*np.sqrt(TRADING_DAYS) if x.std()>0 else 0)

        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=df.index, y=roll_bh,
            name="Buy & Hold", line=dict(color="#8b949e", width=1.5, dash="dot")))
        fig_rs.add_trace(go.Scatter(x=df.index, y=roll_s,
            name="Strategy", line=dict(color="#58a6ff", width=1.5)))
        fig_rs.add_hline(y=1, line_dash="dash", line_color="#238636", opacity=0.5,
                         annotation_text="1.0", annotation_font_color="#3fb950",
                         annotation_font_size=10)
        mobile_layout(fig_rs, height=240, title="Rolling Sharpe (252d)")
        st.plotly_chart(fig_rs, use_container_width=True, config={"displayModeBar": False})

    # ── BACKTESTING ───────────────────────────────────────────
    elif page == "🔬 Backtesting":

        if not run_multi:
            st.info("Enable **'Test all combinations'** in the sidebar.")
            c1, c2 = st.columns(2)
            c1.metric("Total Return", f"{r['total_return']*100:.1f}%")
            c2.metric("CAGR",         f"{r['cagr']*100:.1f}%")
            c3, c4 = st.columns(2)
            c3.metric("Sharpe",       f"{r['sharpe']:.2f}")
            c4.metric("Max DD",       f"{r['max_dd']*100:.1f}%")
        else:
            combos  = [(10,50),(10,100),(10,200),(20,50),(20,100),(20,200),(50,100),(50,200)]
            results = []
            prog    = st.progress(0, text="Running backtests...")
            for i, (f, s) in enumerate(combos):
                res = backtest(df_raw, f, s)
                results.append({
                    "Combo":   f"SMA {f}/{s}",
                    "Return":  round(res["total_return"]*100, 1),
                    "CAGR":    round(res["cagr"]*100, 1),
                    "Sharpe":  round(res["sharpe"], 3),
                    "Calmar":  round(res["calmar"], 3),
                    "Max DD":  round(res["max_dd"]*100, 1),
                    "Win %":   round(res["win_rate"]*100, 1),
                    "Trades":  res["n_trades"],
                    "_eq":     res["df"]["Strategy_Cum"],
                    "_df":     res["df"],
                })
                prog.progress((i+1)/len(combos), text=f"SMA {f}/{s} done...")
            prog.empty()

            df_res = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
            best   = df_res.iloc[0]
            st.success(f"🏆 **{best['Combo']}** · Sharpe {best['Sharpe']} · "
                       f"Return {best['Return']}% · Max DD {best['Max DD']}%")

            st.dataframe(df_res.drop(columns=["_eq","_df"]).set_index("Combo"),
                         use_container_width=True)

            fig_b = go.Figure(go.Scatter(
                x=df_res["Sharpe"], y=df_res["Return"],
                mode="markers+text", text=df_res["Combo"],
                textposition="top center", textfont=dict(size=9),
                marker=dict(size=-df_res["Max DD"], color=df_res["Sharpe"],
                            colorscale="RdYlGn", showscale=True,
                            colorbar=dict(title="Sharpe", thickness=10),
                            line=dict(width=1, color="#30363d")),
                hovertemplate="<b>%{text}</b><br>Sharpe: %{x:.3f}<br>Return: %{y:.1f}%<extra></extra>"
            ))
            mobile_layout(fig_b, height=320, title="Sharpe vs Return")
            fig_b.update_xaxes(title="Sharpe")
            fig_b.update_yaxes(title="Return (%)")
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

            fig_all = go.Figure()
            fig_all.add_trace(go.Scatter(
                x=results[0]["_df"].index, y=results[0]["_df"]["Market_Cum"],
                name="B&H", line=dict(color="#8b949e", width=2, dash="dot")))
            colors = px.colors.qualitative.Set2
            for i, res in enumerate(results):
                fig_all.add_trace(go.Scatter(
                    x=res["_eq"].index, y=res["_eq"],
                    name=res["Combo"], line=dict(color=colors[i % len(colors)], width=1.5)))
            mobile_layout(fig_all, height=320, title="All Equity Curves vs B&H")
            fig_all.update_yaxes(tickformat=".1f")
            st.plotly_chart(fig_all, use_container_width=True, config={"displayModeBar": False})
