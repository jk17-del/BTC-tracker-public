# ============================================================
# BTC MA Strategy – Streamlit App (PWA-ready)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

# ── Seitenkonfiguration ──────────────────────────────────────
st.set_page_config(
    page_title="BTC MA Strategy",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Hintergrund & Schrift */
    [data-testid="stAppViewContainer"] { background-color: #0d1117; }
    [data-testid="stSidebar"]          { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, p, label              { color: #e6edf3 !important; }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"]  { color: #58a6ff !important; font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"]  { color: #8b949e !important; }
    [data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }

    /* Sidebar Slider */
    .stSlider > div > div > div { background: #58a6ff !important; }

    /* Buttons */
    .stButton > button {
        background: #238636; color: #fff; border: none;
        border-radius: 6px; padding: 8px 20px; font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #2ea043; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Plotly charts transparent bg */
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Konstanten ───────────────────────────────────────────────
TRADING_DAYS = 252

# ── Backtesting-Funktion ─────────────────────────────────────
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
        "df":             df,
        "total_return":   total_return,
        "cagr":           cagr,
        "sharpe":         sharpe,
        "calmar":         calmar,
        "max_dd":         max_dd,
        "volatility":     volatility,
        "win_rate":       win_rate,
        "n_trades":       n_trades,
        "golden_cross":   df[df["Cross"] ==  1],
        "death_cross":    df[df["Cross"] == -1],
    }

def bh_kpis(df):
    rf_daily   = 0.02 / TRADING_DAYS
    n_years    = len(df) / TRADING_DAYS
    total      = df["Market_Cum"].iloc[-1] - 1
    cagr       = df["Market_Cum"].iloc[-1] ** (1 / n_years) - 1
    excess     = df["Market_Return"] - rf_daily
    sharpe     = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)
    roll_max   = df["Market_Cum"].cummax()
    dd         = (df["Market_Cum"] - roll_max) / roll_max
    max_dd     = dd.min()
    vol        = df["Market_Return"].std() * np.sqrt(TRADING_DAYS)
    calmar     = cagr / abs(max_dd)
    return {"total": total, "cagr": cagr, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "vol": vol, "dd": dd}

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
)

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ₿ BTC MA Strategy")
    st.markdown("---")

    ticker = st.selectbox("Asset", ["BTC-EUR", "BTC-USD", "ETH-EUR", "ETH-USD"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date(2020, 1, 1))
    with col2:
        end_date = st.date_input("Ende", value=date.today())

    st.markdown("### Moving Averages")
    sma_fast = st.slider("SMA Fast", min_value=5,   max_value=100, value=10,  step=5)
    sma_slow = st.slider("SMA Slow", min_value=20,  max_value=400, value=100, step=10)

    if sma_fast >= sma_slow:
        st.error("SMA Fast muss kleiner als SMA Slow sein!")
        st.stop()

    st.markdown("### Backtesting-Vergleich")
    run_multi = st.checkbox("Mehrere Kombinationen testen", value=False)

    if run_multi:
        st.caption("Getestete Kombinationen:")
        st.caption("10/50 · 10/100 · 10/200\n20/50 · 20/100 · 20/200\n50/100 · 50/200")

    run = st.button("🚀 Analyse starten")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
st.markdown(f"# {ticker.split('-')[0]} Moving Average Strategy")
currency = "€" if "EUR" in ticker else "$"

if not run:
    st.info("👈 Parameter in der Sidebar einstellen und **Analyse starten** klicken.")
    st.stop()

# ── Daten laden ──────────────────────────────────────────────
with st.spinner("Daten werden geladen..."):
    df_raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if df_raw.empty:
        st.error("Keine Daten gefunden. Bitte Zeitraum oder Asset anpassen.")
        st.stop()
    df_raw = df_raw[["Close"]].dropna()
    df_raw.columns = ["Close"]

# ── Hauptstrategie berechnen ─────────────────────────────────
r   = backtest(df_raw, sma_fast, sma_slow)
df  = r["df"]
bh  = bh_kpis(df)

# ════════════════════════════════════════════════════════════
# TAB-LAYOUT
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Chart & KPIs", "⚔️ vs. Buy & Hold", "🔬 Backtesting"])

# ────────────────────────────────────────────────────────────
# TAB 1 – Chart & KPIs
# ────────────────────────────────────────────────────────────
with tab1:

    # KPI-Zeile
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return",   f"{r['total_return']*100:.1f}%",
              delta=f"{(r['total_return']-bh['total'])*100:+.1f}% vs B&H")
    c2.metric("CAGR",           f"{r['cagr']*100:.1f}%")
    c3.metric("Sharpe Ratio",   f"{r['sharpe']:.2f}",
              delta=f"{r['sharpe']-bh['sharpe']:+.2f} vs B&H")
    c4.metric("Max Drawdown",   f"{r['max_dd']*100:.1f}%",
              delta=f"{(r['max_dd']-bh['max_dd'])*100:+.1f}% vs B&H", delta_color="inverse")
    c5.metric("Anzahl Trades",  str(r["n_trades"]))

    st.markdown("---")

    # Preis-Chart mit MAs
    fig = go.Figure()

    # Trend-Hintergründe
    prev_idx    = df.index[0]
    prev_signal = df["Signal"].iloc[0]
    shapes = []
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

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"].squeeze(),
        name="Preis", line=dict(color="#8b949e", width=1),
        hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Preis: " + currency + "%{y:,.0f}<extra></extra>"
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
        xaxis=dict(**PLOT_THEME["xaxis"],
            rangeslider=dict(visible=True),
            rangeselector=dict(
                bgcolor="#161b22", activecolor="#238636",
                buttons=[
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1J", step="year",  stepmode="backward"),
                    dict(step="all", label="Alle"),
                ]
            )
        ),
        yaxis=dict(**PLOT_THEME["yaxis"],
            title=f"Preis ({currency})", tickprefix=currency, tickformat=","
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weitere KPIs
    with st.expander("📊 Alle KPIs anzeigen"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Strategie**")
            st.dataframe(pd.DataFrame({
                "KPI": ["Total Return", "CAGR", "Sharpe Ratio", "Calmar Ratio",
                        "Max Drawdown", "Volatilität", "Win Rate", "Trades"],
                "Wert": [
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
                        "Max Drawdown", "Volatilität", "–", "–"],
                "Wert": [
                    f"{bh['total']*100:.2f}%",
                    f"{bh['cagr']*100:.2f}%",
                    f"{bh['sharpe']:.3f}",
                    f"{bh['calmar']:.3f}",
                    f"{bh['max_dd']*100:.2f}%",
                    f"{bh['vol']*100:.2f}%",
                    "–", "–",
                ]
            }), hide_index=True, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 2 – vs. Buy & Hold
# ────────────────────────────────────────────────────────────
with tab2:

    # Equity-Kurven
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
        hovertemplate=f"<b>Strategie</b><br>%{{x|%d.%m.%Y}}<br>%{{y:.2f}}x<extra></extra>"
    ))
    fig_eq.update_layout(
        **PLOT_THEME,
        title="Equity-Kurven",
        yaxis=dict(**PLOT_THEME["yaxis"], title="Wachstum (1x = Start)", tickformat=".1f"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        height=400,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown
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
        hovertemplate="<b>Strategie DD</b><br>%{x|%d.%m.%Y}<br>%{y:.1f}%<extra></extra>"
    ))
    fig_dd.update_layout(
        **PLOT_THEME,
        title="Drawdown-Vergleich",
        yaxis=dict(**PLOT_THEME["yaxis"], title="Drawdown (%)", ticksuffix="%"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        height=300,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Rollierender Sharpe
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
        hovertemplate="<b>Strategie Sharpe</b><br>%{x|%d.%m.%Y}<br>%{y:.2f}<extra></extra>"
    ))
    fig_rs.add_hline(y=0, line_dash="dash", line_color="#30363d")
    fig_rs.add_hline(y=1, line_dash="dash", line_color="#238636", opacity=0.5,
                     annotation_text="Sharpe = 1", annotation_font_color="#3fb950")
    fig_rs.update_layout(
        **PLOT_THEME,
        title="Rollierender Sharpe Ratio (252 Tage)",
        yaxis=dict(**PLOT_THEME["yaxis"], title="Sharpe Ratio"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        height=300,
    )
    st.plotly_chart(fig_rs, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 3 – Multi-Backtesting
# ────────────────────────────────────────────────────────────
with tab3:
    if not run_multi:
        st.info("In der Sidebar **'Mehrere Kombinationen testen'** aktivieren.")
    else:
        combos = [(10,50),(10,100),(10,200),(20,50),(20,100),(20,200),(50,100),(50,200)]
        results = []
        prog = st.progress(0, text="Backtesting läuft...")
        for i, (f, s) in enumerate(combos):
            res = backtest(df_raw, f, s)
            results.append({
                "Kombination": f"SMA {f}/{s}",
                "Total Return (%)": round(res["total_return"] * 100, 2),
                "CAGR (%)":         round(res["cagr"] * 100, 2),
                "Sharpe Ratio":     round(res["sharpe"], 3),
                "Calmar Ratio":     round(res["calmar"], 3),
                "Max Drawdown (%)": round(res["max_dd"] * 100, 2),
                "Win Rate (%)":     round(res["win_rate"] * 100, 2),
                "Trades":           res["n_trades"],
                "_eq":              res["df"]["Strategy_Cum"],
            })
            prog.progress((i + 1) / len(combos), text=f"SMA {f}/{s} fertig...")
        prog.empty()

        df_res = pd.DataFrame(results).sort_values("Sharpe Ratio", ascending=False)

        # Beste Strategie hervorheben
        best_label = df_res.iloc[0]["Kombination"]
        st.success(f"🏆 Beste Strategie: **{best_label}** | "
                   f"Sharpe: {df_res.iloc[0]['Sharpe Ratio']} | "
                   f"Return: {df_res.iloc[0]['Total Return (%)']}% | "
                   f"MaxDD: {df_res.iloc[0]['Max Drawdown (%)']}%")

        # Tabelle
        st.dataframe(
            df_res.drop(columns=["_eq"]).set_index("Kombination"),
            use_container_width=True
        )

        # Bubble Chart
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=df_res["Sharpe Ratio"],
            y=df_res["Total Return (%)"],
            mode="markers+text",
            text=df_res["Kombination"],
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
            title="Sharpe vs. Return (Blasengröße = Max Drawdown)",
            xaxis=dict(**PLOT_THEME["xaxis"], title="Sharpe Ratio"),
            yaxis=dict(**PLOT_THEME["yaxis"], title="Total Return (%)"),
            height=450,
        )
        st.plotly_chart(fig_b, use_container_width=True)

        # Equity-Kurven aller Strategien
        fig_all = go.Figure()
        bh_eq = results[0]["_eq"]
        r0 = backtest(df_raw, combos[0][0], combos[0][1])
        fig_all.add_trace(go.Scatter(
            x=r0["df"].index, y=r0["df"]["Market_Cum"],
            name="Buy & Hold", line=dict(color="#8b949e", width=2, dash="dot"),
        ))
        colors = px.colors.qualitative.Set2
        for i, res in enumerate(results):
            fig_all.add_trace(go.Scatter(
                x=res["_eq"].index, y=res["_eq"],
                name=res["Kombination"],
                line=dict(color=colors[i % len(colors)], width=1.8),
            ))
        fig_all.update_layout(
            **PLOT_THEME,
            title="Alle Equity-Kurven vs. Buy & Hold",
            yaxis=dict(**PLOT_THEME["yaxis"], title="Wachstum (1x = Start)", tickformat=".1f"),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08, x=0),
            height=500,
        )
        st.plotly_chart(fig_all, use_container_width=True)
