"""
Streamlit market risk dashboard.

Reads live data from Postgres on Neon (populated by notebook 01 + run_risk_pipeline.py /
the GitHub Actions cron). No computation happens here beyond light aggregation --
VaR/CVaR, backtest, and stress test numbers are all pre-computed and stored;
this file just queries and displays them.

Run locally:  streamlit run dashboard/app.py   (from project root)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.config import DatabaseConfig
from src.universe import PORTFOLIO_NAME, PORTFOLIO_DESCRIPTION
from src.risk_engine.backtesting import kupiec_pof_test
from src.risk_engine.stress_testing import stress_vs_var_gap
from src.risk_engine.portfolio import portfolio_returns, ewma_volatility, TRADING_DAYS_PER_YEAR
from src.risk_config import HISTORICAL_SCENARIOS, ROLLING_WINDOW_DAYS, EWMA_LAMBDA, ASSUMED_PORTFOLIO_VALUE

st.set_page_config(page_title="Market Risk Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Connection -- st.secrets first (Streamlit Community Cloud), .env fallback
# (local dev), matching DatabaseConfig.from_env()'s expected variable name.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    secret_keys = ["DATABASE_URL"]

    # st.secrets raises StreamlitSecretNotFoundError outright if no
    # secrets.toml exists anywhere (not just an empty dict) -- which is the
    # normal case for local dev using .env instead. Treat that as "no
    # secrets configured" rather than letting it crash the app.
    try:
        found_secrets = any(k in st.secrets for k in secret_keys)
    except Exception:
        found_secrets = False

    if found_secrets:
        for k in secret_keys:
            if k in st.secrets:
                os.environ[k] = str(st.secrets[k])
    else:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")

    return DatabaseConfig.from_env().create_engine()


engine = get_engine()


@st.cache_data(ttl=3600)
def get_portfolio_id() -> int:
    return int(pd.read_sql(
        "SELECT portfolio_id FROM dim_portfolio WHERE portfolio_name = %(name)s",
        engine, params={"name": PORTFOLIO_NAME},
    )["portfolio_id"].iloc[0])


@st.cache_data(ttl=3600)
def load_composition(portfolio_id: int) -> pd.DataFrame:
    query = """
        SELECT da.ticker, da.company_name, da.sector, da.market_cap_tier, dpw.weight
        FROM dim_asset da
        JOIN dim_portfolio_weight dpw ON da.asset_id = dpw.asset_id
        WHERE dpw.portfolio_id = %(pid)s
        ORDER BY da.ticker;
    """
    return pd.read_sql(query, engine, params={"pid": portfolio_id})


@st.cache_data(ttl=3600)
def load_price_history() -> pd.DataFrame:
    query = """
        SELECT fp.date_id AS date, da.ticker, fp.close
        FROM fact_price fp
        JOIN dim_asset da ON fp.asset_id = da.asset_id
        ORDER BY fp.date_id;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="close")


@st.cache_data(ttl=3600)
def load_returns() -> pd.DataFrame:
    query = """
        SELECT fr.date_id AS date, da.ticker, fr.simple_return
        FROM fact_returns fr
        JOIN dim_asset da ON fr.asset_id = da.asset_id
        ORDER BY fr.date_id;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="simple_return").dropna(how="any").astype(float)


@st.cache_data(ttl=3600)
def load_risk_metrics(portfolio_id: int) -> pd.DataFrame:
    query = """
        SELECT date_id, method, confidence, var_pct, cvar_pct, window_days, computed_at
        FROM fact_risk_metrics
        WHERE portfolio_id = %(pid)s
        ORDER BY date_id DESC;
    """
    return pd.read_sql(query, engine, params={"pid": portfolio_id})


@st.cache_data(ttl=3600)
def load_backtest(portfolio_id: int) -> pd.DataFrame:
    query = """
        SELECT date_id, method, confidence, predicted_var_pct, realized_return_pct, breach
        FROM fact_var_backtest
        WHERE portfolio_id = %(pid)s
        ORDER BY date_id;
    """
    df = pd.read_sql(query, engine, params={"pid": portfolio_id})
    df["date_id"] = pd.to_datetime(df["date_id"])
    return df


@st.cache_data(ttl=3600)
def load_stress_tests(portfolio_id: int) -> pd.DataFrame:
    query = """
        SELECT scenario_name, scenario_type, start_date, end_date,
               portfolio_pnl_pct, portfolio_pnl_value, computed_at
        FROM fact_stress_test
        WHERE portfolio_id = %(pid)s
        ORDER BY computed_at DESC;
    """
    return pd.read_sql(query, engine, params={"pid": portfolio_id})


# ---------------------------------------------------------------------------
# Load everything up front
# ---------------------------------------------------------------------------
portfolio_id = get_portfolio_id()
composition = load_composition(portfolio_id)
price_history = load_price_history()
risk_metrics = load_risk_metrics(portfolio_id)
backtest = load_backtest(portfolio_id)
stress_tests = load_stress_tests(portfolio_id)
returns_df = load_returns()
weights = composition.set_index("ticker")["weight"].astype(float)
port_ret = portfolio_returns(returns_df, weights)

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
PORTFOLIO_COLOR = "#1F3A5F"      # navy -- the portfolio / headline series
VAR_COLOR = "#D9822B"            # amber -- VaR thresholds
BREACH_COLOR = "#C0392B"         # brick red -- breaches / losses
RETURN_COLOR = "#8FA3BF"         # slate -- realized returns (background series)
CONTEXT_GREY = "#C8CDD5"         # un-highlighted lines
BAND_COLOR = "#E8ECF2"           # stress-scenario shading
# muted categorical palette (Tableau 10) for highlighted tickers / methods
COLORWAY = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2",
            "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
# 14 distinct colours for highlighted tickers (Tableau 20 picks, no navy -- that's the portfolio)
HIGHLIGHT_PALETTE = ["#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1", "#FF9DA7",
                     "#9C755F", "#4E79A7", "#8CD17D", "#B6992D", "#D37295", "#499894", "#79706E"]
METHOD_LABELS = {
    "historical": "Historical", "parametric": "Parametric",
    "parametric_ewma": "Parametric (EWMA)", "monte_carlo": "Monte Carlo",
}


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def add_scenario_bands(fig, x_min=None, x_max=None):
    """Shade each historical stress scenario (from risk_config) on a time-series chart."""
    for name, (start, end) in HISTORICAL_SCENARIOS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if (x_min is not None and e < x_min) or (x_max is not None and s > x_max):
            continue
        fig.add_vrect(x0=s, x1=e, fillcolor=BAND_COLOR, opacity=0.8, line_width=0, layer="below",
                      annotation_text=name.split(" (")[0], annotation_position="top left",
                      annotation_font=dict(size=11, color="#5A6577"))


def log_tickvals(lo: float, hi: float) -> list[float]:
    """Readable 1-2-5 ticks for a log axis (Plotly's defaults print bare '2' and '5')."""
    ticks = []
    for exp in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** exp
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def base_layout(fig, height=480, y_fmt=None):
    fig.update_layout(
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        template="plotly_white", colorway=COLORWAY,
        font=dict(color="#2B3442"), title_font=dict(size=15),
    )
    if y_fmt:
        fig.update_yaxes(tickformat=y_fmt)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor="#9AA5B4",
                     gridcolor="#EEF1F5")
    fig.update_yaxes(gridcolor="#EEF1F5", zerolinecolor="#D5DAE1")
    return fig


ROLLING_LABEL = f"{ROLLING_WINDOW_DAYS}-day rolling"
EWMA_LABEL = f"EWMA (λ = {EWMA_LAMBDA})"


@st.cache_data(ttl=3600)
def volatility_table(_returns_df: pd.DataFrame, _port_ret: pd.Series, key: str) -> dict:
    """
    Annualized 250-day rolling vs EWMA volatility for the portfolio and every
    ticker, keyed by name. `key` (latest date) busts the cache on new data.
    EWMA is a one-day-ahead forecast (no look-ahead) -- see portfolio.ewma_volatility.
    """
    ann = np.sqrt(TRADING_DAYS_PER_YEAR)
    series = {"Portfolio": _port_ret, **{t: _returns_df[t] for t in _returns_df.columns}}
    out = {}
    for name, r in series.items():
        out[name] = pd.DataFrame({
            ROLLING_LABEL: r.rolling(ROLLING_WINDOW_DAYS).std() * ann,
            EWMA_LABEL: ewma_volatility(r, EWMA_LAMBDA) * ann,
        }).iloc[ROLLING_WINDOW_DAYS:]
    return out

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Market Risk Dashboard")
st.caption(PORTFOLIO_DESCRIPTION)

last_price_date = price_history.index.max()

c1, c2 = st.columns(2)
c1.metric("Portfolio", PORTFOLIO_NAME)
c2.metric("Latest price data", last_price_date.strftime("%Y-%m-%d") if pd.notna(last_price_date) else "N/A")

tab_overview, tab_var, tab_vol, tab_backtest, tab_stress = st.tabs(
    ["Overview", "VaR / CVaR", "Volatility (EWMA)", "Backtesting", "Stress Test"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Composition")
        st.dataframe(
            composition.rename(columns={
                "ticker": "Ticker", "company_name": "Company", "sector": "Sector",
                "market_cap_tier": "Cap", "weight": "Weight",
            }).style.format({"Weight": "{:.2%}"}),
            hide_index=True, width="stretch",
        )

    with col2:
        st.subheader("Sector Allocation")
        sector_weights = composition.groupby("sector")["weight"].sum().reset_index()
        fig = px.pie(sector_weights, names="sector", values="weight", hole=0.5,
                     color_discrete_sequence=COLORWAY, template="plotly_white")
        fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color="white", width=2)))
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Indexed Price Performance")
    min_d, max_d = price_history.index.min().date(), price_history.index.max().date()
    tickers = list(price_history.columns)

    f1, f2 = st.columns([1, 1])
    start_d, end_d = f1.slider("Period", min_value=min_d, max_value=max_d,
                               value=(min_d, max_d), format="YYYY-MM")
    h1, h2 = f2.columns([5, 1])
    highlight_all = h2.checkbox("All", value=False, help="Highlight every ticker at once")
    highlight = h1.multiselect("Highlight tickers (others stay grey)", tickers, default=[],
                               disabled=highlight_all)
    if highlight_all:
        highlight = tickers
    g1, g2, g3, g4 = st.columns(4)
    freq = g1.radio("Frequency", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True,
                    help="Weekly/monthly closes remove day-to-day noise from long-horizon views.")
    show_others = g2.toggle("Show other tickers", value=True)
    show_port = g3.toggle("Portfolio line", value=True)
    log_scale = g4.toggle("Log scale", value=True)

    rule = {"Daily": None, "Weekly": "W-FRI", "Monthly": "ME"}[freq]

    def resample_last(obj):
        return obj if rule is None else obj.resample(rule).last().dropna(how="all")

    window = price_history.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
    # rebase to 100 at the start of the SELECTED period, so the slider changes the comparison
    indexed = resample_last(window / window.bfill().iloc[0] * 100)

    fig = go.Figure()
    if show_others:
        for t in [t for t in tickers if t not in highlight]:
            fig.add_trace(go.Scatter(
                x=indexed.index, y=indexed[t], name=t, mode="lines", showlegend=False,
                line=dict(width=0.8, color=CONTEXT_GREY),
                hovertemplate=f"{t}: %{{y:,.0f}}<extra></extra>",
            ))
    for i, t in enumerate(highlight):
        fig.add_trace(go.Scatter(
            x=indexed.index, y=indexed[t], name=t, mode="lines",
            line=dict(width=1.4, color=HIGHLIGHT_PALETTE[i % len(HIGHLIGHT_PALETTE)]),
            hovertemplate=f"<b>{t}</b>: %{{y:,.0f}}<extra></extra>",
        ))
    if show_port:
        pr = port_ret.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
        port_idx = resample_last((1 + pr).cumprod() / (1 + pr.iloc[0]) * 100)
        fig.add_trace(go.Scatter(
            x=port_idx.index, y=port_idx, name="Portfolio (equal-weight)", mode="lines",
            line=dict(width=2.2, color=PORTFOLIO_COLOR),
            hovertemplate="<b>Portfolio</b>: %{y:,.0f}<extra></extra>",
        ))

    if not fig.data:
        st.info("Nothing to plot -- highlight a ticker or switch on the portfolio / other tickers.")
    else:
        add_scenario_bands(fig, pd.Timestamp(start_d), pd.Timestamp(end_d))
        base_layout(fig, height=540)
        fig.update_layout(title=f"Start of period = 100 · {freq.lower()} closes{' · log scale' if log_scale else ''}")
        if log_scale:
            ys = [tr.y for tr in fig.data]
            lo = min(np.nanmin(y) for y in ys)
            hi = max(np.nanmax(y) for y in ys)
            fig.update_yaxes(type="log", tickvals=log_tickvals(lo, hi), tickformat=",")
        else:
            fig.update_yaxes(tickformat=",")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------
with tab_var:
    if risk_metrics.empty:
        st.info("No risk metrics found yet -- run the risk pipeline to populate fact_risk_metrics.")
    else:
        latest_date = risk_metrics["date_id"].max()
        latest = risk_metrics[risk_metrics["date_id"] == latest_date].sort_values(["confidence", "method"]).copy()
        latest["Method"] = latest["method"].map(method_label)
        latest["confidence_label"] = latest["confidence"].map(lambda c: f"{c:.0%}")

        st.subheader(f"Current VaR / CVaR (as of {latest_date})")
        st.caption(
            "**Historical**: empirical distribution of the last 250 days. "
            "**Parametric**: closed-form normal (Delta-Normal), equally weighted covariance. "
            "**Parametric EWMA**: same formula, RiskMetrics EWMA covariance (λ = 0.94) -- "
            "recent days dominate, so it reacts fastest to volatility spikes. "
            "**Monte Carlo**: 10,000 correlated scenarios simulated from the window's "
            "mean and covariance (Cholesky)."
        )

        c1, c2 = st.columns([2, 1])
        metric = c2.radio("Show", ["VaR", "CVaR"], horizontal=True)
        in_dollars = c2.toggle(f"In $ (on ${ASSUMED_PORTFOLIO_VALUE/1e6:.0f}M notional)", value=False)
        col = "var_pct" if metric == "VaR" else "cvar_pct"
        plot_df = latest.assign(value=latest[col] * (ASSUMED_PORTFOLIO_VALUE if in_dollars else 1))

        fig = px.bar(
            plot_df, x="Method", y="value", color="confidence_label", barmode="group",
            labels={"value": metric, "confidence_label": "Confidence"},
            text=plot_df["value"].map(lambda v: f"${v:,.0f}" if in_dollars else f"{v:.2%}"),
            title=f"One-day {metric} by method",
            color_discrete_sequence=[PORTFOLIO_COLOR, VAR_COLOR], template="plotly_white",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat="$,.0f" if in_dollars else ".1%")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        c1.plotly_chart(fig, width="stretch")

        c2.dataframe(
            latest[["Method", "confidence", "var_pct", "cvar_pct"]]
            .rename(columns={"confidence": "Conf.", "var_pct": "VaR", "cvar_pct": "CVaR"})
            .style.format({"Conf.": "{:.0%}", "VaR": "{:.2%}", "CVaR": "{:.2%}"}),
            hide_index=True, width="stretch",
        )

# ---------------------------------------------------------------------------
# Volatility (EWMA)
# ---------------------------------------------------------------------------
with tab_vol:
    vols = volatility_table(returns_df, port_ret, str(returns_df.index.max()))
    eff_memory = 1 / (1 - EWMA_LAMBDA)

    st.subheader("EWMA volatility")
    st.caption(
        f"RiskMetrics EWMA: σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·r²ₜ₋₁ with λ = {EWMA_LAMBDA} -- yesterday's return gets "
        f"{1 - EWMA_LAMBDA:.0%} weight and older days decay geometrically (effective memory ≈ "
        f"{eff_memory:.0f} trading days, vs. {ROLLING_WINDOW_DAYS} equally weighted days for the rolling "
        "window). All figures annualized (× √252). This is the σ behind the Parametric (EWMA) VaR."
    )

    sel = st.selectbox("Series", list(vols), index=0,
                       help="The equal-weight portfolio, or any single stock")
    v = vols[sel]
    now_ewma, now_roll = v[EWMA_LABEL].iloc[-1], v[ROLLING_LABEL].iloc[-1]
    ratio = now_ewma / now_roll
    regime = "Elevated" if ratio > 1.2 else "Calm" if ratio < 0.8 else "Normal"
    pct_rank = (v[EWMA_LABEL] <= now_ewma).mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("EWMA volatility (today)", f"{now_ewma:.1%}")
    m2.metric(f"{ROLLING_WINDOW_DAYS}-day volatility", f"{now_roll:.1%}")
    m3.metric("EWMA ÷ rolling", f"{ratio:.2f}×", regime,
              delta_color="inverse" if regime == "Elevated" else "off")
    m4.metric("Historical percentile", f"{pct_rank:.0%}",
              "of all days since " + f"{v.index.min():%Y}", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v.index, y=v[ROLLING_LABEL], name=ROLLING_LABEL, mode="lines",
                             line=dict(color="#9AA5B4", width=1.8),
                             hovertemplate="Rolling: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=v.index, y=v[EWMA_LABEL], name=EWMA_LABEL, mode="lines",
                             line=dict(color=PORTFOLIO_COLOR, width=1.0),
                             hovertemplate="EWMA: %{y:.1%}<extra></extra>"))
    add_scenario_bands(fig)
    base_layout(fig, height=440, y_fmt=".0%")
    fig.update_layout(title=f"{sel}: annualized volatility, EWMA vs. rolling window")
    fig.update_xaxes(rangeselector=dict(bgcolor="#EEF1F5", activecolor="#C8D3E3", x=1, xanchor="right", y=1.02, buttons=[
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=5, label="5Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ]))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Read it: EWMA spikes within days of a shock and decays back; the rolling window climbs "
        "slowly and then stays high for a year. That lag is why equal-weighted VaR under-predicts "
        "going into a crisis -- and why the EWMA model passes the Kupiec backtest."
    )

    st.subheader("Today's volatility by stock")
    snap = pd.DataFrame({
        "Ticker": [t for t in vols if t != "Portfolio"],
        EWMA_LABEL: [vols[t][EWMA_LABEL].iloc[-1] for t in vols if t != "Portfolio"],
        ROLLING_LABEL: [vols[t][ROLLING_LABEL].iloc[-1] for t in vols if t != "Portfolio"],
    }).sort_values(EWMA_LABEL, ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=snap["Ticker"], x=snap[ROLLING_LABEL], name=ROLLING_LABEL, orientation="h",
                         marker_color="#C8CDD5", hovertemplate="Rolling: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(y=snap["Ticker"], x=snap[EWMA_LABEL], name=EWMA_LABEL, orientation="h",
                         marker_color=PORTFOLIO_COLOR, hovertemplate="EWMA: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=now_ewma if sel == "Portfolio" else vols["Portfolio"][EWMA_LABEL].iloc[-1],
                  line=dict(color=VAR_COLOR, dash="dash", width=1.5),
                  annotation_text="Portfolio EWMA", annotation_position="top")
    base_layout(fig, height=520, y_fmt=None)
    fig.update_layout(barmode="group", hovermode="y unified",
                      title="Annualized volatility, sorted by EWMA (the dashed line shows diversification)")
    fig.update_xaxes(tickformat=".0%", showspikes=False)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
with tab_backtest:
    if backtest.empty:
        st.info("No backtest results found yet -- run the backtest and push to fact_var_backtest.")
    else:
        methods = sorted(backtest["method"].unique())
        confidences = sorted(backtest["confidence"].unique())

        # focus periods: full history, each stress scenario (padded), last 12 months
        bt_min, bt_max = backtest["date_id"].min(), backtest["date_id"].max()
        focus_options = {"Full history": (bt_min, bt_max)}
        for name, (s, e) in HISTORICAL_SCENARIOS.items():
            focus_options[name] = (pd.Timestamp(s) - pd.DateOffset(months=3), pd.Timestamp(e) + pd.DateOffset(months=6))
        focus_options["Last 12 months"] = (bt_max - pd.DateOffset(years=1), bt_max)

        col1, col2, col3 = st.columns(3)
        sel_method = col1.selectbox("Method", methods, index=0, format_func=method_label)
        sel_confidence = col2.selectbox("Confidence", confidences, index=0,
                                        format_func=lambda c: f"{c:.0%}")
        sel_focus = col3.selectbox("Zoom to", list(focus_options))
        f_start, f_end = focus_options[sel_focus]

        subset = backtest[(backtest["method"] == sel_method) & (backtest["confidence"] == sel_confidence)]
        view = subset[(subset["date_id"] >= f_start) & (subset["date_id"] <= f_end)]
        breaches = view[view["breach"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view["date_id"], y=view["realized_return_pct"], mode="lines",
                                 name="Realized return", line=dict(width=0.6, color=RETURN_COLOR),
                                 hovertemplate="Return: %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=view["date_id"], y=-view["predicted_var_pct"], mode="lines",
                                 name=f"−{sel_confidence:.0%} VaR", line=dict(color=VAR_COLOR, width=1.4, shape="hv"),
                                 hovertemplate="VaR: %{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=breaches["date_id"], y=breaches["realized_return_pct"], mode="markers",
                                 name=f"Breaches in view (n={len(breaches)})",
                                 marker=dict(color=BREACH_COLOR, size=4.5, line=dict(width=0.5, color="white")),
                                 hovertemplate="Breach: %{y:.2%}<extra></extra>"))
        add_scenario_bands(fig, f_start, f_end)
        base_layout(fig, height=500, y_fmt=".1%")
        fig.update_layout(title=f"Realized returns vs. {method_label(sel_method)} VaR -- {sel_focus}")
        fig.update_xaxes(range=[f_start, f_end])
        st.plotly_chart(fig, width="stretch")

        kupiec = kupiec_pof_test(subset["breach"], confidence=sel_confidence)
        st.caption("Kupiec POF test on the full backtest history (not just the zoomed view):")
        m1, m2, m3 = st.columns(3)
        m1.metric("Observations", f"{kupiec['n_observations']:,}")
        m2.metric("Breaches", kupiec["n_breaches"],
                  f"{kupiec['observed_breach_rate']:.2%} vs {kupiec['expected_breach_rate']:.2%} expected",
                  delta_color="off")
        m3.metric("Kupiec p-value", f"{kupiec['p_value']:.4f}",
                  "Rejected" if kupiec["reject_h0_at_5pct"] else "Not rejected",
                  delta_color="inverse" if kupiec["reject_h0_at_5pct"] else "normal")
        st.write(kupiec["verdict"])

        # ------------------------------------------------------------------
        st.subheader("Compare methods")
        rows = []
        for m in methods:
            k = kupiec_pof_test(backtest[(backtest["method"] == m) & (backtest["confidence"] == sel_confidence)]["breach"],
                                confidence=sel_confidence)
            rows.append({"Method": method_label(m), "Breaches": k["n_breaches"],
                         "Breach rate": k["observed_breach_rate"], "Expected": k["expected_breach_rate"],
                         "p-value": k["p_value"],
                         "Verdict": "❌ Rejected" if k["reject_h0_at_5pct"] else "✅ Not rejected"})
        st.dataframe(
            pd.DataFrame(rows).style.format({"Breach rate": "{:.2%}", "Expected": "{:.2%}", "p-value": "{:.4f}"}),
            hide_index=True, width="stretch",
        )

        compare = st.multiselect("Methods to overlay", methods, default=methods, format_func=method_label)
        fig = go.Figure()
        realized = subset[(subset["date_id"] >= f_start) & (subset["date_id"] <= f_end)]
        fig.add_trace(go.Scatter(x=realized["date_id"], y=realized["realized_return_pct"], mode="lines",
                                 name="Realized return", line=dict(width=0.5, color=CONTEXT_GREY),
                                 hovertemplate="Return: %{y:.2%}<extra></extra>"))
        for m in compare:
            mv = backtest[(backtest["method"] == m) & (backtest["confidence"] == sel_confidence)
                          & (backtest["date_id"] >= f_start) & (backtest["date_id"] <= f_end)]
            fig.add_trace(go.Scatter(x=mv["date_id"], y=-mv["predicted_var_pct"], mode="lines",
                                     name=method_label(m), line=dict(width=1.3, shape="hv"),
                                     hovertemplate=f"{method_label(m)}: %{{y:.2%}}<extra></extra>"))
        add_scenario_bands(fig, f_start, f_end)
        base_layout(fig, height=460, y_fmt=".1%")
        fig.update_layout(title=f"{sel_confidence:.0%} VaR thresholds by method -- {sel_focus}")
        fig.update_xaxes(range=[f_start, f_end])
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Stress Test
# ---------------------------------------------------------------------------
with tab_stress:
    if stress_tests.empty:
        st.info("No stress test results found yet -- run the stress test and push to fact_stress_test.")
    else:
        scen = stress_tests.copy()
        scen["start_date"] = pd.to_datetime(scen["start_date"])
        scen["end_date"] = pd.to_datetime(scen["end_date"])
        scen = scen.sort_values("start_date")

        c1, c2, c3 = st.columns(3)
        sel_scen = c1.selectbox("Scenario", scen["scenario_name"].tolist())
        bt_methods = sorted(backtest["method"].unique()) if not backtest.empty else []
        var_method = c2.selectbox("Compare against VaR method", bt_methods or ["historical"],
                                  format_func=method_label)
        show_stocks = c3.toggle("Show individual stocks", value=False)

        row = scen[scen["scenario_name"] == sel_scen].iloc[0]
        s, e = row["start_date"], row["end_date"]
        win = returns_df.loc[s:e]
        pr = port_ret.loc[s:e]
        cum = (1 + pr).cumprod() - 1
        n_days = len(pr)

        # VaR the model was predicting on the scenario's first day (from the backtest) --
        # "what did the model say going in?", not today's VaR
        pre = backtest[(backtest["method"] == var_method) & (backtest["date_id"] >= s)].sort_values("date_id")
        if not pre.empty:
            var_conf = pre["confidence"].min()
            var_pct = float(pre[pre["confidence"] == var_conf].iloc[0]["predicted_var_pct"])
            var_note = f"{method_label(var_method)} {var_conf:.0%} VaR on {pre.iloc[0]['date_id']:%Y-%m-%d}"
        else:
            hist = risk_metrics[(risk_metrics["method"] == "historical") & (risk_metrics["confidence"] == 0.95)]
            var_conf, var_pct = 0.95, float(hist.iloc[0]["var_pct"])
            var_note = "current Historical 95% VaR"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Window", f"{s:%d %b %Y} → {e:%d %b %Y}", f"{n_days} trading days", delta_color="off")
        m2.metric("Cumulative P&L", f"{row['portfolio_pnl_pct']:.2%}")
        m3.metric("P&L value", f"${row['portfolio_pnl_value']:,.0f}")
        m4.metric("Worst single day", f"{pr.min():.2%}", f"{pr.idxmin():%d %b %Y}", delta_color="off")

        left, right = st.columns([3, 2])

        # path of the portfolio through the crisis vs. the VaR "cone" (VaR × √t)
        fig = go.Figure()
        if show_stocks:
            stock_cum = (1 + win).cumprod() - 1
            for i, t in enumerate(stock_cum.columns):
                fig.add_trace(go.Scatter(x=stock_cum.index, y=stock_cum[t], mode="lines",
                                         name="Individual stocks", legendgroup="stocks", showlegend=i == 0,
                                         line=dict(width=0.8, color=CONTEXT_GREY),
                                         hovertemplate=f"{t}: %{{y:.1%}}<extra></extra>"))
        cone = -var_pct * np.sqrt(np.arange(1, n_days + 1))
        fig.add_trace(go.Scatter(x=pr.index, y=cone, name="VaR × √t (normal-times bound)", mode="lines",
                                 line=dict(color=VAR_COLOR, dash="dash", width=1.6),
                                 hovertemplate="VaR cone: %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Portfolio", mode="lines",
                                 line=dict(color=PORTFOLIO_COLOR, width=2.4),
                                 fill="tozeroy", fillcolor="rgba(192,57,43,0.08)",
                                 hovertemplate="Portfolio: %{y:.1%}<extra></extra>"))
        base_layout(fig, height=480, y_fmt=".0%")
        fig.update_layout(title=f"{sel_scen}: cumulative P&L path")
        left.plotly_chart(fig, width="stretch")

        loss = abs(row["portfolio_pnl_pct"])
        scaled = var_pct * np.sqrt(n_days)
        fig = go.Figure(go.Bar(
            x=["1-day VaR", f"VaR × √{n_days}", "Actual loss"],
            y=[var_pct, scaled, loss],
            marker_color=[VAR_COLOR, "#A8611C", BREACH_COLOR],
            text=[f"{var_pct:.1%}", f"{scaled:.1%}", f"{loss:.1%}"], textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(title="Model vs. reality", yaxis_tickformat=".0%", height=480, template="plotly_white",
                          margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        right.plotly_chart(fig, width="stretch")

        gap = stress_vs_var_gap(row["portfolio_pnl_pct"], var_pct)
        st.markdown(
            f"Going in, the model ({var_note}) put a bad day at **{var_pct:.2%}**. "
            f"The {sel_scen} loss was **{gap['gap_multiple']:.1f}×** that, and still "
            f"**{loss / scaled:.1f}×** the VaR scaled to {n_days} days with the √t rule -- "
            "VaR describes a typical bad day; it says nothing about a sustained crisis."
        )

        st.subheader("All stress scenarios")
        display_cols = ["scenario_name", "scenario_type", "start_date", "end_date",
                        "portfolio_pnl_pct", "portfolio_pnl_value"]
        st.dataframe(
            scen[display_cols].rename(columns={
                "scenario_name": "Scenario", "scenario_type": "Type",
                "start_date": "Start", "end_date": "End",
                "portfolio_pnl_pct": "P&L %", "portfolio_pnl_value": "P&L ($)",
            }).style.format({"P&L %": "{:.2%}", "P&L ($)": "${:,.0f}",
                             "Start": "{:%Y-%m-%d}", "End": "{:%Y-%m-%d}"}),
            hide_index=True, width="stretch",
        )
