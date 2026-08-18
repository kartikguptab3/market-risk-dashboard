"""
Streamlit market risk dashboard.

Reads live data from Supabase (populated by notebook 01 + run_risk_pipeline.py /
the GitHub Actions cron). No computation happens here beyond light aggregation --
VaR/CVaR, backtest, and stress test numbers are all pre-computed and stored;
this file just queries and displays them.

Run locally:  streamlit run dashboard/app.py   (from project root)
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.config import SupabaseConfig
from src.universe import PORTFOLIO_NAME, PORTFOLIO_DESCRIPTION
from src.risk_engine.backtesting import kupiec_pof_test
from src.risk_engine.stress_testing import stress_vs_var_gap

st.set_page_config(page_title="Market Risk Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Connection -- st.secrets first (Streamlit Community Cloud), .env fallback
# (local dev), matching SupabaseConfig.from_env()'s expected variable names.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    secret_keys = [
        "SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_DB_HOST", "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME", "SUPABASE_DB_USER", "SUPABASE_DB_PASSWORD",
    ]

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

    cfg = SupabaseConfig.from_env()
    return create_engine(cfg.sqlalchemy_url())


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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Market Risk Dashboard")
st.caption(PORTFOLIO_DESCRIPTION)

last_price_date = price_history.index.max()

c1, c2 = st.columns(2)
c1.metric("Portfolio", PORTFOLIO_NAME)
c2.metric("Latest price data", last_price_date.strftime("%Y-%m-%d") if pd.notna(last_price_date) else "N/A")

tab_overview, tab_var, tab_backtest, tab_stress = st.tabs(
    ["Overview", "VaR / CVaR", "Backtesting", "Stress Test"]
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
            hide_index=True, use_container_width=True,
        )

    with col2:
        st.subheader("Sector Allocation")
        sector_weights = composition.groupby("sector")["weight"].sum().reset_index()
        fig = px.pie(sector_weights, names="sector", values="weight", hole=0.4)
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Indexed Price Performance (Start = 100, Log Scale)")
    indexed = price_history / price_history.iloc[0] * 100
    fig = px.line(indexed, labels={"value": "Indexed Price", "date": "Date"}, log_y=True)
    fig.update_layout(legend_title_text="Ticker")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------
with tab_var:
    if risk_metrics.empty:
        st.info("No risk metrics found yet -- run the risk pipeline to populate fact_risk_metrics.")
    else:
        latest_date = risk_metrics["date_id"].max()
        latest = risk_metrics[risk_metrics["date_id"] == latest_date].sort_values(["confidence", "method"])

        st.subheader(f"Current VaR / CVaR (as of {latest_date})")
        st.dataframe(
            latest[["method", "confidence", "var_pct", "cvar_pct", "window_days"]]
            .rename(columns={
                "method": "Method", "confidence": "Confidence",
                "var_pct": "VaR", "cvar_pct": "CVaR", "window_days": "Window (days)",
            }).style.format({"Confidence": "{:.0%}", "VaR": "{:.2%}", "CVaR": "{:.2%}"}),
            hide_index=True, use_container_width=True,
        )

        latest = latest.copy()
        latest["confidence_label"] = latest["confidence"].map(lambda c: f"{c:.0%}")

        fig = px.bar(
            latest, x="method", y="var_pct", color="confidence_label", barmode="group",
            labels={"method": "Method", "var_pct": "VaR", "confidence_label": "Confidence"},
            title="VaR by Method and Confidence Level",
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
with tab_backtest:
    if backtest.empty:
        st.info("No backtest results found yet -- run the backtest and push to fact_var_backtest.")
    else:
        methods = sorted(backtest["method"].unique())
        confidences = sorted(backtest["confidence"].unique())
        col1, col2 = st.columns(2)
        sel_method = col1.selectbox("Method", methods, index=0)
        sel_confidence = col2.selectbox("Confidence", confidences, index=0,
                                         format_func=lambda c: f"{c:.0%}")

        subset = backtest[(backtest["method"] == sel_method) & (backtest["confidence"] == sel_confidence)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset["date_id"], y=subset["realized_return_pct"],
                                  mode="lines", name="Realized return", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=subset["date_id"], y=-subset["predicted_var_pct"],
                                  mode="lines", name=f"{sel_confidence:.0%} VaR threshold",
                                  line=dict(color="orange", width=1.5)))
        breaches = subset[subset["breach"]]
        fig.add_trace(go.Scatter(x=breaches["date_id"], y=breaches["realized_return_pct"],
                                  mode="markers", name=f"Breaches (n={len(breaches)})",
                                  marker=dict(color="red", size=6)))
        fig.update_layout(title="Realized Returns vs VaR Threshold", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

        kupiec = kupiec_pof_test(subset["breach"], confidence=sel_confidence)
        m1, m2, m3 = st.columns(3)
        m1.metric("Observations", kupiec["n_observations"])
        m2.metric("Breaches", kupiec["n_breaches"],
                   f"{kupiec['observed_breach_rate']:.2%} vs {kupiec['expected_breach_rate']:.2%} expected")
        m3.metric("Kupiec p-value", f"{kupiec['p_value']:.4f}",
                   "Rejected" if kupiec["reject_h0_at_5pct"] else "Not rejected")
        st.write(kupiec["verdict"])

# ---------------------------------------------------------------------------
# Stress Test
# ---------------------------------------------------------------------------
with tab_stress:
    if stress_tests.empty:
        st.info("No stress test results found yet -- run the stress test and push to fact_stress_test.")
    else:
        st.subheader("Historical Scenario: 2008 GFC")
        row = stress_tests.iloc[0]

        # Pair against the closest available full-sample 95% historical VaR, if present
        var_95_hist = risk_metrics[
            (risk_metrics["method"] == "historical") & (risk_metrics["confidence"] == 0.95)
        ].sort_values("date_id", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Scenario window", f"{row['start_date']} to {row['end_date']}")
        c2.metric("Cumulative P&L", f"{row['portfolio_pnl_pct']:.2%}")
        c3.metric("P&L value", f"${row['portfolio_pnl_value']:,.0f}")

        if not var_95_hist.empty:
            var_pct = var_95_hist.iloc[0]["var_pct"]
            gap = stress_vs_var_gap(row["portfolio_pnl_pct"], var_pct)

            fig = go.Figure(go.Bar(
                x=["95% VaR (single day)", "2008 GFC (cumulative)"],
                y=[var_pct, abs(row["portfolio_pnl_pct"])],
                marker_color=["orange", "darkred"],
                text=[f"{var_pct:.1%}", f"{abs(row['portfolio_pnl_pct']):.1%}"],
                textposition="outside",
            ))
            fig.update_layout(title="Statistical VaR vs. Actual 2008 Crisis Loss", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"The 2008 GFC scenario produced a loss **{gap['gap_multiple']:.1f}x** the single-day "
                f"95% VaR estimate -- a reminder that VaR characterizes a typical bad day, not a "
                f"sustained crisis."
            )

        st.subheader("All stress test runs")
        display_cols = ["scenario_name", "scenario_type", "start_date", "end_date",
                         "portfolio_pnl_pct", "portfolio_pnl_value"]
        st.dataframe(
            stress_tests[display_cols].rename(columns={
                "scenario_name": "Scenario", "scenario_type": "Type",
                "start_date": "Start", "end_date": "End",
                "portfolio_pnl_pct": "P&L %", "portfolio_pnl_value": "P&L ($)",
            }).style.format({"P&L %": "{:.2%}", "P&L ($)": "${:,.0f}"}),
            hide_index=True, use_container_width=True,
        )
