"""
Risk pipeline orchestrator: pulls fact_returns from Supabase, computes
VaR/CVaR (parametric + historical, both confidence levels), runs the
backtest, runs stress tests, and writes everything back to Supabase.

This is the ONLY module allowed to import from both data_pipeline/ and
risk_engine/ — it's the glue layer. risk_engine/ itself stays free of any
Supabase-specific code.

Scope note: Monte Carlo VaR and the hypothetical sector-shock scenarios are
out of scope for this pipeline (see var_cvar.py / stress_testing.py) --
only the parametric + historical methods and the historical scenarios in
HISTORICAL_SCENARIOS run here.

Run standalone (after fetch_data.py has populated prices/returns):
    python -m src.risk_engine.run_risk_pipeline
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/ on sys.path

import pandas as pd
from sqlalchemy import text

from universe import PORTFOLIO_NAME
from risk_config import ROLLING_WINDOW_DAYS, CONFIDENCE_LEVELS, ASSUMED_PORTFOLIO_VALUE, HISTORICAL_SCENARIOS
from data_pipeline.db import get_engine, upsert_dataframe, read_table
from risk_engine.var_cvar import compute_all_methods
from risk_engine.backtesting import rolling_backtest, kupiec_pof_test
from risk_engine.stress_testing import historical_scenario_pnl


def load_returns_matrix(engine):
    portfolio_id = int(read_table(
        "SELECT portfolio_id FROM dim_portfolio WHERE portfolio_name = :name",
        engine=engine, params={"name": PORTFOLIO_NAME},
    )["portfolio_id"].iloc[0])

    weights_df = read_table(
        """
        SELECT da.ticker, dpw.weight
        FROM dim_portfolio_weight dpw
        JOIN dim_asset da ON dpw.asset_id = da.asset_id
        WHERE dpw.portfolio_id = :pid
        """,
        engine=engine, params={"pid": portfolio_id},
    )
    weights = weights_df.set_index("ticker")["weight"]

    returns_long = read_table(
        """
        SELECT fr.date_id AS date, da.ticker, fr.simple_return
        FROM fact_returns fr
        JOIN dim_asset da ON fr.asset_id = da.asset_id
        ORDER BY fr.date_id
        """,
        engine=engine,
    )
    returns_long["date"] = pd.to_datetime(returns_long["date"])
    returns_df = returns_long.pivot(index="date", columns="ticker", values="simple_return")
    returns_df = returns_df.dropna(how="any")

    return returns_df, weights, portfolio_id


def compute_current_var_cvar(engine, returns_df: pd.DataFrame, weights: pd.Series, portfolio_id: int) -> pd.DataFrame:
    window_df = returns_df.tail(ROLLING_WINDOW_DAYS)
    as_of_date = window_df.index.max().date()

    results = pd.concat(
        [compute_all_methods(window_df, weights, confidence=c) for c in CONFIDENCE_LEVELS],
        ignore_index=True,
    )

    results["date_id"] = as_of_date
    results["portfolio_id"] = portfolio_id
    results["window_days"] = ROLLING_WINDOW_DAYS
    results["var_value"] = results["var_pct"] * ASSUMED_PORTFOLIO_VALUE
    results["cvar_value"] = results["cvar_pct"] * ASSUMED_PORTFOLIO_VALUE

    cols = ["date_id", "portfolio_id", "method", "confidence",
            "var_pct", "var_value", "cvar_pct", "cvar_value", "window_days"]
    return results[cols]


def run_backtest_and_store(engine, returns_df: pd.DataFrame, weights: pd.Series, portfolio_id: int) -> None:
    method, confidence = "historical", CONFIDENCE_LEVELS[0]  # 95% -- fast relative to Monte Carlo

    backtest_df = rolling_backtest(returns_df, weights, method=method, confidence=confidence, window=ROLLING_WINDOW_DAYS)
    if backtest_df.empty:
        print(f"Not enough history for a {ROLLING_WINDOW_DAYS}-day backtest window yet -- skipping.")
        return

    backtest_df["portfolio_id"] = portfolio_id
    backtest_df["date_id"] = pd.to_datetime(backtest_df["date_id"]).dt.date

    cols = ["date_id", "portfolio_id", "method", "confidence", "predicted_var_pct", "realized_return_pct", "breach"]
    upsert_dataframe(
        backtest_df[cols], "fact_var_backtest",
        pk_cols=["date_id", "portfolio_id", "method", "confidence"], engine=engine,
    )

    kupiec = kupiec_pof_test(backtest_df["breach"], confidence=confidence)
    print(f"Kupiec POF test ({method}, {confidence:.0%}, n={kupiec['n_observations']}): {kupiec['verdict']}")


def run_stress_tests_and_store(engine, returns_df: pd.DataFrame, weights: pd.Series, portfolio_id: int) -> None:
    rows = []
    for scenario_name, (start_date, end_date) in HISTORICAL_SCENARIOS.items():
        result = historical_scenario_pnl(returns_df, weights, start_date, end_date)
        if result["cumulative_pnl_pct"] is None:
            continue
        rows.append({
            "portfolio_id": portfolio_id,
            "scenario_name": scenario_name,
            "scenario_type": "historical",
            "start_date": result["start_date"],
            "end_date": result["end_date"],
            "portfolio_pnl_pct": result["cumulative_pnl_pct"],
            "portfolio_pnl_value": result["cumulative_pnl_pct"] * ASSUMED_PORTFOLIO_VALUE,
        })

    if not rows:
        print("No stress scenarios overlapped available return history -- skipping.")
        return

    stress_df = pd.DataFrame(rows)

    # fact_stress_test has only a serial PK (no natural unique key) -- clear
    # this portfolio's prior rows for these scenario names before inserting
    # fresh ones, so a daily rerun doesn't pile up duplicates for scenario
    # windows whose underlying historical prices never change.
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM fact_stress_test WHERE portfolio_id = :pid AND scenario_name = ANY(:names)"),
            {"pid": portfolio_id, "names": list(stress_df["scenario_name"])},
        )
    stress_df.to_sql("fact_stress_test", engine, if_exists="append", index=False)
    print(f"Stored {len(stress_df)} stress test result(s).")


def run() -> None:
    engine = get_engine()
    returns_df, weights, portfolio_id = load_returns_matrix(engine)

    risk_metrics_df = compute_current_var_cvar(engine, returns_df, weights, portfolio_id)
    upsert_dataframe(
        risk_metrics_df, "fact_risk_metrics",
        pk_cols=["date_id", "portfolio_id", "method", "confidence"], engine=engine,
    )
    print(f"Stored {len(risk_metrics_df)} risk metric row(s).")

    run_backtest_and_store(engine, returns_df, weights, portfolio_id)
    run_stress_tests_and_store(engine, returns_df, weights, portfolio_id)


if __name__ == "__main__":
    run()
