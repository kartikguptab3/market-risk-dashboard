"""
Stress testing & scenario analysis.

Scope: historical scenario only (2008 GFC). Hypothetical sector-shock
scenarios (HYPOTHETICAL_SCENARIOS in risk_config.py) were cut to keep
project scope manageable -- historical_scenario_pnl below is written
generically so any entry in HISTORICAL_SCENARIOS can be run, but the
notebook only calls it for 2008.

The important output isn't the stress loss number alone -- it's the GAP
between stressed loss and the VaR/CVaR estimate. That gap is the entire
justification for stress testing existing as a complement to statistical VaR.
"""
import pandas as pd

from risk_engine.portfolio import portfolio_returns


def historical_scenario_pnl(
    price_returns_df: pd.DataFrame,
    weights: pd.Series,
    start_date: str,
    end_date: str,
) -> dict:
    window_df = price_returns_df.loc[start_date:end_date]

    if window_df.empty:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "cumulative_pnl_pct": None,
            "n_trading_days": 0,
            "worst_single_day_pct": None,
        }

    port_returns = portfolio_returns(window_df, weights)
    cumulative_pnl = (1 + port_returns).prod() - 1

    return {
        "start_date": start_date,
        "end_date": end_date,
        "cumulative_pnl_pct": cumulative_pnl,
        "n_trading_days": len(port_returns),
        "worst_single_day_pct": port_returns.min(),
    }


def stress_vs_var_gap(stressed_loss_pct: float, var_pct: float) -> dict:
    gap = abs(stressed_loss_pct) - var_pct
    gap_multiple = abs(stressed_loss_pct) / var_pct

    return {
        "stressed_loss_pct": stressed_loss_pct,
        "var_pct": var_pct,
        "gap": gap,
        "gap_multiple": gap_multiple,
    }
