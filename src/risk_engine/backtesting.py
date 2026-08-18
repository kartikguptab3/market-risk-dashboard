"""
VaR backtesting: rolling-window breach detection + Kupiec Proportion-of-Failures
(POF) test.

This is the piece that shows model VALIDATION skill, not just calculation --
core to what a market risk analyst actually does day to day.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from risk_engine.portfolio import portfolio_returns
from risk_engine.var_cvar import historical_var_cvar, parametric_var_cvar

_METHOD_FUNCS = {
    "historical": historical_var_cvar,
    "parametric": parametric_var_cvar,
}


def rolling_backtest(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    method: str = "historical",
    confidence: float = 0.95,
    window: int = 250,
) -> pd.DataFrame:
    if method not in _METHOD_FUNCS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHOD_FUNCS)}.")

    var_func = _METHOD_FUNCS[method]
    port_returns = portfolio_returns(returns_df, weights)

    records = []
    for t in range(window, len(returns_df)):
        # trailing window strictly BEFORE day t -- no look-ahead
        window_df = returns_df.iloc[t - window : t]

        predicted_var, _ = var_func(window_df, weights, confidence)

        date_id = returns_df.index[t]
        realized_return = port_returns.iloc[t]
        breach = realized_return < -predicted_var

        records.append({
            "date_id": date_id,
            "method": method,
            "confidence": confidence,
            "predicted_var_pct": predicted_var,
            "realized_return_pct": realized_return,
            "breach": breach,
        })

    return pd.DataFrame(records)


def kupiec_pof_test(breach_series: pd.Series, confidence: float = 0.95) -> dict:
    n = len(breach_series)
    x = int(breach_series.sum())

    p_expected = 1 - confidence
    p_observed = x / n

    eps = 1e-10  # guards log(0) when x == 0 or x == n

    def log_likelihood(p: float) -> float:
        p = min(max(p, eps), 1 - eps)
        return (n - x) * np.log(1 - p) + x * np.log(p)

    lr_statistic = -2 * (log_likelihood(p_expected) - log_likelihood(p_observed))
    p_value = 1 - chi2.cdf(lr_statistic, df=1)
    reject_h0 = p_value < 0.05

    if not reject_h0:
        verdict = (
            f"Model NOT rejected at 5% significance (p={p_value:.4f}). Observed breach "
            f"rate ({p_observed:.2%}) is statistically consistent with the expected rate "
            f"({p_expected:.2%}) -- the model appears well-calibrated."
        )
    elif p_observed > p_expected:
        verdict = (
            f"Model REJECTED at 5% significance (p={p_value:.4f}). Observed breach rate "
            f"({p_observed:.2%}) is significantly HIGHER than expected ({p_expected:.2%}) -- "
            f"the model is underestimating risk (too many losses exceed predicted VaR)."
        )
    else:
        verdict = (
            f"Model REJECTED at 5% significance (p={p_value:.4f}). Observed breach rate "
            f"({p_observed:.2%}) is significantly LOWER than expected ({p_expected:.2%}) -- "
            f"the model is overestimating risk (too conservative, predicted losses too large)."
        )

    return {
        "n_observations": n,
        "n_breaches": x,
        "expected_breach_rate": p_expected,
        "observed_breach_rate": p_observed,
        "lr_statistic": lr_statistic,
        "p_value": p_value,
        "reject_h0_at_5pct": reject_h0,
        "verdict": verdict,
    }
