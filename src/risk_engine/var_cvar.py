"""
Core VaR / CVaR (Expected Shortfall) calculation engine.

Keep every function here PURE -- take numpy/pandas data in, return numbers out.
No database calls, no yfinance calls. This is what makes these functions
independently unit-testable (see tests/test_var_cvar.py) and reusable from
both notebooks and run_risk_pipeline.py.

Weight alignment, portfolio returns, and the asset-level covariance matrix
are NOT reimplemented here -- they live in portfolio.py as the single source
of truth, and backtesting.py / stress_testing.py should follow the same
pattern rather than duplicating this logic again.

Convention: VaR and CVaR are returned as POSITIVE numbers representing a
loss fraction of portfolio value (e.g. 0.023 = a loss of 2.3%).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from risk_engine.portfolio import align_weights, portfolio_returns, covariance_matrix


def parametric_var_cvar(returns_df: pd.DataFrame, weights: pd.Series, confidence: float = 0.95):
    """
    Variance-covariance (Delta-Normal) method. Assumes portfolio returns are
    normally distributed.

    mu, sigma computed from the mean vector and covariance matrix of
    returns_df, weighted by `weights` (sigma_p = sqrt(w^T * Cov * w)) --
    not from portfolio_returns().std(), since this generalizes to
    stress testing / risk decomposition later, where the full covariance
    structure (not just the aggregate series) is needed.

    Returns (var, cvar), both as positive numbers.
    """
    aligned_weights = align_weights(weights, returns_df)
    cov = covariance_matrix(returns_df)

    mu = (returns_df.mean() * aligned_weights).sum()
    sigma = np.sqrt(aligned_weights.values @ cov.values @ aligned_weights.values)

    z = norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / (1 - confidence))

    return var, cvar


def historical_var_cvar(returns_df: pd.DataFrame, weights: pd.Series, confidence: float = 0.95):
    """
    Historical simulation method. No distributional assumption -- uses the
    empirical distribution of actual historical portfolio returns.

    VaR = -1 * the (1-confidence) percentile of portfolio returns for the
    window (e.g. for 95% confidence, the 5th percentile).
    CVaR = -1 * the average of all returns worse than -VaR (the tail
    beyond the VaR cutoff).

    Returns (var, cvar), both as positive numbers.
    """
    port_returns = portfolio_returns(returns_df, weights)

    var = -np.percentile(port_returns, (1 - confidence) * 100)
    tail_losses = port_returns[port_returns <= -var]
    cvar = -tail_losses.mean()

    return var, cvar


def monte_carlo_var_cvar(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    random_seed: int | None = 42,
):
    """
    Monte Carlo simulation method.

    NOT YET IMPLEMENTED -- deliberately left as a stub. Cut from project
    scope alongside hypothetical stress scenarios to manage time; parametric
    + historical VaR/CVaR still cover two independent estimation approaches.
    compute_all_methods() skips this method until/unless it's filled in.
    """
    raise NotImplementedError


def compute_all_methods(returns_df: pd.DataFrame, weights: pd.Series, confidence: float = 0.95,
                         n_simulations: int = 10_000) -> pd.DataFrame:
    """
    Calls historical and parametric methods (Monte Carlo pending -- see
    monte_carlo_var_cvar) and returns a tidy DataFrame with columns
    [method, confidence, var_pct, cvar_pct] -- one row per method.
    """
    results = []

    var_h, cvar_h = historical_var_cvar(returns_df, weights, confidence)
    results.append({"method": "historical", "confidence": confidence, "var_pct": var_h, "cvar_pct": cvar_h})

    var_p, cvar_p = parametric_var_cvar(returns_df, weights, confidence)
    results.append({"method": "parametric", "confidence": confidence, "var_pct": var_p, "cvar_pct": cvar_p})

    return pd.DataFrame(results)
