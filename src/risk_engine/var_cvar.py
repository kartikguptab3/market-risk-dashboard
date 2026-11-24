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

from risk_engine.portfolio import align_weights, portfolio_returns, covariance_matrix, ewma_covariance_matrix


def parametric_var_cvar(returns_df: pd.DataFrame, weights: pd.Series, confidence: float = 0.95,
                        volatility_model: str = "sample", ewma_lambda: float = 0.94):
    """
    Variance-covariance (Delta-Normal) method. Assumes portfolio returns are
    normally distributed.

    mu, sigma computed from the mean vector and covariance matrix of
    returns_df, weighted by `weights` (sigma_p = sqrt(w^T * Cov * w)) --
    not from portfolio_returns().std(), since this generalizes to
    stress testing / risk decomposition later, where the full covariance
    structure (not just the aggregate series) is needed.

    volatility_model:
      "sample" -- equally weighted covariance over the whole window, sample mean.
      "ewma"   -- RiskMetrics EWMA covariance (decay `ewma_lambda`), zero mean.
                  Recent days dominate, so VaR reacts quickly to volatility
                  clustering instead of lagging by months.

    Returns (var, cvar), both as positive numbers.
    """
    if volatility_model not in ("sample", "ewma"):
        raise ValueError("volatility_model must be 'sample' or 'ewma'")

    aligned_weights = align_weights(weights, returns_df)

    if volatility_model == "ewma":
        cov = ewma_covariance_matrix(returns_df, lam=ewma_lambda)
        mu = 0.0  # RiskMetrics convention
    else:
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
    distribution: str = "normal",
    t_df: int = 5,
):
    """
    Monte Carlo simulation method.

    1. Estimate the mean vector and covariance matrix from returns_df
       (same inputs as the parametric method -- via portfolio.py).
    2. Cholesky-decompose the covariance: Cov = L L^T.
    3. Draw n_simulations vectors of independent shocks Z and correlate
       them: simulated asset returns = mu + Z L^T. This preserves the
       cross-asset correlation structure.
    4. Collapse each simulated scenario into a portfolio return (w . r) and
       read VaR / CVaR off the simulated distribution exactly like the
       historical method does.

    distribution:
      "normal"    -- multivariate normal shocks. Converges to parametric VaR
                     as n_simulations grows (a useful sanity check).
      "student_t" -- multivariate Student-t shocks with `t_df` degrees of
                     freedom, rescaled to the same covariance. Fatter tails,
                     so 99% VaR / CVaR come out higher than under normality.

    random_seed fixes the draw so results are reproducible run to run.

    Returns (var, cvar), both as positive numbers.
    """
    if distribution not in ("normal", "student_t"):
        raise ValueError("distribution must be 'normal' or 'student_t'")

    aligned_weights = align_weights(weights, returns_df)
    mu = returns_df.mean().values
    cov = covariance_matrix(returns_df).values

    # tiny diagonal jitter guards against a near-singular covariance
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))

    rng = np.random.default_rng(random_seed)
    z = rng.standard_normal((n_simulations, len(mu)))

    if distribution == "student_t":
        # multivariate t = normal / sqrt(chi2/df); rescale so variance matches cov
        chi2_draws = rng.chisquare(t_df, size=(n_simulations, 1))
        z = z / np.sqrt(chi2_draws / t_df) * np.sqrt((t_df - 2) / t_df)

    simulated_asset_returns = mu + z @ L.T          # (n_simulations, n_assets)
    simulated_port_returns = simulated_asset_returns @ aligned_weights.values

    var = -np.percentile(simulated_port_returns, (1 - confidence) * 100)
    tail_losses = simulated_port_returns[simulated_port_returns <= -var]
    cvar = -tail_losses.mean()

    return var, cvar


def compute_all_methods(returns_df: pd.DataFrame, weights: pd.Series, confidence: float = 0.95,
                         n_simulations: int = 10_000, ewma_lambda: float = 0.94) -> pd.DataFrame:
    """
    Calls the historical, parametric (sample and EWMA) and Monte Carlo methods
    and returns a tidy DataFrame with columns [method, confidence, var_pct,
    cvar_pct] -- one row per method.
    """
    results = []

    var_h, cvar_h = historical_var_cvar(returns_df, weights, confidence)
    results.append({"method": "historical", "confidence": confidence, "var_pct": var_h, "cvar_pct": cvar_h})

    var_p, cvar_p = parametric_var_cvar(returns_df, weights, confidence)
    results.append({"method": "parametric", "confidence": confidence, "var_pct": var_p, "cvar_pct": cvar_p})

    var_e, cvar_e = parametric_var_cvar(returns_df, weights, confidence,
                                        volatility_model="ewma", ewma_lambda=ewma_lambda)
    results.append({"method": "parametric_ewma", "confidence": confidence, "var_pct": var_e, "cvar_pct": cvar_e})

    var_mc, cvar_mc = monte_carlo_var_cvar(returns_df, weights, confidence, n_simulations=n_simulations)
    results.append({"method": "monte_carlo", "confidence": confidence, "var_pct": var_mc, "cvar_pct": cvar_mc})

    return pd.DataFrame(results)
