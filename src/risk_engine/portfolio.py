"""
Portfolio construction utilities.

Pure functions: DataFrame/Series in, numbers or DataFrame/Series out.
No database awareness -- inputs (returns, weights) are always passed in
by the caller (typically run_risk_pipeline.py, which is the only module
allowed to bridge data_pipeline/ and risk_engine/).

Consumed by var_cvar.py, backtesting.py, and stress_testing.py, so this
file owns the single source of truth for "what is the portfolio return
on a given day" and "what is the portfolio's risk (covariance/vol)".
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def align_weights(weights: pd.Series, returns_df: pd.DataFrame) -> pd.Series:
    """
    Reindex `weights` (indexed by ticker) to match the column order of
    `returns_df` (columns = tickers). This is the safety net against
    silently multiplying the wrong weight against the wrong ticker if
    the two inputs arrive in different orders.

    Raises if any ticker in returns_df is missing a weight, or vice versa,
    rather than silently filling with NaN/0 -- a missing weight is a data
    problem that should surface immediately, not get masked.
    """
    missing_weights = set(returns_df.columns) - set(weights.index)
    missing_returns = set(weights.index) - set(returns_df.columns)

    if missing_weights:
        raise ValueError(f"No weight found for tickers: {missing_weights}")
    if missing_returns:
        raise ValueError(f"No return data found for weighted tickers: {missing_returns}")

    return weights.reindex(returns_df.columns)


def portfolio_returns(returns_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Combine per-asset returns into a single portfolio return series.

    returns_df: index = date, columns = ticker, values = simple or log return
    weights:    index = ticker, values = portfolio weight (should sum to ~1)

    Returns a Series indexed by date.
    """
    aligned_weights = align_weights(weights, returns_df)
    return returns_df.mul(aligned_weights, axis=1).sum(axis=1)


def covariance_matrix(returns_df: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    """
    Asset-level covariance matrix from a returns DataFrame
    (index = date, columns = ticker).

    annualize: if True, scales daily covariance up to annual terms
    (covariance scales linearly with time, unlike volatility which
    scales with sqrt(time)).
    """
    cov = returns_df.cov()
    if annualize:
        cov = cov * TRADING_DAYS_PER_YEAR
    return cov


def portfolio_volatility(weights: pd.Series, cov_matrix: pd.DataFrame, annualize: bool = False) -> float:
    """
    Portfolio standard deviation: sqrt(w' Σ w).

    weights:    index = ticker, values = portfolio weight
    cov_matrix: asset-level covariance matrix (output of covariance_matrix(),
                with annualize matching the annualize flag used there --
                don't mix a daily cov_matrix with annualize=True here, or
                you'll double-scale)
    """
    aligned_weights = weights.reindex(cov_matrix.columns)
    variance = aligned_weights.values @ cov_matrix.values @ aligned_weights.values
    vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol
