"""
Unit tests for the risk_engine pure functions.

Because var_cvar.py has zero database dependency, these run entirely
against synthetic data -- no Supabase connection needed.

Scope note: Monte Carlo VaR is not implemented (see var_cvar.py), so these
tests cover the parametric and historical methods only.

Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from risk_engine.var_cvar import parametric_var_cvar, historical_var_cvar

METHODS = {
    "parametric": parametric_var_cvar,
    "historical": historical_var_cvar,
}


@pytest.fixture
def synthetic_returns():
    """
    500 days x 4 tickers of correlated normal returns with a known mean
    vector and covariance matrix, so the historical and parametric methods
    are estimating the same well-understood distribution.
    """
    rng = np.random.default_rng(42)
    tickers = ["A", "B", "C", "D"]
    mean = np.array([0.0005, 0.0003, 0.0004, 0.0002])
    cov = np.array([
        [0.00040, 0.00010, 0.00008, 0.00005],
        [0.00010, 0.00030, 0.00006, 0.00004],
        [0.00008, 0.00006, 0.00025, 0.00003],
        [0.00005, 0.00004, 0.00003, 0.00020],
    ])
    data = rng.multivariate_normal(mean, cov, size=500)
    returns_df = pd.DataFrame(data, columns=tickers)
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=tickers)
    return returns_df, weights


def test_var_increases_with_confidence(synthetic_returns):
    """99% VaR should always be >= 95% VaR -- it covers more of the tail."""
    returns_df, weights = synthetic_returns
    for var_func in METHODS.values():
        var_95, _ = var_func(returns_df, weights, confidence=0.95)
        var_99, _ = var_func(returns_df, weights, confidence=0.99)
        assert var_99 >= var_95


def test_cvar_is_at_least_var(synthetic_returns):
    """CVaR (expected loss GIVEN a breach) should be >= VaR at the same confidence."""
    returns_df, weights = synthetic_returns
    for confidence in (0.95, 0.99):
        for var_func in METHODS.values():
            var, cvar = var_func(returns_df, weights, confidence=confidence)
            assert cvar >= var


def test_parametric_matches_known_normal_case():
    """
    Single-asset, single-period case where the answer is known by hand:
    mu=0, sigma=0.01, 95% confidence -> VaR = 1.645 * 0.01.
    """
    rng = np.random.default_rng(7)
    returns_df = pd.DataFrame({"A": rng.normal(loc=0.0, scale=0.01, size=5000)})
    weights = pd.Series([1.0], index=["A"])

    var_95, _ = parametric_var_cvar(returns_df, weights, confidence=0.95)

    assert np.isclose(var_95, 1.645 * 0.01, atol=0.001)
