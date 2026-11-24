"""
Unit tests for the risk_engine pure functions.

Because var_cvar.py has zero database dependency, these run entirely
against synthetic data -- no database connection needed.

Covers all three methods: parametric, historical and Monte Carlo.

Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from risk_engine.var_cvar import parametric_var_cvar, historical_var_cvar, monte_carlo_var_cvar

METHODS = {
    "parametric": parametric_var_cvar,
    "historical": historical_var_cvar,
    "monte_carlo": monte_carlo_var_cvar,
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


def test_monte_carlo_converges_to_parametric(synthetic_returns):
    """
    With normal shocks, Monte Carlo is simulating the exact distribution the
    parametric method assumes -- so with enough paths the two should agree.
    """
    returns_df, weights = synthetic_returns
    for confidence in (0.95, 0.99):
        var_p, cvar_p = parametric_var_cvar(returns_df, weights, confidence)
        var_mc, cvar_mc = monte_carlo_var_cvar(returns_df, weights, confidence, n_simulations=200_000)
        assert np.isclose(var_mc, var_p, rtol=0.02)
        assert np.isclose(cvar_mc, cvar_p, rtol=0.03)


def test_monte_carlo_is_reproducible_with_seed(synthetic_returns):
    """Same seed -> identical result, so stored daily numbers don't jitter."""
    returns_df, weights = synthetic_returns
    assert monte_carlo_var_cvar(returns_df, weights, random_seed=1) == \
           monte_carlo_var_cvar(returns_df, weights, random_seed=1)


def test_student_t_has_fatter_tail_than_normal(synthetic_returns):
    """At 99%, fat-tailed shocks with the same covariance should give a larger CVaR."""
    returns_df, weights = synthetic_returns
    _, cvar_normal = monte_carlo_var_cvar(returns_df, weights, 0.99, n_simulations=100_000)
    _, cvar_t = monte_carlo_var_cvar(returns_df, weights, 0.99, n_simulations=100_000,
                                     distribution="student_t", t_df=4)
    assert cvar_t > cvar_normal


# ---------------------------------------------------------------------------
# EWMA volatility
# ---------------------------------------------------------------------------
from risk_engine.portfolio import ewma_weights, ewma_covariance_matrix, ewma_volatility


def test_ewma_weights_sum_to_one_and_favor_recent():
    w = ewma_weights(250, lam=0.94)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) > 0)  # oldest first, so weights increase


def test_ewma_covariance_matches_recursion():
    """Matrix form must equal the textbook recursion Σ_t = λΣ_{t-1} + (1-λ) r rᵀ."""
    rng = np.random.default_rng(0)
    returns_df = pd.DataFrame(rng.normal(0, 0.01, size=(300, 3)), columns=["A", "B", "C"])
    lam = 0.94
    cov = np.zeros((3, 3))
    for r in returns_df.values:
        cov = lam * cov + (1 - lam) * np.outer(r, r)
    cov /= 1 - lam ** len(returns_df)  # same normalization as ewma_weights
    assert np.allclose(ewma_covariance_matrix(returns_df, lam).values, cov)


def test_ewma_volatility_series_matches_covariance_diagonal():
    """Last forecast of the series (+1 step) equals sqrt of the matrix diagonal, burn-in aside."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(0, 0.01, size=1000))
    vol = ewma_volatility(s, lam=0.94)
    # next-day forecast from the recursion vs matrix form on the full series
    next_var = 0.94 * vol.iloc[-1] ** 2 + 0.06 * s.iloc[-1] ** 2
    matrix_var = ewma_covariance_matrix(s.to_frame("A"), 0.94).iloc[0, 0]
    assert np.isclose(np.sqrt(next_var), np.sqrt(matrix_var), rtol=1e-6)


def test_ewma_var_reacts_to_recent_vol_spike():
    """Calm 230 days then a 20-day high-vol burst: EWMA VaR must jump far more than equal-weighted."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, size=230)
    stressed = rng.normal(0, 0.03, size=20)
    returns_df = pd.DataFrame({"A": np.concatenate([calm, stressed])})
    weights = pd.Series([1.0], index=["A"])
    var_sample, _ = parametric_var_cvar(returns_df, weights, 0.99)
    var_ewma, _ = parametric_var_cvar(returns_df, weights, 0.99, volatility_model="ewma")
    assert var_ewma > 1.5 * var_sample


def test_parametric_ewma_obeys_var_cvar_ordering(synthetic_returns):
    returns_df, weights = synthetic_returns
    v95, c95 = parametric_var_cvar(returns_df, weights, 0.95, volatility_model="ewma")
    v99, c99 = parametric_var_cvar(returns_df, weights, 0.99, volatility_model="ewma")
    assert v99 >= v95 and c95 >= v95 and c99 >= v99


def test_parametric_rejects_unknown_volatility_model(synthetic_returns):
    returns_df, weights = synthetic_returns
    with pytest.raises(ValueError):
        parametric_var_cvar(returns_df, weights, volatility_model="garch")
