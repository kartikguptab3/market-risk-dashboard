"""
Risk-engine parameters: rolling windows, confidence levels, simulation counts,
and stress-test scenario definitions.
Consumed by var_cvar.py, backtesting.py, and stress_testing.py.
"""

DATA_START_DATE = "2005-01-01"

ROLLING_WINDOW_DAYS = 250          # ~1 trading year, standard for VaR estimation
CONFIDENCE_LEVELS = [0.95, 0.99]
MONTE_CARLO_SIMULATIONS = 10_000
ASSUMED_PORTFOLIO_VALUE = 10_000_000  # $10M notional, purely for currency-denominated VaR display

HISTORICAL_SCENARIOS = {
    "2008 GFC (Lehman collapse)": ("2008-09-01", "2008-11-30"),
    "COVID-19 Crash":             ("2020-02-19", "2020-03-23"),
    "2022 Rate-Hike Selloff":     ("2022-01-01", "2022-06-30"),
}

HYPOTHETICAL_SCENARIOS = {
    "Equity -20% Broad Shock":        {"ALL": -0.20},
    "Energy Sector Crash -35%":       {"Energy": -0.35, "OTHER": -0.05},
    "Rate Shock (Financials -15%, Utilities -10%)": {"Financials": -0.15, "Utilities": -0.10, "OTHER": -0.03},
}
