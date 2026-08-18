"""
Static reference data: the equity universe and portfolio metadata.
This is pure reference data (no logic) — consumed by the seed/insertion step,
portfolio.py, stress_testing.py, and the dashboard.
"""

TICKERS = {
    "AAPL": {"name": "Apple Inc.",              "sector": "Technology",        "cap": "large"},
    "MSFT": {"name": "Microsoft Corp.",          "sector": "Technology",        "cap": "large"},
    "JPM":  {"name": "JPMorgan Chase & Co.",     "sector": "Financials",        "cap": "large"},
    "GS":   {"name": "Goldman Sachs Group",      "sector": "Financials",        "cap": "large"},
    "XOM":  {"name": "Exxon Mobil Corp.",        "sector": "Energy",            "cap": "large"},
    "JNJ":  {"name": "Johnson & Johnson",        "sector": "Healthcare",        "cap": "large"},
    "MDT":  {"name": "Medtronic plc",            "sector": "Healthcare",        "cap": "large"},
    "PG":   {"name": "Procter & Gamble",         "sector": "Consumer Staples",  "cap": "large"},
    "WMT":  {"name": "Walmart Inc.",             "sector": "Consumer Staples",  "cap": "large"},
    "CAT":  {"name": "Caterpillar Inc.",         "sector": "Industrials",       "cap": "large"},
    "NEE":  {"name": "NextEra Energy",           "sector": "Utilities",         "cap": "large"},
    "DIS":  {"name": "Walt Disney Co.",          "sector": "Communication",     "cap": "large"},
    "F":    {"name": "Ford Motor Co.",           "sector": "Consumer Discretionary", "cap": "mid"},
    "NEM":  {"name": "Newmont Corp.",            "sector": "Materials",         "cap": "mid"},
}

PORTFOLIO_NAME = "Diversified Equity Portfolio"
PORTFOLIO_DESCRIPTION = "Equal-weight 14-stock portfolio across 10 sectors, large + mid cap, for market risk demonstration."
