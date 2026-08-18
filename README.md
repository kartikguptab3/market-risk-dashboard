# Market Risk Dashboard

A market risk engine and dashboard for a 14-stock, 10-sector US equity
portfolio: VaR/CVaR via two methods (parametric, historical simulation),
rolling backtesting with the Kupiec POF test, and historical scenario
stress testing, served through a Streamlit dashboard.

**Live demo:** https://market-risk-dashboard-kartikguptab3.streamlit.app/
(free-tier hosting, so it may take ~30-60s to wake up on a cold visit)

## Architecture

```
yfinance ──▶ fetch_data.py ──▶ Supabase (Postgres, star schema) ──▶ run_risk_pipeline.py ──▶ Supabase (risk metrics)
                                                                                                       │
                                                                                                       ▼
                                                                                          Streamlit dashboard (reads only)
```

- **Compute** (`fetch_data.py`, `run_risk_pipeline.py`) and **serving**
  (`dashboard/app.py`) are fully separated. The dashboard never recomputes
  anything, it only reads pre-computed tables. Keeps it fast, and matches
  how a real production risk system would be split.
- **`risk_engine/`** has zero database awareness: every function is pure
  (DataFrame/Series in, numbers out). That's what makes it unit-testable
  without a live database.

## Star schema

See `sql/schema.sql`. Dimension tables (`dim_asset`, `dim_date`,
`dim_portfolio`, `dim_portfolio_weight`) describe the "who/what/when";
fact tables (`fact_price`, `fact_returns`, `fact_risk_metrics`,
`fact_var_backtest`, `fact_stress_test`) hold the measures.

## Setup

### 1. Create a Supabase project
1. Sign up at [supabase.com](https://supabase.com) (free tier is enough
   for this data volume).
2. Create a new project and set a strong database password (you'll need
   it below).
3. Under **Project Settings → Database**, note the connection parameters
   (host, port, database name, user).

### 2. Run the schema
1. Open **SQL Editor** in the Supabase dashboard.
2. Paste in `sql/schema.sql` and run it.
3. Confirm the tables show up under **Table Editor**.

### 3. Local environment
```bash
git clone https://github.com/kartikguptab3/market-risk-dashboard.git
cd market-risk-dashboard
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in .env with your Supabase values from step 1
```

### 4. Run the pipeline
```bash
python -m src.data_pipeline.fetch_data       # pulls prices, populates dim/fact price tables
python -m src.risk_engine.run_risk_pipeline  # computes VaR/CVaR/backtest/stress, populates risk tables
```

### 5. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Run the tests
```bash
pytest tests/
```

### 7. Deploy the dashboard (Streamlit Community Cloud)
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. Deploy this repo, pointing at `dashboard/app.py`.
3. Under the app's **Settings → Secrets**, add the same Supabase values
   in TOML format:
   ```toml
   SUPABASE_DB_HOST = "..."
   SUPABASE_DB_PORT = "5432"
   SUPABASE_DB_NAME = "postgres"
   SUPABASE_DB_USER = "..."
   SUPABASE_DB_PASSWORD = "..."
   ```
   `dashboard/app.py`'s `get_engine()` checks `st.secrets` first and falls
   back to `.env` locally, so no extra wiring is needed either way.

## Methodology summary

- **VaR/CVaR**: 250-trading-day rolling window, 95% and 99% confidence,
  two methods:
  - *Parametric*: assumes normally distributed returns (closed-form).
  - *Historical simulation*: uses the empirical return distribution, no
    normality assumption.
- **Backtesting**: rolling out-of-sample VaR predictions checked against
  realized returns, validated with the Kupiec (1995) Proportion-of-Failures
  test.
- **Stress testing**: historical scenario replay (2008 GFC, COVID-19 crash,
  2022 rate-hike selloff) against current portfolio weights, compared
  against the VaR/CVaR baseline to show the gap between "normal times"
  statistical models and genuine crisis outcomes.

## Known limitations

See `notebooks/04_stress_test_writeup.ipynb` for the full discussion:
normality assumptions, historical-window dependence, correlation breakdown
in crises, and the arbitrary nature of the 250-day window are all worth
being upfront about.

## Project structure

```
market-risk-dashboard/
├── sql/schema.sql                    Star schema DDL
├── notebooks/                        Exploration + prototyping + final writeup
├── src/
│   ├── config.py                     Supabase connection config
│   ├── universe.py                   Tickers, sectors, portfolio metadata
│   ├── risk_config.py                Rolling windows, confidence levels, scenarios
│   ├── data_pipeline/                yfinance → Supabase ETL
│   └── risk_engine/                  VaR/CVaR, backtesting, stress testing
├── dashboard/app.py                  Streamlit app
└── tests/                            Unit tests for risk_engine
```

## License

MIT, see [LICENSE](LICENSE).
