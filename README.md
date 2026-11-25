# Market Risk Dashboard

A market risk engine and dashboard for a 14-stock, 10-sector US equity
portfolio: VaR/CVaR via four methods (parametric, parametric with EWMA
volatility, historical simulation, Monte Carlo), rolling backtesting with the Kupiec POF test, and historical
scenario stress testing, served through a Streamlit dashboard.

**Live demo:** https://market-risk-dashboard-kartikguptab3.streamlit.app/
(free-tier hosting, so it may take ~30-60s to wake up on a cold visit)

## Architecture

```
yfinance ──▶ fetch_data.py ──▶ Neon (Postgres, star schema) ──▶ run_risk_pipeline.py ──▶ Neon (risk metrics)
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
- **Database**: serverless Postgres on Neon. Compute scales to zero when
  idle and resumes on the next query, so the free plan never pauses the
  project.

## Star schema

See `sql/schema.sql`. Dimension tables (`dim_asset`, `dim_date`,
`dim_portfolio`, `dim_portfolio_weight`) describe the "who/what/when";
fact tables (`fact_price`, `fact_returns`, `fact_risk_metrics`,
`fact_var_backtest`, `fact_stress_test`) hold the measures.

## Setup

### 1. Create a Neon project
1. Sign up at [neon.com](https://neon.com) (the free plan is enough for
   this data volume).
2. Create a new project. Pick a US region if the dashboard will run on
   Streamlit Community Cloud (its servers are in the US).
3. Click **Connect**, keep **Connection pooling** on, and copy the
   connection string. That single value is your `DATABASE_URL`.

### 2. Run the schema
1. Open **SQL Editor** in the Neon console.
2. Paste in `sql/schema.sql` and run it.
3. Confirm the tables show up under **Tables**.

### 3. Local environment
```bash
git clone https://github.com/kartikguptab3/market-risk-dashboard.git
cd market-risk-dashboard
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in DATABASE_URL with your Neon connection string from step 1
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
3. Under the app's **Settings → Secrets**, add the connection string in
   TOML format:
   ```toml
   DATABASE_URL = "postgresql://...neon.tech/neondb?sslmode=require"
   ```
   `dashboard/app.py`'s `get_engine()` checks `st.secrets` first and falls
   back to `.env` locally, so no extra wiring is needed either way.

### 8. GitHub Actions (optional)
Two workflows in `.github/workflows/`, both with a manual **Run workflow**
button in the **Actions** tab:
- `daily_pipeline.yml`: runs step 4's two commands on demand. The daily
  weekday schedule is switched off to stay within Neon's free-tier storage
  (each run re-upserts the full price history). Needs a repository secret
  `DATABASE_URL`.
- `keep_awake.yml`: opens the hosted dashboard every 6 hours (and clicks
  the wake-up button if it has gone to sleep). Needs a repository variable
  `APP_URL` set to the app's link.

## Methodology summary

- **VaR/CVaR**: 250-trading-day rolling window, 95% and 99% confidence,
  four methods:
  - *Parametric*: assumes normally distributed returns (closed-form),
    equally weighted covariance over the window.
  - *Parametric EWMA*: same Delta-Normal formula, but the covariance is the
    RiskMetrics exponentially weighted forecast (λ = 0.94, zero mean). Recent
    days carry most of the weight (effective memory ≈ 17 days), so VaR adapts
    to volatility clustering instead of lagging a 250-day average.
  - *Historical simulation*: uses the empirical return distribution, no
    normality assumption.
  - *Monte Carlo*: 10,000 simulated one-day scenarios drawn from the
    window's mean vector and covariance matrix, correlated via Cholesky
    decomposition. Multivariate normal by default (converges to the
    parametric result, a built-in sanity check); an optional Student-t
    mode (`distribution="student_t"`) fattens the tails.
- **Backtesting**: rolling out-of-sample VaR predictions checked against
  realized returns, validated with the Kupiec (1995) Proportion-of-Failures
  test.
- **Stress testing**: historical scenario replay of the 2008 GFC (Lehman
  collapse, Sep-Nov 2008) and the COVID-19 crash (Feb-Mar 2020) against
  current portfolio weights, compared against the VaR/CVaR baseline to show
  the gap between "normal times" statistical models and genuine crisis
  outcomes.

## Backtest results (95% VaR, ~5,250 daily observations since 2006)

| Method | Breach rate (expected 5.00%) | Kupiec p-value | Verdict |
|---|---|---|---|
| Historical | 5.67% | 0.029 | Rejected -- underestimates risk |
| Parametric (equal-weighted) | 5.67% | 0.029 | Rejected -- underestimates risk |
| Parametric (EWMA, λ = 0.94) | 5.40% | 0.186 | Not rejected |
| Monte Carlo | 5.55% | 0.070 | Not rejected |

The two equal-weighted 250-day models react too slowly when volatility
jumps (2008, 2020) and get rejected; swapping the same Delta-Normal formula
onto an EWMA covariance fixes the calibration.

## Dashboard

Light-themed Streamlit app, four tabs:

- **Overview** -- composition, sector mix, and indexed price performance
  with a period slider, ticker highlighting, daily/weekly/monthly smoothing
  and an equal-weight portfolio line.
- **VaR / CVaR** -- current one-day VaR or CVaR by method (% or $), plus
  EWMA vs. 250-day rolling portfolio volatility.
- **Backtesting** -- realized returns vs. each method's VaR line with
  breaches marked, a "zoom to crisis" selector, a Kupiec table across all
  methods and an overlay of every method's VaR threshold.
- **Stress Test** -- pick 2008 GFC or COVID-19: the portfolio's cumulative
  P&L path against a VaR × √t bound, and the actual loss vs. the VaR the
  model was predicting the day the crisis began.

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
│   ├── config.py                     Database (Neon) connection config
│   ├── universe.py                   Tickers, sectors, portfolio metadata
│   ├── risk_config.py                Rolling windows, confidence levels, scenarios
│   ├── data_pipeline/                yfinance → Postgres ETL
│   └── risk_engine/                  VaR/CVaR (incl. EWMA), backtesting, stress testing
├── dashboard/app.py                  Streamlit app
├── .streamlit/config.toml            Dashboard theme (light)
├── .github/workflows/                On-demand pipeline + keep-awake (optional)
└── tests/                            Unit tests for risk_engine
```

## License

MIT, see [LICENSE](LICENSE).
