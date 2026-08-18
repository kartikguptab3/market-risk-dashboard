# Market Risk Dashboard — Full Project Walkthrough

This is a script-by-script explanation of the project: what each file does, the
math/methodology behind it, why it's built the way it is, and the interview
angles it opens up. Read this the way you'd prep for a case-study interview —
not just "what does this function return" but "why does this exist, what
would break it, what would you change."

---

## 1. The big picture

```
yfinance ──▶ fetch_data.py ──▶ Supabase (Postgres, star schema) ──▶ run_risk_pipeline.py ──▶ Supabase (risk metrics)
                                                                                                       │
                                                                                                       ▼
                                                                                          Streamlit dashboard (reads only)
```

Three layers, each with one job:

- **`data_pipeline/`** — gets raw market data into the database. Knows about
  yfinance and Postgres. Knows nothing about VaR.
- **`risk_engine/`** — turns returns into risk numbers. Pure functions only:
  DataFrame/Series in, numbers out. Knows nothing about databases or
  yfinance. This is deliberate — it's what makes `tests/test_var_cvar.py`
  possible without a live database connection, and it's what makes the same
  functions reusable from notebooks, the pipeline, and (if you ever add it) a
  second dashboard or API.
- **`dashboard/app.py`** — reads pre-computed tables and renders them. Never
  recomputes anything. This is why the dashboard is fast: it's just SQL
  SELECTs and Plotly charts, not statistics.

`run_risk_pipeline.py` is the one exception to "layers don't know about each
other" — it's explicitly the glue module allowed to import from both
`data_pipeline/` and `risk_engine/`, because *something* has to wire raw data
to risk math to storage.

**Interview framing**: this is the compute/serve split you'd defend in a
system design question — batch-compute expensive numbers on a schedule,
serve cheap reads to users. It's also why a live incident in the dashboard
(bad query, Streamlit crash) can never corrupt risk data, and why a bad
pipeline run can't take down the dashboard for users looking at yesterday's
numbers.

---

## 2. The star schema (`sql/schema.sql`)

Classic dimension/fact design:

**Dimensions** (the "who/what/when", slowly changing or static):
- `dim_asset` — one row per ticker (name, sector, market cap tier)
- `dim_date` — one row per calendar date (year/quarter/month/day-of-week,
  pre-computed so downstream queries never call `EXTRACT()` at query time)
- `dim_portfolio` — one row per named portfolio (this project only ever has
  one, "Diversified Equity Portfolio", but the schema supports more without
  any changes)
- `dim_portfolio_weight` — which assets are in which portfolio, at what
  weight, as of what `effective_date`. The `effective_date` column is what
  makes this a *slowly changing dimension* — if you ever rebalanced the
  portfolio, you'd insert a new row with a new `effective_date` rather than
  overwrite the old weights, preserving history.

**Facts** (the measures, one row per event/observation):
- `fact_price`, `fact_returns` — one row per (date, asset)
- `fact_risk_metrics` — one row per (date, portfolio, method, confidence) —
  this shape is what lets the dashboard show parametric vs. historical VaR
  side by side at both 95% and 99% without any pivoting logic
- `fact_var_backtest` — one row per (date, portfolio, method, confidence),
  the exceedance log described below
- `fact_stress_test` — one row per scenario run (no natural composite key,
  just a serial `scenario_id` — every other fact table has a real composite
  primary key)

**Interview framing**: "Why a star schema instead of one wide table?" —
normalization avoids repeating ticker/sector text on every one of 76,000
price rows, and it's the standard shape for BI/reporting workloads where you
read far more than you write. "Why does `fact_stress_test` not have a
composite key?" — because a stress scenario re-run isn't naturally unique
per day the way a price is; the code handles this explicitly (see §5) with a
delete-then-insert pattern instead of upsert.

---

## 3. Configuration layer

Three small files, deliberately separated by concern rather than dumped into
one `config.py`:

- **`src/config.py`** — `SupabaseConfig`, a dataclass built from environment
  variables (`SUPABASE_DB_HOST`, etc.), plus `sqlalchemy_url()` which builds
  the `postgresql+psycopg2://...` connection string. This is the *only*
  place that knows how to talk to Postgres — swap Supabase for RDS someday
  and this is the only file that changes.
- **`src/universe.py`** — the actual investable universe: `TICKERS` (14
  stocks, sector, market cap tier), `PORTFOLIO_NAME`/`PORTFOLIO_DESCRIPTION`.
  Business/reference data, not connection plumbing.
- **`src/risk_config.py`** — every risk *parameter*: `ROLLING_WINDOW_DAYS`
  (250, ≈1 trading year), `CONFIDENCE_LEVELS` (`[0.95, 0.99]`),
  `MONTE_CARLO_SIMULATIONS` (10,000 — unused until Monte Carlo VaR is
  implemented), `ASSUMED_PORTFOLIO_VALUE` ($10M, purely for turning a
  percentage VaR into a dollar figure for display),
  `HISTORICAL_SCENARIOS` (2008 GFC, COVID crash, 2022 rate-hike selloff, as
  date ranges), and `HYPOTHETICAL_SCENARIOS` (sector-shock definitions —
  defined but currently unused; see §11).

**Interview framing**: separating "how do I connect" from "what do I trade"
from "how do I measure risk" means a change to the confidence level never
touches database code, and a change to the ticker universe never touches
risk math. Small thing, but it's the kind of separation reviewers notice.

---

## 4. Data pipeline

### `src/data_pipeline/db.py`

The only file that talks raw SQL to Postgres.

- **`get_engine()`** — builds a SQLAlchemy engine from `SupabaseConfig`,
  caches it in a module-level global (so repeated calls within one process
  don't reopen connections), calls `load_dotenv()` first (a no-op if `.env`
  doesn't exist or if the variables are already set — which is exactly what
  happens in GitHub Actions, where the real values arrive as environment
  variables from repository secrets, not a checked-in `.env` file), and
  raises a clear `RuntimeError` naming exactly which env var is missing
  rather than failing deep inside a psycopg2 stack trace.

- **`upsert_dataframe(df, table, pk_cols, engine)`** — the generic write
  path every table goes through. Builds
  `INSERT INTO table (...) VALUES %s ON CONFLICT (pk_cols) DO UPDATE SET ...`
  dynamically from `df.columns`, so no table needs its own hand-written SQL.
  Uses `psycopg2.extras.execute_values` to send each 1,000-row chunk as
  **one** multi-row `INSERT` statement rather than one round-trip per row —
  this matters a lot at scale (see the "lessons learned" section, §12, for
  the real incident this fixes). NaN values are converted to `None` first
  (`df.astype(object).where(df.notna(), None)`) since Postgres numeric
  columns don't accept `NaN` the way pandas does.

- **`read_table(query, engine, params)`** — thin wrapper over
  `pandas.read_sql` with a `sqlalchemy.text()` query, so callers get
  parameterized queries (safe against injection) instead of hand-formatting
  strings.

**Interview framing**: "How would this scale to 10x the tickers?" — the
chunked `execute_values` approach scales roughly linearly; the real
bottleneck would become Supabase's connection/compute tier, not this code.
"Why upsert instead of insert?" — idempotency. The whole pipeline is
designed to be safely re-run (see `fetch_data.py` below) without special
"has this already run today" logic.

### `src/data_pipeline/fetch_data.py`

The ETL: yfinance → star schema. `run()` does, in order:

1. **`upsert_dim_asset()`** — writes the 14-ticker universe into `dim_asset`,
   keyed on `ticker` (which has a `unique` constraint in the schema, so
   `ON CONFLICT (ticker)` works even though `asset_id` is the actual primary
   key). Reads back `{ticker: asset_id}` so downstream steps can map tickers
   to foreign keys.
2. **`upsert_dim_portfolio()`** — same pattern, keyed on `portfolio_name`.
3. **`fetch_prices()`** — `yf.download(..., group_by="ticker", auto_adjust=True)`.
   Two things worth knowing cold for an interview:
   - `auto_adjust=True` means yfinance's `Close` is *already* split- and
     dividend-adjusted — there's no separate "Adj Close" column to reach
     for. Since the schema requires both `close` and `adj_close` as
     non-null columns, this code sets `adj_close = close` deliberately (see
     the docstring — this isn't an oversight).
   - Multi-ticker downloads come back as a wide DataFrame with a MultiIndex
     column (ticker × OHLCV). `raw.stack(level=0, future_stack=True)`
     reshapes that into one row per (date, ticker) — the long format the
     star schema needs.
4. The earliest trading date actually returned by yfinance (**not** the
   `DATA_START_DATE` config constant, which may land on a weekend/holiday)
   is used as `dim_portfolio_weight.effective_date` — this matters because
   `effective_date` is part of that table's primary key, so using the wrong
   date would insert a *second*, slightly-different weight row on every
   rerun instead of updating the existing one.
5. **`build_dim_date()`** — vectorized year/quarter/month/day/day-of-week via
   pandas' `.dt` accessor, upserted into `dim_date`.
6. `fact_price` upsert, keyed on `(date_id, asset_id)`.
7. **`compute_returns()`** — `simple_return = pct_change()`,
   `log_return = diff of log(price)`, both computed **per ticker**
   (`groupby("ticker")`) so day 1 of ticker B never gets compared against the
   last price of ticker A. First row per ticker (no prior day to diff
   against) is dropped.
8. `fact_returns` upsert, same PK pattern.

**Interview framing**: "Why log returns *and* simple returns?" — simple
returns are what you actually realize (compound correctly for P&L); log
returns are additive over time and symmetric, which is why VaR/CVaR
literature and the parametric normal-distribution assumption often prefer
them. This project stores both and lets each consumer pick.

### GitHub Actions (`.github/workflows/daily_pipeline.yml`)

Runs `fetch_data.py` then `run_risk_pipeline.py` on a cron (`30 22 * * 1-5` —
22:30 UTC, weekdays, a buffer after US market close), plus
`workflow_dispatch` for manual runs. Secrets are injected as real environment
variables, which is exactly the path `get_engine()`'s `load_dotenv()`
no-op is designed around.

---

## 5. Risk engine — the actual math

Every function here is pure: DataFrame/Series in, numbers out, zero I/O.
That's what makes `tests/test_var_cvar.py` run in under 5 seconds with no
network or database.

### `src/risk_engine/portfolio.py` — shared primitives

- **`align_weights(weights, returns_df)`** — reindexes the weights Series to
  match the returns DataFrame's column order, and **raises** if any ticker
  is missing a weight or vice versa. This is a deliberate fail-loud choice:
  a missing weight silently defaulting to 0 or NaN would quietly understate
  portfolio risk, which is the worst kind of bug in a risk system —
  wrong, but not obviously wrong.
- **`portfolio_returns(returns_df, weights)`** — `r_p = Σ wᵢrᵢ`, i.e.
  `returns_df.mul(weights, axis=1).sum(axis=1)`.
- **`covariance_matrix(returns_df, annualize)`** — `returns_df.cov()`,
  optionally scaled by 252 (covariance scales *linearly* with time; this is
  a common interview gotcha — **volatility** scales with `√time`, not
  linearly, which is why `portfolio_volatility()` below multiplies by
  `√252`, not `252`, when annualizing).
- **`portfolio_volatility(weights, cov_matrix, annualize)`** —
  `σ_p = √(wᵀΣw)`.

### `src/risk_engine/var_cvar.py` — VaR and CVaR (Expected Shortfall)

Convention used throughout: VaR and CVaR are **positive numbers**
representing a loss fraction (0.023 = a 2.3% loss), not signed returns.

- **`parametric_var_cvar(returns_df, weights, confidence)`** — the
  variance-covariance / Delta-Normal method. Assumes portfolio returns are
  normally distributed.
  - `μ = Σ(mean return of each asset × its weight)`
  - `σ = √(wᵀΣw)` — note this is computed from the **full covariance
    matrix**, not `portfolio_returns(...).std()`. Mathematically these
    should agree for a linear portfolio, but computing it from Σ directly is
    what generalizes to risk decomposition (which asset contributes how
    much variance) if you ever extend this — computing from the aggregate
    series alone would throw that structure away.
  - `z = Φ⁻¹(1 − confidence)` (`norm.ppf`) — e.g. at 95% confidence,
    `z ≈ -1.645`.
  - `VaR = -(μ + zσ)` — for `μ≈0`, this collapses to the number every risk
    textbook quotes: **95% VaR ≈ 1.645σ**, **99% VaR ≈ 2.326σ**. This exact
    identity is what `tests/test_var_cvar.py::test_parametric_matches_known_normal_case`
    checks against a hand-computed value.
  - `CVaR = -(μ − σ·φ(z)/(1−confidence))` — the closed-form expected
    shortfall of a normal distribution (`φ` is the normal PDF). This is a
    standard formula worth being able to write from memory in an interview.

- **`historical_var_cvar(returns_df, weights, confidence)`** — the
  historical simulation method. No distributional assumption — uses the
  empirical distribution of realized portfolio returns directly.
  - `VaR = -percentile(portfolio_returns, (1−confidence)×100)` — e.g. the
    5th percentile of the return series for 95% confidence.
  - `CVaR = -mean(all returns ≤ −VaR)` — the average of the tail *beyond*
    the VaR cutoff, i.e. "given that you're in the bad 5% of days, what's
    the average loss."

- **`monte_carlo_var_cvar()`** — **intentionally not implemented**
  (`raise NotImplementedError`). Scoped out to manage project time; the
  docstring documents this explicitly rather than leaving it silently
  broken. `compute_all_methods()` only calls the historical and parametric
  functions, so the pipeline never touches this stub. See §11 for how you'd
  build it if you extend the project later.

**Interview framing — know this cold**: "Why do CVaR/parametric VaR and
historical VaR usually disagree?" — parametric assumes a normal
distribution (thin tails); real returns are fat-tailed (excess kurtosis).
When historical VaR is meaningfully larger than parametric VaR at the same
confidence, that gap *is* your fat-tail diagnostic — you don't need a
separate kurtosis test to see it, the model disagreement tells you directly.
This project's `notebooks/04_stress_test_writeup.ipynb` computes and
discusses exactly this gap.

### `src/risk_engine/backtesting.py` — model validation

This is the piece that demonstrates *validation* skill, not just
calculation — arguably the most interview-relevant file in the project,
since backtesting/model validation is core day-to-day work for a market
risk analyst.

- **`rolling_backtest(returns_df, weights, method, confidence, window)`** —
  for each day `t` from `window` onward: compute VaR using **only** the
  trailing `window` days *strictly before* `t`
  (`returns_df.iloc[t-window:t]` — pandas `iloc` slicing excludes the right
  endpoint, so this never includes day `t` itself), then check whether day
  `t`'s **realized** return breached that prediction
  (`realized_return < -predicted_var`). This strict no-look-ahead
  construction is the single most important correctness property of a
  backtest — a backtest that lets the model see the day it's predicting
  will always look better calibrated than it really is.

- **`kupiec_pof_test(breach_series, confidence)`** — the Kupiec (1995)
  Proportion-of-Failures test. Tests the null hypothesis that the observed
  breach rate equals the expected rate (`1 − confidence`), via a likelihood
  ratio statistic:
  ```
  LR = -2 × [ log-likelihood(p_expected) − log-likelihood(p_observed) ]
  ```
  where the log-likelihood of a breach rate `p` given `n` observations and
  `x` breaches is `(n−x)·ln(1−p) + x·ln(p)`. Under the null, `LR ~ χ²(1)`,
  so `p_value = 1 − χ²CDF(LR, df=1)`; reject the model's calibration if
  `p_value < 0.05`. The code also classifies *which direction* the model
  failed in — breach rate too high (underestimating risk, the dangerous
  direction) vs. too low (overestimating risk, i.e. too conservative) —
  because those two failure modes have very different real-world
  implications for a bank's capital requirements.

**This project's actual result** (95% confidence, historical method, 5,188
observations after truncating and rebuilding the full pipeline): 293
breaches vs. 259 expected (5.65% vs. 5.00%), `p = 0.0358` → **rejected** at
5% significance, in the "underestimating risk" direction. This is a
genuinely useful, defensible finding, not a bug — see §12 for why a
*marginal* rejection like this is actually more credible than either a
clean pass or a dramatic failure, and why it's consistent with the sample
containing three separate crisis periods (2008, 2020, 2022).

### `src/risk_engine/stress_testing.py` — scenario analysis

- **`historical_scenario_pnl(price_returns_df, weights, start_date, end_date)`**
  — replays actual historical returns over a named date window (e.g.
  2008-09-01 to 2008-11-30) against **current** portfolio weights, and
  computes the cumulative compounded P&L: `(1 + returns).prod() − 1`. This
  answers "if the portfolio I hold *today* had existed during the 2008
  crash, what would it have lost" — a genuinely different question from
  "what did the market do in 2008," because it's weighted by this specific
  portfolio's composition.
- **`stress_vs_var_gap(stressed_loss_pct, var_pct)`** — the whole point of
  stress testing as a discipline: `gap_multiple = stressed_loss / var`.
  This project's actual number: the 2008 GFC scenario produced a loss
  **19.9× the single-day 95% VaR estimate**. That ratio is the sound bite —
  VaR characterizes a *typical* bad day under normal-times statistics;
  stress testing is what tells you sustained crises look nothing like that.
- **`hypothetical_shock_pnl()`** — **does not exist in this file.**
  `HYPOTHETICAL_SCENARIOS` is defined in `risk_config.py` but scoped out of
  this implementation entirely (see §11). Worth knowing so you don't get
  caught off guard if asked to walk through the code live.

---

## 6. Orchestration — `src/risk_engine/run_risk_pipeline.py`

The glue layer. `run()`, in order:

1. **`load_returns_matrix(engine)`** — reads `portfolio_id`, weights (via
   `dim_portfolio_weight` joined to `dim_asset`), and the full
   `fact_returns` history pivoted into a date × ticker matrix
   (`simple_return`), dropping any date with a missing ticker so every row
   the risk engine sees is a complete cross-section.
2. **`compute_current_var_cvar()`** — takes the trailing 250-day window,
   calls `compute_all_methods()` for both 95% and 99% confidence, attaches
   `date_id`/`portfolio_id`/`window_days`, and converts percentage VaR/CVaR
   into dollar terms via `ASSUMED_PORTFOLIO_VALUE`. Upserted into
   `fact_risk_metrics`.
3. **`run_backtest_and_store()`** — runs `rolling_backtest()` at 95%
   confidence, historical method (chosen as the default because it's cheap
   relative to what Monte Carlo would cost once implemented), upserts into
   `fact_var_backtest`, then prints the Kupiec verdict to stdout (this is
   what shows up in the GitHub Actions log on every automated run).
4. **`run_stress_tests_and_store()`** — loops `HISTORICAL_SCENARIOS`,
   computes each, and writes to `fact_stress_test`. Because that table has
   no natural unique key (just a serial PK), this step explicitly
   `DELETE`s this portfolio's prior rows for the scenario names about to be
   re-inserted first — otherwise a daily rerun would accumulate a new
   duplicate row every single day for a historical window (like 2008) whose
   underlying prices never change.

**Interview framing**: "Why upsert some tables but delete-then-insert
others?" — it maps directly to whether the table has a natural composite
key. `fact_risk_metrics` and `fact_var_backtest` do (date + portfolio +
method + confidence); `fact_stress_test` doesn't, so idempotency has to be
enforced explicitly instead of leaning on `ON CONFLICT`.

---

## 7. Dashboard — `dashboard/app.py`

Streamlit, read-only, four tabs. A few implementation details worth knowing:

- **Connection resolution** (`get_engine()`, `@st.cache_resource`): tries
  `st.secrets` first (how Streamlit Community Cloud injects credentials),
  falls back to `.env` via `load_dotenv()` for local dev. `st.secrets`
  raises an exception outright (not just "empty") when no `secrets.toml`
  exists anywhere, which is the normal case locally — the code catches that
  specifically rather than letting it crash the app.
- **Caching**: every data-loading function is `@st.cache_data(ttl=3600)` —
  since the dashboard never computes anything itself, caching query results
  for an hour means repeat views (e.g. switching tabs) don't re-hit
  Supabase at all.
- **Overview tab**: composition table + sector pie chart (`groupby("sector")
  .sum()` on weights) + indexed price performance, log-scaled, rebased to
  100 at the start of history — log scale is the standard choice for
  multi-year price charts so early cheap-stock volatility doesn't visually
  dwarf later high-price moves.
- **VaR/CVaR tab**: grouped bar chart, method × confidence, for the latest
  `date_id` in `fact_risk_metrics`.
- **Backtesting tab**: user picks method/confidence via selectboxes;
  realized returns plotted against the (negated) VaR threshold line, with
  breaches marked as red dots, plus the live Kupiec verdict computed
  on-the-fly for the selected subset (this is the one place the dashboard
  does real computation — `kupiec_pof_test()` is cheap enough to run
  per-request rather than pre-store every possible method/confidence
  combination).
- **Stress Test tab**: headline 2008 GFC scenario pulled out for a
  dedicated comparison chart against the closest available 95% historical
  VaR, plus a table of every stored scenario run.

---

## 8. Tests — `tests/test_var_cvar.py`

Three tests, all against synthetic data (`np.random.default_rng(42)
.multivariate_normal(...)`), zero database dependency:

1. **`test_var_increases_with_confidence`** — 99% VaR ≥ 95% VaR must hold
   for both methods (you're covering more of the tail).
2. **`test_cvar_is_at_least_var`** — CVaR ≥ VaR by definition (the tail
   average can't be better than the tail cutoff).
3. **`test_parametric_matches_known_normal_case`** — single-asset, known
   `μ=0, σ=0.01`, asserts 95% VaR is within `0.001` of the hand-computed
   `1.645 × 0.01`. This is the test that actually validates the formula is
   implemented correctly, not just internally consistent.

**Interview framing**: "Why test invariants (VaR increases with confidence)
instead of just checking against a hardcoded expected number everywhere?"
— invariant tests catch a broader class of bugs and don't need to be
updated every time you touch the input data; the one hand-computed test
exists specifically to catch the case where every method is *consistently*
wrong in the same direction, which invariant tests alone wouldn't catch.

---

## 9. Notebooks

`notebooks/01`–`04` are prototyping and writeup, not production code —
`01_data_exploration` is where the ETL logic was first worked out (and where
you can see the raw yfinance download shape before it became
`fetch_data.py`), `02_var_cvar_prototyping` and `03_backtesting_analysis`
validate the risk_engine functions interactively, and
`04_stress_test_writeup` is the closest thing this project has to a
findings report — portfolio overview, VaR/CVaR summary, backtest results,
the 2008 GFC comparison, and a written limitations section. If an
interviewer wants the "so what did you conclude" version of this project in
prose rather than code, that notebook is it.

One thing to know: the notebooks write to Supabase with plain
`to_sql(if_exists="append")`, not the idempotent upsert pattern the real
pipeline scripts use — that was fine for one-time bootstrapping, but
re-running those specific cells today (now that the pipeline scripts own
those tables) would hit unique-constraint violations. That's expected and
fine — they did their job.

---

## 10. Known limitations (own these in an interview — don't wait to be asked)

- **Normality assumption** in parametric VaR understates tail risk — real
  returns are fat-tailed. The historical-vs-parametric gap (§5) is the
  project's own evidence of this.
- **Historical VaR assumes the future resembles the lookback window** — a
  250-day window that doesn't include a crisis will underestimate risk
  right up until it does; a window that just *finished* including one will
  overestimate risk for a while after. This directly explains the Kupiec
  backtest result.
- **Correlations break down in crises** — the covariance matrix used for
  parametric VaR is estimated from "normal times" data; diversification
  benefits it assumes shrink exactly when you need them most. This is
  *why* stress testing exists as a separate discipline rather than "just
  use a longer VaR window."
- **The 250-day window itself is a convention**, not a law — ~1 trading
  year is standard in practice (and in this project) but arbitrary; a
  shorter window reacts faster to regime changes at the cost of noisier
  estimates, a longer window is smoother but slower to adapt.
- **Equal weighting** is a simplification for demonstration purposes — a
  real portfolio mandate would have position sizing driven by conviction,
  risk budget, or liquidity, not 1/N.

---

## 11. What's explicitly out of scope (and how you'd build it if asked)

Two things are deliberately unimplemented, documented as such rather than
silently missing:

- **Monte Carlo VaR** (`var_cvar.py::monte_carlo_var_cvar`) — the README
  describes the intended approach: simulate `n_simulations` (10,000)
  correlated return paths via Cholesky decomposition of the historical
  covariance matrix (`L = cholesky(Σ)`, then `simulated = μ + L @ z` for
  standard-normal `z`), compute portfolio P&L for each simulated path, then
  take the same percentile/tail-average approach as historical VaR but on
  simulated rather than realized data. It should converge toward the
  parametric VaR for large `n_simulations` on normally-distributed data —
  that convergence is exactly what you'd unit test.
- **Hypothetical sector-shock scenarios** (`HYPOTHETICAL_SCENARIOS` in
  `risk_config.py`, e.g. "Energy Sector Crash −35%") — unlike historical
  replay, these apply a hand-specified shock per sector to *current*
  weights rather than replaying realized returns. You'd sum
  `weight × shock` grouped by sector (with an "OTHER" bucket for
  unaffected sectors, which the config already anticipates) to get a
  portfolio-level P&L, no historical data required at all.

Being able to describe *how* you'd build the missing piece, even though you
didn't, is a strong interview answer in itself — it shows you understand
the scope boundary you drew, not just that you drew one.

---

## 12. Engineering lessons from actually running this in production

Two real incidents happened when the pipeline was rebuilt from scratch
against live Supabase — both are legitimate "tell me about a time you
debugged a production issue" material:

**Row-by-row inserts don't scale.** The first version of
`upsert_dataframe()` used a plain SQLAlchemy `text()` statement with
executemany-style chunking — which, for a raw textual `INSERT` (as opposed
to SQLAlchemy's Core `Insert` construct), does **not** batch under the
hood; it sends one network round-trip per row. For the ~76,000-row
`fact_price`/`fact_returns` writes, that meant tens of thousands of
round-trips to a Supabase host in a different region — functionally stuck.
Fix: `psycopg2.extras.execute_values`, which builds one genuine multi-row
`INSERT` per 1,000-row chunk. Benchmarked improvement: **~10,700 rows/sec**
against the real database, vs. an approach that hadn't finished after many
minutes.

**A killed process can leave a lock behind.** After stopping a stuck run
mid-flight, every subsequent write to `fact_price` started failing with
`psycopg2.errors.QueryCanceled: canceling statement due to statement
timeout` — at first glance, indistinguishable from "still too slow."
Querying `pg_stat_activity` and `pg_locks` directly showed the real cause:
the killed process's database session hadn't cleanly rolled back — it was
sitting `idle in transaction`, still holding a `RowExclusiveLock` on
`fact_price`, blocking every new writer until they timed out waiting for a
lock, not because any single insert was actually slow. Fix:
`pg_terminate_backend(pid)` on the orphaned session. The general lesson —
*"statement timeout" and "lock contention" produce the identical symptom
from the outside; you have to look at `pg_stat_activity` to tell them
apart* — is a genuinely useful thing to say out loud in an interview.

---

## 13. Fast answers to likely interview questions

- **"Walk me through your VaR methodology."** 250-day rolling window, 95%
  and 99% confidence, two independent methods (parametric/Delta-Normal and
  historical simulation) computed in parallel so I can compare them — the
  gap between them is itself a fat-tail diagnostic.
- **"How do you know your VaR model is any good?"** I backtest it — rolling
  out-of-sample predictions with zero look-ahead, checked against realized
  returns, formally validated with the Kupiec Proportion-of-Failures test
  rather than eyeballing a chart.
- **"What did the backtest tell you?"** The 95% historical model was
  rejected at 5% significance (p=0.036) — 5.65% observed breach rate vs.
  5.00% expected, in the "underestimating risk" direction. Marginal, not
  dramatic, and consistent with known laggard behavior of rolling-window
  VaR through the three crisis periods in the sample.
- **"Why also do stress testing if you already have VaR?"** Because VaR is
  a statistical, normal-times estimate — this project's own 2008 GFC replay
  lost 19.9× the single-day VaR estimate. Stress testing answers "what does
  an actual crisis do to *this* portfolio," which no amount of tuning the
  VaR window fixes.
- **"What would you improve first?"** Monte Carlo VaR as the third
  independent method, and a VaR-over-time trend chart in the dashboard —
  the schema already supports it, it's just a matter of accumulating daily
  runs.
- **"What's the hardest bug you hit building this?"** The lock-contention
  incident in §12 — it's a good story because the fix required looking past
  the symptom (a timeout) to the actual mechanism (an orphaned session
  holding a lock), which is a debugging instinct, not a code fix.
