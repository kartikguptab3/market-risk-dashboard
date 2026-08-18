-- ============================================================
-- Market Risk Dashboard -- Supabase / Postgres Schema
-- Star schema: dimension tables + fact tables
-- Run this once in the Supabase SQL Editor (or via psql) to set up.
-- ============================================================

-- ------------------------------------------------------------
-- DIMENSION TABLES
-- ------------------------------------------------------------

create table if not exists dim_asset (
    asset_id        serial primary key,
    ticker          text not null unique,
    company_name    text,
    sector          text,
    market_cap_tier text,          -- 'large', 'mid', 'small'
    is_active       boolean default true,
    created_at      timestamptz default now()
);

create table if not exists dim_date (
    date_id         date primary key,   -- the actual calendar date, used as PK directly
    year            int not null,
    quarter         int not null,
    month           int not null,
    day             int not null,
    day_of_week     int not null,       -- 0=Monday .. 6=Sunday
    is_trading_day  boolean default true
);

create table if not exists dim_portfolio (
    portfolio_id    serial primary key,
    portfolio_name  text not null unique,
    description     text,
    created_at      timestamptz default now()
);

-- weights of each asset within a portfolio (allows multiple portfolios/mandates later)
create table if not exists dim_portfolio_weight (
    portfolio_id    int references dim_portfolio(portfolio_id),
    asset_id        int references dim_asset(asset_id),
    weight          numeric(6,5) not null,   -- e.g. 0.07692 for equal-weight of 13
    effective_date  date not null,
    primary key (portfolio_id, asset_id, effective_date)
);

-- ------------------------------------------------------------
-- FACT TABLES
-- ------------------------------------------------------------

create table if not exists fact_price (
    date_id         date references dim_date(date_id),
    asset_id        int references dim_asset(asset_id),
    open            numeric(14,4),
    high            numeric(14,4),
    low             numeric(14,4),
    close           numeric(14,4) not null,
    adj_close       numeric(14,4) not null,
    volume          bigint,
    primary key (date_id, asset_id)
);

create table if not exists fact_returns (
    date_id         date references dim_date(date_id),
    asset_id        int references dim_asset(asset_id),
    simple_return   numeric(10,6),
    log_return      numeric(10,6),
    primary key (date_id, asset_id)
);

-- pre-computed portfolio-level risk metrics, one row per (date, portfolio, method, confidence)
create table if not exists fact_risk_metrics (
    date_id         date references dim_date(date_id),
    portfolio_id    int references dim_portfolio(portfolio_id),
    method          text not null,      -- 'parametric', 'historical', 'monte_carlo'
    confidence      numeric(4,3) not null,  -- 0.95 or 0.99
    var_pct         numeric(10,6) not null, -- VaR as a fraction of portfolio value (e.g. 0.021 = 2.1%)
    var_value       numeric(14,2),          -- VaR in currency terms, given an assumed portfolio value
    cvar_pct        numeric(10,6) not null,
    cvar_value      numeric(14,2),
    window_days     int not null default 250,
    computed_at     timestamptz default now(),
    primary key (date_id, portfolio_id, method, confidence)
);

-- backtesting exceedance log: did realized portfolio loss breach the prior day's VaR?
create table if not exists fact_var_backtest (
    date_id             date references dim_date(date_id),
    portfolio_id        int references dim_portfolio(portfolio_id),
    method              text not null,
    confidence          numeric(4,3) not null,
    predicted_var_pct   numeric(10,6) not null,
    realized_return_pct numeric(10,6) not null,
    breach              boolean not null,
    primary key (date_id, portfolio_id, method, confidence)
);

-- stress test results: scenario-based shocks applied to current portfolio weights
create table if not exists fact_stress_test (
    scenario_id     serial primary key,
    portfolio_id    int references dim_portfolio(portfolio_id),
    scenario_name   text not null,        -- e.g. '2008 GFC', 'COVID Crash', 'Rate Shock +200bps'
    scenario_type   text not null,        -- 'historical' or 'hypothetical'
    start_date      date,
    end_date        date,
    portfolio_pnl_pct numeric(10,6) not null,
    portfolio_pnl_value numeric(14,2),
    computed_at     timestamptz default now()
);

-- ------------------------------------------------------------
-- INDEXES for dashboard query performance
-- ------------------------------------------------------------
create index if not exists idx_fact_price_date on fact_price(date_id);
create index if not exists idx_fact_returns_date on fact_returns(date_id);
create index if not exists idx_fact_risk_metrics_date on fact_risk_metrics(date_id);
create index if not exists idx_fact_risk_metrics_portfolio on fact_risk_metrics(portfolio_id);
