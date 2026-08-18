"""
ETL: pulls daily prices from yfinance, transforms into the star schema,
and upserts into Supabase.

Idempotent -- safe to re-run daily (this is what the GitHub Actions cron
job calls). Every write goes through db.upsert_dataframe(), keyed on each
table's real primary key, so re-running with overlapping dates just
updates existing rows instead of duplicating them.

Run standalone:
    python -m src.data_pipeline.fetch_data
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))  # src/ on sys.path

import numpy as np
import pandas as pd
import yfinance as yf

from universe import TICKERS, PORTFOLIO_NAME, PORTFOLIO_DESCRIPTION
from risk_config import DATA_START_DATE
from data_pipeline.db import get_engine, upsert_dataframe, read_table


def upsert_dim_asset(engine) -> dict[str, int]:
    dim_asset_df = pd.DataFrame([
        {"ticker": t, "company_name": v["name"], "sector": v["sector"], "market_cap_tier": v["cap"]}
        for t, v in TICKERS.items()
    ])
    upsert_dataframe(dim_asset_df, "dim_asset", pk_cols=["ticker"], engine=engine)

    asset_lookup = read_table("SELECT asset_id, ticker FROM dim_asset", engine=engine)
    return dict(zip(asset_lookup["ticker"], asset_lookup["asset_id"]))


def upsert_dim_portfolio(engine) -> int:
    dim_portfolio_df = pd.DataFrame([{
        "portfolio_name": PORTFOLIO_NAME,
        "description": PORTFOLIO_DESCRIPTION,
    }])
    upsert_dataframe(dim_portfolio_df, "dim_portfolio", pk_cols=["portfolio_name"], engine=engine)

    result = read_table(
        "SELECT portfolio_id FROM dim_portfolio WHERE portfolio_name = :name",
        engine=engine, params={"name": PORTFOLIO_NAME},
    )
    return int(result["portfolio_id"].iloc[0])


def upsert_dim_portfolio_weights(engine, portfolio_id: int, asset_map: dict[str, int], effective_date) -> None:
    equal_weight = round(1 / len(TICKERS), 5)
    weights_df = pd.DataFrame([
        {
            "portfolio_id": portfolio_id,
            "asset_id": asset_map[ticker],
            "weight": equal_weight,
            "effective_date": effective_date,
        }
        for ticker in TICKERS
    ])
    upsert_dataframe(
        weights_df, "dim_portfolio_weight",
        pk_cols=["portfolio_id", "asset_id", "effective_date"], engine=engine,
    )


def build_dim_date(dates) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    return pd.DataFrame({
        "date_id": dates.date,
        "year": dates.year,
        "quarter": dates.quarter,
        "month": dates.month,
        "day": dates.day,
        "day_of_week": dates.dayofweek,
        "is_trading_day": True,
    })


def fetch_prices() -> pd.DataFrame:
    """
    Downloads OHLCV for all TICKERS from DATA_START_DATE to today.

    auto_adjust=True means yfinance's "Close" is already split/dividend
    adjusted; adj_close is set equal to it (matches the notebook-validated
    data already in fact_price, and both columns are required by schema.sql).
    """
    tickers = list(TICKERS.keys())
    raw = yf.download(
        tickers=tickers, start=DATA_START_DATE, group_by="ticker",
        auto_adjust=True, threads=True,
    )

    long_df = raw.stack(level=0, future_stack=True)
    long_df.index.names = ["date", "ticker"]
    long_df = long_df.reset_index()

    long_df = pd.DataFrame({
        "date": long_df["date"],
        "ticker": long_df["ticker"],
        "open": long_df["Open"],
        "high": long_df["High"],
        "low": long_df["Low"],
        "close": long_df["Close"],
        "adj_close": long_df["Close"],
        "volume": long_df["Volume"],
    })

    long_df = long_df.dropna(subset=["close", "adj_close"])
    return long_df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.sort_values(["ticker", "date"]).copy()
    df["simple_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df["log_return"] = df.groupby("ticker")["adj_close"].transform(lambda x: np.log(x / x.shift(1)))
    return df.dropna(subset=["simple_return", "log_return"])[["date", "ticker", "simple_return", "log_return"]]


def run() -> None:
    engine = get_engine()

    asset_map = upsert_dim_asset(engine)
    portfolio_id = upsert_dim_portfolio(engine)

    price_df = fetch_prices()

    # effective_date is part of dim_portfolio_weight's primary key -- use the
    # actual earliest trading day returned by yfinance (not the DATA_START_DATE
    # config constant, which may land on a non-trading day) so reruns update
    # the existing weight row instead of inserting a duplicate under a
    # slightly different date.
    earliest_trading_date = price_df["date"].min().date()
    upsert_dim_portfolio_weights(engine, portfolio_id, asset_map, earliest_trading_date)

    dim_date_df = build_dim_date(price_df["date"])
    upsert_dataframe(dim_date_df, "dim_date", pk_cols=["date_id"], engine=engine)

    fact_price_df = price_df.copy()
    fact_price_df["date_id"] = pd.to_datetime(fact_price_df["date"]).dt.date
    fact_price_df["asset_id"] = fact_price_df["ticker"].map(asset_map)
    fact_price_df = fact_price_df[["date_id", "asset_id", "open", "high", "low", "close", "adj_close", "volume"]]
    upsert_dataframe(fact_price_df, "fact_price", pk_cols=["date_id", "asset_id"], engine=engine)

    returns_df = compute_returns(price_df)
    fact_returns_df = returns_df.copy()
    fact_returns_df["date_id"] = pd.to_datetime(fact_returns_df["date"]).dt.date
    fact_returns_df["asset_id"] = fact_returns_df["ticker"].map(asset_map)
    fact_returns_df = fact_returns_df[["date_id", "asset_id", "simple_return", "log_return"]]
    upsert_dataframe(fact_returns_df, "fact_returns", pk_cols=["date_id", "asset_id"], engine=engine)

    print(
        f"Upserted {len(fact_price_df)} price rows and {len(fact_returns_df)} "
        f"return rows across {len(asset_map)} tickers."
    )


if __name__ == "__main__":
    run()
