"""
Database connection + upsert helpers for Supabase Postgres.

Raw SQLAlchemy + psycopg2 against the Postgres connection string Supabase
gives you (Project Settings -> Database -> Connection string) -- no
supabase-py client needed, this gives full SQL control.
"""
import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import SupabaseConfig

_engine: Engine | None = None


def get_engine() -> Engine:
    """
    Build (and cache) a SQLAlchemy engine from SupabaseConfig.from_env().
    Raises a clear error if required connection env vars are missing, rather
    than failing deep inside a psycopg2 connection attempt.
    """
    global _engine
    if _engine is not None:
        return _engine

    # No-op if .env doesn't exist, and never overrides vars already set in the
    # environment -- so this is a no-op in GitHub Actions too, where the real
    # values come from repository secrets, not a checked-in .env file.
    load_dotenv()

    cfg = SupabaseConfig.from_env()
    required = {
        "SUPABASE_DB_HOST": cfg.db_host,
        "SUPABASE_DB_USER": cfg.db_user,
        "SUPABASE_DB_PASSWORD": cfg.db_password,
        "SUPABASE_DB_NAME": cfg.db_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in .env (local) or as repository secrets (GitHub Actions)."
        )

    _engine = create_engine(cfg.sqlalchemy_url())
    return _engine


def upsert_dataframe(df: pd.DataFrame, table: str, pk_cols: list[str], engine: Engine = None) -> None:
    """
    Upsert a DataFrame into a Postgres table via INSERT ... ON CONFLICT DO UPDATE,
    built dynamically from df.columns so callers don't hand-write SQL per table.

    Uses psycopg2.extras.execute_values to send each 1000-row chunk as a
    single multi-row INSERT statement (one network round-trip per chunk).
    A plain SQLAlchemy text()-based executemany sends one round-trip PER ROW
    instead -- fine for small tables, but a non-starter for the ~75k-row
    fact_price/fact_returns writes (would be tens of thousands of round-trips
    to a remote Supabase host).
    """
    if df.empty:
        return

    engine = engine or get_engine()

    # NaN isn't valid for most Postgres column types outside float -- normalize
    # to None so it becomes SQL NULL instead of erroring on insert.
    df = df.astype(object).where(df.notna(), None)

    columns = list(df.columns)
    update_cols = [c for c in columns if c not in pk_cols]

    insert_cols_sql = ", ".join(columns)
    conflict_cols_sql = ", ".join(pk_cols)
    conflict_action_sql = (
        "DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        if update_cols else "DO NOTHING"
    )
    sql = (
        f"INSERT INTO {table} ({insert_cols_sql}) VALUES %s "
        f"ON CONFLICT ({conflict_cols_sql}) {conflict_action_sql}"
    )

    records = list(df[columns].itertuples(index=False, name=None))

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            execute_values(cur, sql, records, page_size=1000)
        raw_conn.commit()
    finally:
        raw_conn.close()


def read_table(query: str, engine: Engine = None, params: dict = None) -> pd.DataFrame:
    """Run a SQL query (optionally parameterized) and return the result as a DataFrame."""
    engine = engine or get_engine()
    return pd.read_sql(text(query), engine, params=params)
