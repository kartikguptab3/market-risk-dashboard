"""
Central configuration for the Postgres connection (hosted on Neon).
All secrets are read from environment variables (.env locally, GitHub Secrets
in CI, st.secrets on Streamlit Cloud).

Neon hands you a single connection string, so that's the only thing we read:
    DATABASE_URL=postgresql://user:pass@ep-xxxx-pooler.<region>.aws.neon.tech/neondb?sslmode=require
"""
import os
from dataclasses import dataclass

import sqlalchemy
from sqlalchemy.engine import Engine


@dataclass
class DatabaseConfig:
    url: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(url=os.environ.get("DATABASE_URL", "").strip())

    def sqlalchemy_url(self) -> str:
        # Neon gives "postgresql://..." (some tools emit "postgres://");
        # SQLAlchemy needs the driver spelled out to use psycopg2.
        url = self.url
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg2://" + url[len(prefix):]
        return url

    def create_engine(self) -> Engine:
        if not self.url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add your Neon connection string to .env "
                "(local), repository secrets (GitHub Actions) or st.secrets (Streamlit Cloud)."
            )
        return sqlalchemy.create_engine(
            self.sqlalchemy_url(),
            # Neon scales compute to zero after ~5 min idle, which drops open
            # connections -- pre_ping swaps a dead pooled connection for a fresh
            # one instead of erroring on the first query after a pause.
            pool_pre_ping=True,
            pool_recycle=300,
        )
