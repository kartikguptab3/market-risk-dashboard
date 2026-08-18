"""
Central configuration for the Supabase/Postgres connection.
All secrets are read from environment variables (.env locally, GitHub Secrets in CI).
"""
import os
from dataclasses import dataclass


@dataclass
class SupabaseConfig:
    url: str
    key: str
    db_host: str
    db_port: str
    db_name: str
    db_user: str
    db_password: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        return cls(
            url=os.environ.get("SUPABASE_URL", ""),
            key=os.environ.get("SUPABASE_KEY", ""),
            db_host=os.environ.get("SUPABASE_DB_HOST", ""),
            db_port=os.environ.get("SUPABASE_DB_PORT", "5432"),
            db_name=os.environ.get("SUPABASE_DB_NAME", "postgres"),
            db_user=os.environ.get("SUPABASE_DB_USER", "postgres"),
            db_password=os.environ.get("SUPABASE_DB_PASSWORD", ""),
        )

    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
