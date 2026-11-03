"""Configuration: API keys from `.env`, paths, and API endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DB_API_BASE = "https://db.satnogs.org/api"
DB_HOST = "db.satnogs.org"
NETWORK_API_BASE = "https://network.satnogs.org/api"
NETWORK_HOST = "network.satnogs.org"
USER_AGENT = "satnogs-tlm-agg/0.1 (+https://github.com/parker-research/satnogs-tlm-agg)"


@dataclass(frozen=True)
class Settings:
    """Runtime settings. API keys come from the environment / `.env` file."""

    db_api_key: str | None
    network_api_key: str | None
    data_dir: Path
    out_dir: Path

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "satnogs.duckdb"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def load_settings(env_file: Path, data_dir: Path, out_dir: Path) -> Settings:
    """Load settings, reading `SATNOGS_DB_API_KEY` / `SATNOGS_NETWORK_API_KEY` from `.env`."""
    load_dotenv(env_file, override=False)

    def _key(name: str) -> str | None:
        value = os.environ.get(name, "").strip()
        # Treat the placeholder values from `.env.sample` as "unset".
        if not value or set(value) <= {"x"}:
            return None
        return value

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        db_api_key=_key("SATNOGS_DB_API_KEY"),
        network_api_key=_key("SATNOGS_NETWORK_API_KEY"),
        data_dir=data_dir,
        out_dir=out_dir,
    )
