"""Resumable per-satellite fetch state (DuckDB `fetch_progress`; see the schema comment)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import duckdb

from .store import WRITE_LOCK
from .timeutil import now


@dataclass
class Progress:
    kind: str
    sat_id: str
    covered_until: datetime | None = None
    window_end: datetime | None = None
    window_ceiling: datetime | None = None
    window_oldest: datetime | None = None
    window_new: int = 0
    next_url: str | None = None
    next_check_at: datetime | None = None
    check_interval_h: float | None = None
    final: bool = False
    oldest_ts: datetime | None = None
    newest_ts: datetime | None = None
    n_records: int = 0
    n_requests: int = 0
    last_error: str | None = None

    @property
    def in_window(self) -> bool:
        return self.window_ceiling is not None

    @property
    def backfill_done(self) -> bool:
        return self.covered_until is not None


_COLS = [f.name for f in fields(Progress)]


def load(con: duckdb.DuckDBPyConnection, kind: str, sat_id: str) -> Progress:
    rows = con.execute(
        f"SELECT {', '.join(_COLS)} FROM fetch_progress WHERE kind = ? AND sat_id = ?", [kind, sat_id]
    ).fetchall()
    if not rows:
        return Progress(kind, sat_id)
    p = Progress(**dict(zip(_COLS, rows[0], strict=True)))
    p.window_new = int(p.window_new or 0)
    p.n_records = int(p.n_records or 0)
    p.n_requests = int(p.n_requests or 0)
    p.final = bool(p.final)
    return p


def save(con: duckdb.DuckDBPyConnection, p: Progress) -> None:
    cols = [*_COLS, "backfill_done", "last_run", "layout_version"]
    values = [getattr(p, c) for c in _COLS] + [p.backfill_done, now(), 2]
    with WRITE_LOCK:
        con.execute(
            f"INSERT OR REPLACE INTO fetch_progress ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            values,
        )
