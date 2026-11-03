"""Resumable, request-minimising fetch of a satellite's records, tracked per day in DuckDB.

Order in which a satellite's mission (per source `kind`) is fetched:

1. **Probe**: one request over the whole history. Satellites with no data are done after it;
   otherwise it returns the newest page (for a re-entered satellite: its last frames).
2. **End of life**: the `eol_days` before decay (or before the last record, if still in
   orbit), newest first.
3. **Random chunks** of the rest of the mission. A chunk is 1-30 days, sized from the observed
   record rate so it costs about one page.
4. **Caught up**: new days are fetched once a chunk's worth has accumulated. Re-entered
   satellites stop `final_after_decay` after decay.

Each chunk is walked newest-first with the server's cursor, saved after every page
(`fetch_state.next_url`), so a restart continues exactly there. Every day a walk fully passes is
written to `fetch_days`; the remaining days are always `expected days - fetch_days`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
from loguru import logger

from .http import ApiError, RateLimited, SatnogsClient
from .store import WRITE_LOCK
from .timeutil import now

Record = dict[str, Any]
# (lower bound or None, upper bound, upper inclusive?) -> query params selecting that range
WindowParams = Callable[[datetime | None, datetime, bool], dict[str, Any]]

# Days before this (or before launch) aren't expected to hold SatNOGS data.
HISTORY_START = date(2013, 1, 1)
PAGE_SIZE = 25  # fixed server-side for both SatNOGS DB telemetry and Network observations

# Timestamps of the stored records of each kind, used to count records per day.
_RECORD_TIMES = {
    "db_telemetry": "SELECT timestamp AS t FROM frames WHERE source = 'db' AND sat_id = $sat",
    "network_obs": "SELECT start AS t FROM network_observations WHERE sat_id = $sat",
}

_rng = random.Random()


@dataclass
class Schedule:
    settle: timedelta = timedelta(hours=6)
    """A day is fetched only once it ended at least this long ago (late uploads)."""
    final_after_decay: timedelta = timedelta(days=14)
    """Re-entered satellites are expected to have data until this long after decay."""
    eol_days: int = 60
    """Days before decay / last record that are fetched first."""
    max_chunk_days: int = 30
    min_gap_after_turn: timedelta | None = None
    """Wait between turns (e.g. the 1/day 'frequency violator' throttle)."""


@dataclass
class State:
    kind: str
    sat_id: str
    probed: bool = False
    window_start: date | None = None  # NULL while probing (whole history)
    window_end: date | None = None  # exclusive; NULL = no walk in progress
    next_url: str | None = None
    oldest_fetched: datetime | None = None
    n_requests: int = 0
    not_before: datetime | None = None
    last_error: str | None = None


_STATE_COLS = [f.name for f in fields(State)]


def load_state(con: duckdb.DuckDBPyConnection, kind: str, sat_id: str) -> State:
    rows = con.execute(
        f"SELECT {', '.join(_STATE_COLS)} FROM fetch_state WHERE kind = ? AND sat_id = ?", [kind, sat_id]
    ).fetchall()
    if not rows:
        return State(kind, sat_id)
    st = State(**dict(zip(_STATE_COLS, rows[0], strict=True)))
    st.probed, st.n_requests = bool(st.probed), int(st.n_requests or 0)
    return st


def save_state(con: duckdb.DuckDBPyConnection, st: State) -> None:
    cols = [*_STATE_COLS, "updated_at"]
    with WRITE_LOCK:
        con.execute(
            f"INSERT OR REPLACE INTO fetch_state ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [getattr(st, c) for c in _STATE_COLS] + [now()],
        )


def expected_range(launched: datetime | None, decayed: datetime | None, sched: Schedule) -> tuple[date, date]:
    """Days [start, end) that should be fetched for a satellite (must match `fetch_coverage`)."""
    start = max(launched.date(), HISTORY_START) if launched is not None else HISTORY_START
    end = (now() - sched.settle).date()
    if decayed is not None:
        end = min(end, decayed.date() + sched.final_after_decay + timedelta(days=1))
    return start, end


def mark_days(con: duckdb.DuckDBPyConnection, kind: str, sat_id: str, first: date, last: date) -> None:
    """Record days [first, last] as completely fetched, with their record counts."""
    if first > last:
        return
    with WRITE_LOCK:
        con.execute(
            f"""
            INSERT OR REPLACE INTO fetch_days
            SELECT $kind, $sat, g.day::DATE, coalesce(c.n, 0), now()
            FROM (SELECT unnest(generate_series($first::TIMESTAMP, $last::TIMESTAMP, INTERVAL 1 DAY)) AS day) g
            LEFT JOIN (
                SELECT t::DATE AS day, count(*) AS n FROM ({_RECORD_TIMES[kind]})
                WHERE t >= $first::TIMESTAMPTZ AND t < ($last + 1)::TIMESTAMPTZ
                GROUP BY 1
            ) c ON c.day = g.day::DATE
            """,
            {"kind": kind, "sat": sat_id, "first": first, "last": last},
        )


def remaining_days(
    con: duckdb.DuckDBPyConnection, kind: str, sat_id: str, start: date, end: date
) -> list[date]:
    if start >= end:
        return []
    rows = con.execute(
        """
        SELECT g.day::DATE FROM (
            SELECT unnest(generate_series($start::TIMESTAMP, $last::TIMESTAMP, INTERVAL 1 DAY)) AS day
        ) g
        ANTI JOIN (SELECT day FROM fetch_days WHERE kind = $kind AND sat_id = $sat) f ON f.day = g.day::DATE
        ORDER BY 1
        """,
        {"start": start, "last": end - timedelta(days=1), "kind": kind, "sat": sat_id},
    ).fetchall()
    return [r[0] for r in rows]


def _chunk_days(con: duckdb.DuckDBPyConnection, kind: str, sat_id: str, max_days: int) -> int:
    """Chunk length (days) expected to hold about one page of records."""
    n_days, n_records = con.execute(
        "SELECT count(*), coalesce(sum(n_records), 0) FROM fetch_days WHERE kind = ? AND sat_id = ?",
        [kind, sat_id],
    ).fetchall()[0]
    if n_days == 0:
        return 1
    rate = n_records / n_days
    return max(1, min(max_days, math.floor(PAGE_SIZE / rate) if rate > 0 else max_days))


def _eol_anchor(
    con: duckdb.DuckDBPyConnection, kind: str, sat_id: str, decayed: datetime | None
) -> date | None:
    if decayed is not None:
        return decayed.date()
    last = con.execute(f"SELECT max(t) FROM ({_RECORD_TIMES[kind]})", {"sat": sat_id}).fetchall()[0][0]
    return last.date() if last is not None else None


def plan_chunk(
    remaining: list[date],
    *,
    end: date,
    fixed_end: bool,
    n: int,
    eol_anchor: date | None,
    eol_days: int,
    rng: random.Random = _rng,
) -> tuple[date, date] | None:
    """Pick the next chunk [start, end) of remaining days, or None if nothing should be fetched."""
    if not remaining:
        return None
    rem = set(remaining)
    # Caught up on a live satellite: wait until a whole chunk of new days has accumulated.
    if not fixed_end and remaining[0] >= end - timedelta(days=n) and len(remaining) < n:
        return None
    eol = [d for d in remaining if eol_anchor and eol_anchor - timedelta(days=eol_days) <= d <= eol_anchor]
    if eol:  # newest remaining end-of-life day, extended backwards
        last = eol[-1]
        first = last
        while (first - timedelta(days=1)) in rem and (last - first).days + 1 < n:
            first -= timedelta(days=1)
    else:  # random remaining day, extended forwards
        first = rng.choice(remaining)
        last = first
        while (last + timedelta(days=1)) in rem and (last - first).days + 1 < n:
            last += timedelta(days=1)
    return first, last + timedelta(days=1)


def _day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


@dataclass
class WalkResult:
    new_records: int = 0
    requests: int = 0


def walk(
    client: SatnogsClient,
    con: duckdb.DuckDBPyConnection,
    *,
    kind: str,
    sat_id: str,
    label: str,
    url: str,
    base_params: dict[str, Any],
    window_params: WindowParams,
    ts_of: Callable[[Record], datetime | None],
    ingest: Callable[[list[Record]], int],
    max_pages: int,
    schedule: Schedule,
    launched: datetime | None = None,
    decayed: datetime | None = None,
) -> WalkResult:
    """One fetch turn (at most `max_pages` requests) for one satellite."""
    st = load_state(con, kind, sat_id)
    res = WalkResult()
    start, end = expected_range(launched, decayed, schedule)

    try:
        while res.requests < max_pages:
            if st.window_end is None:  # choose what to fetch next
                if not st.probed:
                    if end <= start:
                        break
                    st.window_start, st.window_end = None, end
                else:
                    chunk = plan_chunk(
                        remaining_days(con, kind, sat_id, start, end),
                        end=end,
                        fixed_end=decayed is not None,
                        n=_chunk_days(con, kind, sat_id, schedule.max_chunk_days),
                        eol_anchor=_eol_anchor(con, kind, sat_id, decayed),
                        eol_days=schedule.eol_days,
                    )
                    if chunk is None:
                        break
                    st.window_start, st.window_end = chunk
                st.next_url, st.oldest_fetched = None, None

            if st.next_url:
                req_url, params = st.next_url, None
            else:
                lower = _day_start(st.window_start) if st.window_start else None
                if st.oldest_fetched is not None:  # resuming by time after a stale cursor
                    params = window_params(lower, st.oldest_fetched, True)
                else:
                    params = window_params(lower, _day_start(st.window_end), False)
                req_url, params = url, {**base_params, **params}
            try:
                page = client.get_page(req_url, params)
            except ApiError as exc:
                if st.next_url and exc.status in (400, 404) and not isinstance(exc, RateLimited):
                    logger.warning("{} {}: stale cursor, resuming by time", kind, label)
                    st.next_url = None
                    continue
                raise
            res.requests += 1
            st.n_requests += 1
            res.new_records += ingest(page.records)

            prev_oldest = st.oldest_fetched
            tss = [t for r in page.records if (t := ts_of(r)) is not None]
            if tss:
                st.oldest_fetched = min(tss) if prev_oldest is None else min(prev_oldest, min(tss))
            hi = prev_oldest.date() if prev_oldest else st.window_end - timedelta(days=1)
            probing = st.window_start is None

            if not page.next_url:  # every day of the window is stored
                first = st.window_start
                if probing:
                    first = min(start, st.oldest_fetched.date()) if st.oldest_fetched else start
                mark_days(con, kind, sat_id, first, hi)
                st.window_start = st.window_end = st.next_url = st.oldest_fetched = None
                st.probed = True
            else:
                # Days strictly after the day of `oldest_fetched` are complete (newest-first).
                if st.oldest_fetched is not None:
                    mark_days(con, kind, sat_id, st.oldest_fetched.date() + timedelta(days=1), hi)
                if probing:  # the probe only ever takes one page
                    st.window_start = st.window_end = st.next_url = st.oldest_fetched = None
                    st.probed = True
                else:
                    st.next_url = page.next_url
            st.last_error = None
            save_state(con, st)
        if res.requests and schedule.min_gap_after_turn:
            st.not_before = now() + schedule.min_gap_after_turn
        else:
            st.not_before = None
    except RateLimited as exc:
        st.not_before = now() + timedelta(seconds=exc.retry_after)
        st.last_error = f"rate limited for {exc.retry_after:.0f}s"
        logger.info("{} {}: rate limited, next try in {:.1f}h", kind, label, exc.retry_after / 3600)
    except ApiError as exc:
        st.last_error = str(exc)[:1000]
        logger.error("{} {}: {}", kind, label, exc)
        if exc.status in (401, 403):
            save_state(con, st)
            raise
        st.not_before = now() + timedelta(days=1)  # unexpected client error: retry tomorrow
    if res.requests or st.last_error:
        save_state(con, st)
    if res.new_records:
        logger.info("{} {}: +{} new in {} req", kind, label, res.new_records, res.requests)
    return res
