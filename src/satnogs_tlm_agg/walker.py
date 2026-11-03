"""Resumable, request-minimising walk over a satellite's time-ordered SatNOGS records.

A satellite's history is fetched as consecutive time windows `(covered_until, ceiling]`:

1. First window: `(-inf, now - settle]`, i.e. the full history, newest-first.
2. Later windows: `(covered_until, now - settle]`, only what arrived since.

Inside a window the server's cursor (`next` link) is followed and saved after every page, so a
restart continues with the exact next page. Each page's raw body is archived *before* it is
parsed or stored. `settle` keeps the newest few hours out of every window so late uploads land
in the following window instead of needing an overlapping re-fetch.

Per call, at most `max_pages` requests are made (round-robin fairness across satellites).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import duckdb
from loguru import logger

from . import progress
from .archive import RawArchive
from .http import ApiError, RateLimited, SatnogsClient
from .timeutil import now

Record = dict[str, Any]
# (covered_until, window_end) -> query params selecting that window
WindowParams = Callable[[datetime | None, datetime], dict[str, Any]]


@dataclass
class Schedule:
    """How often to re-check a satellite once its current window is complete."""

    settle: timedelta = timedelta(hours=6)
    base_interval: timedelta = timedelta(days=1)
    max_interval: timedelta = timedelta(days=30)
    # A re-entered satellite is final once coverage extends this far past its decay date.
    final_after_decay: timedelta = timedelta(days=14)
    # Minimum gap between calls while a window is in progress (e.g. 1/day violator throttle).
    min_gap_in_window: timedelta | None = None


@dataclass
class WalkResult:
    new_records: int = 0
    requests: int = 0
    window_complete: bool = False


def _reschedule_after_window(p: progress.Progress, sched: Schedule, decayed: datetime | None) -> None:
    if (
        decayed is not None
        and p.covered_until is not None
        and p.covered_until >= decayed + sched.final_after_decay
    ):
        p.final = True
        p.next_check_at = None
        return
    base_h = sched.base_interval.total_seconds() / 3600
    if p.window_new > 0 or p.check_interval_h is None:
        interval_h = base_h
    else:
        interval_h = min(p.check_interval_h * 2, sched.max_interval.total_seconds() / 3600)
    p.check_interval_h = interval_h
    p.next_check_at = now() + timedelta(hours=interval_h)


def walk(
    client: SatnogsClient,
    con: duckdb.DuckDBPyConnection,
    archive: RawArchive,
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
    decayed: datetime | None = None,
) -> WalkResult:
    p = progress.load(con, kind, sat_id)
    res = WalkResult()
    if p.final:
        return res

    if not p.in_window:
        ceiling = now() - schedule.settle
        if p.covered_until is not None and ceiling <= p.covered_until:
            return res
        p.window_end = p.window_ceiling = ceiling
        p.window_oldest = None
        p.window_new = 0
        p.next_url = None
    p.next_check_at = None

    try:
        while res.requests < max_pages:
            if p.next_url:
                req_url, params = p.next_url, None
            else:
                req_url, params = url, {**base_params, **window_params(p.covered_until, p.window_end)}
            try:
                page = client.get_page(req_url, params)
            except ApiError as exc:
                if p.next_url and exc.status in (400, 404) and not isinstance(exc, RateLimited):
                    # Cursor no longer valid: re-query the rest of the window by time instead.
                    logger.warning("{} {}: stale cursor, resuming window by time", kind, label)
                    p.next_url = None
                    if p.window_oldest is not None:
                        p.window_end = p.window_oldest
                    continue
                raise
            res.requests += 1
            p.n_requests += 1
            # Archive first: from here on this page never has to be requested again.
            archive.write(kind, {"sat_id": sat_id, "url": page.url, "next": page.next_url, "body": page.body})
            n_new = ingest(page.records)
            res.new_records += n_new
            p.n_records += n_new
            p.window_new += n_new
            tss = [t for r in page.records if (t := ts_of(r)) is not None]
            if tss:
                lo, hi = min(tss), max(tss)
                p.window_oldest = lo if p.window_oldest is None else min(p.window_oldest, lo)
                p.oldest_ts = lo if p.oldest_ts is None else min(p.oldest_ts, lo)
                p.newest_ts = hi if p.newest_ts is None else max(p.newest_ts, hi)
            p.next_url = page.next_url
            p.last_error = None
            if not page.next_url:
                p.covered_until = p.window_ceiling
                p.window_end = p.window_ceiling = p.window_oldest = None
                res.window_complete = True
                _reschedule_after_window(p, schedule, decayed)
                p.window_new = 0
                break
            progress.save(con, p)
        if not res.window_complete and schedule.min_gap_in_window is not None:
            p.next_check_at = now() + schedule.min_gap_in_window
    except RateLimited as exc:
        p.next_check_at = now() + timedelta(seconds=exc.retry_after)
        p.last_error = f"rate limited for {exc.retry_after:.0f}s"
        logger.info("{} {}: rate limited, next try in {:.1f}h", kind, label, exc.retry_after / 3600)
    except ApiError as exc:
        p.last_error = str(exc)[:1000]
        logger.error("{} {}: {}", kind, label, exc)
        if exc.status in (401, 403):
            progress.save(con, p)
            raise
        # Unexpected client error for this satellite: retry after a day, not every cycle.
        p.next_check_at = now() + schedule.base_interval
    progress.save(con, p)
    if res.new_records or res.window_complete:
        logger.info(
            "{} {}: +{} new in {} req{} (total {})",
            kind,
            label,
            res.new_records,
            res.requests,
            ", window complete" if res.window_complete else "",
            p.n_records,
        )
    return res
