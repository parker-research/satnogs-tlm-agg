"""Re-ingest everything from the raw archive into DuckDB, without any API requests."""

from __future__ import annotations

from typing import Any

import duckdb
import polars as pl
from loguru import logger

from .archive import RawArchive
from .satnogs_db import (
    FRAME_SCHEMA,
    ingest_satellites,
    ingest_telemetry_page,
    ingest_transmitters,
    insert_frames,
)
from .satnogs_network import _OBS_META_COLS, demod_frame_row, ingest_observations_page, record_demod_file
from .store import WRITE_LOCK


def _records(body: Any) -> list[dict[str, Any]]:
    if isinstance(body, dict):
        return body.get("results") or []
    return body or []


def rebuild(con: duckdb.DuckDBPyConnection, archive: RawArchive) -> None:
    last: dict[str, Any] = {}
    for kind in ("db_satellites", "db_transmitters"):
        for rec in archive.iter(kind):
            last[kind] = rec["body"]
    if "db_satellites" in last:
        ingest_satellites(con, last["db_satellites"])
    if "db_transmitters" in last:
        ingest_transmitters(con, last["db_transmitters"])

    for kind, ingest in (("db_telemetry", ingest_telemetry_page), ("network_obs", ingest_observations_page)):
        pages = new = 0
        for rec in archive.iter(kind):
            pages += 1
            new += ingest(con, _records(rec["body"]))
        logger.info("rebuild {}: {} archived pages, {} new records", kind, pages, new)

    rows: list[dict[str, Any]] = []
    meta_cache: dict[int, tuple | None] = {}
    files = 0
    for rec in archive.iter("network_demod"):
        files += 1
        obs_id = rec["obs_id"]
        if obs_id not in meta_cache:
            found = con.execute(
                f"SELECT {_OBS_META_COLS} FROM network_observations WHERE id = ?", [obs_id]
            ).fetchall()
            meta_cache[obs_id] = found[0] if found else None
        meta = meta_cache[obs_id]
        if meta is None:
            continue
        content = bytes.fromhex(rec["content_hex"])
        r = demod_frame_row(meta, rec["url"], content)
        if r is not None:
            rows.append(r)
        record_demod_file(con, rec["url"], obs_id, 200, len(content), r["frame_id"] if r else None)
        if len(rows) >= 10_000:
            insert_frames(con, pl.DataFrame(rows, schema=FRAME_SCHEMA))
            rows.clear()
    insert_frames(con, pl.DataFrame(rows, schema=FRAME_SCHEMA))
    # An observation is complete only when all of its frame-file URLs are recorded.
    with WRITE_LOCK:
        con.execute(
            """
            UPDATE network_observations o SET demod_fetched = TRUE
            WHERE NOT demod_fetched AND len(demod_urls) > 0
              AND NOT EXISTS (
                  SELECT 1 FROM (SELECT unnest(o.demod_urls) AS url) u
                  WHERE u.url NOT IN (SELECT url FROM demod_files)
                    AND NOT regexp_matches(lower(u.url), '\\.(png|jpe?g|wav|ogg)$'))
            """
        )
    logger.info("rebuild network_demod: {} archived files", files)
