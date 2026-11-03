"""SatNOGS Network: observation metadata (incl. .ogg audio URLs) and demodulated frame files."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Literal

import duckdb
import polars as pl
from loguru import logger

from .config import NETWORK_API_BASE
from .http import ApiError, SatnogsClient
from .satnogs_db import FRAME_SCHEMA, _int, frame_id, insert_frames
from .store import WRITE_LOCK
from .timeutil import iso, now, parse_ts
from .walker import Schedule, WalkResult, walk

DemodMode = Literal["gap-fill", "all", "off"]

# e.g. .../data_15337318_2026-11-02T22-43-56 or ..._2026-11-02T22-43-56_g0
_DEMOD_TS_RE = re.compile(r"data_\d+_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})")
_NON_FRAME_EXT = (".png", ".jpg", ".jpeg", ".wav", ".ogg", ".txt.png")


def _demod_urls(obs: dict[str, Any]) -> list[str]:
    out = []
    for d in obs.get("demoddata") or []:
        url = d.get("payload_demod") if isinstance(d, dict) else None
        if url:
            out.append(url)
    return out


def observations_to_df(records: list[dict[str, Any]]) -> pl.DataFrame:
    fetched = now()
    rows = [
        {
            "id": int(o["id"]),
            "sat_id": o.get("sat_id"),
            "norad_cat_id": _int(o.get("norad_cat_id")),
            "start": parse_ts(o.get("start")),
            "end": parse_ts(o.get("end")),
            "status": o.get("status"),
            "ground_station": _int(o.get("ground_station")),
            "observer": o.get("observer"),
            "station_lat": o.get("station_lat"),
            "station_lng": o.get("station_lng"),
            "transmitter_uuid": o.get("transmitter_uuid") or o.get("transmitter"),
            "transmitter_mode": o.get("transmitter_mode"),
            "transmitter_baud": o.get("transmitter_baud"),
            "payload_url": o.get("payload"),
            "waterfall_url": o.get("waterfall"),
            "archive_url": o.get("archive_url"),
            "demod_urls": _demod_urls(o),
            "raw_json": json.dumps(o),
            "fetched_at": fetched,
        }
        for o in records
    ]
    return pl.DataFrame(
        rows,
        schema_overrides={
            "station_lat": pl.Float64,
            "station_lng": pl.Float64,
            "transmitter_baud": pl.Float64,
            "demod_urls": pl.List(pl.String),
            "start": pl.Datetime("us", "UTC"),
            "end": pl.Datetime("us", "UTC"),
            "fetched_at": pl.Datetime("us", "UTC"),
        },
        infer_schema_length=None,
    )


def insert_observations(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0
    cols = ", ".join(f'"{c}"' for c in df.columns)
    with WRITE_LOCK:
        con.register("_obs", df.to_arrow())
        try:
            n_new = con.execute(
                "SELECT count(*) FROM _obs o ANTI JOIN network_observations n USING (id)"
            ).fetchall()[0][0]
            # Keep `demod_fetched` when an observation is seen again, unless its demod URLs changed.
            con.execute(
                f"""
                INSERT INTO network_observations ({cols}) SELECT {cols} FROM _obs
                ON CONFLICT (id) DO UPDATE SET
                    status = excluded.status,
                    archive_url = excluded.archive_url,
                    payload_url = excluded.payload_url,
                    waterfall_url = excluded.waterfall_url,
                    demod_fetched = CASE WHEN network_observations.demod_urls = excluded.demod_urls
                                         THEN network_observations.demod_fetched ELSE FALSE END,
                    demod_urls = excluded.demod_urls,
                    raw_json = excluded.raw_json,
                    fetched_at = excluded.fetched_at
                """
            )
        finally:
            con.unregister("_obs")
    return n_new


def _network_window(lower: datetime | None, upper: datetime, inclusive: bool) -> dict[str, str]:
    # Network filters: start = obs start >= x, start__lt = obs start < x
    params = {"start__lt": iso(upper + timedelta(microseconds=1) if inclusive else upper)}
    if lower is not None:
        params["start"] = iso(lower)
    return params


def ingest_observations_page(con: duckdb.DuckDBPyConnection, records: list[dict[str, Any]]) -> int:
    return insert_observations(con, observations_to_df(records))


def fetch_observations(
    client: SatnogsClient,
    con: duckdb.DuckDBPyConnection,
    sat_id: str,
    label: str,
    *,
    max_pages: int,
    schedule: Schedule,
    launched: datetime | None,
    decayed: datetime | None,
) -> WalkResult:
    return walk(
        client,
        con,
        kind="network_obs",
        sat_id=sat_id,
        label=label,
        url=f"{NETWORK_API_BASE}/observations/",
        base_params={"sat_id": sat_id, "format": "json"},
        window_params=_network_window,
        ts_of=lambda r: parse_ts(r.get("start")),
        ingest=lambda records: ingest_observations_page(con, records),
        max_pages=max_pages,
        schedule=schedule,
        launched=launched,
        decayed=decayed,
    )


def _demod_timestamp(url: str, fallback: Any) -> Any:
    m = _DEMOD_TS_RE.search(url)
    if not m:
        return fallback
    return parse_ts(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}Z") or fallback


def _looks_like_frame(content: bytes) -> bytes | None:
    """Demod files are usually raw frame bytes; some older ones are hex text."""
    if not content:
        return None
    try:
        text = content.decode("ascii").strip()
    except UnicodeDecodeError:
        return content
    compact = re.sub(r"\s+", "", text)
    if compact and len(compact) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", compact):
        return bytes.fromhex(compact)
    return content


_OBS_META_COLS = "id, sat_id, norad_cat_id, start, observer, ground_station, transmitter_uuid"


def demod_frame_row(obs: tuple, url: str, content: bytes) -> dict[str, Any] | None:
    obs_id, sat_id, norad, start, observer, station, tx = obs
    frame = _looks_like_frame(content)
    if frame is None:
        return None
    frame_hex = frame.hex().upper()
    return {
        "frame_id": frame_id("network_demod", sat_id, url, frame_hex, observer),
        "source": "network_demod",
        "sat_id": sat_id,
        "norad_cat_id": norad,
        "timestamp": _demod_timestamp(url, start),
        "frame_hex": frame_hex,
        "transmitter": tx,
        "app_source": "network",
        "observer": observer,
        "station_id": station,
        "observation_id": obs_id,
        "version": None,
        "raw_json": json.dumps({"url": url}),
        "fetched_at": now(),
    }


def record_demod_file(con: duckdb.DuckDBPyConnection, url: str, obs_id: int, status: int) -> None:
    with WRITE_LOCK:
        con.execute("INSERT OR IGNORE INTO demod_files VALUES (?, ?, ?, ?)", [url, obs_id, status, now()])


def pending_demod_observations(
    con: duckdb.DuckDBPyConnection, mode: DemodMode, sat_ids: list[str] | None, limit: int
) -> list[tuple]:
    """Observations whose demod files are worth downloading, decodable satellites first.

    `gap-fill` (default): SatNOGS Network forwards demodulated frames to SatNOGS DB, so the
    files are mostly duplicates of DB telemetry. Only download them for observations inside a
    day the DB fetch has fully covered and for which the DB returned no frames.
    """
    if mode == "off":
        return []
    where = ["NOT o.demod_fetched", "len(o.demod_urls) > 0"]
    params: dict[str, Any] = {"limit": limit}
    if sat_ids:
        where.append("o.sat_id IN (SELECT unnest($sat_ids))")
        params["sat_ids"] = sat_ids
    if mode == "gap-fill":
        where.append(
            """EXISTS (SELECT 1 FROM fetch_days fd WHERE fd.kind = 'db_telemetry'
                       AND fd.sat_id = coalesce(prim.primary_id, o.sat_id) AND fd.day = o.start::DATE)"""
        )
        where.append("NOT EXISTS (SELECT 1 FROM frames f WHERE f.source = 'db' AND f.observation_id = o.id)")
    return con.execute(
        f"""
        WITH prim AS (  -- DB telemetry of merged satellites is served under the primary
            SELECT unnest(associated_satellites) AS child, sat_id AS primary_id FROM satellites
        )
        SELECT {", ".join("o." + c.strip() for c in _OBS_META_COLS.split(","))}, o.demod_urls
        FROM network_observations o
        LEFT JOIN prim ON prim.child = o.sat_id
        WHERE {" AND ".join(where)}
        ORDER BY EXISTS (SELECT 1 FROM satellite_decoders d WHERE d.sat_id = o.sat_id) DESC, o.start DESC
        LIMIT $limit
        """,
        params,
    ).fetchall()


def fetch_demod_frames(
    client: SatnogsClient,
    con: duckdb.DuckDBPyConnection,
    *,
    mode: DemodMode = "gap-fill",
    sat_ids: list[str] | None = None,
    limit_observations: int = 200,
) -> tuple[int, int]:
    """Download demod files for pending observations. Returns (observations, new frames)."""
    obs_rows = pending_demod_observations(con, mode, sat_ids, limit_observations)
    total = 0
    for row in obs_rows:
        meta, urls = row[:-1], row[-1]
        obs_id = meta[0]
        urls = [u for u in urls if not u.lower().endswith(_NON_FRAME_EXT)]
        done = {
            r[0]
            for r in con.execute(
                "SELECT url FROM demod_files WHERE url IN (SELECT unnest(?))", [urls]
            ).fetchall()
        }
        retry_later = False
        for url in urls:
            if url in done:
                continue
            try:
                content = client.get(url).content
                status = 200
            except ApiError as exc:
                logger.warning("demod {} obs {}: {}", url, obs_id, exc)
                if exc.status not in (403, 404, 410):
                    retry_later = True  # transient: try this file again later
                    continue
                content, status = b"", exc.status
            r = demod_frame_row(meta, url, content) if content else None
            if r is not None:
                total += insert_frames(con, pl.DataFrame([r], schema=FRAME_SCHEMA))
            record_demod_file(con, url, obs_id, status)
        if not retry_later:
            with WRITE_LOCK:
                con.execute("UPDATE network_observations SET demod_fetched = TRUE WHERE id = ?", [obs_id])
    if obs_rows:
        logger.info("Fetched demod files for {} observations (+{} frames)", len(obs_rows), total)
    return len(obs_rows), total
