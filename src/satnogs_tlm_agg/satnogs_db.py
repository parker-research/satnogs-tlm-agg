"""SatNOGS DB: satellites, transmitters, and telemetry frames."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import duckdb
import polars as pl
from loguru import logger

from .archive import RawArchive
from .config import DB_API_BASE
from .http import SatnogsClient
from .store import WRITE_LOCK, upsert_df
from .timeutil import iso, now, parse_ts
from .walker import Schedule, WalkResult, walk

FRAME_SCHEMA = {
    "frame_id": pl.String,
    "source": pl.String,
    "sat_id": pl.String,
    "norad_cat_id": pl.Int64,
    "timestamp": pl.Datetime("us", "UTC"),
    "frame_hex": pl.String,
    "transmitter": pl.String,
    "app_source": pl.String,
    "observer": pl.String,
    "station_id": pl.Int64,
    "observation_id": pl.Int64,
    "version": pl.String,
    "raw_json": pl.String,
    "fetched_at": pl.Datetime("us", "UTC"),
}


def frame_id(
    source: str,
    sat_id: str | None,
    timestamp: str | None,
    frame_hex: str,
    observer: str | None,
) -> str:
    key = f"{source}|{sat_id}|{timestamp}|{frame_hex.upper()}|{observer}"
    return hashlib.sha1(key.encode()).hexdigest()


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def sync_satellites(client: SatnogsClient, con: duckdb.DuckDBPyConnection, archive: RawArchive) -> int:
    sats = client.get_json(f"{DB_API_BASE}/satellites/", {"format": "json"})
    archive.write("db_satellites", {"body": sats})
    return ingest_satellites(con, sats)


def ingest_satellites(con: duckdb.DuckDBPyConnection, sats: list[dict[str, Any]]) -> int:
    synced = now()
    rows = [
        {
            "sat_id": s["sat_id"],
            "norad_cat_id": _int(s.get("norad_cat_id")),
            "norad_follow_id": _int(s.get("norad_follow_id")),
            "name": s.get("name"),
            "names": s.get("names"),
            "status": s.get("status"),
            "decayed": parse_ts(s.get("decayed")),
            "launched": parse_ts(s.get("launched")),
            "deployed": parse_ts(s.get("deployed")),
            "operator": s.get("operator"),
            "countries": s.get("countries"),
            "decoders": [t["decoder"] for t in s.get("telemetries") or [] if t.get("decoder")],
            "associated_satellites": [str(a) for a in s.get("associated_satellites") or []],
            "reception_status": s.get("reception_status"),
            "is_frequency_violator": bool(s.get("is_frequency_violator")),
            "updated": parse_ts(s.get("updated")),
            "raw_json": json.dumps(s),
            "synced_at": synced,
        }
        for s in sats
    ]
    df = pl.DataFrame(rows, infer_schema_length=None)
    upsert_df(con, "satellites", df)
    logger.info(
        "Synced {} satellites ({} with decoders)",
        df.height,
        df.filter(pl.col("decoders").list.len() > 0).height,
    )
    return df.height


def sync_transmitters(client: SatnogsClient, con: duckdb.DuckDBPyConnection, archive: RawArchive) -> int:
    txs = client.get_json(f"{DB_API_BASE}/transmitters/", {"format": "json"})
    archive.write("db_transmitters", {"body": txs})
    return ingest_transmitters(con, txs)


def ingest_transmitters(con: duckdb.DuckDBPyConnection, txs: list[dict[str, Any]]) -> int:
    synced = now()
    rows = [
        {
            "uuid": t["uuid"],
            "sat_id": t.get("sat_id"),
            "norad_cat_id": _int(t.get("norad_cat_id")),
            "description": t.get("description"),
            "alive": t.get("alive"),
            "type": t.get("type"),
            "status": t.get("status"),
            "mode": t.get("mode"),
            "baud": float(t["baud"]) if t.get("baud") is not None else None,
            "downlink_low": _int(t.get("downlink_low")),
            "downlink_high": _int(t.get("downlink_high")),
            "raw_json": json.dumps(t),
            "synced_at": synced,
        }
        for t in txs
    ]
    df = pl.DataFrame(rows, infer_schema_length=None)
    upsert_df(con, "transmitters", df)
    logger.info("Synced {} transmitters", df.height)
    return df.height


def telemetry_to_df(records: list[dict[str, Any]]) -> pl.DataFrame:
    fetched = now()
    rows = []
    for r in records:
        frame_hex = (r.get("frame") or "").strip()
        if not frame_hex:
            continue
        rows.append(
            {
                "frame_id": frame_id(
                    "db",
                    r.get("sat_id"),
                    r.get("timestamp"),
                    frame_hex,
                    r.get("observer"),
                ),
                "source": "db",
                "sat_id": r.get("sat_id"),
                "norad_cat_id": _int(r.get("norad_cat_id")),
                "timestamp": parse_ts(r.get("timestamp")),
                "frame_hex": frame_hex.upper(),
                "transmitter": r.get("transmitter"),
                "app_source": r.get("app_source"),
                "observer": r.get("observer"),
                "station_id": _int(r.get("station_id")),
                "observation_id": _int(r.get("observation_id")),
                "version": r.get("version"),
                "raw_json": json.dumps({k: v for k, v in r.items() if k != "frame"}),
                "fetched_at": fetched,
            }
        )
    return pl.DataFrame(rows, schema=FRAME_SCHEMA)


def insert_frames(con: duckdb.DuckDBPyConnection, df: pl.DataFrame) -> int:
    """Insert frames, ignoring duplicates. Returns the number of genuinely new rows."""
    if df.is_empty():
        return 0
    df = df.unique("frame_id")
    with WRITE_LOCK:
        con.register("_new_frames", df.select("frame_id").to_arrow())
        try:
            n_new = con.execute(
                "SELECT count(*) FROM _new_frames n ANTI JOIN frames f USING (frame_id)"
            ).fetchall()[0][0]
        finally:
            con.unregister("_new_frames")
        upsert_df(con, "frames", df, mode="ignore")
    return n_new


def _db_window(lower: datetime | None, upper: datetime) -> dict[str, str]:
    # DB filters: start = timestamp >= x, end = timestamp <= x
    params = {"end": iso(upper)}
    if lower is not None:
        params["start"] = iso(lower)
    return params


def ingest_telemetry_page(con: duckdb.DuckDBPyConnection, records: list[dict[str, Any]]) -> int:
    return insert_frames(con, telemetry_to_df(records))


def fetch_telemetry(
    client: SatnogsClient,
    con: duckdb.DuckDBPyConnection,
    archive: RawArchive,
    sat_id: str,
    label: str,
    *,
    max_pages: int,
    schedule: Schedule,
    decayed: datetime | None = None,
) -> WalkResult:
    return walk(
        client,
        con,
        archive,
        kind="db_telemetry",
        sat_id=sat_id,
        label=label,
        url=f"{DB_API_BASE}/telemetry/",
        base_params={"sat_id": sat_id, "format": "json"},
        window_params=_db_window,
        ts_of=lambda r: parse_ts(r.get("timestamp")),
        ingest=lambda records: ingest_telemetry_page(con, records),
        max_pages=max_pages,
        schedule=schedule,
        decayed=decayed,
    )
