"""DuckDB working store for intermediate data (raw frames, progress state, decoded fields)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl

SCHEMA = """
CREATE TABLE IF NOT EXISTS satellites (
    sat_id          VARCHAR PRIMARY KEY,
    norad_cat_id    BIGINT,
    norad_follow_id BIGINT,
    name            VARCHAR,
    names           VARCHAR,
    status          VARCHAR,
    decayed         TIMESTAMPTZ,
    launched        TIMESTAMPTZ,
    deployed        TIMESTAMPTZ,
    operator        VARCHAR,
    countries       VARCHAR,
    decoders        VARCHAR[],
    associated_satellites VARCHAR[],
    reception_status VARCHAR,
    is_frequency_violator BOOLEAN,
    updated         TIMESTAMPTZ,
    raw_json        JSON,
    synced_at       TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS transmitters (
    uuid            VARCHAR PRIMARY KEY,
    sat_id          VARCHAR,
    norad_cat_id    BIGINT,
    description     VARCHAR,
    alive           BOOLEAN,
    type            VARCHAR,
    status          VARCHAR,
    mode            VARCHAR,
    baud            DOUBLE,
    downlink_low    BIGINT,
    downlink_high   BIGINT,
    raw_json        JSON,
    synced_at       TIMESTAMPTZ
);

-- One row per received frame. `source` is 'db' (SatNOGS DB telemetry API)
-- or 'network_demod' (demodulated data files attached to Network observations).
CREATE TABLE IF NOT EXISTS frames (
    frame_id        VARCHAR PRIMARY KEY,   -- sha1 of (source, sat_id, timestamp, frame_hex, observer)
    source          VARCHAR NOT NULL,
    sat_id          VARCHAR,
    norad_cat_id    BIGINT,
    timestamp       TIMESTAMPTZ,
    frame_hex       VARCHAR,
    transmitter     VARCHAR,
    app_source      VARCHAR,
    observer        VARCHAR,
    station_id      BIGINT,
    observation_id  BIGINT,
    version         VARCHAR,
    raw_json        JSON,
    fetched_at      TIMESTAMPTZ
);

-- Per-satellite fetch state ('db_telemetry' | 'network_obs').
--
-- Each satellite's timeline is fetched as a sequence of time *windows*, walked newest-first
-- with the server's cursor. After every page the `next_url` cursor is saved, so an interrupted
-- walk continues exactly where it stopped: no page is ever requested twice.
--   covered_until : everything with timestamp <= this is stored (NULL = nothing complete yet)
--   window_end    : upper bound of the window being walked (query parameter)
--   window_ceiling: becomes `covered_until` when the window completes
--   next_url      : saved server cursor for the next page of the current window
-- When a window completes the satellite is rescheduled (`next_check_at`) with exponential
-- back-off while nothing new arrives; re-entered satellites become `final` (never re-queried).
CREATE TABLE IF NOT EXISTS fetch_progress (
    kind            VARCHAR NOT NULL,
    sat_id          VARCHAR NOT NULL,
    backfill_done   BOOLEAN DEFAULT FALSE,
    oldest_ts       TIMESTAMPTZ,
    newest_ts       TIMESTAMPTZ,
    n_records       BIGINT DEFAULT 0,
    last_run        TIMESTAMPTZ,
    last_error      VARCHAR,
    PRIMARY KEY (kind, sat_id)
);
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS covered_until TIMESTAMPTZ;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS window_end TIMESTAMPTZ;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS window_ceiling TIMESTAMPTZ;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS window_oldest TIMESTAMPTZ;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS window_new BIGINT DEFAULT 0;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS next_url VARCHAR;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS next_check_at TIMESTAMPTZ;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS check_interval_h DOUBLE;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS final BOOLEAN DEFAULT FALSE;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS n_requests BIGINT DEFAULT 0;
ALTER TABLE fetch_progress ADD COLUMN IF NOT EXISTS layout_version INTEGER DEFAULT 1;

CREATE TABLE IF NOT EXISTS network_observations (
    id              BIGINT PRIMARY KEY,
    sat_id          VARCHAR,
    norad_cat_id    BIGINT,
    start           TIMESTAMPTZ,
    "end"           TIMESTAMPTZ,
    status          VARCHAR,
    ground_station  BIGINT,
    observer        VARCHAR,
    station_lat     DOUBLE,
    station_lng     DOUBLE,
    transmitter_uuid VARCHAR,
    transmitter_mode VARCHAR,
    transmitter_baud DOUBLE,
    payload_url     VARCHAR,               -- .ogg audio, for later re-demodulation
    waterfall_url   VARCHAR,
    archive_url     VARCHAR,
    demod_urls      VARCHAR[],
    demod_fetched   BOOLEAN DEFAULT FALSE,
    raw_json        JSON,
    fetched_at      TIMESTAMPTZ
);

-- Every demod file URL ever requested (so an interrupted observation never re-downloads).
CREATE TABLE IF NOT EXISTS demod_files (
    url             VARCHAR PRIMARY KEY,
    observation_id  BIGINT,
    http_status     INTEGER,
    n_bytes         INTEGER,
    frame_id        VARCHAR,
    fetched_at      TIMESTAMPTZ
);

-- Decoder results in long form. One row per (frame, decoder, field).
CREATE TABLE IF NOT EXISTS decoded_fields (
    frame_id        VARCHAR NOT NULL,
    decoder         VARCHAR NOT NULL,
    field           VARCHAR NOT NULL,
    value_double    DOUBLE,
    value_text      VARCHAR
);

CREATE TABLE IF NOT EXISTS decode_attempts (
    frame_id        VARCHAR NOT NULL,
    decoder         VARCHAR NOT NULL,
    ok              BOOLEAN NOT NULL,
    n_fields        INTEGER,
    error           VARCHAR,
    decoder_version VARCHAR,
    attempted_at    TIMESTAMPTZ,
    PRIMARY KEY (frame_id, decoder)
);

ALTER TABLE satellites ADD COLUMN IF NOT EXISTS is_frequency_violator BOOLEAN;

-- Telemetry of merged ("associated") satellites is served and attributed under the primary
-- satellite, so a primary inherits its children's decoders.
CREATE OR REPLACE VIEW satellite_decoders AS
    SELECT DISTINCT sat_id, decoder FROM (
        SELECT sat_id, unnest(decoders) AS decoder FROM satellites
        UNION ALL
        SELECT p.sat_id, unnest(c.decoders) AS decoder
        FROM (SELECT sat_id, unnest(associated_satellites) AS child_id FROM satellites) p
        JOIN satellites c ON c.sat_id = p.child_id
    );

-- Satellites whose telemetry is served via another (primary) satellite.
CREATE OR REPLACE VIEW merged_satellites AS
    SELECT DISTINCT unnest(associated_satellites) AS sat_id FROM satellites;
"""

# Convert progress rows written by the first (overlap-based) fetch layout.
MIGRATIONS = """
UPDATE fetch_progress SET covered_until = newest_ts, layout_version = 2
    WHERE layout_version = 1 AND backfill_done;
UPDATE fetch_progress SET window_end = oldest_ts, window_ceiling = newest_ts, layout_version = 2
    WHERE layout_version = 1 AND NOT backfill_done AND oldest_ts IS NOT NULL;
UPDATE fetch_progress SET layout_version = 2 WHERE layout_version = 1;
"""

# DuckDB allows concurrent cursors, but we serialise writes from the fetcher threads so
# conflicting transactions never abort. Writes are tiny compared to network waits.
WRITE_LOCK = threading.RLock()


def connect(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    con.execute("SET GLOBAL TimeZone = 'UTC'")
    if not read_only:
        con.execute(SCHEMA)
        con.execute(MIGRATIONS)
    return con


def upsert_df(
    con: duckdb.DuckDBPyConnection,
    table: str,
    df: pl.DataFrame,
    *,
    mode: Literal["replace", "ignore", "append"] = "replace",
) -> int:
    """Insert a Polars frame into `table` by column name. Returns rows in `df`.

    `replace`/`ignore` resolve primary-key conflicts; `append` is for tables without keys.
    """
    if df.is_empty():
        return 0
    cols = ", ".join(f'"{c}"' for c in df.columns)
    verb = {"replace": "INSERT OR REPLACE", "ignore": "INSERT OR IGNORE", "append": "INSERT"}[mode]
    with WRITE_LOCK:
        con.register("_incoming", df.to_arrow())
        try:
            con.execute(f"{verb} INTO {table} ({cols}) SELECT {cols} FROM _incoming")
        finally:
            con.unregister("_incoming")
    return df.height


def query_pl(con: duckdb.DuckDBPyConnection, sql: str, params: list | dict | None = None) -> pl.DataFrame:
    """Run a query and return a Polars DataFrame."""
    return con.sql(sql, params=params).pl()
