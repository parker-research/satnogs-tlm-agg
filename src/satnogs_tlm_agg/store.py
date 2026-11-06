"""DuckDB store: everything fetched, the incremental fetch progress, and decode results."""

from __future__ import annotations

import os
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

-- Days whose records are completely stored, per satellite and source ('db_telemetry' =
-- SatNOGS DB frames, 'network_obs' = SatNOGS Network observations). A day is recorded the
-- moment the newest-first walk has passed below its start; `n_records` may be 0.
CREATE TABLE IF NOT EXISTS fetch_days (
    kind            VARCHAR NOT NULL,
    sat_id          VARCHAR NOT NULL,
    day             DATE NOT NULL,
    n_records       BIGINT NOT NULL,
    fetched_at      TIMESTAMPTZ,
    PRIMARY KEY (kind, sat_id, day)
);

-- Per satellite and source: the walk in progress (if any). See walker.py for the fetch order
-- (probe -> end of life -> random chunks). A walk covers the days [window_start, window_end),
-- newest first; `next_url` (the server cursor) is saved after every page.
CREATE TABLE IF NOT EXISTS fetch_state (
    kind            VARCHAR NOT NULL,
    sat_id          VARCHAR NOT NULL,
    probed          BOOLEAN DEFAULT FALSE, -- the one-request whole-history probe is done
    window_start    DATE,                  -- NULL while probing
    window_end      DATE,                  -- exclusive; NULL = no walk in progress
    next_url        VARCHAR,
    oldest_fetched  TIMESTAMPTZ,           -- the walk has stored everything in (this, window_end)
    n_requests      BIGINT DEFAULT 0,      -- total requests made for this satellite/source
    not_before      TIMESTAMPTZ,           -- don't request before this (rate limits, errors)
    last_error      VARCHAR,
    updated_at      TIMESTAMPTZ,
    PRIMARY KEY (kind, sat_id)
);

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

-- Every demod file URL already requested (so an interrupted observation never re-downloads).
CREATE TABLE IF NOT EXISTS demod_files (
    url             VARCHAR PRIMARY KEY,
    observation_id  BIGINT,
    http_status     INTEGER,
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

-- Historical TLE mean elements (Hugging Face juliensimon/space-track-tle-history), for SGP4.
CREATE TABLE IF NOT EXISTS tles (
    norad_cat_id    INTEGER NOT NULL,
    epoch           TIMESTAMPTZ NOT NULL,
    inclination     FLOAT,
    raan            FLOAT,
    eccentricity    FLOAT,
    arg_perigee     FLOAT,
    mean_anomaly    FLOAT,
    mean_motion     DOUBLE,
    mean_motion_dot DOUBLE,
    bstar           DOUBLE,
    PRIMARY KEY (norad_cat_id, epoch)
);

-- Which satellites have been loaded from which version of each yearly TLE file.
CREATE TABLE IF NOT EXISTS tle_files (
    file            VARCHAR NOT NULL,
    version         VARCHAR NOT NULL,
    norad_cat_id    INTEGER NOT NULL,
    loaded_at       TIMESTAMPTZ,
    PRIMARY KEY (file, norad_cat_id)
);

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

-- Per satellite and source: fetched vs. remaining days. Expected days run from launch (not
-- before 2013, pre-SatNOGS) to yesterday, or to 14 days after decay. Must match
-- walker.expected_range.
CREATE OR REPLACE VIEW fetch_coverage AS
    SELECT k.kind, s.sat_id, s.name, s.norad_cat_id, s.status, s.launched, s.decayed,
           greatest(coalesce(s.launched::DATE, DATE '2013-01-01'), DATE '2013-01-01') AS expected_from,
           least(current_date, coalesce(s.decayed::DATE + 15, current_date)) AS expected_to,
           d.first_day, d.last_day,
           coalesce(d.days_fetched, 0) AS days_fetched,
           greatest(date_diff('day', expected_from, expected_to) - coalesce(d.days_fetched_expected, 0), 0)
               AS days_remaining,
           coalesce(d.n_records, 0) AS n_records,
           coalesce(st.n_requests, 0) AS n_requests,
           coalesce(st.probed, FALSE) AS probed,
           st.window_end IS NOT NULL AS walking,
           st.oldest_fetched AS walking_at,
           st.not_before,
           st.last_error
    FROM satellites s
    CROSS JOIN (VALUES ('db_telemetry'), ('network_obs')) k(kind)
    LEFT JOIN (
        SELECT f.kind, f.sat_id, min(day) AS first_day, max(day) AS last_day,
               count(*) AS days_fetched, sum(n_records) AS n_records,
               count(*) FILTER (WHERE day >= DATE '2013-01-01'
                                  AND day >= coalesce(sat.launched::DATE, DATE '2013-01-01')
                                  AND day < coalesce(sat.decayed::DATE + 15, current_date))
                   AS days_fetched_expected
        FROM fetch_days f JOIN satellites sat USING (sat_id) GROUP BY f.kind, f.sat_id
    ) d ON d.kind = k.kind AND d.sat_id = s.sat_id
    LEFT JOIN fetch_state st ON st.kind = k.kind AND st.sat_id = s.sat_id;

-- Every expected day not yet fetched (filter by sat_id/kind when querying).
CREATE OR REPLACE VIEW fetch_days_remaining AS
    SELECT e.kind, e.sat_id, e.name, e.day
    FROM (
        SELECT kind, sat_id, name,
               unnest(generate_series(expected_from::TIMESTAMP, (expected_to - 1)::TIMESTAMP,
                                      INTERVAL 1 DAY))::DATE AS day
        FROM fetch_coverage
    ) e
    ANTI JOIN fetch_days f ON f.kind = e.kind AND f.sat_id = e.sat_id AND f.day = e.day;
"""

# DuckDB allows concurrent cursors, but we serialise writes from the fetcher threads so
# conflicting transactions never abort. Writes are tiny compared to network waits.
WRITE_LOCK = threading.RLock()


def connect(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the store. Memory settings keep a long `run` from exhausting DuckDB's buffer pool:
    once a checkpoint can't allocate, DuckDB invalidates the database until restart.

    `SATNOGS_TLM_AGG_MEMORY_LIMIT` (e.g. '4GB') and `SATNOGS_TLM_AGG_THREADS` override the
    defaults (DuckDB's 80% of RAM, 4 threads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, str] = {
        "threads": os.environ.get("SATNOGS_TLM_AGG_THREADS", "").strip() or "4",
        # Large sorts/joins spill to disk instead of failing.
        "temp_directory": str(path.parent / ".duckdb_tmp"),
        "preserve_insertion_order": "false",
    }
    if limit := os.environ.get("SATNOGS_TLM_AGG_MEMORY_LIMIT", "").strip():
        config["memory_limit"] = limit
    con = duckdb.connect(str(path), read_only=read_only, config=config)
    # The TLE sync streams ~12 GB of remote Parquet; don't keep it in the buffer pool.
    con.execute("SET enable_external_file_cache = false")
    con.execute("SET GLOBAL TimeZone = 'UTC'")
    if not read_only:
        con.execute(SCHEMA)
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
