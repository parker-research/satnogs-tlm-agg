"""Export the DuckDB working store to Parquet tables (written with Polars).

Layout under `out/`:
    satellites.parquet, transmitters.parquet, network_observations.parquet
    coverage.parquet                      per-satellite frame/decode counts and date ranges
    frames/sat_id=<id>/frames.parquet     every raw frame (hex), all sources
    decoded/decoder=<d>/<sat_id>.parquet  wide tables: one row per frame, one column per field
    fields_long/decoder=<d>.parquet       long numeric table of every decoded field
    temperatures.parquet                  long table of temperature-like fields, de-duplicated
                                          across stations, with mission-phase columns
    temperatures_daily.parquet            per satellite/field/day stats
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import duckdb
import polars as pl
from loguru import logger

from .config import BLACKLIST_NORAD_IDS
from .orbit import orbit_states
from .store import query_pl

# Case-insensitive RE2 patterns (DuckDB). Matches e.g. "batt_temp", "BatteryTemperature",
# "obc_tmp", "thermistor_3"; the exclusion drops e.g. "boot_attempts".
DEFAULT_TEMP_REGEX = r"temp|tmp|therm"
DEFAULT_TEMP_EXCLUDE_REGEX = r"attempt"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(path)  # atomic, so readers never see half-written files


def export_metadata(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    for table in ("satellites", "transmitters", "network_observations"):
        df = query_pl(con, f"SELECT * FROM {table}")
        _write(df, out / f"{table}.parquet")
        logger.info("Exported {} ({} rows)", table, df.height)

    coverage = query_pl(
        con,
        """
        WITH fr AS (
            SELECT sat_id, count(*) AS n_frames,
                   count(*) FILTER (WHERE source = 'db') AS n_frames_db,
                   count(*) FILTER (WHERE source = 'network_demod') AS n_frames_network,
                   min(timestamp) AS first_frame, max(timestamp) AS last_frame
            FROM frames GROUP BY sat_id
        ), dec AS (
            SELECT f.sat_id, count(DISTINCT a.frame_id) FILTER (WHERE a.ok) AS n_decoded_frames
            FROM decode_attempts a JOIN frames f USING (frame_id) GROUP BY f.sat_id
        ), obs AS (
            SELECT sat_id, count(*) AS n_network_obs, count(*) FILTER (WHERE status = 'good') AS n_network_obs_good
            FROM network_observations GROUP BY sat_id
        )
        SELECT s.sat_id, s.norad_cat_id, s.name, s.status, s.launched, s.decayed, s.decoders,
               coalesce(fr.n_frames, 0) AS n_frames, coalesce(fr.n_frames_db, 0) AS n_frames_db,
               coalesce(fr.n_frames_network, 0) AS n_frames_network,
               coalesce(dec.n_decoded_frames, 0) AS n_decoded_frames,
               fr.first_frame, fr.last_frame,
               date_diff('day', fr.last_frame, s.decayed) AS days_last_frame_before_decay,
               coalesce(obs.n_network_obs, 0) AS n_network_obs,
               coalesce(obs.n_network_obs_good, 0) AS n_network_obs_good,
               c.days_fetched AS db_days_fetched, c.days_remaining AS db_days_remaining,
               c.last_error AS db_last_error
        FROM satellites s
        LEFT JOIN fr USING (sat_id) LEFT JOIN dec USING (sat_id) LEFT JOIN obs USING (sat_id)
        LEFT JOIN fetch_coverage c ON c.kind = 'db_telemetry' AND c.sat_id = s.sat_id
        ORDER BY n_frames DESC
        """,
    )
    _write(coverage, out / "coverage.parquet")


def export_frames(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    sat_ids = [
        r[0] for r in con.execute("SELECT DISTINCT sat_id FROM frames WHERE sat_id IS NOT NULL").fetchall()
    ]
    for sat_id in sat_ids:
        df = query_pl(
            con,
            "SELECT * EXCLUDE (raw_json) FROM frames WHERE sat_id = ? ORDER BY timestamp",
            [sat_id],
        )
        _write(df, out / "frames" / f"sat_id={_safe(sat_id)}" / "frames.parquet")
    logger.info("Exported raw frames for {} satellites", len(sat_ids))


def _decoded_long(con: duckdb.DuckDBPyConnection, decoder: str, sat_id: str | None = None) -> pl.DataFrame:
    sql = """
        SELECT f.sat_id, f.norad_cat_id, f.timestamp, f.frame_id, f.source, f.observer,
               d.field, d.value_double, d.value_text
        FROM decoded_fields d JOIN frames f USING (frame_id)
        WHERE d.decoder = ?
    """
    params: list = [decoder]
    if sat_id is not None:
        sql += " AND f.sat_id = ?"
        params.append(sat_id)
    return query_pl(con, sql, params)


def export_decoded(con: duckdb.DuckDBPyConnection, out: Path) -> None:
    decoders = [
        r[0] for r in con.execute("SELECT DISTINCT decoder FROM decoded_fields ORDER BY 1").fetchall()
    ]
    meta = ["sat_id", "norad_cat_id", "timestamp", "frame_id", "source", "observer"]
    for decoder in decoders:
        sat_ids = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT f.sat_id FROM decode_attempts a JOIN frames f USING (frame_id) WHERE a.decoder = ? AND a.ok",
                [decoder],
            ).fetchall()
        ]
        long_parts = []
        for sat_id in sat_ids:
            long = _decoded_long(con, decoder, sat_id)
            if long.is_empty():
                continue
            # Fields with any text value become text columns; the rest stay numeric.
            text_fields = long.filter(pl.col("value_text").is_not_null())["field"].unique().to_list()
            num = long.filter(~pl.col("field").is_in(text_fields))
            txt = long.filter(pl.col("field").is_in(text_fields))
            wide = long.select(meta).unique("frame_id")
            if not num.is_empty():
                wide = wide.join(
                    num.pivot(
                        on="field",
                        index="frame_id",
                        values="value_double",
                        aggregate_function="first",
                    ),
                    on="frame_id",
                    how="left",
                )
            if not txt.is_empty():
                wide = wide.join(
                    txt.pivot(
                        on="field",
                        index="frame_id",
                        values="value_text",
                        aggregate_function="first",
                    ),
                    on="frame_id",
                    how="left",
                )
            _write(
                wide.sort("timestamp"),
                out / "decoded" / f"decoder={_safe(decoder)}" / f"{_safe(sat_id)}.parquet",
            )
            long_parts.append(num.select(meta + ["field", pl.col("value_double").alias("value")]))
        if long_parts:
            _write(
                pl.concat(long_parts),
                out / "fields_long" / f"decoder={_safe(decoder)}.parquet",
            )
        logger.info(
            "Exported decoded tables for decoder {} ({} satellites)",
            decoder,
            len(sat_ids),
        )


def export_temperatures(
    con: duckdb.DuckDBPyConnection,
    out: Path,
    temp_regex: str = DEFAULT_TEMP_REGEX,
    temp_exclude_regex: str = DEFAULT_TEMP_EXCLUDE_REGEX,
) -> None:
    temps = query_pl(
        con,
        """
        WITH t AS (
            SELECT f.sat_id, f.norad_cat_id, d.decoder, d.field, d.value_double AS value,
                   f.timestamp, f.frame_hex, f.observer, f.source
            FROM decoded_fields d JOIN frames f USING (frame_id)
            WHERE d.value_double IS NOT NULL
              AND regexp_matches(d.field, $inc, 'i') AND NOT regexp_matches(d.field, $exc, 'i')
        ), dedup AS (
            -- The same packet is often received by many stations: keep its earliest reception.
            SELECT sat_id, norad_cat_id, decoder, field, value,
                   min(timestamp) AS timestamp, count(*) AS n_receptions,
                   list(DISTINCT observer) AS observers, list(DISTINCT source) AS sources
            FROM t GROUP BY sat_id, norad_cat_id, decoder, field, value, frame_hex
        )
        SELECT d.*, s.name, s.status, s.launched, s.decayed,
               date_diff('second', s.launched, d.timestamp) / 86400.0 AS mission_day,
               date_diff('second', d.timestamp, s.decayed) / 86400.0 AS days_before_decay,
               d.value BETWEEN -150 AND 250 AS plausible_celsius
        FROM dedup d JOIN satellites s USING (sat_id)
        WHERE coalesce(d.norad_cat_id, -1) NOT IN (SELECT unnest($blacklist))
        ORDER BY d.sat_id, d.field, d.timestamp
        """,
        {"inc": temp_regex, "exc": temp_exclude_regex, "blacklist": sorted(BLACKLIST_NORAD_IDS)},
    )
    # Orbit state at each reading, propagated with SGP4 from the nearest historical TLE.
    orbits = orbit_states(con, temps.select("norad_cat_id", "timestamp"))
    temps = temps.join(orbits, on=["norad_cat_id", "timestamp"], how="left")
    _write(temps, out / "temperatures.parquet")
    logger.info(
        "Exported {} temperature readings ({} satellites, {} fields)",
        temps.height,
        temps["sat_id"].n_unique(),
        temps.select("sat_id", "field").unique().height,
    )

    daily = (
        temps.with_columns(day=pl.col("timestamp").dt.truncate("1d"))
        .group_by(
            "sat_id",
            "norad_cat_id",
            "name",
            "status",
            "decayed",
            "decoder",
            "field",
            "day",
        )
        .agg(
            n=pl.len(),
            min=pl.col("value").min(),
            max=pl.col("value").max(),
            mean=pl.col("value").mean(),
            median=pl.col("value").median(),
            days_before_decay=pl.col("days_before_decay").min(),
            altitude_km_mean=pl.col("altitude_km").mean(),
            altitude_km_min=pl.col("altitude_km").min(),
            speed_km_s_mean=pl.col("speed_km_s").mean(),
        )
        .sort("sat_id", "field", "day")
    )
    _write(daily, out / "temperatures_daily.parquet")


def export_all(
    con: duckdb.DuckDBPyConnection,
    out: Path,
    *,
    frames: bool = True,
    temp_regex: str = DEFAULT_TEMP_REGEX,
    temp_exclude_regex: str = DEFAULT_TEMP_EXCLUDE_REGEX,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    export_metadata(con, out)
    if frames:
        export_frames(con, out)
    for sub in ("decoded", "fields_long"):
        shutil.rmtree(out / sub, ignore_errors=True)  # drop tables for decoders that disappeared
    export_decoded(con, out)
    export_temperatures(con, out, temp_regex, temp_exclude_regex)
