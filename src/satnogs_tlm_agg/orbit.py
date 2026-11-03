"""Orbit state (altitude, speed, sub-satellite point) at given times, via satkit SGP4."""

from __future__ import annotations

from datetime import timedelta

import duckdb
import numpy as np
import polars as pl
import satkit as sk
from loguru import logger

from .tles import TLE_COLUMNS

ORBIT_SCHEMA = {
    "norad_cat_id": pl.Int64,
    "timestamp": pl.Datetime("us", "UTC"),
    "tle_epoch": pl.Datetime("us", "UTC"),
    "tle_age_days": pl.Float64,
    "altitude_km": pl.Float64,
    "speed_km_s": pl.Float64,
    "latitude_deg": pl.Float64,
    "longitude_deg": pl.Float64,
}


def _omm(t: dict) -> dict:
    return {
        "EPOCH": t["epoch"],
        "MEAN_MOTION": t["mean_motion"],
        "ECCENTRICITY": float(t["eccentricity"]),
        "INCLINATION": float(t["inclination"]),
        "RA_OF_ASC_NODE": float(t["raan"]),
        "ARG_OF_PERICENTER": float(t["arg_perigee"]),
        "MEAN_ANOMALY": float(t["mean_anomaly"]),
        "BSTAR": t["bstar"],
        "MEAN_MOTION_DOT": t["mean_motion_dot"],
        "MEAN_MOTION_DDOT": 0.0,
    }


def orbit_states(
    con: duckdb.DuckDBPyConnection, points: pl.DataFrame, max_tle_age: timedelta = timedelta(days=7)
) -> pl.DataFrame:
    """Propagate each (norad_cat_id, timestamp) with the TLE nearest in time.

    altitude_km: height above the WGS-84 ellipsoid. speed_km_s: inertial (TEME) speed.
    Points without a TLE within `max_tle_age` get nulls.
    """
    points = (
        points.select("norad_cat_id", "timestamp").drop_nulls().unique().sort("norad_cat_id", "timestamp")
    )
    if points.is_empty():
        return pl.DataFrame(schema=ORBIT_SCHEMA)
    tles = con.sql(
        f"""SELECT norad_cat_id::BIGINT AS norad_cat_id, epoch AS tle_epoch, {", ".join(TLE_COLUMNS)}
            FROM tles WHERE norad_cat_id IN (SELECT unnest($ids)) ORDER BY norad_cat_id, epoch""",
        params={"ids": points["norad_cat_id"].unique().to_list()},
    ).pl()
    matched = points.join_asof(
        tles.with_columns(pl.col("tle_epoch").dt.convert_time_zone("UTC")),
        left_on="timestamp",
        right_on="tle_epoch",
        by="norad_cat_id",
        strategy="nearest",
        check_sortedness=False,  # both sides are sorted by (norad_cat_id, time) above
    ).with_columns(
        tle_age_days=(pl.col("timestamp") - pl.col("tle_epoch")).dt.total_seconds() / 86400,
    )
    usable = matched.filter(pl.col("tle_age_days").abs() <= max_tle_age.total_seconds() / 86400)

    rows = []
    for (norad, epoch), grp in usable.group_by("norad_cat_id", "tle_epoch", maintain_order=True):
        tle = {**grp.row(0, named=True), "epoch": epoch}
        stamps = grp["timestamp"].to_list()
        times = [sk.time.from_datetime(t) for t in stamps]
        try:
            pos, vel, err = sk.sgp4(_omm(tle), times, errflag=True)
        except RuntimeError as exc:
            logger.debug("sgp4 failed for {} @ {}: {}", norad, epoch, exc)
            continue
        quats = sk.frametransform.qteme2itrf(times)
        for t, p, v, e, q in zip(stamps, pos, vel, err, quats, strict=True):
            if e != 0 or not np.all(np.isfinite(p)):
                continue
            coord = sk.itrfcoord(q * p)
            rows.append(
                (
                    norad,
                    t,
                    coord.altitude / 1e3,
                    float(np.linalg.norm(v)) / 1e3,
                    coord.latitude_deg,
                    coord.longitude_deg,
                )
            )
    states = pl.DataFrame(
        rows,
        schema={
            "norad_cat_id": pl.Int64,
            "timestamp": pl.Datetime("us", "UTC"),
            "altitude_km": pl.Float64,
            "speed_km_s": pl.Float64,
            "latitude_deg": pl.Float64,
            "longitude_deg": pl.Float64,
        },
        orient="row",
    )
    out = matched.select("norad_cat_id", "timestamp", "tle_epoch", "tle_age_days").join(
        states, on=["norad_cat_id", "timestamp"], how="left"
    )
    # TLE columns only where a state was actually computed.
    out = out.with_columns(
        [
            pl.when(pl.col("altitude_km").is_not_null()).then(pl.col(c)).alias(c)
            for c in ("tle_epoch", "tle_age_days")
        ]
    )
    return out.select(list(ORBIT_SCHEMA)).cast(ORBIT_SCHEMA)
