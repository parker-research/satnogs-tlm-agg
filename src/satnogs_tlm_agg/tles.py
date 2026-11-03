"""Historical TLEs from the `juliensimon/space-track-tle-history` Hugging Face dataset.

The dataset has one Parquet file per year (~12 GB in total) that is not sorted by NORAD ID, so
reading the rows of even one satellite means scanning the whole yearly file. Each yearly file
is therefore scanned at most once for all SatNOGS satellites that could appear in it (years
from launch, but not before 2013, to decay). `tle_files` records which (file version,
satellite) pairs are loaded, so files are only re-read when they change upstream (the current
year's file grows) or when a new satellite appears in the catalogue.
"""

from __future__ import annotations

import os
from datetime import date

import duckdb
import polars as pl
import requests
from loguru import logger

from .config import BLACKLIST_NORAD_IDS
from .store import WRITE_LOCK, upsert_df
from .timeutil import now
from .walker import HISTORY_START

DATASET = "juliensimon/space-track-tle-history"
_TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/data"
_FILE_URL = "hf://datasets/" + DATASET + "/{path}"

TLE_COLUMNS = (
    "inclination",
    "raan",
    "eccentricity",
    "arg_perigee",
    "mean_anomaly",
    "mean_motion",
    "mean_motion_dot",
    "bstar",
)


def _list_files() -> dict[int, tuple[str, str]]:
    """{year: (path, version id)} of the dataset's yearly files."""
    resp = requests.get(_TREE_URL, timeout=60)
    resp.raise_for_status()
    out = {}
    for f in resp.json():
        name = f["path"].rsplit("/", 1)[-1]
        if f["type"] == "file" and name.startswith("tle_") and name.endswith(".parquet"):
            version = (f.get("lfs") or {}).get("oid") or f["oid"]
            out[int(name[4:8])] = (f["path"], version)
    return out


def _targets(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """NORAD IDs with the year range in which they can have SatNOGS data."""
    return con.sql(
        """
        SELECT norad_cat_id,
               greatest(year(coalesce(launched, DATE '2013-01-01')), $floor) AS first_year,
               year(coalesce(decayed + INTERVAL 15 DAY, now())) AS last_year
        FROM satellites
        WHERE norad_cat_id IS NOT NULL AND status IN ('in orbit', 're-entered')
          AND norad_cat_id NOT IN (SELECT unnest($blacklist))
        """,
        params={"floor": HISTORY_START.year, "blacklist": sorted(BLACKLIST_NORAD_IDS)},
    ).pl()


def _read_tles(con: duckdb.DuckDBPyConnection, path: str, norad_ids: list[int]) -> pl.DataFrame:
    return con.sql(
        f"""
        SELECT DISTINCT ON (norad_id, epoch)
               norad_id::INTEGER AS norad_cat_id, epoch, {", ".join(TLE_COLUMNS)}
        FROM read_parquet('{_FILE_URL.format(path=path)}')
        WHERE norad_id IN (SELECT unnest($ids))
        """,
        params={"ids": norad_ids},
    ).pl()


def sync_tles(con: duckdb.DuckDBPyConnection) -> int:
    """Load any missing TLEs. Returns the number of yearly files scanned."""
    files = _list_files()
    targets = _targets(con)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if token := os.environ.get("HF_TOKEN", "").strip():  # optional: higher Hugging Face rate limits
        con.execute("CREATE OR REPLACE SECRET hf_token (TYPE huggingface, TOKEN ?)", [token])
    scanned = 0
    for year in range(HISTORY_START.year, date.today().year + 1):
        if year not in files:
            continue
        path, version = files[year]
        wanted = targets.filter((pl.col("first_year") <= year) & (pl.col("last_year") >= year))[
            "norad_cat_id"
        ].to_list()
        loaded = {
            r[0]
            for r in con.execute(
                "SELECT norad_cat_id FROM tle_files WHERE file = ? AND version = ?", [path, version]
            ).fetchall()
        }
        missing = sorted(set(wanted) - loaded)
        if not missing:
            continue
        logger.info("TLEs: scanning {} for {} satellites", path, len(missing))
        try:
            df = _read_tles(con, path, missing)
        except duckdb.HTTPException as exc:
            logger.warning("TLEs: {} not read ({}); will retry next time", path, str(exc).splitlines()[0])
            continue
        loaded_rows = pl.DataFrame(
            {"file": path, "version": version, "norad_cat_id": missing, "loaded_at": now()},
            schema_overrides={"norad_cat_id": pl.Int32},
        )
        with WRITE_LOCK:
            upsert_df(con, "tles", df, mode="ignore")
            con.execute("DELETE FROM tle_files WHERE file = ? AND version <> ?", [path, version])
            upsert_df(con, "tle_files", loaded_rows)
        logger.info("TLEs: {} -> {} element sets", path, df.height)
        scanned += 1
    return scanned
