"""Command-line interface (tyro subcommands).

Two strictly separated stages, both working on one DuckDB file (`data/satnogs_tlm_agg.duckdb`):

1. **Fetch** (`fetch`, or the fetch half of `run`): SatNOGS API requests only. Frames are stored
   and progress (`fetch_state`, `fetch_days`) saved after every page. Three channels run
   concurrently, each paced under its own server throttle: SatNOGS DB telemetry, SatNOGS
   Network observations, and Network demod files.
2. **Process** (`tles`, `decode`, `export`, or the periodic half of `run`): loads historical
   TLEs, decodes the stored frames, propagates orbits, writes Parquet. No SatNOGS requests.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl
import tyro
from loguru import logger

from . import decode as decode_mod
from . import export as export_mod
from . import satnogs_db, satnogs_network, store, tles
from .config import (
    BLACKLIST_NORAD_IDS,
    DB_API_BASE,
    DB_HOST,
    NETWORK_API_BASE,
    NETWORK_HOST,
    USER_AGENT,
    Settings,
    load_settings,
)
from .http import ApiError, SatnogsClient
from .store import query_pl
from .timeutil import now
from .walker import Schedule

# --------------------------------------------------------------------------- options


@dataclass
class Common:
    """Options shared by all subcommands."""

    env_file: Path = Path(".env")
    """`.env` file holding SATNOGS_DB_API_KEY / SATNOGS_NETWORK_API_KEY (see `.env.sample`)."""
    data_dir: Path = Path("data")
    """Holds the DuckDB database (`satnogs_tlm_agg.duckdb`)."""
    out_dir: Path = Path("out")
    """Parquet output directory."""
    log_level: str = "INFO"


@dataclass
class SatFilter:
    """Which satellites to process. Default: all except the space-station blacklist,
    prioritised (re-entered with decoders first)."""

    norad: list[int] = field(default_factory=list)
    """Only these NORAD IDs."""
    sat_id: list[str] = field(default_factory=list)
    """Only these SatNOGS sat_ids."""
    status: list[Literal["re-entered", "in orbit", "future"]] = field(
        default_factory=lambda: ["re-entered", "in orbit"]
    )
    only_with_decoders: bool = False
    """Skip satellites that have no satnogs-decoders decoder assigned."""
    limit: int | None = None
    """Process at most this many satellites (after prioritisation)."""


@dataclass
class FetchOptions:
    """Request pacing and fetch order. Defaults stay under the SatNOGS server throttles
    (DB telemetry: 6 requests/min per user; Network observations: 240/hour per user)."""

    db: bool = True
    """Fetch SatNOGS DB telemetry (main frame source)."""
    network: bool = True
    """Fetch SatNOGS Network observation metadata (incl. .ogg/demod URLs)."""
    demod: satnogs_network.DemodMode = "gap-fill"
    """Network demod files: 'gap-fill' = only observations on fetched DB days without DB
    frames (they're otherwise duplicates of DB telemetry); 'all'; or 'off'."""
    db_request_interval: float = 10.5
    """Seconds between DB API requests (6/min server throttle)."""
    network_request_interval: float = 15.5
    """Seconds between Network API requests (240/hour server throttle)."""
    media_request_interval: float = 0.25
    """Seconds between demod file downloads (not API-throttled)."""
    db_pages_per_turn: int = 30
    """Max DB pages (25 frames each) per satellite before moving to the next one."""
    network_pages_per_turn: int = 10
    """Max Network pages (25 observations each) per satellite per turn."""
    demod_batch: int = 200
    """Observations per demod download batch."""
    eol_days: int = 60
    """Days before decay (or before the last record) fetched first."""
    max_chunk_days: int = 30
    """Longest random chunk (chunks are sized to ~1 page from the observed record rate)."""
    settle_hours: float = 6.0
    """A day is only fetched once it ended this long ago, so late uploads are included."""
    final_after_decay_days: float = 14.0
    """Re-entered satellites are fetched until this long after decay, then never again."""
    catalogue_refresh_hours: float = 24.0
    """Re-sync the satellite/transmitter catalogues this often."""

    def schedule(self, violator: bool = False) -> Schedule:
        return Schedule(
            settle=timedelta(hours=self.settle_hours),
            final_after_decay=timedelta(days=self.final_after_decay_days),
            eol_days=self.eol_days,
            max_chunk_days=self.max_chunk_days,
            # SatNOGS allows one telemetry request per day for "frequency violator" satellites.
            min_gap_after_turn=timedelta(hours=24.1) if violator else None,
        )


# --------------------------------------------------------------------------- helpers


@dataclass
class Ctx:
    settings: Settings
    con: duckdb.DuckDBPyConnection


def _setup(common: Common) -> Ctx:
    logger.remove()
    logger.add(sys.stderr, level=common.log_level, enqueue=True)
    settings = load_settings(common.env_file, common.data_dir, common.out_dir)
    return Ctx(settings, store.connect(settings.duckdb_path))


def _db_client(ctx: Ctx, opts: FetchOptions | None = None) -> SatnogsClient:
    if not ctx.settings.db_api_key:
        logger.warning("SATNOGS_DB_API_KEY not set: the DB telemetry endpoint requires it")
    return SatnogsClient(
        ctx.settings.db_api_key,
        auth_host=DB_HOST,
        min_interval=opts.db_request_interval if opts else 10.5,
        user_agent=USER_AGENT,
    )


def _network_client(ctx: Ctx, opts: FetchOptions | None = None) -> SatnogsClient:
    if not ctx.settings.network_api_key:
        logger.warning("SATNOGS_NETWORK_API_KEY not set: anonymous Network access is limited to 60 req/hour")
    return SatnogsClient(
        ctx.settings.network_api_key,
        auth_host=NETWORK_HOST,
        min_interval=opts.network_request_interval if opts else 15.5,
        media_min_interval=opts.media_request_interval if opts else 0.25,
        user_agent=USER_AGENT,
    )


def _sync_catalogue_if_stale(con: duckdb.DuckDBPyConnection, client: SatnogsClient, max_age_h: float) -> None:
    last = con.execute("SELECT max(synced_at) FROM satellites").fetchall()[0][0]
    if last is None or now() - last > timedelta(hours=max_age_h):
        satnogs_db.sync_satellites(client, con)
        satnogs_db.sync_transmitters(client, con)


def _due_satellites(con: duckdb.DuckDBPyConnection, kind: str, f: SatFilter) -> pl.DataFrame:
    """Satellites with days left to fetch for `kind`, in priority order."""
    df = query_pl(
        con,
        """
        SELECT c.sat_id, c.norad_cat_id, c.name, c.status, c.launched, c.decayed,
               coalesce(s.is_frequency_violator, FALSE) AS violator,
               EXISTS (SELECT 1 FROM satellite_decoders d WHERE d.sat_id = c.sat_id) AS has_decoder,
               c.sat_id IN (SELECT sat_id FROM merged_satellites) AS merged
        FROM fetch_coverage c JOIN satellites s USING (sat_id)
        WHERE c.kind = $kind
          AND (c.days_remaining > 0 OR c.walking OR NOT c.probed)
          AND (c.not_before IS NULL OR c.not_before <= now())
          AND coalesce(c.norad_cat_id, -1) NOT IN (SELECT unnest($blacklist))
        """,
        {"kind": kind, "blacklist": sorted(BLACKLIST_NORAD_IDS)},
    )
    df = df.filter(pl.col("status").is_in(f.status))
    if kind == "db_telemetry":
        # Merged satellites' telemetry is returned when querying their primary.
        df = df.filter(~pl.col("merged"))
    if f.norad:
        df = df.filter(pl.col("norad_cat_id").is_in(f.norad))
    if f.sat_id:
        df = df.filter(pl.col("sat_id").is_in(f.sat_id))
    if f.only_with_decoders:
        df = df.filter(pl.col("has_decoder"))
    reentered = pl.col("status") == "re-entered"
    df = df.with_columns(
        priority=pl.when(reentered & pl.col("has_decoder"))
        .then(0)
        .when(pl.col("has_decoder"))
        .then(1)
        .when(reentered)
        .then(2)
        .otherwise(3)
    ).sort(
        ["violator", "priority", "decayed", "norad_cat_id"],
        descending=[False, False, True, False],
        nulls_last=True,
    )
    return df.head(f.limit) if f.limit is not None else df


def _label(row: dict) -> str:
    return f"{row['name']} ({row['norad_cat_id']})"


# --------------------------------------------------------------------------- fetch channels

IDLE_WAIT_S = 900.0


def _walk_channel(ctx, sats, opts, stop, until_idle, *, kind, client, fetch_fn, pages, sync=False) -> None:
    con = ctx.con.cursor()
    client.stop_event = stop
    while not stop.is_set():
        if sync:
            _sync_catalogue_if_stale(con, client, opts.catalogue_refresh_hours)
        due = _due_satellites(con, kind, sats)
        requests = 0
        for row in due.iter_rows(named=True):
            if stop.is_set():
                return
            res = fetch_fn(
                client,
                con,
                row["sat_id"],
                _label(row),
                max_pages=1 if row["violator"] else pages,
                schedule=opts.schedule(violator=row["violator"]),
                launched=row["launched"],
                decayed=row["decayed"],
            )
            requests += res.requests
        if requests == 0:  # everything up to date (or waiting): check again later
            if until_idle:
                return
            stop.wait(IDLE_WAIT_S)


def _db_channel(ctx, sats, opts, stop, until_idle) -> None:
    _walk_channel(
        ctx,
        sats,
        opts,
        stop,
        until_idle,
        kind="db_telemetry",
        client=_db_client(ctx, opts),
        fetch_fn=satnogs_db.fetch_telemetry,
        pages=opts.db_pages_per_turn,
        sync=True,
    )


def _network_channel(ctx, sats, opts, stop, until_idle) -> None:
    _walk_channel(
        ctx,
        sats,
        opts,
        stop,
        until_idle,
        kind="network_obs",
        client=_network_client(ctx, opts),
        fetch_fn=satnogs_network.fetch_observations,
        pages=opts.network_pages_per_turn,
    )


def _demod_channel(
    ctx: Ctx, sats: SatFilter, opts: FetchOptions, stop: threading.Event, until_idle: bool
) -> None:
    con = ctx.con.cursor()
    client = _network_client(ctx, opts)
    client.stop_event = stop
    sat_ids = sats.sat_id or None
    if sats.norad:
        sat_ids = [
            r[0]
            for r in con.execute(
                "SELECT sat_id FROM satellites WHERE norad_cat_id IN (SELECT unnest(?))", [sats.norad]
            ).fetchall()
        ]
    while not stop.is_set():
        n_obs, _ = satnogs_network.fetch_demod_frames(
            client, con, mode=opts.demod, sat_ids=sat_ids, limit_observations=opts.demod_batch
        )
        if n_obs == 0:
            if until_idle:
                return
            stop.wait(IDLE_WAIT_S)


def _run_channel(name: str, fn, ctx: Ctx, sats, opts, stop: threading.Event, until_idle: bool) -> None:
    while not stop.is_set():
        try:
            fn(ctx, sats, opts, stop, until_idle)
            return
        except KeyboardInterrupt:
            return
        except ApiError as exc:
            if exc.status in (401, 403):
                logger.error("{}: authentication failed - check API keys in .env: {}", name, exc)
                stop.set()
                return
            logger.exception("{} channel error; restarting in 5 min", name)
        except Exception:
            logger.exception("{} channel crashed; restarting in 5 min", name)
        stop.wait(300)


def _start_channels(ctx: Ctx, sats: SatFilter, opts: FetchOptions, stop: threading.Event, until_idle: bool):
    channels = []
    if opts.db:
        channels.append(("db", _db_channel))
    if opts.network:
        channels.append(("network", _network_channel))
    if opts.demod != "off":
        channels.append(("demod", _demod_channel))
    threads = [
        threading.Thread(
            target=_run_channel, args=(n, fn, ctx, sats, opts, stop, until_idle), name=n, daemon=True
        )
        for n, fn in channels
    ]
    for t in threads:
        t.start()
    return threads


def _process(con: duckdb.DuckDBPyConnection, out_dir: Path, workers: int) -> None:
    """TLEs -> decode -> export. No SatNOGS requests."""
    try:
        tles.sync_tles(con)
    except Exception:
        logger.exception("TLE sync failed; exporting with the TLEs already stored")
    decode_mod.run_decode(con, workers=workers)
    export_mod.export_all(con, out_dir)
    with store.WRITE_LOCK:
        con.execute("CHECKPOINT")


# --------------------------------------------------------------------------- commands


def sync(common: Common) -> None:
    """Refresh the satellite and transmitter catalogues from SatNOGS DB."""
    ctx = _setup(common)
    client = _db_client(ctx)
    satnogs_db.sync_satellites(client, ctx.con)
    satnogs_db.sync_transmitters(client, ctx.con)


def fetch(common: Common, sats: SatFilter, opts: FetchOptions, until_idle: bool = False) -> None:
    """Fetch only (no processing). Runs until Ctrl-C, or with --until-idle until nothing is due."""
    ctx = _setup(common)
    _sync_catalogue_if_stale(ctx.con, _db_client(ctx, opts), opts.catalogue_refresh_hours)
    stop = threading.Event()
    threads = _start_channels(ctx, sats, opts, stop, until_idle)
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.warning("Stopping (progress is saved after every page)...")
        stop.set()
        for t in threads:
            t.join()
    ctx.con.execute("CHECKPOINT")


def tles_cmd(common: Common) -> None:
    """Load historical TLEs for the catalogue from Hugging Face (incremental)."""
    ctx = _setup(common)
    tles.sync_tles(ctx.con)


def decode(common: Common, decoder: tuple[str, ...] = (), workers: int = 4, redecode: bool = False) -> None:
    """Decode stored frames with satnogs-decoders (local only, incremental)."""
    ctx = _setup(common)
    decode_mod.run_decode(ctx.con, decoders=list(decoder) or None, workers=workers, redecode=redecode)


def export(
    common: Common,
    frames: bool = True,
    temp_regex: str = export_mod.DEFAULT_TEMP_REGEX,
    temp_exclude_regex: str = export_mod.DEFAULT_TEMP_EXCLUDE_REGEX,
) -> None:
    """Write all Parquet output tables (local only)."""
    ctx = _setup(common)
    export_mod.export_all(
        ctx.con,
        ctx.settings.out_dir,
        frames=frames,
        temp_regex=temp_regex,
        temp_exclude_regex=temp_exclude_regex,
    )


def run(
    common: Common,
    sats: SatFilter,
    opts: FetchOptions,
    process_every_hours: float = 12.0,
    workers: int = 4,
) -> None:
    """Long-running mode: fetch continuously; every `process_every_hours` load TLEs, decode the
    stored frames, and export Parquet."""
    ctx = _setup(common)
    _sync_catalogue_if_stale(ctx.con, _db_client(ctx, opts), opts.catalogue_refresh_hours)
    stop = threading.Event()
    threads = _start_channels(ctx, sats, opts, stop, until_idle=False)
    proc_con = ctx.con.cursor()
    try:
        while not stop.is_set():
            if stop.wait(process_every_hours * 3600):
                break
            logger.info("Processing stored data (TLEs, decode, export)")
            try:
                _process(proc_con, ctx.settings.out_dir, workers)
            except Exception:
                logger.exception("Processing failed; fetching continues")
    except KeyboardInterrupt:
        logger.warning("Stopping (progress is saved after every page)...")
    stop.set()
    for t in threads:
        t.join()
    ctx.con.execute("CHECKPOINT")


def status(common: Common, top: int = 30) -> None:
    """What has been fetched, and what remains."""
    ctx = _setup(common)
    con = ctx.con
    pl.Config.set_tbl_rows(top)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(40)
    print(
        query_pl(
            con, "SELECT source, count(*) AS frames, count(DISTINCT sat_id) AS sats FROM frames GROUP BY 1"
        )
    )
    print(
        query_pl(
            con,
            """
            SELECT kind, count(*) FILTER (WHERE probed) AS sats_probed,
                   count(*) FILTER (WHERE probed AND days_remaining = 0) AS sats_complete,
                   sum(days_fetched)::BIGINT AS days_fetched, sum(days_remaining)::BIGINT AS days_remaining,
                   sum(n_records)::BIGINT AS records, sum(n_requests)::BIGINT AS requests,
                   count(*) FILTER (WHERE last_error IS NOT NULL) AS with_errors
            FROM fetch_coverage WHERE coalesce(norad_cat_id, -1) NOT IN (SELECT unnest($bl))
              AND status IN ('in orbit', 're-entered')
            GROUP BY 1 ORDER BY 1
            """,
            {"bl": sorted(BLACKLIST_NORAD_IDS)},
        )
    )
    print(
        query_pl(
            con,
            """SELECT count(*) AS observations, count(*) FILTER (WHERE len(demod_urls) > 0) AS with_demod,
                      count(*) FILTER (WHERE demod_fetched) AS demod_fetched FROM network_observations""",
        )
    )
    print(query_pl(con, "SELECT count(DISTINCT norad_cat_id) AS tle_sats, count(*) AS tles FROM tles"))
    print(
        query_pl(
            con,
            """SELECT decoder, count(*) AS attempts, count(*) FILTER (WHERE ok) AS ok
               FROM decode_attempts GROUP BY 1 ORDER BY ok DESC""",
        ).head(top)
    )
    print(
        query_pl(
            con,
            """SELECT name, norad_cat_id, status, n_records, n_requests, days_fetched, days_remaining,
                      first_day, last_day
               FROM fetch_coverage WHERE kind = 'db_telemetry' ORDER BY n_records DESC""",
        ).head(top)
    )


def probe(common: Common, norad: int = 43803) -> None:
    """Fetch one page from each telemetry/observation endpoint and print it (2 requests)."""
    ctx = _setup(common)
    _sync_catalogue_if_stale(ctx.con, _db_client(ctx), 24 * 365)
    rows = ctx.con.execute("SELECT sat_id, name FROM satellites WHERE norad_cat_id = ?", [norad]).fetchall()
    if not rows:
        raise SystemExit(f"NORAD {norad} not in the satellite catalogue")
    sat_id, name = rows[0]
    print(f"{name}: sat_id={sat_id}")
    page = _db_client(ctx).get_page(f"{DB_API_BASE}/telemetry/", {"sat_id": sat_id, "format": "json"})
    print(f"DB telemetry: {len(page.records)} records, next={page.next_url}")
    print("first record:", page.records[0] if page.records else None)
    page = _network_client(ctx).get_page(
        f"{NETWORK_API_BASE}/observations/", {"sat_id": sat_id, "format": "json"}
    )
    print(f"Network: {len(page.records)} observations, next={page.next_url}")


def decoders(common: Common) -> None:
    """List decoders assigned in SatNOGS DB and whether satnogs-decoders provides them."""
    ctx = _setup(common)
    used = decode_mod.decoders_in_use(ctx.con)
    missing = [d for d in used if decode_mod.decoder_class(d) is None]
    print(
        f"satnogs-decoders {decode_mod.DECODER_VERSION}: {len(decode_mod.available_decoders())} decoder classes"
    )
    print(f"{len(used)} decoders referenced by SatNOGS DB; missing locally: {missing or 'none'}")


def main() -> None:
    try:
        tyro.extras.subcommand_cli_from_dict(
            {
                "run": run,
                "fetch": fetch,
                "tles": tles_cmd,
                "decode": decode,
                "export": export,
                "status": status,
                "sync": sync,
                "probe": probe,
                "decoders": decoders,
            },
            description="Aggregate SatNOGS telemetry for as many satellites as possible.",
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except ApiError as exc:
        if exc.status in (401, 403):
            logger.error("Authentication failed - set valid API keys in .env (see .env.sample): {}", exc)
            raise SystemExit(1) from exc
        raise
