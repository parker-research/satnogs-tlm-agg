# satnogs-tlm-agg

Tools to fetch and aggregate telemetry across hundreds of satellite missions on SatNOGS, summarizing temperature during orbital decay

## Summary

Aggregate as much satellite telemetry as possible from [SatNOGS](https://satnogs.org/), for
every satellite in the SatNOGS DB (alive and re-entered), decode it with
[satnogs-decoders](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders), and
write large Parquet tables, with a focus on **temperatures over each mission, especially
the final weeks before re-entry**.

## Setup

```sh
uv sync
cp .env.sample .env   # then fill in your keys
```

- `SATNOGS_DB_API_KEY`: required for the DB telemetry endpoint (from your db.satnogs.org profile).
- `SATNOGS_NETWORK_API_KEY`: optional. Anonymous Network access works but is heavily rate limited.
- `HF_TOKEN`: optional. Gives higher Hugging Face rate limits when loading historical TLEs.

**Polars runtime:** the dependency includes the Polars build for older CPUs (`rtcompat`), so it
runs on machines without AVX2 etc., and Polars uses it automatically. On a modern machine
you can switch to the faster default build with `POLARS_FORCE_PKG=32`. Fetching is bound by
API rate limits, so the difference is small here.

## Usage

```sh
uv run satnogs-tlm-agg probe --norad 43803   # sanity-check keys + response shapes (2 requests)
uv run satnogs-tlm-agg run                    # long-running mode (leave it on a machine)
```

Everything incremental lives in one DuckDB file, `data/satnogs_tlm_agg.duckdb`: fetched
frames and observations, fetch progress, TLEs, and decode results. `out/` holds only Parquet
exports, regenerated from the database.

`run` fetches continuously. Every `--process-every-hours` (default 12) it loads any missing
TLEs, decodes the stored frames, and exports Parquet; that step makes no SatNOGS requests.
Each stage can also be run on its own:

| command    | what it does |
|------------|--------------|
| `fetch`    | SatNOGS fetching only, until Ctrl-C (or `--until-idle`) |
| `tles`     | load missing historical TLEs from Hugging Face |
| `decode`   | decode all not-yet-decoded stored frames (local) |
| `export`   | write Parquet outputs (local) |
| `status`   | fetched vs. remaining days, requests made, TLEs, decodes |
| `sync`     | refresh satellite + transmitter catalogues (also automatic every 24 h) |
| `decoders` | which DB-assigned decoders the installed satnogs-decoders provides |

Only one process can open the database at a time (DuckDB lock). Stop `run` before using
another command, or read the Parquet files in `out/`.

Narrow the satellites with `--norad 39472 61762`, `--status re-entered`,
`--only-with-decoders`, or `--limit 50`. Toggle fetch channels with `--no-db`, `--no-network`,
`--demod {gap-fill,all,off}`. Space stations are always excluded: see
`BLACKLIST_NORAD_IDS` in `src/satnogs_tlm_agg/config.py`.

## How fetching works

SatNOGS API requests are treated as expensive. Server limits, from the SatNOGS source code:

| channel | page size | server throttle | our default pace |
|---------|-----------|-----------------|------------------|
| DB telemetry | 25 frames | 6 req/min per user; **1 req/day per satellite** for "frequency violators" | 1 req / 10.5 s (~200k frames/day max) |
| Network observations | 25 observations | 240 req/h per user (60 anonymous) | 1 req / 15.5 s |
| Network demod files | 1 frame per file | none | 4 files/s |

For each satellite (and each source), its mission is fetched in this order:

1. **Probe**: one request over the whole history. Satellites without data are finished
   after it. Otherwise it returns the newest page.
2. **End of life**: the 60 days (`--eol-days`) before decay, or before the last record if
   the satellite is still in orbit, newest first.
3. **Random chunks** over the rest of the mission, so a representative sample builds up
   early. Chunks are 1–30 days, sized from the observed record rate to cost about one page.
4. **Caught up**: new days are fetched once a chunk's worth has accumulated. Re-entered
   satellites are fetched until 14 days after decay, then never again.

Each chunk is walked newest-first with the server's cursor. After every page the frames are
stored, the cursor is saved in `fetch_state`, and every completed day is written to
`fetch_days` with its record count. A restart continues with exactly the next page, and no
time range is requested twice. Days are only fetched once they ended 6 h ago
(`--settle-hours`), so late uploads are included.

Progress queries:

```sql
SELECT * FROM fetch_coverage WHERE kind = 'db_telemetry' ORDER BY days_remaining DESC;
SELECT * FROM fetch_days WHERE sat_id = '...' ORDER BY day;            -- fetched days
SELECT * FROM fetch_days_remaining WHERE sat_id = '...' ORDER BY day;  -- remaining days
```

Network demod files are tracked per URL (`demod_files`). By default (`gap-fill`) they are
only downloaded for observations on fully fetched DB days that have no DB frames. SatNOGS
Network forwards its frames to the DB, so the others would be duplicates.

## Orbits

Historical TLEs come from the Hugging Face dataset
[juliensimon/space-track-tle-history](https://huggingface.co/datasets/juliensimon/space-track-tle-history)
(one Parquet file per year). The files aren't sorted by NORAD ID, so each yearly file is
scanned in full once, for every catalogue satellite that can appear in it (2013 onwards).
`tle_files` records what was loaded, so a file is only re-read when it changes upstream or a
new satellite appears. The first load reads ~7.8 GB and stores ~10M element sets.

Each temperature reading is propagated with [satkit](https://pypi.org/project/satkit/)
SGP4 from the TLE nearest in time (at most 7 days away), giving altitude above the WGS-84
ellipsoid, inertial speed, and the sub-satellite point.

## Outputs (`out/`)

| file | contents |
|------|----------|
| `satellites.parquet`, `transmitters.parquet`, `network_observations.parquet` | catalogues/metadata (observations include `payload_url`: the `.ogg` recording) |
| `coverage.parquet` | per satellite: frame counts by source, decoded frames, first/last frame, DB days fetched/remaining |
| `frames/sat_id=<id>/frames.parquet` | every raw frame (hex) from all sources |
| `decoded/decoder=<d>/<sat_id>.parquet` | wide: one row per decoded frame, one column per decoder field |
| `fields_long/decoder=<d>.parquet` | long: every numeric decoded field |
| `temperatures.parquet` | long: temperature-like fields, de-duplicated across receiving stations, with `mission_day`, `days_before_decay`, `n_receptions`, `plausible_celsius`, and orbit state (`altitude_km`, `speed_km_s`, `latitude_deg`, `longitude_deg`, `tle_epoch`, `tle_age_days`) |
| `temperatures_daily.parquet` | per satellite/field/day: n, min, max, mean, median, days_before_decay, mean/min altitude, mean speed |

Example: last 30 days of re-entered satellites.

```python
import polars as pl
t = pl.read_parquet("out/temperatures.parquet")
t.filter(pl.col("status") == "re-entered", pl.col("days_before_decay") < 30)
```

### Caveats

- **Units vary by decoder.** Many decoders already apply calibration (°C), but some expose raw
  ADC counts. `plausible_celsius` (value within −150…250) is only a rough flag. Check each
  field against its decoder's `.ksy` before drawing physical conclusions.
- Temperature fields are picked by name (`temp|tmp|therm`, minus `attempt`). Adjust this with
  `export --temp-regex ... --temp-exclude-regex ...`.
- A satellite only decodes if SatNOGS DB assigns it a decoder (~220 satellites). Raw frames are
  still collected for every satellite, so they can be decoded later when decoders appear.
  After a satnogs-decoders upgrade (`uv lock --upgrade-package satnogs-decoders`), failed
  attempts are retried automatically. Use `decode --redecode` to redo successful ones as well.
- Frames uploaded more than 6 h after their day ended are missed if that day was already
  fetched.
- Re-demodulating Network `.ogg` recordings (gr-satnogs) is not implemented yet. The URLs are
  stored in `network_observations.payload_url` for that future step.
