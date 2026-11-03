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

## Usage

```sh
uv run satnogs-tlm-agg probe --norad 43803   # sanity-check keys + response shapes (2 requests)
uv run satnogs-tlm-agg run                    # long-running mode (leave it on a machine)
```

The work is split into two strictly separate stages:

1. **Fetch** (API requests only). Every response is first appended raw to `data/raw/`, then its
   frames are stored in DuckDB (`data/satnogs.duckdb`). Nothing is decoded at this stage.
2. **Process** (no requests). Decodes the stored frames and exports Parquet.

`run` keeps the fetch stage going continuously and runs the process stage every
`--process-every-hours` (default 12). You can also run each stage on its own:

| command    | what it does |
|------------|--------------|
| `fetch`    | fetch only, until Ctrl-C (or `--until-idle`) |
| `decode`   | decode all not-yet-decoded stored frames (local) |
| `export`   | write Parquet outputs (local) |
| `status`   | progress: satellites covered/in progress/final, requests made, frames, decodes |
| `rebuild`  | re-ingest the raw archive into DuckDB, no requests |
| `sync`     | refresh satellite + transmitter catalogues (also automatic every 24 h) |
| `decoders` | which DB-assigned decoders the installed satnogs-decoders provides |

Narrow the set of satellites with `--norad 39472 61762`, `--status re-entered`,
`--only-with-decoders`, or `--limit 50`. Fetch channels can be toggled with `--no-db`,
`--no-network`, `--demod {gap-fill,all,off}`. See `--help` on each command.

## How fetching works (and why requests are never repeated)

API requests are treated as expensive. Server-side limits, from the SatNOGS source code:

| channel | page size | server throttle | our default pace | max throughput |
|---------|-----------|-----------------|------------------|----------------|
| DB telemetry (`/api/telemetry/`) | 25 frames | 6 req/min per user; **1 req/day per satellite** for "frequency violators" | 1 req / 10.5 s | ~8.5k frames/h, ~200k/day |
| Network observations | 25 observations | 240 req/h per user (60 anonymous) | 1 req / 15.5 s | ~5.8k observations/h |
| Network demod files (S3) | 1 frame per file | none | 4 files/s | — |

Pages can't be made bigger, since the servers have no page-size parameter. Each satellite's
history is fetched as **time windows, walked newest → oldest with the server's cursor**:

- **First window = the satellite's whole history** up to `now − 6 h` (the settle time, so
  late uploads are included). It isn't split into days, because empty days would still cost a
  request each.
- **After every page** the raw response is archived, its frames are stored, and the cursor
  for the next page is saved. Ctrl-C, a crash or a reboot resumes at exactly the next page.
- **Fairness:** each satellite gets up to `--db-pages-per-turn` (30 pages = 750 frames) per
  turn, then the next one goes. Priority: re-entered with decoders → with decoders →
  re-entered → others. Violators are limited to 1 page per day and go last.
- **Once the history is complete**, later windows only cover `(previous end, now − 6 h]`.
  There's no overlap, so already-fetched data is never fetched again. Re-checks start
  after 1 day and double while nothing new arrives, up to 30 days.
- **Re-entered satellites** become *final* once their coverage reaches 14 days past the decay
  date. They are never queried again.
- **Network demod files** are recorded per URL, so an interrupted observation never
  re-downloads a file. By default (`gap-fill`) they are only downloaded for observations the
  DB fetch has fully covered without returning frames for them. SatNOGS Network forwards its
  frames to the DB, so the rest would be duplicates.

A first pass over all ~2,800 satellites costs at least ~2,800 DB requests (about 8 hours),
most of which find little or nothing. Busy satellites then take much longer. For example,
1M frames = 40k requests ≈ 5 days.

## Outputs (`out/`)

| file | contents |
|------|----------|
| `satellites.parquet`, `transmitters.parquet`, `network_observations.parquet` | catalogues/metadata (observations include `payload_url` → `.ogg`) |
| `coverage.parquet` | per satellite: frame counts by source, decoded frames, first/last frame, days between last frame and decay |
| `frames/sat_id=<id>/frames.parquet` | every raw frame (hex) from all sources |
| `decoded/decoder=<d>/<sat_id>.parquet` | wide: one row per decoded frame, one column per decoder field |
| `fields_long/decoder=<d>.parquet` | long: every numeric decoded field |
| `temperatures.parquet` | long: temperature-like fields, de-duplicated across receiving stations, with `mission_day`, `days_before_decay`, `n_receptions`, `plausible_celsius` |
| `temperatures_daily.parquet` | per satellite/field/day: n, min, max, mean, median, days_before_decay |

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
- Frames uploaded more than 6 h (`--settle-hours`) after reception, into an already-covered
  window, are missed. That's the price of never re-reading covered windows.
- Re-demodulating Network `.ogg` recordings (gr-satnogs) is not implemented yet. The URLs are
  stored in `network_observations.payload_url` for that future step.
