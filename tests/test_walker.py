import json
import random
from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest

from satnogs_tlm_agg import store
from satnogs_tlm_agg.http import ApiError, Page
from satnogs_tlm_agg.orbit import orbit_states
from satnogs_tlm_agg.satnogs_db import _db_window, ingest_telemetry_page
from satnogs_tlm_agg.timeutil import parse_ts
from satnogs_tlm_agg.walker import Schedule, plan_chunk, walk

LAUNCH = datetime(2024, 1, 1, tzinfo=UTC)
DECAY = datetime(2024, 4, 10, tzinfo=UTC)  # 100 days of mission
PAGE = 25


def rec(ts: datetime, i: int) -> dict:
    return {
        "sat_id": "S",
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "frame": f"{i:06X}",
        "observer": "o",
    }


class FakeDbApi:
    """Mimics SatNOGS DB telemetry: newest-first, cursor pagination, start (gte) / end (lte)."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.requests: list[str] = []
        self.invalid_cursors = False

    def get_page(self, url, params):
        if params is None:  # following a cursor
            if self.invalid_cursors:
                raise ApiError(404, url, "Invalid cursor")
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        else:
            q = {**params, "offset": "0"}
        self.requests.append(json.dumps(q, sort_keys=True))
        rows = sorted(self.records, key=lambda r: r["timestamp"], reverse=True)
        if "start" in q:
            rows = [r for r in rows if parse_ts(r["timestamp"]) >= parse_ts(q["start"])]
        if "end" in q:
            rows = [r for r in rows if parse_ts(r["timestamp"]) <= parse_ts(q["end"])]
        off = int(q["offset"])
        chunk = rows[off : off + PAGE]
        nxt = None
        if off + PAGE < len(rows):
            nq = {k: v for k, v in q.items() if k != "offset"} | {"offset": str(off + PAGE)}
            nxt = "https://db/api/telemetry/?" + "&".join(f"{k}={v}" for k, v in nq.items())
        return Page(url, chunk, nxt, {"next": nxt, "results": chunk})


def _walk(api, con, max_pages=1000, decayed=DECAY):
    return walk(
        api,
        con,
        kind="db_telemetry",
        sat_id="S",
        label="S",
        url="https://db/api/telemetry/",
        base_params={"sat_id": "S"},
        window_params=_db_window,
        ts_of=lambda r: parse_ts(r["timestamp"]),
        ingest=lambda recs: ingest_telemetry_page(con, recs),
        max_pages=max_pages,
        schedule=Schedule(eol_days=10),
        launched=LAUNCH,
        decayed=decayed,
    )


def q(con, sql):
    return con.execute(sql).fetchall()


@pytest.fixture
def con(tmp_path):
    return store.connect(tmp_path / "t.duckdb")


def test_satellite_without_data_costs_one_request(con):
    api = FakeDbApi([])
    _walk(api, con)
    assert len(api.requests) == 1
    # Every expected day (launch .. decay + 14 days) is recorded as fetched, with 0 records.
    assert q(con, "SELECT count(*), sum(n_records) FROM fetch_days")[0] == (100 + 15, 0)
    _walk(api, con)
    assert len(api.requests) == 1


def test_full_mission_fetched_eol_first_without_repeats(con):
    rng = random.Random(1)
    records = [rec(LAUNCH + timedelta(minutes=rng.randrange(100 * 24 * 60)), i) for i in range(2000)]
    api = FakeDbApi(records)
    turns = 0
    while True:
        turns += 1
        r = _walk(api, con, max_pages=3)  # small budget: many resumptions
        if r.requests == 0:
            break
    assert q(con, "SELECT count(*) FROM frames")[0][0] == 2000
    assert len(api.requests) == len(set(api.requests)), "a request was repeated"
    assert q(con, "SELECT count(*), sum(n_records) FROM fetch_days")[0] == (115, 2000)
    # After the 1-page probe, the next requests target the end-of-life window (last 10 days).
    second = json.loads(api.requests[1])
    assert parse_ts(second["end"]) >= DECAY - timedelta(days=10)


def test_stale_cursor_resumes_by_time(con):
    records = [rec(DECAY - timedelta(hours=h), h) for h in range(200)]  # dense end of life
    api = FakeDbApi(records)
    _walk(api, con, max_pages=3)
    api.invalid_cursors = True
    _walk(api, con, max_pages=2)
    api.invalid_cursors = False
    while _walk(api, con, max_pages=50).requests:
        pass
    assert q(con, "SELECT count(*) FROM frames")[0][0] == 200


def test_plan_chunk():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    # End of life first, newest first, extended backwards.
    assert plan_chunk(
        days, end=date(2024, 4, 10), fixed_end=True, n=5, eol_anchor=date(2024, 3, 1), eol_days=10
    ) == (
        date(2024, 2, 26),
        date(2024, 3, 2),
    )
    # Random chunks never exceed n days and only cover remaining days.
    rng = random.Random(0)
    for _ in range(50):
        a, b = plan_chunk(
            days[::2], end=date(2024, 4, 10), fixed_end=True, n=7, eol_anchor=None, eol_days=10, rng=rng
        )
        assert (b - a).days == 1  # every other day missing -> single days
    # Caught up on a live satellite: wait until a chunk's worth of days accumulates.
    end = date(2024, 4, 10)
    assert (
        plan_chunk([end - timedelta(days=1)], end=end, fixed_end=False, n=7, eol_anchor=None, eol_days=10)
        is None
    )
    assert plan_chunk([end - timedelta(days=1)], end=end, fixed_end=True, n=7, eol_anchor=None, eol_days=10)


def test_orbit_states_iss(con):
    # ISS (ZARYA) element set: 1 25544U 98067A 21264.51782528 .00002893 00000-0 58680-4 0 9991
    epoch = datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=264.51782528 - 1)
    con.execute(
        "INSERT INTO tles VALUES (25544, ?, 51.6442, 208.5856, 0.0001458, 47.2277, 50.1624, 15.48919419, 0.00002893, 0.58680e-4)",
        [epoch],
    )
    pts = pl.DataFrame(
        {
            "norad_cat_id": [25544, 25544, 25544],
            "timestamp": [epoch, epoch + timedelta(minutes=30), epoch + timedelta(days=30)],
        },
        schema={"norad_cat_id": pl.Int64, "timestamp": pl.Datetime("us", "UTC")},
    )
    out = orbit_states(con, pts).sort("timestamp")
    alt, speed = out["altitude_km"].to_list(), out["speed_km_s"].to_list()
    assert 400 < alt[0] < 440 and 400 < alt[1] < 440
    assert 7.6 < speed[0] < 7.7
    assert alt[2] is None  # no TLE within 7 days
