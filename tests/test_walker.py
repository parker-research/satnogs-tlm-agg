import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from satnogs_tlm_agg import progress, store
from satnogs_tlm_agg.archive import RawArchive
from satnogs_tlm_agg.http import ApiError, Page
from satnogs_tlm_agg.rebuild import rebuild
from satnogs_tlm_agg.satnogs_db import _db_window, ingest_telemetry_page
from satnogs_tlm_agg.timeutil import parse_ts
from satnogs_tlm_agg.walker import Schedule, walk

T0 = datetime(2024, 1, 1, tzinfo=UTC)
PAGE = 3


def rec(i: int, observer: str = "obs", base: datetime = T0) -> dict:
    ts = (base + timedelta(minutes=i)).isoformat().replace("+00:00", "Z")
    return {"sat_id": "S", "timestamp": ts, "frame": f"{i:04X}", "observer": observer}


class FakeDbApi:
    """Mimics SatNOGS DB telemetry: newest-first, cursor pagination, start (gte) / end (lte)."""

    def __init__(self, n: int):
        self.records = [rec(i) for i in range(n)]
        self.requests: list[str] = []
        self.invalid_cursors = False

    def get_page(self, url, params):
        if params is None:  # following a cursor
            q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
            if self.invalid_cursors:
                raise ApiError(404, url, "Invalid cursor")
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
        return Page(url, chunk, nxt, {"next": nxt, "previous": None, "results": chunk})


SCHED = Schedule(settle=timedelta(0))


def _walk(api, con, archive, max_pages, decayed=None):
    return walk(
        api,
        con,
        archive,
        kind="db_telemetry",
        sat_id="S",
        label="S",
        url="https://db/api/telemetry/",
        base_params={"sat_id": "S"},
        window_params=_db_window,
        ts_of=lambda r: parse_ts(r["timestamp"]),
        ingest=lambda recs: ingest_telemetry_page(con, recs),
        max_pages=max_pages,
        schedule=SCHED,
        decayed=decayed,
    )


def n_frames(con):
    return con.execute("SELECT count(*) FROM frames").fetchall()[0][0]


@pytest.fixture
def env(tmp_path):
    return store.connect(tmp_path / "t.duckdb"), RawArchive(tmp_path / "raw")


def test_no_page_is_requested_twice(env):
    con, archive = env
    api = FakeDbApi(20)  # 7 pages of 3
    calls = 0
    while True:
        calls += 1
        r = _walk(api, con, archive, max_pages=2)  # budget forces 4 resumptions
        if r.window_complete:
            break
    assert calls == 4
    assert len(api.requests) == 7 == len(set(api.requests))
    assert n_frames(con) == 20
    p = progress.load(con, "db_telemetry", "S")
    assert p.backfill_done and p.next_check_at is not None and p.next_url is None


def test_incremental_window_fetches_only_new(env):
    con, archive = env
    api = FakeDbApi(10)
    while not _walk(api, con, archive, max_pages=50).window_complete:
        pass
    api.requests.clear()
    # New frames arrive after the first window's ceiling.
    recent = datetime.now(UTC)
    api.records += [rec(0, "a", base=recent), rec(0, "b", base=recent)]
    r = _walk(api, con, archive, max_pages=50)
    assert r.new_records == 2 and len(api.requests) == 1
    assert n_frames(con) == 12
    # Nothing new: one request, then back-off doubles the re-check interval.
    first = progress.load(con, "db_telemetry", "S").check_interval_h
    _walk(api, con, archive, max_pages=50)
    assert progress.load(con, "db_telemetry", "S").check_interval_h == 2 * first


def test_reentered_satellite_becomes_final(env):
    con, archive = env
    api = FakeDbApi(5)
    _walk(api, con, archive, max_pages=50, decayed=T0)
    assert progress.load(con, "db_telemetry", "S").final
    api.requests.clear()
    _walk(api, con, archive, max_pages=50, decayed=T0)
    assert api.requests == []


def test_stale_cursor_falls_back_to_time_query(env):
    con, archive = env
    api = FakeDbApi(20)
    _walk(api, con, archive, max_pages=2)
    api.invalid_cursors = True
    _walk(api, con, archive, max_pages=1)  # cursor rejected -> requery by time
    api.invalid_cursors = False
    while not _walk(api, con, archive, max_pages=50).window_complete:
        pass
    assert n_frames(con) == 20


def test_rebuild_from_archive_without_requests(env, tmp_path):
    con, archive = env
    api = FakeDbApi(10)
    while not _walk(api, con, archive, max_pages=50).window_complete:
        pass
    fresh = store.connect(tmp_path / "fresh.duckdb")
    rebuild(fresh, archive)
    assert n_frames(fresh) == 10
