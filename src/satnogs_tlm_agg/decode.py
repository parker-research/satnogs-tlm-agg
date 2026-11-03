"""Decode raw frames with the satnogs-decoders (Kaitai Struct) library.

Each satellite's decoders come from the SatNOGS DB `telemetries` field. Results land in
DuckDB `decoded_fields` in long form (frame, decoder, field, value), with one
`decode_attempts` row per (frame, decoder) so work is never repeated. Failed attempts made
with an older satnogs-decoders release are retried automatically after an upgrade.
"""

from __future__ import annotations

import math
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from typing import Any

import duckdb
import polars as pl
import satnogsdecoders
from loguru import logger
from satnogsdecoders import decoder as sd

from .store import WRITE_LOCK, upsert_df
from .timeutil import now

DECODER_VERSION: str = satnogsdecoders.__version__

FIELD_SCHEMA = {
    "frame_id": pl.String,
    "decoder": pl.String,
    "field": pl.String,
    "value_double": pl.Float64,
    "value_text": pl.String,
}
ATTEMPT_SCHEMA = {
    "frame_id": pl.String,
    "decoder": pl.String,
    "ok": pl.Boolean,
    "n_fields": pl.Int32,
    "error": pl.String,
    "decoder_version": pl.String,
    "attempted_at": pl.Datetime("us", "UTC"),
}


@cache
def decoder_class(name: str) -> Any | None:
    """Resolve a DB decoder name (e.g. 'fo29', 'oresat05') to its Kaitai class."""
    for candidate in (name.capitalize(), name.replace("_", "").capitalize()):
        cls = getattr(sd, candidate, None)
        if isinstance(cls, type):
            return cls
    return None


def available_decoders() -> list[str]:
    return sorted(
        n for n in dir(sd) if n[:1].isupper() and isinstance(getattr(sd, n), type) and n != "KaitaiStruct"
    )


def _to_value(v: Any) -> tuple[float | None, str | None]:
    if isinstance(v, bool):
        return float(v), None
    if isinstance(v, int):
        return (float(v), None) if abs(v) < 2**63 else (None, str(v))
    if isinstance(v, float):
        return (v, None) if math.isfinite(v) else (None, str(v))
    if isinstance(v, (bytes, bytearray)):
        return None, bytes(v).hex()
    if hasattr(v, "name") and hasattr(v, "value"):  # Kaitai enums
        return (float(v.value) if isinstance(v.value, (int, float)) else None), str(v.name)
    return None, str(v)


def decode_chunk(decoder_name: str, items: list[tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """Worker: decode (frame_id, frame_hex) pairs. Runs in a subprocess."""
    cls = decoder_class(decoder_name)
    fields: list[dict] = []
    attempts: list[dict] = []
    ts = now()
    for fid, frame_hex in items:
        try:
            struct = cls.from_bytes(bytes.fromhex(frame_hex))
            values = sd.get_fields(struct)
            for k, v in values.items():
                vd, vt = _to_value(v)
                fields.append(
                    {
                        "frame_id": fid,
                        "decoder": decoder_name,
                        "field": k,
                        "value_double": vd,
                        "value_text": vt,
                    }
                )
            attempts.append(
                {
                    "frame_id": fid,
                    "decoder": decoder_name,
                    "ok": bool(values),
                    "n_fields": len(values),
                    "error": None if values else "no fields",
                    "decoder_version": DECODER_VERSION,
                    "attempted_at": ts,
                }
            )
        except Exception as exc:  # noqa: BLE001 - Kaitai raises many types for non-matching frames
            attempts.append(
                {
                    "frame_id": fid,
                    "decoder": decoder_name,
                    "ok": False,
                    "n_fields": 0,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                    "decoder_version": DECODER_VERSION,
                    "attempted_at": ts,
                }
            )
    return fields, attempts


def _chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def decoders_in_use(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("SELECT DISTINCT decoder FROM satellite_decoders ORDER BY 1").fetchall()
    return [r[0] for r in rows]


def reset_decoder(con: duckdb.DuckDBPyConnection, decoder_name: str) -> None:
    with WRITE_LOCK:
        con.execute("DELETE FROM decoded_fields WHERE decoder = ?", [decoder_name])
        con.execute("DELETE FROM decode_attempts WHERE decoder = ?", [decoder_name])


def run_decode(
    con: duckdb.DuckDBPyConnection,
    *,
    decoders: list[str] | None = None,
    workers: int = 4,
    batch_size: int = 50_000,
    chunk_size: int = 2_000,
    redecode: bool = False,
) -> int:
    names = decoders or decoders_in_use(con)
    total_ok = 0
    # Retry failures from older decoder releases (decoders get fixed/extended over time).
    with WRITE_LOCK:
        con.execute("DELETE FROM decode_attempts WHERE NOT ok AND decoder_version <> ?", [DECODER_VERSION])

    # `spawn`: forking a process that has live fetcher threads is unsafe.
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    try:
        for name in names:
            if decoder_class(name) is None:
                logger.warning(
                    "Decoder '{}' not found in satnogs-decoders {}; skipping",
                    name,
                    DECODER_VERSION,
                )
                continue
            if redecode:
                reset_decoder(con, name)
            while True:
                pending = con.execute(
                    f"""
                    SELECT f.frame_id, f.frame_hex
                    FROM frames f
                    JOIN satellite_decoders sd ON sd.sat_id = f.sat_id AND sd.decoder = $name
                    WHERE NOT EXISTS (
                        SELECT 1 FROM decode_attempts a WHERE a.frame_id = f.frame_id AND a.decoder = $name
                    )
                    LIMIT {int(batch_size)}
                    """,
                    {"name": name},
                ).fetchall()
                if not pending:
                    break
                chunks = list(_chunks(pending, chunk_size))
                if pool is not None:
                    results = list(pool.map(decode_chunk, [name] * len(chunks), chunks))
                else:
                    results = [decode_chunk(name, c) for c in chunks]
                fields = [f for r in results for f in r[0]]
                attempts = [a for r in results for a in r[1]]
                with WRITE_LOCK:
                    con.execute("BEGIN")
                    try:
                        upsert_df(
                            con, "decoded_fields", pl.DataFrame(fields, schema=FIELD_SCHEMA), mode="append"
                        )
                        upsert_df(con, "decode_attempts", pl.DataFrame(attempts, schema=ATTEMPT_SCHEMA))
                        con.execute("COMMIT")
                    except Exception:
                        con.execute("ROLLBACK")
                        raise
                n_ok = sum(a["ok"] for a in attempts)
                total_ok += n_ok
                logger.info(
                    "decode {}: {} frames, {} ok, {} fields",
                    name,
                    len(attempts),
                    n_ok,
                    len(fields),
                )
                if len(pending) < batch_size:
                    break
    finally:
        if pool is not None:
            pool.shutdown()
    return total_ok
