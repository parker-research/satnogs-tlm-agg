"""Append-only archive of every raw API response, written *before* anything is parsed.

This is the source of truth: if parsing, the DuckDB schema, or the database itself changes or
breaks, `rebuild` re-ingests everything from here without one API request.

Layout: `data/raw/<kind>/<YYYY-MM-DD>.jsonl.gz`, one JSON object per line. Each write is its
own gzip member (valid concatenated gzip), so a crash loses at most the line being written.
"""

from __future__ import annotations

import gzip
import json
import threading
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from .timeutil import now


class RawArchive:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def write(self, kind: str, record: dict[str, Any]) -> None:
        ts = now()
        record = {"archived_at": ts.isoformat(), **record}
        line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        path = self.root / kind / f"{ts:%Y-%m-%d}.jsonl.gz"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                f.write(gzip.compress(line, compresslevel=6))
                f.flush()

    def iter(self, kind: str) -> Iterator[dict[str, Any]]:
        for path in sorted((self.root / kind).glob("*.jsonl.gz")):
            try:
                with gzip.open(path, "rt") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            except (EOFError, gzip.BadGzipFile, zlib.error, json.JSONDecodeError) as exc:
                logger.warning("Archive {}: stopped at damaged tail ({})", path, exc)

    def kinds(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("*") if p.is_dir()) if self.root.exists() else []
