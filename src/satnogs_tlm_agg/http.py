"""Polite, resilient HTTP client built on `requests`.

- Token auth (`Authorization: Token <key>`), sent only to the API host (never to S3/CDN hosts).
- Per-host minimum interval between requests, set to stay *under* the server-side throttles
  so requests are never wasted on 429 responses.
- Short 429 waits are slept through; long ones (e.g. the 1/day "frequency violator" throttle)
  raise `RateLimited` so the caller can reschedule that satellite and move on.
- `get_page` handles both DRF pagination styles used by SatNOGS:
  JSON body `{"next": ..., "results": [...]}` and plain list + `Link: <...>; rel="next"`.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from loguru import logger

_RETRY_AFTER_WAIT_RE = re.compile(r"(\d+)\s*seconds?")


class ApiError(RuntimeError):
    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:500]}")
        self.status = status
        self.url = url
        self.body = body


class RateLimited(ApiError):
    """429 with a wait too long to sleep through inline."""

    def __init__(self, url: str, retry_after: float, body: str) -> None:
        super().__init__(429, url, body)
        self.retry_after = retry_after


@dataclass
class Page:
    url: str  # final request URL (with query string)
    records: list[dict[str, Any]]
    next_url: str | None
    body: Any  # parsed JSON body, for the raw archive


class SatnogsClient:
    def __init__(
        self,
        api_key: str | None,
        *,
        auth_host: str,
        min_interval: float,
        media_min_interval: float = 0.25,
        user_agent: str = "satnogs-tlm-agg",
        timeout: float = 90.0,
        max_retries: int = 8,
        max_inline_wait: float = 180.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept"] = "application/json"
        self._auth_header = {"Authorization": f"Token {api_key}"} if api_key else {}
        self.auth_host = auth_host
        self.min_interval = min_interval
        self.media_min_interval = media_min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_inline_wait = max_inline_wait
        self._last_request: dict[str, float] = {}
        self.n_requests = 0
        self.n_rate_limited = 0
        self.stop_event: threading.Event | None = None

    def _sleep(self, seconds: float) -> None:
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise KeyboardInterrupt
        else:
            time.sleep(seconds)

    def _throttle(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            interval = self.min_interval if host == self.auth_host else self.media_min_interval
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = time.monotonic()

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """GET with retries. Raises `ApiError` on non-retryable failures."""
        host = urlparse(url).netloc
        headers = self._auth_header if host == self.auth_host else {}
        backoff = 10.0
        for attempt in range(1, self.max_retries + 1):
            self._throttle(host)
            self.n_requests += 1
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.warning("Request error ({}/{}) {}: {}", attempt, self.max_retries, url, exc)
                self._sleep(backoff)
                backoff = min(backoff * 2, 900)
                continue

            if resp.status_code == 429:
                self.n_rate_limited += 1
                wait = self._retry_after(resp)
                if wait > self.max_inline_wait:
                    raise RateLimited(resp.url, wait, resp.text)
                logger.warning("Rate limited (429) on {} - sleeping {:.0f}s", host, wait)
                self._sleep(wait)
                continue
            if resp.status_code >= 500:
                logger.warning(
                    "HTTP {} ({}/{}) {} - retrying in {:.0f}s",
                    resp.status_code,
                    attempt,
                    self.max_retries,
                    resp.url,
                    backoff,
                )
                self._sleep(backoff)
                backoff = min(backoff * 2, 900)
                continue
            if resp.status_code >= 400:
                raise ApiError(resp.status_code, resp.url, resp.text)
            return resp
        raise ApiError(-1, url, f"gave up after {self.max_retries} attempts")

    @staticmethod
    def _retry_after(resp: requests.Response) -> float:
        wait: float | None = None
        header = resp.headers.get("Retry-After")
        if header:
            try:
                wait = float(header)
            except ValueError:
                wait = None
        if wait is None:
            # DRF puts "Expected available in N seconds." in the body.
            m = _RETRY_AFTER_WAIT_RE.search(resp.text or "")
            wait = float(m.group(1)) if m else 60.0
        return max(wait, 1.0) + 1.0

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return self.get(url, params).json()

    def get_page(self, url: str, params: dict[str, Any] | None = None) -> Page:
        """Fetch exactly one page. Pass `params=None` when `url` is a `next` link."""
        resp = self.get(url, params)
        body = resp.json()
        if isinstance(body, dict) and "results" in body:
            records = body["results"] or []
            next_url = body.get("next")
        elif isinstance(body, list):
            records = body
            next_url = resp.links.get("next", {}).get("url")
        else:
            raise ApiError(resp.status_code, resp.url, f"unexpected body shape: {str(body)[:300]}")
        return Page(resp.url, records, next_url, body)
