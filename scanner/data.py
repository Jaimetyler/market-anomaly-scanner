from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from scanner.config import MASSIVE_API_KEY


BASE_URL = "https://api.massive.com"

DAILY_BAR_CACHE_DIR = Path("data/daily_bars")

# Hard bounds: no runaway retries.
HTTP_ATTEMPTS = 2
HTTP_CONNECT_TIMEOUT_SECONDS = 15.0
HTTP_READ_TIMEOUT_SECONDS = 45.0

# One lock per exact cache file. This prevents two worker threads from
# simultaneously fetching/writing the same ticker/date range.
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _normalize_ticker(ticker: str) -> str:
    normalized = ticker.upper().strip()

    if not normalized:
        raise ValueError("ticker cannot be empty.")

    return normalized


def _cache_path(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
) -> Path:
    safe_ticker = (
        _normalize_ticker(ticker)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    return (
        DAILY_BAR_CACHE_DIR
        / safe_ticker
        / f"{start_date}_{end_date}.json"
    )


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())

    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)

        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock

        return lock


def _load_cached_bars(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        # A corrupt/incomplete cache must never poison research.
        # Treat it as a miss and replace it from Massive.
        return None

    if not isinstance(payload, dict):
        return None

    if payload.get("cache_version") != 1:
        return None

    bars = payload.get("bars")

    if not isinstance(bars, list):
        return None

    return bars


def _atomic_save_cached_bars(
    *,
    path: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    bars: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "cache_version": 1,
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "adjusted": True,
        "bars": bars,
    }

    temp_path = path.with_suffix(
        path.suffix
        + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                separators=(",", ":"),
            )

        # Atomic on the same filesystem. Readers see either the old
        # complete file or the new complete file, never a partial JSON.
        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _fetch_daily_bars_from_massive(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    url = (
        f"{BASE_URL}"
        f"/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{start_date}/{end_date}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": MASSIVE_API_KEY,
    }

    timeout = httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT_SECONDS,
        read=HTTP_READ_TIMEOUT_SECONDS,
        write=HTTP_CONNECT_TIMEOUT_SECONDS,
        pool=HTTP_CONNECT_TIMEOUT_SECONDS,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        HTTP_ATTEMPTS + 1,
    ):
        try:
            response = httpx.get(
                url,
                params=params,
                timeout=timeout,
            )

            response.raise_for_status()

            payload = response.json()

            results = (
                payload.get(
                    "results",
                    [],
                )
                or []
            )

            if not isinstance(results, list):
                raise RuntimeError(
                    "Massive daily aggregate response "
                    "did not contain a list of results."
                )

            return results

        except (
            httpx.HTTPError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc

            if attempt >= HTTP_ATTEMPTS:
                break

            # Exactly one bounded retry.
            time.sleep(0.5)

    assert last_error is not None

    raise RuntimeError(
        f"Unable to fetch daily bars for {ticker} "
        f"after {HTTP_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def get_daily_bars(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Return adjusted Massive daily aggregate bars.

    Historical requests are cached locally by exact:

        ticker + start_date + end_date

    First request:
        Massive REST -> atomic local JSON cache.

    Subsequent identical requests:
        local disk -> no network request.

    `force_refresh=True` bypasses an existing cache and replaces it.

    The public positional/keyword interface used throughout the scanner
    remains compatible with the original get_daily_bars().
    """

    normalized_ticker = _normalize_ticker(
        ticker
    )

    cache_path = _cache_path(
        ticker=normalized_ticker,
        start_date=start_date,
        end_date=end_date,
    )

    if not force_refresh:
        cached = _load_cached_bars(
            cache_path
        )

        if cached is not None:
            return cached

    lock = _cache_lock(
        cache_path
    )

    with lock:
        # Another worker may have filled the cache while this thread
        # was waiting for the per-file lock.
        if not force_refresh:
            cached = _load_cached_bars(
                cache_path
            )

            if cached is not None:
                return cached

        bars = _fetch_daily_bars_from_massive(
            ticker=normalized_ticker,
            start_date=start_date,
            end_date=end_date,
        )

        _atomic_save_cached_bars(
            path=cache_path,
            ticker=normalized_ticker,
            start_date=start_date,
            end_date=end_date,
            bars=bars,
        )

        return bars