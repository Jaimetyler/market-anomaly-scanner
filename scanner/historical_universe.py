from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Iterable

import httpx

from scanner.config import MASSIVE_API_KEY
from scanner.data import BASE_URL


CACHE_DIR = Path("data/historical_universe")

ALLOWED_PRIMARY_EXCHANGES = {
    "XNYS",  # NYSE
    "XNAS",  # Nasdaq
    "XASE",  # NYSE American
}

ALLOWED_TYPES = {
    "CS",  # Common Stock
}


@dataclass(frozen=True)
class HistoricalTicker:
    ticker: str
    name: str | None
    active: bool | None
    type: str | None
    primary_exchange: str | None
    cik: str | None
    composite_figi: str | None
    share_class_figi: str | None
    list_date: str | None
    delisted_utc: str | None


@dataclass(frozen=True)
class HistoricalUniverseResult:
    as_of_date: date
    tickers: list[HistoricalTicker]
    total_count: int
    fetched_from_cache: bool


@dataclass(frozen=True)
class RollingUniverseResult:
    """
    Point-in-time universe snapshots for a sequence
    of historical market sessions.

    snapshots:
        Maps ISO session date -> eligible ticker list.

    first_seen:
        First requested session on which a ticker
        appears in the eligible point-in-time universe.

    last_seen:
        Last requested session on which a ticker
        appears in the eligible point-in-time universe.

    securities:
        Best-known HistoricalTicker metadata keyed
        by ticker.

    fetched_snapshots:
        Number of snapshots retrieved from Massive.

    cached_snapshots:
        Number of snapshots loaded from local cache.

    membership_dates_by_ticker:
        Precomputed inverse membership index. Maps ticker to
        the exact requested session dates on which it belonged
        to the point-in-time universe.
    """

    snapshots: dict[str, list[HistoricalTicker]]
    first_seen: dict[str, str]
    last_seen: dict[str, str]
    securities: dict[str, HistoricalTicker]
    fetched_snapshots: int
    cached_snapshots: int
    membership_dates_by_ticker: dict[str, frozenset[str]] = field(
        default_factory=dict
    )


def _normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(value)


def _cache_path(
    as_of_date: date,
) -> Path:
    return (
        CACHE_DIR
        / f"{as_of_date.isoformat()}.json"
    )


def _load_cache(
    as_of_date: date,
) -> dict[str, Any] | None:
    path = _cache_path(as_of_date)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _save_cache(
    as_of_date: date,
    payload: dict[str, Any],
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = _cache_path(as_of_date)

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )


def _parse_ticker(
    raw: dict[str, Any],
) -> HistoricalTicker:
    return HistoricalTicker(
        ticker=(
            raw.get("ticker")
            or ""
        ).upper().strip(),
        name=raw.get("name"),
        active=raw.get("active"),
        type=raw.get("type"),
        primary_exchange=raw.get(
            "primary_exchange"
        ),
        cik=raw.get("cik"),
        composite_figi=raw.get(
            "composite_figi"
        ),
        share_class_figi=raw.get(
            "share_class_figi"
        ),
        list_date=raw.get("list_date"),
        delisted_utc=raw.get(
            "delisted_utc"
        ),
    )


def _eligible_ticker(
    ticker: HistoricalTicker,
) -> bool:
    if not ticker.ticker:
        return False

    if ticker.active is not True:
        return False

    if ticker.type not in ALLOWED_TYPES:
        return False

    if (
        ticker.primary_exchange
        not in ALLOWED_PRIMARY_EXCHANGES
    ):
        return False

    return True


def _fetch_active_tickers(
    *,
    as_of_date: date,
) -> list[dict[str, Any]]:
    url = (
        f"{BASE_URL}"
        f"/v3/reference/tickers"
    )

    params = {
        "market": "stocks",
        "locale": "us",
        "date": as_of_date.isoformat(),
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
        "apiKey": MASSIVE_API_KEY,
    }

    results: list[
        dict[str, Any]
    ] = []

    seen_urls: set[str] = set()

    max_pages = 20

    for _ in range(max_pages):
        if url in seen_urls:
            raise RuntimeError(
                "Historical universe pagination "
                "loop detected."
            )

        seen_urls.add(url)

        response = httpx.get(
            url,
            params=params,
            timeout=60.0,
        )

        response.raise_for_status()

        payload = response.json()

        page_results = (
            payload.get(
                "results",
                [],
            )
            or []
        )

        results.extend(
            page_results
        )

        next_url = payload.get(
            "next_url"
        )

        if not next_url:
            break

        if "apiKey=" not in next_url:
            separator = (
                "&"
                if "?" in next_url
                else "?"
            )

            next_url = (
                f"{next_url}"
                f"{separator}"
                f"apiKey="
                f"{MASSIVE_API_KEY}"
            )

        url = next_url

        # Massive next_url already contains
        # cursor and original query state.
        params = None

    else:
        raise RuntimeError(
            "Historical universe pagination "
            f"exceeded {max_pages} pages."
        )

    return results


def get_historical_universe(
    as_of_date: str | date | datetime,
    *,
    force_refresh: bool = False,
) -> HistoricalUniverseResult:
    """
    Return stocks actively traded on the requested
    historical date.

    The Massive point-in-time query is authoritative.

    We intentionally request active=true only.
    """

    normalized_date = _normalize_date(
        as_of_date
    )

    cached = None

    if not force_refresh:
        cached = _load_cache(
            normalized_date
        )

    fetched_from_cache = (
        cached is not None
    )

    if cached is None:
        raw_tickers = (
            _fetch_active_tickers(
                as_of_date=normalized_date,
            )
        )

        payload = {
            "as_of_date": (
                normalized_date.isoformat()
            ),
            "active_only": True,
            "tickers": raw_tickers,
        }

        _save_cache(
            normalized_date,
            payload,
        )

    else:
        payload = cached

        # Reject caches from the older
        # active+inactive implementation.
        if (
            payload.get("active_only")
            is not True
        ):
            raw_tickers = (
                _fetch_active_tickers(
                    as_of_date=normalized_date,
                )
            )

            payload = {
                "as_of_date": (
                    normalized_date.isoformat()
                ),
                "active_only": True,
                "tickers": raw_tickers,
            }

            _save_cache(
                normalized_date,
                payload,
            )

            fetched_from_cache = False

    raw_tickers = (
        payload.get(
            "tickers",
            [],
        )
        or []
    )

    parsed = [
        _parse_ticker(raw)
        for raw in raw_tickers
    ]

    eligible = [
        ticker
        for ticker in parsed
        if _eligible_ticker(ticker)
    ]

    deduped: dict[
        tuple[str, str | None],
        HistoricalTicker,
    ] = {}

    for ticker in eligible:
        identity = (
            ticker.ticker,
            ticker.composite_figi,
        )

        if identity not in deduped:
            deduped[
                identity
            ] = ticker

    final_tickers = sorted(
        deduped.values(),
        key=lambda item: (
            item.ticker,
            item.composite_figi
            or "",
        ),
    )

    return HistoricalUniverseResult(
        as_of_date=normalized_date,
        tickers=final_tickers,
        total_count=len(
            final_tickers
        ),
        fetched_from_cache=(
            fetched_from_cache
        ),
    )


def get_rolling_historical_universe(
    session_dates: Iterable[
        str | date | datetime
    ],
    *,
    force_refresh: bool = False,
) -> RollingUniverseResult:
    """
    Build point-in-time universe membership across
    the supplied market sessions.

    IMPORTANT:
    Every supplied session is independently resolved
    through get_historical_universe().

    This intentionally favors correctness over clever
    inference. Local caching makes subsequent research
    runs inexpensive.

    No ticker is allowed into an earlier session just
    because it exists in a later snapshot.
    """

    normalized_dates = sorted(
        {
            _normalize_date(value)
            for value in session_dates
        }
    )

    if not normalized_dates:
        raise ValueError(
            "session_dates cannot be empty."
        )

    snapshots: dict[
        str,
        list[HistoricalTicker],
    ] = {}

    first_seen: dict[
        str,
        str,
    ] = {}

    last_seen: dict[
        str,
        str,
    ] = {}

    securities: dict[
        str,
        HistoricalTicker,
    ] = {}

    membership_dates_mutable: dict[
        str,
        set[str],
    ] = {}

    fetched_snapshots = 0
    cached_snapshots = 0

    total = len(
        normalized_dates
    )

    for index, session_date in enumerate(
        normalized_dates,
        start=1,
    ):
        result = get_historical_universe(
            as_of_date=session_date,
            force_refresh=force_refresh,
        )

        if result.fetched_from_cache:
            cached_snapshots += 1
            source = "CACHE"
        else:
            fetched_snapshots += 1
            source = "API"

        session_key = (
            session_date.isoformat()
        )

        snapshots[
            session_key
        ] = result.tickers

        for security in result.tickers:
            ticker = security.ticker

            membership_dates_mutable.setdefault(
                ticker,
                set(),
            ).add(
                session_key
            )

            securities[
                ticker
            ] = security

            if ticker not in first_seen:
                first_seen[
                    ticker
                ] = session_key

            last_seen[
                ticker
            ] = session_key

        print(
            f"  Universe "
            f"[{index:>3}/{total:<3}] "
            f"{session_key} "
            f"{len(result.tickers):>5,} "
            f"{source}"
        )

    membership_dates_by_ticker = {
        ticker: frozenset(dates)
        for ticker, dates
        in membership_dates_mutable.items()
    }

    return RollingUniverseResult(
        snapshots=snapshots,
        first_seen=first_seen,
        last_seen=last_seen,
        securities=securities,
        fetched_snapshots=(
            fetched_snapshots
        ),
        cached_snapshots=(
            cached_snapshots
        ),
        membership_dates_by_ticker=(
            membership_dates_by_ticker
        ),
    )


def ticker_is_member_on_date(
    *,
    ticker: str,
    session_date: str | date | datetime,
    rolling_universe: RollingUniverseResult,
) -> bool:
    """
    Exact point-in-time membership lookup.
    """

    normalized = _normalize_date(
        session_date
    ).isoformat()

    securities = (
        rolling_universe.snapshots.get(
            normalized
        )
    )

    if securities is None:
        raise KeyError(
            "No universe snapshot exists for "
            f"{normalized}."
        )

    target = ticker.upper().strip()

    return any(
        security.ticker == target
        for security in securities
    )