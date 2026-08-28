import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from scanner.config import MASSIVE_API_KEY
from scanner.data import BASE_URL


CACHE_PATH = Path("data/universe.json")


def _load_cache() -> dict[str, dict[str, Any]] | None:
    if not CACHE_PATH.exists():
        return None

    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(
    universe: dict[str, dict[str, Any]],
) -> None:
    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            universe,
            f,
            indent=2,
        )


def _add_api_key(url: str) -> str:
    """
    Add our API key to a Massive next_url without changing
    its cursor or any other pagination parameters.
    """

    parts = urlsplit(url)

    query = dict(
        parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
    )

    query["apiKey"] = MASSIVE_API_KEY

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def get_active_stock_universe(
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Fetch active U.S. common stocks and cache them locally.

    Massive's server-side type=CS filter keeps ETFs,
    warrants, preferreds, funds, etc. out of this universe.
    """

    if not force_refresh:
        cached = _load_cache()

        if cached:
            return cached

    first_url = (
        f"{BASE_URL}"
        "/v3/reference/tickers"
    )

    first_params = {
        "market": "stocks",
        "type": "CS",
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
        "apiKey": MASSIVE_API_KEY,
    }

    universe: dict[str, dict[str, Any]] = {}

    seen_urls: set[str] = set()

    page = 0
    url = first_url
    params = first_params

    with httpx.Client(timeout=60.0) as client:

        while url:
            page += 1

            if page > 100:
                raise RuntimeError(
                    "Universe pagination exceeded 100 pages. "
                    "Stopping because this almost certainly "
                    "indicates a pagination loop."
                )

            print(
                f"Fetching common-stock universe page {page}..."
            )

            response = client.get(
                url,
                params=params,
            )

            response.raise_for_status()

            payload = response.json()

            results = payload.get(
                "results",
                [],
            )

            if not results:
                break

            first_ticker = results[0].get("ticker")
            last_ticker = results[-1].get("ticker")

            print(
                f"  {len(results):,} records: "
                f"{first_ticker} -> {last_ticker}"
            )

            for item in results:
                ticker = item.get("ticker")

                if not ticker:
                    continue

                universe[ticker] = item

            next_url = payload.get("next_url")

            if not next_url:
                break

            next_url = _add_api_key(
                next_url
            )

            if next_url in seen_urls:
                raise RuntimeError(
                    "Massive returned a repeated next_url. "
                    "Pagination loop detected."
                )

            seen_urls.add(
                next_url
            )

            # IMPORTANT:
            # next_url contains the cursor.
            # Do not send our original filters again.
            url = next_url
            params = None

    if not universe:
        raise RuntimeError(
            "Massive returned no active common stocks."
        )

    _save_cache(
        universe
    )

    print()
    print(
        f"Cached {len(universe):,} active common stocks."
    )

    return universe