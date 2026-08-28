from dataclasses import dataclass
from typing import Any

import httpx

from scanner.config import MASSIVE_API_KEY
from scanner.data import BASE_URL


@dataclass
class MarketCandidate:
    ticker: str
    price: float
    volume: float
    dollar_volume: float
    day_change_pct: float | None
    snapshot: dict[str, Any]


def get_full_market_snapshot() -> list[dict[str, Any]]:
    """
    Fetch the Massive full U.S. stocks snapshot.

    This is the cheap first-stage market scan.
    """

    url = (
        f"{BASE_URL}"
        "/v2/snapshot/locale/us/markets/stocks/tickers"
    )

    params = {
        "include_otc": "false",
        "apiKey": MASSIVE_API_KEY,
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.get(
            url,
            params=params,
        )

        response.raise_for_status()

        payload = response.json()

    if payload.get("status") not in {
        "OK",
        "DELAYED",
    }:
        raise RuntimeError(
            "Massive returned an unexpected "
            f"market snapshot response: {payload}"
        )

    return payload.get(
        "tickers",
        [],
    )


def build_prefilter_candidates(
    snapshots: list[dict[str, Any]],
    min_price: float = 1.0,
    min_dollar_volume: float = 2_000_000,
    min_day_change_pct: float = 5.0,
) -> list[MarketCandidate]:
    """
    Cheap market-wide prefilter.

    Passing this does NOT mean the stock is an anomaly.

    It only means the stock is sufficiently liquid and
    active today to justify downloading deeper history.
    """

    candidates: list[MarketCandidate] = []

    for item in snapshots:
        ticker = item.get(
            "ticker"
        )

        day = (
            item.get("day")
            or {}
        )

        price = day.get(
            "c"
        )

        volume = day.get(
            "v"
        )

        day_change_pct = item.get(
            "todaysChangePerc"
        )

        if (
            not ticker
            or price is None
            or volume is None
            or day_change_pct is None
        ):
            continue

        try:
            price = float(
                price
            )

            volume = float(
                volume
            )

            day_change_pct = float(
                day_change_pct
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if price < min_price:
            continue

        dollar_volume = (
            price * volume
        )

        if (
            dollar_volume
            < min_dollar_volume
        ):
            continue

        # V0.1 currently hunts upside anomalies.
        #
        # Keep this deliberately broad so we do not
        # throw away potentially interesting names
        # before doing the deeper historical analysis.
        if (
            day_change_pct
            < min_day_change_pct
        ):
            continue

        candidates.append(
            MarketCandidate(
                ticker=ticker,
                price=price,
                volume=volume,
                dollar_volume=dollar_volume,
                day_change_pct=day_change_pct,
                snapshot=item,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.day_change_pct
            if candidate.day_change_pct is not None
            else -999.0
        ),
        reverse=True,
    )

    return candidates