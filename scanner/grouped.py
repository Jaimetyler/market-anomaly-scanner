from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from scanner.config import MASSIVE_API_KEY
from scanner.data import BASE_URL


def get_grouped_daily_bars(
    session_date: str,
) -> list[dict[str, Any]]:
    """
    Fetch one full-market daily OHLCV snapshot for a trading day.
    """

    url = (
        f"{BASE_URL}"
        "/v2/aggs/grouped/locale/us/market/stocks/"
        f"{session_date}"
    )

    params = {
        "adjusted": "true",
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
            "Massive returned an unexpected grouped "
            f"daily response for {session_date}: {payload}"
        )

    return payload.get(
        "results",
        [],
    )


def get_recent_grouped_sessions(
    trading_sessions: int = 25,
    calendar_lookback_days: int = 45,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fetch enough recent calendar days to collect a desired number
    of actual trading sessions.

    Returns:
        {
            "2026-08-27": [...],
            "2026-08-26": [...],
            ...
        }
    """

    if trading_sessions <= 0:
        raise ValueError(
            "trading_sessions must be greater than zero."
        )

    today = date.today()

    sessions: dict[str, list[dict[str, Any]]] = {}

    for offset in range(calendar_lookback_days + 1):
        day = today - timedelta(days=offset)

        if day.weekday() >= 5:
            continue

        day_text = day.isoformat()

        try:
            bars = get_grouped_daily_bars(
                day_text
            )

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code

            # Holidays / no-data days can occur.
            if status in {
                404,
            }:
                continue

            raise

        if not bars:
            continue

        sessions[day_text] = bars

        if len(sessions) >= trading_sessions:
            break

    return dict(
        sorted(
            sessions.items(),
            key=lambda item: item[0],
        )
    )


def grouped_sessions_to_dataframe(
    sessions: dict[str, list[dict[str, Any]]],
    eligible_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Convert grouped-market results into a normalized dataframe.

    One row = one ticker/session.
    """

    rows: list[dict[str, Any]] = []

    for session_date, bars in sessions.items():
        for bar in bars:
            ticker = bar.get("T")

            if not ticker:
                continue

            if (
                eligible_symbols is not None
                and ticker not in eligible_symbols
            ):
                continue

            rows.append(
                {
                    "ticker": ticker,
                    "session_date": session_date,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "transactions": bar.get("n"),
                    "vwap": bar.get("vw"),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "session_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "transactions",
                "vwap",
            ]
        )

    df = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "transactions",
        "vwap",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df["session_date"] = pd.to_datetime(
        df["session_date"]
    )

    df = (
        df.dropna(
            subset=[
                "ticker",
                "session_date",
                "close",
                "volume",
            ]
        )
        .sort_values(
            [
                "ticker",
                "session_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


def _percent_change(
    current: float,
    previous: float,
) -> float | None:
    if previous == 0:
        return None

    return (
        (current / previous) - 1
    ) * 100


def build_grouped_prefilter(
    df: pd.DataFrame,
    min_price: float = 1.0,
    min_dollar_volume: float = 2_000_000,
) -> list[dict[str, Any]]:
    """
    Build a broad anomaly candidate list using only recent
    grouped-market data.

    This is intentionally broader than the deep detector.

    It catches:
    - fresh one-day spikes
    - multi-day acceleration
    - post-spike extension
    - abnormal volume
    """

    candidates: list[dict[str, Any]] = []

    if df.empty:
        return candidates

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):
        group = (
            group.sort_values(
                "session_date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(group) < 6:
            continue

        latest = group.iloc[-1]

        price = float(
            latest["close"]
        )

        volume = float(
            latest["volume"]
        )

        dollar_volume = (
            price * volume
        )

        if price < min_price:
            continue

        if dollar_volume < min_dollar_volume:
            continue

        closes = group["close"]
        volumes = group["volume"]

        def session_return(
            sessions_back: int,
        ) -> float | None:
            if len(closes) <= sessions_back:
                return None

            return _percent_change(
                float(closes.iloc[-1]),
                float(
                    closes.iloc[
                        -1 - sessions_back
                    ]
                ),
            )

        return_1d = session_return(1)
        return_3d = session_return(3)
        return_5d = session_return(5)
        return_10d = session_return(10)
        return_20d = session_return(20)

        avg_volume_20d = None
        relative_volume = None

        prior_volumes = (
            volumes.iloc[:-1]
            .dropna()
            .tail(20)
        )

        if len(prior_volumes) >= 10:
            avg_volume_20d = float(
                prior_volumes.mean()
            )

            if avg_volume_20d > 0:
                relative_volume = (
                    volume
                    / avg_volume_20d
                )

        triggers: list[str] = []

        if (
            return_1d is not None
            and return_1d >= 10
        ):
            triggers.append(
                "1D_MOVE"
            )

        if (
            return_3d is not None
            and return_3d >= 20
        ):
            triggers.append(
                "3D_MOVE"
            )

        if (
            return_5d is not None
            and return_5d >= 25
        ):
            triggers.append(
                "5D_MOVE"
            )

        if (
            return_10d is not None
            and return_10d >= 40
        ):
            triggers.append(
                "10D_MOVE"
            )

        if (
            return_20d is not None
            and return_20d >= 60
        ):
            triggers.append(
                "20D_MOVE"
            )

        if (
            relative_volume is not None
            and relative_volume >= 3
        ):
            triggers.append(
                "RVOL"
            )

        if not triggers:
            continue

        candidates.append(
            {
                "ticker": ticker,
                "session_date": (
                    latest[
                        "session_date"
                    ].date().isoformat()
                ),
                "price": price,
                "volume": volume,
                "dollar_volume": dollar_volume,
                "return_1d": return_1d,
                "return_3d": return_3d,
                "return_5d": return_5d,
                "return_10d": return_10d,
                "return_20d": return_20d,
                "avg_volume_20d": avg_volume_20d,
                "relative_volume": relative_volume,
                "prefilter_triggers": triggers,
            }
        )

    def ranking_value(
        item: dict[str, Any],
    ) -> float:
        score = 0.0

        score += max(
            item.get("return_1d")
            or 0,
            0,
        )

        score += max(
            item.get("return_3d")
            or 0,
            0,
        ) * 0.8

        score += max(
            item.get("return_5d")
            or 0,
            0,
        ) * 0.6

        score += max(
            item.get("return_10d")
            or 0,
            0,
        ) * 0.3

        score += max(
            item.get("return_20d")
            or 0,
            0,
        ) * 0.15

        score += min(
            item.get("relative_volume")
            or 0,
            25,
        ) * 3

        return score

    candidates.sort(
        key=ranking_value,
        reverse=True,
    )

    return candidates