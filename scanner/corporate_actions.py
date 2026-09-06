from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
import json

import httpx

from scanner.config import MASSIVE_API_KEY
from scanner.data import BASE_URL


CACHE_DIR = Path("data/corporate_actions")


@dataclass
class CorporateActionEvent:
    event_type: str
    execution_date: date
    split_from: float | None = None
    split_to: float | None = None
    ticker: str | None = None
    historical_adjustment_factor: float | None = None
    raw: dict[str, Any] | None = None


@dataclass
class CorporateActionCheck:
    flagged: bool
    exclude_from_research: bool
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    nearby_actions: list[CorporateActionEvent] = field(
        default_factory=list
    )


def _normalize_date(
    value: str | date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(value)


def classify_split(
    split_from: float,
    split_to: float,
) -> str:
    split_from = float(split_from)
    split_to = float(split_to)

    if split_from <= 0 or split_to <= 0:
        raise ValueError(
            "Split values must be greater than zero."
        )

    ratio = split_to / split_from

    if ratio > 1:
        return "FORWARD_SPLIT"

    if ratio < 1:
        return "REVERSE_SPLIT"

    return "NEUTRAL_SPLIT"


def split_adjustment_factor(
    split_from: float,
    split_to: float,
) -> float:
    split_from = float(split_from)
    split_to = float(split_to)

    if split_from <= 0 or split_to <= 0:
        raise ValueError(
            "Split values must be greater than zero."
        )

    return split_to / split_from


def parse_split_event(
    raw: dict[str, Any],
) -> CorporateActionEvent:
    execution_date = (
        raw.get("execution_date")
        or raw.get("ex_date")
        or raw.get("date")
    )

    if execution_date is None:
        raise ValueError(
            "Split event is missing execution date."
        )

    split_from = raw.get("split_from")
    split_to = raw.get("split_to")

    if split_from is None or split_to is None:
        raise ValueError(
            "Split event must contain split_from "
            "and split_to."
        )

    api_adjustment_type = raw.get(
        "adjustment_type"
    )

    if api_adjustment_type == "reverse_split":
        event_type = "REVERSE_SPLIT"

    elif api_adjustment_type == "forward_split":
        event_type = "FORWARD_SPLIT"

    elif api_adjustment_type == "stock_dividend":
        event_type = "STOCK_DIVIDEND"

    else:
        event_type = classify_split(
            split_from=split_from,
            split_to=split_to,
        )

    historical_factor = raw.get(
        "historical_adjustment_factor"
    )

    if historical_factor is not None:
        historical_factor = float(
            historical_factor
        )

    return CorporateActionEvent(
        event_type=event_type,
        execution_date=_normalize_date(
            execution_date
        ),
        split_from=float(split_from),
        split_to=float(split_to),
        ticker=raw.get("ticker"),
        historical_adjustment_factor=(
            historical_factor
        ),
        raw=raw,
    )


def _cache_path(
    ticker: str,
) -> Path:
    ticker = ticker.upper().strip()

    return (
        CACHE_DIR
        / f"{ticker}_splits.json"
    )


def _load_split_cache(
    ticker: str,
) -> list[dict[str, Any]] | None:
    path = _cache_path(
        ticker
    )

    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

        if not isinstance(
            payload,
            list,
        ):
            return None

        return payload

    except Exception:
        return None


def _save_split_cache(
    ticker: str,
    splits: list[dict[str, Any]],
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = _cache_path(
        ticker
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            splits,
            f,
            indent=2,
        )


def get_stock_splits(
    ticker: str,
    force_refresh: bool = False,
) -> list[CorporateActionEvent]:
    """
    Retrieve all known Massive split events for
    one ticker.

    Uses the current:
        GET /stocks/v1/splits

    Results are cached locally so historical
    replays do not repeatedly hit the endpoint.
    """

    ticker = (
        ticker
        .upper()
        .strip()
    )

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    if not force_refresh:
        cached = _load_split_cache(
            ticker
        )

        if cached is not None:
            return [
                parse_split_event(
                    item
                )
                for item in cached
            ]

    url = (
        f"{BASE_URL}"
        "/stocks/v1/splits"
    )

    params = {
        "ticker": ticker,
        "limit": 1000,
        "sort": "execution_date.asc",
        "apiKey": MASSIVE_API_KEY,
    }

    raw_splits = []

    # Hard bound protects us from another
    # accidental pagination nightmare.
    max_pages = 20
    page = 0

    with httpx.Client(
        timeout=60.0
    ) as client:

        while url:
            page += 1

            if page > max_pages:
                raise RuntimeError(
                    "Split pagination exceeded "
                    f"{max_pages} pages for {ticker}."
                )

            response = client.get(
                url,
                params=params,
            )

            response.raise_for_status()

            payload = response.json()

            status = payload.get(
                "status"
            )

            if status not in {
                "OK",
                "DELAYED",
            }:
                raise RuntimeError(
                    "Massive returned an unexpected "
                    f"split response for {ticker}: "
                    f"{payload}"
                )

            results = payload.get(
                "results",
                [],
            )

            raw_splits.extend(
                results
            )

            next_url = payload.get(
                "next_url"
            )

            if not next_url:
                break

            # Massive next_url may omit the key.
            separator = (
                "&"
                if "?" in next_url
                else "?"
            )

            if "apiKey=" not in next_url:
                next_url = (
                    f"{next_url}"
                    f"{separator}"
                    f"apiKey={MASSIVE_API_KEY}"
                )

            url = next_url
            params = None

    _save_split_cache(
        ticker,
        raw_splits,
    )

    return [
        parse_split_event(
            item
        )
        for item in raw_splits
    ]


def find_nearby_actions(
    signal_date: str | date | datetime,
    actions: list[CorporateActionEvent],
    lookback_days: int = 10,
    lookforward_days: int = 3,
) -> list[CorporateActionEvent]:
    signal_date = _normalize_date(
        signal_date
    )

    start = (
        signal_date
        - timedelta(
            days=lookback_days
        )
    )

    end = (
        signal_date
        + timedelta(
            days=lookforward_days
        )
    )

    return sorted(
        [
            action
            for action in actions
            if (
                start
                <= action.execution_date
                <= end
            )
        ],
        key=lambda action: (
            action.execution_date
        ),
    )


def check_corporate_action_risk(
    signal_date: str | date | datetime,
    actions: list[CorporateActionEvent],
    return_1d: float | None = None,
    return_5d: float | None = None,
    return_20d: float | None = None,
) -> CorporateActionCheck:
    nearby = find_nearby_actions(
        signal_date=signal_date,
        actions=actions,
    )

    flags = []
    reasons = []
    exclude = False

    for action in nearby:

        if (
            action.event_type
            == "REVERSE_SPLIT"
        ):
            if (
                "RECENT_REVERSE_SPLIT"
                not in flags
            ):
                flags.append(
                    "RECENT_REVERSE_SPLIT"
                )

            reasons.append(
                "A reverse split occurred near "
                "the anomaly signal."
            )

            exclude = True

        elif (
            action.event_type
            == "FORWARD_SPLIT"
        ):
            if (
                "RECENT_FORWARD_SPLIT"
                not in flags
            ):
                flags.append(
                    "RECENT_FORWARD_SPLIT"
                )

            reasons.append(
                "A forward split occurred near "
                "the anomaly signal."
            )

        elif (
            action.event_type
            == "STOCK_DIVIDEND"
        ):
            if (
                "RECENT_STOCK_DIVIDEND"
                not in flags
            ):
                flags.append(
                    "RECENT_STOCK_DIVIDEND"
                )

            reasons.append(
                "A stock dividend occurred near "
                "the anomaly signal."
            )

    extreme_move = any(
        value is not None
        and abs(float(value)) >= threshold
        for value, threshold in [
            (
                return_1d,
                100,
            ),
            (
                return_5d,
                300,
            ),
            (
                return_20d,
                500,
            ),
        ]
    )

    if extreme_move:
        if (
            "EXTREME_MOVE_NEEDS_CA_CHECK"
            not in flags
        ):
            flags.append(
                "EXTREME_MOVE_NEEDS_CA_CHECK"
            )

        reasons.append(
            "Observed return is extreme enough "
            "to require corporate-action validation."
        )

    return CorporateActionCheck(
        flagged=bool(flags),
        exclude_from_research=exclude,
        flags=flags,
        reasons=reasons,
        nearby_actions=nearby,
    )

# ============================================================================
# CASH DIVIDENDS
# ============================================================================


@dataclass(frozen=True)
class DividendEvent:
    ticker: str
    ex_dividend_date: date
    cash_amount: float | None = None
    split_adjusted_cash_amount: float | None = None
    distribution_type: str | None = None
    historical_adjustment_factor: float | None = None
    declaration_date: date | None = None
    record_date: date | None = None
    pay_date: date | None = None
    raw: dict[str, Any] | None = None


def _optional_date(
    value: Any,
) -> date | None:
    if value in (
        None,
        "",
    ):
        return None

    return _normalize_date(
        value
    )


def parse_dividend_event(
    raw: dict[str, Any],
) -> DividendEvent:
    ex_date = raw.get(
        "ex_dividend_date"
    )

    if ex_date is None:
        raise ValueError(
            "Dividend event is missing ex_dividend_date."
        )

    cash_amount = raw.get(
        "cash_amount"
    )

    split_adjusted = raw.get(
        "split_adjusted_cash_amount"
    )

    historical_factor = raw.get(
        "historical_adjustment_factor"
    )

    return DividendEvent(
        ticker=str(
            raw.get(
                "ticker",
                "",
            )
        ),
        ex_dividend_date=(
            _normalize_date(
                ex_date
            )
        ),
        cash_amount=(
            float(cash_amount)
            if cash_amount is not None
            else None
        ),
        split_adjusted_cash_amount=(
            float(split_adjusted)
            if split_adjusted is not None
            else None
        ),
        distribution_type=(
            str(
                raw[
                    "distribution_type"
                ]
            )
            if raw.get(
                "distribution_type"
            )
            is not None
            else None
        ),
        historical_adjustment_factor=(
            float(
                historical_factor
            )
            if historical_factor
            is not None
            else None
        ),
        declaration_date=(
            _optional_date(
                raw.get(
                    "declaration_date"
                )
            )
        ),
        record_date=(
            _optional_date(
                raw.get(
                    "record_date"
                )
            )
        ),
        pay_date=(
            _optional_date(
                raw.get(
                    "pay_date"
                )
            )
        ),
        raw=raw,
    )


def _dividend_cache_path(
    ticker: str,
) -> Path:
    safe_ticker = (
        ticker
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

    return (
        CACHE_DIR
        / f"{safe_ticker}_dividends.json"
    )


def _load_dividend_cache(
    ticker: str,
) -> list[dict[str, Any]] | None:
    path = (
        _dividend_cache_path(
            ticker
        )
    )

    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(
                handle
            )

        if not isinstance(
            payload,
            list,
        ):
            return None

        return payload

    except Exception:
        return None


def _save_dividend_cache(
    ticker: str,
    dividends: list[
        dict[str, Any]
    ],
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        _dividend_cache_path(
            ticker
        )
    )

    temp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with temp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            dividends,
            handle,
            indent=2,
        )

    temp.replace(
        path
    )


def get_stock_dividends(
    ticker: str,
    force_refresh: bool = False,
) -> list[DividendEvent]:
    """
    Retrieve Massive cash-dividend records for one exact-case ticker.

    Endpoint:
        GET /stocks/v1/dividends

    Unlike the legacy split helper above, this intentionally preserves
    provider ticker case.
    """

    ticker = ticker.strip()

    if not ticker:
        raise ValueError(
            "Ticker cannot be empty."
        )

    if not force_refresh:
        cached = (
            _load_dividend_cache(
                ticker
            )
        )

        if cached is not None:
            return [
                parse_dividend_event(
                    item
                )
                for item in cached
            ]

    url = (
        f"{BASE_URL}"
        "/stocks/v1/dividends"
    )

    params: dict[
        str,
        Any,
    ] = {
        "ticker": ticker,
        "limit": 1000,
        "sort": (
            "ex_dividend_date.asc"
        ),
        "apiKey": MASSIVE_API_KEY,
    }

    raw_events: list[
        dict[str, Any]
    ] = []

    with httpx.Client(
        timeout=30.0,
    ) as client:
        next_url: (
            str
            | None
        ) = url

        first_page = True

        while next_url:
            if first_page:
                response = client.get(
                    next_url,
                    params=params,
                )

                first_page = False

            else:
                response = client.get(
                    next_url,
                    params={
                        "apiKey": (
                            MASSIVE_API_KEY
                        )
                    },
                )

            response.raise_for_status()

            payload = (
                response.json()
            )

            results = payload.get(
                "results",
                [],
            )

            if isinstance(
                results,
                list,
            ):
                raw_events.extend(
                    item
                    for item in results
                    if isinstance(
                        item,
                        dict,
                    )
                )

            next_url = payload.get(
                "next_url"
            )

    _save_dividend_cache(
        ticker,
        raw_events,
    )

    return [
        parse_dividend_event(
            item
        )
        for item in raw_events
    ]
