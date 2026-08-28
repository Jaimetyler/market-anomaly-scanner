from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


@dataclass
class CorporateActionEvent:
    event_type: str
    execution_date: date
    split_from: float | None = None
    split_to: float | None = None
    ticker: str | None = None
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
    """
    Classify a split from Massive-style split terms.

    Examples:

        1 -> 5
        shareholder receives 5 shares for each 1
        = FORWARD_SPLIT

        10 -> 1
        ten old shares become one new share
        = REVERSE_SPLIT

        1 -> 1
        = NEUTRAL_SPLIT
    """

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
    """
    Return the share-count adjustment factor.

    1-for-5 style forward split:
        1 -> 5 = 5.0

    10-for-1 reverse split:
        10 -> 1 = 0.1
    """

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
    """
    Convert a split record into our internal format.

    Accepts common field names so the research
    layer does not depend tightly on one API
    response representation.
    """

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

    event_type = classify_split(
        split_from=split_from,
        split_to=split_to,
    )

    return CorporateActionEvent(
        event_type=event_type,
        execution_date=_normalize_date(
            execution_date
        ),
        split_from=float(split_from),
        split_to=float(split_to),
        ticker=raw.get("ticker"),
        raw=raw,
    )


def find_nearby_actions(
    signal_date: str | date | datetime,
    actions: list[CorporateActionEvent],
    lookback_days: int = 10,
    lookforward_days: int = 3,
) -> list[CorporateActionEvent]:
    """
    Find corporate actions near a signal.

    We deliberately allow a small look-forward
    window ONLY for research-quality diagnostics.

    IMPORTANT:
    A future corporate action found here must never
    be used as an input feature for a historical
    trading signal.

    This function is for contamination/sanity checks,
    not signal generation.
    """

    signal_date = _normalize_date(
        signal_date
    )

    start = (
        signal_date
        - timedelta(days=lookback_days)
    )

    end = (
        signal_date
        + timedelta(days=lookforward_days)
    )

    return sorted(
        [
            action
            for action in actions
            if start
            <= action.execution_date
            <= end
        ],
        key=lambda action: action.execution_date,
    )


def check_corporate_action_risk(
    signal_date: str | date | datetime,
    actions: list[CorporateActionEvent],
    return_1d: float | None = None,
    return_5d: float | None = None,
    return_20d: float | None = None,
) -> CorporateActionCheck:
    """
    Determine whether a historical anomaly may be
    contaminated by a corporate action.

    V1 policy:

    - Nearby reverse split:
        flag and exclude from clean research sample.

    - Nearby forward split:
        flag, but do not automatically exclude.

    - Extremely large observed return:
        flag for corporate-action verification.

    This is intentionally conservative.
    """

    nearby = find_nearby_actions(
        signal_date=signal_date,
        actions=actions,
    )

    flags = []
    reasons = []
    exclude = False

    for action in nearby:
        if action.event_type == "REVERSE_SPLIT":
            if "RECENT_REVERSE_SPLIT" not in flags:
                flags.append(
                    "RECENT_REVERSE_SPLIT"
                )

            reasons.append(
                "A reverse split occurred near "
                "the anomaly signal."
            )

            exclude = True

        elif action.event_type == "FORWARD_SPLIT":
            if "RECENT_FORWARD_SPLIT" not in flags:
                flags.append(
                    "RECENT_FORWARD_SPLIT"
                )

            reasons.append(
                "A forward split occurred near "
                "the anomaly signal."
            )

    extreme_move = any(
        value is not None
        and abs(float(value)) >= threshold
        for value, threshold in [
            (return_1d, 100),
            (return_5d, 300),
            (return_20d, 500),
        ]
    )

    if extreme_move:
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