from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Sequence

from scanner.corporate_actions import DividendEvent
from scanner.turtle import bar_close, bar_open


DEFAULT_DISCONTINUITY_THRESHOLD = 0.35


# Explicitly researched history boundaries not captured cleanly by the
# automated corporate-action feeds.
#
# These are NOT inferred from price movement.
#
# AABA:
# 2019-09-24 followed Altaba's confirmed $51.50/share liquidating
# distribution and represents a non-market price-history discontinuity.
#
# AAN:
# 2020-12-01 crosses a confirmed corporate-separation / security-lineage
# boundary.
#
# AAC:
# 2021-03-25 begins a different security lineage under the AAC ticker.
# The earlier 2019 history belongs to a different security and must not
# be treated as continuous price history.
#
# Keep this registry intentionally small and auditable.
CONFIRMED_LINEAGE_BREAKS: dict[str, frozenset[str]] = {
    "AABA": frozenset(
        {
            "2019-09-24",
        }
    ),
    "AAN": frozenset(
        {
            "2020-12-01",
        }
    ),
    "AAC": frozenset(
        {
            "2021-03-25",
        }
    ),
}


@dataclass(frozen=True)
class TurtleHistoryCandidate:
    break_index: int
    previous_close: float
    new_open: float
    open_gap_pct: float
    previous_date: str | None
    new_date: str | None
    candidate_reason: str


@dataclass(frozen=True)
class TurtleHistoryBreak:
    break_index: int
    previous_close: float
    new_open: float
    open_gap_pct: float
    previous_date: str | None
    new_date: str | None
    reason: str
    evidence: str


@dataclass(frozen=True)
class TurtleHistorySegment:
    start_index: int
    end_index: int
    bars: tuple[Any, ...]

    @property
    def length(self) -> int:
        return len(self.bars)


def _bar_date(bar: Any) -> str | None:
    if isinstance(bar, dict):
        for key in (
            "session_date",
            "date",
            "signal_date",
        ):
            value = bar.get(key)

            if value is not None:
                if hasattr(value, "isoformat"):
                    return value.isoformat()

                return str(value)

        return None

    for key in (
        "session_date",
        "date",
        "signal_date",
    ):
        if hasattr(bar, key):
            value = getattr(bar, key)

            if value is not None:
                if hasattr(value, "isoformat"):
                    return value.isoformat()

                return str(value)

    return None


def _date_string(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def overnight_gap_pct(
    previous_bar: Any,
    current_bar: Any,
) -> float:
    previous_close = bar_close(previous_bar)
    current_open = bar_open(current_bar)

    if previous_close <= 0:
        raise ValueError(
            "previous close must be positive"
        )

    return (
        current_open / previous_close - 1.0
    ) * 100.0


def is_large_gap_candidate(
    previous_bar: Any,
    current_bar: Any,
    *,
    threshold: float = DEFAULT_DISCONTINUITY_THRESHOLD,
) -> bool:
    """
    Detect a suspiciously large overnight gap.

    IMPORTANT:
    A large gap is only a candidate for investigation.
    It is not automatically treated as a history break.
    """

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    previous_close = bar_close(previous_bar)
    current_open = bar_open(current_bar)

    if previous_close <= 0 or current_open <= 0:
        return True

    ratio = current_open / previous_close

    return (
        ratio <= 1.0 - threshold
        or ratio >= 1.0 + threshold
    )


def find_discontinuity_candidates(
    bars: Sequence[Any],
    *,
    threshold: float = DEFAULT_DISCONTINUITY_THRESHOLD,
) -> list[TurtleHistoryCandidate]:
    """
    Find large-gap boundaries that deserve corporate-action review.

    No history is removed here.
    """

    candidates: list[TurtleHistoryCandidate] = []

    for index in range(1, len(bars)):
        previous = bars[index - 1]
        current = bars[index]

        if not is_large_gap_candidate(
            previous,
            current,
            threshold=threshold,
        ):
            continue

        candidates.append(
            TurtleHistoryCandidate(
                break_index=index,
                previous_close=bar_close(previous),
                new_open=bar_open(current),
                open_gap_pct=overnight_gap_pct(
                    previous,
                    current,
                ),
                previous_date=_bar_date(previous),
                new_date=_bar_date(current),
                candidate_reason=(
                    "LARGE_OVERNIGHT_GAP"
                ),
            )
        )

    return candidates


def _dividend_cash_amount(
    event: DividendEvent,
) -> float | None:
    if (
        event.split_adjusted_cash_amount
        is not None
    ):
        return float(
            event.split_adjusted_cash_amount
        )

    if event.cash_amount is not None:
        return float(event.cash_amount)

    return None


def _matching_dividends(
    dividends: Iterable[DividendEvent],
    *,
    session_date: str,
) -> list[DividendEvent]:
    return [
        event
        for event in dividends
        if _date_string(
            event.ex_dividend_date
        )
        == session_date
    ]


def dividend_explains_gap(
    candidate: TurtleHistoryCandidate,
    dividend: DividendEvent,
    *,
    relative_error_tolerance: float = 0.35,
) -> bool:
    """
    Determine whether a cash distribution quantitatively explains
    a downward overnight discontinuity.

    Example:

        previous close = 70.80
        distribution   = 51.50
        expected ex-price ~ 19.30
        actual open     = 19.32

    This prevents us from classifying every ex-dividend date as a
    history break.
    """

    cash = _dividend_cash_amount(dividend)

    if cash is None or cash <= 0:
        return False

    observed_drop = (
        candidate.previous_close
        - candidate.new_open
    )

    if observed_drop <= 0:
        return False

    denominator = max(
        abs(observed_drop),
        abs(cash),
    )

    if denominator <= 0:
        return False

    relative_error = (
        abs(observed_drop - cash)
        / denominator
    )

    return (
        relative_error
        <= relative_error_tolerance
    )


def _confirmed_lineage_break(
    *,
    ticker: str | None,
    new_date: str | None,
) -> bool:
    if not ticker or not new_date:
        return False

    dates = CONFIRMED_LINEAGE_BREAKS.get(
        ticker,
        frozenset(),
    )

    return new_date in dates


def find_history_breaks(
    bars: Sequence[Any],
    *,
    ticker: str | None = None,
    dividends: Sequence[DividendEvent] = (),
    threshold: float = DEFAULT_DISCONTINUITY_THRESHOLD,
) -> list[TurtleHistoryBreak]:
    """
    Return only evidence-confirmed history boundaries.

    A large overnight gap by itself is NOT enough.
    """

    candidates = find_discontinuity_candidates(
        bars,
        threshold=threshold,
    )

    breaks: list[TurtleHistoryBreak] = []

    for candidate in candidates:
        if candidate.new_date is not None:
            matching = _matching_dividends(
                dividends,
                session_date=candidate.new_date,
            )

            explaining = [
                event
                for event in matching
                if dividend_explains_gap(
                    candidate,
                    event,
                )
            ]

            if explaining:
                event = max(
                    explaining,
                    key=lambda item: (
                        _dividend_cash_amount(item)
                        or 0.0
                    ),
                )

                cash = _dividend_cash_amount(event)

                breaks.append(
                    TurtleHistoryBreak(
                        break_index=candidate.break_index,
                        previous_close=(
                            candidate.previous_close
                        ),
                        new_open=candidate.new_open,
                        open_gap_pct=(
                            candidate.open_gap_pct
                        ),
                        previous_date=(
                            candidate.previous_date
                        ),
                        new_date=candidate.new_date,
                        reason=(
                            "CONFIRMED_CASH_DISTRIBUTION"
                        ),
                        evidence=(
                            f"Massive dividend "
                            f"ex_date={candidate.new_date} "
                            f"cash={cash}"
                        ),
                    )
                )

                continue

        if _confirmed_lineage_break(
            ticker=ticker,
            new_date=candidate.new_date,
        ):
            breaks.append(
                TurtleHistoryBreak(
                    break_index=candidate.break_index,
                    previous_close=(
                        candidate.previous_close
                    ),
                    new_open=candidate.new_open,
                    open_gap_pct=(
                        candidate.open_gap_pct
                    ),
                    previous_date=(
                        candidate.previous_date
                    ),
                    new_date=candidate.new_date,
                    reason=(
                        "CONFIRMED_LINEAGE_EVENT"
                    ),
                    evidence=(
                        "MANUAL_CONFIRMED_EVENT"
                    ),
                )
            )

    return breaks


def split_continuous_history(
    bars: Sequence[Any],
    *,
    ticker: str | None = None,
    dividends: Sequence[DividendEvent] = (),
    threshold: float = DEFAULT_DISCONTINUITY_THRESHOLD,
    minimum_segment_bars: int = 1,
) -> tuple[
    list[TurtleHistorySegment],
    list[TurtleHistoryBreak],
]:
    """
    Split history only at evidence-confirmed boundaries.

    Large unexplained market gaps remain in the tradable series.
    """

    if minimum_segment_bars < 1:
        raise ValueError(
            "minimum_segment_bars must be >= 1"
        )

    if not bars:
        return [], []

    breaks = find_history_breaks(
        bars,
        ticker=ticker,
        dividends=dividends,
        threshold=threshold,
    )

    break_indexes = {
        item.break_index
        for item in breaks
    }

    segments: list[TurtleHistorySegment] = []

    start = 0

    for index in range(1, len(bars)):
        if index not in break_indexes:
            continue

        piece = tuple(
            bars[start:index]
        )

        if len(piece) >= minimum_segment_bars:
            segments.append(
                TurtleHistorySegment(
                    start_index=start,
                    end_index=index - 1,
                    bars=piece,
                )
            )

        start = index

    final_piece = tuple(bars[start:])

    if len(final_piece) >= minimum_segment_bars:
        segments.append(
            TurtleHistorySegment(
                start_index=start,
                end_index=len(bars) - 1,
                bars=final_piece,
            )
        )

    return segments, breaks


@dataclass(frozen=True)
class TurtleHistoryAudit:
    ticker: str
    break_index: int
    previous_date: str | None
    new_date: str | None
    previous_close: float
    new_open: float
    open_gap_pct: float
    classification: str
    reason: str
    evidence: str
    confirmed_break: bool


def audit_history_candidates(
    bars,
    *,
    ticker: str,
    dividends=(),
    discontinuity_threshold: float = DEFAULT_DISCONTINUITY_THRESHOLD,
) -> list[TurtleHistoryAudit]:
    """
    Classify every large-gap history candidate.

    Important:
    - Large price movement alone never confirms a break.
    - Confirmed corporate-action/history boundaries are marked as breaks.
    - Unexplained large gaps remain tradable and are surfaced for audit.
    """
    candidates = find_discontinuity_candidates(
        bars,
        threshold=discontinuity_threshold,
    )

    confirmed_breaks = find_history_breaks(
        bars,
        ticker=ticker,
        dividends=dividends,
        threshold=discontinuity_threshold,
    )

    confirmed_by_index = {
        item.break_index: item
        for item in confirmed_breaks
    }

    rows: list[TurtleHistoryAudit] = []

    for candidate in candidates:
        confirmed = confirmed_by_index.get(
            candidate.break_index
        )

        if confirmed is not None:
            classification = confirmed.reason
            reason = confirmed.reason
            evidence = confirmed.evidence
            confirmed_break = True
        else:
            classification = "UNCONFIRMED_LARGE_GAP"
            reason = candidate.candidate_reason
            evidence = (
                "No confirmed corporate-action or manually "
                "researched history-boundary evidence."
            )
            confirmed_break = False

        rows.append(
            TurtleHistoryAudit(
                ticker=ticker,
                break_index=candidate.break_index,
                previous_date=candidate.previous_date,
                new_date=candidate.new_date,
                previous_close=candidate.previous_close,
                new_open=candidate.new_open,
                open_gap_pct=candidate.open_gap_pct,
                classification=classification,
                reason=reason,
                evidence=evidence,
                confirmed_break=confirmed_break,
            )
        )

    return rows
