from __future__ import annotations

from datetime import date

import pytest

from scanner.corporate_actions import (
    DividendEvent,
)
from scanner.turtle_history import (
    dividend_explains_gap,
    find_discontinuity_candidates,
    find_history_breaks,
    overnight_gap_pct,
    split_continuous_history,
)


def bar(
    date_: str,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> dict:
    return {
        "session_date": date_,
        "o": open_,
        "h": high,
        "l": low,
        "c": close,
    }


def test_overnight_gap_pct():
    previous = bar(
        "2020-01-01",
        open_=100,
        high=101,
        low=99,
        close=100,
    )

    current = bar(
        "2020-01-02",
        open_=80,
        high=81,
        low=79,
        close=80,
    )

    assert (
        overnight_gap_pct(
            previous,
            current,
        )
        == pytest.approx(
            -20.0
        )
    )


def test_large_gap_is_candidate_not_automatically_break():
    bars = [
        bar(
            "2025-03-13",
            open_=20,
            high=21,
            low=19,
            close=20,
        ),
        bar(
            "2025-03-14",
            open_=30,
            high=35,
            low=29,
            close=34,
        ),
    ]

    candidates = (
        find_discontinuity_candidates(
            bars
        )
    )

    breaks = (
        find_history_breaks(
            bars,
            ticker="AAOI",
        )
    )

    assert len(
        candidates
    ) == 1

    assert breaks == []


def test_aaba_distribution_confirms_break():
    bars = [
        bar(
            "2019-09-23",
            open_=70.41,
            high=70.845,
            low=70.40,
            close=70.80,
        ),
        bar(
            "2019-09-24",
            open_=19.32,
            high=19.66,
            low=19.25,
            close=19.51,
        ),
    ]

    dividend = DividendEvent(
        ticker="AABA",
        ex_dividend_date=(
            date(
                2019,
                9,
                24,
            )
        ),
        cash_amount=51.50,
        split_adjusted_cash_amount=51.50,
        distribution_type="special",
    )

    candidates = (
        find_discontinuity_candidates(
            bars
        )
    )

    assert len(
        candidates
    ) == 1

    assert (
        dividend_explains_gap(
            candidates[0],
            dividend,
        )
    )

    breaks = (
        find_history_breaks(
            bars,
            ticker="AABA",
            dividends=[
                dividend
            ],
        )
    )

    assert len(
        breaks
    ) == 1

    assert (
        breaks[0].reason
        == "CONFIRMED_CASH_DISTRIBUTION"
    )


def test_aan_confirmed_lineage_break():
    bars = [
        bar(
            "2020-11-30",
            open_=63.32,
            high=64.17,
            low=62.06,
            close=62.93,
        ),
        bar(
            "2020-12-01",
            open_=21.70,
            high=22.11,
            low=16.74,
            close=18.41,
        ),
    ]

    breaks = (
        find_history_breaks(
            bars,
            ticker="AAN",
        )
    )

    assert len(
        breaks
    ) == 1

    assert (
        breaks[0].reason
        == "CONFIRMED_LINEAGE_EVENT"
    )


@pytest.mark.parametrize(
    (
        "ticker",
        "previous_close",
        "new_open",
    ),
    [
        (
            "AAOI",
            20.0,
            30.0,
        ),
        (
            "AAP",
            30.0,
            42.0,
        ),
        (
            "AAME",
            4.0,
            6.4,
        ),
    ],
)
def test_real_large_market_gaps_remain_continuous(
    ticker: str,
    previous_close: float,
    new_open: float,
):
    bars = [
        bar(
            "2025-01-01",
            open_=previous_close,
            high=(
                previous_close
                * 1.02
            ),
            low=(
                previous_close
                * 0.98
            ),
            close=previous_close,
        ),
        bar(
            "2025-01-02",
            open_=new_open,
            high=(
                new_open
                * 1.05
            ),
            low=(
                new_open
                * 0.95
            ),
            close=new_open,
        ),
    ]

    segments, breaks = (
        split_continuous_history(
            bars,
            ticker=ticker,
        )
    )

    assert breaks == []
    assert len(
        segments
    ) == 1

    assert len(
        segments[0].bars
    ) == 2


def test_history_splits_only_on_confirmed_break():
    bars = [
        bar(
            "2020-11-29",
            open_=62,
            high=64,
            low=61,
            close=63,
        ),
        bar(
            "2020-11-30",
            open_=63,
            high=64,
            low=62,
            close=62.93,
        ),
        bar(
            "2020-12-01",
            open_=21.70,
            high=22,
            low=17,
            close=18.41,
        ),
        bar(
            "2020-12-02",
            open_=18.5,
            high=19,
            low=18,
            close=18.7,
        ),
    ]

    segments, breaks = (
        split_continuous_history(
            bars,
            ticker="AAN",
        )
    )

    assert len(
        breaks
    ) == 1

    assert len(
        segments
    ) == 2

    assert (
        segments[0]
        .end_index
        == 1
    )

    assert (
        segments[1]
        .start_index
        == 2
    )


def test_threshold_validation():
    bars = [
        bar(
            "d1",
            open_=100,
            high=101,
            low=99,
            close=100,
        ),
        bar(
            "d2",
            open_=101,
            high=102,
            low=100,
            close=101,
        ),
    ]

    with pytest.raises(
        ValueError
    ):
        find_discontinuity_candidates(
            bars,
            threshold=0.0,
        )



def test_audit_marks_real_market_gap_unconfirmed():
    from scanner.turtle_history import (
        audit_history_candidates,
    )

    bars = [
        bar(
            "2025-03-13",
            open_=10.00,
            high=10.50,
            low=9.80,
            close=10.00,
        ),
        bar(
            "2025-03-14",
            open_=15.00,
            high=16.00,
            low=14.80,
            close=15.50,
        ),
    ]

    rows = audit_history_candidates(
        bars,
        ticker="AAOI",
        dividends=[],
    )

    assert len(rows) == 1

    assert (
        rows[0].classification
        == "UNCONFIRMED_LARGE_GAP"
    )

    assert (
        rows[0].confirmed_break
        is False
    )


def test_audit_marks_aaba_manual_break_confirmed():
    from scanner.turtle_history import (
        audit_history_candidates,
    )

    bars = [
        bar(
            "2019-09-23",
            open_=70.41,
            high=70.845,
            low=70.40,
            close=70.80,
        ),
        bar(
            "2019-09-24",
            open_=19.32,
            high=19.66,
            low=19.25,
            close=19.51,
        ),
    ]

    rows = audit_history_candidates(
        bars,
        ticker="AABA",
        dividends=[],
    )

    assert len(rows) == 1

    assert (
        rows[0].confirmed_break
        is True
    )

    assert (
        rows[0].classification
        == "CONFIRMED_LINEAGE_EVENT"
    )


def test_aac_ticker_reuse_confirms_lineage_break():
    bars = [
        bar(
            "2019-10-25",
            open_=0.70,
            high=0.72,
            low=0.68,
            close=0.70,
        ),
        bar(
            "2021-03-25",
            open_=14.51,
            high=15.00,
            low=14.00,
            close=14.50,
        ),
    ]

    breaks = find_history_breaks(
        bars,
        ticker="AAC",
    )

    assert len(breaks) == 1

    assert (
        breaks[0].new_date
        == "2021-03-25"
    )

    assert (
        breaks[0].reason
        == "CONFIRMED_LINEAGE_EVENT"
    )

