from datetime import date

import pandas as pd
import pytest

from scanner.historical import (
    bar_session_date,
    bars_to_outcome_dataframe,
    build_snapshot_as_of,
    calculate_historical_outcomes,
    create_historical_slice,
)


def make_timestamp(
    day: str,
) -> int:
    return int(
        pd.Timestamp(
            day,
            tz="UTC",
        ).timestamp()
        * 1000
    )


def make_bar(
    day: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1_000_000,
):
    return {
        "t": make_timestamp(day),
        "o": open_price,
        "h": high,
        "l": low,
        "c": close,
        "v": volume,
    }


def test_bar_session_date():
    bar = make_bar(
        "2026-01-05",
        100,
        105,
        95,
        102,
    )

    assert (
        bar_session_date(bar)
        == date(2026, 1, 5)
    )


def test_historical_slice_enforces_information_wall():
    bars = [
        make_bar(
            "2026-01-05",
            90,
            101,
            89,
            100,
        ),
        make_bar(
            "2026-01-06",
            100,
            110,
            95,
            105,
        ),
        make_bar(
            "2026-01-07",
            120,
            130,
            110,
            125,
        ),
        make_bar(
            "2026-01-08",
            125,
            140,
            115,
            135,
        ),
    ]

    result = create_historical_slice(
        bars=bars,
        signal_date="2026-01-06",
    )

    assert len(
        result.history_bars
    ) == 2

    assert len(
        result.future_bars
    ) == 2

    assert (
        bar_session_date(
            result.history_bars[-1]
        )
        == date(2026, 1, 6)
    )

    assert (
        bar_session_date(
            result.future_bars[0]
        )
        == date(2026, 1, 7)
    )

    assert (
        result.signal_close
        == 105.0
    )

    assert (
        result.next_open
        == 120.0
    )


def test_signal_date_must_exist():
    bars = [
        make_bar(
            "2026-01-05",
            100,
            105,
            95,
            102,
        ),
        make_bar(
            "2026-01-07",
            105,
            110,
            100,
            108,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="No bar exists",
    ):
        create_historical_slice(
            bars=bars,
            signal_date="2026-01-06",
        )


def test_outcome_dataframe_starts_after_signal():
    bars = [
        make_bar(
            "2026-01-07",
            120,
            130,
            110,
            125,
        ),
        make_bar(
            "2026-01-08",
            125,
            140,
            115,
            135,
        ),
    ]

    df = bars_to_outcome_dataframe(
        bars
    )

    assert len(df) == 2

    assert (
        df.iloc[0]["open"]
        == 120
    )

    assert (
        df.iloc[0]["close"]
        == 125
    )


def test_research_and_executable_outcomes_are_different():
    bars = [
        make_bar(
            "2026-01-05",
            95,
            102,
            94,
            100,
        ),

        # Signal day closes at 100.
        make_bar(
            "2026-01-06",
            98,
            105,
            95,
            100,
        ),

        # Trader cannot enter at yesterday's
        # close because the next session gaps
        # all the way to 120.
        make_bar(
            "2026-01-07",
            120,
            130,
            110,
            125,
        ),

        make_bar(
            "2026-01-08",
            124,
            128,
            90,
            95,
        ),
    ]

    result = (
        calculate_historical_outcomes(
            bars=bars,
            signal_date="2026-01-06",
        )
    )

    assert (
        result.signal_close
        == 100.0
    )

    assert (
        result.next_open
        == 120.0
    )

    # Research:
    # signal close 100 -> next close 125
    #
    # = +25%
    assert (
        result.research.return_1d
        == pytest.approx(25.0)
    )

    # Executable:
    # next open 120 -> same session close 125
    #
    # = +4.1667%
    assert (
        result.executable
        is not None
    )

    assert (
        result.executable.return_1d
        == pytest.approx(
            4.1666666667
        )
    )

    # Research model sees a maximum rise
    # from 100 -> 130 = +30%.
    assert (
        result.research.mae_pct
        == pytest.approx(30.0)
    )

    # Realistic next-open model sees
    # 120 -> 130 = +8.333%.
    assert (
        result.executable.mae_pct
        == pytest.approx(
            8.3333333333
        )
    )


def test_snapshot_cannot_see_future_price_explosion():
    """
    Critical anti-look-ahead test.

    Build two datasets with IDENTICAL history
    through the signal date.

    Their future is wildly different.

    Historical snapshot values MUST still match.
    """

    bars_a = []
    bars_b = []

    start = pd.Timestamp(
        "2025-01-01",
        tz="UTC",
    )

    price = 50.0

    for i in range(230):
        day = (
            start
            + pd.Timedelta(
                days=i
            )
        )

        day_text = (
            day.date().isoformat()
        )

        price += 0.10

        bar = make_bar(
            day_text,
            price - 0.20,
            price + 0.50,
            price - 0.50,
            price,
            1_000_000,
        )

        bars_a.append(
            bar.copy()
        )

        bars_b.append(
            bar.copy()
        )

    signal_date = (
        bar_session_date(
            bars_a[-1]
        )
    )

    # Add radically different futures.

    future_day_1 = (
        pd.Timestamp(
            signal_date,
            tz="UTC",
        )
        + pd.Timedelta(
            days=1
        )
    )

    future_day_2 = (
        future_day_1
        + pd.Timedelta(
            days=1
        )
    )

    bars_a.append(
        make_bar(
            future_day_1.date().isoformat(),
            75,
            76,
            70,
            72,
        )
    )

    bars_a.append(
        make_bar(
            future_day_2.date().isoformat(),
            72,
            74,
            65,
            68,
        )
    )

    # Completely insane future.
    bars_b.append(
        make_bar(
            future_day_1.date().isoformat(),
            500,
            700,
            450,
            650,
        )
    )

    bars_b.append(
        make_bar(
            future_day_2.date().isoformat(),
            900,
            1200,
            800,
            1100,
        )
    )

    snapshot_a = (
        build_snapshot_as_of(
            bars=bars_a,
            signal_date=signal_date,
        )
    )

    snapshot_b = (
        build_snapshot_as_of(
            bars=bars_b,
            signal_date=signal_date,
        )
    )

    keys_to_compare = [
        "close",
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "relative_volume",
        "rsi_14",
        "distance_sma_20_pct",
        "distance_sma_50_pct",
        "atr_expansion",
    ]

    for key in keys_to_compare:
        value_a = snapshot_a.get(
            key
        )

        value_b = snapshot_b.get(
            key
        )

        if (
            value_a is None
            or value_b is None
        ):
            assert value_a == value_b

        else:
            assert (
                value_a
                == pytest.approx(
                    value_b
                )
            )


def test_no_future_sessions_produces_empty_outcomes():
    bars = [
        make_bar(
            "2026-01-05",
            95,
            102,
            94,
            100,
        ),
        make_bar(
            "2026-01-06",
            98,
            105,
            95,
            100,
        ),
    ]

    result = (
        calculate_historical_outcomes(
            bars=bars,
            signal_date="2026-01-06",
        )
    )

    assert (
        result.next_open
        is None
    )

    assert (
        result.executable
        is None
    )

    assert (
        result.research.return_1d
        is None
    )

    assert (
        result.research.mfe_pct
        is None
    )