from datetime import datetime, timedelta, timezone

import pytest

from scanner.feature_frame import snapshots_by_session
from scanner.historical import build_snapshot_as_of


def make_bars(count: int = 260) -> list[dict]:
    start = datetime(
        2024,
        1,
        2,
        5,
        0,
        tzinfo=timezone.utc,
    )

    bars = []
    price = 20.0

    for index in range(count):
        # Deliberately non-flat deterministic path so every rolling
        # feature receives enough variation to expose semantic drift.
        drift = 0.09 + ((index % 9) - 4) * 0.035
        price = max(1.0, price + drift)

        open_price = price - 0.11
        high = price + 0.42 + (index % 3) * 0.03
        low = price - 0.38 - (index % 4) * 0.02
        volume = 100_000 + (index * 1_731) % 180_000

        timestamp = start + timedelta(days=index)

        bars.append(
            {
                "o": open_price,
                "h": high,
                "l": low,
                "c": price,
                "v": volume,
                "t": int(timestamp.timestamp() * 1000),
            }
        )

    return bars


def assert_snapshots_match(expected: dict, actual: dict) -> None:
    assert set(actual) == set(expected)

    for key, expected_value in expected.items():
        actual_value = actual[key]

        if expected_value is None:
            assert actual_value is None, key
            continue

        if isinstance(expected_value, (int, float)):
            assert actual_value == pytest.approx(
                expected_value,
                rel=1e-12,
                abs=1e-12,
            ), key
            continue

        assert actual_value == expected_value, key


def test_precomputed_snapshots_match_legacy_prefix_builder():
    bars = make_bars()
    precomputed = snapshots_by_session(bars)

    # Exercise early, medium, and fully warmed-up indicator states.
    indexes = [5, 15, 25, 40, 75, 150, 225, 259]

    for index in indexes:
        signal_date = datetime.fromtimestamp(
            bars[index]["t"] / 1000,
            tz=timezone.utc,
        ).date().isoformat()

        expected = build_snapshot_as_of(
            bars=bars,
            signal_date=signal_date,
        )

        actual = precomputed[
            signal_date
        ]

        assert_snapshots_match(
            expected,
            actual,
        )