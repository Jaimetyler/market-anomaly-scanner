from __future__ import annotations

import json

import scanner.data as data_module


def test_daily_bars_first_call_fetches_then_second_call_uses_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        data_module,
        "DAILY_BAR_CACHE_DIR",
        tmp_path,
    )

    calls = []

    expected = [
        {
            "t": 1,
            "o": 10.0,
            "h": 11.0,
            "l": 9.0,
            "c": 10.5,
            "v": 1000,
        }
    ]

    def fake_fetch(
        *,
        ticker,
        start_date,
        end_date,
    ):
        calls.append(
            (
                ticker,
                start_date,
                end_date,
            )
        )

        return expected

    monkeypatch.setattr(
        data_module,
        "_fetch_daily_bars_from_massive",
        fake_fetch,
    )

    first = data_module.get_daily_bars(
        ticker="aapl",
        start_date="2025-01-01",
        end_date="2025-03-31",
    )

    second = data_module.get_daily_bars(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-03-31",
    )

    assert first == expected
    assert second == expected
    assert calls == [
        (
            "AAPL",
            "2025-01-01",
            "2025-03-31",
        )
    ]


def test_force_refresh_replaces_existing_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        data_module,
        "DAILY_BAR_CACHE_DIR",
        tmp_path,
    )

    responses = [
        [{"t": 1, "c": 10.0}],
        [{"t": 2, "c": 11.0}],
    ]

    def fake_fetch(
        *,
        ticker,
        start_date,
        end_date,
    ):
        return responses.pop(0)

    monkeypatch.setattr(
        data_module,
        "_fetch_daily_bars_from_massive",
        fake_fetch,
    )

    first = data_module.get_daily_bars(
        ticker="AA",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    refreshed = data_module.get_daily_bars(
        ticker="AA",
        start_date="2025-01-01",
        end_date="2025-01-31",
        force_refresh=True,
    )

    cached_again = data_module.get_daily_bars(
        ticker="AA",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert first == [{"t": 1, "c": 10.0}]
    assert refreshed == [{"t": 2, "c": 11.0}]
    assert cached_again == refreshed


def test_corrupt_cache_is_treated_as_miss(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        data_module,
        "DAILY_BAR_CACHE_DIR",
        tmp_path,
    )

    path = data_module._cache_path(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "{this-is-not-json",
        encoding="utf-8",
    )

    expected = [
        {
            "t": 1,
            "c": 10.0,
        }
    ]

    monkeypatch.setattr(
        data_module,
        "_fetch_daily_bars_from_massive",
        lambda **kwargs: expected,
    )

    result = data_module.get_daily_bars(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert result == expected

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["cache_version"] == 1
    assert payload["bars"] == expected


def test_different_date_ranges_use_different_cache_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        data_module,
        "DAILY_BAR_CACHE_DIR",
        tmp_path,
    )

    calls = []

    def fake_fetch(
        *,
        ticker,
        start_date,
        end_date,
    ):
        calls.append(
            (
                ticker,
                start_date,
                end_date,
            )
        )

        return [
            {
                "t": len(calls),
            }
        ]

    monkeypatch.setattr(
        data_module,
        "_fetch_daily_bars_from_massive",
        fake_fetch,
    )

    data_module.get_daily_bars(
        ticker="AAPL",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    data_module.get_daily_bars(
        ticker="AAPL",
        start_date="2025-02-01",
        end_date="2025-02-28",
    )

    assert len(calls) == 2