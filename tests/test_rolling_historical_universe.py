from datetime import date

import pytest

import scanner.historical_universe as hu


def make_ticker(
    ticker: str,
) -> hu.HistoricalTicker:
    return hu.HistoricalTicker(
        ticker=ticker,
        name=f"{ticker} Company",
        active=True,
        type="CS",
        primary_exchange="XNAS",
        cik=None,
        composite_figi=f"FIGI-{ticker}",
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )


def make_result(
    session_date: date,
    tickers: list[str],
    *,
    cached: bool = False,
) -> hu.HistoricalUniverseResult:
    securities = [
        make_ticker(ticker)
        for ticker in tickers
    ]

    return hu.HistoricalUniverseResult(
        as_of_date=session_date,
        tickers=securities,
        total_count=len(securities),
        fetched_from_cache=cached,
    )


def test_rolling_universe_tracks_new_listing(
    monkeypatch,
):
    snapshots = {
        date(2025, 1, 2): [
            "AAA",
            "BBB",
        ],
        date(2025, 1, 3): [
            "AAA",
            "BBB",
            "NEW",
        ],
    }

    def fake_get(
        as_of_date,
        *,
        force_refresh=False,
    ):
        session_date = hu._normalize_date(
            as_of_date
        )

        return make_result(
            session_date,
            snapshots[session_date],
        )

    monkeypatch.setattr(
        hu,
        "get_historical_universe",
        fake_get,
    )

    result = (
        hu.get_rolling_historical_universe(
            [
                "2025-01-02",
                "2025-01-03",
            ]
        )
    )

    assert not hu.ticker_is_member_on_date(
        ticker="NEW",
        session_date="2025-01-02",
        rolling_universe=result,
    )

    assert hu.ticker_is_member_on_date(
        ticker="NEW",
        session_date="2025-01-03",
        rolling_universe=result,
    )

    assert (
        result.first_seen["NEW"]
        == "2025-01-03"
    )


def test_rolling_universe_tracks_disappearance(
    monkeypatch,
):
    snapshots = {
        date(2025, 1, 2): [
            "AAA",
            "OLD",
        ],
        date(2025, 1, 3): [
            "AAA",
        ],
    }

    def fake_get(
        as_of_date,
        *,
        force_refresh=False,
    ):
        session_date = hu._normalize_date(
            as_of_date
        )

        return make_result(
            session_date,
            snapshots[session_date],
        )

    monkeypatch.setattr(
        hu,
        "get_historical_universe",
        fake_get,
    )

    result = (
        hu.get_rolling_historical_universe(
            [
                "2025-01-02",
                "2025-01-03",
            ]
        )
    )

    assert hu.ticker_is_member_on_date(
        ticker="OLD",
        session_date="2025-01-02",
        rolling_universe=result,
    )

    assert not hu.ticker_is_member_on_date(
        ticker="OLD",
        session_date="2025-01-03",
        rolling_universe=result,
    )

    assert (
        result.last_seen["OLD"]
        == "2025-01-02"
    )


def test_rolling_universe_counts_cache_sources(
    monkeypatch,
):
    def fake_get(
        as_of_date,
        *,
        force_refresh=False,
    ):
        session_date = hu._normalize_date(
            as_of_date
        )

        return make_result(
            session_date,
            ["AAA"],
            cached=(
                session_date
                == date(2025, 1, 2)
            ),
        )

    monkeypatch.setattr(
        hu,
        "get_historical_universe",
        fake_get,
    )

    result = (
        hu.get_rolling_historical_universe(
            [
                "2025-01-02",
                "2025-01-03",
            ]
        )
    )

    assert result.cached_snapshots == 1
    assert result.fetched_snapshots == 1


def test_rolling_universe_dedupes_dates(
    monkeypatch,
):
    calls = []

    def fake_get(
        as_of_date,
        *,
        force_refresh=False,
    ):
        session_date = hu._normalize_date(
            as_of_date
        )

        calls.append(
            session_date
        )

        return make_result(
            session_date,
            ["AAA"],
        )

    monkeypatch.setattr(
        hu,
        "get_historical_universe",
        fake_get,
    )

    hu.get_rolling_historical_universe(
        [
            "2025-01-02",
            date(2025, 1, 2),
            "2025-01-03",
        ]
    )

    assert calls == [
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]


def test_rolling_universe_rejects_empty_dates():
    with pytest.raises(
        ValueError,
        match="session_dates cannot be empty",
    ):
        hu.get_rolling_historical_universe(
            []
        )


def test_membership_requires_known_snapshot(
    monkeypatch,
):
    def fake_get(
        as_of_date,
        *,
        force_refresh=False,
    ):
        session_date = hu._normalize_date(
            as_of_date
        )

        return make_result(
            session_date,
            ["AAA"],
        )

    monkeypatch.setattr(
        hu,
        "get_historical_universe",
        fake_get,
    )

    result = (
        hu.get_rolling_historical_universe(
            ["2025-01-02"]
        )
    )

    with pytest.raises(
        KeyError,
        match="No universe snapshot exists",
    ):
        hu.ticker_is_member_on_date(
            ticker="AAA",
            session_date="2025-01-03",
            rolling_universe=result,
        )