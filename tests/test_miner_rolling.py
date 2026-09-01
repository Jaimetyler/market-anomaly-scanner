from scanner.historical_universe import (
    HistoricalTicker,
    RollingUniverseResult,
)
from scanner.miner import (
    build_event_rows,
    build_rolling_research_selection,
    membership_dates_for_ticker,
    membership_dates_for_security,
    security_identity,
)


def make_ticker(
    ticker: str,
) -> HistoricalTicker:
    return HistoricalTicker(
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


def make_rolling() -> RollingUniverseResult:
    aaa = make_ticker("AAA")
    bbb = make_ticker("BBB")
    new = make_ticker("NEW")
    old = make_ticker("OLD")

    return RollingUniverseResult(
        snapshots={
            "2025-01-02": [
                aaa,
                bbb,
                old,
            ],
            "2025-01-03": [
                aaa,
                bbb,
                new,
            ],
        },
        first_seen={
            "AAA": "2025-01-02",
            "BBB": "2025-01-02",
            "OLD": "2025-01-02",
            "NEW": "2025-01-03",
        },
        last_seen={
            "AAA": "2025-01-03",
            "BBB": "2025-01-03",
            "OLD": "2025-01-02",
            "NEW": "2025-01-03",
        },
        securities={
            "AAA": aaa,
            "BBB": bbb,
            "OLD": old,
            "NEW": new,
        },
        fetched_snapshots=2,
        cached_snapshots=0,
    )


def test_membership_dates_allow_later_listing():
    rolling = make_rolling()

    dates = membership_dates_for_ticker(
        ticker="NEW",
        session_dates=[
            "2025-01-02",
            "2025-01-03",
        ],
        rolling_universe=rolling,
    )

    assert dates == {
        "2025-01-03",
    }


def test_membership_dates_stop_after_disappearance():
    rolling = make_rolling()

    dates = membership_dates_for_ticker(
        ticker="OLD",
        session_dates=[
            "2025-01-02",
            "2025-01-03",
        ],
        rolling_universe=rolling,
    )

    assert dates == {
        "2025-01-02",
    }


def test_build_selection_uses_union_of_snapshots(
    monkeypatch,
):
    rolling = make_rolling()

    monkeypatch.setattr(
        "scanner.miner.get_rolling_historical_universe",
        lambda **kwargs: rolling,
    )

    selected, result, raw_count, core_count = (
        build_rolling_research_selection(
            session_dates=[
                "2025-01-02",
                "2025-01-03",
            ],
            max_tickers=10,
        )
    )

    assert result is rolling

    assert {
        item.ticker
        for item in selected
    } == {
        "AAA",
        "BBB",
        "NEW",
        "OLD",
    }

    assert raw_count == 4
    assert core_count == 4


def test_build_selection_is_deterministic_and_bounded(
    monkeypatch,
):
    rolling = make_rolling()

    monkeypatch.setattr(
        "scanner.miner.get_rolling_historical_universe",
        lambda **kwargs: rolling,
    )

    selected, _, _, _ = (
        build_rolling_research_selection(
            session_dates=[
                "2025-01-02",
                "2025-01-03",
            ],
            max_tickers=2,
        )
    )

    assert [
        item.ticker
        for item in selected
    ] == [
        "AAA",
        "BBB",
    ]



def make_ticker_with_figi(ticker: str, figi: str) -> HistoricalTicker:
    return HistoricalTicker(
        ticker=ticker,
        name=f"{ticker} Company",
        active=True,
        type="CS",
        primary_exchange="XNAS",
        cik=None,
        composite_figi=figi,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )


def test_security_identity_prefers_composite_figi():
    security = make_ticker_with_figi("DUP", "BBG-ONE")
    assert security_identity(security) == "FIGI:BBG-ONE"


def test_security_membership_separates_ticker_reuse_by_figi():
    old = make_ticker_with_figi("DUP", "BBG-OLD")
    new = make_ticker_with_figi("DUP", "BBG-NEW")

    rolling = RollingUniverseResult(
        snapshots={
            "2025-01-02": [old],
            "2025-01-03": [new],
        },
        first_seen={"DUP": "2025-01-02"},
        last_seen={"DUP": "2025-01-03"},
        securities={"DUP": new},
        fetched_snapshots=2,
        cached_snapshots=0,
    )

    session_dates = ["2025-01-02", "2025-01-03"]

    assert membership_dates_for_security(
        security=old,
        session_dates=session_dates,
        rolling_universe=rolling,
    ) == {"2025-01-02"}

    assert membership_dates_for_security(
        security=new,
        session_dates=session_dates,
        rolling_universe=rolling,
    ) == {"2025-01-03"}


def test_security_membership_excludes_ambiguous_same_ticker_session():
    one = make_ticker_with_figi("DUP", "BBG-ONE")
    two = make_ticker_with_figi("DUP", "BBG-TWO")

    rolling = RollingUniverseResult(
        snapshots={
            "2025-01-02": [one, two],
            "2025-01-03": [one],
        },
        first_seen={"DUP": "2025-01-02"},
        last_seen={"DUP": "2025-01-03"},
        securities={"DUP": one},
        fetched_snapshots=2,
        cached_snapshots=0,
    )

    session_dates = ["2025-01-02", "2025-01-03"]

    assert membership_dates_for_security(
        security=one,
        session_dates=session_dates,
        rolling_universe=rolling,
    ) == {"2025-01-03"}

    assert membership_dates_for_security(
        security=two,
        session_dates=session_dates,
        rolling_universe=rolling,
    ) == set()



def test_event_clustering_does_not_merge_reused_ticker_identities():
    rows = [
        {
            "ticker": "DUP",
            "security_id": "FIGI:OLD",
            "signal_date": "2025-01-02",
            "research_eligible": True,
            "primary_setup": "FRESH_SPIKE",
        },
        {
            "ticker": "DUP",
            "security_id": "FIGI:NEW",
            "signal_date": "2025-01-03",
            "research_eligible": True,
            "primary_setup": "FRESH_SPIKE",
        },
    ]

    events = build_event_rows(
        rows=rows,
        session_dates=["2025-01-02", "2025-01-03"],
    )

    assert len(events) == 2
    assert {event["security_id"] for event in events} == {
        "FIGI:OLD",
        "FIGI:NEW",
    }
    assert {event["ticker"] for event in events} == {"DUP"}