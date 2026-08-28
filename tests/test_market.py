from scanner.market import (
    build_prefilter_candidates,
)


def make_market_snapshot(
    ticker="TEST",
    price=10.0,
    volume=1_000_000,
    change_pct=10.0,
):
    return {
        "ticker": ticker,
        "day": {
            "c": price,
            "v": volume,
        },
        "todaysChangePerc": change_pct,
    }


def test_prefilter_accepts_liquid_gainer():
    market = [
        make_market_snapshot()
    ]

    results = build_prefilter_candidates(
        market
    )

    assert len(results) == 1
    assert results[0].ticker == "TEST"


def test_prefilter_rejects_penny_stock():
    market = [
        make_market_snapshot(
            price=0.50,
        )
    ]

    results = build_prefilter_candidates(
        market
    )

    assert results == []


def test_prefilter_rejects_illiquid_stock():
    market = [
        make_market_snapshot(
            price=2.00,
            volume=100_000,
        )
    ]

    results = build_prefilter_candidates(
        market
    )

    assert results == []


def test_prefilter_rejects_small_move():
    market = [
        make_market_snapshot(
            change_pct=2.0,
        )
    ]

    results = build_prefilter_candidates(
        market
    )

    assert results == []


def test_prefilter_orders_largest_gainers_first():
    market = [
        make_market_snapshot(
            ticker="AAA",
            change_pct=7.0,
        ),
        make_market_snapshot(
            ticker="BBB",
            change_pct=25.0,
        ),
        make_market_snapshot(
            ticker="CCC",
            change_pct=12.0,
        ),
    ]

    results = build_prefilter_candidates(
        market
    )

    assert [
        item.ticker
        for item in results
    ] == [
        "BBB",
        "CCC",
        "AAA",
    ]