from datetime import date

import pytest

from scanner.historical_universe import (
    HistoricalTicker,
    _eligible_ticker,
    _normalize_date,
    _parse_ticker,
)


def test_normalize_string_date():
    result = _normalize_date(
        "2025-06-30"
    )

    assert result == date(
        2025,
        6,
        30,
    )


def test_parse_ticker():
    raw = {
        "ticker": "abc",
        "name": "ABC Corp",
        "active": True,
        "type": "CS",
        "primary_exchange": "XNYS",
        "cik": "123",
        "composite_figi": "FIGI1",
        "share_class_figi": "FIGI2",
        "list_date": "2020-01-01",
        "delisted_utc": None,
    }

    ticker = _parse_ticker(
        raw
    )

    assert ticker.ticker == "ABC"
    assert ticker.name == "ABC Corp"
    assert ticker.active is True
    assert ticker.type == "CS"
    assert (
        ticker.primary_exchange
        == "XNYS"
    )


def test_common_stock_on_nyse_is_eligible():
    ticker = HistoricalTicker(
        ticker="ABC",
        name="ABC Corp",
        active=True,
        type="CS",
        primary_exchange="XNYS",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is True
    )


def test_common_stock_on_nasdaq_is_eligible():
    ticker = HistoricalTicker(
        ticker="XYZ",
        name="XYZ Corp",
        active=True,
        type="CS",
        primary_exchange="XNAS",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is True
    )


def test_common_stock_on_american_is_eligible():
    ticker = HistoricalTicker(
        ticker="XYZ",
        name="XYZ Corp",
        active=True,
        type="CS",
        primary_exchange="XASE",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is True
    )


def test_etf_is_not_eligible():
    ticker = HistoricalTicker(
        ticker="ETF",
        name="ETF Fund",
        active=True,
        type="ETF",
        primary_exchange="XNAS",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is False
    )


def test_otc_is_not_eligible():
    ticker = HistoricalTicker(
        ticker="OTC",
        name="OTC Corp",
        active=True,
        type="CS",
        primary_exchange="OTCM",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is False
    )


def test_empty_ticker_is_not_eligible():
    ticker = HistoricalTicker(
        ticker="",
        name=None,
        active=True,
        type="CS",
        primary_exchange="XNYS",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )

    assert (
        _eligible_ticker(
            ticker
        )
        is False
    )