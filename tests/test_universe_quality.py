from scanner.historical_universe import HistoricalTicker
from scanner.universe_quality import (
    SecurityBucket,
    build_core_research_universe,
    classify_security,
)


def make_ticker(
    ticker: str,
    name: str,
) -> HistoricalTicker:
    return HistoricalTicker(
        ticker=ticker,
        name=name,
        active=True,
        type="CS",
        primary_exchange="XNAS",
        cik=None,
        composite_figi=None,
        share_class_figi=None,
        list_date=None,
        delisted_utc=None,
    )


def test_normal_operating_company_is_included():
    ticker = make_ticker(
        "AAPL",
        "Apple Inc.",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.OPERATING_COMPANY
    )

    assert (
        result.include_in_core_research
        is True
    )


def test_spac_acquisition_corporation_is_tagged():
    ticker = make_ticker(
        "AACT",
        "Ares Acquisition Corporation II",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.SPAC
    )

    assert (
        result.include_in_core_research
        is False
    )


def test_spac_acquisition_inc_is_tagged():
    ticker = make_ticker(
        "AACB",
        "Artius II Acquisition Inc.",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.SPAC
    )


def test_senior_notes_are_excluded():
    ticker = make_ticker(
        "ABLLL",
        (
            "Abacus Global Management, Inc. "
            "9.875% Fixed Rate Senior Notes due 2028"
        ),
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.DEBT
    )

    assert (
        result.include_in_core_research
        is False
    )


def test_convertible_notes_are_excluded():
    ticker = make_ticker(
        "TEST",
        "Example Corp Convertible Notes due 2030",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.DEBT
    )


def test_preferred_stock_is_excluded():
    ticker = make_ticker(
        "TESTP",
        "Example Corp Series A Preferred Stock",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.PREFERRED
    )


def test_preferred_units_are_excluded_as_preferred():
    ticker = make_ticker(
        "TESTP",
        (
            "Example Partners L.P. "
            "6.25% Preferred Units, Series 1"
        ),
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.PREFERRED
    )

    assert (
        result.include_in_core_research
        is False
    )


def test_warrant_is_excluded():
    ticker = make_ticker(
        "TESTW",
        "Example Corp Warrants",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.WARRANT
    )


def test_actual_rights_security_is_excluded():
    ticker = make_ticker(
        "TESTR",
        "Example Corp Rights",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.RIGHTS
    )


def test_brightspire_does_not_false_match_rights():
    ticker = make_ticker(
        "BRSP",
        "BrightSpire Capital, Inc.",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.OPERATING_COMPANY
    )

    assert (
        result.include_in_core_research
        is True
    )


def test_partnership_units_are_not_automatically_excluded():
    ticker = make_ticker(
        "BIP",
        (
            "Brookfield Infrastructure Partners L.P. "
            "Limited Partnership Units"
        ),
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.OPERATING_COMPANY
    )

    assert (
        result.include_in_core_research
        is True
    )


def test_spac_bundle_unit_is_excluded():
    ticker = make_ticker(
        "TESTU",
        (
            "Example Acquisition Corp. "
            "Units Consisting of One Share "
            "and One-Half Warrant"
        ),
    )

    result = classify_security(
        ticker
    )

    #
    # Warrants are checked before generic units,
    # so either exclusion category is acceptable
    # from a core-research perspective.
    #
    assert (
        result.include_in_core_research
        is False
    )

    assert result.bucket in {
        SecurityBucket.WARRANT,
        SecurityBucket.UNIT,
        SecurityBucket.SPAC,
    }


def test_class_a_common_stock_is_not_accidentally_excluded():
    ticker = make_ticker(
        "ABNB",
        "Airbnb, Inc. Class A Common Stock",
    )

    result = classify_security(
        ticker
    )

    assert (
        result.bucket
        == SecurityBucket.OPERATING_COMPANY
    )

    assert (
        result.include_in_core_research
        is True
    )


def test_core_research_universe_only_returns_included_names():
    apple = make_ticker(
        "AAPL",
        "Apple Inc.",
    )

    spac = make_ticker(
        "SPAC",
        "Example Acquisition Corporation",
    )

    debt = make_ticker(
        "NOTE",
        "Example Corp Senior Notes due 2030",
    )

    result = build_core_research_universe(
        [
            apple,
            spac,
            debt,
        ]
    )

    assert result == [
        apple
    ]