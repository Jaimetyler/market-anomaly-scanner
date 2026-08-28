from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from scanner.historical_universe import HistoricalTicker


class SecurityBucket(str, Enum):
    OPERATING_COMPANY = "OPERATING_COMPANY"
    SPAC = "SPAC"
    DEBT = "DEBT"
    PREFERRED = "PREFERRED"
    WARRANT = "WARRANT"
    RIGHTS = "RIGHTS"
    UNIT = "UNIT"
    OTHER_EXCLUDED = "OTHER_EXCLUDED"


@dataclass(frozen=True)
class UniverseClassification:
    ticker: HistoricalTicker
    bucket: SecurityBucket
    include_in_core_research: bool
    reasons: tuple[str, ...]


DEBT_PATTERNS = (
    r"\bsenior notes?\b",
    r"\bsubordinated notes?\b",
    r"\bconvertible notes?\b",
    r"\bfixed[- ]rate notes?\b",
    r"\bfloating[- ]rate notes?\b",
    r"\bdebentures?\b",
    r"\bbonds?\s+due\b",
    r"\bnotes?\s+due\b",
)

PREFERRED_PATTERNS = (
    r"\bpreferred stock\b",
    r"\bpreferred shares?\b",
    r"\bpreferred units?\b",
    r"\bpreferred interest\b",
    r"\bdepositary shares?\b",
    r"\bdepositary receipts?.*preferred\b",
)

WARRANT_PATTERNS = (
    r"\bwarrants?\b",
)

RIGHTS_PATTERNS = (
    r"\brights\b",
    r"\bright to receive\b",
)

#
# Only use UNIT for obvious bundled/security units.
#
# We intentionally do NOT exclude generic:
#
#   "Limited Partnership Units"
#
# because securities such as BIP and BBU are actual
# operating-company equity-like instruments rather
# than SPAC bundles or financing instruments.
#
UNIT_PATTERNS = (
    r"\bunits consisting\b",
    r"\bunit consisting\b",
    r"\beach unit consists\b",
    r"\beach unit consisting\b",
)

SPAC_PATTERNS = (
    r"\bacquisition corp\b",
    r"\bacquisition corporation\b",
    r"\bacquisition company\b",
    r"\bacquisition co\.\b",
    r"\bacquisition inc\.?\b",
    r"\bblank check\b",
)


def _normalized_name(
    ticker: HistoricalTicker,
) -> str:
    return (
        ticker.name
        or ""
    ).strip().lower()


def _matches_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def classify_security(
    ticker: HistoricalTicker,
) -> UniverseClassification:
    """
    Classify a Massive ticker into a research bucket.

    Massive's ticker type is useful, but type='CS'
    alone does not guarantee normal operating-company
    common equity.

    This function intentionally uses regex word
    boundaries rather than raw substring matching
    so names such as "BrightSpire" do not accidentally
    match the word "rights".

    The raw point-in-time universe is never destroyed.
    This layer only determines the core research bucket.
    """

    name = _normalized_name(
        ticker
    )

    if _matches_any(
        name,
        DEBT_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.DEBT,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_DEBT_SECURITY",
            ),
        )

    #
    # Preferred classification comes before UNIT.
    #
    # Example:
    # "Preferred Units, Series 1"
    #
    # should be categorized as preferred rather
    # than simply as a unit.
    #
    if _matches_any(
        name,
        PREFERRED_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.PREFERRED,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_PREFERRED_SECURITY",
            ),
        )

    if _matches_any(
        name,
        WARRANT_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.WARRANT,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_WARRANT",
            ),
        )

    if _matches_any(
        name,
        RIGHTS_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.RIGHTS,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_RIGHTS_SECURITY",
            ),
        )

    if _matches_any(
        name,
        UNIT_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.UNIT,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_UNIT_SECURITY",
            ),
        )

    if _matches_any(
        name,
        SPAC_PATTERNS,
    ):
        return UniverseClassification(
            ticker=ticker,
            bucket=SecurityBucket.SPAC,
            include_in_core_research=False,
            reasons=(
                "NAME_LOOKS_LIKE_SPAC",
            ),
        )

    return UniverseClassification(
        ticker=ticker,
        bucket=SecurityBucket.OPERATING_COMPANY,
        include_in_core_research=True,
        reasons=(),
    )


def classify_universe(
    tickers: list[HistoricalTicker],
) -> list[UniverseClassification]:
    return [
        classify_security(
            ticker
        )
        for ticker in tickers
    ]


def build_core_research_universe(
    tickers: list[HistoricalTicker],
) -> list[HistoricalTicker]:
    """
    Return securities eligible for the main
    operating-company anomaly research dataset.

    Excluded/tagged securities remain preserved
    in the raw historical universe for possible
    separate research later.
    """

    classifications = classify_universe(
        tickers
    )

    return [
        item.ticker
        for item in classifications
        if item.include_in_core_research
    ]