"""Offline, reviewed boundaries; dates begin a new independent history.

These boundaries prevent cross-security windows; they do not certify identities
or completeness of the underlying prices.
"""
from bisect import bisect_right

CONFIRMED_LINEAGE_BREAKS = {
    "AABA": frozenset({"2019-09-24"}),
    "AAN": frozenset({"2020-12-01"}),
    "AAC": frozenset({"2021-03-25"}),
    "ACET": frozenset({"2020-09-16"}),
}


def lineage_segment(ticker: str, day: str) -> int:
    return bisect_right(sorted(CONFIRMED_LINEAGE_BREAKS.get(ticker, ())), day)
