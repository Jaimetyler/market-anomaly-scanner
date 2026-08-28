from dataclasses import dataclass, field
from typing import Any


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class DetectionResult:
    eligible: bool
    candidate: bool

    failed_filters: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    values: dict[str, Any] = field(default_factory=dict)


# ============================================================
# CONFIG
# ============================================================

MIN_PRICE = 1.00
MIN_DOLLAR_VOLUME = 2_000_000


# Primary anomaly triggers
TRIGGER_RETURN_1D = 15.0
TRIGGER_RETURN_3D = 25.0
TRIGGER_RETURN_5D = 35.0
TRIGGER_RVOL = 4.0
TRIGGER_SMA20_EXTENSION = 25.0


# Supporting evidence
EVIDENCE_RSI = 75.0
EVIDENCE_RVOL = 2.0
EVIDENCE_ATR_EXPANSION = 1.5
EVIDENCE_SMA20_EXTENSION = 15.0
EVIDENCE_SMA50_EXTENSION = 25.0
EVIDENCE_RETURN_10D = 40.0
EVIDENCE_RETURN_20D = 60.0


# ============================================================
# HELPERS
# ============================================================

def _gte(
    value: float | int | None,
    threshold: float,
) -> bool:
    if value is None:
        return False

    return float(value) >= threshold


def _format_pct(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.2f}%"


# ============================================================
# DETECTOR
# ============================================================

def detect_anomaly(
    snapshot: dict[str, Any],
) -> DetectionResult:
    """
    Evaluate one quantitative snapshot.

    Important:

    This detector identifies unusual upside behavior.

    It does NOT:
        - recommend a short
        - calculate an anomaly score
        - evaluate fundamentals
        - evaluate catalyst quality
        - determine trade direction

    The goal is broad candidate discovery for research.
    """

    failed_filters: list[str] = []
    triggers: list[str] = []
    evidence: list[str] = []

    price = snapshot.get("price")
    dollar_volume = snapshot.get("dollar_volume")

    return_1d = snapshot.get("return_1d")
    return_3d = snapshot.get("return_3d")
    return_5d = snapshot.get("return_5d")
    return_10d = snapshot.get("return_10d")
    return_20d = snapshot.get("return_20d")

    relative_volume = snapshot.get("relative_volume")
    rsi_14 = snapshot.get("rsi_14")

    distance_sma_20 = snapshot.get(
        "distance_sma_20_pct"
    )

    distance_sma_50 = snapshot.get(
        "distance_sma_50_pct"
    )

    atr_expansion = snapshot.get(
        "atr_expansion"
    )

    # --------------------------------------------------------
    # ELIGIBILITY FILTERS
    # --------------------------------------------------------

    if price is None:
        failed_filters.append(
            "Missing price"
        )

    elif price < MIN_PRICE:
        failed_filters.append(
            f"Price below ${MIN_PRICE:.2f}"
        )

    if dollar_volume is None:
        failed_filters.append(
            "Missing dollar volume"
        )

    elif dollar_volume < MIN_DOLLAR_VOLUME:
        failed_filters.append(
            "Dollar volume below $2M"
        )

    eligible = len(failed_filters) == 0

    # --------------------------------------------------------
    # PRIMARY TRIGGERS
    # --------------------------------------------------------

    if _gte(
        return_1d,
        TRIGGER_RETURN_1D,
    ):
        triggers.append(
            (
                "PRICE_1D_15: "
                f"1-day return {_format_pct(return_1d)}"
            )
        )

    if _gte(
        return_3d,
        TRIGGER_RETURN_3D,
    ):
        triggers.append(
            (
                "PRICE_3D_25: "
                f"3-day return {_format_pct(return_3d)}"
            )
        )

    if _gte(
        return_5d,
        TRIGGER_RETURN_5D,
    ):
        triggers.append(
            (
                "PRICE_5D_35: "
                f"5-day return {_format_pct(return_5d)}"
            )
        )

    if _gte(
        relative_volume,
        TRIGGER_RVOL,
    ):
        triggers.append(
            (
                "RVOL_4: "
                f"relative volume {relative_volume:.2f}x"
            )
        )

    if _gte(
        distance_sma_20,
        TRIGGER_SMA20_EXTENSION,
    ):
        triggers.append(
            (
                "SMA20_EXT_25: "
                f"{_format_pct(distance_sma_20)} "
                "above SMA20"
            )
        )

    # --------------------------------------------------------
    # SUPPORTING EVIDENCE
    # --------------------------------------------------------

    if _gte(
        rsi_14,
        EVIDENCE_RSI,
    ):
        evidence.append(
            (
                "RSI_75: "
                f"RSI(14) {rsi_14:.2f}"
            )
        )

    if _gte(
        relative_volume,
        EVIDENCE_RVOL,
    ):
        evidence.append(
            (
                "RVOL_2: "
                f"relative volume {relative_volume:.2f}x"
            )
        )

    if _gte(
        atr_expansion,
        EVIDENCE_ATR_EXPANSION,
    ):
        evidence.append(
            (
                "ATR_EXPANSION_1_5: "
                f"ATR expansion {atr_expansion:.2f}x"
            )
        )

    if _gte(
        distance_sma_20,
        EVIDENCE_SMA20_EXTENSION,
    ):
        evidence.append(
            (
                "SMA20_EXT_15: "
                f"{_format_pct(distance_sma_20)} "
                "above SMA20"
            )
        )

    if _gte(
        distance_sma_50,
        EVIDENCE_SMA50_EXTENSION,
    ):
        evidence.append(
            (
                "SMA50_EXT_25: "
                f"{_format_pct(distance_sma_50)} "
                "above SMA50"
            )
        )

    if _gte(
        return_10d,
        EVIDENCE_RETURN_10D,
    ):
        evidence.append(
            (
                "PRICE_10D_40: "
                f"10-day return {_format_pct(return_10d)}"
            )
        )

    if _gte(
        return_20d,
        EVIDENCE_RETURN_20D,
    ):
        evidence.append(
            (
                "PRICE_20D_60: "
                f"20-day return {_format_pct(return_20d)}"
            )
        )

    candidate = eligible and len(triggers) > 0

    return DetectionResult(
        eligible=eligible,
        candidate=candidate,
        failed_filters=failed_filters,
        triggers=triggers,
        evidence=evidence,
        values={
            "price": price,
            "dollar_volume": dollar_volume,
            "return_1d": return_1d,
            "return_3d": return_3d,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_20d": return_20d,
            "relative_volume": relative_volume,
            "rsi_14": rsi_14,
            "distance_sma_20_pct": distance_sma_20,
            "distance_sma_50_pct": distance_sma_50,
            "atr_expansion": atr_expansion,
        },
    )