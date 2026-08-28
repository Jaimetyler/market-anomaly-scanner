from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassificationResult:
    primary_setup: str
    tags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)


def _gte(
    value: float | None,
    threshold: float,
) -> bool:
    return (
        value is not None
        and float(value) >= threshold
    )


def _lte(
    value: float | None,
    threshold: float,
) -> bool:
    return (
        value is not None
        and float(value) <= threshold
    )


def classify_setup(
    snapshot: dict[str, Any],
) -> ClassificationResult:
    """
    Deterministically classify the current anomaly.

    IMPORTANT:
    These classifications describe market behavior.
    They are NOT trade recommendations.

    A stock may receive multiple tags, while
    primary_setup identifies the dominant pattern.
    """

    return_1d = snapshot.get("return_1d")
    return_3d = snapshot.get("return_3d")
    return_5d = snapshot.get("return_5d")
    return_10d = snapshot.get("return_10d")
    return_20d = snapshot.get("return_20d")

    relative_volume = snapshot.get(
        "relative_volume"
    )

    rsi_14 = snapshot.get(
        "rsi_14"
    )

    sma20_extension = snapshot.get(
        "distance_sma_20_pct"
    )

    sma50_extension = snapshot.get(
        "distance_sma_50_pct"
    )

    atr_expansion = snapshot.get(
        "atr_expansion"
    )

    tags: list[str] = []
    reasons: list[str] = []

    # --------------------------------------------------
    # FRESH SPIKE
    # --------------------------------------------------

    if _gte(return_1d, 20):
        tags.append(
            "FRESH_SPIKE"
        )

        reasons.append(
            "One-day return is at least +20%."
        )

    # --------------------------------------------------
    # VOLUME SHOCK
    # --------------------------------------------------

    if _gte(relative_volume, 5):
        tags.append(
            "VOLUME_SHOCK"
        )

        reasons.append(
            "Relative volume is at least 5x "
            "the recent average."
        )

    # --------------------------------------------------
    # MULTI-DAY ACCELERATION
    # --------------------------------------------------

    multi_day_acceleration = (
        _gte(return_3d, 25)
        and _gte(return_5d, 35)
    )

    if multi_day_acceleration:
        tags.append(
            "MULTI_DAY_ACCELERATION"
        )

        reasons.append(
            "Three-day and five-day returns "
            "show strong upside acceleration."
        )

    # --------------------------------------------------
    # PARABOLIC EXTENSION
    # --------------------------------------------------

    parabolic = (
        (
            _gte(return_5d, 75)
            or _gte(return_10d, 100)
            or _gte(return_20d, 150)
        )
        and (
            _gte(sma20_extension, 50)
            or _gte(rsi_14, 85)
        )
    )

    if parabolic:
        tags.append(
            "PARABOLIC_EXTENSION"
        )

        reasons.append(
            "Price has experienced an extreme "
            "multi-session move and remains highly "
            "extended from its recent trend."
        )

    # --------------------------------------------------
    # POST-SPIKE PULLBACK
    # --------------------------------------------------

    prior_extension = (
        _gte(return_3d, 25)
        or _gte(return_5d, 50)
        or _gte(return_10d, 75)
        or _gte(return_20d, 100)
    )

    pulling_back = (
        _lte(return_1d, -5)
    )

    still_extended = (
        _gte(sma20_extension, 20)
    )

    if (
        prior_extension
        and pulling_back
        and still_extended
    ):
        tags.append(
            "POST_SPIKE_PULLBACK"
        )

        reasons.append(
            "Stock is pulling back after a major "
            "recent advance while remaining extended "
            "above its 20-day trend."
        )

    # --------------------------------------------------
    # EXTENDED TREND
    # --------------------------------------------------

    extended_trend = (
        _gte(return_20d, 50)
        and _gte(sma20_extension, 25)
    )

    if extended_trend:
        tags.append(
            "EXTENDED_TREND"
        )

        reasons.append(
            "Twenty-day return and SMA20 extension "
            "indicate a sustained extended trend."
        )

    # --------------------------------------------------
    # VOLATILITY EXPANSION
    # --------------------------------------------------

    if _gte(atr_expansion, 2):
        tags.append(
            "VOLATILITY_EXPANSION"
        )

        reasons.append(
            "ATR is at least 2x its recent baseline."
        )

    # --------------------------------------------------
    # Determine dominant setup
    # --------------------------------------------------

    priority = [
        "POST_SPIKE_PULLBACK",
        "PARABOLIC_EXTENSION",
        "FRESH_SPIKE",
        "MULTI_DAY_ACCELERATION",
        "VOLUME_SHOCK",
        "EXTENDED_TREND",
        "VOLATILITY_EXPANSION",
    ]

    primary_setup = (
        "UNCLASSIFIED_ANOMALY"
    )

    for setup in priority:
        if setup in tags:
            primary_setup = setup
            break

    return ClassificationResult(
        primary_setup=primary_setup,
        tags=tags,
        reasons=reasons,
        values={
            "return_1d": return_1d,
            "return_3d": return_3d,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_20d": return_20d,
            "relative_volume": relative_volume,
            "rsi_14": rsi_14,
            "distance_sma_20_pct": (
                sma20_extension
            ),
            "distance_sma_50_pct": (
                sma50_extension
            ),
            "atr_expansion": (
                atr_expansion
            ),
        },
    )