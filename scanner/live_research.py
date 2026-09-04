from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MEMBERS_PATH = Path(
    "data/research/consolidation_reports/archetype_members.csv"
)

BUCKET_COLUMNS = (
    "price_bucket",
    "rvol_bucket",
    "sma20_extension_bucket",
    "move_1d_bucket",
    "move_3d_bucket",
    "move_5d_bucket",
)


@dataclass(frozen=True)
class LiveResearchMatch:
    signal_id: str
    archetype_id: str
    segmentation: str
    initial_setup: str
    horizon_days: int
    condition: str

    historical_n: int
    historical_win_rate: float
    historical_median_return: float

    oos_n: int
    oos_win_rate: float
    oos_median_return: float
    worst_oos_median_return: float

    folds_tested: int
    validated_folds: int
    validation_rate: float
    promotion_score: float


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    return text


def _as_int(value: Any) -> int:
    if value is None:
        return 0

    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    return int(float(value))


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0

    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass

    return float(value)


def price_bucket(price: float | None) -> str | None:
    """
    Price buckets are currently not represented in the promoted
    archetype-member set.

    This helper is retained so live feature construction stays
    compatible with future research outputs that may promote
    price-segmented signals.
    """
    if price is None:
        return None

    if price < 1:
        return "<$1"

    if price < 5:
        return "$1-5"

    if price < 10:
        return "$5-10"

    if price < 20:
        return "$10-20"

    if price < 50:
        return "$20-50"

    return "$50+"


def rvol_bucket(rvol: float | None) -> str | None:
    """
    Match the RVOL segmentation labels used by the historical
    research pipeline.
    """
    if rvol is None:
        return None

    if rvol < 1:
        return "<1x"

    if rvol < 1.5:
        return "1-1.5x"

    if rvol < 2:
        return "1.5-2x"

    if rvol < 3:
        return "2-3x"

    if rvol < 5:
        return "3-5x"

    if rvol < 10:
        return "5-10x"

    return "10x+"


def sma20_extension_bucket(
    extension_pct: float | None,
) -> str | None:
    """
    Only the extreme SMA20 buckets currently survive promotion.

    Returning None for the middle region is intentional: there is
    currently no promoted SMA20 condition to match there.
    """
    if extension_pct is None:
        return None

    if extension_pct < -50:
        return "<-50%"

    if extension_pct >= 50:
        return "+50%+"

    return None


def move_bucket(move_pct: float | None) -> str | None:
    """
    The promoted research set currently contains only the extreme
    move buckets: below -50% and +100% or greater.

    Returning None between those thresholds prevents us from
    inventing bucket labels that are not present in the promoted
    evidence.
    """
    if move_pct is None:
        return None

    if move_pct < -50:
        return "<-50%"

    if move_pct >= 100:
        return "+100%+"

    return None


def build_live_features(
    snapshot: dict[str, Any],
    initial_setup: str,
) -> dict[str, str | None]:
    return {
        "initial_setup": initial_setup,
        "price_bucket": price_bucket(
            snapshot.get("price")
        ),
        "rvol_bucket": rvol_bucket(
            snapshot.get("relative_volume")
        ),
        "sma20_extension_bucket": (
            sma20_extension_bucket(
                snapshot.get("distance_sma_20_pct")
            )
        ),
        "move_1d_bucket": move_bucket(
            snapshot.get("return_1d")
        ),
        "move_3d_bucket": move_bucket(
            snapshot.get("return_3d")
        ),
        "move_5d_bucket": move_bucket(
            snapshot.get("return_5d")
        ),
    }


def load_promoted_archetype_members(
    path: str | Path = DEFAULT_MEMBERS_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Research archetype file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "signal_id",
        "archetype_id",
        "segmentation",
        "initial_setup",
        "horizon_days",
        "condition",
        "disposition",
    }

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "Archetype member file is missing required "
            f"columns: {', '.join(missing)}"
        )

    df = df[
        df["disposition"]
        .astype(str)
        .str.upper()
        .eq("PROMOTE")
    ].copy()

    return df


def row_matches_live_features(
    row: pd.Series,
    live_features: dict[str, str | None],
) -> bool:
    required_setup = _clean_text(
        row.get("initial_setup")
    )

    if required_setup is not None:
        if (
            live_features.get("initial_setup")
            != required_setup
        ):
            return False

    for column in BUCKET_COLUMNS:
        required_value = _clean_text(
            row.get(column)
        )

        if required_value is None:
            continue

        live_value = live_features.get(column)

        if live_value != required_value:
            return False

    return True


def match_live_snapshot(
    snapshot: dict[str, Any],
    initial_setup: str,
    members_df: pd.DataFrame | None = None,
) -> list[LiveResearchMatch]:
    if members_df is None:
        members_df = (
            load_promoted_archetype_members()
        )

    live_features = build_live_features(
        snapshot=snapshot,
        initial_setup=initial_setup,
    )

    matches: list[LiveResearchMatch] = []

    for _, row in members_df.iterrows():
        if not row_matches_live_features(
            row=row,
            live_features=live_features,
        ):
            continue

        matches.append(
            LiveResearchMatch(
                signal_id=(
                    _clean_text(
                        row.get("signal_id")
                    )
                    or ""
                ),
                archetype_id=(
                    _clean_text(
                        row.get("archetype_id")
                    )
                    or ""
                ),
                segmentation=(
                    _clean_text(
                        row.get("segmentation")
                    )
                    or ""
                ),
                initial_setup=(
                    _clean_text(
                        row.get("initial_setup")
                    )
                    or ""
                ),
                horizon_days=_as_int(
                    row.get("horizon_days")
                ),
                condition=(
                    _clean_text(
                        row.get("condition")
                    )
                    or ""
                ),
                historical_n=_as_int(
                    row.get("n")
                ),
                historical_win_rate=_as_float(
                    row.get("win_rate")
                ),
                historical_median_return=_as_float(
                    row.get(
                        "median_contrarian_return"
                    )
                ),
                oos_n=_as_int(
                    row.get("total_test_n")
                ),
                oos_win_rate=_as_float(
                    row.get(
                        "weighted_test_win_rate"
                    )
                ),
                oos_median_return=_as_float(
                    row.get(
                        "median_test_median_return"
                    )
                ),
                worst_oos_median_return=_as_float(
                    row.get(
                        "worst_test_median_return"
                    )
                ),
                folds_tested=_as_int(
                    row.get("folds_tested")
                ),
                validated_folds=_as_int(
                    row.get("validated_folds")
                ),
                validation_rate=_as_float(
                    row.get("validation_rate")
                ),
                promotion_score=_as_float(
                    row.get("promotion_score")
                ),
            )
        )

    matches.sort(
        key=lambda item: (
            item.promotion_score,
            item.validation_rate,
            item.oos_n,
        ),
        reverse=True,
    )

    return matches


def best_match_per_archetype(
    matches: list[LiveResearchMatch],
) -> list[LiveResearchMatch]:
    """
    Collapse overlapping promoted definitions so one live stock
    does not appear to have multiple independent confirmations
    simply because several near-equivalent definitions belong to
    the same consolidated archetype.
    """
    best: dict[str, LiveResearchMatch] = {}

    for match in matches:
        current = best.get(
            match.archetype_id
        )

        if current is None:
            best[match.archetype_id] = match
            continue

        candidate_key = (
            match.promotion_score,
            match.validation_rate,
            match.oos_n,
        )

        current_key = (
            current.promotion_score,
            current.validation_rate,
            current.oos_n,
        )

        if candidate_key > current_key:
            best[match.archetype_id] = match

    result = list(best.values())

    result.sort(
        key=lambda item: (
            item.promotion_score,
            item.validation_rate,
            item.oos_n,
        ),
        reverse=True,
    )

    return result