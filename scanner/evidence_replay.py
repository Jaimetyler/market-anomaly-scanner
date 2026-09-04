from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_WALKFORWARD_PATH = Path(
    "data/research/walkforward_reports/all_walkforward_folds.csv"
)

CONDITION_COLUMNS = (
    "initial_setup",
    "rvol_bucket",
    "sma20_extension_bucket",
    "move_1d_bucket",
    "move_3d_bucket",
    "move_5d_bucket",
)


@dataclass(frozen=True)
class HistoricalEvidenceMatch:
    segmentation: str
    test_year: int
    horizon_days: int
    train_rank: int

    train_n: int
    train_win_rate: float
    train_mean_return: float
    train_median_return: float
    train_reversal_rate: float
    train_edge_score: float

    initial_setup: str | None
    rvol_bucket: str | None
    sma20_extension_bucket: str | None
    move_1d_bucket: str | None
    move_3d_bucket: str | None
    move_5d_bucket: str | None

    @property
    def condition_count(self) -> int:
        return sum(
            value is not None
            for value in (
                self.initial_setup,
                self.rvol_bucket,
                self.sma20_extension_bucket,
                self.move_1d_bucket,
                self.move_3d_bucket,
                self.move_5d_bucket,
            )
        )


def _clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _as_int(value: object) -> int:
    if pd.isna(value):
        return 0

    return int(value)


def _as_float(value: object) -> float:
    if pd.isna(value):
        return 0.0

    return float(value)


def load_walkforward_folds(
    path: str | Path = DEFAULT_WALKFORWARD_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Walk-forward fold file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "segmentation",
        "test_year",
        "horizon_days",
        "train_rank",
        "train_n",
        "train_win_rate",
        "train_mean_return",
        "train_median_return",
        "train_reversal_rate",
        "train_edge_score",
        *CONDITION_COLUMNS,
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "Walk-forward fold file is missing required columns: "
            + ", ".join(missing)
        )

    return df


def event_features_from_row(row: pd.Series) -> dict[str, str | None]:
    return {
        column: _clean_text(row.get(column))
        for column in CONDITION_COLUMNS
    }


def fold_matches_event(
    fold_row: pd.Series,
    event_features: dict[str, str | None],
) -> bool:
    """
    Return True when every condition used by the fold is satisfied
    by the historical event.

    Null condition columns in the fold are ignored.
    """
    for column in CONDITION_COLUMNS:
        required_value = _clean_text(fold_row.get(column))

        if required_value is None:
            continue

        if event_features.get(column) != required_value:
            return False

    return True


def match_historical_event(
    event_row: pd.Series,
    folds_df: pd.DataFrame,
    *,
    horizons: Iterable[int] | None = None,
) -> list[HistoricalEvidenceMatch]:
    """
    Match one historical event against evidence that existed BEFORE
    that event's test year.

    The walk-forward report already encodes the leakage-safe training
    window for each test year, so matching by test_year prevents future
    years from contributing to the event's evidence.
    """
    event_date = pd.to_datetime(event_row["date"])
    test_year = int(event_date.year)

    eligible = folds_df[
        folds_df["test_year"].astype(int) == test_year
    ]

    if horizons is not None:
        allowed_horizons = {int(h) for h in horizons}
        eligible = eligible[
            eligible["horizon_days"].astype(int).isin(allowed_horizons)
        ]

    features = event_features_from_row(event_row)

    matches: list[HistoricalEvidenceMatch] = []

    for _, fold in eligible.iterrows():
        if not fold_matches_event(fold, features):
            continue

        matches.append(
            HistoricalEvidenceMatch(
                segmentation=_clean_text(fold["segmentation"]) or "",
                test_year=_as_int(fold["test_year"]),
                horizon_days=_as_int(fold["horizon_days"]),
                train_rank=_as_int(fold["train_rank"]),
                train_n=_as_int(fold["train_n"]),
                train_win_rate=_as_float(fold["train_win_rate"]),
                train_mean_return=_as_float(fold["train_mean_return"]),
                train_median_return=_as_float(
                    fold["train_median_return"]
                ),
                train_reversal_rate=_as_float(
                    fold["train_reversal_rate"]
                ),
                train_edge_score=_as_float(fold["train_edge_score"]),
                initial_setup=_clean_text(fold.get("initial_setup")),
                rvol_bucket=_clean_text(fold.get("rvol_bucket")),
                sma20_extension_bucket=_clean_text(
                    fold.get("sma20_extension_bucket")
                ),
                move_1d_bucket=_clean_text(
                    fold.get("move_1d_bucket")
                ),
                move_3d_bucket=_clean_text(
                    fold.get("move_3d_bucket")
                ),
                move_5d_bucket=_clean_text(
                    fold.get("move_5d_bucket")
                ),
            )
        )

    matches.sort(
        key=lambda match: (
            match.train_edge_score,
            match.train_n,
            match.condition_count,
        ),
        reverse=True,
    )

    return matches


def best_match_by_horizon(
    matches: Iterable[HistoricalEvidenceMatch],
) -> dict[int, HistoricalEvidenceMatch]:
    """
    Select the strongest training-only evidence match for each horizon.

    Because match_historical_event sorts by training edge first, the
    first match encountered for a horizon is its current anchor.
    """
    best: dict[int, HistoricalEvidenceMatch] = {}

    for match in matches:
        if match.horizon_days not in best:
            best[match.horizon_days] = match

    return best