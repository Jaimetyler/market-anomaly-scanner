from __future__ import annotations

from pathlib import Path

import pandas as pd

from scanner.evidence_replay import CONDITION_COLUMNS


DEFAULT_EVENTS_PATH = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)

DEFAULT_FOLDS_PATH = Path(
    "data/research/walkforward_reports/all_walkforward_folds.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/research/replay_reports/historical_evidence_replay.csv"
)


def _condition_mask(
    events: pd.DataFrame,
    fold: pd.Series,
) -> pd.Series:
    mask = pd.Series(True, index=events.index)

    for column in CONDITION_COLUMNS:
        value = fold.get(column)

        if pd.isna(value):
            continue

        mask &= events[column].astype("string") == str(value)

    return mask


def build_historical_replay(
    events_df: pd.DataFrame,
    folds_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a leakage-safe historical replay dataset.

    Each output row represents one historical event matched to one
    walk-forward evidence definition at one horizon.

    The evidence columns come only from the training window that
    preceded the event's test year. Realized outcome columns come from
    the event itself and are never used for matching/ranking.
    """
    events = events_df.copy()
    folds = folds_df.copy()

    events["date"] = pd.to_datetime(events["date"])
    events["test_year"] = events["date"].dt.year.astype(int)

    rows: list[pd.DataFrame] = []

    for test_year in sorted(folds["test_year"].dropna().unique()):
        test_year = int(test_year)

        year_events = events[
            events["test_year"] == test_year
        ].copy()

        year_folds = folds[
            folds["test_year"].astype(int) == test_year
        ]

        if year_events.empty or year_folds.empty:
            continue

        print(
            f"Replay year {test_year}: "
            f"{len(year_events):,} events, "
            f"{len(year_folds):,} fold definitions"
        )

        for _, fold in year_folds.iterrows():
            horizon = int(fold["horizon_days"])

            outcome_col = f"contrarian_return_{horizon}d"
            availability_col = f"outcome_available_{horizon}d"

            if outcome_col not in year_events.columns:
                continue

            mask = _condition_mask(year_events, fold)

            if availability_col in year_events.columns:
                mask &= year_events[availability_col].fillna(False).astype(bool)

            matched = year_events.loc[mask]

            if matched.empty:
                continue

            block = pd.DataFrame(
                {
                    "event_id": matched["event_id"].values,
                    "ticker": matched["ticker"].values,
                    "date": matched["date"].values,
                    "test_year": test_year,
                    "event_direction": matched[
                        "event_direction"
                    ].values,
                    "initial_setup": matched[
                        "initial_setup"
                    ].values,
                    "horizon_days": horizon,
                    "segmentation": fold["segmentation"],
                    "train_rank": int(fold["train_rank"]),
                    "train_n": int(fold["train_n"]),
                    "train_win_rate": float(
                        fold["train_win_rate"]
                    ),
                    "train_mean_return": float(
                        fold["train_mean_return"]
                    ),
                    "train_median_return": float(
                        fold["train_median_return"]
                    ),
                    "train_reversal_rate": float(
                        fold["train_reversal_rate"]
                    ),
                    "train_edge_score": float(
                        fold["train_edge_score"]
                    ),
                    "train_start_year": int(
                        fold["train_start_year"]
                    ),
                    "train_end_year": int(
                        fold["train_end_year"]
                    ),
                    "realized_contrarian_return": matched[
                        outcome_col
                    ].values,
                }
            )

            block["realized_win"] = (
                block["realized_contrarian_return"] > 0
            )

            condition_parts = []

            for column in CONDITION_COLUMNS:
                value = fold.get(column)

                if pd.isna(value):
                    continue

                condition_parts.append(
                    f"{column}={value}"
                )

            block["condition"] = " | ".join(
                condition_parts
            )

            block["condition_count"] = len(
                condition_parts
            )

            rows.append(block)

    if not rows:
        return pd.DataFrame()

    replay = pd.concat(
        rows,
        ignore_index=True,
    )

    replay = replay.sort_values(
        [
            "date",
            "event_id",
            "horizon_days",
            "train_edge_score",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return replay


def load_inputs(
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    folds_path: str | Path = DEFAULT_FOLDS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(events_path)
    folds = pd.read_csv(folds_path)

    return events, folds


def write_replay(
    replay_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_df.to_csv(
        output_path,
        index=False,
    )

    return output_path