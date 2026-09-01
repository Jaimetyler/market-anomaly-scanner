from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scanner.segmentation import summarize_segments


@dataclass(frozen=True)
class WalkForwardConfig:
    date_col: str = "date"
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    min_train_n: int = 30
    min_test_n: int = 10
    top_n_per_fold: int = 25
    min_train_win_rate: float = 0.50
    min_train_median_return: float = 0.0


def _validate_config(config: WalkForwardConfig) -> None:
    if config.min_train_n < 1:
        raise ValueError("min_train_n must be at least 1.")
    if config.min_test_n < 1:
        raise ValueError("min_test_n must be at least 1.")
    if config.top_n_per_fold < 1:
        raise ValueError("top_n_per_fold must be at least 1.")
    if not config.horizons:
        raise ValueError("horizons must not be empty.")
    if min(config.horizons) <= 0:
        raise ValueError("horizons must contain positive integers.")


def _normalize_dates(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in frame.columns:
        raise ValueError(f"Missing date column: {date_col}")

    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")

    if result[date_col].isna().any():
        bad = int(result[date_col].isna().sum())
        raise ValueError(
            f"{date_col} contains {bad} invalid/missing date values."
        )

    result["walkforward_year"] = result[date_col].dt.year.astype(int)
    return result


def _segment_mask(
    frame: pd.DataFrame,
    *,
    group_by: Sequence[str],
    segment_row: pd.Series,
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)

    for column in group_by:
        if column not in frame.columns:
            raise ValueError(f"Missing grouping column: {column}")

        value = segment_row[column]

        if pd.isna(value):
            mask &= frame[column].isna()
        else:
            mask &= frame[column].astype(str).eq(str(value))

    return mask


def _evaluate_test_segment(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> dict[str, float | int]:
    suffix = f"{horizon}d"

    required = [
        f"outcome_available_{suffix}",
        f"contrarian_return_{suffix}",
        f"reversed_{suffix}",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing test outcome columns for {horizon}d: {missing}"
        )

    usable = frame[
        frame[f"outcome_available_{suffix}"].fillna(False)
    ].copy()

    returns = pd.to_numeric(
        usable[f"contrarian_return_{suffix}"],
        errors="coerce",
    )
    valid = returns.notna()
    usable = usable.loc[valid]
    returns = returns.loc[valid]

    n = int(len(returns))

    if n == 0:
        return {
            "test_n": 0,
            "test_win_rate": np.nan,
            "test_mean_return": np.nan,
            "test_median_return": np.nan,
            "test_reversal_rate": np.nan,
        }

    reversed_values = usable[f"reversed_{suffix}"].astype("boolean")

    return {
        "test_n": n,
        "test_win_rate": float((returns > 0).mean()),
        "test_mean_return": float(returns.mean()),
        "test_median_return": float(returns.median()),
        "test_reversal_rate": (
            float(reversed_values.mean())
            if reversed_values.notna().any()
            else np.nan
        ),
    }


def walk_forward_validate(
    outcomes: pd.DataFrame,
    *,
    group_by: Sequence[str],
    config: WalkForwardConfig | None = None,
    first_test_year: int | None = None,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation.

    Each fold trains only on years before the test year, ranks the training
    segments, freezes the top definitions, then evaluates those exact segments
    in the untouched test year.
    """
    config = config or WalkForwardConfig()
    _validate_config(config)

    group_by = list(group_by)
    if not group_by:
        raise ValueError("group_by must contain at least one column.")

    frame = _normalize_dates(outcomes, config.date_col)

    missing_groups = [
        column for column in group_by if column not in frame.columns
    ]
    if missing_groups:
        raise ValueError(f"Missing grouping columns: {missing_groups}")

    # A frozen segment must have a concrete value for every dimension.
    # Do not turn unavailable feature data into a null-valued segment.
    frame = frame.dropna(subset=group_by).copy()

    if frame.empty:
        return pd.DataFrame()

    years = sorted(frame["walkforward_year"].unique().tolist())
    if len(years) < 2:
        raise ValueError(
            "Walk-forward validation requires at least two calendar years."
        )

    if first_test_year is None:
        first_test_year = years[1]

    test_years = [
        year
        for year in years
        if year >= int(first_test_year)
        and any(train_year < year for train_year in years)
    ]

    rows: list[dict] = []

    for test_year in test_years:
        train = frame[frame["walkforward_year"] < test_year].copy()
        test = frame[frame["walkforward_year"] == test_year].copy()

        if train.empty or test.empty:
            continue

        train_start_year = int(train["walkforward_year"].min())
        train_end_year = int(train["walkforward_year"].max())

        summary = summarize_segments(
            train,
            group_by=group_by,
            horizons=config.horizons,
            min_n=config.min_train_n,
        )

        if summary.empty:
            continue

        for horizon in config.horizons:
            candidates = summary[
                (summary["horizon_days"] == int(horizon))
                & (
                    summary["contrarian_win_rate"]
                    >= float(config.min_train_win_rate)
                )
                & (
                    summary["median_contrarian_return"]
                    >= float(config.min_train_median_return)
                )
            ].copy()

            if candidates.empty:
                continue

            candidates = (
                candidates.sort_values(
                    ["edge_score", "n"],
                    ascending=[False, False],
                    kind="stable",
                )
                .head(config.top_n_per_fold)
                .reset_index(drop=True)
            )

            for rank, (_, candidate) in enumerate(
                candidates.iterrows(),
                start=1,
            ):
                mask = _segment_mask(
                    test,
                    group_by=group_by,
                    segment_row=candidate,
                )
                test_segment = test.loc[mask].copy()

                test_metrics = _evaluate_test_segment(
                    test_segment,
                    horizon=horizon,
                )

                test_n = int(test_metrics["test_n"])
                test_win_rate = test_metrics["test_win_rate"]
                test_median = test_metrics["test_median_return"]

                validated = bool(
                    test_n >= config.min_test_n
                    and pd.notna(test_win_rate)
                    and float(test_win_rate) > 0.50
                    and pd.notna(test_median)
                    and float(test_median) > 0.0
                )

                row = {
                    "test_year": int(test_year),
                    "train_start_year": train_start_year,
                    "train_end_year": train_end_year,
                    "horizon_days": int(horizon),
                    "train_rank": rank,
                    **{
                        column: candidate[column]
                        for column in group_by
                    },
                    "train_n": int(candidate["n"]),
                    "train_win_rate": float(
                        candidate["contrarian_win_rate"]
                    ),
                    "train_mean_return": float(
                        candidate["mean_contrarian_return"]
                    ),
                    "train_median_return": float(
                        candidate["median_contrarian_return"]
                    ),
                    "train_reversal_rate": float(
                        candidate["reversal_rate"]
                    ),
                    "train_edge_score": float(
                        candidate["edge_score"]
                    ),
                    **test_metrics,
                    "validated": validated,
                }

                if (
                    pd.notna(test_metrics["test_median_return"])
                    and candidate["median_contrarian_return"] != 0
                ):
                    row["median_retention_ratio"] = float(
                        test_metrics["test_median_return"]
                        / candidate["median_contrarian_return"]
                    )
                else:
                    row["median_retention_ratio"] = np.nan

                if pd.notna(test_metrics["test_win_rate"]):
                    row["win_rate_delta"] = float(
                        test_metrics["test_win_rate"]
                        - candidate["contrarian_win_rate"]
                    )
                else:
                    row["win_rate_delta"] = np.nan

                rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "test_year",
            "horizon_days",
            "validated",
            "train_rank",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def summarize_walk_forward(
    results: pd.DataFrame,
    *,
    group_by: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse fold-level results into persistence statistics for each setup.
    """
    group_by = list(group_by)

    required = [
        *group_by,
        "horizon_days",
        "test_year",
        "test_n",
        "test_win_rate",
        "test_median_return",
        "validated",
    ]
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise ValueError(
            f"Walk-forward results missing columns: {missing}"
        )

    if results.empty:
        return pd.DataFrame()

    keys = [*group_by, "horizon_days"]
    rows: list[dict] = []

    complete_results = results.dropna(subset=group_by)

    grouped = complete_results.groupby(
        keys[0] if len(keys) == 1 else keys,
        observed=True,
        sort=True,
    )

    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        key_values = dict(zip(keys, key))
        valid_test = frame[frame["test_n"] > 0].copy()

        if valid_test.empty:
            continue

        folds = int(len(valid_test))
        validated_folds = int(valid_test["validated"].sum())
        weighted_n = int(valid_test["test_n"].sum())

        weighted_win_rate = (
            float(
                np.average(
                    valid_test["test_win_rate"],
                    weights=valid_test["test_n"],
                )
            )
            if weighted_n > 0
            else np.nan
        )

        rows.append(
            {
                **key_values,
                "folds_tested": folds,
                "validated_folds": validated_folds,
                "validation_rate": validated_folds / folds,
                "total_test_n": weighted_n,
                "weighted_test_win_rate": weighted_win_rate,
                "median_test_median_return": float(
                    valid_test["test_median_return"].median()
                ),
                "mean_test_median_return": float(
                    valid_test["test_median_return"].mean()
                ),
                "worst_test_median_return": float(
                    valid_test["test_median_return"].min()
                ),
                "best_test_median_return": float(
                    valid_test["test_median_return"].max()
                ),
                "first_test_year": int(
                    valid_test["test_year"].min()
                ),
                "last_test_year": int(
                    valid_test["test_year"].max()
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        [
            "horizon_days",
            "validation_rate",
            "total_test_n",
            "median_test_median_return",
        ],
        ascending=[True, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
