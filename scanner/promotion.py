from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PromotionConfig:
    """
    Conservative promotion thresholds for research setups.
    """

    promote_min_validation_rate: float = 0.60
    promote_min_folds: int = 2
    promote_min_test_n: int = 30
    promote_min_weighted_win_rate: float = 0.55
    promote_min_test_median_return: float = 0.0

    watch_min_validation_rate: float = 0.40
    watch_min_folds: int = 1
    watch_min_test_n: int = 10
    watch_min_weighted_win_rate: float = 0.50


def _normalize_key_value(value):
    if pd.isna(value):
        return None
    return str(value)


def _segment_key(
    row: pd.Series,
    *,
    segmentation: str,
    group_columns: Sequence[str],
) -> tuple:
    return (
        segmentation,
        int(row["horizon_days"]),
        *(
            _normalize_key_value(row.get(column))
            for column in group_columns
        ),
    )


def classify_evidence(
    row: pd.Series,
    *,
    config: PromotionConfig,
) -> str:
    """
    Classify one merged research row as PROMOTE, WATCH, or REJECT.
    """

    globally_supported = bool(
        row.get("globally_supported", False)
    )

    validation_rate = float(
        row.get("validation_rate", 0.0)
        if pd.notna(row.get("validation_rate", np.nan))
        else 0.0
    )

    folds_tested = int(
        row.get("folds_tested", 0)
        if pd.notna(row.get("folds_tested", np.nan))
        else 0
    )

    total_test_n = int(
        row.get("total_test_n", 0)
        if pd.notna(row.get("total_test_n", np.nan))
        else 0
    )

    weighted_test_win_rate = float(
        row.get("weighted_test_win_rate", 0.0)
        if pd.notna(row.get("weighted_test_win_rate", np.nan))
        else 0.0
    )

    median_test_median_return = float(
        row.get("median_test_median_return", 0.0)
        if pd.notna(
            row.get("median_test_median_return", np.nan)
        )
        else 0.0
    )

    if (
        globally_supported
        and validation_rate >= config.promote_min_validation_rate
        and folds_tested >= config.promote_min_folds
        and total_test_n >= config.promote_min_test_n
        and weighted_test_win_rate
        >= config.promote_min_weighted_win_rate
        and median_test_median_return
        > config.promote_min_test_median_return
    ):
        return "PROMOTE"

    if (
        globally_supported
        or (
            validation_rate >= config.watch_min_validation_rate
            and folds_tested >= config.watch_min_folds
            and total_test_n >= config.watch_min_test_n
            and weighted_test_win_rate
            >= config.watch_min_weighted_win_rate
        )
    ):
        return "WATCH"

    return "REJECT"


def combine_evidence(
    *,
    significance: pd.DataFrame,
    persistence: pd.DataFrame,
    grouping_map: dict[str, list[str]],
    config: PromotionConfig | None = None,
) -> pd.DataFrame:
    """
    Merge global statistical-significance results with walk-forward
    persistence results and assign a final research disposition.

    Matching is done by:
      segmentation
      horizon_days
      the grouping columns for that segmentation
    """
    config = config or PromotionConfig()

    required_sig = {
        "segmentation",
        "horizon_days",
        "globally_supported",
        "confidence_score",
        "global_q_value",
        "n",
        "win_rate",
        "median_contrarian_return",
    }

    required_persist = {
        "segmentation",
        "horizon_days",
        "folds_tested",
        "validation_rate",
        "total_test_n",
        "weighted_test_win_rate",
        "median_test_median_return",
    }

    missing_sig = sorted(required_sig - set(significance.columns))
    missing_persist = sorted(
        required_persist - set(persistence.columns)
    )

    if missing_sig:
        raise ValueError(
            f"Significance data missing columns: {missing_sig}"
        )
    if missing_persist:
        raise ValueError(
            f"Persistence data missing columns: {missing_persist}"
        )

    sig_rows: list[pd.Series] = []
    persist_lookup: dict[tuple, pd.Series] = {}

    for _, row in persistence.iterrows():
        segmentation = str(row["segmentation"])
        group_columns = grouping_map.get(segmentation)

        if group_columns is None:
            continue

        key = _segment_key(
            row,
            segmentation=segmentation,
            group_columns=group_columns,
        )
        persist_lookup[key] = row

    for _, sig_row in significance.iterrows():
        segmentation = str(sig_row["segmentation"])
        group_columns = grouping_map.get(segmentation)

        if group_columns is None:
            continue

        key = _segment_key(
            sig_row,
            segmentation=segmentation,
            group_columns=group_columns,
        )

        merged = sig_row.copy()
        persist_row = persist_lookup.get(key)

        if persist_row is not None:
            for column in persist_row.index:
                if column in {
                    "segmentation",
                    "horizon_days",
                    *group_columns,
                }:
                    continue

                if column not in merged.index:
                    merged[column] = persist_row[column]
                else:
                    merged[f"wf_{column}"] = persist_row[column]

        sig_rows.append(merged)

    if not sig_rows:
        return pd.DataFrame()

    combined = pd.DataFrame(sig_rows)

    for column, default in (
        ("folds_tested", 0),
        ("validated_folds", 0),
        ("validation_rate", 0.0),
        ("total_test_n", 0),
        ("weighted_test_win_rate", np.nan),
        ("median_test_median_return", np.nan),
        ("worst_test_median_return", np.nan),
    ):
        if column not in combined.columns:
            combined[column] = default

    combined["disposition"] = combined.apply(
        lambda row: classify_evidence(
            row,
            config=config,
        ),
        axis=1,
    )

    disposition_rank = {
        "PROMOTE": 0,
        "WATCH": 1,
        "REJECT": 2,
    }

    combined["_disposition_rank"] = combined[
        "disposition"
    ].map(disposition_rank)

    combined["promotion_score"] = (
        combined["globally_supported"].astype(int) * 100.0
        + combined["validation_rate"].fillna(0.0) * 40.0
        + (
            np.maximum(
                combined["weighted_test_win_rate"].fillna(0.0)
                - 0.50,
                0.0,
            )
            * 100.0
        )
        + (
            np.maximum(
                combined["median_test_median_return"].fillna(0.0),
                0.0,
            )
            * 100.0
        )
        + np.log1p(combined["total_test_n"].fillna(0.0))
        + combined["confidence_score"].fillna(0.0)
    )

    combined = combined.sort_values(
        [
            "_disposition_rank",
            "promotion_score",
            "horizon_days",
        ],
        ascending=[True, False, True],
        kind="stable",
    ).drop(columns=["_disposition_rank"])

    return combined.reset_index(drop=True)


def build_promotion_report(
    *,
    significance_path: Path,
    persistence_path: Path,
    output_path: Path,
    grouping_map: dict[str, list[str]],
    config: PromotionConfig | None = None,
) -> pd.DataFrame:
    if not significance_path.exists():
        raise FileNotFoundError(
            f"Missing significance file: {significance_path}"
        )
    if not persistence_path.exists():
        raise FileNotFoundError(
            f"Missing persistence file: {persistence_path}"
        )

    significance = pd.read_csv(
        significance_path,
        low_memory=False,
    )
    persistence = pd.read_csv(
        persistence_path,
        low_memory=False,
    )

    combined = combine_evidence(
        significance=significance,
        persistence=persistence,
        grouping_map=grouping_map,
        config=config,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    combined.to_csv(output_path, index=False)

    return combined