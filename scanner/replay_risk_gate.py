from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_SAMPLES_PATH = Path(
    "data/research/replay_reports/raw_median_top_n_samples.csv"
)

DEFAULT_EVENTS_PATH = Path(
    "data/research/edge_reports/event_outcomes_adapted.csv"
)

DEFAULT_REPORT_DIR = Path(
    "data/research/replay_reports"
)

FOCUS_TOP_N = 5

DEVELOPMENT_YEARS = (
    2021,
    2022,
    2023,
    2024,
)

VALIDATION_YEAR = 2025


@dataclass(frozen=True)
class RiskGate:
    name: str
    description: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def _always(
    df: pd.DataFrame,
) -> pd.Series:
    return pd.Series(
        True,
        index=df.index,
    )


def _price_ge(
    threshold: float,
) -> Callable[[pd.DataFrame], pd.Series]:
    def predicate(
        df: pd.DataFrame,
    ) -> pd.Series:
        return (
            pd.to_numeric(
                df["event_close"],
                errors="coerce",
            )
            >= threshold
        )

    return predicate


def _rvol_lt_10(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        pd.to_numeric(
            df["rvol"],
            errors="coerce",
        )
        < 10.0
    )


def _move5_lt_100(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        pd.to_numeric(
            df["return_5d_event"],
            errors="coerce",
        )
        < 100.0
    )


def _and(
    *predicates: Callable[
        [pd.DataFrame],
        pd.Series,
    ],
) -> Callable[
    [pd.DataFrame],
    pd.Series,
]:
    def combined(
        df: pd.DataFrame,
    ) -> pd.Series:
        mask = pd.Series(
            True,
            index=df.index,
        )

        for predicate in predicates:
            mask &= predicate(
                df
            ).fillna(False)

        return mask

    return combined


RISK_GATES = (
    RiskGate(
        name="BASELINE",
        description="No structural risk gate",
        predicate=_always,
    ),
    RiskGate(
        name="PRICE_GE_1",
        description="Event close >= $1",
        predicate=_price_ge(1.0),
    ),
    RiskGate(
        name="PRICE_GE_2",
        description="Event close >= $2",
        predicate=_price_ge(2.0),
    ),
    RiskGate(
        name="PRICE_GE_5",
        description="Event close >= $5",
        predicate=_price_ge(5.0),
    ),
    RiskGate(
        name="RVOL_LT_10",
        description="Relative volume < 10x",
        predicate=_rvol_lt_10,
    ),
    RiskGate(
        name="MOVE5_LT_100",
        description="5D event move < +100%",
        predicate=_move5_lt_100,
    ),
    RiskGate(
        name="PRICE_GE_1_RVOL_LT_10",
        description="$1+ and RVOL < 10x",
        predicate=_and(
            _price_ge(1.0),
            _rvol_lt_10,
        ),
    ),
    RiskGate(
        name="PRICE_GE_2_RVOL_LT_10",
        description="$2+ and RVOL < 10x",
        predicate=_and(
            _price_ge(2.0),
            _rvol_lt_10,
        ),
    ),
    RiskGate(
        name="PRICE_GE_5_RVOL_LT_10",
        description="$5+ and RVOL < 10x",
        predicate=_and(
            _price_ge(5.0),
            _rvol_lt_10,
        ),
    ),
    RiskGate(
        name="PRICE_GE_2_MOVE5_LT_100",
        description="$2+ and 5D move < +100%",
        predicate=_and(
            _price_ge(2.0),
            _move5_lt_100,
        ),
    ),
    RiskGate(
        name="PRICE_GE_2_RVOL_LT_10_MOVE5_LT_100",
        description="$2+, RVOL < 10x, 5D move < +100%",
        predicate=_and(
            _price_ge(2.0),
            _rvol_lt_10,
            _move5_lt_100,
        ),
    ),
)


def load_event_risk_metadata(
    path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Event outcomes file not found: {path}"
        )

    columns = [
        "event_id",
        "event_close",
        "rvol",
        "return_5d_event",
        "price_bucket",
        "rvol_bucket",
        "move_5d_bucket",
        "initial_setup",
        "event_direction",
    ]

    header = pd.read_csv(
        path,
        nrows=0,
    )

    missing = sorted(
        set(columns)
        - set(header.columns)
    )

    if missing:
        raise ValueError(
            "Event outcomes file is missing: "
            + ", ".join(missing)
        )

    print(
        f"Loading risk metadata: {path}"
    )

    metadata = pd.read_csv(
        path,
        usecols=columns,
    )

    metadata = (
        metadata.drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Loaded metadata for "
        f"{metadata['event_id'].nunique():,} events."
    )

    return metadata


def load_top5_samples(
    samples_path: str | Path = DEFAULT_SAMPLES_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
) -> pd.DataFrame:
    samples_path = Path(
        samples_path
    )

    if not samples_path.exists():
        raise FileNotFoundError(
            f"RAW_MEDIAN samples not found: "
            f"{samples_path}"
        )

    print(
        f"Loading RAW_MEDIAN samples: "
        f"{samples_path}"
    )

    samples = pd.read_csv(
        samples_path,
        parse_dates=["date"],
    )

    samples = samples[
        samples["top_n"] == FOCUS_TOP_N
    ].copy()

    metadata = load_event_risk_metadata(
        events_path
    )

    metadata_columns = [
        "event_id",
        "event_close",
        "rvol",
        "return_5d_event",
        "price_bucket",
        "rvol_bucket",
        "move_5d_bucket",
        "initial_setup",
        "event_direction",
    ]

    existing = {
        column
        for column in metadata_columns
        if (
            column in samples.columns
            and column != "event_id"
        )
    }

    if existing:
        samples = samples.drop(
            columns=list(
                existing
            )
        )

    samples = samples.merge(
        metadata[
            metadata_columns
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    numeric_columns = [
        "test_year",
        "horizon_days",
        "event_close",
        "rvol",
        "return_5d_event",
        "realized_contrarian_return",
    ]

    for column in numeric_columns:
        samples[column] = (
            pd.to_numeric(
                samples[column],
                errors="coerce",
            )
        )

    print(
        f"Top-5 rows: {len(samples):,}"
    )

    return samples


def apply_gate(
    samples: pd.DataFrame,
    gate: RiskGate,
) -> pd.DataFrame:
    mask = gate.predicate(
        samples
    )

    return samples[
        mask.fillna(False)
    ].copy()


def tail_mean(
    values: pd.Series,
    quantile: float = 0.05,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    cutoff = clean.quantile(
        quantile
    )

    tail = clean[
        clean <= cutoff
    ]

    if tail.empty:
        return np.nan

    return float(
        tail.mean()
    )


def profit_factor(
    values: pd.Series,
) -> float:
    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    gross_profit = clean[
        clean > 0
    ].sum()

    gross_loss = clean[
        clean < 0
    ].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf

        return np.nan

    return float(
        gross_profit
        / abs(gross_loss)
    )


def summarize_group(
    group: pd.DataFrame,
    *,
    baseline_n: int,
) -> dict[str, float]:
    returns = (
        pd.to_numeric(
            group[
                "realized_contrarian_return"
            ],
            errors="coerce",
        )
        .dropna()
    )

    if returns.empty:
        return {
            "observations": 0,
            "coverage": 0.0,
        }

    return {
        "observations": int(
            len(returns)
        ),
        "coverage": (
            len(returns)
            / baseline_n
            if baseline_n
            else np.nan
        ),
        "win_rate": float(
            (returns > 0).mean()
        ),
        "mean_return": float(
            returns.mean()
        ),
        "median_return": float(
            returns.median()
        ),
        "p05_return": float(
            returns.quantile(
                0.05
            )
        ),
        "p10_return": float(
            returns.quantile(
                0.10
            )
        ),
        "cvar_05": tail_mean(
            returns,
            0.05,
        ),
        "worst_return": float(
            returns.min()
        ),
        "loss_le_50_rate": float(
            (returns <= -50).mean()
        ),
        "loss_le_100_rate": float(
            (returns <= -100).mean()
        ),
        "loss_le_200_rate": float(
            (returns <= -200).mean()
        ),
        "profit_factor": profit_factor(
            returns
        ),
    }


def build_gate_summary(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for period_name, years in (
        (
            "DEVELOPMENT_2021_2024",
            DEVELOPMENT_YEARS,
        ),
        (
            "VALIDATION_2025",
            (VALIDATION_YEAR,),
        ),
    ):
        period = samples[
            samples[
                "test_year"
            ].isin(
                years
            )
        ].copy()

        for horizon in sorted(
            period[
                "horizon_days"
            ]
            .dropna()
            .unique()
        ):
            horizon_df = period[
                period[
                    "horizon_days"
                ]
                == horizon
            ].copy()

            baseline_n = len(
                horizon_df
            )

            for gate in RISK_GATES:
                filtered = apply_gate(
                    horizon_df,
                    gate,
                )

                metrics = summarize_group(
                    filtered,
                    baseline_n=baseline_n,
                )

                rows.append(
                    {
                        "period": period_name,
                        "horizon_days": int(
                            horizon
                        ),
                        "gate": gate.name,
                        "description": gate.description,
                        **metrics,
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_bucket_summary(
    samples: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    bucket_columns = (
        "price_bucket",
        "rvol_bucket",
        "move_5d_bucket",
        "initial_setup",
    )

    for column in bucket_columns:
        for (
            horizon,
            bucket_value,
        ), group in samples.groupby(
            [
                "horizon_days",
                column,
            ],
            observed=True,
            dropna=False,
        ):
            returns = (
                pd.to_numeric(
                    group[
                        "realized_contrarian_return"
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if returns.empty:
                continue

            rows.append(
                {
                    "feature": column,
                    "bucket": str(
                        bucket_value
                    ),
                    "horizon_days": int(
                        horizon
                    ),
                    "observations": len(
                        returns
                    ),
                    "win_rate": (
                        returns > 0
                    ).mean(),
                    "mean_return": (
                        returns.mean()
                    ),
                    "median_return": (
                        returns.median()
                    ),
                    "p05_return": (
                        returns.quantile(
                            0.05
                        )
                    ),
                    "cvar_05": tail_mean(
                        returns
                    ),
                    "loss_le_100_rate": (
                        returns <= -100
                    ).mean(),
                    "loss_le_200_rate": (
                        returns <= -200
                    ).mean(),
                    "profit_factor": (
                        profit_factor(
                            returns
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def write_reports(
    gate_summary: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_REPORT_DIR,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate_summary.to_csv(
        output_dir
        / "raw_median_risk_gate_summary.csv",
        index=False,
    )

    bucket_summary.to_csv(
        output_dir
        / "raw_median_risk_bucket_summary.csv",
        index=False,
    )