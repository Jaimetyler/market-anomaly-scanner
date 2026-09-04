from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_POLICY_SAMPLES_PATH = Path(
    "data/research/replay_reports/scanner_policy_samples.csv"
)

DEFAULT_REPORT_DIR = Path(
    "data/research/replay_reports"
)

POLICY = "PRICE_GE_5"
TOP_N = 5


def load_policy_samples(
    path: str | Path = DEFAULT_POLICY_SAMPLES_PATH,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Policy samples not found: {path}"
        )

    print(
        f"Loading policy samples: {path}"
    )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    df = df[
        df["top_n"] == TOP_N
    ].copy()

    print(
        f"Loaded {len(df):,} top-{TOP_N} rows."
    )

    return df


def compare_baseline_to_policy(
    samples: pd.DataFrame,
    *,
    policy: str = POLICY,
) -> pd.DataFrame:
    baseline = samples[
        samples["policy"] == "BASELINE"
    ].copy()

    gated = samples[
        samples["policy"] == policy
    ].copy()

    key = [
        "date",
        "horizon_days",
    ]

    baseline_members = (
        baseline.groupby(
            key,
            observed=True,
        )["event_id"]
        .agg(set)
        .rename(
            "baseline_event_ids"
        )
        .reset_index()
    )

    gated = gated.merge(
        baseline_members,
        on=key,
        how="left",
        validate="many_to_one",
    )

    gated["is_replacement"] = gated.apply(
        lambda row: (
            row["event_id"]
            not in row["baseline_event_ids"]
        ),
        axis=1,
    )

    return gated


def summarize_replacements(
    displaced: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        test_year,
        horizon,
        daily_rank,
        replacement,
    ), group in displaced.groupby(
        [
            "test_year",
            "horizon_days",
            "daily_rank",
            "is_replacement",
        ],
        observed=True,
    ):
        returns = pd.to_numeric(
            group[
                "realized_contrarian_return"
            ],
            errors="coerce",
        ).dropna()

        if returns.empty:
            continue

        gains = returns[
            returns > 0
        ].sum()

        losses = returns[
            returns < 0
        ].sum()

        profit_factor = (
            gains / abs(losses)
            if losses != 0
            else float("inf")
        )

        rows.append(
            {
                "test_year": int(
                    test_year
                ),
                "horizon_days": int(
                    horizon
                ),
                "daily_rank": int(
                    daily_rank
                ),
                "is_replacement": bool(
                    replacement
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
                "worst_return": (
                    returns.min()
                ),
                "loss_le_100_rate": (
                    returns <= -100
                ).mean(),
                "loss_le_200_rate": (
                    returns <= -200
                ).mean(),
                "profit_factor": (
                    profit_factor
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_periods(
    displaced: pd.DataFrame,
) -> pd.DataFrame:
    df = displaced.copy()

    df["period"] = df[
        "test_year"
    ].apply(
        lambda year: (
            "DEVELOPMENT_2021_2024"
            if year in (
                2021,
                2022,
                2023,
                2024,
            )
            else (
                "VALIDATION_2025"
                if year == 2025
                else "OTHER"
            )
        )
    )

    df = df[
        df["period"] != "OTHER"
    ]

    rows = []

    for (
        period,
        horizon,
        daily_rank,
        replacement,
    ), group in df.groupby(
        [
            "period",
            "horizon_days",
            "daily_rank",
            "is_replacement",
        ],
        observed=True,
    ):
        returns = pd.to_numeric(
            group[
                "realized_contrarian_return"
            ],
            errors="coerce",
        ).dropna()

        if returns.empty:
            continue

        gains = returns[
            returns > 0
        ].sum()

        losses = returns[
            returns < 0
        ].sum()

        pf = (
            gains / abs(losses)
            if losses != 0
            else float("inf")
        )

        rows.append(
            {
                "period": period,
                "horizon_days": int(
                    horizon
                ),
                "daily_rank": int(
                    daily_rank
                ),
                "is_replacement": bool(
                    replacement
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
                "worst_return": (
                    returns.min()
                ),
                "loss_le_100_rate": (
                    returns <= -100
                ).mean(),
                "loss_le_200_rate": (
                    returns <= -200
                ).mean(),
                "profit_factor": pf,
            }
        )

    return pd.DataFrame(
        rows
    )


def worst_replacements(
    displaced: pd.DataFrame,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    replacements = displaced[
        displaced["is_replacement"]
    ].copy()

    return (
        replacements.sort_values(
            "realized_contrarian_return",
            ascending=True,
        )
        .head(limit)
        .reset_index(drop=True)
    )


def write_reports(
    displaced: pd.DataFrame,
    yearly: pd.DataFrame,
    periods: pd.DataFrame,
    worst: pd.DataFrame,
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

    displaced.to_csv(
        output_dir
        / "scanner_policy_displacement_rows.csv",
        index=False,
    )

    yearly.to_csv(
        output_dir
        / "scanner_policy_displacement_by_year.csv",
        index=False,
    )

    periods.to_csv(
        output_dir
        / "scanner_policy_displacement_summary.csv",
        index=False,
    )

    worst.to_csv(
        output_dir
        / "scanner_policy_worst_replacements.csv",
        index=False,
    )
    