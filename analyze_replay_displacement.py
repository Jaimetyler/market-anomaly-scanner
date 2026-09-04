from __future__ import annotations

import pandas as pd

from scanner.replay_displacement import (
    compare_baseline_to_policy,
    load_policy_samples,
    summarize_periods,
    summarize_replacements,
    worst_replacements,
    write_reports,
)


def pct(
    value: float,
) -> str:
    if pd.isna(value):
        return "NA"

    return f"{value:.1%}"


def ret(
    value: float,
) -> str:
    if pd.isna(value):
        return "NA"

    return f"{value:+.2f}%"


def ratio(
    value: float,
) -> str:
    if pd.isna(value):
        return "NA"

    if value == float("inf"):
        return "INF"

    return f"{value:.2f}"


def print_period(
    summary: pd.DataFrame,
    period: str,
) -> None:
    rows = summary[
        summary["period"]
        == period
    ].copy()

    print()
    print("=" * 160)
    print(
        f"{period} — $5 GATE DISPLACEMENT"
    )
    print("=" * 160)

    for horizon in sorted(
        rows[
            "horizon_days"
        ].unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 160)

        horizon_rows = rows[
            rows[
                "horizon_days"
            ]
            == horizon
        ].sort_values(
            [
                "daily_rank",
                "is_replacement",
            ]
        )

        for _, row in horizon_rows.iterrows():
            label = (
                "REPLACEMENT"
                if row[
                    "is_replacement"
                ]
                else "ORIGINAL"
            )

            print(
                f"RANK {int(row['daily_rank'])} | "
                f"{label:<11} | "
                f"N={int(row['observations']):>5,} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"WORST={ret(row['worst_return']):>12} | "
                f"<=-100={pct(row['loss_le_100_rate']):>6} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def print_worst(
    worst: pd.DataFrame,
) -> None:
    print()
    print("=" * 160)
    print(
        "WORST $5+ REPLACEMENT EVENTS"
    )
    print("=" * 160)

    for _, row in worst.head(
        30
    ).iterrows():
        print(
            f"{str(row['ticker']):<8} | "
            f"{str(row['date'])[:10]} | "
            f"{int(row['horizon_days']):>2}D | "
            f"RANK={int(row['daily_rank'])} | "
            f"PRICE=${row['event_close']:.2f} | "
            f"RETURN={ret(row['realized_contrarian_return']):>12} | "
            f"SCORE={row['ranking_score']:.2f}"
        )


def main() -> None:
    print("=" * 160)
    print(
        "SCANNER POLICY DISPLACEMENT ANALYSIS"
    )
    print("=" * 160)

    samples = load_policy_samples()

    displaced = (
        compare_baseline_to_policy(
            samples
        )
    )

    replacement_count = int(
        displaced[
            "is_replacement"
        ].sum()
    )

    print(
        f"\n$5+ replacement rows: "
        f"{replacement_count:,}"
    )

    yearly = (
        summarize_replacements(
            displaced
        )
    )

    periods = summarize_periods(
        displaced
    )

    worst = worst_replacements(
        displaced
    )

    write_reports(
        displaced,
        yearly,
        periods,
        worst,
    )

    print_period(
        periods,
        "DEVELOPMENT_2021_2024",
    )

    print_period(
        periods,
        "VALIDATION_2025",
    )

    print_worst(
        worst
    )

    print()
    print("=" * 160)
    print("REPORTS WRITTEN")
    print("=" * 160)

    print(
        "data/research/replay_reports/"
        "scanner_policy_displacement_rows.csv"
    )

    print(
        "data/research/replay_reports/"
        "scanner_policy_displacement_by_year.csv"
    )

    print(
        "data/research/replay_reports/"
        "scanner_policy_displacement_summary.csv"
    )

    print(
        "data/research/replay_reports/"
        "scanner_policy_worst_replacements.csv"
    )


if __name__ == "__main__":
    main()