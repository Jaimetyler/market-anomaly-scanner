from __future__ import annotations

import pandas as pd

from scanner.replay_policy import (
    build_policy_samples,
    load_inputs,
    summarize_periods,
    summarize_yearly,
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


def print_top5(
    periods: pd.DataFrame,
    period: str,
) -> None:
    rows = periods[
        (
            periods["period"]
            == period
        )
        & (
            periods["top_n"]
            == 5
        )
    ].copy()

    print()
    print("=" * 165)
    print(
        f"{period} — RERANKED TOP 5"
    )
    print("=" * 165)

    for horizon in sorted(
        rows["horizon_days"]
        .unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 165)

        horizon_rows = rows[
            rows["horizon_days"]
            == horizon
        ]

        for _, row in horizon_rows.iterrows():
            print(
                f"{row['policy']:<14} | "
                f"N={int(row['observations']):>6,} | "
                f"DAYS={int(row['trading_days']):>4} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"CVaR5={ret(row['cvar_05']):>11} | "
                f"<=-100={pct(row['loss_le_100_rate']):>6} | "
                f"<=-200={pct(row['loss_le_200_rate']):>6} | "
                f"PF={ratio(row['profit_factor']):>5} | "
                f"MED PRICE=${row['median_entry_price']:.2f}"
            )


def print_topn_20d(
    periods: pd.DataFrame,
    period: str,
) -> None:
    rows = periods[
        (
            periods["period"]
            == period
        )
        & (
            periods["horizon_days"]
            == 20
        )
    ].copy()

    print()
    print("=" * 165)
    print(
        f"{period} — 20D TOP-N COMPARISON"
    )
    print("=" * 165)

    for top_n in sorted(
        rows["top_n"]
        .unique()
    ):
        print()
        print(
            f"TOP {int(top_n)}"
        )
        print("-" * 165)

        top_rows = rows[
            rows["top_n"]
            == top_n
        ]

        for _, row in top_rows.iterrows():
            print(
                f"{row['policy']:<14} | "
                f"N={int(row['observations']):>6,} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"CVaR5={ret(row['cvar_05']):>11} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def main() -> None:
    print("=" * 165)
    print(
        "LIVE-SCANNER POLICY REPLAY"
    )
    print("=" * 165)

    anchors = load_inputs()

    print(
        "\nApplying eligibility BEFORE ranking..."
    )

    samples = build_policy_samples(
        anchors
    )

    print(
        f"Policy sample rows: "
        f"{len(samples):,}"
    )

    print(
        "Calculating year-by-year performance..."
    )

    yearly = summarize_yearly(
        samples
    )

    print(
        "Calculating development/validation performance..."
    )

    periods = summarize_periods(
        samples
    )

    write_reports(
        samples,
        yearly,
        periods,
    )

    print_top5(
        periods,
        "DEVELOPMENT_2021_2024",
    )

    print_top5(
        periods,
        "VALIDATION_2025",
    )

    print_topn_20d(
        periods,
        "DEVELOPMENT_2021_2024",
    )

    print_topn_20d(
        periods,
        "VALIDATION_2025",
    )

    print()
    print("=" * 165)
    print("REPORTS WRITTEN")
    print("=" * 165)

    print(
        "data/research/replay_reports/"
        "scanner_policy_samples.csv"
    )

    print(
        "data/research/replay_reports/"
        "scanner_policy_by_year.csv"
    )

    print(
        "data/research/replay_reports/"
        "scanner_policy_summary.csv"
    )

    print()
    print(
        "NOTE: 2025 is validation only. "
        "It has already been inspected and is not a pristine holdout."
    )


if __name__ == "__main__":
    main()