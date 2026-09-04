from __future__ import annotations

import pandas as pd

from scanner.replay_risk_gate import (
    build_bucket_summary,
    build_gate_summary,
    load_top5_samples,
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
    subset = summary[
        summary["period"] == period
    ].copy()

    print()
    print("=" * 170)
    print(period)
    print("=" * 170)

    for horizon in sorted(
        subset[
            "horizon_days"
        ].unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 170)

        rows = subset[
            subset[
                "horizon_days"
            ]
            == horizon
        ]

        for _, row in rows.iterrows():
            print(
                f"{row['gate']:<38} | "
                f"N={int(row['observations']):>6,} | "
                f"KEEP={pct(row['coverage']):>6} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"CVaR5={ret(row['cvar_05']):>11} | "
                f"<=-100={pct(row['loss_le_100_rate']):>6} | "
                f"<=-200={pct(row['loss_le_200_rate']):>6} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def print_bucket_extremes(
    buckets: pd.DataFrame,
) -> None:
    print()
    print("=" * 170)
    print(
        "20D STRUCTURAL RISK — WORST BUCKETS BY PROFIT FACTOR"
    )
    print("=" * 170)

    subset = buckets[
        buckets[
            "horizon_days"
        ]
        == 20
    ].copy()

    for feature in (
        "price_bucket",
        "rvol_bucket",
        "move_5d_bucket",
        "initial_setup",
    ):
        print()
        print(feature.upper())
        print("-" * 170)

        rows = (
            subset[
                subset["feature"]
                == feature
            ]
            .sort_values(
                [
                    "profit_factor",
                    "loss_le_200_rate",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

        for _, row in rows.iterrows():
            print(
                f"{str(row['bucket']):<28} | "
                f"N={int(row['observations']):>6,} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"CVaR5={ret(row['cvar_05']):>11} | "
                f"<=-100={pct(row['loss_le_100_rate']):>6} | "
                f"<=-200={pct(row['loss_le_200_rate']):>6} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def main() -> None:
    print("=" * 170)
    print(
        "RAW_MEDIAN STRUCTURAL RISK-GATE ANALYSIS"
    )
    print("=" * 170)

    samples = load_top5_samples()

    print(
        "\nTesting pre-trade structural gates..."
    )

    gates = build_gate_summary(
        samples
    )

    print(
        "Calculating structural bucket risk..."
    )

    buckets = build_bucket_summary(
        samples
    )

    write_reports(
        gates,
        buckets,
    )

    print_period(
        gates,
        "DEVELOPMENT_2021_2024",
    )

    print_period(
        gates,
        "VALIDATION_2025",
    )

    print_bucket_extremes(
        buckets
    )

    print()
    print("=" * 170)
    print("REPORTS WRITTEN")
    print("=" * 170)

    print(
        "data/research/replay_reports/"
        "raw_median_risk_gate_summary.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_risk_bucket_summary.csv"
    )

    print()
    print(
        "IMPORTANT: 2025 has already been inspected in prior "
        "analysis, so it is now treated as validation rather "
        "than a pristine untouched holdout."
    )


if __name__ == "__main__":
    main()