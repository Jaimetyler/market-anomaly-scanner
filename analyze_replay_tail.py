from __future__ import annotations

import pandas as pd

from scanner.replay_tail import (
    FOCUS_TOP_N,
    add_tail_flags,
    focus_samples,
    load_samples,
    repeated_tail_events,
    summarize_direction,
    summarize_segmentation_tail,
    summarize_setup_tail,
    summarize_year_tail,
    worst_events,
    write_tail_reports,
)


def pct(
    value: float,
) -> str:
    if pd.isna(value):
        return "NA"

    return f"{value:.2%}"


def ret(
    value: float,
) -> str:
    if pd.isna(value):
        return "NA"

    return f"{value:+.2f}%"


def print_direction_summary(
    summary: pd.DataFrame,
) -> None:
    print()
    print("=" * 145)
    print(
        f"TOP {FOCUS_TOP_N} — TAIL RISK BY EVENT DIRECTION"
    )
    print("=" * 145)

    for horizon in sorted(
        summary["horizon_days"]
        .unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 145)

        rows = summary[
            summary["horizon_days"]
            == horizon
        ]

        for _, row in rows.iterrows():
            print(
                f"{row['direction_label']:<20} | "
                f"N={int(row['observations']):>6,} | "
                f"WR={pct(row['win_rate']):>7} | "
                f"MED={ret(row['median_return']):>10} | "
                f"MEAN={ret(row['mean_return']):>10} | "
                f"P05={ret(row['p05_return']):>10} | "
                f"WORST={ret(row['worst_return']):>12} | "
                f"<=-50={pct(row['loss_le_50_rate']):>7} | "
                f"<=-100={pct(row['loss_le_100_rate']):>7} | "
                f"<=-200={pct(row['loss_le_200_rate']):>7}"
            )


def print_worst_events(
    worst: pd.DataFrame,
) -> None:
    print()
    print("=" * 145)
    print("WORST 10 EVENTS PER HORIZON")
    print("=" * 145)

    for horizon in sorted(
        worst["horizon_days"]
        .unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 145)

        rows = (
            worst[
                worst["horizon_days"]
                == horizon
            ]
            .head(10)
        )

        for _, row in rows.iterrows():
            print(
                f"{str(row['ticker']):<8} | "
                f"{str(row['date'])[:10]} | "
                f"{row['direction_label']:<20} | "
                f"{str(row['initial_setup']):<24} | "
                f"{str(row['segmentation']):<28} | "
                f"RETURN={ret(row['realized_contrarian_return']):>12} | "
                f"TRAIN MED={ret(row['train_median_return']):>10} | "
                f"TRAIN N={int(row['train_n']):>6,}"
            )


def print_repeated_tail(
    repeated: pd.DataFrame,
) -> None:
    print()
    print("=" * 145)
    print(
        "EVENTS WITH <= -50% LOSSES ACROSS MULTIPLE HORIZONS"
    )
    print("=" * 145)

    if repeated.empty:
        print(
            "No repeated tail events found."
        )
        return

    for _, row in repeated.head(
        30
    ).iterrows():
        print(
            f"{str(row['ticker']):<8} | "
            f"{str(row['date'])[:10]} | "
            f"{row['direction_label']:<20} | "
            f"{str(row['initial_setup']):<24} | "
            f"HORIZONS={int(row['horizons_hit'])} | "
            f"WORST={ret(row['worst_return']):>12}"
        )


def main() -> None:
    print("=" * 145)
    print(
        "RAW_MEDIAN TAIL ATTRIBUTION"
    )
    print("=" * 145)

    samples = load_samples()

    focused = focus_samples(
        samples
    )

    print(
        f"\nTop-{FOCUS_TOP_N} focused rows: "
        f"{len(focused):,}"
    )

    focused = add_tail_flags(
        focused
    )

    print(
        "Calculating direction attribution..."
    )
    direction = summarize_direction(
        focused
    )

    print(
        "Calculating setup attribution..."
    )
    setup = summarize_setup_tail(
        focused
    )

    print(
        "Calculating segmentation attribution..."
    )
    segmentation = (
        summarize_segmentation_tail(
            focused
        )
    )

    print(
        "Calculating year attribution..."
    )
    yearly = summarize_year_tail(
        focused
    )

    print(
        "Extracting worst events..."
    )
    worst = worst_events(
        focused
    )

    print(
        "Finding repeated tail events..."
    )
    repeated = repeated_tail_events(
        focused
    )

    write_tail_reports(
        direction_summary=direction,
        setup_summary=setup,
        segmentation_summary=segmentation,
        year_summary=yearly,
        worst=worst,
        repeated=repeated,
    )

    print_direction_summary(
        direction
    )

    print_worst_events(
        worst
    )

    print_repeated_tail(
        repeated
    )

    print()
    print("=" * 145)
    print("REPORTS WRITTEN")
    print("=" * 145)

    print(
        "data/research/replay_reports/"
        "raw_median_tail_by_direction.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_tail_by_setup.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_tail_by_segmentation.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_tail_by_year.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_worst_events.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_repeated_tail_events.csv"
    )


if __name__ == "__main__":
    main()