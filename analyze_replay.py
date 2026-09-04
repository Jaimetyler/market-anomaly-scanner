from __future__ import annotations

import pandas as pd

from scanner.replay_analysis import (
    ANALYSIS_YEARS,
    HOLDOUT_YEAR,
    add_daily_ranks,
    add_ranking_scores,
    build_method_anchors,
    load_replay,
    method_scorecard,
    summarize_daily_top_n,
    summarize_method_performance,
    summarize_score_deciles,
    write_reports,
)


def pct(value: float) -> str:
    return f"{value:.1%}"


def ret(value: float) -> str:
    return f"{value:+.2f}%"


def print_development_scorecard(
    scorecard: pd.DataFrame,
) -> None:
    print()
    print("=" * 120)
    print(
        "DEVELOPMENT SCORECARD — TOP 5 PER DAY "
        f"({min(ANALYSIS_YEARS)}–{max(ANALYSIS_YEARS)})"
    )
    print("=" * 120)

    for horizon in sorted(
        scorecard["horizon_days"].unique()
    ):
        print()
        print(f"{int(horizon)}D HORIZON")
        print("-" * 120)

        subset = scorecard[
            scorecard["horizon_days"]
            == horizon
        ]

        for _, row in subset.iterrows():
            print(
                f"{row['ranking_method']:<12} | "
                f"N={int(row['observations']):>7,} | "
                f"AVG WR={pct(row['avg_yearly_win_rate']):>6} | "
                f"WORST WR={pct(row['worst_year_win_rate']):>6} | "
                f"AVG MED={ret(row['avg_yearly_median_return']):>9} | "
                f"WORST MED={ret(row['worst_year_median_return']):>9} | "
                f"AVG MEAN={ret(row['avg_yearly_mean_return']):>9}"
            )


def print_holdout(
    top_n: pd.DataFrame,
) -> None:
    print()
    print("=" * 120)
    print(
        f"{HOLDOUT_YEAR} HOLDOUT — TOP 5 PER DAY"
    )
    print("=" * 120)

    holdout = top_n[
        (top_n["test_year"] == HOLDOUT_YEAR)
        & (top_n["top_n"] == 5)
    ].copy()

    holdout = holdout.sort_values(
        [
            "horizon_days",
            "realized_median_return",
        ],
        ascending=[
            True,
            False,
        ],
    )

    for horizon in sorted(
        holdout["horizon_days"].unique()
    ):
        print()
        print(f"{int(horizon)}D HORIZON")
        print("-" * 120)

        subset = holdout[
            holdout["horizon_days"]
            == horizon
        ]

        for _, row in subset.iterrows():
            print(
                f"{row['ranking_method']:<12} | "
                f"N={int(row['observations']):>7,} | "
                f"WR={pct(row['realized_win_rate']):>6} | "
                f"MED={ret(row['realized_median_return']):>9} | "
                f"MEAN={ret(row['realized_mean_return']):>9}"
            )


def main() -> None:
    print("=" * 120)
    print("SCANNER RANKING CALIBRATION")
    print("=" * 120)
    print()

    replay = load_replay()

    print()
    print("Computing training-only ranking scores...")
    scored = add_ranking_scores(replay)

    print()
    anchors = build_method_anchors(
        scored
    )

    print(
        f"\nAnchor rows: "
        f"{len(anchors):,}"
    )

    overall = summarize_method_performance(
        anchors
    )

    print("Ranking anchors within each historical day...")
    ranked = add_daily_ranks(
        anchors
    )

    print("Calculating top-N scanner performance...")
    top_n = summarize_daily_top_n(
        ranked
    )

    print("Calculating score deciles...")
    deciles = summarize_score_deciles(
        anchors
    )

    scorecard = method_scorecard(
        top_n
    )

    write_reports(
        anchors=anchors,
        overall=overall,
        top_n=top_n,
        deciles=deciles,
        scorecard=scorecard,
    )

    print_development_scorecard(
        scorecard
    )

    # IMPORTANT:
    # We print 2025 only after the development-period scorecard
    # has already been constructed independently of it.
    print_holdout(
        top_n
    )

    print()
    print("=" * 120)
    print("REPORTS WRITTEN")
    print("=" * 120)
    print(
        "data/research/replay_reports/"
        "replay_method_anchors.csv"
    )
    print(
        "data/research/replay_reports/"
        "replay_method_performance.csv"
    )
    print(
        "data/research/replay_reports/"
        "replay_daily_top_n.csv"
    )
    print(
        "data/research/replay_reports/"
        "replay_score_deciles.csv"
    )
    print(
        "data/research/replay_reports/"
        "replay_development_scorecard.csv"
    )


if __name__ == "__main__":
    main()