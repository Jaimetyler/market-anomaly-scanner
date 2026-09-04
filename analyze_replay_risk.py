from __future__ import annotations

import pandas as pd

from scanner.replay_risk import (
    HOLDOUT_YEAR,
    RANKING_METHOD,
    TOP_LEVELS,
    add_daily_ranks,
    attach_excursions,
    build_top_n_samples,
    load_excursion_outcomes,
    load_raw_median_anchors,
    summarize_by_year,
    summarize_period,
    write_risk_reports,
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

    if subset.empty:
        return

    print()
    print("=" * 150)
    print(period)
    print("=" * 150)

    for horizon in sorted(
        subset["horizon_days"].unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 150)

        horizon_rows = subset[
            subset["horizon_days"]
            == horizon
        ].sort_values(
            "top_n"
        )

        for _, row in horizon_rows.iterrows():
            print(
                f"TOP {int(row['top_n']):>2} | "
                f"N={int(row['observations']):>6,} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>9} | "
                f"P10={ret(row['p10_return']):>9} | "
                f"P05={ret(row['p05_return']):>9} | "
                f"CVaR5={ret(row['cvar_05']):>9} | "
                f"AVG LOSS={ret(row['average_loss']):>9} | "
                f"PAYOFF={ratio(row['payoff_ratio']):>5} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def print_excursion_coverage(
    summary: pd.DataFrame,
) -> None:
    print()
    print("=" * 150)
    print("MFE / MAE COVERAGE CHECK")
    print("=" * 150)

    rows = summary[
        (
            summary["period"]
            == "HOLDOUT_2025"
        )
        & (
            summary["top_n"]
            == 5
        )
    ].sort_values(
        "horizon_days"
    )

    for _, row in rows.iterrows():
        print(
            f"{int(row['horizon_days']):>2}D | "
            f"MFE COVERAGE={pct(row['mfe_coverage']):>6} | "
            f"MAE COVERAGE={pct(row['mae_coverage']):>6} | "
            f"MED MFE={ret(row['median_mfe']):>9} | "
            f"P90 MFE={ret(row['p90_mfe']):>9} | "
            f"MED MAE={ret(row['median_mae']):>9} | "
            f"P10 MAE={ret(row['p10_mae']):>9}"
        )


def print_year_consistency(
    yearly: pd.DataFrame,
) -> None:
    print()
    print("=" * 150)
    print(
        f"{RANKING_METHOD} — TOP 5 YEAR-BY-YEAR"
    )
    print("=" * 150)

    subset = yearly[
        yearly["top_n"] == 5
    ].copy()

    for horizon in sorted(
        subset["horizon_days"].unique()
    ):
        print()
        print(
            f"{int(horizon)}D HORIZON"
        )
        print("-" * 150)

        rows = subset[
            subset["horizon_days"]
            == horizon
        ].sort_values(
            "test_year"
        )

        for _, row in rows.iterrows():
            print(
                f"{int(row['test_year'])} | "
                f"N={int(row['observations']):>5,} | "
                f"WR={pct(row['win_rate']):>6} | "
                f"MED={ret(row['median_return']):>9} | "
                f"MEAN={ret(row['mean_return']):>9} | "
                f"P05={ret(row['p05_return']):>9} | "
                f"CVaR5={ret(row['cvar_05']):>9} | "
                f"PF={ratio(row['profit_factor']):>5}"
            )


def main() -> None:
    print("=" * 150)
    print(
        "RAW_MEDIAN SCANNER RISK ANALYSIS"
    )
    print("=" * 150)
    print()

    anchors = (
        load_raw_median_anchors()
    )

    print(
        "Ranking anchors within each "
        "historical day..."
    )

    ranked = add_daily_ranks(
        anchors
    )

    excursions = (
        load_excursion_outcomes()
    )

    print(
        f"Excursion rows: "
        f"{len(excursions):,}"
    )

    ranked = attach_excursions(
        ranked,
        excursions,
    )

    print(
        "Building top "
        f"{'/'.join(str(x) for x in TOP_LEVELS)} "
        "samples..."
    )

    samples = build_top_n_samples(
        ranked
    )

    print(
        f"Top-N sample rows: "
        f"{len(samples):,}"
    )

    print(
        "Calculating year-by-year "
        "risk distributions..."
    )

    yearly = summarize_by_year(
        samples
    )

    print(
        "Calculating development/holdout "
        "risk distributions..."
    )

    period = summarize_period(
        samples
    )

    write_risk_reports(
        top_samples=samples,
        yearly_summary=yearly,
        period_summary=period,
    )

    print_period(
        period,
        "DEVELOPMENT_2021_2024",
    )

    print_period(
        period,
        "HOLDOUT_2025",
    )

    print_year_consistency(
        yearly
    )

    print_excursion_coverage(
        period
    )

    print()
    print("=" * 150)
    print("REPORTS WRITTEN")
    print("=" * 150)

    print(
        "data/research/replay_reports/"
        "raw_median_top_n_samples.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_risk_by_year.csv"
    )

    print(
        "data/research/replay_reports/"
        "raw_median_risk_summary.csv"
    )

    print()
    print(
        "NOTE: Profit factor/payoff statistics above "
        "describe the historical event return distribution. "
        "They are not a portfolio backtest because events can "
        "overlap and no execution, sizing, borrowing, slippage, "
        "or capital constraints are modeled."
    )


if __name__ == "__main__":
    main()