from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from scanner.corporate_actions import (
    get_stock_dividends,
    get_stock_splits,
)
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits
from scanner.flatfile_history import read_many_ticker_histories
from scanner.turtle import SYSTEM_1, SYSTEM_2, TurtleSystem
from scanner.turtle_history import split_continuous_history
from scanner.turtle_portfolio import (
    TurtlePortfolioConfig,
    run_turtle_portfolio,
)
from scanner.turtle_simulation import (
    TurtleTrade,
    simulate_turtle_system,
)


DEFAULT_MASTER = Path(
    "data/research/master_anomalies_2019_2025.csv"
)
DEFAULT_REPORT_DIR = Path(
    "data/research/turtle_reports"
)

DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Turtle Trading sanity backtests against the "
            "project's local Massive daily flat-file history."
        )
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START,
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help=(
            "Maximum exact-case tickers to test. "
            "Use 0 for all tickers."
        ),
    )

    parser.add_argument(
        "--system",
        choices=("1", "2", "both"),
        default="both",
    )

    parser.add_argument(
        "--starting-equity",
        type=float,
        default=5000.0,
    )

    parser.add_argument(
        "--risk",
        type=float,
        default=0.01,
        help=(
            "Fraction of current equity allocated to "
            "one 1N unit. Default = 0.01."
        ),
    )

    return parser.parse_args()


def _first_present(
    row: dict[str, str],
    names: Sequence[str],
) -> str | None:
    for name in names:
        value = row.get(name)

        if value is not None:
            value = value.strip()

            if value:
                return value

    return None


def _master_tickers(
    path: Path,
) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Master event file not found: {path}"
        )

    tickers: set[str] = set()

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"No header found in {path}"
            )

        for row in reader:
            ticker = _first_present(
                row,
                (
                    "ticker",
                    "symbol",
                    "security_ticker",
                ),
            )

            if ticker:
                # IMPORTANT:
                # Provider identity is case-sensitive.
                # Do not .upper() here.
                tickers.add(ticker)

    if not tickers:
        raise RuntimeError(
            "No ticker column could be resolved from "
            f"{path}. Fields: {reader.fieldnames}"
        )

    return sorted(tickers)


def _bar_date(
    bar: Any,
) -> str | None:
    if isinstance(bar, dict):
        for key in (
            "session_date",
            "date",
            "signal_date",
        ):
            value = bar.get(key)

            if value is not None:
                if hasattr(value, "isoformat"):
                    return value.isoformat()

                return str(value)

        ns = bar.get("window_start_ns")

    else:
        for key in (
            "session_date",
            "date",
            "signal_date",
        ):
            if hasattr(bar, key):
                value = getattr(bar, key)

                if value is not None:
                    if hasattr(value, "isoformat"):
                        return value.isoformat()

                    return str(value)

        ns = getattr(
            bar,
            "window_start_ns",
            None,
        )

    if ns is not None:
        return datetime.fromtimestamp(
            int(ns) / 1_000_000_000,
            tz=timezone.utc,
        ).date().isoformat()

    return None


def _safe_bar_date(
    bars: Sequence[Any],
    index: int,
) -> str:
    if 0 <= index < len(bars):
        value = _bar_date(
            bars[index]
        )

        if value:
            return value

    return ""


def _systems_from_arg(
    value: str,
) -> tuple[TurtleSystem, ...]:
    if value == "1":
        return (SYSTEM_1,)

    if value == "2":
        return (SYSTEM_2,)

    return (
        SYSTEM_1,
        SYSTEM_2,
    )


def _median(
    values: Iterable[float],
) -> float:
    clean = [
        float(value)
        for value in values
        if math.isfinite(
            float(value)
        )
    ]

    if not clean:
        return float("nan")

    return statistics.median(
        clean
    )


def _mean(
    values: Iterable[float],
) -> float:
    clean = [
        float(value)
        for value in values
        if math.isfinite(
            float(value)
        )
    ]

    if not clean:
        return float("nan")

    return statistics.fmean(
        clean
    )


def _fmt(
    value: float,
    digits: int = 2,
) -> str:
    if not math.isfinite(value):
        return ""

    return f"{value:.{digits}f}"


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fieldnames
            ),
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    temp.replace(path)


def _trade_row(
    *,
    ticker: str,
    trade: TurtleTrade,
    bars: Sequence[Any],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "system": trade.system,
        "side": trade.side,
        "entry_date": (
            trade.entry_date
            or _safe_bar_date(
                bars,
                trade.entry_index,
            )
        ),
        "exit_date": (
            trade.exit_date
            or _safe_bar_date(
                bars,
                trade.exit_index,
            )
        ),
        "entry_index": (
            trade.entry_index
        ),
        "exit_index": (
            trade.exit_index
        ),
        "initial_entry_price": (
            trade.initial_entry_price
        ),
        "average_entry_price": (
            trade.average_entry_price
        ),
        "exit_price": (
            trade.exit_price
        ),
        "units": trade.units,
        "initial_n": (
            trade.initial_n
        ),
        "initial_stop_price": (
            trade.initial_stop_price
        ),
        "final_stop_price": (
            trade.final_stop_price
        ),
        "exit_reason": (
            trade.exit_reason
        ),
        "return_pct": (
            trade.return_pct
        ),
        "unit_weighted_return_pct": (
            trade.unit_weighted_return_pct
        ),
        "bars_held": (
            trade.bars_held
        ),
        "ambiguous_bars": (
            trade.ambiguous_bars
        ),
    }


def _year_from_date(
    value: str,
) -> str:
    if len(value) >= 4:
        prefix = value[:4]

        if prefix.isdigit():
            return prefix

    return "UNKNOWN"


def _build_yearly_rows(
    trade_rows: Sequence[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in trade_rows:
        year = _year_from_date(
            str(
                row.get(
                    "exit_date",
                    "",
                )
            )
        )

        groups[
            (
                str(row["system"]),
                year,
            )
        ].append(row)

    output: list[
        dict[str, Any]
    ] = []

    for (
        system,
        year,
    ), rows in sorted(
        groups.items()
    ):
        returns = [
            float(
                row[
                    "unit_weighted_return_pct"
                ]
            )
            for row in rows
        ]

        winners = sum(
            value > 0
            for value in returns
        )

        output.append(
            {
                "system": system,
                "year": year,
                "trades": len(rows),
                "win_rate_pct": (
                    winners
                    / len(rows)
                    * 100.0
                ),
                "median_trade_return_pct": (
                    _median(
                        returns
                    )
                ),
                "mean_trade_return_pct": (
                    _mean(
                        returns
                    )
                ),
            }
        )

    return output


def _build_system_rows(
    *,
    trades_by_system: dict[
        str,
        list[TurtleTrade],
    ],
    starting_equity: float,
    risk_fraction: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    summary_rows: list[
        dict[str, Any]
    ] = []

    equity_rows: list[
        dict[str, Any]
    ] = []

    for system_name in sorted(
        trades_by_system
    ):
        trades = trades_by_system[
            system_name
        ]

        if not trades:
            summary_rows.append(
                {
                    "system": (
                        system_name
                    ),
                    "trades": 0,
                    "win_rate_pct": "",
                    "median_trade_return_pct": "",
                    "mean_trade_return_pct": "",
                    "starting_equity": (
                        starting_equity
                    ),
                    "ending_equity_proxy": (
                        starting_equity
                    ),
                    "return_pct_proxy": 0.0,
                    "max_drawdown_pct_proxy": 0.0,
                    "portfolio_model": (
                        "SEQUENTIAL_PROXY"
                    ),
                }
            )

            continue

        returns = [
            trade.unit_weighted_return_pct
            for trade in trades
        ]

        winners = sum(
            value > 0
            for value in returns
        )

        portfolio = (
            run_turtle_portfolio(
                trades,
                config=(
                    TurtlePortfolioConfig(
                        starting_equity=(
                            starting_equity
                        ),
                        risk_fraction_per_unit=(
                            risk_fraction
                        ),
                    )
                ),
            )
        )

        summary_rows.append(
            {
                "system": system_name,
                "trades": len(trades),
                "win_rate_pct": (
                    winners
                    / len(trades)
                    * 100.0
                ),
                "median_trade_return_pct": (
                    _median(
                        returns
                    )
                ),
                "mean_trade_return_pct": (
                    _mean(
                        returns
                    )
                ),
                "starting_equity": (
                    starting_equity
                ),
                "ending_equity_proxy": (
                    portfolio.ending_equity
                ),
                "return_pct_proxy": (
                    portfolio.total_return_pct
                ),
                "max_drawdown_pct_proxy": (
                    portfolio.max_drawdown_pct
                ),
                "portfolio_model": (
                    "SEQUENTIAL_PROXY"
                ),
            }
        )

        for point in (
            portfolio.equity_curve
        ):
            equity_rows.append(
                {
                    "system": (
                        system_name
                    ),
                    "trade_number": (
                        point.trade_number
                    ),
                    "equity": (
                        point.equity
                    ),
                    "peak_equity": (
                        point.peak_equity
                    ),
                    "drawdown_pct": (
                        point.drawdown_pct
                    ),
                    "portfolio_model": (
                        "SEQUENTIAL_PROXY"
                    ),
                }
            )

    return (
        summary_rows,
        equity_rows,
    )


def main() -> None:
    args = _parse_args()

    if args.limit < 0:
        raise ValueError(
            "--limit cannot be negative"
        )

    if args.starting_equity <= 0:
        raise ValueError(
            "--starting-equity must be positive"
        )

    if not (
        0.0
        < args.risk
        < 1.0
    ):
        raise ValueError(
            "--risk must be between 0 and 1"
        )

    systems = _systems_from_arg(
        args.system
    )

    all_tickers = _master_tickers(
        args.master
    )

    if args.limit:
        tickers = all_tickers[
            : args.limit
        ]
    else:
        tickers = all_tickers

    print()
    print("=" * 78)
    print("TURTLE REAL-DATA SANITY BACKTEST")
    print("=" * 78)
    print(
        f"Master:             {args.master}"
    )
    print(
        f"Exact-case universe:{len(all_tickers):>9,}"
    )
    print(
        f"Tickers selected:   {len(tickers):>9,}"
    )
    print(
        f"History window:     {args.start} -> {args.end}"
    )
    print(
        "Systems:            "
        + ", ".join(
            system.name
            for system in systems
        )
    )
    print(
        f"Starting equity:    ${args.starting_equity:,.2f}"
    )
    print(
        f"Risk / 1N unit:     {args.risk * 100:.2f}%"
    )
    print()

    print(
        "Loading local Massive flat-file histories "
        "in one pass..."
    )

    histories = (
        read_many_ticker_histories(
            tickers,
            args.start,
            args.end,
            progress=True,
        )
    )

    print()
    print(
        "Applying production split adjustment "
        "and simulating Turtle trades..."
    )
    print()

    trade_rows: list[
        dict[str, Any]
    ] = []

    dated_trades_by_system: dict[
        str,
        list[
            tuple[
                str,
                str,
                TurtleTrade,
            ]
        ],
    ] = defaultdict(list)

    no_history = 0
    too_short = 0
    processed = 0
    failed = 0

    for ordinal, ticker in enumerate(
        tickers,
        start=1,
    ):
        raw_history = histories.get(
            ticker,
            [],
        )

        if not raw_history:
            no_history += 1
            continue

        try:
            split_events = (
                get_stock_splits(
                    ticker=ticker
                )
            )

            bars = (
                adjust_flatfile_history_for_splits(
                    raw_history,
                    split_events,
                )
            )

            if len(bars) < 80:
                too_short += 1
                continue

            # ----------------------------------------------------------
            # Stock-specific Turtle history adaptation.
            #
            # Split-adjusted bars can still contain large discontinuities
            # caused by distributions, spinoffs, ticker-lineage changes,
            # or other corporate actions that are not ordinary market P&L.
            #
            # Never carry:
            #   - an open Turtle position
            #   - N
            #   - breakout channels
            #   - System 1 winner/loser state
            #
            # across one of those quarantined boundaries.
            # ----------------------------------------------------------

            dividend_events = (
                get_stock_dividends(
                    ticker
                )
            )

            segments, history_breaks = (
                split_continuous_history(
                    bars,
                    ticker=ticker,
                    dividends=dividend_events,
                    minimum_segment_bars=80,
                )
            )

            if history_breaks:
                print(
                    f"[CA] {ticker}: "
                    f"{len(history_breaks)} "
                    f"discontinuity break(s), "
                    f"{len(segments)} usable segment(s)"
                )

                for item in history_breaks:
                    print(
                        f"     "
                        f"{item.previous_date or '?'} -> "
                        f"{item.new_date or '?'} "
                        f"gap={item.open_gap_pct:+.2f}% "
                        f"({item.reason})"
                    )

            if not segments:
                too_short += 1
                continue

            for segment_number, segment in enumerate(
                segments,
                start=1,
            ):
                segment_bars = list(
                    segment.bars
                )

                for system in systems:
                    trades = (
                        simulate_turtle_system(
                            segment_bars,
                            system=system,
                        )
                    )

                    for trade in trades:
                        row = _trade_row(
                            ticker=ticker,
                            trade=trade,
                            bars=segment_bars,
                        )

                        # Preserve enough information to audit exactly
                        # which continuous price segment produced a trade.
                        row["segment_number"] = (
                            segment_number
                        )

                        row["segment_start_index"] = (
                            segment.start_index
                        )

                        row["segment_end_index"] = (
                            segment.end_index
                        )

                        row["global_entry_index"] = (
                            segment.start_index
                            + trade.entry_index
                        )

                        row["global_exit_index"] = (
                            segment.start_index
                            + trade.exit_index
                        )

                        trade_rows.append(
                            row
                        )

                        entry_date = str(
                            row[
                                "entry_date"
                            ]
                        )

                        dated_trades_by_system[
                            system.name
                        ].append(
                            (
                                entry_date,
                                ticker,
                                trade,
                            )
                        )

            processed += 1

        except Exception as exc:
            failed += 1

            print(
                f"[WARN] {ticker}: "
                f"{type(exc).__name__}: {exc}"
            )

        if (
            ordinal % 10 == 0
            or ordinal
            == len(tickers)
        ):
            print(
                f"[{ordinal:>4}/{len(tickers):>4}] "
                f"processed={processed:,} "
                f"trades={len(trade_rows):,} "
                f"no_history={no_history:,} "
                f"short={too_short:,} "
                f"failed={failed:,}"
            )

    # Stable chronological order for reports.
    trade_rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "entry_date",
                    "",
                )
            ),
            str(
                row.get(
                    "ticker",
                    "",
                )
            ),
            str(
                row.get(
                    "system",
                    "",
                )
            ),
            int(
                row.get(
                    "entry_index",
                    0,
                )
            ),
        )
    )

    trades_by_system: dict[
        str,
        list[TurtleTrade],
    ] = {}

    for system in systems:
        ordered = sorted(
            dated_trades_by_system.get(
                system.name,
                [],
            ),
            key=lambda item: (
                item[0],
                item[1],
                item[2].entry_index,
            ),
        )

        trades_by_system[
            system.name
        ] = [
            item[2]
            for item in ordered
        ]

    yearly_rows = (
        _build_yearly_rows(
            trade_rows
        )
    )

    (
        system_rows,
        equity_rows,
    ) = _build_system_rows(
        trades_by_system=(
            trades_by_system
        ),
        starting_equity=(
            args.starting_equity
        ),
        risk_fraction=(
            args.risk
        ),
    )

    report_dir = args.report_dir

    trades_path = (
        report_dir
        / "trades.csv"
    )

    yearly_path = (
        report_dir
        / "yearly_performance.csv"
    )

    system_path = (
        report_dir
        / "system_comparison.csv"
    )

    equity_path = (
        report_dir
        / "equity_curve.csv"
    )

    trade_fields = (
        "ticker",
        "system",
        "side",
        "entry_date",
        "exit_date",
        "entry_index",
        "exit_index",
        "initial_entry_price",
        "average_entry_price",
        "exit_price",
        "units",
        "initial_n",
        "initial_stop_price",
        "final_stop_price",
        "exit_reason",
        "return_pct",
        "unit_weighted_return_pct",
        "bars_held",
        "ambiguous_bars",
        "segment_number",
        "segment_start_index",
        "segment_end_index",
        "global_entry_index",
        "global_exit_index",
    )

    _write_csv(
        trades_path,
        trade_rows,
        fieldnames=trade_fields,
    )

    _write_csv(
        yearly_path,
        yearly_rows,
        fieldnames=(
            "system",
            "year",
            "trades",
            "win_rate_pct",
            "median_trade_return_pct",
            "mean_trade_return_pct",
        ),
    )

    _write_csv(
        system_path,
        system_rows,
        fieldnames=(
            "system",
            "trades",
            "win_rate_pct",
            "median_trade_return_pct",
            "mean_trade_return_pct",
            "starting_equity",
            "ending_equity_proxy",
            "return_pct_proxy",
            "max_drawdown_pct_proxy",
            "portfolio_model",
        ),
    )

    _write_csv(
        equity_path,
        equity_rows,
        fieldnames=(
            "system",
            "trade_number",
            "equity",
            "peak_equity",
            "drawdown_pct",
            "portfolio_model",
        ),
    )

    print()
    print("=" * 78)
    print("TURTLE SANITY BACKTEST COMPLETE")
    print("=" * 78)
    print(
        f"Selected tickers:   {len(tickers):,}"
    )
    print(
        f"Processed tickers:  {processed:,}"
    )
    print(
        f"No history:         {no_history:,}"
    )
    print(
        f"Too little history: {too_short:,}"
    )
    print(
        f"Failures:           {failed:,}"
    )
    print(
        f"Total trades:       {len(trade_rows):,}"
    )
    print()

    for row in system_rows:
        print(
            f"{row['system']}:"
        )
        print(
            f"  Trades:            "
            f"{row['trades']:,}"
        )

        if row["trades"]:
            print(
                "  Win rate:          "
                f"{float(row['win_rate_pct']):.2f}%"
            )
            print(
                "  Median trade:      "
                f"{float(row['median_trade_return_pct']):+.2f}%"
            )
            print(
                "  Mean trade:        "
                f"{float(row['mean_trade_return_pct']):+.2f}%"
            )
            print(
                "  $5k ending proxy:  "
                f"${float(row['ending_equity_proxy']):,.2f}"
            )
            print(
                "  Proxy return:      "
                f"{float(row['return_pct_proxy']):+.2f}%"
            )
            print(
                "  Proxy max DD:      "
                f"{float(row['max_drawdown_pct_proxy']):.2f}%"
            )

        print()

    print("Reports:")
    print(
        f"  {trades_path}"
    )
    print(
        f"  {yearly_path}"
    )
    print(
        f"  {system_path}"
    )
    print(
        f"  {equity_path}"
    )
    print()
    print(
        "IMPORTANT: the $5,000 equity result above is currently a "
        "SEQUENTIAL PROXY. It does not yet model overlapping positions, "
        "portfolio unit limits, sector/correlation limits, margin, "
        "short borrow availability, commissions, or slippage."
    )
    print(
        "Use this run to validate trade mechanics and data integrity, "
        "not as the final historical Turtle performance claim."
    )
    print()


if __name__ == "__main__":
    main()
