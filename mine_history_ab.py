from __future__ import annotations

# DESTINATION: repo root / mine_history_ab.py
#
# Purpose:
#   A/B the existing REST historical miner against the new local Massive
#   daily-flat-file history without replacing scanner/miner.py.
#
# Examples:
#   REST control:
#       ./.venv/Scripts/python.exe mine_history_ab.py \
#         --source rest --start 2025-01-02 --end 2025-03-31 --limit 250
#
#   Flat-file experiment:
#       ./.venv/Scripts/python.exe mine_history_ab.py \
#         --source flatfiles --start 2025-01-02 --end 2025-03-31 --limit 250

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, timedelta
from statistics import mean, median
import time
from typing import Any

from scanner.classifier import classify_setup
from scanner.corporate_actions import (
    check_corporate_action_risk,
    get_stock_splits,
)
from scanner.detector import detect_anomaly
from scanner.feature_frame import snapshots_by_session
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits
from scanner.flatfile_history import read_many_ticker_histories
from scanner.historical import calculate_historical_outcomes
from scanner.miner import (
    EVENT_MAX_GAP_SESSIONS,
    HISTORY_LOOKBACK_DAYS,
    MARKET_CALENDAR_TICKER,
    OUTCOME_LOOKFORWARD_DAYS,
    TICKER_WORKERS,
    HistoricalMinerConfig,
    HistoricalMinerResult,
    build_event_rows,
    build_research_row,
    build_rolling_research_selection,
    membership_dates_for_security,
    security_identity,
    mine_history as mine_history_rest,
    signal_dates_in_window,
    trading_dates_from_bars,
)


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _outcome_stats(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float | int | None]:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "negative": None,
            "down10": None,
            "down20": None,
            "up10": None,
            "up20": None,
        }

    count = len(values)

    return {
        "count": count,
        "mean": mean(values),
        "median": median(values),
        "negative": 100.0 * sum(v < 0 for v in values) / count,
        "down10": 100.0 * sum(v <= -10 for v in values) / count,
        "down20": 100.0 * sum(v <= -20 for v in values) / count,
        "up10": 100.0 * sum(v >= 10 for v in values) / count,
        "up20": 100.0 * sum(v >= 20 for v in values) / count,
    }


def _print_stats(
    title: str,
    rows: list[dict[str, Any]],
    field: str,
) -> None:
    stats = _outcome_stats(rows, field)

    print()
    print(title)
    print("-" * 72)
    print(f"Observations/events: {stats['count']:>10,}")

    if not stats["count"]:
        return

    print(f"Mean:                {_pct(stats['mean']):>10}")
    print(f"Median:              {_pct(stats['median']):>10}")
    print(f"Finished negative:   {_pct(stats['negative']):>10}")
    print(f"Down >= 10%:         {_pct(stats['down10']):>10}")
    print(f"Down >= 20%:         {_pct(stats['down20']):>10}")
    print(f"Up >= 10%:           {_pct(stats['up10']):>10}")
    print(f"Up >= 20%:           {_pct(stats['up20']):>10}")


def _flat_bars_to_massive_dicts(flat_bars) -> list[dict[str, object]]:
    return [
        bar.as_massive_aggregate()
        for bar in flat_bars
    ]


def _mine_one_flat_ticker(
    *,
    ticker: str,
    security_id: str,
    bars: list[dict],
    start_date: date,
    end_date: date,
    eligible_session_dates: set[str],
    split_events,
) -> tuple[list[dict[str, Any]], int, float]:
    started = time.perf_counter()

    if not bars:
        return [], 0, time.perf_counter() - started

    signal_dates = signal_dates_in_window(
        bars=bars,
        start_date=start_date,
        end_date=end_date,
    )

    signal_dates = [
        session
        for session in signal_dates
        if session in eligible_session_dates
    ]

    if not signal_dates:
        return [], 0, time.perf_counter() - started

    snapshot_index = snapshots_by_session(bars)

    candidates: list[
        tuple[str, dict[str, Any], Any, Any]
    ] = []

    for signal_date in signal_dates:
        snapshot = snapshot_index.get(signal_date)

        if snapshot is None:
            continue

        detection = detect_anomaly(snapshot)

        if not detection.candidate:
            continue

        classification = classify_setup(snapshot)

        candidates.append(
            (
                signal_date,
                snapshot,
                detection,
                classification,
            )
        )

    rows: list[dict[str, Any]] = []

    for (
        signal_date,
        snapshot,
        detection,
        classification,
    ) in candidates:
        corporate_action_check = check_corporate_action_risk(
            signal_date=signal_date,
            actions=split_events,
            return_1d=snapshot.get("return_1d"),
            return_5d=snapshot.get("return_5d"),
            return_20d=snapshot.get("return_20d"),
        )

        # INFORMATION WALL:
        # signal features/detection/classification are complete before
        # forward outcomes are calculated.
        outcomes = calculate_historical_outcomes(
            bars=bars,
            signal_date=signal_date,
        )

        rows.append(
            build_research_row(
                {
                    "ticker": ticker,
                    "security_id": security_id,
                    "signal_date": signal_date,
                    "snapshot": snapshot,
                    "detection": detection,
                    "classification": classification,
                    "corporate_action_check": corporate_action_check,
                    "outcomes": outcomes,
                }
            )
        )

    elapsed = time.perf_counter() - started
    return rows, len(signal_dates), elapsed


def mine_history_flatfiles(
    config: HistoricalMinerConfig,
    *,
    force_universe_refresh: bool = False,
    build_events: bool = True,
) -> HistoricalMinerResult:
    """
    Flat-file equivalent of scanner.miner.mine_history().

    Key differences:
      1. All selected ticker histories are read by scanning each local
         day-aggregate gzip ONCE via read_many_ticker_histories().
      2. Unadjusted flat-file OHLCV is normalized for stock splits before
         feature generation.
      3. No per-ticker REST aggregate requests are made.

    Reference/universe and split metadata still use the project's existing
    cached/API helpers.
    """
    if config.end_date < config.start_date:
        raise ValueError("end_date cannot be before start_date")

    fetch_start = config.start_date - timedelta(
        days=HISTORY_LOOKBACK_DAYS
    )
    fetch_end = config.end_date + timedelta(
        days=OUTCOME_LOOKFORWARD_DAYS
    )

    print("Building market-wide session calendar from local SPY bars...")

    # Read SPY separately only for the signal-window calendar. This scans the
    # local files once here; the bulk ticker pass below scans once more.
    spy_map = read_many_ticker_histories(
        [MARKET_CALENDAR_TICKER],
        config.start_date,
        config.end_date,
        progress=False,
    )
    spy_bars = _flat_bars_to_massive_dicts(
        spy_map.get(MARKET_CALENDAR_TICKER, [])
    )

    market_session_dates = [
        session
        for session in trading_dates_from_bars(spy_bars)
        if config.start_date
        <= date.fromisoformat(session)
        <= config.end_date
    ]

    if not market_session_dates:
        raise RuntimeError(
            "No local SPY sessions found for requested signal window. "
            "Download the required Massive daily aggregate files first."
        )

    print(f"Market sessions: {len(market_session_dates):,}")
    print()
    print("Building rolling point-in-time universe...")

    (
        selected,
        rolling_universe,
        raw_universe_count,
        core_universe_count,
    ) = build_rolling_research_selection(
        session_dates=market_session_dates,
        max_tickers=config.max_tickers,
        force_universe_refresh=force_universe_refresh,
    )

    print(
        f"Universe snapshots from API:   "
        f"{rolling_universe.fetched_snapshots:,}"
    )
    print(
        f"Universe snapshots from cache: "
        f"{rolling_universe.cached_snapshots:,}"
    )

    # One ticker history can serve multiple non-overlapping historical
    # security identities. Load each ticker only once.
    tickers = sorted({security.ticker for security in selected})

    print()
    print(
        "Loading local flat-file histories in one pass..."
    )
    print(
        f"Requested local history: {fetch_start.isoformat()} "
        f"-> {fetch_end.isoformat()}"
    )

    histories = read_many_ticker_histories(
        tickers,
        fetch_start,
        fetch_end,
        progress=True,
    )

    print()
    print(
        "Loading split metadata and normalizing local histories..."
    )
    print(
        f"Split lookups are hard-bounded to {len(tickers):,} selected tickers "
        f"with at most {min(TICKER_WORKERS, len(tickers))} concurrent workers."
    )

    split_events_by_ticker: dict[str, list] = {}
    adjusted_bars_by_ticker: dict[str, list[dict]] = {}

    split_worker_count = min(
        TICKER_WORKERS,
        max(1, len(tickers)),
    )

    def load_and_adjust(ticker: str):
        events = get_stock_splits(ticker=ticker)
        adjusted = adjust_flatfile_history_for_splits(
            histories.get(ticker, []),
            events,
        )
        return (
            ticker,
            events,
            _flat_bars_to_massive_dicts(adjusted),
        )

    with ThreadPoolExecutor(
        max_workers=split_worker_count
    ) as executor:
        futures = {
            executor.submit(load_and_adjust, ticker): ticker
            for ticker in tickers
        }

        done_count = 0

        for future in futures:
            ticker, events, bars = future.result()
            done_count += 1
            split_events_by_ticker[ticker] = events
            adjusted_bars_by_ticker[ticker] = bars

            if (
                done_count == 1
                or done_count % 25 == 0
                or done_count == len(tickers)
            ):
                print(
                    f"  normalized {done_count:>3}/{len(tickers):<3} "
                    f"tickers"
                )

    print()
    worker_count = min(
        TICKER_WORKERS,
        max(1, len(tickers)),
    )
    print(
        f"Mining local histories with {worker_count} bounded workers..."
    )
    print()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    sessions_tested = 0
    tickers_completed = 0

    completed_results: dict[
        int,
        tuple[str, list[dict[str, Any]], int, float]
    ] = {}

    total = len(selected)

    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        next_to_submit = 0
        in_flight: dict[Any, tuple[int, str]] = {}

        def submit_one(selection_index: int) -> None:
            security = selected[selection_index]
            ticker = security.ticker

            security_id = security_identity(security)

            eligible_session_dates = membership_dates_for_security(
                security=security,
                session_dates=market_session_dates,
                rolling_universe=rolling_universe,
            )

            future = executor.submit(
                _mine_one_flat_ticker,
                ticker=ticker,
                security_id=security_id,
                bars=adjusted_bars_by_ticker.get(ticker, []),
                start_date=config.start_date,
                end_date=config.end_date,
                eligible_session_dates=eligible_session_dates,
                split_events=split_events_by_ticker.get(ticker, []),
            )

            in_flight[future] = (
                selection_index,
                ticker,
            )

        while (
            next_to_submit < total
            and len(in_flight) < worker_count
        ):
            submit_one(next_to_submit)
            next_to_submit += 1

        done_count = 0

        while in_flight:
            done, _ = wait(
                in_flight,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                selection_index, ticker = in_flight.pop(
                    future
                )
                done_count += 1

                try:
                    (
                        ticker_rows,
                        ticker_sessions,
                        elapsed,
                    ) = future.result()

                    completed_results[selection_index] = (
                        ticker,
                        ticker_rows,
                        ticker_sessions,
                        elapsed,
                    )

                    print(
                        f"[{done_count:>3}/{total:<3} done] "
                        f"#{selection_index + 1:<3} "
                        f"{ticker:<8} "
                        f"{elapsed:>6.2f}s "
                        f"sessions={ticker_sessions:>3} "
                        f"anomalies={len(ticker_rows):>3}"
                    )

                except Exception as exc:
                    completed_results[selection_index] = (
                        ticker,
                        [],
                        0,
                        0.0,
                    )

                    errors.append(
                        {
                            "ticker": ticker,
                            "error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )

                    print(
                        f"[{done_count:>3}/{total:<3} done] "
                        f"#{selection_index + 1:<3} "
                        f"{ticker:<8} FAILED "
                        f"{type(exc).__name__}: {exc}"
                    )

                if next_to_submit < total:
                    submit_one(next_to_submit)
                    next_to_submit += 1

    for selection_index in range(total):
        result = completed_results.get(selection_index)

        if result is None:
            continue

        (
            ticker,
            ticker_rows,
            ticker_sessions,
            _elapsed,
        ) = result

        if any(
            error["ticker"] == ticker
            for error in errors
        ):
            continue

        rows.extend(ticker_rows)
        sessions_tested += ticker_sessions
        tickers_completed += 1

    research_eligible = sum(
        1
        for row in rows
        if row["research_eligible"]
    )

    ca_exclusions = len(rows) - research_eligible

    if build_events:
        event_rows = build_event_rows(
            rows=rows,
            session_dates=market_session_dates,
        )
    else:
        # Master-dataset production checkpoints DAILY observations first and
        # clusters events once, globally, after all years are safely on disk.
        event_rows = []

    return HistoricalMinerResult(
        config=config,
        universe_date=config.start_date,
        raw_universe_count=raw_universe_count,
        core_universe_count=core_universe_count,
        selected_ticker_count=len(selected),
        tickers_completed=tickers_completed,
        tickers_failed=len(errors),
        market_sessions=len(market_session_dates),
        sessions_tested=sessions_tested,
        anomaly_observations=len(rows),
        research_eligible_observations=research_eligible,
        corporate_action_exclusions=ca_exclusions,
        anomaly_events=len(event_rows),
        rows=rows,
        event_rows=event_rows,
        errors=errors,
    )


def print_result(
    *,
    source: str,
    result: HistoricalMinerResult,
) -> None:
    print()
    print("=" * 100)
    print(f"MINER SUMMARY — {source.upper()}")
    print("=" * 100)
    print(
        f"Raw universe union:           "
        f"{result.raw_universe_count:>8,}"
    )
    print(
        f"Core research universe union: "
        f"{result.core_universe_count:>8,}"
    )
    print(
        f"Tickers selected:             "
        f"{result.selected_ticker_count:>8,}"
    )
    print(
        f"Tickers completed:            "
        f"{result.tickers_completed:>8,}"
    )
    print(
        f"Tickers failed:               "
        f"{result.tickers_failed:>8,}"
    )
    print(
        f"Market sessions:              "
        f"{result.market_sessions:>8,}"
    )
    print(
        f"Ticker sessions tested:       "
        f"{result.sessions_tested:>8,}"
    )
    print(
        f"Daily anomaly observations:   "
        f"{result.anomaly_observations:>8,}"
    )
    print(
        f"Research eligible:            "
        f"{result.research_eligible_observations:>8,}"
    )
    print(
        f"Corporate-action exclusions:  "
        f"{result.corporate_action_exclusions:>8,}"
    )
    print(
        f"Clustered anomaly events:     "
        f"{result.anomaly_events:>8,}"
    )

    eligible_rows = [
        row
        for row in result.rows
        if row.get("research_eligible")
    ]

    _print_stats(
        "DAILY OBSERVATION NEXT-OPEN -> 5-SESSION OUTCOMES",
        eligible_rows,
        "executable_return_5d",
    )

    _print_stats(
        "EVENT-ENTRY NEXT-OPEN -> 5-SESSION OUTCOMES",
        result.event_rows,
        "executable_return_5d",
    )

    if result.errors:
        print()
        print("ERRORS")
        print("-" * 72)
        for error in result.errors:
            print(
                f"{error['ticker']}: {error['error']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B historical miner: existing REST bars vs local "
            "Massive day-aggregate flat files."
        )
    )
    parser.add_argument(
        "--source",
        choices=("rest", "flatfiles"),
        required=True,
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--force-universe-refresh",
        action="store_true",
    )

    args = parser.parse_args()

    config = HistoricalMinerConfig(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        max_tickers=args.limit,
    )

    print("=" * 100)
    print("HISTORICAL MINER A/B")
    print("=" * 100)
    print(f"Source:        {args.source}")
    print(f"Signal window: {args.start} -> {args.end}")
    print(f"Ticker limit:  {args.limit}")
    print()

    started = time.perf_counter()

    if args.source == "rest":
        result = mine_history_rest(
            config,
            force_universe_refresh=(
                args.force_universe_refresh
            ),
        )
    else:
        result = mine_history_flatfiles(
            config,
            force_universe_refresh=(
                args.force_universe_refresh
            ),
        )

    elapsed = time.perf_counter() - started

    print_result(
        source=args.source,
        result=result,
    )

    print()
    print(f"Elapsed: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())