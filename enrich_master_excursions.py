from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from scanner.excursions import (
    DEFAULT_HORIZONS,
    calculate_horizon_excursions,
)
from scanner.corporate_actions import get_stock_splits
from scanner.flatfile_adjustments import adjust_flatfile_history_for_splits


DEFAULT_EVENTS = Path(
    "data/research/master_anomalies_2019_2025.csv"
)

DEFAULT_OUTPUT = Path(
    "data/research/"
    "master_anomalies_2019_2025_excursions.csv"
)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Master event CSV is missing required columns: "
            + ", ".join(missing)
        )


def _normalise_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)

    frame = history.copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )

    # If date lives in the index, expose it as a column.
    if "date" not in frame.columns:
        index_name = frame.index.name

        if index_name:
            frame = frame.reset_index()

        if "date" not in frame.columns:
            candidates = [
                column
                for column in frame.columns
                if str(column).lower()
                in {
                    "date",
                    "session_date",
                    "timestamp",
                    "datetime",
                    "time",
                }
            ]

            if candidates:
                frame = frame.rename(
                    columns={
                        candidates[0]: "date"
                    }
                )

    # Case-insensitive OHLC/date normalization.
    rename: dict[str, str] = {}

    for column in frame.columns:
        lower = str(column).lower()

        if lower in {
            "date",
            "session_date",
            "timestamp",
            "datetime",
            "time",
        }:
            rename[column] = "date"

        elif lower == "open":
            rename[column] = "open"

        elif lower == "high":
            rename[column] = "high"

        elif lower == "low":
            rename[column] = "low"

        elif lower == "close":
            rename[column] = "close"

    frame = frame.rename(
        columns=rename
    )

    required = {
        "date",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Flat-file history is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.normalize()

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = (
        frame.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return frame


def _history_reader() -> Callable:
    """
    Keep the enrichment layer thin: the provider-specific flat-file reader
    remains the source of truth for ticker identity and local bar loading.
    """
    try:
        from scanner.flatfile_history import (
            read_ticker_history,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Could not import "
            "scanner.flatfile_history.read_ticker_history"
        ) from exc

    return read_ticker_history


def _load_history_compat(
    reader: Callable,
    *,
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Call the existing flat-file reader without forcing a new provider API.

    The project has changed signatures during development, so this wrapper
    accepts the common ticker/start/end forms while keeping provider ticker
    case EXACT. We never uppercase ticker here.
    """
    start_text = (
        start_date.strftime("%Y-%m-%d")
    )
    end_text = (
        end_date.strftime("%Y-%m-%d")
    )

    attempts = [
        lambda: reader(
            ticker,
            start=start_text,
            end=end_text,
        ),
        lambda: reader(
            ticker,
            start_date=start_text,
            end_date=end_text,
        ),
        lambda: reader(
            ticker=ticker,
            start=start_text,
            end=end_text,
        ),
        lambda: reader(
            ticker=ticker,
            start_date=start_text,
            end_date=end_text,
        ),
        lambda: reader(
            symbol=ticker,
            start=start_text,
            end=end_text,
        ),
        lambda: reader(
            symbol=ticker,
            start_date=start_text,
            end_date=end_text,
        ),
        lambda: reader(
            ticker
        ),
        lambda: reader(
            ticker=ticker
        ),
        lambda: reader(
            symbol=ticker
        ),
    ]

    errors: list[str] = []

    for attempt in attempts:
        try:
            history = attempt()
            frame = _normalise_history(
                history
            )

            return frame[
                frame["date"].between(
                    start_date,
                    end_date,
                )
            ].reset_index(drop=True)

        except TypeError as exc:
            errors.append(str(exc))

    raise RuntimeError(
        "Could not call read_ticker_history() "
        f"for exact ticker {ticker!r}.\n"
        "Observed call errors:\n- "
        + "\n- ".join(errors[:6])
    )


def _entry_matches_bar_open(
    expected_next_open: float,
    actual_open: float,
) -> bool:
    if (
        not np.isfinite(expected_next_open)
        or not np.isfinite(actual_open)
    ):
        return False

    tolerance = max(
        0.01,
        abs(expected_next_open)
        * 1e-5,
    )

    return (
        abs(
            expected_next_open
            - actual_open
        )
        <= tolerance
    )


def enrich_master_events(
    events: pd.DataFrame,
    *,
    reader: Callable,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    horizons = tuple(
        sorted(
            {
                int(horizon)
                for horizon in horizons
            }
        )
    )

    required = [
        "ticker",
        "entry_signal_date",
        "entry_next_open",
    ]

    required.extend(
        f"executable_return_{horizon}d"
        for horizon in horizons
    )

    _require_columns(
        events,
        required,
    )

    result = events.copy()

    result["_event_date"] = pd.to_datetime(
        result["entry_signal_date"],
        errors="coerce",
    ).dt.normalize()

    result["_next_open_numeric"] = (
        pd.to_numeric(
            result["entry_next_open"],
            errors="coerce",
        )
    )

    for horizon in horizons:
        suffix = f"{horizon}d"
        result[
            f"contrarian_mfe_{suffix}"
        ] = np.nan
        result[
            f"contrarian_mae_{suffix}"
        ] = np.nan

    result[
        "excursion_entry_open_verified"
    ] = False

    result[
        "excursion_error"
    ] = None

    grouped = result.groupby(
        "ticker",
        sort=False,
        dropna=False,
    )

    ticker_count = len(grouped)

    print(
        f"Exact ticker groups: {ticker_count:,}"
    )

    processed = 0
    enriched_rows = 0
    entry_mismatches = 0
    no_history = 0

    for ticker_value, indexes in grouped.groups.items():
        processed += 1

        ticker = str(ticker_value)

        group = result.loc[
            list(indexes)
        ]

        valid_dates = group[
            "_event_date"
        ].dropna()

        if valid_dates.empty:
            result.loc[
                list(indexes),
                "excursion_error",
            ] = "invalid_event_date"
            continue

        # 20 sessions is usually < 30 calendar days.
        # Use a generous range without loading the entire provider archive.
        start_date = (
            valid_dates.min()
            - pd.Timedelta(days=7)
        )
        end_date = (
            valid_dates.max()
            + pd.Timedelta(days=60)
        )

        try:
            raw_history = _load_history_compat(
                reader,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )

            split_events = get_stock_splits(
                ticker=ticker
            )

            history = adjust_flatfile_history_for_splits(
                raw_history,
                split_events,
            )

            history = _normalise_history(
                history
            )
        except Exception as exc:
            result.loc[
                list(indexes),
                "excursion_error",
            ] = (
                "history_load_failed: "
                + str(exc)[:180]
            )
            no_history += len(group)
            continue

        if history.empty:
            result.loc[
                list(indexes),
                "excursion_error",
            ] = "no_history"
            no_history += len(group)
            continue

        for row_index in indexes:
            signal_date = result.at[
                row_index,
                "_event_date",
            ]

            entry_price = result.at[
                row_index,
                "_next_open_numeric",
            ]

            if (
                pd.isna(signal_date)
                or not np.isfinite(entry_price)
                or entry_price <= 0
            ):
                result.at[
                    row_index,
                    "excursion_error",
                ] = "invalid_entry"
                continue

            # Executable entry occurs at the NEXT session's open.
            future = history.loc[
                history["date"] > signal_date
            ].reset_index(drop=True)

            if future.empty:
                result.at[
                    row_index,
                    "excursion_error",
                ] = "no_future_bars"
                continue

            actual_first_open = float(
                future.iloc[0]["open"]
            )

            if not _entry_matches_bar_open(
                float(entry_price),
                actual_first_open,
            ):
                result.at[
                    row_index,
                    "excursion_error",
                ] = (
                    "next_open_mismatch:"
                    f" master={float(entry_price):.8f}"
                    f" flatfile={actual_first_open:.8f}"
                )
                entry_mismatches += 1
                continue

            result.at[
                row_index,
                "excursion_entry_open_verified",
            ] = True

            metrics = (
                calculate_horizon_excursions(
                    future,
                    entry_price=float(
                        entry_price
                    ),
                    horizons=horizons,
                )
            )

            wrote_any = False

            for horizon in horizons:
                suffix = f"{horizon}d"

                # Preserve the existing research availability convention:
                # if the executable horizon outcome is unavailable, excursion
                # at that horizon must also remain unavailable.
                outcome_value = pd.to_numeric(
                    pd.Series(
                        [
                            result.at[
                                row_index,
                                f"executable_return_{suffix}",
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if pd.isna(outcome_value):
                    continue

                mfe = metrics[
                    f"contrarian_mfe_{suffix}"
                ]
                mae = metrics[
                    f"contrarian_mae_{suffix}"
                ]

                result.at[
                    row_index,
                    f"contrarian_mfe_{suffix}",
                ] = mfe

                result.at[
                    row_index,
                    f"contrarian_mae_{suffix}",
                ] = mae

                if (
                    np.isfinite(mfe)
                    and np.isfinite(mae)
                ):
                    wrote_any = True

            if wrote_any:
                enriched_rows += 1

        if (
            processed % 250 == 0
            or processed == ticker_count
        ):
            print(
                f"[{processed:,}/{ticker_count:,} tickers] "
                f"enriched_events={enriched_rows:,} "
                f"entry_mismatches={entry_mismatches:,} "
                f"no_history={no_history:,}"
            )

    result = result.drop(
        columns=[
            "_event_date",
            "_next_open_numeric",
        ]
    )

    print()
    print(
        "Excursion enrichment complete:"
    )
    print(
        f"  rows:              {len(result):,}"
    )
    print(
        f"  enriched rows:     {enriched_rows:,}"
    )
    print(
        f"  entry mismatches:  {entry_mismatches:,}"
    )
    print(
        f"  no history rows:   {no_history:,}"
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add horizon-specific executable "
            "contrarian MFE/MAE to the finalized "
            "historical anomaly master dataset."
        )
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.events.exists():
        raise FileNotFoundError(
            f"Master event CSV does not exist: "
            f"{args.events}"
        )

    print(
        f"Loading master events: {args.events}"
    )

    events = pd.read_csv(
        args.events,
        low_memory=False,
    )

    print(
        f"Master events loaded: {len(events):,}"
    )

    reader = _history_reader()

    enriched = enrich_master_events(
        events,
        reader=reader,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = args.output.with_name(
        args.output.name + ".tmp"
    )

    enriched.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(
        args.output
    )

    print(
        f"Wrote: {args.output}"
    )


if __name__ == "__main__":
    main()
