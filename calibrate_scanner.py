from __future__ import annotations

from scanner.replay_dataset import (
    DEFAULT_OUTPUT_PATH,
    build_historical_replay,
    load_inputs,
    write_replay,
)


def main() -> None:
    print("=" * 88)
    print("HISTORICAL SCANNER EVIDENCE REPLAY")
    print("=" * 88)
    print()

    events, folds = load_inputs()

    print(f"Historical events: {len(events):,}")
    print(f"Walk-forward folds: {len(folds):,}")
    print()

    replay = build_historical_replay(
        events_df=events,
        folds_df=folds,
    )

    if replay.empty:
        raise RuntimeError(
            "Replay produced no matched evidence rows."
        )

    output_path = write_replay(
        replay,
        DEFAULT_OUTPUT_PATH,
    )

    print()
    print("=" * 88)
    print("REPLAY COMPLETE")
    print("=" * 88)

    print(f"Evidence rows:       {len(replay):,}")
    print(
        f"Unique events:       "
        f"{replay['event_id'].nunique():,}"
    )
    print(
        f"Years:               "
        f"{replay['test_year'].min()}–"
        f"{replay['test_year'].max()}"
    )
    print(
        f"Horizons:            "
        f"{sorted(replay['horizon_days'].unique())}"
    )
    print(
        f"Segmentations:       "
        f"{replay['segmentation'].nunique():,}"
    )
    print(f"Output:              {output_path}")

    print()
    print("MATCH COUNTS BY HORIZON")
    print("-" * 88)

    counts = (
        replay.groupby("horizon_days")
        .size()
        .sort_index()
    )

    for horizon, count in counts.items():
        print(
            f"{int(horizon):>2}D: "
            f"{int(count):>10,}"
        )


if __name__ == "__main__":
    main()