from __future__ import annotations

import pandas as pd
import pytest

from research_master_dataset import adapt_master_events


def _master_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_number": 1,
                "ticker": "AAA",
                "event_start_date": "2024-01-02",
                "initial_setup": "FRESH_SPIKE",
                "entry_signal_close": 10.0,
                "entry_relative_volume": 4.2,
                "entry_distance_sma_20_pct": 0.30,
                "entry_return_1d": 0.20,
                "entry_return_3d": 0.35,
                "entry_return_5d": 0.50,
                "executable_return_1d": -0.05,
                "executable_return_2d": -0.08,
                "executable_return_3d": -0.10,
                "executable_return_5d": -0.15,
                "executable_return_10d": -0.20,
                "executable_return_20d": -0.25,
                "contrarian_mfe_1d": 4.0,
                "contrarian_mae_1d": -3.0,
                "contrarian_mfe_2d": 7.0,
                "contrarian_mae_2d": -5.0,
                "contrarian_mfe_3d": 9.0,
                "contrarian_mae_3d": -6.0,
                "contrarian_mfe_5d": 18.0,
                "contrarian_mae_5d": -8.0,
                "contrarian_mfe_10d": 24.0,
                "contrarian_mae_10d": -11.0,
                "contrarian_mfe_20d": 31.0,
                "contrarian_mae_20d": -15.0,
            }
        ]
    )


def test_adapt_master_events_maps_features() -> None:
    result = adapt_master_events(_master_events())
    row = result.iloc[0]

    assert row["event_id"] == "AAA|2024-01-02|1"
    assert row["event_direction"] == 1
    assert row["event_close"] == pytest.approx(10.0)
    assert row["rvol"] == pytest.approx(4.2)
    assert row["sma20_extension_pct"] == pytest.approx(0.30)
    assert row["return_5d_event"] == pytest.approx(0.50)


def test_adapt_master_events_converts_short_side_returns() -> None:
    result = adapt_master_events(_master_events())
    row = result.iloc[0]

    assert row["return_5d"] == pytest.approx(-0.15)
    assert row["contrarian_return_5d"] == pytest.approx(0.15)
    assert bool(row["reversed_5d"]) is True
    assert bool(row["outcome_available_5d"]) is True


def test_adapt_master_events_maps_horizon_mfe_mae() -> None:
    result = adapt_master_events(_master_events())
    row = result.iloc[0]

    assert row["contrarian_mfe_5d"] == pytest.approx(18.0)
    assert row["contrarian_mae_5d"] == pytest.approx(-8.0)


def test_missing_forward_return_is_marked_unavailable() -> None:
    frame = _master_events()
    frame.loc[0, "executable_return_20d"] = None

    result = adapt_master_events(frame)
    row = result.iloc[0]

    assert bool(row["outcome_available_20d"]) is False
    assert pd.isna(row["contrarian_return_20d"])


def test_missing_required_column_fails_loudly() -> None:
    frame = _master_events().drop(columns=["entry_relative_volume"])

    with pytest.raises(ValueError, match="entry_relative_volume"):
        adapt_master_events(frame)
