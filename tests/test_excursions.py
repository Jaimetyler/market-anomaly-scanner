from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.excursions import (
    calculate_horizon_excursions,
    contrarian_short_path_excursion,
)


def test_short_path_excursion_uses_intraperiod_high_low() -> None:
    bars = pd.DataFrame(
        {
            "high": [
                105.0,
                120.0,
                110.0,
            ],
            "low": [
                95.0,
                90.0,
                70.0,
            ],
        }
    )

    mfe, mae = (
        contrarian_short_path_excursion(
            bars,
            entry_price=100.0,
        )
    )

    assert mfe == pytest.approx(30.0)
    assert mae == pytest.approx(-20.0)


def test_horizon_excursions_use_only_each_horizon_path() -> None:
    bars = pd.DataFrame(
        {
            "high": [
                110.0,
                125.0,
                130.0,
                135.0,
                140.0,
            ],
            "low": [
                95.0,
                90.0,
                80.0,
                75.0,
                70.0,
            ],
        }
    )

    result = calculate_horizon_excursions(
        bars,
        entry_price=100.0,
        horizons=(1, 2, 5),
    )

    assert result[
        "contrarian_mfe_1d"
    ] == pytest.approx(5.0)

    assert result[
        "contrarian_mae_1d"
    ] == pytest.approx(-10.0)

    assert result[
        "contrarian_mfe_2d"
    ] == pytest.approx(10.0)

    assert result[
        "contrarian_mae_2d"
    ] == pytest.approx(-25.0)

    assert result[
        "contrarian_mfe_5d"
    ] == pytest.approx(30.0)

    assert result[
        "contrarian_mae_5d"
    ] == pytest.approx(-40.0)


def test_missing_horizon_remains_nan() -> None:
    bars = pd.DataFrame(
        {
            "high": [
                105.0,
                110.0,
            ],
            "low": [
                95.0,
                90.0,
            ],
        }
    )

    result = calculate_horizon_excursions(
        bars,
        entry_price=100.0,
        horizons=(1, 3),
    )

    assert np.isfinite(
        result["contrarian_mfe_1d"]
    )

    assert np.isnan(
        result["contrarian_mfe_3d"]
    )

    assert np.isnan(
        result["contrarian_mae_3d"]
    )


def test_invalid_entry_price_fails_loudly() -> None:
    bars = pd.DataFrame(
        {
            "high": [10.0],
            "low": [9.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="entry_price",
    ):
        calculate_horizon_excursions(
            bars,
            entry_price=0.0,
            horizons=(1,),
        )
