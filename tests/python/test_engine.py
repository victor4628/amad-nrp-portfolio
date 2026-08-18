from datetime import date, timedelta

import numpy as np
import polars as pl

from amad_nrp import (
    NRP_AMAD_Config_QWIM,
    NRP_Daily_Approximation_Config_QWIM,
    apply_AMAD_returns_NRP_QWIM,
    calculate_NRP_daily_approximation_allocation_QWIM,
    run_SP100_benchmark_backtest_QWIM,
)


def _dated_returns(values: list[float]) -> pl.DataFrame:
    start = date(2025, 1, 1)
    return pl.DataFrame(
        {
            "Date": [start + timedelta(days=index) for index in range(len(values))],
            "Asset": values,
        }
    )


def test_amad_is_causal_and_dampens_large_observations() -> None:
    prefix = [0.001, -0.001, 0.002, -0.002, 0.0015, 0.08]
    first = apply_AMAD_returns_NRP_QWIM(
        returns_data=_dated_returns(prefix + [0.0]),
        config=NRP_AMAD_Config_QWIM(window_size=5, threshold=3.0),
    )
    second = apply_AMAD_returns_NRP_QWIM(
        returns_data=_dated_returns(prefix + [0.25]),
        config=NRP_AMAD_Config_QWIM(window_size=5, threshold=3.0),
    )

    assert first.get_column("Asset")[: len(prefix)].to_list() == second.get_column("Asset")[: len(prefix)].to_list()
    assert abs(first.get_column("Asset")[5]) < abs(prefix[5])


def test_amad_matches_the_documented_logarithmic_formula() -> None:
    result = apply_AMAD_returns_NRP_QWIM(
        returns_data=_dated_returns([-1.0, 0.0, 1.0, 2.0, 10.0]),
        config=NRP_AMAD_Config_QWIM(window_size=5, threshold=3.0),
    )

    # The current observation is part of the window: median=1, MAD=1, z=9.
    expected = 1.0 + 3.0 + np.log(9.0 - (3.0 - 1.0))
    assert np.isclose(result.get_column("Asset")[-1], expected)


def test_network_allocation_is_long_only_and_fully_invested() -> None:
    covariance = np.array(
        [
            [0.040, 0.012, 0.004],
            [0.012, 0.050, 0.010],
            [0.004, 0.010, 0.030],
        ]
    )
    result = calculate_NRP_daily_approximation_allocation_QWIM(
        asset_names=("A", "B", "C"),
        realized_covariance_matrices=(covariance,),
        config=NRP_Daily_Approximation_Config_QWIM(estimation_months=1),
    )

    assert np.isclose(result.weights.sum(), 1.0)
    assert np.all(result.weights > 0.0)
    assert result.edge_data.height == 2


def test_sp100_benchmark_uses_the_standard_benchmark_contract() -> None:
    dates = [
        date(2025, 1, 1),
        date(2025, 1, 31),
        date(2025, 2, 28),
        date(2025, 3, 31),
        date(2025, 4, 30),
    ]
    prices = pl.DataFrame(
        {
            "Date": dates,
            "SP100Benchmark": [100.0, 101.0, 102.0, 100.0, 104.0],
        },
    )
    result = run_SP100_benchmark_backtest_QWIM(
        benchmark_price_data=prices,
        comparison_dates=pl.Series("Date", dates[1:]),
    )

    assert result.daily_returns.height == 4
    assert result.metrics.get_column("Strategy").item() == "S&P 100 Benchmark"
