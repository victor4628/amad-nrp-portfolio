# ruff: noqa: N801, N999

"""Build the adjusted-price S&P 100 benchmark used by the research engine.

Adjusted prices incorporate cash distributions and splits. The benchmark is
aligned to the exact out-of-sample trading dates produced by the point-in-time
NRP backtest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import polars as pl

from .errors import (
    Exception_Calculation,
    Exception_Data_Validation_Error,
    Exception_Validation_Input,
)
from .errors import get_logger


if TYPE_CHECKING:
    from datetime import date

_LOGGER = get_logger(name=__name__)
_EPSILON = 1.0e-12
_MONTHS_PER_YEAR = 12
_REQUIRED_BENCHMARK_COLUMNS = {"Date", "SP100Benchmark"}


@attrs.define(kw_only=True, frozen=True, slots=True)
class SP100_Benchmark_Backtest_Result_QWIM:
    """Store the aligned S&P 100 benchmark result.

    Parameters
    ----------
    daily_returns : polars.DataFrame
        Daily adjusted-price returns and cumulative benchmark value.
    monthly_returns : polars.DataFrame
        Compounded monthly returns and cumulative benchmark value.
    metrics : polars.DataFrame
        Annualized performance metrics using the NRP reporting schema.
    """

    daily_returns: pl.DataFrame
    monthly_returns: pl.DataFrame
    metrics: pl.DataFrame


def run_SP100_benchmark_backtest_QWIM(
    *,
    benchmark_price_data: pl.DataFrame,
    comparison_dates: pl.Series,
) -> SP100_Benchmark_Backtest_Result_QWIM:
    """Run the adjusted-price S&P 100 benchmark on exact strategy dates.

    Parameters
    ----------
    benchmark_price_data : polars.DataFrame
        Adjusted prices with ``Date`` and ``SP100Benchmark`` columns. The panel must
        include one trading observation before the first comparison date.
    comparison_dates : polars.Series
        Exact daily out-of-sample dates from the NRP strategy.

    Returns
    -------
    SP100_Benchmark_Backtest_Result_QWIM
        Aligned daily/monthly returns and common performance metrics.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when prices or dates are malformed, incomplete, or non-positive.
    Exception_Validation_Input
        Raised when fewer than two comparison dates are supplied.
    Exception_Calculation
        Raised when benchmark returns or cumulative values are non-finite.

    Notes
    -----
    The function is pure: it does not mutate inputs or perform file I/O.
    Benchmark transaction costs are zero because the reported path represents
    one initial benchmark purchase held throughout the sample.
    """
    prepared_prices = _prepare_SP100_benchmark_prices_QWIM(
        benchmark_price_data=benchmark_price_data,
    )
    prepared_dates = _prepare_SP100_benchmark_comparison_dates_QWIM(
        comparison_dates=comparison_dates,
    )
    benchmark_returns = prepared_prices.with_columns(
        (
            pl.col("SP100Benchmark") / pl.col("SP100Benchmark").shift(1) - 1.0
        ).alias("NetReturn"),
    ).select("Date", "NetReturn")
    daily_returns = (
        pl.DataFrame({"Date": prepared_dates})
        .join(benchmark_returns, on="Date", how="left")
        .sort("Date")
    )
    if daily_returns.get_column("NetReturn").null_count() > 0:
        missing_dates = daily_returns.filter(pl.col("NetReturn").is_null()).get_column("Date")
        raise Exception_Data_Validation_Error(
            "S&P 100 benchmark prices do not cover every NRP out-of-sample date.",
            field="comparison_dates",
            value=missing_dates.to_list(),
        )
    return_values = daily_returns.get_column("NetReturn").to_numpy()
    if not np.all(np.isfinite(return_values)) or np.any(return_values <= -1.0):
        raise Exception_Calculation(
            "S&P 100 benchmark returns must be finite and greater than negative one.",
        )
    daily_returns = (
        daily_returns.with_columns(
            pl.col("NetReturn").alias("GrossReturn"),
            pl.lit(0.0).alias("TransactionCost"),
        )
        .with_columns(
            (1.0 + pl.col("GrossReturn")).cum_prod().alias("GrossValue"),
            (1.0 + pl.col("NetReturn")).cum_prod().alias("NetValue"),
        )
        .select(
            "Date",
            "GrossReturn",
            "TransactionCost",
            "NetReturn",
            "GrossValue",
            "NetValue",
        )
    )
    monthly_returns = _calculate_SP100_benchmark_monthly_returns_QWIM(
        daily_returns=daily_returns,
    )
    metrics = _calculate_SP100_benchmark_metrics_QWIM(
        monthly_returns=monthly_returns,
    )
    _LOGGER.info(
        "Completed S&P 100 benchmark: days=%s months=%s",
        daily_returns.height,
        monthly_returns.height,
    )
    return SP100_Benchmark_Backtest_Result_QWIM(
        daily_returns=daily_returns,
        monthly_returns=monthly_returns,
        metrics=metrics,
    )


def _prepare_SP100_benchmark_prices_QWIM(
    *,
    benchmark_price_data: pl.DataFrame,
) -> pl.DataFrame:
    """Validate and normalize the S&P 100 adjusted-price input."""
    if not isinstance(benchmark_price_data, pl.DataFrame):
        raise Exception_Data_Validation_Error(
            "S&P 100 benchmark prices must be a Polars DataFrame.",
            field="benchmark_price_data",
            value=type(benchmark_price_data).__name__,
        )
    missing_columns = _REQUIRED_BENCHMARK_COLUMNS - set(benchmark_price_data.columns)
    if missing_columns:
        raise Exception_Data_Validation_Error(
            "S&P 100 benchmark prices are missing required columns.",
            field="benchmark_price_data",
            value=sorted(missing_columns),
        )
    prepared_prices = benchmark_price_data.select(
        pl.col("Date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.strptime(pl.Date, strict=False)
        .alias("Date"),
        pl.col("SP100Benchmark").cast(pl.Float64, strict=False).fill_nan(None),
    ).sort("Date")
    invalid_prices = prepared_prices.filter(
        pl.col("Date").is_null()
        | pl.col("SP100Benchmark").is_null()
        | (pl.col("SP100Benchmark") <= 0.0),
    )
    if invalid_prices.height > 0 or prepared_prices.height < 3:
        raise Exception_Data_Validation_Error(
            "S&P 100 benchmark prices require at least three valid positive observations.",
            field="benchmark_price_data",
            value=invalid_prices.height,
        )
    if prepared_prices.get_column("Date").n_unique() != prepared_prices.height:
        raise Exception_Data_Validation_Error(
            "S&P 100 benchmark dates must be unique.",
            field="Date",
            value="duplicates",
        )
    return prepared_prices


def _prepare_SP100_benchmark_comparison_dates_QWIM(
    *,
    comparison_dates: pl.Series,
) -> list[date]:
    """Validate the exact strategy-date alignment requested for the benchmark."""
    if not isinstance(comparison_dates, pl.Series):
        raise Exception_Data_Validation_Error(
            "S&P 100 comparison_dates must be a Polars Series.",
            field="comparison_dates",
            value=type(comparison_dates).__name__,
        )
    prepared_dates = comparison_dates.cast(pl.Date, strict=False)
    if (
        prepared_dates.len() < 2
        or prepared_dates.null_count() > 0
        or prepared_dates.n_unique() != prepared_dates.len()
    ):
        raise Exception_Validation_Input(
            "S&P 100 comparison_dates require at least two unique non-null dates.",
            field_name="comparison_dates",
            actual_value=prepared_dates.len(),
        )
    return prepared_dates.sort().to_list()


def _calculate_SP100_benchmark_monthly_returns_QWIM(
    *,
    daily_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Compound aligned daily S&P 100 returns into calendar-month returns."""
    return (
        daily_returns.with_columns(
            pl.col("Date").dt.truncate("1mo").alias("Month"),
        )
        .group_by("Month")
        .agg(
            pl.col("Date").min().alias("EffectiveDate"),
            ((1.0 + pl.col("GrossReturn")).product() - 1.0).alias("GrossReturn"),
            pl.lit(0.0).alias("TransactionCost"),
            ((1.0 + pl.col("NetReturn")).product() - 1.0).alias("NetReturn"),
        )
        .sort("Month")
        .with_columns(
            (1.0 + pl.col("NetReturn")).cum_prod().alias("NetValue"),
        )
    )


def _calculate_SP100_benchmark_metrics_QWIM(
    *,
    monthly_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate common performance metrics for the S&P 100 benchmark."""
    return_values = monthly_returns.get_column("NetReturn").to_numpy().astype(np.float64)
    if return_values.size < 2:
        raise Exception_Calculation(
            "S&P 100 benchmark needs at least two out-of-sample months.",
        )
    annualized_return = float(
        np.prod(1.0 + return_values) ** (_MONTHS_PER_YEAR / return_values.size) - 1.0,
    )
    annualized_volatility = float(
        np.std(return_values, ddof=1) * np.sqrt(_MONTHS_PER_YEAR),
    )
    sharpe_ratio = (
        annualized_return / annualized_volatility if annualized_volatility > _EPSILON else 0.0
    )
    wealth_index = np.cumprod(1.0 + return_values)
    maximum_drawdown = float(
        np.min(wealth_index / np.maximum.accumulate(wealth_index) - 1.0),
    )
    return pl.DataFrame(
        [
            {
                "Strategy": "S&P 100 Benchmark",
                "ReturnPreprocessing": "benchmark",
                "OutOfSampleMonths": int(return_values.size),
                "AnnualizedReturn": annualized_return,
                "AnnualizedVolatility": annualized_volatility,
                "SharpeRatio": sharpe_ratio,
                "MaximumDrawdown": maximum_drawdown,
                "AverageMaximumWeight": None,
                "WorstMaximumWeight": None,
                "AverageEffectiveHoldings": None,
                "AverageTurnover": 0.0,
            },
        ],
        schema_overrides={
            "AverageMaximumWeight": pl.Float64,
            "WorstMaximumWeight": pl.Float64,
            "AverageEffectiveHoldings": pl.Float64,
        },
    )
