# ruff: noqa: N801, N999

"""Backtest NRP with point-in-time index membership.

This module keeps the NRP allocation mathematics in
``_NRP_daily_approximation`` unchanged and adds only the variable-universe
research layer required for historical S&P 100 membership. At each monthly
rebalance, the eligible universe is selected from the membership intervals
known on the effective date and from prices available over the complete
estimation and holding window.

Notes
-----
The implementation uses functional data transformations and pure numerical
helpers. File reads, artifact writes, and plotting remain in the CLI script.
No predecessor history or synthetic price backfill is permitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import polars as pl

from .amad import (
    NRP_AMAD_Config_QWIM,
    apply_AMAD_returns_NRP_QWIM,
)
from .core import (
    _calculate_monthly_metrics_NRP_daily_QWIM,
    _drift_weights_NRP_daily_QWIM,
    calculate_daily_log_returns_NRP_daily_approximation_QWIM,
    calculate_monthly_realized_covariances_NRP_daily_approximation_QWIM,
    calculate_NRP_daily_approximation_allocation_QWIM,
)
from .errors import (
    Exception_Data_Validation_Error,
    Exception_Validation_Input,
)
from .errors import get_logger


if TYPE_CHECKING:
    from datetime import date

    from src.models.portfolio_optimization._NRP_daily_approximation import (
        NRP_Daily_Allocation_QWIM,
        NRP_Daily_Approximation_Config_QWIM,
    )

_LOGGER = get_logger(name=__name__)
_REQUIRED_MEMBERSHIP_COLUMNS = {
    "Membership_Start",
    "Membership_End_Exclusive",
    "Price_Ticker",
}


@attrs.define(kw_only=True, frozen=True, slots=True)
class NRP_Point_In_Time_Backtest_Result_QWIM:
    """Store a complete variable-universe NRP backtest.

    Parameters
    ----------
    strategy_name : str
        Explicit strategy label including preprocessing conventions.
    daily_returns : polars.DataFrame
        Daily out-of-sample gross and net returns with cumulative net value.
    monthly_returns : polars.DataFrame
        Monthly out-of-sample returns and cumulative net value.
    rebalance_weights : polars.DataFrame
        Long-form target weight history for every effective date.
    allocation_diagnostics : polars.DataFrame
        Asset centrality and target weight at every rebalance.
    network_edges : polars.DataFrame
        Selected MST edges for every rebalance.
    rebalance_diagnostics : polars.DataFrame
        Universe, concentration, spectral, and turnover diagnostics.
    membership_audit : polars.DataFrame
        Active-member eligibility decision at every rebalance.
    metrics : polars.DataFrame
        Annualized performance and concentration summary.
    """

    strategy_name: str
    daily_returns: pl.DataFrame
    monthly_returns: pl.DataFrame
    rebalance_weights: pl.DataFrame
    allocation_diagnostics: pl.DataFrame
    network_edges: pl.DataFrame
    rebalance_diagnostics: pl.DataFrame
    membership_audit: pl.DataFrame
    metrics: pl.DataFrame


def combine_adjusted_price_panels_NRP_point_in_time_QWIM(
    *,
    price_data_base: pl.DataFrame,
    price_data_supplemental: pl.DataFrame,
) -> pl.DataFrame:
    """Combine a base adjusted-price panel with supplemental ticker columns.

    Parameters
    ----------
    price_data_base : polars.DataFrame
        Primary wide adjusted-price panel with ``Date`` first.
    price_data_supplemental : polars.DataFrame
        Supplemental wide panel on the same trading-date convention.

    Returns
    -------
    polars.DataFrame
        Base trading calendar with non-overlapping supplemental columns joined
        by ``Date``.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised for malformed dates, duplicate columns, or missing supplemental
        overlap with the base calendar.
    """
    base_data = _prepare_price_data_NRP_point_in_time_QWIM(
        price_data=price_data_base,
    )
    supplemental_data = _prepare_price_data_NRP_point_in_time_QWIM(
        price_data=price_data_supplemental,
        minimum_asset_count=1,
    )
    duplicate_assets = (set(base_data.columns) & set(supplemental_data.columns)) - {"Date"}
    if duplicate_assets:
        raise Exception_Data_Validation_Error(
            "Supplemental adjusted-price columns must not duplicate base assets.",
            field="price_data_supplemental",
            value=sorted(duplicate_assets),
        )
    combined_data = base_data.join(
        supplemental_data,
        on="Date",
        how="left",
    ).sort("Date")
    supplemental_assets = [
        item_column for item_column in supplemental_data.columns if item_column != "Date"
    ]
    if all(
        combined_data.get_column(item_asset).null_count() == combined_data.height
        for item_asset in supplemental_assets
    ):
        raise Exception_Data_Validation_Error(
            "Supplemental prices do not overlap the base trading calendar.",
            field="Date",
            value="no_overlap",
        )
    return combined_data


def run_NRP_point_in_time_backtest_QWIM(
    *,
    price_data: pl.DataFrame,
    membership_intervals: pl.DataFrame,
    config: NRP_Daily_Approximation_Config_QWIM,
) -> NRP_Point_In_Time_Backtest_Result_QWIM:
    """Run monthly NRP using only members and histories known at each date.

    Parameters
    ----------
    price_data : polars.DataFrame
        Wide adjusted-price panel. Nulls are allowed outside an asset's
        available history but not inside an eligible estimation/holding window.
    membership_intervals : polars.DataFrame
        Point-in-time intervals with inclusive ``Membership_Start`` and
        exclusive ``Membership_End_Exclusive``.
    config : NRP_Daily_Approximation_Config_QWIM
        Existing NRP mathematical and preprocessing settings.

    Returns
    -------
    NRP_Point_In_Time_Backtest_Result_QWIM
        Daily and monthly paths, full weights, networks, membership audit, and
        performance diagnostics.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when price or membership inputs are malformed.
    Exception_Validation_Input
        Raised when fewer than two eligible assets remain at a rebalance.
    Exception_Calculation
        Raised when NRP allocation or holding-period drift is undefined.

    Notes
    -----
    Membership is evaluated on the first trading day of each holding month.
    The preceding ``config.estimation_months`` calendar covariance matrices
    are used without holding-month information. AMAD, when enabled, affects
    estimation returns only. Target weights become effective on the first
    trading day and drift through the month.
    """
    prepared_prices = _prepare_price_data_NRP_point_in_time_QWIM(
        price_data=price_data,
    )
    prepared_membership = _prepare_membership_NRP_point_in_time_QWIM(
        membership_intervals=membership_intervals,
    )
    month_data = _create_month_calendar_NRP_point_in_time_QWIM(
        price_data=prepared_prices,
    )
    if month_data.height <= config.estimation_months:
        raise Exception_Validation_Input(
            "Point-in-time NRP requires more months than its estimation window.",
            field_name="estimation_months",
            actual_value={
                "available_months": month_data.height,
                "required_months": config.estimation_months + 1,
            },
        )

    daily_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    network_frames: list[pl.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    membership_frames: list[pl.DataFrame] = []
    previous_end_weights: dict[str, float] = {}
    gross_portfolio_value = 1.0
    net_portfolio_value = 1.0

    for idx_month in range(config.estimation_months, month_data.height):
        effective_month = month_data.get_column("Month").item(idx_month)
        effective_date = month_data.get_column("EffectiveDate").item(idx_month)
        holding_end_date = month_data.get_column("HoldingEndDate").item(idx_month)
        estimation_months = tuple(
            month_data.get_column("Month")
            .slice(idx_month - config.estimation_months, config.estimation_months)
            .to_list(),
        )
        membership_audit, eligible_assets = _select_eligible_members_NRP_point_in_time_QWIM(
            price_data=prepared_prices,
            membership_intervals=prepared_membership,
            effective_date=effective_date,
            holding_end_date=holding_end_date,
            estimation_months=estimation_months,
        )
        membership_frames.append(membership_audit)
        if len(eligible_assets) < 2:
            raise Exception_Validation_Input(
                "Point-in-time NRP requires at least two eligible members.",
                field_name="eligible_assets",
                actual_value={
                    "effective_date": effective_date,
                    "eligible_assets": len(eligible_assets),
                },
            )
        allocation = _calculate_allocation_NRP_point_in_time_QWIM(
            price_data=prepared_prices,
            eligible_assets=eligible_assets,
            effective_month=effective_month,
            estimation_months=estimation_months,
            config=config,
        )
        target_weights = {
            item_asset: float(allocation.weights[idx_asset])
            for idx_asset, item_asset in enumerate(eligible_assets)
        }
        turnover = _calculate_turnover_NRP_point_in_time_QWIM(
            target_weights=target_weights,
            previous_end_weights=previous_end_weights,
        )
        (
            daily_month_rows,
            previous_end_weights,
            gross_portfolio_value,
            net_portfolio_value,
            gross_monthly_return,
            net_monthly_return,
        ) = _simulate_holding_month_NRP_point_in_time_QWIM(
            price_data=prepared_prices,
            eligible_assets=eligible_assets,
            effective_month=effective_month,
            target_weights=target_weights,
            turnover=turnover,
            transaction_cost_bps=config.transaction_cost_bps,
            gross_portfolio_value=gross_portfolio_value,
            net_portfolio_value=net_portfolio_value,
            strategy_name=f"NRP-{config.return_preprocessing}",
        )
        daily_rows.extend(daily_month_rows)
        monthly_rows.append(
            {
                "Month": effective_month,
                "EffectiveDate": effective_date,
                "EstimationStartMonth": estimation_months[0],
                "EstimationEndMonth": estimation_months[-1],
                "GrossReturn": gross_monthly_return,
                "TransactionCost": turnover * config.transaction_cost_bps / 10_000.0,
                "NetReturn": net_monthly_return,
                "NetValue": net_portfolio_value,
                "EligibleAssets": len(eligible_assets),
            },
        )
        weight_rows.extend(
            {
                "EffectiveDate": effective_date,
                "EstimationEndMonth": estimation_months[-1],
                "Asset": item_asset,
                "Weight": target_weights[item_asset],
            }
            for item_asset in eligible_assets
        )
        allocation_rows.extend(
            {
                "EffectiveDate": effective_date,
                "EstimationEndMonth": estimation_months[-1],
                "Asset": item_asset,
                "Weight": target_weights[item_asset],
                "Centrality": float(allocation.centrality_values[idx_asset]),
            }
            for idx_asset, item_asset in enumerate(eligible_assets)
        )
        network_frames.append(
            allocation.edge_data.with_columns(
                pl.lit(effective_date).alias("EffectiveDate"),
                pl.lit(estimation_months[-1]).alias("EstimationEndMonth"),
            ).select(
                "EffectiveDate",
                "EstimationEndMonth",
                *allocation.edge_data.columns,
            ),
        )
        rebalance_rows.append(
            {
                "EffectiveDate": effective_date,
                "EstimationStartMonth": estimation_months[0],
                "EstimationEndMonth": estimation_months[-1],
                "ActiveMembers": membership_audit.height,
                "EligibleAssets": len(eligible_assets),
                "ExcludedAssets": membership_audit.height - len(eligible_assets),
                "CentralityNormalization": config.centrality_normalization,
                "WeightNormalization": config.weight_normalization,
                "AdjacencyWeighting": config.adjacency_weighting,
                "ReturnPreprocessing": config.return_preprocessing,
                "AMADWindow": config.AMAD_window,
                "AMADThreshold": config.AMAD_threshold,
                "MaximumWeight": float(np.max(allocation.weights)),
                "MinimumWeight": float(np.min(allocation.weights)),
                "EffectiveHoldings": float(1.0 / np.sum(np.square(allocation.weights))),
                "MaximumMSTDegree": allocation.maximum_degree,
                "SpectralRadius": allocation.spectral_radius,
                "Turnover": turnover,
            },
        )

    daily_returns = pl.DataFrame(daily_rows).sort("Date")
    monthly_returns = pl.DataFrame(monthly_rows).sort("Month")
    rebalance_weights = pl.DataFrame(weight_rows).sort(["EffectiveDate", "Asset"])
    allocation_diagnostics = pl.DataFrame(allocation_rows).sort(["EffectiveDate", "Asset"])
    network_edges = pl.concat(network_frames, how="vertical").sort(
        ["EffectiveDate", "AssetLeft", "AssetRight"],
    )
    rebalance_diagnostics = pl.DataFrame(rebalance_rows).sort("EffectiveDate")
    membership_audit = pl.concat(membership_frames, how="vertical").sort(
        ["EffectiveDate", "Price_Ticker"],
    )
    metrics = _calculate_monthly_metrics_NRP_daily_QWIM(
        monthly_returns=monthly_returns,
        rebalance_diagnostics=rebalance_diagnostics,
        weight_normalization=config.weight_normalization,
        adjacency_weighting=config.adjacency_weighting,
        return_preprocessing=config.return_preprocessing,
    ).with_columns(
        pl.lit(
            "Point-in-time S&P 100 NRP "
            f"({config.return_preprocessing}, {config.adjacency_weighting}, "
            f"{config.weight_normalization})",
        ).alias("Strategy"),
    )
    strategy_name = metrics.get_column("Strategy").item(0)
    _LOGGER.info(
        "Completed point-in-time NRP: preprocessing=%s months=%s latest_assets=%s",
        config.return_preprocessing,
        monthly_returns.height,
        rebalance_diagnostics.get_column("EligibleAssets").tail(1).item(),
    )
    return NRP_Point_In_Time_Backtest_Result_QWIM(
        strategy_name=strategy_name,
        daily_returns=daily_returns,
        monthly_returns=monthly_returns,
        rebalance_weights=rebalance_weights,
        allocation_diagnostics=allocation_diagnostics,
        network_edges=network_edges,
        rebalance_diagnostics=rebalance_diagnostics,
        membership_audit=membership_audit,
        metrics=metrics,
    )


def _prepare_price_data_NRP_point_in_time_QWIM(
    *,
    price_data: pl.DataFrame,
    minimum_asset_count: int = 2,
) -> pl.DataFrame:
    """Validate and normalize a wide adjusted-price panel."""
    if not isinstance(price_data, pl.DataFrame) or "Date" not in price_data.columns:
        raise Exception_Data_Validation_Error(
            "Point-in-time prices require a Polars DataFrame with Date.",
            field="price_data",
            value=type(price_data).__name__,
        )
    asset_columns = [item_column for item_column in price_data.columns if item_column != "Date"]
    if len(asset_columns) < minimum_asset_count:
        raise Exception_Data_Validation_Error(
            "Point-in-time prices do not contain enough asset columns.",
            field="price_data",
            value=asset_columns,
        )
    prepared_prices = price_data.select(
        pl.col("Date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.strptime(pl.Date, strict=False)
        .alias("Date"),
        *[
            pl.col(item_asset).cast(pl.Float64, strict=False).fill_nan(None).alias(item_asset)
            for item_asset in asset_columns
        ],
    ).sort("Date")
    if prepared_prices.get_column("Date").null_count() > 0:
        raise Exception_Data_Validation_Error(
            "Point-in-time price dates cannot contain null values.",
            field="Date",
            value="contains_null",
        )
    if prepared_prices.get_column("Date").n_unique() != prepared_prices.height:
        raise Exception_Data_Validation_Error(
            "Point-in-time price dates must be unique.",
            field="Date",
            value="duplicate_dates",
        )
    return prepared_prices


def _prepare_membership_NRP_point_in_time_QWIM(
    *,
    membership_intervals: pl.DataFrame,
) -> pl.DataFrame:
    """Validate and normalize historical membership intervals."""
    if not isinstance(membership_intervals, pl.DataFrame):
        raise Exception_Data_Validation_Error(
            "Point-in-time membership must be a Polars DataFrame.",
            field="membership_intervals",
            value=type(membership_intervals).__name__,
        )
    missing_columns = _REQUIRED_MEMBERSHIP_COLUMNS - set(membership_intervals.columns)
    if missing_columns:
        raise Exception_Data_Validation_Error(
            "Point-in-time membership is missing required columns.",
            field="membership_intervals",
            value=sorted(missing_columns),
        )
    prepared_membership = membership_intervals.with_columns(
        pl.col("Membership_Start").cast(pl.Date),
        pl.col("Membership_End_Exclusive").cast(pl.Date),
        pl.col("Price_Ticker").cast(pl.String),
    ).sort(["Membership_Start", "Price_Ticker"])
    if (
        prepared_membership.select(
            pl.col("Membership_Start").is_null() | pl.col("Price_Ticker").is_null(),
        )
        .to_series()
        .any()
    ):
        raise Exception_Data_Validation_Error(
            "Membership starts and price tickers cannot be null.",
            field="membership_intervals",
            value="contains_null",
        )
    invalid_intervals = prepared_membership.filter(
        pl.col("Membership_End_Exclusive").is_not_null()
        & (pl.col("Membership_End_Exclusive") <= pl.col("Membership_Start")),
    )
    if invalid_intervals.height > 0:
        raise Exception_Data_Validation_Error(
            "Membership end must be later than membership start.",
            field="Membership_End_Exclusive",
            value=invalid_intervals.select(
                "Price_Ticker",
                "Membership_Start",
                "Membership_End_Exclusive",
            ).to_dicts(),
        )
    return prepared_membership


def _create_month_calendar_NRP_point_in_time_QWIM(
    *,
    price_data: pl.DataFrame,
) -> pl.DataFrame:
    """Create first/last trading dates for every observed calendar month."""
    return (
        price_data.select("Date")
        .with_columns(
            pl.col("Date").dt.truncate("1mo").cast(pl.Date).alias("Month"),
        )
        .group_by("Month")
        .agg(
            pl.col("Date").min().alias("EffectiveDate"),
            pl.col("Date").max().alias("HoldingEndDate"),
        )
        .sort("Month")
    )


def _select_eligible_members_NRP_point_in_time_QWIM(
    *,
    price_data: pl.DataFrame,
    membership_intervals: pl.DataFrame,
    effective_date: date,
    holding_end_date: date,
    estimation_months: tuple[date, ...],
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Select active members with complete estimation and holding prices."""
    active_members = membership_intervals.filter(
        (pl.col("Membership_Start") <= effective_date)
        & (
            pl.col("Membership_End_Exclusive").is_null()
            | (pl.col("Membership_End_Exclusive") > effective_date)
        ),
    )
    if active_members.get_column("Price_Ticker").n_unique() != active_members.height:
        raise Exception_Data_Validation_Error(
            "Active membership contains duplicate price tickers.",
            field="Price_Ticker",
            value=effective_date,
        )
    estimation_start_month = estimation_months[0]
    required_prices = price_data.filter(
        pl.col("Date").is_between(
            estimation_start_month,
            holding_end_date,
            closed="both",
        ),
    )
    required_observations = required_prices.height
    available_columns = set(price_data.columns) - {"Date"}
    audit_rows: list[dict[str, object]] = []
    eligible_assets: list[str] = []
    for item_member in active_members.iter_rows(named=True):
        ticker_symbol = str(item_member["Price_Ticker"])
        if ticker_symbol not in available_columns:
            observations_available = 0
            status_reason = "MissingPriceColumn"
            eligible = False
        else:
            ticker_prices = required_prices.get_column(ticker_symbol)
            valid_prices = ticker_prices.is_not_null() & (ticker_prices > 0.0)
            observations_available = int(valid_prices.sum())
            eligible = observations_available == required_observations
            status_reason = (
                "EligibleCompleteEstimationAndHoldingWindow"
                if eligible
                else "IncompleteEstimationOrHoldingWindow"
            )
        if eligible:
            eligible_assets.append(ticker_symbol)
        audit_rows.append(
            {
                "EffectiveDate": effective_date,
                "EstimationStartMonth": estimation_months[0],
                "EstimationEndMonth": estimation_months[-1],
                "HoldingEndDate": holding_end_date,
                "Price_Ticker": ticker_symbol,
                "Eligible": eligible,
                "EligibilityReason": status_reason,
                "RequiredPriceObservations": required_observations,
                "AvailablePriceObservations": observations_available,
            },
        )
    return (
        pl.DataFrame(audit_rows).sort(["EffectiveDate", "Price_Ticker"]),
        tuple(sorted(eligible_assets)),
    )


def _calculate_allocation_NRP_point_in_time_QWIM(
    *,
    price_data: pl.DataFrame,
    eligible_assets: tuple[str, ...],
    effective_month: date,
    estimation_months: tuple[date, ...],
    config: NRP_Daily_Approximation_Config_QWIM,
) -> NRP_Daily_Allocation_QWIM:
    """Calculate one NRP target without any holding-month information."""
    estimation_price_history = (
        price_data.filter(pl.col("Date") < effective_month)
        .select("Date", *eligible_assets)
        .drop_nulls()
    )
    raw_log_returns = calculate_daily_log_returns_NRP_daily_approximation_QWIM(
        price_data=estimation_price_history,
    )
    estimation_log_returns = (
        apply_AMAD_returns_NRP_QWIM(
            returns_data=raw_log_returns,
            config=NRP_AMAD_Config_QWIM(
                window_size=config.AMAD_window,
                threshold=config.AMAD_threshold,
            ),
        )
        if config.return_preprocessing == "AMAD"
        else raw_log_returns
    )
    estimation_window_returns = (
        estimation_log_returns.with_columns(
            pl.col("Date").dt.truncate("1mo").cast(pl.Date).alias("Month"),
        )
        .filter(pl.col("Month").is_in(estimation_months))
        .drop("Month")
    )
    monthly_covariances = calculate_monthly_realized_covariances_NRP_daily_approximation_QWIM(
        daily_log_returns=estimation_window_returns,
    )
    if monthly_covariances.month_starts != estimation_months:
        raise Exception_Data_Validation_Error(
            "Eligible assets did not produce the exact estimation-month sequence.",
            field="estimation_months",
            value={
                "expected": estimation_months,
                "received": monthly_covariances.month_starts,
            },
        )
    return calculate_NRP_daily_approximation_allocation_QWIM(
        asset_names=eligible_assets,
        realized_covariance_matrices=monthly_covariances.covariance_matrices,
        config=config,
    )


def _calculate_turnover_NRP_point_in_time_QWIM(
    *,
    target_weights: dict[str, float],
    previous_end_weights: dict[str, float],
) -> float:
    """Calculate half-L1 turnover across the union of changing universes."""
    union_assets = set(target_weights) | set(previous_end_weights)
    return float(
        0.5
        * sum(
            abs(
                target_weights.get(item_asset, 0.0) - previous_end_weights.get(item_asset, 0.0),
            )
            for item_asset in union_assets
        ),
    )


def _simulate_holding_month_NRP_point_in_time_QWIM(
    *,
    price_data: pl.DataFrame,
    eligible_assets: tuple[str, ...],
    effective_month: date,
    target_weights: dict[str, float],
    turnover: float,
    transaction_cost_bps: float,
    gross_portfolio_value: float,
    net_portfolio_value: float,
    strategy_name: str,
) -> tuple[
    list[dict[str, object]],
    dict[str, float],
    float,
    float,
    float,
    float,
]:
    """Simulate one holding month and drift weights through daily returns."""
    holding_prices = price_data.filter(
        pl.col("Date").dt.truncate("1mo").cast(pl.Date) == effective_month,
    ).select("Date", *eligible_assets)
    if holding_prices.is_empty():
        raise Exception_Data_Validation_Error(
            "Holding month has no trading observations.",
            field="effective_month",
            value=effective_month,
        )
    effective_date = holding_prices.get_column("Date").min()
    previous_date = (
        price_data.filter(
            pl.col("Date") < effective_date,
        )
        .get_column("Date")
        .max()
    )
    if previous_date is None:
        raise Exception_Data_Validation_Error(
            "Holding month requires one prior adjusted-price observation.",
            field="effective_date",
            value=effective_date,
        )
    holding_prices = price_data.filter(
        pl.col("Date").is_between(previous_date, holding_prices.get_column("Date").max()),
    ).select("Date", *eligible_assets)
    price_values = holding_prices.select(eligible_assets).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0.0):
        raise Exception_Data_Validation_Error(
            "Eligible holding prices must be finite and positive.",
            field="holding_prices",
            value=effective_month,
        )
    daily_asset_returns = price_values[1:] / price_values[:-1] - 1.0
    return_dates = holding_prices.get_column("Date").to_list()[1:]
    current_weights = np.asarray(
        [target_weights[item_asset] for item_asset in eligible_assets],
        dtype=np.float64,
    )
    transaction_cost = turnover * transaction_cost_bps / 10_000.0
    gross_value_start = gross_portfolio_value
    net_value_start = net_portfolio_value
    daily_rows: list[dict[str, object]] = []
    for idx_return, item_date in enumerate(return_dates):
        gross_return = float(current_weights @ daily_asset_returns[idx_return])
        daily_cost = transaction_cost if idx_return == 0 else 0.0
        net_return = (1.0 - daily_cost) * (1.0 + gross_return) - 1.0
        gross_portfolio_value *= 1.0 + gross_return
        net_portfolio_value *= 1.0 + net_return
        daily_rows.append(
            {
                "Date": item_date,
                "Month": effective_month,
                "Strategy": strategy_name,
                "GrossReturn": gross_return,
                "TransactionCost": daily_cost,
                "NetReturn": net_return,
                "NetValue": net_portfolio_value,
            },
        )
        current_weights = _drift_weights_NRP_daily_QWIM(
            weights=current_weights,
            asset_returns=daily_asset_returns[idx_return],
            portfolio_return=gross_return,
        )
    end_weights = {
        item_asset: float(current_weights[idx_asset])
        for idx_asset, item_asset in enumerate(eligible_assets)
    }
    return (
        daily_rows,
        end_weights,
        gross_portfolio_value,
        net_portfolio_value,
        gross_portfolio_value / gross_value_start - 1.0,
        net_portfolio_value / net_value_start - 1.0,
    )
