# ruff: noqa: N801, N999

"""Implement a daily-frequency Network Risk Parity research engine.

The network construction follows Ciciretti and Pallotta (2024). The current
research default uses their distance-weighted MST and a scale-invariant
proportional normalization of inverse centrality. The paper's literal softmax
and its Equation (2) unweighted interpretation remain available only as named
diagnostic comparators. The module excludes AMAD, Dantzig estimation, Ridge
regularisation, portfolio caps, and Dashboard concerns.

The paper builds monthly realised covariances from hourly log returns; the
local S&P 500 input is daily, so this implementation sums outer products of
daily log returns within each calendar month instead. All result labels make
the chosen conventions explicit.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import attrs
import numpy as np
import polars as pl

from pydantic import BaseModel, ConfigDict, Field

from .amad import (
    NRP_AMAD_Config_QWIM,
    apply_AMAD_returns_NRP_QWIM,
)
from .errors import (
    Exception_Calculation,
    Exception_Data_Validation_Error,
    Exception_Validation_Input,
)
from .errors import get_logger


_LOGGER = get_logger(name=__name__)
_EPSILON = 1.0e-12
_MONTHS_PER_YEAR = 12
Centrality_Normalization_NRP_Daily = Literal["l2"]
Weight_Normalization_NRP_Daily = Literal["paper_softmax", "inverse_proportional"]
Adjacency_Weighting_NRP_Daily = Literal["correlation_distance", "unweighted"]
Return_Preprocessing_NRP_Daily = Literal["raw", "AMAD"]


class NRP_Daily_Approximation_Config_Input_QWIM(BaseModel):
    """Validate external settings for a daily-frequency NRP backtest.

    Parameters
    ----------
    estimation_months : int, default=24
        Number of completed calendar months of realised covariance used at a
        rebalance.  The paper uses two years, hence the default of 24.
    transaction_cost_bps : float, default=0.0
        One-way trading cost used only for an optional implementation-cost
        sensitivity.  Zero reproduces the paper's no-cost reporting setup.
    centrality_normalization : {"l2"}, default="l2"
        Explicit numerical scale of the Perron eigenvector before the paper's
        allocation transformation. An L2 scale matches standard symmetric
        eigensolver output and makes the softmax baseline reproducible.
    weight_normalization : {"paper_softmax", "inverse_proportional"}, default="inverse_proportional"
        ``paper_softmax`` applies the paper's stated softmax to inverse
        centrality. ``inverse_proportional`` instead scales inverse
        centralities to sum to one. The latter is the scale-invariant current
        research default, not an assertion about the paper authors'
        implementation.
    adjacency_weighting : {"correlation_distance", "unweighted"}, default="correlation_distance"
        ``correlation_distance`` assigns each selected MST edge its paper
        distance ``1 - rho_ij**2``. ``unweighted`` selects the identical
        Kruskal MST but assigns every selected edge one, implementing the
        unweighted neighbor sum written in the paper's Equation (2).
    return_preprocessing : {"raw", "AMAD"}, default="raw"
        Estimation-return preprocessing. ``AMAD`` applies the cited trailing
        median/MAD transformation before monthly realised covariance; holding
        returns always remain raw.
    AMAD_window : int, default=63
        Trailing observations used for AMAD rolling statistics.
    AMAD_threshold : float, default=3.0
        Standardized absolute-deviation threshold where AMAD dampening begins.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    estimation_months: int = Field(default=24, ge=1)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0)
    centrality_normalization: Centrality_Normalization_NRP_Daily = "l2"
    weight_normalization: Weight_Normalization_NRP_Daily = "inverse_proportional"
    adjacency_weighting: Adjacency_Weighting_NRP_Daily = "correlation_distance"
    return_preprocessing: Return_Preprocessing_NRP_Daily = "raw"
    AMAD_window: int = Field(default=63, ge=2)
    AMAD_threshold: float = Field(default=3.0, gt=0.0)


@attrs.define(kw_only=True, frozen=True, slots=True)
class NRP_Daily_Approximation_Config_QWIM:
    """Store immutable settings for the daily-frequency NRP approximation.

    Parameters
    ----------
    estimation_months : int
        Number of prior monthly realised covariance matrices used at each
        rebalance.
    transaction_cost_bps : float
        One-way turnover cost in basis points.  Zero is the paper baseline.
    centrality_normalization : {"l2"}
        Explicit normalization applied to the non-negative Perron vector.
    weight_normalization : {"paper_softmax", "inverse_proportional"}
        Fully-invested transformation applied after inverse centrality.
    adjacency_weighting : {"correlation_distance", "unweighted"}
        Edge values in the adjacency matrix used for Perron centrality.
    return_preprocessing : {"raw", "AMAD"}
        Estimation-return preprocessing applied before realised covariance.
    AMAD_window : int
        Trailing observations used for AMAD median and MAD.
    AMAD_threshold : float
        Standardized deviation where logarithmic dampening begins.
    """

    estimation_months: int = attrs.field(default=24, converter=int)
    transaction_cost_bps: float = attrs.field(default=0.0, converter=float)
    centrality_normalization: Centrality_Normalization_NRP_Daily = attrs.field(default="l2")
    weight_normalization: Weight_Normalization_NRP_Daily = attrs.field(
        default="inverse_proportional",
    )
    adjacency_weighting: Adjacency_Weighting_NRP_Daily = attrs.field(default="correlation_distance")
    return_preprocessing: Return_Preprocessing_NRP_Daily = attrs.field(default="raw")
    AMAD_window: int = attrs.field(default=63, converter=int)
    AMAD_threshold: float = attrs.field(default=3.0, converter=float)

    def __attrs_post_init__(self) -> None:
        """Validate the immutable domain settings through the input boundary."""
        NRP_Daily_Approximation_Config_Input_QWIM(
            estimation_months=self.estimation_months,
            transaction_cost_bps=self.transaction_cost_bps,
            centrality_normalization=self.centrality_normalization,
            weight_normalization=self.weight_normalization,
            adjacency_weighting=self.adjacency_weighting,
            return_preprocessing=self.return_preprocessing,
            AMAD_window=self.AMAD_window,
            AMAD_threshold=self.AMAD_threshold,
        )


@attrs.define(kw_only=True, frozen=True, slots=True)
class Monthly_Realized_Covariances_NRP_Daily_QWIM:
    """Store calendar-month realised covariance matrices in a fixed asset order.

    Parameters
    ----------
    asset_names : tuple[str, ...]
        Ordered stock identifiers matching both axes of every covariance matrix.
    month_starts : tuple[datetime.date, ...]
        First calendar date of each month represented in ``covariance_matrices``.
    covariance_matrices : tuple[numpy.ndarray, ...]
        One positive-semidefinite realised covariance matrix for each month.
    """

    asset_names: tuple[str, ...]
    month_starts: tuple[date, ...]
    covariance_matrices: tuple[np.ndarray, ...]


@attrs.define(kw_only=True, frozen=True, slots=True)
class NRP_Daily_Allocation_QWIM:
    """Store one paper-formula NRP allocation and its network diagnostics.

    Parameters
    ----------
    asset_names : tuple[str, ...]
        Ordered stock identifiers for all allocation arrays.
    weights : numpy.ndarray
        Fully invested long-only NRP weights from the configured
        inverse-centrality transformation.
    centrality_values : numpy.ndarray
        L2-normalized Perron eigenvector of the configured MST adjacency matrix.
    correlation_matrix : numpy.ndarray
        Correlation matrix derived from the rolling realised covariance estimate.
    adjacency_matrix : numpy.ndarray
        Configured MST adjacency matrix used to calculate centrality.
    edge_data : polars.DataFrame
        MST edges with their correlation distances.
    spectral_radius : float
        Largest eigenvalue of the configured MST adjacency matrix.
    maximum_degree : int
        Largest unweighted degree in the MST.
    """

    asset_names: tuple[str, ...]
    weights: np.ndarray
    centrality_values: np.ndarray
    correlation_matrix: np.ndarray
    adjacency_matrix: np.ndarray
    edge_data: pl.DataFrame
    spectral_radius: float
    maximum_degree: int


@attrs.define(kw_only=True, frozen=True, slots=True)
class NRP_Daily_Approximation_Backtest_Result_QWIM:
    """Store reproducible outputs of a pure daily-frequency NRP backtest.

    Parameters
    ----------
    strategy_name : str
        Explicit label identifying the daily-frequency NRP approximation.
    monthly_returns : polars.DataFrame
        One realised portfolio-return row for every out-of-sample calendar month.
    rebalance_weights : polars.DataFrame
        Long-form target weights effective on each monthly rebalance date.
    allocation_diagnostics : polars.DataFrame
        Long-form centrality and weight diagnostics for each rebalance.
    network_edges : polars.DataFrame
        All MST edges selected at every rebalance.
    rebalance_diagnostics : polars.DataFrame
        Rebalance-level covariance window and concentration diagnostics.
    metrics : polars.DataFrame
        One-row monthly-backtest performance and concentration summary.
    """

    strategy_name: str
    monthly_returns: pl.DataFrame
    rebalance_weights: pl.DataFrame
    allocation_diagnostics: pl.DataFrame
    network_edges: pl.DataFrame
    rebalance_diagnostics: pl.DataFrame
    metrics: pl.DataFrame


def create_NRP_daily_approximation_config_QWIM(
    *,
    config_input: NRP_Daily_Approximation_Config_Input_QWIM,
) -> NRP_Daily_Approximation_Config_QWIM:
    """Create immutable daily-NRP settings from a validated external model.

    Parameters
    ----------
    config_input : NRP_Daily_Approximation_Config_Input_QWIM
        Validated user-facing backtest configuration.

    Returns
    -------
    NRP_Daily_Approximation_Config_QWIM
        Immutable configuration consumed by the pure calculation functions.
    """
    return NRP_Daily_Approximation_Config_QWIM(**config_input.model_dump())


def calculate_daily_log_returns_NRP_daily_approximation_QWIM(
    *,
    price_data: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate finite daily log returns from a complete close-price panel.

    Parameters
    ----------
    price_data : polars.DataFrame
        Strictly positive close prices with one ``Date`` column and at least two
        stock columns.  Every stock must have an observation for every date.

    Returns
    -------
    polars.DataFrame
        Date-indexed daily log returns in the input stock-column order.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when dates or price values are missing, duplicated, non-finite,
        non-positive, or otherwise unsuitable for realised covariance.

    Notes
    -----
    This is the intentional daily substitute for the paper's hourly log-return
    input.  NumPy is used only at the dense numerical boundary.
    """
    prepared_prices = _validate_close_price_data_NRP_daily_QWIM(price_data=price_data)
    asset_names = _get_asset_columns_NRP_daily_QWIM(data_frame=prepared_prices)
    price_values = prepared_prices.select(asset_names).to_numpy().astype(np.float64)
    log_return_values = np.log(price_values[1:] / price_values[:-1])
    if not np.all(np.isfinite(log_return_values)):
        raise Exception_Data_Validation_Error(
            "Daily log-return construction produced non-finite values.",
            field="price_data",
            value="non_finite_log_return",
        )
    return pl.DataFrame(
        {
            "Date": prepared_prices.get_column("Date").slice(1),
            **{
                item_asset: log_return_values[:, idx_asset]
                for idx_asset, item_asset in enumerate(asset_names)
            },
        },
    )


def calculate_monthly_realized_covariances_NRP_daily_approximation_QWIM(
    *,
    daily_log_returns: pl.DataFrame,
) -> Monthly_Realized_Covariances_NRP_Daily_QWIM:
    """Aggregate daily log-return cross-products into calendar-month matrices.

    Parameters
    ----------
    daily_log_returns : polars.DataFrame
        Finite daily log returns with a ``Date`` column and one column per stock.

    Returns
    -------
    Monthly_Realized_Covariances_NRP_Daily_QWIM
        Chronologically ordered monthly matrices where each matrix is
        ``sum(r_t r_t.T)`` over that calendar month's daily observations.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when dates or return values are missing, duplicated, or
        non-finite.

    Notes
    -----
    The aggregation preserves equation (1) of the paper, but substitutes daily
    observations for the unavailable hourly observations.
    """
    prepared_returns = _validate_return_data_NRP_daily_QWIM(return_data=daily_log_returns)
    asset_names = tuple(_get_asset_columns_NRP_daily_QWIM(data_frame=prepared_returns))
    return_values = prepared_returns.select(asset_names).to_numpy().astype(np.float64)
    return_dates = prepared_returns.get_column("Date").to_list()
    month_to_indices: dict[date, list[int]] = {}
    for idx_return, item_date in enumerate(return_dates):
        month_start = _get_month_start_NRP_daily_QWIM(item_date=item_date)
        month_to_indices.setdefault(month_start, []).append(idx_return)

    month_starts = tuple(sorted(month_to_indices))
    covariance_matrices = tuple(
        return_values[np.asarray(month_to_indices[item_month], dtype=np.intp)].T
        @ return_values[np.asarray(month_to_indices[item_month], dtype=np.intp)]
        for item_month in month_starts
    )
    return Monthly_Realized_Covariances_NRP_Daily_QWIM(
        asset_names=asset_names,
        month_starts=month_starts,
        covariance_matrices=covariance_matrices,
    )


def calculate_NRP_daily_approximation_allocation_QWIM(
    *,
    asset_names: tuple[str, ...],
    realized_covariance_matrices: tuple[np.ndarray, ...],
    config: NRP_Daily_Approximation_Config_QWIM,
) -> NRP_Daily_Allocation_QWIM:
    """Calculate one daily-approximation NRP allocation from realised covariances.

    Parameters
    ----------
    asset_names : tuple[str, ...]
        Ordered stock identifiers that match every covariance-matrix axis.
    realized_covariance_matrices : tuple[numpy.ndarray, ...]
        Consecutive completed monthly realised covariance matrices in the
        rolling estimation window.
    config : NRP_Daily_Approximation_Config_QWIM
        Validated NRP settings.  The method requires exactly
        ``config.estimation_months`` matrices.

    Returns
    -------
    NRP_Daily_Allocation_QWIM
        The configured-MST, L2 Perron-centrality allocation using the
        configured fully invested inverse-centrality transformation.

    Raises
    ------
    Exception_Validation_Input
        Raised when the number or dimensions of covariance matrices do not
        match the required stock universe.
    Exception_Calculation
        Raised if a zero-distance edge or an underflowed Perron component makes
        inverse-centrality allocation undefined.

    Notes
    -----
    The paper leaves the scale of the eigenvector centrality unspecified even
    though softmax is scale-sensitive.  This implementation fixes an L2 scale
    for the ``paper_softmax`` baseline.  The alternative
    ``inverse_proportional`` normalization cancels that arbitrary scale and is
    the current research default after the softmax concentration diagnostic.
    """
    _validate_realized_covariance_window_NRP_daily_QWIM(
        asset_names=asset_names,
        realized_covariance_matrices=realized_covariance_matrices,
        estimation_months=config.estimation_months,
    )
    covariance_matrix = np.sum(np.stack(realized_covariance_matrices, axis=0), axis=0)
    correlation_matrix = _calculate_correlation_NRP_daily_QWIM(covariance_matrix=covariance_matrix)
    distance_matrix = 1.0 - np.square(correlation_matrix)
    np.fill_diagonal(distance_matrix, 0.0)
    adjacency_matrix, edge_data = _calculate_MST_adjacency_NRP_daily_QWIM(
        asset_names=asset_names,
        distance_matrix=distance_matrix,
        correlation_matrix=correlation_matrix,
        adjacency_weighting=config.adjacency_weighting,
    )
    centrality_values, spectral_radius = _calculate_l2_perron_centrality_NRP_daily_QWIM(
        adjacency_matrix=adjacency_matrix,
    )
    weights = _calculate_inverse_centrality_weights_NRP_daily_QWIM(
        centrality_values=centrality_values,
        weight_normalization=config.weight_normalization,
    )
    maximum_degree = int(np.max(np.count_nonzero(adjacency_matrix, axis=1)))
    return NRP_Daily_Allocation_QWIM(
        asset_names=asset_names,
        weights=weights,
        centrality_values=centrality_values,
        correlation_matrix=correlation_matrix,
        adjacency_matrix=adjacency_matrix,
        edge_data=edge_data,
        spectral_radius=spectral_radius,
        maximum_degree=maximum_degree,
    )


def run_NRP_daily_approximation_backtest_QWIM(
    *,
    price_data: pl.DataFrame,
    config: NRP_Daily_Approximation_Config_QWIM | None = None,
) -> NRP_Daily_Approximation_Backtest_Result_QWIM:
    """Backtest pure NRP with daily realised-covariance approximation.

    Parameters
    ----------
    price_data : polars.DataFrame
        Complete daily close-price panel for one fixed stock universe.  The
        current project dataset intentionally uses current S&P 500 membership.
    config : NRP_Daily_Approximation_Config_QWIM | None, default=None
        NRP settings.  When omitted, this uses the paper's 24-month rolling
        window, monthly rebalancing, L2 centrality convention, and zero costs.

    Returns
    -------
    NRP_Daily_Approximation_Backtest_Result_QWIM
        Monthly out-of-sample returns, target weights, network diagnostics, and
        annualized performance metrics.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when the close-price input is incomplete or malformed.
    Exception_Validation_Input
        Raised when the input lacks more completed months than the requested
        estimation window.
    Exception_Calculation
        Raised when the literal paper allocation is numerically undefined.

    Notes
    -----
    Weights use only completed months before their effective month.  They are
    set on the first trading day of the next calendar month and drift through
    daily simple returns until the next monthly rebalance.  This avoids
    look-ahead bias while preserving the paper's monthly holding convention.
    """
    effective_config = config if config is not None else NRP_Daily_Approximation_Config_QWIM()
    prepared_prices = _validate_close_price_data_NRP_daily_QWIM(price_data=price_data)
    asset_names = tuple(_get_asset_columns_NRP_daily_QWIM(data_frame=prepared_prices))
    raw_daily_log_returns = calculate_daily_log_returns_NRP_daily_approximation_QWIM(
        price_data=prepared_prices,
    )
    estimation_daily_log_returns = (
        apply_AMAD_returns_NRP_QWIM(
            returns_data=raw_daily_log_returns,
            config=NRP_AMAD_Config_QWIM(
                window_size=effective_config.AMAD_window,
                threshold=effective_config.AMAD_threshold,
            ),
        )
        if effective_config.return_preprocessing == "AMAD"
        else raw_daily_log_returns
    )
    monthly_covariances = calculate_monthly_realized_covariances_NRP_daily_approximation_QWIM(
        daily_log_returns=estimation_daily_log_returns,
    )
    if len(monthly_covariances.month_starts) <= effective_config.estimation_months:
        raise Exception_Validation_Input(
            "NRP requires more completed months than its estimation window.",
            field_name="estimation_months",
            actual_value={
                "available_months": len(monthly_covariances.month_starts),
                "required_months": effective_config.estimation_months + 1,
            },
        )

    daily_simple_returns, return_dates = _calculate_simple_returns_NRP_daily_QWIM(
        prepared_prices=prepared_prices,
        asset_names=asset_names,
    )
    month_to_return_indices = _get_month_to_return_indices_NRP_daily_QWIM(return_dates=return_dates)
    monthly_return_rows: list[dict[str, object]] = []
    rebalance_weight_rows: list[dict[str, object]] = []
    allocation_diagnostic_rows: list[dict[str, object]] = []
    network_edge_frames: list[pl.DataFrame] = []
    rebalance_diagnostic_rows: list[dict[str, object]] = []
    previous_end_weights = np.zeros(len(asset_names), dtype=np.float64)
    gross_portfolio_value = 1.0
    portfolio_value = 1.0
    cost_rate = effective_config.transaction_cost_bps / 10_000.0

    for idx_month in range(
        effective_config.estimation_months,
        len(monthly_covariances.month_starts),
    ):
        effective_month = monthly_covariances.month_starts[idx_month]
        estimation_months = monthly_covariances.month_starts[
            idx_month - effective_config.estimation_months : idx_month
        ]
        allocation = calculate_NRP_daily_approximation_allocation_QWIM(
            asset_names=asset_names,
            realized_covariance_matrices=monthly_covariances.covariance_matrices[
                idx_month - effective_config.estimation_months : idx_month
            ],
            config=effective_config,
        )
        return_indices = month_to_return_indices[effective_month]
        effective_date = return_dates[return_indices[0]]
        turnover = float(0.5 * np.sum(np.abs(allocation.weights - previous_end_weights)))
        transaction_cost = turnover * cost_rate
        gross_value_start = gross_portfolio_value
        net_value_start = portfolio_value
        current_weights = allocation.weights.copy()
        for idx_return in return_indices:
            asset_returns = daily_simple_returns[idx_return]
            gross_return = float(current_weights @ asset_returns)
            daily_cost = transaction_cost if idx_return == return_indices[0] else 0.0
            net_return = (1.0 - daily_cost) * (1.0 + gross_return) - 1.0
            gross_portfolio_value *= 1.0 + gross_return
            portfolio_value *= 1.0 + net_return
            current_weights = _drift_weights_NRP_daily_QWIM(
                weights=current_weights,
                asset_returns=asset_returns,
                portfolio_return=gross_return,
            )

        gross_monthly_return = gross_portfolio_value / gross_value_start - 1.0
        net_monthly_return = portfolio_value / net_value_start - 1.0
        monthly_return_rows.append(
            {
                "Month": effective_month,
                "EffectiveDate": effective_date,
                "EstimationStartMonth": estimation_months[0],
                "EstimationEndMonth": estimation_months[-1],
                "GrossReturn": gross_monthly_return,
                "TransactionCost": transaction_cost,
                "NetReturn": net_monthly_return,
                "NetValue": portfolio_value,
            },
        )
        rebalance_weight_rows.extend(
            {
                "EffectiveDate": effective_date,
                "EstimationEndMonth": estimation_months[-1],
                "Asset": item_asset,
                "Weight": float(allocation.weights[idx_asset]),
            }
            for idx_asset, item_asset in enumerate(asset_names)
        )
        allocation_diagnostic_rows.extend(
            {
                "EffectiveDate": effective_date,
                "EstimationEndMonth": estimation_months[-1],
                "Asset": item_asset,
                "Weight": float(allocation.weights[idx_asset]),
                "Centrality": float(allocation.centrality_values[idx_asset]),
            }
            for idx_asset, item_asset in enumerate(asset_names)
        )
        network_edge_frames.append(
            allocation.edge_data.with_columns(
                pl.lit(effective_date).alias("EffectiveDate"),
                pl.lit(estimation_months[-1]).alias("EstimationEndMonth"),
            ).select(["EffectiveDate", "EstimationEndMonth", *allocation.edge_data.columns]),
        )
        rebalance_diagnostic_rows.append(
            {
                "EffectiveDate": effective_date,
                "EstimationStartMonth": estimation_months[0],
                "EstimationEndMonth": estimation_months[-1],
                "CentralityNormalization": effective_config.centrality_normalization,
                "WeightNormalization": effective_config.weight_normalization,
                "AdjacencyWeighting": effective_config.adjacency_weighting,
                "ReturnPreprocessing": effective_config.return_preprocessing,
                "AMADWindow": effective_config.AMAD_window,
                "AMADThreshold": effective_config.AMAD_threshold,
                "MaximumWeight": float(np.max(allocation.weights)),
                "MinimumWeight": float(np.min(allocation.weights)),
                "EffectiveHoldings": float(1.0 / np.sum(np.square(allocation.weights))),
                "MaximumMSTDegree": allocation.maximum_degree,
                "SpectralRadius": allocation.spectral_radius,
                "Turnover": turnover,
            },
        )
        previous_end_weights = current_weights

    monthly_returns = pl.DataFrame(monthly_return_rows)
    rebalance_weights = pl.DataFrame(rebalance_weight_rows)
    allocation_diagnostics = pl.DataFrame(allocation_diagnostic_rows)
    network_edges = pl.concat(network_edge_frames, how="vertical")
    rebalance_diagnostics = pl.DataFrame(rebalance_diagnostic_rows)
    metrics = _calculate_monthly_metrics_NRP_daily_QWIM(
        monthly_returns=monthly_returns,
        rebalance_diagnostics=rebalance_diagnostics,
        weight_normalization=effective_config.weight_normalization,
        adjacency_weighting=effective_config.adjacency_weighting,
        return_preprocessing=effective_config.return_preprocessing,
    )
    _LOGGER.info(
        "Completed daily-frequency NRP approximation: assets=%s rebalances=%s",
        len(asset_names),
        len(rebalance_diagnostic_rows),
    )
    return NRP_Daily_Approximation_Backtest_Result_QWIM(
        strategy_name=(
            "NRP daily-frequency approximation "
            f"({effective_config.return_preprocessing}, "
            f"{effective_config.adjacency_weighting}, {effective_config.weight_normalization})"
        ),
        monthly_returns=monthly_returns,
        rebalance_weights=rebalance_weights,
        allocation_diagnostics=allocation_diagnostics,
        network_edges=network_edges,
        rebalance_diagnostics=rebalance_diagnostics,
        metrics=metrics,
    )


def _get_asset_columns_NRP_daily_QWIM(*, data_frame: pl.DataFrame) -> list[str]:
    """Return non-date asset columns from a validated NRP data frame."""
    if "Date" not in data_frame.columns:
        raise Exception_Data_Validation_Error(
            "NRP data must include a Date column.",
            field="Date",
            value=data_frame.columns,
        )
    asset_names = [item_column for item_column in data_frame.columns if item_column != "Date"]
    if len(asset_names) < 2:
        raise Exception_Data_Validation_Error(
            "NRP requires at least two asset columns.",
            field="asset_columns",
            value=asset_names,
        )
    return asset_names


def _validate_close_price_data_NRP_daily_QWIM(*, price_data: pl.DataFrame) -> pl.DataFrame:
    """Validate and chronologically normalize a complete daily close-price panel."""
    if not isinstance(price_data, pl.DataFrame):
        raise Exception_Data_Validation_Error(
            "NRP close prices must be supplied as a Polars DataFrame.",
            field="price_data",
            value=type(price_data).__name__,
        )
    asset_names = _get_asset_columns_NRP_daily_QWIM(data_frame=price_data)
    prepared_prices = (
        price_data.select(
            [
                pl.col("Date")
                .cast(pl.Utf8)
                .str.slice(0, 10)
                .str.strptime(pl.Date, strict=False)
                .alias("Date"),
                *[
                    pl.col(item_asset).cast(pl.Float64, strict=False).alias(item_asset)
                    for item_asset in asset_names
                ],
            ],
        )
        .sort("Date")
        .with_columns([pl.col(item_asset).fill_nan(None) for item_asset in asset_names])
    )
    if prepared_prices.height < 3:
        raise Exception_Data_Validation_Error(
            "NRP needs at least three close-price observations.",
            field="price_data",
            value=prepared_prices.height,
        )
    if prepared_prices.null_count().sum_horizontal().sum() > 0:
        raise Exception_Data_Validation_Error(
            "NRP close-price data must be complete and non-null.",
            field="price_data",
            value="contains_null",
        )
    if prepared_prices.get_column("Date").n_unique() != prepared_prices.height:
        raise Exception_Data_Validation_Error(
            "NRP close-price dates must be unique.",
            field="Date",
            value="duplicate_dates",
        )
    price_values = prepared_prices.select(asset_names).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(price_values)) or np.any(price_values <= 0.0):
        raise Exception_Data_Validation_Error(
            "NRP close prices must be finite and strictly positive.",
            field="price_data",
            value="non_finite_or_non_positive",
        )
    return prepared_prices


def _validate_return_data_NRP_daily_QWIM(*, return_data: pl.DataFrame) -> pl.DataFrame:
    """Validate and chronologically normalize a daily log-return panel."""
    if not isinstance(return_data, pl.DataFrame):
        raise Exception_Data_Validation_Error(
            "NRP returns must be supplied as a Polars DataFrame.",
            field="daily_log_returns",
            value=type(return_data).__name__,
        )
    asset_names = _get_asset_columns_NRP_daily_QWIM(data_frame=return_data)
    prepared_returns = (
        return_data.select(
            [
                pl.col("Date")
                .cast(pl.Utf8)
                .str.slice(0, 10)
                .str.strptime(pl.Date, strict=False)
                .alias("Date"),
                *[
                    pl.col(item_asset).cast(pl.Float64, strict=False).alias(item_asset)
                    for item_asset in asset_names
                ],
            ],
        )
        .sort("Date")
        .with_columns([pl.col(item_asset).fill_nan(None) for item_asset in asset_names])
    )
    if prepared_returns.null_count().sum_horizontal().sum() > 0:
        raise Exception_Data_Validation_Error(
            "NRP log returns must be complete and non-null.",
            field="daily_log_returns",
            value="contains_null",
        )
    if prepared_returns.get_column("Date").n_unique() != prepared_returns.height:
        raise Exception_Data_Validation_Error(
            "NRP log-return dates must be unique.",
            field="Date",
            value="duplicate_dates",
        )
    return_values = prepared_returns.select(asset_names).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(return_values)):
        raise Exception_Data_Validation_Error(
            "NRP log returns must be finite.",
            field="daily_log_returns",
            value="non_finite_return",
        )
    return prepared_returns


def _get_month_start_NRP_daily_QWIM(*, item_date: date) -> date:
    """Return the first calendar date of the month containing one date."""
    return date(year=item_date.year, month=item_date.month, day=1)


def _validate_realized_covariance_window_NRP_daily_QWIM(
    *,
    asset_names: tuple[str, ...],
    realized_covariance_matrices: tuple[np.ndarray, ...],
    estimation_months: int,
) -> None:
    """Validate dimensions and finiteness of a fixed realised-covariance window."""
    if len(realized_covariance_matrices) != estimation_months:
        raise Exception_Validation_Input(
            "NRP allocation needs exactly estimation_months realised covariance matrices.",
            field_name="realized_covariance_matrices",
            actual_value={
                "received": len(realized_covariance_matrices),
                "required": estimation_months,
            },
        )
    expected_shape = (len(asset_names), len(asset_names))
    if len(asset_names) < 2:
        raise Exception_Validation_Input(
            "NRP allocation requires at least two assets.",
            field_name="asset_names",
            actual_value=len(asset_names),
        )
    for idx_matrix, item_matrix in enumerate(realized_covariance_matrices):
        matrix_values = np.asarray(item_matrix, dtype=np.float64)
        if matrix_values.shape != expected_shape or not np.all(np.isfinite(matrix_values)):
            raise Exception_Validation_Input(
                "Each NRP realised covariance matrix must be finite and match asset_names.",
                field_name="realized_covariance_matrices",
                actual_value={
                    "index": idx_matrix,
                    "shape": matrix_values.shape,
                    "expected": expected_shape,
                },
            )


def _calculate_correlation_NRP_daily_QWIM(*, covariance_matrix: np.ndarray) -> np.ndarray:
    """Convert a finite realised covariance matrix to a symmetric correlation matrix."""
    symmetric_covariance = (covariance_matrix + covariance_matrix.T) / 2.0
    variances = np.diag(symmetric_covariance)
    if np.any(variances <= _EPSILON):
        raise Exception_Calculation("NRP realised covariance has a non-positive asset variance.")
    standard_deviations = np.sqrt(variances)
    correlation_matrix = symmetric_covariance / np.outer(standard_deviations, standard_deviations)
    correlation_matrix = np.clip((correlation_matrix + correlation_matrix.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix


def _calculate_MST_adjacency_NRP_daily_QWIM(
    *,
    asset_names: tuple[str, ...],
    distance_matrix: np.ndarray,
    correlation_matrix: np.ndarray,
    adjacency_weighting: Adjacency_Weighting_NRP_Daily,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Build a Kruskal MST and its configured adjacency matrix.

    Edge selection always uses the paper's correlation distance. The two
    options differ only in the values recorded in the adjacency matrix for
    Perron centrality, making the paper's weighted-graph description and its
    unweighted Equation (2) directly comparable.
    """
    n_assets = len(asset_names)
    candidate_edges = [
        (float(distance_matrix[idx_left, idx_right]), idx_left, idx_right)
        for idx_left in range(n_assets)
        for idx_right in range(idx_left + 1, n_assets)
    ]
    candidate_edges.sort(key=lambda item_edge: (item_edge[0], item_edge[1], item_edge[2]))
    parent_nodes = list(range(n_assets))

    def find_root(node_index: int) -> int:
        """Return the union-find root of one temporary Kruskal node."""
        while parent_nodes[node_index] != node_index:
            parent_nodes[node_index] = parent_nodes[parent_nodes[node_index]]
            node_index = parent_nodes[node_index]
        return node_index

    adjacency_matrix = np.zeros((n_assets, n_assets), dtype=np.float64)
    edge_rows: list[dict[str, object]] = []
    for distance_value, idx_left, idx_right in candidate_edges:
        root_left = find_root(idx_left)
        root_right = find_root(idx_right)
        if root_left == root_right:
            continue
        if distance_value <= _EPSILON and adjacency_weighting == "correlation_distance":
            raise Exception_Calculation(
                "NRP MST has a zero-distance edge, so weighted centrality is undefined.",
            )
        parent_nodes[root_right] = root_left
        adjacency_value = distance_value if adjacency_weighting == "correlation_distance" else 1.0
        adjacency_matrix[idx_left, idx_right] = adjacency_value
        adjacency_matrix[idx_right, idx_left] = adjacency_value
        edge_rows.append(
            {
                "AssetLeft": asset_names[idx_left],
                "AssetRight": asset_names[idx_right],
                "Distance": distance_value,
                "Correlation": float(correlation_matrix[idx_left, idx_right]),
                "AdjacencyWeight": adjacency_value,
            },
        )
        if len(edge_rows) == n_assets - 1:
            break
    if len(edge_rows) != n_assets - 1:
        raise Exception_Calculation("Kruskal construction did not produce a connected NRP tree.")
    return adjacency_matrix, pl.DataFrame(edge_rows)


def _calculate_l2_perron_centrality_NRP_daily_QWIM(
    *,
    adjacency_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculate a strictly positive L2-normalized Perron eigenvector."""
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    maximum_index = int(np.argmax(eigenvalues))
    spectral_radius = float(eigenvalues[maximum_index])
    centrality_values = eigenvectors[:, maximum_index]
    if float(np.sum(centrality_values)) < 0.0:
        centrality_values = -centrality_values
    centrality_values = np.abs(centrality_values)
    centrality_norm = float(np.linalg.norm(centrality_values))
    if spectral_radius <= _EPSILON or centrality_norm <= _EPSILON:
        raise Exception_Calculation(
            "NRP centrality calculation returned a degenerate Perron eigenvector.",
        )
    centrality_values = centrality_values / centrality_norm
    if not np.all(np.isfinite(centrality_values)) or np.any(centrality_values <= 0.0):
        raise Exception_Calculation(
            "NRP centrality contains numerical zeros; inverse-centrality allocation is undefined.",
        )
    return centrality_values, spectral_radius


def _calculate_inverse_centrality_weights_NRP_daily_QWIM(
    *,
    centrality_values: np.ndarray,
    weight_normalization: Weight_Normalization_NRP_Daily,
) -> np.ndarray:
    """Transform positive inverse Perron centralities into portfolio weights.

    ``paper_softmax`` retains the paper's stated exponential normalization.
    ``inverse_proportional`` uses ``(1 / xi_i) / sum_j(1 / xi_j)``; unlike a
    softmax, multiplying an eigenvector by any positive constant does not
    affect those weights.
    """
    inverse_centrality = 1.0 / centrality_values
    if weight_normalization == "paper_softmax":
        shifted_scores = inverse_centrality - np.max(inverse_centrality)
        exponentiated_scores = np.exp(shifted_scores)
        score_total = float(np.sum(exponentiated_scores))
        if not np.isfinite(score_total) or score_total <= _EPSILON:
            raise Exception_Calculation(
                "NRP softmax normalization failed to produce positive finite weights.",
            )
        return exponentiated_scores / score_total
    if weight_normalization == "inverse_proportional":
        inverse_total = float(np.sum(inverse_centrality))
        if not np.isfinite(inverse_total) or inverse_total <= _EPSILON:
            raise Exception_Calculation(
                "NRP proportional inverse-centrality normalization failed to produce finite weights.",
            )
        return inverse_centrality / inverse_total
    raise Exception_Validation_Input(
        "weight_normalization must be paper_softmax or inverse_proportional.",
        field_name="weight_normalization",
        actual_value=weight_normalization,
    )


def _calculate_simple_returns_NRP_daily_QWIM(
    *,
    prepared_prices: pl.DataFrame,
    asset_names: tuple[str, ...],
) -> tuple[np.ndarray, list[date]]:
    """Calculate daily simple returns for holding-period simulation."""
    price_values = prepared_prices.select(asset_names).to_numpy().astype(np.float64)
    simple_returns = price_values[1:] / price_values[:-1] - 1.0
    return_dates = prepared_prices.get_column("Date").to_list()[1:]
    return simple_returns, return_dates


def _get_month_to_return_indices_NRP_daily_QWIM(
    *,
    return_dates: list[date],
) -> dict[date, list[int]]:
    """Map each calendar month to its ordered daily-return row indexes."""
    month_to_indices: dict[date, list[int]] = {}
    for idx_return, item_date in enumerate(return_dates):
        month_to_indices.setdefault(
            _get_month_start_NRP_daily_QWIM(item_date=item_date),
            [],
        ).append(idx_return)
    return month_to_indices


def _drift_weights_NRP_daily_QWIM(
    *,
    weights: np.ndarray,
    asset_returns: np.ndarray,
    portfolio_return: float,
) -> np.ndarray:
    """Drift fully invested long-only holdings through one daily return vector."""
    denominator = 1.0 + portfolio_return
    if denominator <= _EPSILON:
        raise Exception_Calculation(
            "NRP portfolio value became non-positive during the holding period.",
        )
    drifted_weights = weights * (1.0 + asset_returns) / denominator
    weight_total = float(np.sum(drifted_weights))
    if weight_total <= _EPSILON or not np.all(np.isfinite(drifted_weights)):
        raise Exception_Calculation("NRP holding-period weights could not be normalized.")
    return drifted_weights / weight_total


def _calculate_monthly_metrics_NRP_daily_QWIM(
    *,
    monthly_returns: pl.DataFrame,
    rebalance_diagnostics: pl.DataFrame,
    weight_normalization: Weight_Normalization_NRP_Daily,
    adjacency_weighting: Adjacency_Weighting_NRP_Daily,
    return_preprocessing: Return_Preprocessing_NRP_Daily,
) -> pl.DataFrame:
    """Calculate annualized performance and concentration metrics from monthly returns."""
    return_values = monthly_returns.get_column("NetReturn").to_numpy().astype(np.float64)
    if return_values.size < 2:
        raise Exception_Calculation(
            "NRP backtest needs at least two out-of-sample months for metrics.",
        )
    annualized_return = float(
        np.prod(1.0 + return_values) ** (_MONTHS_PER_YEAR / return_values.size) - 1.0,
    )
    annualized_volatility = float(np.std(return_values, ddof=1) * np.sqrt(_MONTHS_PER_YEAR))
    sharpe_ratio = (
        annualized_return / annualized_volatility if annualized_volatility > _EPSILON else 0.0
    )
    wealth_index = np.cumprod(1.0 + return_values)
    maximum_drawdown = float(np.min(wealth_index / np.maximum.accumulate(wealth_index) - 1.0))
    return pl.DataFrame(
        [
            {
                "Strategy": (
                    "NRP daily-frequency approximation "
                    f"({return_preprocessing}, {adjacency_weighting}, {weight_normalization})"
                ),
                "ReturnPreprocessing": return_preprocessing,
                "OutOfSampleMonths": int(return_values.size),
                "AnnualizedReturn": annualized_return,
                "AnnualizedVolatility": annualized_volatility,
                "SharpeRatio": sharpe_ratio,
                "MaximumDrawdown": maximum_drawdown,
                "AverageMaximumWeight": float(
                    rebalance_diagnostics.get_column("MaximumWeight").mean(),
                ),
                "WorstMaximumWeight": float(
                    rebalance_diagnostics.get_column("MaximumWeight").max(),
                ),
                "AverageEffectiveHoldings": float(
                    rebalance_diagnostics.get_column("EffectiveHoldings").mean(),
                ),
                "AverageTurnover": float(rebalance_diagnostics.get_column("Turnover").mean()),
            },
        ],
    )
