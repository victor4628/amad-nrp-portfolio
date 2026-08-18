# ruff: noqa: N801, N999

"""Apply AMAD to NRP estimation returns without changing holding returns.

The transformation follows Pallotta and Ciciretti (2025): observations within
``threshold`` rolling median absolute deviations remain unchanged, while more
extreme observations receive logarithmic dampening.  Every rolling statistic
uses data ending at the observation being transformed, so the function does
not introduce future information.
"""

from __future__ import annotations

import attrs
import numpy as np
import polars as pl

from pydantic import BaseModel, ConfigDict, Field

from .errors import Exception_Data_Validation_Error


_EPSILON = 1.0e-12


class NRP_AMAD_Config_Input_QWIM(BaseModel):
    """Validate external AMAD transformation settings.

    Parameters
    ----------
    window_size : int, default=63
        Trailing observations used for the rolling median and MAD.
    threshold : float, default=3.0
        Standardized absolute-deviation level where dampening begins.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)

    window_size: int = Field(default=63, ge=2)
    threshold: float = Field(default=3.0, gt=0.0)


@attrs.define(kw_only=True, frozen=True, slots=True)
class NRP_AMAD_Config_QWIM:
    """Store immutable AMAD settings for NRP estimation returns."""

    window_size: int = attrs.field(default=63, converter=int)
    threshold: float = attrs.field(default=3.0, converter=float)

    def __attrs_post_init__(self) -> None:
        """Validate immutable settings through the external boundary model."""
        NRP_AMAD_Config_Input_QWIM(
            window_size=self.window_size,
            threshold=self.threshold,
        )


def create_NRP_AMAD_config_QWIM(
    *,
    config_input: NRP_AMAD_Config_Input_QWIM,
) -> NRP_AMAD_Config_QWIM:
    """Create immutable AMAD settings from a validated input model."""
    return NRP_AMAD_Config_QWIM(**config_input.model_dump())


def apply_AMAD_returns_NRP_QWIM(
    *,
    returns_data: pl.DataFrame,
    config: NRP_AMAD_Config_QWIM | None = None,
) -> pl.DataFrame:
    """Apply AMAD independently to each asset's estimation-return history.

    Parameters
    ----------
    returns_data : polars.DataFrame
        Date-indexed finite returns.  These can be log returns because AMAD is
        applied only to the NRP estimation path.
    config : NRP_AMAD_Config_QWIM | None, default=None
        AMAD window and threshold.  Defaults reproduce the paper's 63-day,
        threshold-three specification.

    Returns
    -------
    polars.DataFrame
        A frame with identical dates, columns, and row count.  The first
        ``window_size - 1`` observations are unchanged.

    Notes
    -----
    For one asset and observation, define ``z = abs(r - median) / MAD``.
    Values with ``z <= threshold`` remain unchanged.  Otherwise AMAD uses
    ``median + sign(r-median) * MAD * (threshold + log(z-(threshold-1)))``.
    A zero rolling MAD is left unchanged because the paper's standardized
    deviation is undefined in that case.
    """
    effective_config = config if config is not None else NRP_AMAD_Config_QWIM()
    prepared_returns, asset_columns = _validate_AMAD_returns_NRP_QWIM(
        returns_data=returns_data,
    )
    if prepared_returns.height < effective_config.window_size:
        return prepared_returns

    return_values = prepared_returns.select(asset_columns).to_numpy().astype(np.float64)
    transformed_values = return_values.copy()
    for idx_asset in range(return_values.shape[1]):
        transformed_values[:, idx_asset] = _apply_AMAD_series_NRP_QWIM(
            return_values=return_values[:, idx_asset],
            window_size=effective_config.window_size,
            threshold=effective_config.threshold,
        )
    return prepared_returns.with_columns(
        [
            pl.Series(
                name=item_asset,
                values=transformed_values[:, idx_asset],
                dtype=pl.Float64,
            )
            for idx_asset, item_asset in enumerate(asset_columns)
        ],
    )


def _apply_AMAD_series_NRP_QWIM(
    *,
    return_values: np.ndarray,
    window_size: int,
    threshold: float,
) -> np.ndarray:
    """Apply the paper's continuous logarithmic dampening to one return series."""
    transformed_values = np.asarray(return_values, dtype=np.float64).copy()
    value_windows = np.lib.stride_tricks.sliding_window_view(
        transformed_values,
        window_shape=window_size,
    )
    median_values = np.median(value_windows, axis=1)
    mad_values = np.median(
        np.abs(value_windows - median_values[:, np.newaxis]),
        axis=1,
    )
    current_values = transformed_values[window_size - 1 :]
    deviation_values = current_values - median_values
    absolute_deviations = np.abs(deviation_values)
    valid_scale = mad_values > _EPSILON
    anomaly_mask = valid_scale & (absolute_deviations > threshold * mad_values)
    standardized_deviations = np.divide(
        absolute_deviations,
        mad_values,
        out=np.ones_like(absolute_deviations),
        where=valid_scale,
    )
    logarithm_arguments = np.maximum(
        standardized_deviations - (threshold - 1.0),
        1.0,
    )
    damped_values = median_values + np.sign(deviation_values) * mad_values * (
        threshold + np.log(logarithm_arguments)
    )
    transformed_values[window_size - 1 :] = np.where(
        anomaly_mask,
        damped_values,
        current_values,
    )
    return transformed_values


def _validate_AMAD_returns_NRP_QWIM(
    *,
    returns_data: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Validate and chronologically normalize the AMAD return boundary."""
    if not isinstance(returns_data, pl.DataFrame):
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns must be supplied as a Polars DataFrame.",
            field="returns_data",
            value=type(returns_data).__name__,
        )
    if "Date" not in returns_data.columns:
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns must contain a Date column.",
            field="Date",
            value=returns_data.columns,
        )
    asset_columns = tuple(
        item_column for item_column in returns_data.columns if item_column != "Date"
    )
    if not asset_columns:
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns require at least one asset column.",
            field="asset_columns",
            value=asset_columns,
        )
    prepared_returns = returns_data.select(
        [
            pl.col("Date")
            .cast(pl.Utf8)
            .str.slice(0, 10)
            .str.strptime(pl.Date, strict=False)
            .alias("Date"),
            *[
                pl.col(item_asset).cast(pl.Float64, strict=False).fill_nan(None).alias(item_asset)
                for item_asset in asset_columns
            ],
        ],
    ).sort("Date")
    if prepared_returns.height < 2:
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns require at least two observations.",
            field="returns_data",
            value=prepared_returns.height,
        )
    if prepared_returns.null_count().sum_horizontal().sum() > 0:
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns and dates must be complete.",
            field="returns_data",
            value="contains_null",
        )
    if prepared_returns.get_column("Date").n_unique() != prepared_returns.height:
        raise Exception_Data_Validation_Error(
            "NRP AMAD dates must be unique.",
            field="Date",
            value="duplicate_dates",
        )
    return_values = prepared_returns.select(asset_columns).to_numpy().astype(np.float64)
    if not np.all(np.isfinite(return_values)):
        raise Exception_Data_Validation_Error(
            "NRP AMAD returns must be finite.",
            field="returns_data",
            value="non_finite_return",
        )
    return prepared_returns, asset_columns
