# ruff: noqa: N803, N999

"""Run survivorship-corrected S&P 100 NRP research backtests.

The script loads the locally reconstructed point-in-time membership intervals,
joins the supplemental adjusted-price histories, and runs AMAD-NRP and Raw NRP.
Their results are compared with an adjusted-price S&P 100 benchmark. All
research artifacts, including daily net values, monthly
paths, full rebalance weights, and the historical net-value chart, are written
to one immutable run directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from amad_nrp import (
    NRP_Daily_Approximation_Config_QWIM,
    combine_adjusted_price_panels_NRP_point_in_time_QWIM,
    run_NRP_point_in_time_backtest_QWIM,
    run_SP100_benchmark_backtest_QWIM,
)
from amad_nrp.errors import (
    Exception_Data_Validation_Error,
    Exception_Not_Found,
)
from amad_nrp.errors import get_logger


if TYPE_CHECKING:
    from amad_nrp import (
        NRP_Point_In_Time_Backtest_Result_QWIM,
        SP100_Benchmark_Backtest_Result_QWIM,
    )


_LOGGER = get_logger(name=__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PRICE_BASE_PATH = _PROJECT_ROOT / "data" / "inputs" / "sp500_adjusted_prices.parquet"
_DEFAULT_PRICE_SUPPLEMENTAL_PATH = (
    _PROJECT_ROOT / "data" / "inputs" / "sp100_missing_adjusted_prices.parquet"
)
_DEFAULT_MEMBERSHIP_PATH = _PROJECT_ROOT / "data" / "inputs" / "sp100_membership_intervals.csv"
_DEFAULT_BENCHMARK_PATH = (
    _PROJECT_ROOT / "data" / "inputs" / "sp100_benchmark_adjusted_prices.parquet"
)
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "runs"


def _parse_arguments_NRP_SP100_point_in_time_QWIM(
    arguments_raw: list[str] | None = None,
) -> argparse.Namespace:
    """Parse local input paths and fixed research settings."""
    parser = argparse.ArgumentParser(
        description="Run point-in-time S&P 100 AMAD-NRP, Raw NRP, and S&P 100 benchmark.",
    )
    parser.add_argument("--price-base-path", type=Path, default=_DEFAULT_PRICE_BASE_PATH)
    parser.add_argument(
        "--price-supplemental-path",
        type=Path,
        default=_DEFAULT_PRICE_SUPPLEMENTAL_PATH,
    )
    parser.add_argument("--membership-path", type=Path, default=_DEFAULT_MEMBERSHIP_PATH)
    parser.add_argument("--benchmark-path", type=Path, default=_DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--estimation-months", type=int, default=24)
    parser.add_argument("--AMAD-window", type=int, default=63)
    parser.add_argument("--AMAD-threshold", type=float, default=3.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    return parser.parse_args(args=arguments_raw)


def _require_file_NRP_SP100_point_in_time_QWIM(
    *,
    file_path: Path,
    description: str,
) -> Path:
    """Resolve one required local input file.

    Parameters
    ----------
    file_path : pathlib.Path
        Candidate local input.
    description : str
        Human-readable resource description.

    Returns
    -------
    pathlib.Path
        Resolved existing file path.

    Raises
    ------
    Exception_Not_Found
        Raised when the file is absent.
    """
    resolved_path = file_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise Exception_Not_Found(
            f"{description} was not found.",
            resource_type="file",
            resource_id=str(resolved_path),
        )
    return resolved_path


def _write_table_NRP_SP100_point_in_time_QWIM(
    *,
    data_frame: pl.DataFrame,
    run_directory: Path,
    file_stem: str,
) -> None:
    """Write one research table as CSV and compressed Parquet.

    Parameters
    ----------
    data_frame : polars.DataFrame
        Materialized research table.
    run_directory : pathlib.Path
        Immutable run destination.
    file_stem : str
        Shared base name for both output formats.

    Returns
    -------
    None
        The two table files are written as a controlled boundary side effect.
    """
    data_frame.write_csv(run_directory / f"{file_stem}.csv")
    data_frame.write_parquet(
        run_directory / f"{file_stem}.parquet",
        compression="zstd",
    )


def _write_NRP_artifacts_NRP_SP100_point_in_time_QWIM(
    *,
    result: NRP_Point_In_Time_Backtest_Result_QWIM,
    run_directory: Path,
    prefix: str,
) -> None:
    """Persist the complete result contract for one NRP variant.

    Parameters
    ----------
    result : NRP_Point_In_Time_Backtest_Result_QWIM
        Completed dynamic-universe NRP backtest.
    run_directory : pathlib.Path
        Immutable run destination.
    prefix : str
        Artifact prefix identifying Raw or AMAD preprocessing.

    Returns
    -------
    None
        Result artifacts are written to the run directory.
    """
    artifact_tables = {
        "daily_returns": result.daily_returns,
        "monthly_returns": result.monthly_returns,
        "rebalance_weights": result.rebalance_weights,
        "allocation_diagnostics": result.allocation_diagnostics,
        "rebalance_diagnostics": result.rebalance_diagnostics,
        "membership_audit": result.membership_audit,
        "metrics": result.metrics,
    }
    for item_name, item_data in artifact_tables.items():
        _write_table_NRP_SP100_point_in_time_QWIM(
            data_frame=item_data,
            run_directory=run_directory,
            file_stem=f"{prefix}_{item_name}",
        )
    result.network_edges.write_parquet(
        run_directory / f"{prefix}_network_edges.parquet",
        compression="zstd",
    )


def _write_SP100_benchmark_artifacts_NRP_SP100_point_in_time_QWIM(
    *,
    result: SP100_Benchmark_Backtest_Result_QWIM,
    run_directory: Path,
) -> None:
    """Persist the aligned S&P 100 benchmark.

    Parameters
    ----------
    result : SP100_Benchmark_Backtest_Result_QWIM
        Completed adjusted-price S&P 100 benchmark.
    run_directory : pathlib.Path
        Immutable run destination.

    Returns
    -------
    None
        Benchmark artifacts are written to the run directory.
    """
    artifact_tables = {
        "daily_returns": result.daily_returns,
        "monthly_returns": result.monthly_returns,
        "metrics": result.metrics,
    }
    for item_name, item_data in artifact_tables.items():
        _write_table_NRP_SP100_point_in_time_QWIM(
            data_frame=item_data,
            run_directory=run_directory,
            file_stem=f"SP100_Benchmark_{item_name}",
        )


def _create_daily_comparison_NRP_SP100_point_in_time_QWIM(
    *,
    AMAD_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    raw_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    benchmark_result: SP100_Benchmark_Backtest_Result_QWIM,
) -> pl.DataFrame:
    """Join all daily return and net-value paths on identical dates.

    Parameters
    ----------
    AMAD_result : NRP_Point_In_Time_Backtest_Result_QWIM
        AMAD-preprocessed NRP result.
    raw_result : NRP_Point_In_Time_Backtest_Result_QWIM
        Raw-return NRP result.
    benchmark_result : SP100_Benchmark_Backtest_Result_QWIM
        Adjusted-price S&P 100 benchmark result.

    Returns
    -------
    polars.DataFrame
        Daily comparison path sorted by ``Date``.
    """
    AMAD_path = AMAD_result.daily_returns.select(
        "Date",
        pl.col("NetReturn").alias("AMAD_NRP_Return"),
        pl.col("NetValue").alias("AMAD_NRP_NetValue"),
    )
    raw_path = raw_result.daily_returns.select(
        "Date",
        pl.col("NetReturn").alias("Raw_NRP_Return"),
        pl.col("NetValue").alias("Raw_NRP_NetValue"),
    )
    benchmark_path = benchmark_result.daily_returns.select(
        "Date",
        pl.col("NetReturn").alias("SP100BenchmarkReturn"),
        pl.col("NetValue").alias("SP100BenchmarkNetValue"),
    )
    return (
        AMAD_path.join(raw_path, on="Date", how="inner")
        .join(benchmark_path, on="Date", how="inner")
        .sort("Date")
    )


def _create_monthly_comparison_NRP_SP100_point_in_time_QWIM(
    *,
    AMAD_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    raw_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    benchmark_result: SP100_Benchmark_Backtest_Result_QWIM,
) -> pl.DataFrame:
    """Join all monthly return and net-value paths.

    Parameters
    ----------
    AMAD_result : NRP_Point_In_Time_Backtest_Result_QWIM
        AMAD-preprocessed NRP result.
    raw_result : NRP_Point_In_Time_Backtest_Result_QWIM
        Raw-return NRP result.
    benchmark_result : SP100_Benchmark_Backtest_Result_QWIM
        Adjusted-price S&P 100 benchmark result.

    Returns
    -------
    polars.DataFrame
        Monthly comparison path sorted by ``Month``.
    """
    AMAD_path = AMAD_result.monthly_returns.select(
        "Month",
        pl.col("NetReturn").alias("AMAD_NRP_Return"),
        pl.col("NetValue").alias("AMAD_NRP_NetValue"),
    )
    raw_path = raw_result.monthly_returns.select(
        "Month",
        pl.col("NetReturn").alias("Raw_NRP_Return"),
        pl.col("NetValue").alias("Raw_NRP_NetValue"),
    )
    benchmark_path = benchmark_result.monthly_returns.select(
        "Month",
        pl.col("NetReturn").alias("SP100BenchmarkReturn"),
        pl.col("NetValue").alias("SP100BenchmarkNetValue"),
    )
    return (
        AMAD_path.join(raw_path, on="Month", how="inner")
        .join(benchmark_path, on="Month", how="inner")
        .sort("Month")
    )


def _plot_historical_net_value_NRP_SP100_point_in_time_QWIM(
    *,
    comparison_daily: pl.DataFrame,
    run_directory: Path,
) -> Path:
    """Save the daily historical growth-of-one chart.

    Parameters
    ----------
    comparison_daily : polars.DataFrame
        Joined daily net-value path.
    run_directory : pathlib.Path
        Immutable run destination.
    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    figure, axes = plt.subplots(figsize=(12, 6.5))
    date_values = comparison_daily.get_column("Date").to_list()
    series_specifications = (
        ("AMAD_NRP_NetValue", "AMAD-NRP", "#1d4ed8", 2.6),
        ("Raw_NRP_NetValue", "Raw NRP", "#d97706", 2.0),
        ("SP100BenchmarkNetValue", "S&P 100", "#059669", 1.8),
    )
    for column_name, label_name, color_value, line_width in series_specifications:
        axes.plot(
            date_values,
            comparison_daily.get_column(column_name).to_list(),
            label=label_name,
            color=color_value,
            linewidth=line_width,
        )
    axes.set_title("S&P 100 Point-in-Time Backtest — Historical Net Value")
    axes.set_xlabel("Date")
    axes.set_ylabel("Growth of $1")
    axes.grid(alpha=0.25)
    axes.legend(frameon=False)
    figure.tight_layout()
    output_path = run_directory / "SP100_point_in_time_historical_net_value.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _plot_latest_allocation_NRP_SP100_point_in_time_QWIM(
    *,
    AMAD_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    run_directory: Path,
) -> Path:
    """Save the top-25 latest AMAD-NRP target allocation.

    Parameters
    ----------
    AMAD_result : NRP_Point_In_Time_Backtest_Result_QWIM
        Completed AMAD-NRP result.
    run_directory : pathlib.Path
        Immutable run destination.
    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    latest_date = AMAD_result.rebalance_weights.get_column("EffectiveDate").max()
    latest_allocation = (
        AMAD_result.rebalance_weights.filter(
            pl.col("EffectiveDate") == latest_date,
        )
        .sort("Weight", descending=True)
        .head(25)
        .sort("Weight")
    )
    figure, axes = plt.subplots(figsize=(10, 8))
    axes.barh(
        latest_allocation.get_column("Asset").to_list(),
        (latest_allocation.get_column("Weight") * 100.0).to_list(),
        color="#1d4ed8",
    )
    axes.set_title("Latest Point-in-Time AMAD-NRP Allocation — Top 25")
    axes.set_xlabel("Target weight (%)")
    axes.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output_path = run_directory / "AMAD_NRP_latest_allocation_top25.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _validate_paired_universes_NRP_SP100_point_in_time_QWIM(
    *,
    AMAD_result: NRP_Point_In_Time_Backtest_Result_QWIM,
    raw_result: NRP_Point_In_Time_Backtest_Result_QWIM,
) -> None:
    """Require Raw and AMAD NRP to use identical eligible memberships.

    Parameters
    ----------
    AMAD_result : NRP_Point_In_Time_Backtest_Result_QWIM
        AMAD-preprocessed result.
    raw_result : NRP_Point_In_Time_Backtest_Result_QWIM
        Raw-return result.

    Returns
    -------
    None
        Successful validation has no return value.

    Raises
    ------
    Exception_Data_Validation_Error
        Raised when eligible rows differ between the two variants.
    """
    audit_columns = ["EffectiveDate", "Price_Ticker", "Eligible"]
    if not AMAD_result.membership_audit.select(audit_columns).equals(
        raw_result.membership_audit.select(audit_columns),
    ):
        raise Exception_Data_Validation_Error(
            "AMAD-NRP and Raw NRP must use identical point-in-time universes.",
            field="membership_audit",
            value="variant_mismatch",
        )


def run_backtest_study_NRP_SP100_point_in_time_QWIM(
    *,
    estimation_months: int = 24,
    AMAD_window: int = 63,
    AMAD_threshold: float = 3.0,
    transaction_cost_bps: float = 0.0,
    price_base_path: Path = _DEFAULT_PRICE_BASE_PATH,
    price_supplemental_path: Path = _DEFAULT_PRICE_SUPPLEMENTAL_PATH,
    membership_path: Path = _DEFAULT_MEMBERSHIP_PATH,
    benchmark_path: Path = _DEFAULT_BENCHMARK_PATH,
    output_root: Path = _DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Run one point-in-time study and return its immutable output directory."""
    price_base_path = _require_file_NRP_SP100_point_in_time_QWIM(
        file_path=price_base_path,
        description="Base adjusted-price panel",
    )
    price_supplemental_path = _require_file_NRP_SP100_point_in_time_QWIM(
        file_path=price_supplemental_path,
        description="Supplemental adjusted-price panel",
    )
    membership_path = _require_file_NRP_SP100_point_in_time_QWIM(
        file_path=membership_path,
        description="Point-in-time S&P 100 membership intervals",
    )
    benchmark_path = _require_file_NRP_SP100_point_in_time_QWIM(
        file_path=benchmark_path,
        description="S&P 100 adjusted-price benchmark",
    )
    price_data = combine_adjusted_price_panels_NRP_point_in_time_QWIM(
        price_data_base=pl.read_parquet(price_base_path),
        price_data_supplemental=pl.read_parquet(price_supplemental_path),
    )
    membership_data = pl.read_csv(membership_path, try_parse_dates=True)
    benchmark_price_data = pl.read_parquet(benchmark_path)
    raw_config = NRP_Daily_Approximation_Config_QWIM(
        estimation_months=estimation_months,
        transaction_cost_bps=transaction_cost_bps,
        return_preprocessing="raw",
        weight_normalization="inverse_proportional",
        adjacency_weighting="correlation_distance",
    )
    AMAD_config = NRP_Daily_Approximation_Config_QWIM(
        estimation_months=estimation_months,
        transaction_cost_bps=transaction_cost_bps,
        return_preprocessing="AMAD",
        AMAD_window=AMAD_window,
        AMAD_threshold=AMAD_threshold,
        weight_normalization="inverse_proportional",
        adjacency_weighting="correlation_distance",
    )

    _LOGGER.info("Starting point-in-time S&P 100 NRP study")
    started_at = time.perf_counter()
    raw_result = run_NRP_point_in_time_backtest_QWIM(
        price_data=price_data,
        membership_intervals=membership_data,
        config=raw_config,
    )
    AMAD_result = run_NRP_point_in_time_backtest_QWIM(
        price_data=price_data,
        membership_intervals=membership_data,
        config=AMAD_config,
    )
    _validate_paired_universes_NRP_SP100_point_in_time_QWIM(
        AMAD_result=AMAD_result,
        raw_result=raw_result,
    )
    benchmark_result = run_SP100_benchmark_backtest_QWIM(
        benchmark_price_data=benchmark_price_data,
        comparison_dates=AMAD_result.daily_returns.get_column("Date"),
    )
    elapsed_seconds = time.perf_counter() - started_at

    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = output_root.expanduser().resolve()
    run_directory = output_root / f"run_{run_timestamp}_months{AMAD_result.monthly_returns.height}"
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(membership_path, run_directory / "SP100_membership_intervals_source.csv")

    _write_NRP_artifacts_NRP_SP100_point_in_time_QWIM(
        result=AMAD_result,
        run_directory=run_directory,
        prefix="AMAD_NRP",
    )
    _write_NRP_artifacts_NRP_SP100_point_in_time_QWIM(
        result=raw_result,
        run_directory=run_directory,
        prefix="Raw_NRP",
    )
    _write_SP100_benchmark_artifacts_NRP_SP100_point_in_time_QWIM(
        result=benchmark_result,
        run_directory=run_directory,
    )
    comparison_daily = _create_daily_comparison_NRP_SP100_point_in_time_QWIM(
        AMAD_result=AMAD_result,
        raw_result=raw_result,
        benchmark_result=benchmark_result,
    )
    comparison_monthly = _create_monthly_comparison_NRP_SP100_point_in_time_QWIM(
        AMAD_result=AMAD_result,
        raw_result=raw_result,
        benchmark_result=benchmark_result,
    )
    comparison_metrics = pl.concat(
        [AMAD_result.metrics, raw_result.metrics, benchmark_result.metrics],
        how="vertical_relaxed",
    )
    _write_table_NRP_SP100_point_in_time_QWIM(
        data_frame=comparison_daily,
        run_directory=run_directory,
        file_stem="Strategy_comparison_daily_path",
    )
    _write_table_NRP_SP100_point_in_time_QWIM(
        data_frame=comparison_monthly,
        run_directory=run_directory,
        file_stem="Strategy_comparison_monthly_path",
    )
    _write_table_NRP_SP100_point_in_time_QWIM(
        data_frame=comparison_metrics,
        run_directory=run_directory,
        file_stem="Strategy_comparison_metrics",
    )
    historical_chart_path = _plot_historical_net_value_NRP_SP100_point_in_time_QWIM(
        comparison_daily=comparison_daily,
        run_directory=run_directory,
    )
    latest_chart_path = _plot_latest_allocation_NRP_SP100_point_in_time_QWIM(
        AMAD_result=AMAD_result,
        run_directory=run_directory,
    )

    latest_effective_date = AMAD_result.rebalance_weights.get_column("EffectiveDate").max()
    latest_allocation = AMAD_result.rebalance_weights.filter(
        pl.col("EffectiveDate") == latest_effective_date,
    ).sort("Weight", descending=True)
    _write_table_NRP_SP100_point_in_time_QWIM(
        data_frame=latest_allocation,
        run_directory=run_directory,
        file_stem="AMAD_NRP_latest_allocation",
    )
    eligibility_summary = (
        AMAD_result.membership_audit.group_by("EffectiveDate")
        .agg(
            pl.len().alias("ActiveMembers"),
            pl.col("Eligible").sum().alias("EligibleAssets"),
        )
        .sort("EffectiveDate")
    )
    _write_table_NRP_SP100_point_in_time_QWIM(
        data_frame=eligibility_summary,
        run_directory=run_directory,
        file_stem="SP100_monthly_eligibility_summary",
    )
    excluded_latest = AMAD_result.membership_audit.filter(
        (pl.col("EffectiveDate") == latest_effective_date) & ~pl.col("Eligible"),
    ).select("Price_Ticker", "EligibilityReason")
    metadata = {
        "study": "Point-in-time S&P 100 company-level daily-frequency NRP backtest",
        "price_base_path": str(price_base_path),
        "price_supplemental_path": str(price_supplemental_path),
        "membership_path": str(membership_path),
        "benchmark_path": str(benchmark_path),
        "benchmark": (
            "Buy-and-hold adjusted-price S&P 100 benchmark return."
        ),
        "price_basis": "Dividend- and split-adjusted daily close prices",
        "price_start": str(price_data.get_column("Date").min()),
        "price_end": str(price_data.get_column("Date").max()),
        "out_of_sample_start": str(comparison_daily.get_column("Date").min()),
        "out_of_sample_end": str(comparison_daily.get_column("Date").max()),
        "out_of_sample_months": comparison_monthly.height,
        "latest_effective_date": str(latest_effective_date),
        "latest_eligible_assets": int(
            eligibility_summary.get_column("EligibleAssets").tail(1).item(),
        ),
        "latest_excluded_assets": excluded_latest.to_dicts(),
        "membership_rule": (
            "Membership is evaluated on the first trading day of each monthly "
            "holding period using inclusive starts and exclusive ends."
        ),
        "history_rule": (
            "An active member is eligible only with complete positive prices "
            "through the 24-month estimation and current holding window."
        ),
        "synthetic_backfill": False,
        "experimental_control": (
            "AMAD-NRP and Raw NRP use identical point-in-time members, eligible "
            "histories, holding dates, weighted correlation-distance MSTs, "
            "ordinary inverse-centrality normalization, and costs. Only AMAD "
            "preprocessing of estimation returns differs."
        ),
        "config": {
            "estimation_months": estimation_months,
            "transaction_cost_bps": transaction_cost_bps,
            "weight_normalization": AMAD_config.weight_normalization,
            "adjacency_weighting": AMAD_config.adjacency_weighting,
            "AMAD_window": AMAD_window,
            "AMAD_threshold": AMAD_threshold,
        },
        "artifacts": {
            "historical_net_value_chart": historical_chart_path.name,
            "latest_allocation_chart": latest_chart_path.name,
            "daily_net_value_path": "Strategy_comparison_daily_path.parquet",
            "AMAD_weight_history": "AMAD_NRP_rebalance_weights.parquet",
            "benchmark_daily_path": "SP100_Benchmark_daily_returns.parquet",
        },
        "elapsed_seconds": elapsed_seconds,
    }
    (run_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    _LOGGER.info(
        "Completed point-in-time S&P 100 study in %.1f seconds; outputs=%s",
        elapsed_seconds,
        run_directory,
    )
    return run_directory


def main() -> None:
    """Run the full local point-in-time study from command-line settings."""
    arguments = _parse_arguments_NRP_SP100_point_in_time_QWIM()
    run_backtest_study_NRP_SP100_point_in_time_QWIM(
        estimation_months=arguments.estimation_months,
        AMAD_window=arguments.AMAD_window,
        AMAD_threshold=arguments.AMAD_threshold,
        transaction_cost_bps=arguments.transaction_cost_bps,
        price_base_path=arguments.price_base_path,
        price_supplemental_path=arguments.price_supplemental_path,
        membership_path=arguments.membership_path,
        benchmark_path=arguments.benchmark_path,
        output_root=arguments.output_root,
    )


if __name__ == "__main__":
    main()
