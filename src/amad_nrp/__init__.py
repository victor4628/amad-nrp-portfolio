"""Standalone API for dynamic S&P 100 AMAD Network Risk Parity research."""

from .amad import (
    NRP_AMAD_Config_Input_QWIM,
    NRP_AMAD_Config_QWIM,
    apply_AMAD_returns_NRP_QWIM,
    create_NRP_AMAD_config_QWIM,
)
from .benchmark import (
    SP100_Benchmark_Backtest_Result_QWIM,
    run_SP100_benchmark_backtest_QWIM,
)
from .core import (
    Monthly_Realized_Covariances_NRP_Daily_QWIM,
    NRP_Daily_Allocation_QWIM,
    NRP_Daily_Approximation_Backtest_Result_QWIM,
    NRP_Daily_Approximation_Config_Input_QWIM,
    NRP_Daily_Approximation_Config_QWIM,
    calculate_daily_log_returns_NRP_daily_approximation_QWIM,
    calculate_monthly_realized_covariances_NRP_daily_approximation_QWIM,
    calculate_NRP_daily_approximation_allocation_QWIM,
    create_NRP_daily_approximation_config_QWIM,
    run_NRP_daily_approximation_backtest_QWIM,
)
from .point_in_time import (
    NRP_Point_In_Time_Backtest_Result_QWIM,
    combine_adjusted_price_panels_NRP_point_in_time_QWIM,
    run_NRP_point_in_time_backtest_QWIM,
)

__all__ = [name for name in globals() if name.endswith("QWIM")]
