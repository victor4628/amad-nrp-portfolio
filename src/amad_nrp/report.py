"""Generate an original AMAD-NRP research report from one completed run."""

from __future__ import annotations

import json
import math
import shutil
import subprocess

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from amad_nrp.errors import Exception_Not_Found


_SERIES = (
    ("AMAD_NRP_NetValue", "AMAD_NRP_Return", "AMAD-NRP", "#1d4ed8"),
    ("Raw_NRP_NetValue", "Raw_NRP_Return", "Raw NRP", "#d97706"),
    ("SP100BenchmarkNetValue", "SP100BenchmarkReturn", "S&P 100", "#059669"),
)

_DRAWDOWN_SERIES = tuple(item for item in _SERIES if item[2] != "Raw NRP")


def _nice_axis_step(value_range: float, target_intervals: int) -> float:
    """Return a readable 1/2/5-based axis interval."""
    rough_step = max(value_range / target_intervals, np.finfo(float).eps)
    magnitude = 10.0 ** math.floor(math.log10(rough_step))
    normalized_step = rough_step / magnitude
    if normalized_step <= 1.5:
        factor = 1.0
    elif normalized_step <= 3.0:
        factor = 2.0
    elif normalized_step <= 7.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * magnitude


def _clean_axis_limits(
    values: np.ndarray,
    *,
    reference_value: float,
    target_intervals: int,
) -> tuple[float, float, float]:
    """Return clean lower, upper, and interval values for one chart axis."""
    minimum = min(float(np.min(values)), reference_value)
    maximum = max(float(np.max(values)), reference_value)
    value_range = max(maximum - minimum, 0.01)
    step = _nice_axis_step(value_range, target_intervals)
    lower = math.floor(minimum / step) * step
    candidate_upper = math.ceil(maximum / step) * step
    upper = candidate_upper if candidate_upper > lower else lower + step
    return lower, upper, step


def _axis_ticks(lower: float, upper: float, step: float) -> np.ndarray:
    """Build stable ascending ticks including both clean axis boundaries."""
    count = int(round((upper - lower) / step)) + 1
    return lower + np.arange(count, dtype=float) * step


def _date_ticks(dates: list[Any]) -> tuple[list[Any], list[str]]:
    """Match the Dashboard's sample endpoints and annual date markers."""
    ticks = [dates[0]]
    labels = [dates[0].strftime("%Y-%m")]
    first_year = dates[0].year
    last_year = dates[-1].year
    for year in range(first_year + 1, last_year + 1):
        annual_tick = next((item for item in dates if item.year == year), None)
        if annual_tick is not None:
            ticks.append(annual_tick)
            labels.append(str(year))
    if dates[-1] != ticks[-1]:
        ticks.append(dates[-1])
        labels.append(dates[-1].strftime("%Y-%m"))
    return ticks, labels


def _style_report_axes(axes: Any, *, dates: list[Any]) -> None:
    """Apply shared Dashboard-aligned chart styling."""
    date_ticks, date_labels = _date_ticks(dates)
    axes.set_xticks(date_ticks, labels=date_labels)
    axes.grid(color="#e2e8f0", linewidth=0.7)
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(colors="#64748b", labelsize=8)
    axes.yaxis.label.set_color("#475569")


def _sample_standard_deviation(values: np.ndarray) -> float:
    """Return sample standard deviation with a stable small-sample fallback."""
    return float(np.std(values, ddof=1)) if values.size >= 2 else 0.0


def _performance_rows(
    *,
    daily_path: pl.DataFrame,
    source_metrics: pl.DataFrame,
) -> list[dict[str, Any]]:
    """Calculate the dashboard performance table from daily net returns."""
    output: list[dict[str, Any]] = []
    for strategy_key, (_, return_column, label, _) in zip(
        ("AMAD", "raw", "benchmark"),
        _SERIES,
        strict=True,
    ):
        returns = daily_path.get_column(return_column).to_numpy()
        total_return = float(np.prod(1.0 + returns) - 1.0)
        annualized_return = float((1.0 + total_return) ** (252.0 / returns.size) - 1.0)
        annualized_volatility = _sample_standard_deviation(returns) * math.sqrt(252.0)
        downside = returns[returns < 0.0]
        downside_deviation = _sample_standard_deviation(downside) * math.sqrt(252.0)
        wealth = np.cumprod(1.0 + returns)
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        maximum_drawdown = float(np.min(drawdown))
        source = source_metrics.filter(pl.col("ReturnPreprocessing") == strategy_key)
        output.append(
            {
                "strategy": label,
                "months": int(source.get_column("OutOfSampleMonths").item()),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe": annualized_return / annualized_volatility if annualized_volatility else 0.0,
                "sortino": annualized_return / downside_deviation if downside_deviation else 0.0,
                "calmar": annualized_return / abs(maximum_drawdown) if maximum_drawdown else 0.0,
                "maximum_drawdown": maximum_drawdown,
                "turnover": float(source.get_column("AverageTurnover").item() or 0.0),
            },
        )
    return output


def _format_percent(value: float) -> str:
    """Format a decimal value as a percentage for the PDF."""
    return f"{value * 100.0:.2f}%"


def _plot_report_charts(*, daily_path: pl.DataFrame, report_directory: Path) -> tuple[Path, Path]:
    """Render report-native net-value and drawdown charts."""
    dates = daily_path.get_column("Date").to_list()
    plt.style.use("default")

    figure, axes = plt.subplots(figsize=(10.8, 4.8))
    net_value_arrays: list[np.ndarray] = []
    for value_column, _, label, color in _SERIES:
        values = daily_path.get_column(value_column).to_numpy()
        net_value_arrays.append(values)
        axes.plot(
            dates,
            values,
            label=label,
            color=color,
            linewidth=2.2 if label == "AMAD-NRP" else 1.6,
        )
    net_values = np.concatenate(net_value_arrays)
    lower, upper, step = _clean_axis_limits(
        net_values,
        reference_value=1.0,
        target_intervals=7,
    )
    ticks = _axis_ticks(lower, upper, step)
    axes.set_ylim(lower, upper)
    axes.set_yticks(ticks, labels=[f"{item:.4f}".rstrip("0").rstrip(".") for item in ticks])
    axes.set_ylabel("Growth of $1")
    _style_report_axes(axes, dates=dates)
    axes.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
    figure.tight_layout(pad=0.8)
    net_value_path = report_directory / "historical-net-value.png"
    figure.savefig(net_value_path, dpi=190, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(10.8, 4.8))
    drawdown_arrays: list[np.ndarray] = []
    for value_column, _, label, color in _DRAWDOWN_SERIES:
        wealth = daily_path.get_column(value_column).to_numpy()
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        drawdown_arrays.append(drawdown)
        axes.fill_between(dates, drawdown, 0.0, color=color, alpha=0.14, linewidth=0)
        axes.plot(
            dates,
            drawdown,
            label=label,
            color=color,
            linewidth=2.2 if label == "AMAD-NRP" else 1.6,
        )
    drawdowns = np.concatenate(drawdown_arrays)
    lower, upper, step = _clean_axis_limits(
        drawdowns,
        reference_value=0.0,
        target_intervals=5,
    )
    ticks = _axis_ticks(lower, upper, step)
    axes.set_ylim(lower, upper)
    axes.set_yticks(
        ticks,
        labels=[f"{0.0 if abs(item) < 1e-10 else item * 100.0:g}%" for item in ticks],
    )
    axes.axhline(0.0, color="#94a3b8", linewidth=0.8)
    axes.set_ylabel("Drawdown (%)")
    _style_report_axes(axes, dates=dates)
    axes.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
    figure.tight_layout(pad=0.8)
    drawdown_path = report_directory / "drawdown-comparison.png"
    figure.savefig(drawdown_path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return net_value_path, drawdown_path


def _typst_escape(value: Any) -> str:
    """Escape plain text before placing it in Typst content blocks."""
    text = str(value)
    for source, replacement in (
        ("\\", "\\\\"),
        ("#", "\\#"),
        ("[", "\\["),
        ("]", "\\]"),
        ("$", "\\$"),
        ("_", "\\_"),
    ):
        text = text.replace(source, replacement)
    return text


def _table_cells(values: list[str]) -> str:
    """Render Typst table cells as content blocks."""
    return ",\n".join(f"[{_typst_escape(value)}]" for value in values)


def generate_research_report(
    *,
    run_directory: Path,
    output_directory: Path,
) -> Path:
    """Generate and compile a downloadable Typst PDF for one active run."""
    run_directory = run_directory.resolve()
    metadata_path = run_directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise Exception_Not_Found(
            "Backtest metadata was not found.",
            resource_type="file",
            resource_id=str(metadata_path),
        )
    typst_executable = shutil.which("typst")
    if typst_executable is None:
        raise Exception_Not_Found(
            "Typst is required to generate the PDF report. Install it with `brew install typst`.",
            resource_type="executable",
            resource_id="typst",
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    daily_path = pl.read_csv(run_directory / "Strategy_comparison_daily_path.csv", try_parse_dates=True)
    source_metrics = pl.read_csv(run_directory / "Strategy_comparison_metrics.csv")
    latest_allocation = pl.read_csv(run_directory / "AMAD_NRP_latest_allocation.csv")
    output_directory.mkdir(parents=True, exist_ok=True)
    report_directory = output_directory / run_directory.name
    report_directory.mkdir(parents=True, exist_ok=True)
    net_value_path, drawdown_path = _plot_report_charts(
        daily_path=daily_path,
        report_directory=report_directory,
    )
    performance = _performance_rows(daily_path=daily_path, source_metrics=source_metrics)

    AMAD_returns = daily_path.get_column("AMAD_NRP_Return")
    AMAD_source = source_metrics.filter(pl.col("ReturnPreprocessing") == "AMAD")
    diagnostics = {
        "average_maximum_weight": float(AMAD_source.get_column("AverageMaximumWeight").item()),
        "best_day": float(AMAD_returns.max()),
        "worst_day": float(AMAD_returns.min()),
        "effective_holdings": float(AMAD_source.get_column("AverageEffectiveHoldings").item()),
    }

    performance_header = [
        "Strategy", "Months", "Total", "Ann. return", "Ann. vol.",
        "Sharpe", "Sortino", "Calmar", "Max DD", "Turnover",
    ]
    performance_values = performance_header + [
        value
        for row in performance
        for value in (
            row["strategy"],
            str(row["months"]),
            _format_percent(row["total_return"]),
            _format_percent(row["annualized_return"]),
            _format_percent(row["annualized_volatility"]),
            f"{row['sharpe']:.2f}",
            f"{row['sortino']:.2f}",
            f"{row['calmar']:.2f}",
            _format_percent(row["maximum_drawdown"]),
            _format_percent(row["turnover"]),
        )
    ]
    allocation_values = ["Rank", "Asset", "Weight"] + [
        value
        for rank, row in enumerate(latest_allocation.head(20).iter_rows(named=True), start=1)
        for value in (str(rank), str(row["Asset"]), _format_percent(float(row["Weight"])))
    ]
    config = metadata["config"]
    report_source = f"""
#set page(paper: "a4", margin: (x: 17mm, y: 16mm), numbering: "1 / 1")
#set text(font: "New Computer Modern", size: 9pt, fill: rgb("172033"))
#set par(leading: 0.65em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(above: 15pt, below: 8pt, text(size: 16pt, weight: "semibold", it.body))
#show heading.where(level: 2): it => block(above: 10pt, below: 6pt, text(size: 11pt, weight: "semibold", fill: rgb("334155"), it.body))

#let blue = rgb("1d4ed8")
#let slate = rgb("64748b")
#let border = rgb("dce2e8")
#let panel(body) = block(width: 100%, fill: rgb("f8fafc"), stroke: border, inset: 10pt, radius: 3pt, body)

#text(size: 9pt, weight: "bold", fill: blue)[AMAD-NRP RESEARCH REPORT]
#v(9pt)
#text(size: 25pt, weight: "semibold")[AMAD-Enhanced Network Risk Parity for S&P 100]
#v(7pt)
#text(size: 11pt, fill: slate)[Point-in-time constituents | Monthly rebalance | Daily out-of-sample evaluation]
#v(17pt)
#panel[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 12pt,
    [*Sample* \\ {_typst_escape(metadata['out_of_sample_start'])} to {_typst_escape(metadata['out_of_sample_end'])}],
    [*Latest rebalance* \\ {_typst_escape(metadata['latest_effective_date'])}],
    [*Eligible assets* \\ {_typst_escape(metadata['latest_eligible_assets'])}],
  )
]

= Executive summary
AMAD-NRP applies a rolling median and median absolute deviation filter to estimation returns before building the correlation-distance network. Returns beyond the configured threshold are continuously compressed with the implemented logarithmic transformation; holding-period returns remain unmodified.

#table(
  columns: (1.25fr, 0.55fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr, 0.9fr),
  inset: 4pt,
  stroke: border,
  align: (left, right, right, right, right, right, right, right, right, right),
  table.header({_table_cells(performance_values[:10])}),
  {_table_cells(performance_values[10:])}
)

#v(8pt)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  [*Average max position* \\ {_format_percent(diagnostics['average_maximum_weight'])}],
  [*Average effective holdings* \\ {diagnostics['effective_holdings']:.2f}],
  [*Best day* \\ {_format_percent(diagnostics['best_day'])}],
  [*Worst day* \\ {_format_percent(diagnostics['worst_day'])}],
)

#pagebreak()
= Historical net value
#text(fill: slate)[Growth of one dollar after the configured transaction-cost assumption.]
#v(6pt)
#block(width: 100%)[#image("{net_value_path.name}", width: 100%)]

= Drawdown comparison
#text(fill: slate)[Decline from each strategy's running peak.]
#v(6pt)
#block(width: 100%)[#image("{drawdown_path.name}", width: 100%)]

#pagebreak()
= Latest AMAD-NRP allocation
#text(fill: slate)[Top 20 positions at {_typst_escape(metadata['latest_effective_date'])}. The backtest uses the complete allocation.]
#v(7pt)
#table(
  columns: (0.45fr, 1.2fr, 1fr),
  inset: 5pt,
  stroke: border,
  align: (right, left, right),
  table.header({_table_cells(allocation_values[:3])}),
  {_table_cells(allocation_values[3:])}
)

= Run specification
#table(
  columns: (1.2fr, 1fr), inset: 5pt, stroke: border,
  [Estimation window], [{_typst_escape(config['estimation_months'])} months],
  [AMAD window], [{_typst_escape(config['AMAD_window'])} days],
  [AMAD threshold], [{_typst_escape(config['AMAD_threshold'])} MAD],
  [Transaction cost], [{_typst_escape(config['transaction_cost_bps'])} bps],
  [Network], [Weighted correlation-distance MST],
  [Allocation], [Ordinary inverse-centrality, long-only, fully invested],
)

= Method and interpretation
For each asset, the trailing window includes the current observation. An estimation return is flagged when its absolute deviation from the rolling median exceeds the threshold multiplied by the rolling MAD. Only flagged deviations are logarithmically dampened. AMAD-NRP and Raw NRP otherwise use identical point-in-time membership, eligibility rules, network construction, allocation normalization, holding dates, and costs.

{_typst_escape(metadata['membership_rule'])} {_typst_escape(metadata['history_rule'])}

#v(12pt)
#line(length: 100%, stroke: border)
#v(5pt)
#text(size: 8pt, fill: slate)[Research use only. Historical backtests do not predict future performance and are not investment advice.]
"""
    source_path = report_directory / "AMAD-NRP-research-report.typ"
    output_path = report_directory / "AMAD-NRP-research-report.pdf"
    source_path.write_text(report_source.strip() + "\n", encoding="utf-8")
    result = subprocess.run(
        [typst_executable, "compile", str(source_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not output_path.is_file():
        message = result.stderr.strip() or result.stdout.strip() or "Unknown Typst compilation error."
        raise RuntimeError(f"PDF report generation failed: {message}")
    return output_path
