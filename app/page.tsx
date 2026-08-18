"use client";

import { useMemo, useRef, useState, useEffect } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";

type StrategyKey = "AMAD" | "raw" | "benchmark";

type MetricRow = {
  ReturnPreprocessing: StrategyKey;
  OutOfSampleMonths: number;
  AverageMaximumWeight: number | null;
  AverageTurnover: number | null;
};

type PathRow = {
  Date: string;
  AMAD_NRP_Return: number;
  AMAD_NRP_NetValue: number;
  Raw_NRP_Return: number;
  Raw_NRP_NetValue: number;
  SP100BenchmarkReturn: number;
  SP100BenchmarkNetValue: number;
};

type AllocationRow = { Asset: string; Weight: number; EffectiveDate: string };
type RunMetadata = {
  out_of_sample_start: string;
  out_of_sample_end: string;
  latest_effective_date: string;
  latest_eligible_assets: number;
  membership_rule: string;
  history_rule: string;
  synthetic_backfill: boolean;
  elapsed_seconds: number;
  config: {
    estimation_months: number;
    transaction_cost_bps: number;
    weight_normalization: string;
    adjacency_weighting: string;
    AMAD_window: number;
    AMAD_threshold: number;
  };
};

type BacktestResponse = {
  run_id: string;
  metadata: RunMetadata;
  metrics: MetricRow[];
  path: PathRow[];
  allocation: AllocationRow[];
};

type PerformanceRow = {
  key: StrategyKey;
  name: string;
  months: number;
  totalReturn: number;
  annualizedReturn: number;
  annualizedVolatility: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  maximumDrawdown: number;
  turnover: number;
};

const strategyDefinitions: Array<{
  key: StrategyKey;
  name: string;
  color: string;
  valueKey: keyof PathRow;
  returnKey: keyof PathRow;
}> = [
  { key: "AMAD", name: "AMAD-NRP", color: "#1d4ed8", valueKey: "AMAD_NRP_NetValue", returnKey: "AMAD_NRP_Return" },
  { key: "raw", name: "Raw NRP", color: "#d97706", valueKey: "Raw_NRP_NetValue", returnKey: "Raw_NRP_Return" },
  { key: "benchmark", name: "S&P 100", color: "#059669", valueKey: "SP100BenchmarkNetValue", returnKey: "SP100BenchmarkReturn" },
];

const focusedStrategyDefinitions = strategyDefinitions.filter((definition) => definition.key !== "raw");
const lineChartPadding = { top: 18, right: 18, bottom: 30, left: 54 };
const distributionChartPadding = { top: 16, right: 18, bottom: 34, left: 46 };

function formatPercent(value: number, digits = 2) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function formatRatio(value: number) {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function calculateNiceAxisStep(range: number, targetIntervals = 4) {
  const roughStep = Math.max(range / targetIntervals, Number.EPSILON);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalizedStep = roughStep / magnitude;
  const factor = normalizedStep <= 1.5 ? 1 : normalizedStep <= 3 ? 2 : normalizedStep <= 7 ? 5 : 10;
  return factor * magnitude;
}

function formatNetValueAxisTick(value: number) {
  return Number(value.toFixed(4)).toString();
}

function formatDrawdownAxisTick(value: number) {
  const percentage = Math.abs(value) < 1e-10 ? 0 : value * 100;
  return `${Number(percentage.toFixed(2))}%`;
}

function formatPercentageAxisTick(value: number) {
  const percentage = Math.abs(value) < 1e-10 ? 0 : value;
  return `${Number(percentage.toFixed(2))}%`;
}

function sampleStandardDeviation(values: number[]) {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1));
}

function buildPerformanceRows(path: PathRow[], metrics: MetricRow[]): PerformanceRow[] {
  if (path.length === 0) return [];
  return strategyDefinitions.map((definition) => {
    const returns = path.map((item) => Number(item[definition.returnKey]));
    const totalReturn = returns.reduce((wealth, value) => wealth * (1 + value), 1) - 1;
    const annualizedReturn = (1 + totalReturn) ** (252 / returns.length) - 1;
    const annualizedVolatility = sampleStandardDeviation(returns) * Math.sqrt(252);
    const downsideDeviation = sampleStandardDeviation(returns.filter((value) => value < 0)) * Math.sqrt(252);
    let wealth = 1;
    let peak = 1;
    let maximumDrawdown = 0;
    returns.forEach((value) => {
      wealth *= 1 + value;
      peak = Math.max(peak, wealth);
      maximumDrawdown = Math.min(maximumDrawdown, wealth / peak - 1);
    });
    const source = metrics.find((item) => item.ReturnPreprocessing === definition.key);
    return {
      key: definition.key,
      name: definition.name,
      months: Number(source?.OutOfSampleMonths ?? 0),
      totalReturn,
      annualizedReturn,
      annualizedVolatility,
      sharpe: annualizedVolatility > 0 ? annualizedReturn / annualizedVolatility : 0,
      sortino: downsideDeviation > 0 ? annualizedReturn / downsideDeviation : 0,
      calmar: maximumDrawdown < 0 ? annualizedReturn / Math.abs(maximumDrawdown) : 0,
      maximumDrawdown,
      turnover: Number(source?.AverageTurnover ?? 0),
    };
  });
}

function useCanvas(
  draw: (context: CanvasRenderingContext2D, width: number, height: number) => void,
  dependencies: unknown[],
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);
      draw(context, rect.width, rect.height);
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
    // The caller provides the complete redraw dependency list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);
  return canvasRef;
}

function MultiLineChart({ path, mode }: { path: PathRow[]; mode: "value" | "drawdown" }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const canvasRef = useCanvas((context, width, height) => {
    if (path.length < 2) return;
    const padding = lineChartPadding;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const visibleDefinitions = mode === "drawdown" ? focusedStrategyDefinitions : strategyDefinitions;
    const series = visibleDefinitions.map((definition) => {
      const values = path.map((item) => Number(item[definition.valueKey]));
      if (mode === "value") return { ...definition, values };
      let peak = 1;
      return {
        ...definition,
        values: values.map((value) => {
          peak = Math.max(peak, value);
          return value / peak - 1;
        }),
      };
    });
    const allValues = series.flatMap((item) => item.values);
    const minimum = Math.min(...allValues, mode === "drawdown" ? -0.01 : 1);
    const maximum = Math.max(...allValues, mode === "drawdown" ? 0 : 1);
    const spread = Math.max(maximum - minimum, 0.01);
    const axisStep = calculateNiceAxisStep(spread, mode === "value" ? 7 : 5);
    const low = Math.floor(minimum / axisStep) * axisStep;
    const candidateHigh = Math.ceil(maximum / axisStep) * axisStep;
    const high = candidateHigh > low ? candidateHigh : low + axisStep;
    const yTickValues = Array.from(
      { length: Math.round((high - low) / axisStep) + 1 },
      (_, index) => high - index * axisStep,
    );
    const firstYear = Number(path[0].Date.slice(0, 4));
    const lastYear = Number(path.at(-1)!.Date.slice(0, 4));
    const xTicks: Array<{ index: number; label: string; align: CanvasTextAlign }> = [
      { index: 0, label: path[0].Date.slice(0, 7), align: "left" },
    ];
    for (let year = firstYear + 1; year <= lastYear; year += 1) {
      const index = path.findIndex((item) => item.Date.startsWith(`${year}-`));
      if (index >= 0) xTicks.push({ index, label: String(year), align: "center" });
    }
    xTicks.push({ index: path.length - 1, label: path.at(-1)!.Date.slice(0, 7), align: "right" });

    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.strokeStyle = "#e2e8f0";
    context.fillStyle = "#64748b";
    context.lineWidth = 1;
    yTickValues.forEach((value) => {
      const y = padding.top + ((high - value) / (high - low)) * chartHeight;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
      context.textAlign = "right";
      context.fillText(mode === "drawdown" ? formatDrawdownAxisTick(value) : formatNetValueAxisTick(value), padding.left - 8, y + 4);
    });
    xTicks.slice(1, -1).forEach((tick) => {
      const x = padding.left + (tick.index / (path.length - 1)) * chartWidth;
      context.strokeStyle = "#edf0f3";
      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, padding.top + chartHeight);
      context.stroke();
    });
    if (mode === "drawdown") {
      const zeroY = padding.top + ((high - 0) / (high - low)) * chartHeight;
      series.forEach((item) => {
        context.beginPath();
        context.moveTo(padding.left, zeroY);
        item.values.forEach((value, index) => {
          const x = padding.left + (index / (path.length - 1)) * chartWidth;
          const y = padding.top + ((high - value) / (high - low)) * chartHeight;
          context.lineTo(x, y);
        });
        context.lineTo(width - padding.right, zeroY);
        context.closePath();
        context.fillStyle = `${item.color}24`;
        context.fill();
      });
    }
    series.forEach((item) => {
      context.beginPath();
      item.values.forEach((value, index) => {
        const x = padding.left + (index / (path.length - 1)) * chartWidth;
        const y = padding.top + ((high - value) / (high - low)) * chartHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.strokeStyle = item.color;
      context.lineWidth = item.key === "AMAD" ? 2.2 : 1.6;
      context.lineJoin = "round";
      context.stroke();
    });
    context.fillStyle = "#64748b";
    xTicks.forEach((tick) => {
      const x = padding.left + (tick.index / (path.length - 1)) * chartWidth;
      context.strokeStyle = "#94a3b8";
      context.beginPath();
      context.moveTo(x, padding.top + chartHeight);
      context.lineTo(x, padding.top + chartHeight + 4);
      context.stroke();
      context.textAlign = tick.align;
      context.fillText(tick.label, x, height - 7);
    });

    if (hoveredIndex !== null) {
      const activeIndex = Math.min(path.length - 1, Math.max(0, hoveredIndex));
      const pointerX = padding.left + (activeIndex / (path.length - 1)) * chartWidth;
      context.save();
      context.strokeStyle = "#94a3b8";
      context.lineWidth = 1;
      context.setLineDash([3, 3]);
      context.beginPath();
      context.moveTo(pointerX, padding.top);
      context.lineTo(pointerX, padding.top + chartHeight);
      context.stroke();
      context.setLineDash([]);

      series.forEach((item) => {
        const value = item.values[activeIndex];
        const pointerY = padding.top + ((high - value) / (high - low)) * chartHeight;
        context.beginPath();
        context.arc(pointerX, pointerY, 3.5, 0, Math.PI * 2);
        context.fillStyle = item.color;
        context.fill();
        context.strokeStyle = "#ffffff";
        context.lineWidth = 1.5;
        context.stroke();
      });

      const tooltipWidth = 206;
      const tooltipHeight = 34 + series.length * 20;
      const tooltipX = pointerX + tooltipWidth + 12 <= width - padding.right
        ? pointerX + 12
        : Math.max(padding.left, pointerX - tooltipWidth - 12);
      const tooltipY = padding.top + 8;
      context.fillStyle = "rgba(15, 23, 42, 0.94)";
      context.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
      context.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      context.fillStyle = "#f8fafc";
      context.textAlign = "left";
      context.fillText(path[activeIndex].Date, tooltipX + 12, tooltipY + 20);
      context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      series.forEach((item, index) => {
        const rowY = tooltipY + 42 + index * 20;
        context.fillStyle = item.color;
        context.fillRect(tooltipX + 12, rowY - 8, 8, 8);
        context.fillStyle = "#e2e8f0";
        context.textAlign = "left";
        context.fillText(item.name, tooltipX + 27, rowY);
        context.textAlign = "right";
        const displayValue = mode === "drawdown"
          ? `${(item.values[activeIndex] * 100).toFixed(2)}%`
          : item.values[activeIndex].toFixed(4);
        context.fillText(displayValue, tooltipX + tooltipWidth - 12, rowY);
      });
      context.restore();
    }
  }, [path, mode, hoveredIndex]);

  const updateHoveredIndex = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (path.length < 2) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left;
    const chartWidth = bounds.width - lineChartPadding.left - lineChartPadding.right;
    if (pointerX < lineChartPadding.left || pointerX > bounds.width - lineChartPadding.right) {
      setHoveredIndex(null);
      return;
    }
    const index = Math.round(((pointerX - lineChartPadding.left) / chartWidth) * (path.length - 1));
    setHoveredIndex(Math.min(path.length - 1, Math.max(0, index)));
  };

  const moveHoveredIndex = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    setHoveredIndex((currentIndex) => {
      const startingIndex = currentIndex ?? (direction > 0 ? 0 : path.length - 1);
      return Math.min(path.length - 1, Math.max(0, startingIndex + direction));
    });
  };

  return <canvas
    ref={canvasRef}
    className="chart-canvas interactive-chart"
    role="img"
    tabIndex={0}
    aria-label={`${mode === "value" ? "Historical net value comparison" : "Drawdown comparison"}. Move the pointer or use the left and right arrow keys to inspect daily values.`}
    onPointerMove={updateHoveredIndex}
    onPointerLeave={() => setHoveredIndex(null)}
    onFocus={() => setHoveredIndex((currentIndex) => currentIndex ?? path.length - 1)}
    onBlur={() => setHoveredIndex(null)}
    onKeyDown={moveHoveredIndex}
  />;
}

function ReturnDistribution({ path }: { path: PathRow[] }) {
  const canvasRef = useCanvas((context, width, height) => {
    if (path.length < 2) return;
    const padding = distributionChartPadding;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const allReturns = focusedStrategyDefinitions.flatMap((definition) => path.map((item) => Number(item[definition.returnKey]) * 100));
    const rawMinimum = Math.min(...allReturns);
    const rawMaximum = Math.max(...allReturns);
    const maximumAbsoluteReturn = Math.max(Math.abs(rawMinimum), Math.abs(rawMaximum), 0.01);
    const axisStep = calculateNiceAxisStep(maximumAbsoluteReturn * 2, 6);
    const axisLimit = Math.ceil(maximumAbsoluteReturn / axisStep) * axisStep;
    const minimum = -axisLimit;
    const maximum = axisLimit;
    const xTickValues = Array.from(
      { length: Math.round((maximum - minimum) / axisStep) + 1 },
      (_, index) => minimum + index * axisStep,
    );
    const bins = 36;
    const binWidth = (maximum - minimum) / bins;
    const counts = focusedStrategyDefinitions.map((definition) => {
      const values = Array.from({ length: bins }, () => 0);
      path.forEach((item) => {
        const value = Number(item[definition.returnKey]) * 100;
        const index = Math.min(bins - 1, Math.max(0, Math.floor((value - minimum) / binWidth)));
        values[index] += 1;
      });
      return { ...definition, values };
    });
    const maximumCount = Math.max(...counts.flatMap((item) => item.values));
    context.strokeStyle = "#e2e8f0";
    context.fillStyle = "#64748b";
    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    for (let line = 0; line < 4; line += 1) {
      const y = padding.top + (chartHeight * line) / 3;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
    }
    xTickValues.forEach((value) => {
      const x = padding.left + ((value - minimum) / (maximum - minimum)) * chartWidth;
      context.strokeStyle = "#edf0f3";
      context.beginPath();
      context.moveTo(x, padding.top);
      context.lineTo(x, padding.top + chartHeight);
      context.stroke();
    });
    counts.forEach((item) => {
      context.fillStyle = `${item.color}66`;
      item.values.forEach((count, index) => {
        const x = padding.left + (index / bins) * chartWidth;
        const barHeight = (count / maximumCount) * chartHeight;
        context.fillRect(x, padding.top + chartHeight - barHeight, Math.max(1, chartWidth / bins - 1), barHeight);
      });
    });
    const zeroX = padding.left + ((0 - minimum) / (maximum - minimum)) * chartWidth;
    context.strokeStyle = "#475569";
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(zeroX, padding.top);
    context.lineTo(zeroX, padding.top + chartHeight);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#64748b";
    xTickValues.forEach((value, index) => {
      const x = padding.left + ((value - minimum) / (maximum - minimum)) * chartWidth;
      context.textAlign = index === 0 ? "left" : index === xTickValues.length - 1 ? "right" : "center";
      context.fillText(formatPercentageAxisTick(value), x, height - 8);
    });
  }, [path]);

  return <canvas
    ref={canvasRef}
    className="distribution-canvas"
    role="img"
    aria-label="Daily return distribution comparison on a symmetric percentage scale centered at zero."
  />;
}

function ChartLegend({ definitions = strategyDefinitions }: { definitions?: typeof strategyDefinitions }) {
  return <div className="chart-legend">{definitions.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.name}</span>)}</div>;
}

export default function Home() {
  const [estimationMonths, setEstimationMonths] = useState(24);
  const [AMADWindow, setAMADWindow] = useState(126);
  const [transactionCost, setTransactionCost] = useState(0);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [path, setPath] = useState<PathRow[]>([]);
  const [allocation, setAllocation] = useState<AllocationRow[]>([]);
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [runId, setRunId] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [isReporting, setIsReporting] = useState(false);
  const [status, setStatus] = useState("Choose the model settings, then run the backtest.");
  const [error, setError] = useState("");

  const performance = useMemo(() => buildPerformanceRows(path, metrics), [path, metrics]);
  const topAllocation = allocation.slice().sort((a, b) => Number(b.Weight) - Number(a.Weight)).slice(0, 25);
  const AMADReturns = path.map((item) => Number(item.AMAD_NRP_Return));
  const bestDay = AMADReturns.length ? Math.max(...AMADReturns) : 0;
  const worstDay = AMADReturns.length ? Math.min(...AMADReturns) : 0;
  const averageMaximumWeight = Number(metrics.find((item) => item.ReturnPreprocessing === "AMAD")?.AverageMaximumWeight ?? 0);

  const runBacktest = async () => {
    setIsRunning(true);
    setError("");
    setStatus("Running AMAD-NRP and Raw NRP with the selected settings. This usually takes several seconds.");
    try {
      const response = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          estimation_months: estimationMonths,
          AMAD_window: AMADWindow,
          AMAD_threshold: 3,
          transaction_cost_bps: transactionCost,
        }),
      });
      const payload = await response.json() as BacktestResponse | { error: string };
      if (!response.ok || "error" in payload) throw new Error("error" in payload ? payload.error : "Backtest failed.");
      setMetrics(payload.metrics);
      setPath(payload.path);
      setAllocation(payload.allocation);
      setMetadata(payload.metadata);
      setRunId(payload.run_id);
      setStatus(`Live backtest completed in ${payload.metadata.elapsed_seconds.toFixed(1)} seconds.`);
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "Backtest failed.";
      setError(message.includes("fetch") ? "The local backtest service is unavailable. Restart the Dashboard with `npm run dev`." : message);
      setStatus("The backtest did not complete.");
    } finally {
      setIsRunning(false);
    }
  };

  const generateReport = async () => {
    if (!runId) return;
    setIsReporting(true);
    setError("");
    try {
      const response = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
      });
      if (!response.ok) {
        const payload = await response.json() as { error?: string };
        throw new Error(payload.error || "Report generation failed.");
      }
      const reportBlob = await response.blob();
      const reportUrl = URL.createObjectURL(reportBlob);
      const link = document.createElement("a");
      link.href = reportUrl;
      link.download = `AMAD-NRP-report-${metadata?.latest_effective_date ?? "latest"}.pdf`;
      link.click();
      URL.revokeObjectURL(reportUrl);
      setStatus("PDF report generated from the active backtest and downloaded.");
    } catch (reportError) {
      setError(reportError instanceof Error ? reportError.message : "Report generation failed.");
    } finally {
      setIsReporting(false);
    }
  };

  return (
    <main>
      <header className="site-header">
        <div><span className="product-label">AMAD-NRP RESEARCH</span><strong>Strategy Dashboard</strong></div>
        <button className="primary-button" onClick={generateReport} disabled={!runId || isRunning || isReporting}>
          {isReporting ? "Generating PDF…" : "Generate report"}
        </button>
      </header>

      <div className="dashboard-shell">
        <header className="page-heading">
          <div>
            <h1>AMAD-Enhanced Network Risk Parity for S&amp;P 100</h1>
            <p>Point-in-time constituent membership · Monthly rebalance · Daily out-of-sample evaluation</p>
          </div>
          {metadata ? <span className="verified-badge">Live result</span> : null}
        </header>

        <div className="dashboard-layout">
          <aside className="settings-panel">
            <section>
              <h2>Model settings</h2>
              <label className="control-label">Estimation window
                <select value={estimationMonths} onChange={(event) => setEstimationMonths(Number(event.target.value))} disabled={isRunning}>
                  <option value={12}>12 months</option><option value={24}>24 months (default)</option>
                </select>
              </label>
              <label className="control-label">AMAD window
                <select value={AMADWindow} onChange={(event) => setAMADWindow(Number(event.target.value))} disabled={isRunning}>
                  <option value={64}>64 days</option><option value={126}>126 days (default)</option>
                </select>
              </label>
              <label className="control-label">Transaction cost
                <div className="input-suffix"><input type="number" min="0" max="100" step="1" value={transactionCost} onChange={(event) => setTransactionCost(Number(event.target.value))} disabled={isRunning} /><span>bps</span></div>
              </label>
              <div className="fixed-setting"><span>AMAD threshold</span><strong>3.0 MAD</strong></div>
              <button className="run-button" onClick={runBacktest} disabled={isRunning}>{isRunning ? "Running backtest…" : "Run backtest"}</button>
            </section>
            <section>
              <h2>Model specification</h2>
              <ul>
                <li>Point-in-time S&amp;P 100 universe</li>
                <li>AMAD-preprocessed estimation returns</li>
                <li>Weighted correlation-distance MST</li>
                <li>Ordinary inverse-centrality allocation</li>
                <li>Long-only, fully invested, monthly rebalance</li>
              </ul>
            </section>
            {metadata ? <section className="run-details"><h2>Run details</h2><dl className="settings-list">
              <div><dt>Sample</dt><dd>{metadata.out_of_sample_start}<br />{metadata.out_of_sample_end}</dd></div>
              <div><dt>Latest rebalance</dt><dd>{metadata.latest_effective_date}</dd></div>
              <div><dt>Eligible assets</dt><dd>{metadata.latest_eligible_assets}</dd></div>
            </dl></section> : null}
          </aside>

          <div className="dashboard-content">
            <section className="card description-card"><div className="card-header"><h2>Strategy description</h2></div><div className="card-body strategy-copy">
              <p>AMAD-NRP combines Adaptive Market Anomaly Detection with Network Risk Parity for the point-in-time S&amp;P 100. AMAD reduces the influence of unusual short-lived moves in estimation returns before the dependence network is estimated. Stocks with greater network centrality receive smaller allocations; less redundant stocks receive larger allocations.</p>
              <p>Raw NRP uses the same universe, network construction, allocation rule, and holding returns without AMAD preprocessing. The adjusted-price S&amp;P 100 series provides the benchmark comparison.</p>
            </div></section>

            <div className={`status-bar ${error ? "status-error" : ""}`}><strong>{isRunning ? "Backtest in progress" : metadata ? "Active live result" : "Ready"}</strong><span>{error || status}</span></div>

            {!metadata ? <section className="card empty-state"><span className={isRunning ? "loader" : "empty-marker"} /><h2>{isRunning ? "Running the selected study" : "No backtest has been run in this session"}</h2><p>{isRunning ? "The charts and tables will update when both strategy variants and the benchmark are complete." : "Set the estimation window, AMAD window, and transaction cost, then select Run backtest."}</p></section> : <>
              <section className="card chart-card"><div className="card-header"><h2>Historical net value</h2><ChartLegend /></div><div className="card-body"><MultiLineChart path={path} mode="value" /></div><div className="card-footer">Growth of $1 after the selected transaction-cost assumption.</div></section>

              <section className="card chart-card"><div className="card-header"><h2>Drawdown comparison</h2><ChartLegend definitions={focusedStrategyDefinitions} /></div><div className="card-body"><MultiLineChart path={path} mode="drawdown" /></div><div className="card-footer">AMAD-NRP and S&amp;P 100 drawdowns from each portfolio&apos;s running peak.</div></section>

              <section className="card metrics-card"><div className="card-header"><h2>Performance metrics</h2></div><div className="table-wrap"><table><thead><tr><th>Strategy</th><th>Months</th><th>Total return</th><th>Annualized return</th><th>Annualized volatility</th><th>Sharpe</th><th>Sortino</th><th>Calmar</th><th>Maximum drawdown</th><th>Average turnover</th></tr></thead><tbody>{performance.map((item) => <tr key={item.key} className={item.key === "AMAD" ? "highlight-row" : ""}><th>{item.name}</th><td>{item.months}</td><td>{formatPercent(item.totalReturn)}</td><td>{formatPercent(item.annualizedReturn)}</td><td>{formatPercent(item.annualizedVolatility)}</td><td>{formatRatio(item.sharpe)}</td><td>{formatRatio(item.sortino)}</td><td>{formatRatio(item.calmar)}</td><td>{formatPercent(item.maximumDrawdown)}</td><td>{formatPercent(item.turnover)}</td></tr>)}</tbody></table></div><div className="card-footer">Return and risk metrics use daily out-of-sample net returns; turnover is one-way per monthly rebalance.</div></section>

              <div className="two-card-grid">
                <section className="card allocation-card"><div className="card-header"><h2>Latest allocation</h2><span>{metadata.latest_effective_date}</span></div><div className="card-body allocation-list">{topAllocation.map((item, index) => <div className="allocation-row" key={item.Asset}><span>{index + 1}</span><strong>{item.Asset}</strong><div><i style={{ width: `${Math.min(Number(item.Weight) * 300, 100)}%` }} /></div><b>{formatPercent(Number(item.Weight))}</b></div>)}</div><div className="card-footer">Top 25 positions shown; the complete allocation is used in the backtest.</div></section>

                <div className="analysis-stack">
                  <section className="card diagnostics-card"><div className="card-header"><h2>AMAD-NRP diagnostics</h2></div><div className="diagnostic-list"><div><span>Average maximum position</span><strong>{formatPercent(averageMaximumWeight)}</strong><small>Mean largest single-stock weight across rebalances</small></div><div><span>Best day return</span><strong className="positive">{bestDay >= 0 ? "+" : ""}{formatPercent(bestDay)}</strong><small>Highest AMAD-NRP daily net return</small></div><div><span>Worst day return</span><strong className="negative">{formatPercent(worstDay)}</strong><small>Lowest AMAD-NRP daily net return</small></div></div><div className="card-footer">Diagnostics are calculated from the active live run.</div></section>

                  <section className="card distribution-card"><div className="card-header"><h2>Daily return distribution</h2><ChartLegend definitions={focusedStrategyDefinitions} /></div><div className="card-body"><ReturnDistribution path={path} /></div><div className="card-footer">AMAD-NRP and S&amp;P 100 daily net returns on a symmetric scale centered at zero.</div></section>
                </div>
              </div>

              <section className="method-notes"><h2>Method notes</h2><p>For each asset, a trailing median and MAD—including the current observation—flag estimation returns where |r − median| exceeds τ × MAD. Only those deviations receive continuous logarithmic dampening. Holding-period returns remain raw.</p><p>{metadata.membership_rule} {metadata.history_rule} Synthetic backfill: {metadata.synthetic_backfill ? "used" : "not used"}.</p></section>
            </>}
          </div>
        </div>

        <footer><span>AMAD-NRP Research Dashboard</span><span>Historical results do not predict future performance.</span></footer>
      </div>
    </main>
  );
}
