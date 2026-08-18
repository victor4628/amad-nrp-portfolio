# AMAD-Enhanced Network Risk Parity for S&P 100

An independent, lightweight implementation of dynamic S&P 100 AMAD Network Risk Parity. The repository contains a reproducible Python research engine and a Dashboard that runs the selected backtest locally on demand.

## Included

- Point-in-time S&P 100 constituent selection to avoid using today’s membership retrospectively.
- A configurable rolling AMAD transformation applied only to estimation returns.
- Monthly realized covariance, a correlation-distance MST, and inverse-centrality weights.
- Raw NRP and an adjusted-price S&P 100 benchmark for controlled comparisons.
- A responsive Dashboard with live parameterized backtests, comparative analytics, allocation diagnostics, and a downloadable Typst PDF research report.

## Run locally

Requirements:

- Node.js 22.13 or newer
- Python 3.11 or newer
- `uv`
- Typst on `PATH` for PDF report generation

Install the dependencies and start the Dashboard:

```bash
npm install
uv sync
npm run dev
```

On macOS, Typst can be installed with `brew install typst`. Then open [http://localhost:3000](http://localhost:3000). The `npm run dev` command starts both the web interface and the local Python backtest service. Press `Ctrl+C` in the terminal to stop both.

Research engine and tests:

```bash
uv sync
uv run pytest
uv run python scripts/run_backtest.py
```

Each Dashboard run writes a new immutable result to `outputs/runs/`. Reports generated from the active result are written under `output/pdf/`; both directories are local and ignored by Git.

## Research boundary

The default out-of-sample period is July 3, 2023 through July 28, 2026. The Dashboard defaults to a 24-month estimation window, a 126-day AMAD window, and a threshold of 3. Historical backtests are not investment advice.
