"""Local HTTP API for live dashboard backtests and PDF report downloads."""

from __future__ import annotations

import json
import re
import threading

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import polars as pl

from run_backtest import run_backtest_study_NRP_SP100_point_in_time_QWIM

from amad_nrp.report import generate_research_report


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = _PROJECT_ROOT / "outputs" / "runs"
_REPORT_ROOT = _PROJECT_ROOT / "output" / "pdf"
_RUN_ID_PATTERN = re.compile(r"^run_\d{8}T\d{6}Z_months\d+$")
_BACKTEST_LOCK = threading.Lock()


def _serialize_table(file_path: Path) -> list[dict[str, Any]]:
    """Load one run table and return JSON-compatible records."""
    return pl.read_csv(file_path, try_parse_dates=True).to_dicts()


def _load_dashboard_payload(run_directory: Path) -> dict[str, Any]:
    """Load the browser-facing result contract from one completed run."""
    metadata = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    return {
        "run_id": run_directory.name,
        "metadata": metadata,
        "metrics": _serialize_table(run_directory / "Strategy_comparison_metrics.csv"),
        "path": _serialize_table(run_directory / "Strategy_comparison_daily_path.csv"),
        "allocation": _serialize_table(run_directory / "AMAD_NRP_latest_allocation.csv"),
    }


def _validate_backtest_request(payload: dict[str, Any]) -> dict[str, float | int]:
    """Validate the small set of user-selectable research settings."""
    estimation_months = int(payload.get("estimation_months", 24))
    AMAD_window = int(payload.get("AMAD_window", 126))
    AMAD_threshold = float(payload.get("AMAD_threshold", 3.0))
    transaction_cost_bps = float(payload.get("transaction_cost_bps", 0.0))
    if estimation_months not in {12, 24}:
        raise ValueError("Estimation window must be 12 or 24 months.")
    if AMAD_window not in {64, 126}:
        raise ValueError("AMAD window must be 64 or 126 days.")
    if not 1.0 <= AMAD_threshold <= 10.0:
        raise ValueError("AMAD threshold must be between 1 and 10 MAD.")
    if not 0.0 <= transaction_cost_bps <= 100.0:
        raise ValueError("Transaction cost must be between 0 and 100 basis points.")
    return {
        "estimation_months": estimation_months,
        "AMAD_window": AMAD_window,
        "AMAD_threshold": AMAD_threshold,
        "transaction_cost_bps": transaction_cost_bps,
    }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve live backtest and report endpoints to the local Dashboard."""

    server_version = "AMADNRPDashboard/1.0"

    def _send_headers(self, *, status: HTTPStatus, content_type: str, content_length: int) -> None:
        """Write common API response headers."""
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _send_json(self, *, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Serialize and send one JSON response."""
        body = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
        self._send_headers(status=status, content_type="application/json; charset=utf-8", content_length=len(body))
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        """Read a bounded JSON request body."""
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 16_384:
            raise ValueError("A valid JSON request body is required.")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Answer browser preflight requests."""
        self._send_headers(status=HTTPStatus.NO_CONTENT, content_type="text/plain", content_length=0)

    def do_GET(self) -> None:  # noqa: N802
        """Return service health information."""
        if self.path == "/api/health":
            self._send_json(status=HTTPStatus.OK, payload={"status": "ready"})
            return
        self._send_json(status=HTTPStatus.NOT_FOUND, payload={"error": "Endpoint not found."})

    def do_POST(self) -> None:  # noqa: N802
        """Run a live backtest or generate its PDF report."""
        try:
            payload = self._read_json()
            if self.path == "/api/backtest":
                self._handle_backtest(payload)
                return
            if self.path == "/api/report":
                self._handle_report(payload)
                return
            self._send_json(status=HTTPStatus.NOT_FOUND, payload={"error": "Endpoint not found."})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(status=HTTPStatus.BAD_REQUEST, payload={"error": str(error)})
        except Exception as error:  # noqa: BLE001
            self._send_json(status=HTTPStatus.INTERNAL_SERVER_ERROR, payload={"error": str(error)})

    def _handle_backtest(self, payload: dict[str, Any]) -> None:
        """Execute one user-selected backtest and return its full result."""
        settings = _validate_backtest_request(payload)
        if not _BACKTEST_LOCK.acquire(blocking=False):
            self._send_json(
                status=HTTPStatus.CONFLICT,
                payload={"error": "Another backtest is already running. Please wait for it to finish."},
            )
            return
        try:
            run_directory = run_backtest_study_NRP_SP100_point_in_time_QWIM(
                estimation_months=int(settings["estimation_months"]),
                AMAD_window=int(settings["AMAD_window"]),
                AMAD_threshold=float(settings["AMAD_threshold"]),
                transaction_cost_bps=float(settings["transaction_cost_bps"]),
                output_root=_OUTPUT_ROOT,
            )
            self._send_json(status=HTTPStatus.OK, payload=_load_dashboard_payload(run_directory))
        finally:
            _BACKTEST_LOCK.release()

    def _handle_report(self, payload: dict[str, Any]) -> None:
        """Compile and download a Typst report for one completed live run."""
        run_id = str(payload.get("run_id", ""))
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("A valid completed run is required before generating a report.")
        run_directory = _OUTPUT_ROOT / run_id
        if not run_directory.is_dir():
            raise ValueError("The selected backtest run is no longer available.")
        report_path = generate_research_report(
            run_directory=run_directory,
            output_directory=_REPORT_ROOT,
        )
        body = report_path.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", 'attachment; filename="AMAD-NRP-research-report.pdf"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:3000")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *arguments: Any) -> None:
        """Keep local API logs concise."""
        print(f"[dashboard-api] {self.address_string()} {format % arguments}")


def main() -> None:
    """Start the local Dashboard API."""
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardRequestHandler)
    print("AMAD-NRP backtest API ready at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
