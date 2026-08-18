"""Validate the Dashboard's supported backtest settings."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from dashboard_api import _validate_backtest_request  # noqa: E402


def test_dashboard_uses_the_requested_defaults() -> None:
    settings = _validate_backtest_request({})

    assert settings["estimation_months"] == 24
    assert settings["AMAD_window"] == 126


def test_dashboard_accepts_the_alternative_windows() -> None:
    settings = _validate_backtest_request(
        {
            "estimation_months": 12,
            "AMAD_window": 64,
        },
    )

    assert settings["estimation_months"] == 12
    assert settings["AMAD_window"] == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("estimation_months", 36, "12 or 24 months"),
        ("AMAD_window", 63, "64 or 126 days"),
    ],
)
def test_dashboard_rejects_removed_windows(field: str, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_backtest_request({field: value})
