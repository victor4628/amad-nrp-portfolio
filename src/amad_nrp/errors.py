"""Small standalone error and logging boundary for the extracted engine."""

from __future__ import annotations

import logging
from typing import Any


class AMADNRPError(Exception):
    """Base error carrying optional structured context."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context = context


class Exception_Data_Validation_Error(AMADNRPError):
    """Raised when an input table is incomplete or malformed."""


class Exception_Validation_Input(AMADNRPError):
    """Raised when a strategy parameter is outside its valid range."""


class Exception_Calculation(AMADNRPError):
    """Raised when a numerical result cannot be produced safely."""


class Exception_Not_Found(AMADNRPError):
    """Raised when a required local research artifact is missing."""


def get_logger(*, name: str) -> logging.Logger:
    """Return a conventional library logger without configuring global output."""
    return logging.getLogger(name)
