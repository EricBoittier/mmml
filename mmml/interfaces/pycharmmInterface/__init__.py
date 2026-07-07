"""PyCHARMM / CHARMM integration for MM/ML calculators and workflows."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["mlpot"]


def __getattr__(name: str) -> Any:
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
