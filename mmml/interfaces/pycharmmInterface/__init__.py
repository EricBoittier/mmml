"""PyCHARMM / CHARMM integration for MM/ML calculators and workflows."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["mlpot"]


def __getattr__(name: str) -> Any:
    if name == "mlpot":
        return importlib.import_module(".mlpot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
