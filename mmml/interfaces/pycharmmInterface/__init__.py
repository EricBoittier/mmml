"""PyCHARMM / CHARMM integration for MM/ML calculators and workflows."""

from __future__ import annotations

from typing import Any

__all__ = ["mlpot"]


def __getattr__(name: str) -> Any:
    if name == "mlpot":
        from mmml.interfaces.pycharmmInterface import mlpot as _mlpot

        return _mlpot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
