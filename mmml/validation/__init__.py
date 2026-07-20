"""Reproducible validation and smoke-matrix utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .smoke_matrix import SmokeCase, SmokeManifest

__all__ = ["SmokeCase", "SmokeManifest", "load_smoke_manifest", "run_smoke_matrix"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import smoke_matrix

        return getattr(smoke_matrix, name)
    raise AttributeError(name)
