"""Shared result types produced by drivers and samplers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["Trajectory"]


@dataclass(frozen=True)
class Trajectory:
    """Outcome of a driver/sampler run.

    Backend-agnostic handle to the produced trajectory and its bookkeeping.
    Frames live on disk (``path``); ``n_frames`` and ``metadata`` summarize the
    run for the stage-summary / manifest layer.
    """

    path: Path | None = None
    n_frames: int = 0
    exit_code: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
