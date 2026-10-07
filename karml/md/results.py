"""Shared result types produced by drivers and samplers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["Trajectory", "energy_drift_metrics"]


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


def energy_drift_metrics(energies: Any) -> dict[str, float]:
    """Fluctuation and tendency of an energy trace -- a better drift
    diagnostic than a bare endpoint delta (``E[-1] - E[0]``).

    An endpoint delta can be large purely from single-frame noise in a trace
    that is essentially flat on average, or small while the trace trends
    steadily in one direction between two coincidentally-close endpoints.
    Reports both:

    - ``energy_fluctuation_std_ev``: std over the whole trace (noise level).
    - ``energy_trend_ev_per_frame``: slope of a linear least-squares fit
      against frame index (the systematic tendency, robust to which two
      frames the endpoint delta happens to compare).
    - ``energy_trend_total_ev``: the trend line's implied total change
      (``slope * (n_frames - 1)``) -- comparable in scale to the old
      endpoint delta, but reflecting the fitted tendency rather than one
      noisy pair of samples.
    """
    values = np.asarray(energies, dtype=float)
    n = values.shape[0]
    if n < 2:
        return {
            "energy_fluctuation_std_ev": float(values.std()) if n else 0.0,
            "energy_trend_ev_per_frame": 0.0,
            "energy_trend_total_ev": 0.0,
        }
    frames = np.arange(n, dtype=float)
    slope, _intercept = np.polyfit(frames, values, 1)
    return {
        "energy_fluctuation_std_ev": float(values.std()),
        "energy_trend_ev_per_frame": float(slope),
        "energy_trend_total_ev": float(slope * (n - 1)),
    }
