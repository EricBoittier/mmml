"""Optional MLpot callback timing (CHARMM vs ML wall time)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional


def mlpot_profiling_enabled() -> bool:
    return (os.environ.get("MMML_MLPOT_PROFILE") or "").strip().lower() in (
        "1",
        "yes",
        "true",
    )


@dataclass
class MlpotProfileStats:
    ml_calls: int = 0
    ml_seconds: float = 0.0
    charmm_gap_seconds: float = 0.0
    _last_callback_end: Optional[float] = field(default=None, repr=False)

    def record_ml(self, elapsed_s: float) -> None:
        self.ml_calls += 1
        self.ml_seconds += elapsed_s
        self._last_callback_end = time.perf_counter()

    def record_charmm_gap(self) -> None:
        if self._last_callback_end is None:
            return
        self.charmm_gap_seconds += time.perf_counter() - self._last_callback_end

    def summary_line(self) -> str:
        total = self.ml_seconds + self.charmm_gap_seconds
        if total <= 0:
            return "MLpot profile: no samples"
        ml_pct = 100.0 * self.ml_seconds / total
        return (
            f"MLpot profile: {self.ml_calls} ML callbacks, "
            f"ML={self.ml_seconds:.3f}s ({ml_pct:.1f}%), "
            f"CHARMM+overhead={self.charmm_gap_seconds:.3f}s"
        )


_GLOBAL_STATS = MlpotProfileStats()


def get_mlpot_profile_stats() -> MlpotProfileStats:
    return _GLOBAL_STATS


def reset_mlpot_profile_stats() -> None:
    global _GLOBAL_STATS
    _GLOBAL_STATS = MlpotProfileStats()


def maybe_log_mlpot_profile(*, quiet: bool = False) -> None:
    if not mlpot_profiling_enabled() or quiet:
        return
    print(get_mlpot_profile_stats().summary_line(), flush=True)
