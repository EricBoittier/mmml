"""Optional MLpot / ASE calculator timing (CHARMM vs ML wall time)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def mlpot_profiling_enabled() -> bool:
    return (os.environ.get("MMML_MLPOT_PROFILE") or "").strip().lower() in (
        "1",
        "yes",
        "true",
    )


def enable_mlpot_profiling() -> None:
    """Turn on lightweight MLpot/ASE timing and JAX compile timers."""
    os.environ["MMML_MLPOT_PROFILE"] = "1"
    os.environ["MMML_JAX_COMPILE_TIMERS"] = "1"


@dataclass
class MlpotProfileStats:
    ml_calls: int = 0
    ml_seconds: float = 0.0
    charmm_gap_seconds: float = 0.0
    calculate_calls: int = 0
    calculate_seconds: float = 0.0
    chunk_apply_calls: int = 0
    chunk_apply_seconds: float = 0.0
    last_n_gpus: int = 0
    last_n_chunks: int = 0
    last_chunk_size: int = 0
    last_effective_batch_size: int = 0
    max_n_gpus: int = 0
    mm_pair_calls: int = 0
    mm_pair_rebuilds: int = 0
    mm_pair_gpu_rebuilds: int = 0
    _last_callback_end: Optional[float] = field(default=None, repr=False)
    # Per-call samples for steady-state statistics (first ``warmup_calls`` skipped).
    warmup_calls: int = 100
    ml_ms_samples: list = field(default_factory=list, repr=False)
    gap_ms_samples: list = field(default_factory=list, repr=False)
    n_active_samples: list = field(default_factory=list, repr=False)
    chunk_budget_samples: list = field(default_factory=list, repr=False)
    max_active_dimers: int = 0
    chunk_size: int = 0

    def record_ml(self, elapsed_s: float) -> None:
        self.ml_calls += 1
        self.ml_seconds += elapsed_s
        self.ml_ms_samples.append(1000.0 * float(elapsed_s))
        if _SUMMARY_DIR is not None and self.ml_calls % 500 == 0:
            # CHARMM can end the process without running atexit hooks.
            write_mlpot_profile_summary(_SUMMARY_DIR)
        self._last_callback_end = time.perf_counter()

    def record_charmm_gap(self) -> None:
        if self._last_callback_end is None:
            return
        gap = time.perf_counter() - self._last_callback_end
        self.charmm_gap_seconds += gap
        self.gap_ms_samples.append(1000.0 * gap)

    def record_active_dimers(
        self, n_active: int, *, chunk_budget: int, chunk_size: int, max_active_dimers: int
    ) -> None:
        """Sparse ML dimers in range this step and the PhysNet chunks the step ran."""
        self.n_active_samples.append(int(n_active))
        self.chunk_budget_samples.append(int(chunk_budget))
        self.chunk_size = int(chunk_size)
        self.max_active_dimers = int(max_active_dimers)

    def steady_state(self) -> dict[str, Any]:
        """Median / mean / p90 per call after the first ``warmup_calls`` (JIT, budget settling)."""
        import numpy as np

        def _stats(xs: list) -> Optional[dict[str, float]]:
            a = np.asarray(xs[self.warmup_calls :], dtype=float)
            if a.size == 0:
                return None
            return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
                    "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}

        ml, gap = _stats(self.ml_ms_samples), _stats(self.gap_ms_samples)
        n = min(len(self.ml_ms_samples), len(self.gap_ms_samples) + 1)
        step = _stats([m + g for m, g in zip(self.ml_ms_samples[1:n], self.gap_ms_samples[: n - 1])])
        return {
            "warmup_calls": self.warmup_calls,
            "ml_callback_ms": ml,
            "charmm_gap_ms": gap,
            "step_ms": step,
            "n_active_dimers": _stats(self.n_active_samples),
            "chunk_budget": _stats(self.chunk_budget_samples),
            "chunk_size": self.chunk_size,
            "max_active_dimers": self.max_active_dimers,
        }

    def record_calculate(self, elapsed_s: float) -> None:
        """Wall time for one ASE ``Calculator.calculate`` (includes GPU sync)."""
        self.calculate_calls += 1
        self.calculate_seconds += float(elapsed_s)

    def record_chunk_apply(
        self,
        elapsed_s: float,
        *,
        n_gpus: int,
        n_chunks: int,
        chunk_size: int,
        effective_batch_size: int,
    ) -> None:
        """Wall time for PhysNet chunked / multi-GPU apply (includes GPU sync)."""
        self.chunk_apply_calls += 1
        self.chunk_apply_seconds += float(elapsed_s)
        self.last_n_gpus = int(n_gpus)
        self.last_n_chunks = int(n_chunks)
        self.last_chunk_size = int(chunk_size)
        self.last_effective_batch_size = int(effective_batch_size)
        self.max_n_gpus = max(self.max_n_gpus, int(n_gpus))

    def record_mm_pair_stats(self, stats: dict[str, Any]) -> None:
        """Latest cumulative MM pair-list counters (``update_mm_pairs.get_stats()``)."""
        self.mm_pair_calls = int(stats.get("calls", 0))
        self.mm_pair_rebuilds = int(stats.get("updates", 0))
        self.mm_pair_gpu_rebuilds = int(stats.get("gpu_rebuilds", 0))

    def summary_line(self) -> str:
        parts: list[str] = []
        total_cb = self.ml_seconds + self.charmm_gap_seconds
        if total_cb > 0:
            ml_pct = 100.0 * self.ml_seconds / total_cb
            parts.append(
                f"{self.ml_calls} ML callbacks, "
                f"ML={self.ml_seconds:.3f}s ({ml_pct:.1f}%), "
                f"CHARMM+overhead={self.charmm_gap_seconds:.3f}s"
            )
        if self.calculate_calls > 0:
            mean_ms = 1000.0 * self.calculate_seconds / self.calculate_calls
            parts.append(
                f"{self.calculate_calls} ASE calculate, "
                f"total={self.calculate_seconds:.3f}s "
                f"(mean={mean_ms:.2f} ms/call)"
            )
        if self.chunk_apply_calls > 0:
            mean_ms = 1000.0 * self.chunk_apply_seconds / self.chunk_apply_calls
            parts.append(
                f"{self.chunk_apply_calls} chunk-apply, "
                f"total={self.chunk_apply_seconds:.3f}s "
                f"(mean={mean_ms:.2f} ms, last n_gpus={self.last_n_gpus}, "
                f"n_chunks={self.last_n_chunks}, chunk={self.last_chunk_size}, "
                f"batch={self.last_effective_batch_size})"
            )
        if self.mm_pair_calls > 0:
            parts.append(
                f"MM pair list: {self.mm_pair_rebuilds} rebuilds "
                f"({self.mm_pair_gpu_rebuilds} on GPU) / {self.mm_pair_calls} calls"
            )
        if not parts:
            return "MLpot profile: no samples"
        return "MLpot profile: " + "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        mean_calc_ms = (
            1000.0 * self.calculate_seconds / self.calculate_calls
            if self.calculate_calls
            else None
        )
        mean_chunk_ms = (
            1000.0 * self.chunk_apply_seconds / self.chunk_apply_calls
            if self.chunk_apply_calls
            else None
        )
        return {
            "ml_calls": self.ml_calls,
            "ml_seconds": self.ml_seconds,
            "charmm_gap_seconds": self.charmm_gap_seconds,
            "calculate_calls": self.calculate_calls,
            "calculate_seconds": self.calculate_seconds,
            "calculate_mean_ms": mean_calc_ms,
            "chunk_apply_calls": self.chunk_apply_calls,
            "chunk_apply_seconds": self.chunk_apply_seconds,
            "chunk_apply_mean_ms": mean_chunk_ms,
            "last_n_gpus": self.last_n_gpus,
            "last_n_chunks": self.last_n_chunks,
            "last_chunk_size": self.last_chunk_size,
            "last_effective_batch_size": self.last_effective_batch_size,
            "max_n_gpus": self.max_n_gpus,
            "mm_pair_calls": self.mm_pair_calls,
            "mm_pair_rebuilds": self.mm_pair_rebuilds,
            "mm_pair_gpu_rebuilds": self.mm_pair_gpu_rebuilds,
            "steady_state": self.steady_state(),
            "summary": self.summary_line(),
        }


_GLOBAL_STATS = MlpotProfileStats()
_SUMMARY_DIR: Optional[str] = None


def set_mlpot_profile_summary_dir(output_dir: str | os.PathLike[str] | None) -> None:
    """Rewrite ``mlpot_profile.json`` in ``output_dir`` every 500 ML callbacks."""
    global _SUMMARY_DIR
    _SUMMARY_DIR = None if output_dir is None else str(output_dir)


def get_mlpot_profile_stats() -> MlpotProfileStats:
    return _GLOBAL_STATS


def reset_mlpot_profile_stats() -> None:
    global _GLOBAL_STATS
    _GLOBAL_STATS = MlpotProfileStats()


def maybe_log_mlpot_profile(*, quiet: bool = False) -> None:
    if not mlpot_profiling_enabled() or quiet:
        return
    print(get_mlpot_profile_stats().summary_line(), flush=True)


def write_mlpot_profile_summary(
    output_dir: str | os.PathLike[str] | None = None,
    *,
    extra: dict[str, Any] | None = None,
    filename: str = "mlpot_profile.json",
) -> Path | None:
    """Write calculator/chunk timing JSON when profiling is enabled."""
    if not mlpot_profiling_enabled():
        return None
    path = Path(output_dir or ".") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **get_mlpot_profile_stats().to_dict(),
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_output(args: list[str], *, repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def collect_profile_git_metadata(*, argv: list[str] | None = None) -> dict[str, object]:
    """Return git/version metadata for profiling sidecars.

    Profiling runs can be launched through external wrappers, so this metadata is
    intentionally separate from cProfile output and robust to non-git installs.
    """
    root = _repo_root()
    metadata: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "argv": list(argv or []),
    }
    try:
        status_short = _git_output(["status", "--short"], repo_root=root)
        metadata.update(
            {
                "git_commit": _git_output(["rev-parse", "HEAD"], repo_root=root),
                "git_branch": _git_output(["branch", "--show-current"], repo_root=root),
                "git_describe": _git_output(
                    ["describe", "--always", "--dirty", "--tags"], repo_root=root
                ),
                "git_dirty": bool(status_short),
                "git_status_short": status_short.splitlines(),
            }
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        metadata["git_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def write_profile_git_metadata(
    output_dir: str | os.PathLike[str] | None = None,
    *,
    argv: list[str] | None = None,
    extra: dict[str, object] | None = None,
    filename: str = "profile_git_metadata.json",
) -> Path:
    """Write a JSON sidecar with git metadata for profiling output."""
    override = os.environ.get("MMML_PROFILE_GIT_METADATA")
    path = Path(override) if override else Path(output_dir or ".") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = collect_profile_git_metadata(argv=argv)
    if extra:
        metadata.update(extra)
    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
