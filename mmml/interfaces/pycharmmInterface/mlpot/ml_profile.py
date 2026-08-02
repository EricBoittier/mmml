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
    _last_callback_end: Optional[float] = field(default=None, repr=False)

    def record_ml(self, elapsed_s: float) -> None:
        self.ml_calls += 1
        self.ml_seconds += elapsed_s
        self._last_callback_end = time.perf_counter()

    def record_charmm_gap(self) -> None:
        if self._last_callback_end is None:
            return
        self.charmm_gap_seconds += time.perf_counter() - self._last_callback_end

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
            "summary": self.summary_line(),
        }


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
