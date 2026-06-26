"""Optional MLpot callback timing (CHARMM vs ML wall time)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
    filename: str = "profile_git_metadata.json",
) -> Path:
    """Write a JSON sidecar with git metadata for profiling output."""
    override = os.environ.get("MMML_PROFILE_GIT_METADATA")
    path = Path(override) if override else Path(output_dir or ".") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collect_profile_git_metadata(argv=argv), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
