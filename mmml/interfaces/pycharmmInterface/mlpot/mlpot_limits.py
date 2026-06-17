"""Compile-time MLpot limits in ``libcharmm.so`` (``api_func.F90``)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CONSERVATIVE_LIMITS = (100, 100_000)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CHARMMSETUP").is_file():
            return parent
        if (parent / "pyproject.toml").is_file() and (parent / "mmml").is_dir():
            return parent
    return here.parents[4]


def _read_charmmsetup() -> dict[str, str]:
    out: dict[str, str] = {}
    setup = _repo_root() / "CHARMMSETUP"
    if not setup.is_file():
        return out
    for line in setup.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ("CHARMM_HOME", "CHARMM_LIB_DIR"):
            out[key] = value.strip()
    return out


def _charmm_home() -> Path | None:
    home = (os.environ.get("CHARMM_HOME") or "").strip()
    if home:
        return Path(home)
    setup_home = _read_charmmsetup().get("CHARMM_HOME")
    if setup_home:
        return Path(setup_home)
    return None


def _charmm_lib_dir() -> Path | None:
    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip()
    if lib_dir:
        return Path(lib_dir)
    setup_lib = _read_charmmsetup().get("CHARMM_LIB_DIR")
    if setup_lib:
        return Path(setup_lib)
    return None


def _api_func_f90_candidates() -> list[Path]:
    repo = _repo_root()
    home = _charmm_home()
    paths: list[Path] = []
    if home is not None:
        paths.append(home / "source" / "api" / "api_func.F90")
    paths.extend(
        [
            repo / "setup" / "api" / "api_func.F90",
            repo / "setup" / "charmm" / "source" / "api" / "api_func.F90",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _libcharmm_candidates() -> list[Path]:
    home = _charmm_home()
    lib_dir = _charmm_lib_dir()
    paths: list[Path] = []
    if lib_dir is not None:
        paths.append(lib_dir / "libcharmm.so")
    if home is not None:
        paths.extend(
            [
                home / "libcharmm.so",
                home / "lib" / "libcharmm.so",
            ]
        )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            out.append(resolved)
    return out


def _parse_api_func_limits(path: Path) -> tuple[int, int] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    ml = re.search(r"max_Nml\s*=\s*(\d+)", text)
    pr = re.search(r"max_Npr\s*=\s*(\d+)", text)
    if not ml or not pr:
        return None
    return int(ml.group(1)), int(pr.group(1))


@lru_cache(maxsize=1)
def charmm_mlpot_limits_from_source() -> tuple[int, int, Path] | None:
    """Parse ``max_Nml`` / ``max_Npr`` from the first readable ``api_func.F90``."""
    for path in _api_func_f90_candidates():
        if not path.is_file():
            continue
        parsed = _parse_api_func_limits(path)
        if parsed is not None:
            return parsed[0], parsed[1], path
    return None


@dataclass(frozen=True)
class MlpotLimitsStatus:
    max_nml: int
    max_npr: int
    source: str
    api_func_f90: Path | None = None
    libcharmm: Path | None = None
    note: str = ""

    def message(self) -> str:
        lines = [
            f"CHARMM MLpot limits: max_Nml={self.max_nml}, max_Npr={self.max_npr}",
            f"  source: {self.source}",
        ]
        if self.api_func_f90 is not None:
            lines.append(f"  api_func.F90: {self.api_func_f90}")
        if self.libcharmm is not None:
            lines.append(f"  libcharmm.so: {self.libcharmm}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def mlpot_limits_status() -> MlpotLimitsStatus:
    """Explain which limits are in effect and why."""
    env_ml = (os.environ.get("MMML_CHARMM_MLPOT_MAX_ML") or "").strip()
    env_pr = (os.environ.get("MMML_CHARMM_MLPOT_MAX_PAIRS") or "").strip()
    if env_ml and env_pr:
        return MlpotLimitsStatus(
            int(env_ml),
            int(env_pr),
            source="MMML_CHARMM_MLPOT_MAX_ML/PAIRS environment",
        )

    parsed = charmm_mlpot_limits_from_source()
    if parsed is None:
        return MlpotLimitsStatus(
            *_CONSERVATIVE_LIMITS,
            source="conservative fallback (api_func.F90 not found)",
            note=(
                "Set CHARMM_HOME or create CHARMMSETUP with CHARMM_HOME=...; "
                "expected setup/charmm/source/api/api_func.F90"
            ),
        )

    max_ml, max_pr, f90 = parsed
    libs = _libcharmm_candidates()
    if not libs:
        return MlpotLimitsStatus(
            max_ml,
            max_pr,
            source="api_func.F90 (libcharmm.so not located for freshness check)",
            api_func_f90=f90,
            note="Set CHARMM_LIB_DIR or CHARMM_HOME so lib freshness can be checked.",
        )

    lib = max(libs, key=lambda p: p.stat().st_mtime)
    if lib.stat().st_mtime < f90.stat().st_mtime:
        return MlpotLimitsStatus(
            *_CONSERVATIVE_LIMITS,
            source="conservative fallback (libcharmm.so older than api_func.F90)",
            api_func_f90=f90,
            libcharmm=lib,
            note="Run ./scripts/rebuild_charmm_mlpot.sh --clean",
        )

    return MlpotLimitsStatus(
        max_ml,
        max_pr,
        source="api_func.F90 (libcharmm.so is up to date)",
        api_func_f90=f90,
        libcharmm=lib,
    )


def charmm_mlpot_limits() -> tuple[int, int]:
    """Effective MLpot limits for preflight checks."""
    status = mlpot_limits_status()
    return status.max_nml, status.max_npr


def max_mlpot_ml_pairs(n_ml_atoms: int) -> int:
    """Central-cell ML–ML pairs passed to the Python callback (vacuum)."""
    n = int(n_ml_atoms)
    if n <= 1:
        return 0
    return n * (n - 1)


def validate_mlpot_system_size(n_ml_atoms: int) -> None:
    """Fail fast before ``mlpot_set_properties`` writes ``fort.104``."""
    status = mlpot_limits_status()
    n_ml = int(n_ml_atoms)
    if n_ml > status.max_nml:
        detail = status.message()
        raise ValueError(
            f"CHARMM MLpot supports at most {status.max_nml} ML atoms in this "
            f"libcharmm build (api_func.F90 max_Nml); selection has {n_ml}. "
            f"For DCM:90 (450 atoms) rebuild libcharmm.so with max_Nml>=512 and "
            f"max_Npr>=300000:\n"
            f"  ./scripts/rebuild_charmm_mlpot.sh\n"
            f"(source/api/api_func.F90 may already be patched; lib must be rebuilt.)\n"
            f"{detail}"
        )
    n_pairs = max_mlpot_ml_pairs(n_ml)
    if n_pairs > status.max_npr:
        raise ValueError(
            f"CHARMM MLpot pair buffers hold at most {status.max_npr} ML pairs "
            f"(max_Npr); {n_ml} ML atoms need {n_pairs}. Rebuild with larger "
            f"max_Npr:\n"
            f"  ./scripts/rebuild_charmm_mlpot.sh\n"
            f"{status.message()}"
        )


def mlpot_limits_message() -> str:
    return mlpot_limits_status().message()
