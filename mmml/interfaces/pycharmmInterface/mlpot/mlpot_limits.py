"""Compile-time MLpot limits in ``libcharmm.so`` (``api_func.F90``)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path


def _charmm_home() -> Path | None:
    home = (os.environ.get("CHARMM_HOME") or "").strip()
    if home:
        return Path(home)
    repo = Path(__file__).resolve().parents[3]
    setup = repo / "CHARMMSETUP"
    if setup.is_file():
        for line in setup.read_text(encoding="utf-8").splitlines():
            if "CHARMM_HOME" in line and "=" in line:
                return Path(line.split("=", 1)[1].strip())
    return None


@lru_cache(maxsize=1)
def charmm_mlpot_limits_from_source() -> tuple[int, int] | None:
    """Parse ``max_Nml`` / ``max_Npr`` from ``source/api/api_func.F90``."""
    home = _charmm_home()
    if home is None:
        return None
    f90 = home / "source" / "api" / "api_func.F90"
    if not f90.is_file():
        return None
    text = f90.read_text(encoding="utf-8", errors="replace")
    ml = re.search(r"max_Nml\s*=\s*(\d+)", text)
    pr = re.search(r"max_Npr\s*=\s*(\d+)", text)
    if not ml or not pr:
        return None
    return int(ml.group(1)), int(pr.group(1))


def charmm_mlpot_limits() -> tuple[int, int]:
    """Effective MLpot limits for preflight checks."""
    env_ml = (os.environ.get("MMML_CHARMM_MLPOT_MAX_ML") or "").strip()
    env_pr = (os.environ.get("MMML_CHARMM_MLPOT_MAX_PAIRS") or "").strip()
    if env_ml and env_pr:
        return int(env_ml), int(env_pr)

    parsed = charmm_mlpot_limits_from_source()
    if parsed is None:
        return 100, 100_000

    lib_dir = (os.environ.get("CHARMM_LIB_DIR") or "").strip()
    home = _charmm_home()
    lib = Path(lib_dir) / "libcharmm.so" if lib_dir else None
    f90 = home / "source" / "api" / "api_func.F90" if home else None
    if (
        lib is not None
        and f90 is not None
        and lib.is_file()
        and f90.is_file()
        and lib.stat().st_mtime < f90.stat().st_mtime
    ):
        # Source limits raised but libcharmm.so not rebuilt yet.
        return 100, 100_000

    return parsed


def max_mlpot_ml_pairs(n_ml_atoms: int) -> int:
    """Central-cell ML–ML pairs passed to the Python callback (vacuum)."""
    n = int(n_ml_atoms)
    if n <= 1:
        return 0
    return n * (n - 1)


def validate_mlpot_system_size(n_ml_atoms: int) -> None:
    """Fail fast before ``mlpot_set_properties`` writes ``fort.104``."""
    n_ml = int(n_ml_atoms)
    max_ml, max_pr = charmm_mlpot_limits()
    if n_ml > max_ml:
        raise ValueError(
            f"CHARMM MLpot supports at most {max_ml} ML atoms in this libcharmm build "
            f"(api_func.F90 max_Nml); selection has {n_ml}. "
            f"For DCM:90 (450 atoms) rebuild libcharmm.so with max_Nml>=512 and "
            f"max_Npr>=300000:\n"
            f"  ./scripts/rebuild_charmm_mlpot.sh\n"
            f"(source/api/api_func.F90 may already be patched; lib must be rebuilt.)"
        )
    n_pairs = max_mlpot_ml_pairs(n_ml)
    if n_pairs > max_pr:
        raise ValueError(
            f"CHARMM MLpot pair buffers hold at most {max_pr} ML pairs (max_Npr); "
            f"{n_ml} ML atoms need {n_pairs}. Rebuild with larger max_Npr:\n"
            f"  ./scripts/rebuild_charmm_mlpot.sh"
        )


def mlpot_limits_message() -> str:
    max_ml, max_pr = charmm_mlpot_limits()
    return f"CHARMM MLpot limits: max_Nml={max_ml}, max_Npr={max_pr}"
