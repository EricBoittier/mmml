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


def _charmm_home() -> Path | None:
    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths

    home, _ = resolve_charmm_paths(repo_root=_repo_root())
    if home:
        return Path(home)
    return None


def _charmm_lib_dir() -> Path | None:
    from mmml.interfaces.pycharmmInterface.charmm_paths import resolve_charmm_paths

    _, lib_dir = resolve_charmm_paths(repo_root=_repo_root())
    if lib_dir:
        return Path(lib_dir)
    return None


def _tier_api_func_for_lib_dir(lib_dir: Path) -> Path | None:
    """``.../tier_*_nodomdec/lib`` → sibling ``api_func.F90`` patched for that tier."""
    resolved = lib_dir.expanduser().resolve()
    if resolved.name != "lib":
        return None
    candidate = resolved.parent / "api_func.F90"
    return candidate if candidate.is_file() else None


def _api_func_f90_candidates() -> list[Path]:
    repo = _repo_root()
    home = _charmm_home()
    lib_dir = _charmm_lib_dir()
    paths: list[Path] = []
    if lib_dir is not None:
        tier_f90 = _tier_api_func_for_lib_dir(lib_dir)
        if tier_f90 is not None:
            paths.append(tier_f90)
    if home is not None:
        paths.append(home / "source" / "api" / "api_func.F90")
    paths.extend(
        [
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
                "Build libcharmm under setup/charmm or set CHARMM_HOME; "
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


def pbc_image_copies_per_atom(
    n_ml_atoms: int,
    box_side_A: float | None = None,
    *,
    baseline_copies: float = 5.0,
    reference_density_atoms_per_A3: float = 825.0 / (40.0**3),
    ml_pair_radius_A: float = 12.0,
    max_copies: float = 20.0,
) -> float:
    """Heuristic image-atom count per ML atom for dense PBC bulk liquids.

    Calibrated to ~5 copies for 825 ACO atoms in a 40 Å cube (observed fort.104
    ~4.1M pairs). Without ``box_side_A``, returns ``baseline_copies`` for backward
    compatibility with tier selection that only knows ``n_ml``.
    """
    if box_side_A is None or float(box_side_A) <= 0:
        return float(baseline_copies)
    n = max(1, int(n_ml_atoms))
    L = float(box_side_A)
    rho = n / (L**3)
    scale = rho / float(reference_density_atoms_per_A3)
    copies = baseline_copies * max(1.0, scale**0.5)
    spacing = (L**3 / n) ** (1.0 / 3.0)
    copies = max(copies, ml_pair_radius_A / max(spacing, 0.5))
    return min(float(copies), float(max_copies))


def max_mlpot_ml_pairs_pbc(
    n_ml_atoms: int,
    *,
    image_copies_per_atom: float | None = None,
    box_side_A: float | None = None,
) -> int:
    """Upper-bound ML–ML pairs in ``mlpot_update`` when ``Ntrans != 0``.

    Fortran builds ``Nmlp = Nml*(Nml-1) + Nml*Niml`` where ``Niml`` is the
    count of periodic image atoms whose parent is an ML atom. Dense bulk PBC
    liquids typically have ``Niml ~ image_copies_per_atom * Nml`` (observed
    ~5× for N≈800 in 30–40 Å boxes; ~10× for 2000 ACO in L=32 Å).
    """
    n = int(n_ml_atoms)
    if n <= 1:
        return 0
    if image_copies_per_atom is None:
        image_copies_per_atom = pbc_image_copies_per_atom(n, box_side_A)
    central = n * (n - 1)
    niml = int(n * float(image_copies_per_atom))
    return central + n * niml


def validate_mlpot_system_size(
    n_ml_atoms: int,
    *,
    pbc: bool = False,
    box_side_A: float | None = None,
) -> None:
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
    if pbc:
        copies = pbc_image_copies_per_atom(n_ml, box_side_A)
        n_pairs = max_mlpot_ml_pairs_pbc(n_ml, box_side_A=box_side_A)
        pbc_note = f" (PBC image pairs; ~{copies:.1f}× copies"
        if box_side_A is not None:
            pbc_note += f", L={float(box_side_A):g} Å"
        pbc_note += ")"
    else:
        n_pairs = max_mlpot_ml_pairs(n_ml)
        pbc_note = ""
    if n_pairs > status.max_npr:
        rebuild = (
            f"  ./scripts/ensure_charmm_mlpot_limits.sh --n-ml {n_ml}"
            + (" --pbc" if pbc else "")
        )
        if pbc and box_side_A is not None:
            rebuild += f" --box-size {float(box_side_A):g}"
        rebuild += "\n"
        tier_note = ""
        try:
            tier = select_npr_tier_for_build(n_ml, pbc=pbc, box_side_A=box_side_A)
            tier_note = (
                f"Select tier {tier!r} (max_Npr={tier_max_npr(tier)}). "
                f"Then export CHARMM_LIB_DIR before md-system:\n"
            )
        except ValueError:
            pass
        raise ValueError(
            f"CHARMM MLpot pair buffers hold at most {status.max_npr} ML pairs "
            f"(max_Npr); {n_ml} ML atoms need {n_pairs}{pbc_note}. "
            f"{tier_note}Rebuild with larger max_Npr:\n"
            f"{rebuild}"
            f"{status.message()}"
        )


def mlpot_limits_message() -> str:
    return mlpot_limits_status().message()


NPR_TIERS: dict[str, int] = {
    "default": 8_000_000,
    "large": 8_000_000,
    "xlarge": 12_000_000,
    "xxlarge": 36_000_000,
    "xxxlarge": 56_000_000,
}

# CGenFF all-atom monomer sizes for PBC burst solvents (DCM / ACO campaign).
PBC_BURST_ML_ATOMS_PER_MONOMER: dict[str, int] = {
    "DCM": 5,
    "ACO": 10,
}


def required_max_npr(
    n_ml_atoms: int,
    *,
    margin: float = 1.15,
    pbc: bool = False,
    box_side_A: float | None = None,
) -> int:
    """Minimum ``max_Npr`` compile-time limit for ``n_ml_atoms`` ML atoms."""
    pairs = (
        max_mlpot_ml_pairs_pbc(int(n_ml_atoms), box_side_A=box_side_A)
        if pbc
        else max_mlpot_ml_pairs(int(n_ml_atoms))
    )
    return int(pairs * float(margin))


def select_npr_tier(
    n_ml_atoms: int,
    *,
    margin: float = 1.15,
    pbc: bool = False,
    box_side_A: float | None = None,
) -> str:
    """Smallest tier name that fits ``n_ml_atoms``."""
    needed = required_max_npr(
        n_ml_atoms, margin=margin, pbc=pbc, box_side_A=box_side_A
    )
    for name, cap in sorted(NPR_TIERS.items(), key=lambda item: item[1]):
        if needed <= cap:
            return name
    largest = max(NPR_TIERS.items(), key=lambda item: item[1])
    raise ValueError(
        f"{n_ml_atoms} ML atoms need max_Npr>={needed} pairs; largest tier "
        f"{largest[0]}={largest[1]} is insufficient"
    )


def tier_max_npr(tier: str) -> int:
    key = str(tier).strip().lower()
    if key not in NPR_TIERS:
        raise KeyError(f"unknown NPR tier {tier!r}; choose from {sorted(NPR_TIERS)}")
    return NPR_TIERS[key]


def select_npr_tier_for_build(
    n_ml_atoms: int,
    *,
    margin: float = 1.15,
    pbc: bool = False,
    box_side_A: float | None = None,
) -> str:
    """Smallest CHARMM rebuild tier for ``job_shell`` / prebuild.

    When a box-aware estimate exceeds the largest compiled tier, fall back to
    the ``n_ml``-only baseline (``pbc_image_copies_per_atom`` without box).
    That estimate matches fort.104 calibration (~5× images at N≈800) and avoids
    rejecting dense cells like ACO 266 @ L=32 where the density heuristic
    overshoots.
    """
    if box_side_A is not None:
        try:
            return select_npr_tier(
                n_ml_atoms, margin=margin, pbc=pbc, box_side_A=box_side_A
            )
        except ValueError:
            pass
    return select_npr_tier(
        n_ml_atoms, margin=margin, pbc=pbc, box_side_A=None
    )


def pbc_pair_budget_box_side_A(
    n_ml_atoms: int,
    box_side_A: float | None,
    *,
    margin: float = 1.15,
) -> float | None:
    """Box side for MLpot pair-buffer preflight at registration time.

    Uses the box-aware estimate only when it fits a compiled tier; otherwise
    falls back to the ``n_ml``-only baseline (same policy as
    :func:`select_npr_tier_for_build`).
    """
    if box_side_A is None or float(box_side_A) <= 0:
        return None
    try:
        select_npr_tier(
            int(n_ml_atoms),
            margin=margin,
            pbc=True,
            box_side_A=float(box_side_A),
        )
        return float(box_side_A)
    except ValueError:
        return None


def ensure_mlpot_limits_for_system(
    n_ml_atoms: int,
    *,
    margin: float = 1.15,
    pbc: bool = False,
    box_side_A: float | None = None,
) -> None:
    """Raise with rebuild guidance when the loaded lib is too small."""
    status = mlpot_limits_status()
    needed = required_max_npr(
        n_ml_atoms, margin=margin, pbc=pbc, box_side_A=box_side_A
    )
    tier = select_npr_tier(
        n_ml_atoms, margin=margin, pbc=pbc, box_side_A=box_side_A
    )
    target = tier_max_npr(tier)
    if needed > status.max_npr:
        rebuild = (
            f"  ./scripts/ensure_charmm_mlpot_limits.sh --n-ml {int(n_ml_atoms)}"
            f"{' --pbc' if pbc else ''}"
        )
        if pbc and box_side_A is not None:
            rebuild += f" --box-size {float(box_side_A):g}"
        raise ValueError(
            f"CHARMM MLpot pair buffers hold at most {status.max_npr} ML pairs "
            f"(max_Npr); {n_ml_atoms} ML atoms need {needed}"
            f"{' (PBC)' if pbc else ''}. "
            f"Select tier {tier!r} (max_Npr={target}):\n"
            f"{rebuild}\n"
            f"{status.message()}"
        )


def preflight_mlpot_registration_limits(
    n_ml_atoms: int,
    *,
    mlpot_pbc: bool,
    box_side_A: float | None = None,
) -> None:
    """Fail before long CHARMM pretreat when ``libcharmm`` ``max_Npr`` is too small."""
    budget_box = (
        pbc_pair_budget_box_side_A(int(n_ml_atoms), box_side_A)
        if mlpot_pbc
        else None
    )
    validate_mlpot_system_size(
        int(n_ml_atoms),
        pbc=bool(mlpot_pbc),
        box_side_A=budget_box,
    )


def estimate_ml_atoms(
    n_monomers: int,
    *,
    atoms_per_monomer: int | None = None,
    solvent: str | None = None,
) -> int:
    """Campaign helper: ML atom count from monomer count."""
    if solvent is not None:
        key = str(solvent).strip().upper()
        apm = PBC_BURST_ML_ATOMS_PER_MONOMER.get(key)
        if apm is None:
            raise ValueError(
                f"Unknown solvent {solvent!r} for ML atom sizing; "
                f"supported: {sorted(PBC_BURST_ML_ATOMS_PER_MONOMER)}"
            )
        return int(n_monomers) * int(apm)
    apm = int(atoms_per_monomer) if atoms_per_monomer is not None else 5
    return int(n_monomers) * apm
