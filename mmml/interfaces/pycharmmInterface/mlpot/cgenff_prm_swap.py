"""Swap CGENFF .prm between full and zeroed force constants (PSF connectivity kept)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_CgenffPrmMode = Literal["full", "zeroed", "zeroed_bonded"]
_active_mode: _CgenffPrmMode | None = None


def _cgenff_data_dir() -> Path:
    # .../mmml/interfaces/pycharmmInterface/mlpot/cgenff_prm_swap.py -> mmml/data/charmm
    return Path(__file__).resolve().parents[3] / "data" / "charmm"


def cgenff_prm_path() -> Path:
    return _cgenff_data_dir() / "par_all36_cgenff.prm"


def zeroed_cgenff_prm_path(*, bonded_only: bool = False) -> Path:
    name = (
        "zeroed_bonded_par_all36_cgenff.prm"
        if bonded_only
        else "zeroed_par_all36_cgenff.prm"
    )
    return cgenff_prm_path().with_name(name)


def _read_cgenff_prm(path: Path) -> None:
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev

    if not path.is_file():
        raise FileNotFoundError(
            f"CGENFF parameter file not found: {path}\n"
            "Generate zeroed copies with:\n"
            "  uv run python scripts/zero_charmm_prm.py "
            "mmml/data/charmm/par_all36_cgenff.prm "
            "mmml/data/charmm/zeroed_par_all36_cgenff.prm\n"
            "  uv run python scripts/zero_charmm_prm.py "
            "mmml/data/charmm/par_all36_cgenff.prm "
            "mmml/data/charmm/zeroed_bonded_par_all36_cgenff.prm --bonded-only"
        )
    with charmm_relaxed_bomlev():
        # Must match :func:`read_cgenff_toppar` (``read.prm`` without ``flex``).
        # APPEND + FLEX after a non-flex read triggers PARMIO level -2 abort.
        read.prm(str(path), append=True, flex=False)


def psf_bond_count() -> int:
    import pycharmm.psf as psf

    return int(psf.get_nbond())


def assert_psf_bonds_present(*, min_bonds: int = 1, context: str = "CGENFF MM") -> int:
    """Raise if PSF bond count is below *min_bonds* (connectivity must stay intact)."""
    n_bond = psf_bond_count()
    if n_bond < int(min_bonds):
        raise RuntimeError(
            f"{context}: PSF has {n_bond} bonds (expected >= {min_bonds}). "
            "Bonds were deleted or topology was not loaded; reload PSF before MM work."
        )
    return n_bond


def apply_zeroed_cgenff_params(
    *,
    bonded_only: bool = False,
    verbose: bool = False,
) -> None:
    """Re-read CGENFF parameters with zero force constants (append/overrides)."""
    global _active_mode
    path = zeroed_cgenff_prm_path(bonded_only=bonded_only)
    _read_cgenff_prm(path)
    _active_mode = "zeroed_bonded" if bonded_only else "zeroed"
    summary = (
        f"CGENFF params: zeroed bonded only ({path.name})"
        if bonded_only
        else f"CGENFF params: zeroed bonded+nonbond ({path.name})"
    )
    from mmml.utils.rich_report import emit_charmm_block

    emit_charmm_block(summary, verbose=verbose)
    if verbose:
        print(summary, flush=True)


def apply_full_cgenff_params(*, verbose: bool = False) -> None:
    """Restore full CGENFF parameters and verify PSF bonds are still present."""
    global _active_mode
    path = cgenff_prm_path()
    _read_cgenff_prm(path)
    n_bond = assert_psf_bonds_present(context="CGENFF MM restore")
    _active_mode = "full"
    summary = f"CGENFF params: full ({path.name}; PSF bonds={n_bond})"
    from mmml.utils.rich_report import emit_charmm_block

    emit_charmm_block(summary, verbose=verbose)
    if verbose:
        print(summary, flush=True)


def active_cgenff_prm_mode() -> _CgenffPrmMode | None:
    return _active_mode
