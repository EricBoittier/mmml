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


def bonded_cgenff_prm_path() -> Path:
    """Bonded sections only (full constants); safe for READ PARAM APPEND restore."""
    return _cgenff_data_dir() / "bonded_par_all36_cgenff.prm"


def zeroed_cgenff_prm_path(*, bonded_only: bool = False) -> Path:
    name = (
        "zeroed_bonded_par_all36_cgenff.prm"
        if bonded_only
        else "zeroed_par_all36_cgenff.prm"
    )
    return cgenff_prm_path().with_name(name)


def _read_cgenff_prm(path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.nbonds_config import read_cgenff_prm

    if not path.is_file():
        raise FileNotFoundError(
            f"CGENFF parameter file not found: {path}\n"
            "Generate zeroed copies with:\n"
            "  uv run python scripts/zero_charmm_prm.py "
            "mmml/data/charmm/par_all36_cgenff.prm "
            "mmml/data/charmm/zeroed_par_all36_cgenff.prm\n"
            "  uv run python scripts/zero_charmm_prm.py "
            "mmml/data/charmm/par_all36_cgenff.prm "
            "mmml/data/charmm/zeroed_bonded_par_all36_cgenff.prm --bonded-only\n"
            "  uv run python scripts/zero_charmm_prm.py "
            "mmml/data/charmm/par_all36_cgenff.prm "
            "mmml/data/charmm/bonded_par_all36_cgenff.prm --extract-bonded-only"
        )
    read_cgenff_prm(path, append=True)


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


def apply_full_cgenff_params(*, verbose: bool = False, force: bool = False) -> None:
    """Restore bonded CGENFF parameters (append-safe) and verify PSF bonds.

    MLpot registration zeroes bonded force constants via
    ``apply_zeroed_cgenff_params(bonded_only=True)`` so that CHARMM's bonded
    evaluator does not double-count bonds/angles/dihedrals already handled by
    the ML potential.  This function restores them via ``READ PARAM APPEND``
    before any bonded-MM recovery or strain measurement.

    In PBC context, ``READ PARAM APPEND`` calls ``suspend_pbc_before_cgenff_param_append``
    (``crystal free``) and clears IMAGE tables.  Callers that run ``UPDATE`` afterward
    must rebuild crystal + ML exclusions first (see
    ``_finalize_pbc_mlpot_exclusions_after_param_read`` /
    :func:`~mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.prepare_rescue_lists_safe`).

    When ``force=False`` (default) and bonded params were already restored
    (``active_cgenff_prm_mode() == "full"``), this is a no-op.  That avoids
    redundant ``READ PARAM APPEND`` on every ``setup_bonded_only_charmm`` call,
    which otherwise suspends PBC (``crystal free``) and can stall for minutes on
    solvated boxes while CHARMM rebuilds parameter/NBOND state.
    """
    global _active_mode
    if not force and _active_mode == "full":
        if verbose:
            print("CGENFF params: bonded restore skipped (already full)", flush=True)
        return
    path = bonded_cgenff_prm_path()
    _read_cgenff_prm(path)
    n_bond = assert_psf_bonds_present(context="CGENFF MM restore")
    _active_mode = "full"
    summary = f"CGENFF params: bonded restore ({path.name}; PSF bonds={n_bond})"
    from mmml.utils.rich_report import emit_charmm_block

    emit_charmm_block(summary, verbose=verbose)
    if verbose:
        print(summary, flush=True)


def active_cgenff_prm_mode() -> _CgenffPrmMode | None:
    return _active_mode
