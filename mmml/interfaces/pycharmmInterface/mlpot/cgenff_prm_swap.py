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


def apply_full_cgenff_params(*, verbose: bool = False) -> None:
    """Restore bonded CGENFF parameters (append-safe) and verify PSF bonds.

    In PBC+MLpot context the bonded parameters were already loaded at session
    start (full CGenFF read).  ``READ PARAM APPEND`` of the bonded-only file
    is a no-op for bonded terms but has a fatal side effect: it triggers
    ``suspend_pbc_before_cgenff_param_append()`` (``crystal free``) followed by
    ``_finalize_pbc_mlpot_exclusions_after_param_read()`` (``CHARMM UPDATE``),
    which invokes ``enbav2e2b2_`` with PBC crystal + force-switched cutoffs and
    segfaults.  When PBC is active we skip the re-read and update the mode
    tracker only.
    """
    global _active_mode
    if _pbc_crystal_is_active():
        n_bond = assert_psf_bonds_present(context="CGENFF MM restore (PBC skip)")
        _active_mode = "full"
        summary = f"CGENFF params: bonded restore skipped (PBC active; PSF bonds={n_bond})"
        from mmml.utils.rich_report import emit_charmm_block

        emit_charmm_block(summary, verbose=verbose)
        if verbose:
            print(summary, flush=True)
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


def _pbc_crystal_is_active() -> bool:
    """Return True when CHARMM has a live PBC crystal (``pbound > 0`` or ``ntrans > 1``)."""
    try:
        from mmml.interfaces.pycharmmInterface.import_pycharmm import PYCHARMM_AVAILABLE

        if not PYCHARMM_AVAILABLE:
            return False
        try:
            import pycharmm.image as image

            if int(image.get_ntrans()) > 1:
                return True
        except Exception:
            pass
        try:
            import ctypes

            import pycharmm.lib as lib

            sx = ctypes.c_double(0.0)
            sy = ctypes.c_double(0.0)
            sz = ctypes.c_double(0.0)
            lib.charmm.pbound_get_size(ctypes.byref(sx), ctypes.byref(sy), ctypes.byref(sz))
            if min(float(sx.value), float(sy.value), float(sz.value)) > 0.0:
                return True
        except Exception:
            pass
    except Exception:
        pass
    return False


def active_cgenff_prm_mode() -> _CgenffPrmMode | None:
    return _active_mode
