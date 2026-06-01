"""CHARMM BLOCK coefficients for MM-only vs MLpot (keep PSF connectivity intact)."""

from __future__ import annotations

from typing import Any

_ML_BLOCK_NAME = "mmml_ml"
_NO_INTERNAL = "BOND 0.0 ANGL 0.0 DIHEdral 0.0 IMPr 0.0"


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    return pycharmm


def apply_charmm_mm_block() -> None:
    """Full CGENFF internal terms on all atoms (MM / pre-MLpot cluster minimize)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block

    reset_block()


def apply_mlpot_energy_block(ml_selection: Any) -> str:
    """Zero CHARMM internal terms on ML atoms; MM environment keeps bonded terms.

    Uses stored CHARMM selections so the PSF is not modified (no ``delete_bonds``).
    """
    pycharmm = _import_pycharmm()
    n_total = int(pycharmm.coor.get_natom())
    n_ml = len(ml_selection.get_atom_indexes())
    if n_ml <= 0:
        raise ValueError("ML selection is empty")
    if n_ml >= n_total:
        block = f"""BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 {_NO_INTERNAL}
END
"""
        pycharmm.lingo.charmm_script(block)
        return "all"

    name = ml_selection.store(_ML_BLOCK_NAME)
    block = f"""BLOCK
CALL 1 SELE .NOT. @{name} END
CALL 2 SELE @{name} END
COEFF 1 1 1.0
COEFF 2 2 1.0 {_NO_INTERNAL}
COEFF 1 2 0.0
END
"""
    pycharmm.lingo.charmm_script(block)
    return name


def clear_mlpot_energy_block(ml_selection: Any, *, block_tag: str) -> None:
    """Drop stored ML selection used for BLOCK (no-op for ``block_tag=='all'``)."""
    if block_tag == "all":
        return
    try:
        ml_selection.unstore()
    except Exception:
        pass
