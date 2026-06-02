"""CHARMM BLOCK coefficients for MM-only vs MLpot (keep PSF connectivity intact)."""

from __future__ import annotations

from typing import Any

_ML_BLOCK_NAME = "mmml_ml"

# CHARMM COEFF keywords: BOND ANGL DIHEdral ELEC VDW (no IMPR on this line).
# Global coefficient 0.0 also zeros improper dihedrals and any other unnamed terms.
_ML_SELF_ZERO = "0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 0.0 VDW 0.0"


def _import_pycharmm():
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    return pycharmm


def apply_charmm_mm_block() -> None:
    """Full CGENFF internal terms on all atoms (MM / pre-MLpot cluster minimize)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block

    reset_block()


def apply_bonded_mm_only_block() -> None:
    """Bonded MM terms only (BOND/ANGL/DIHE); zero VDW/ELEC for geometry recovery."""
    pycharmm = _import_pycharmm()
    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 BOND 1.0 ANGL 1.0 DIHEdral 1.0 ELEC 0.0 VDW 0.0
END
"""
    pycharmm.lingo.charmm_script(block)


def apply_bonded_vdw_recovery_block() -> None:
    """Bonded MM + VDW for rescue SD; ELEC off (MLpot handles electrostatics).

    Pair with ``NBXMOD 2`` (only 1-2 exclusions) during rescue SD. Production
    ``NBXMOD 5`` is not restored afterward — :func:`restore_workflow_nbonds` is a
    no-op so CHARMM does not rebuild ML exclusion lists (``upinb`` segfault).
    """
    pycharmm = _import_pycharmm()
    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 BOND 1.0 ANGL 1.0 DIHEdral 1.0 ELEC 0.0 VDW 1.0
END
"""
    pycharmm.lingo.charmm_script(block)


def apply_mlpot_energy_block(ml_selection: Any) -> str:
    """Zero CHARMM bonded and nonbonded terms on ML atoms; MLpot supplies the energy.

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
COEFF 1 1 {_ML_SELF_ZERO}
END
"""
        pycharmm.lingo.charmm_script(block)
        return "all"

    name = ml_selection.store(_ML_BLOCK_NAME)
    block = f"""BLOCK
CALL 1 SELE .NOT. @{name} END
CALL 2 SELE @{name} END
COEFF 1 1 1.0
COEFF 2 2 {_ML_SELF_ZERO}
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
