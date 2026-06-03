"""CHARMM BLOCK coefficients for MM-only vs MLpot (keep PSF connectivity intact)."""

from __future__ import annotations

from typing import Any

_ML_BLOCK_NAME = "mmml_ml"

# CHARMM COEFF keywords: BOND ANGL DIHEdral ELEC VDW (no IMPR on this line).
# Global coefficient 0.0 also zeros improper dihedrals and any other unnamed terms.
_ML_SELF_ZERO = "0.0 BOND 0.0 ANGL 0.0 DIHEdral 0.0 ELEC 0.0 VDW 0.0"


def _mlpot_internal_block_coeff_line(mm_internal_scale: float) -> str:
    """BLOCK COEFF line for ML atoms: scaled bonded terms, zero ELEC/VDW."""
    w = float(mm_internal_scale)
    if w < 0.0:
        raise ValueError(f"mm_internal_scale must be >= 0, got {w}")
    if w == 0.0:
        return _ML_SELF_ZERO
    return f"0.0 BOND {w:g} ANGL {w:g} DIHEdral {w:g} ELEC 0.0 VDW 0.0"


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


def apply_mlpot_energy_block(
    ml_selection: Any,
    *,
    mm_internal_scale: float = 0.0,
) -> str:
    """Scale CHARMM bonded terms on ML atoms; MLpot USER supplies ML energy.

    ``mm_internal_scale=0`` (default) zeros BOND/ANGL/DIHE on ML atoms (full ML).
    ``mm_internal_scale=0.1`` keeps 10% CGENFF internal terms alongside MLpot —
    can stiffen X–H and other modes; use as a soft restraint, not a physical mix.

    Uses stored CHARMM selections so the PSF is not modified (no ``delete_bonds``).
    """
    pycharmm = _import_pycharmm()
    coeff = _mlpot_internal_block_coeff_line(mm_internal_scale)
    n_total = int(pycharmm.coor.get_natom())
    n_ml = len(ml_selection.get_atom_indexes())
    if n_ml <= 0:
        raise ValueError("ML selection is empty")
    if n_ml >= n_total:
        block = f"""BLOCK
CALL 1 SELE ALL END
COEFF 1 1 {coeff}
END
"""
        pycharmm.lingo.charmm_script(block)
        return "all"

    name = ml_selection.store(_ML_BLOCK_NAME)
    block = f"""BLOCK
CALL 1 SELE .NOT. @{name} END
CALL 2 SELE @{name} END
COEFF 1 1 1.0
COEFF 2 2 {coeff}
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
