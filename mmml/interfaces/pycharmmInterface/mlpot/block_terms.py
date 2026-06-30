"""CHARMM BLOCK coefficients and CGENFF param swap for MM-only vs MLpot."""

from __future__ import annotations

import os
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


def _truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "yes", "true")


def mlpot_use_block_registration(*, explicit: bool | None = None) -> bool:
    """True when MLpot registration should run CHARMM BLOCK (legacy path).

    Default (unset env / ``explicit=None``): **False** — zero MM on ML atoms via
    PSF edits (:func:`zero_mlpot_psf_mm_terms`) instead of ``eval_charmm_script``.
    Opt in with ``MMML_MLPOT_USE_BLOCK=1`` or ``--mlpot-use-block``.
    """
    if explicit is not None:
        return bool(explicit)
    return _truthy("MMML_MLPOT_USE_BLOCK")


def zero_mlpot_psf_mm_terms(
    ml_selection: Any,
    *,
    mm_internal_scale: float = 0.0,
    verbose: bool = False,
    periodic_external: bool = False,
) -> str:
    """Disable CHARMM MM on ML atoms via zeroed CGENFF params (PSF connectivity kept).

    - Re-reads a zeroed CGENFF .prm (bonded-only when ``periodic_external`` needs
      CHARMM IMAGE VDW; full zero otherwise).
    - Zeros partial charges on ML atoms (ELEC off; MLpot supplies ML electrostatics).
    - Does **not** call ``delete_connectivity`` (no DELTIC bond/angle deletion).

    Hybrid ML+MM may still need legacy BLOCK (``MMML_MLPOT_USE_BLOCK=1``) for
    ML–MM cross VDW when not using periodic CHARMM VDW.
    """
    if float(mm_internal_scale) > 0.0:
        raise ValueError(
            f"mm_internal_scale={mm_internal_scale} requires BLOCK registration "
            "(set MMML_MLPOT_USE_BLOCK=1 or --mlpot-use-block)"
        )
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        apply_zeroed_cgenff_params,
        assert_psf_bonds_present,
    )

    pycharmm = _import_pycharmm()
    n_total = int(pycharmm.coor.get_natom())
    ml_indices = ml_selection.get_atom_indexes()
    n_ml = len(ml_indices)
    if n_ml <= 0:
        raise ValueError("ML selection is empty")

    n_bond_before = assert_psf_bonds_present(context="MLpot registration")

    if n_ml >= n_total:
        tag = "all"
        if periodic_external:
            summary = (
                f"MLpot zeroed CGENFF: periodic external all-ML ({n_total} atoms; "
                f"bonded params zeroed, PSF bonds={n_bond_before}, CHARMM VDW on)"
            )
        else:
            summary = (
                f"MLpot zeroed CGENFF: all-ML ({n_total} atoms; "
                f"bonded+nonbond zeroed, PSF bonds={n_bond_before})"
            )
    else:
        tag = ml_selection.store(_ML_BLOCK_NAME)
        n_mm = n_total - n_ml
        if periodic_external:
            summary = (
                f"MLpot zeroed CGENFF: periodic external hybrid ({n_ml} ML + {n_mm} MM; "
                f"ML bonded zeroed, PSF bonds={n_bond_before}, CHARMM VDW on MM)"
            )
        else:
            summary = (
                f"MLpot zeroed CGENFF: hybrid ({n_ml} ML + {n_mm} MM; "
                f"CGENFF zeroed, PSF bonds={n_bond_before})"
            )

    apply_zeroed_cgenff_params(
        bonded_only=bool(periodic_external),
        verbose=verbose,
    )

    charges = list(pycharmm.psf.get_charges())
    for idx in ml_indices:
        charges[int(idx)] = 0.0
    pycharmm.psf.set_charge(charges)

    assert_psf_bonds_present(context="MLpot registration (after zeroed CGENFF)")

    from mmml.utils.rich_report import emit_charmm_block

    emit_charmm_block(summary, verbose=verbose)
    if verbose:
        print(summary, flush=True)
    return tag


def apply_mlpot_registration_mm_off(
    ml_selection: Any,
    *,
    mm_internal_scale: float = 0.0,
    verbose: bool = False,
    periodic_external: bool = False,
    use_block: bool | None = None,
) -> str:
    """Zero CHARMM MM on ML atoms for MLpot registration (BLOCK or PSF path)."""
    if mlpot_use_block_registration(explicit=use_block):
        if periodic_external:
            return apply_mlpot_periodic_external_block(
                ml_selection,
                mm_internal_scale=float(mm_internal_scale),
                verbose=verbose,
            )
        return apply_mlpot_energy_block(
            ml_selection,
            mm_internal_scale=float(mm_internal_scale),
            verbose=verbose,
        )
    return zero_mlpot_psf_mm_terms(
        ml_selection,
        mm_internal_scale=float(mm_internal_scale),
        verbose=verbose,
        periodic_external=periodic_external,
    )


def _run_block_script(summary: str, script: str, *, verbose: bool = False) -> None:
    """Apply a BLOCK script quietly and optionally emit a one-line Python summary."""
    from mmml.interfaces.pycharmmInterface.charmm_levels import run_charmm_script_quiet
    from mmml.utils.rich_report import emit_charmm_block

    run_charmm_script_quiet(script)
    emit_charmm_block(summary, verbose=verbose)


def apply_charmm_mm_block(*, verbose: bool = False) -> None:
    """Full CGENFF parameters + BLOCK COEFF 1.0 (MM / pre-MLpot cluster minimize)."""
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        apply_full_cgenff_params,
    )

    apply_full_cgenff_params(verbose=verbose)
    reset_block()


def apply_bonded_mm_only_block(*, verbose: bool = False) -> None:
    """Bonded MM terms only (BOND/ANGL/DIHE); zero VDW/ELEC for geometry recovery."""
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        apply_full_cgenff_params,
    )

    apply_full_cgenff_params(verbose=verbose)
    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 BOND 1.0 ANGL 1.0 DIHEdral 1.0 ELEC 0.0 VDW 0.0
END
"""
    _run_block_script(
        "CHARMM BLOCK: bonded-only (BOND/ANGL/DIHE on, ELEC/VDW off)",
        block,
        verbose=verbose,
    )


def apply_bonded_vdw_recovery_block(*, verbose: bool = False) -> None:
    """Bonded MM + VDW for rescue SD; ELEC off (MLpot handles electrostatics).

    Pair with ``NBXMOD 2`` (only 1-2 exclusions) during rescue SD. Production
    ``NBXMOD 5`` is not restored afterward — :func:`restore_workflow_nbonds` is a
    no-op so CHARMM does not rebuild ML exclusion lists (``upinb`` segfault).
    """
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        apply_full_cgenff_params,
    )

    apply_full_cgenff_params(verbose=verbose)
    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 BOND 1.0 ANGL 1.0 DIHEdral 1.0 ELEC 0.0 VDW 1.0
END
"""
    _run_block_script(
        "CHARMM BLOCK: bonded+VDW recovery (ELEC off)",
        block,
        verbose=verbose,
    )


def apply_mlpot_periodic_external_block(
    ml_selection: Any,
    *,
    mm_internal_scale: float = 0.0,
    verbose: bool = False,
) -> str:
    """MLpot BLOCK for periodic external MM: CHARMM IMAGE VDW on, ELEC off (ScaFaCoS).

    Bonded terms on ML atoms follow ``mm_internal_scale`` (same as
    :func:`apply_mlpot_energy_block`).  Nonbond: JAX LJ/Coulomb are disabled;
    CHARMM computes periodic VDW; Coulomb is added in the Python callback.
    """
    pycharmm = _import_pycharmm()
    coeff = _mlpot_internal_block_coeff_line(mm_internal_scale)
    if "ELEC 0.0 VDW 0.0" in coeff:
        periodic_coeff = coeff.replace("ELEC 0.0 VDW 0.0", "ELEC 0.0 VDW 1.0")
    elif "ELEC 0.0 VDW 1.0" in coeff:
        periodic_coeff = coeff
    else:
        periodic_coeff = coeff + " ELEC 0.0 VDW 1.0"
    n_total = int(pycharmm.coor.get_natom())
    n_ml = len(ml_selection.get_atom_indexes())
    if n_ml <= 0:
        raise ValueError("ML selection is empty")
    if n_ml >= n_total:
        block = f"""BLOCK
CALL 1 SELE ALL END
COEFF 1 1 {periodic_coeff}
END
"""
        summary = (
            f"CHARMM BLOCK: periodic external MM ({n_total} atoms, "
            f"CHARMM VDW on, ELEC off → ScaFaCoS)"
        )
        _run_block_script(summary, block, verbose=verbose)
        return "all"

    name = ml_selection.store(_ML_BLOCK_NAME)
    block = f"""BLOCK
CALL 1 SELE .NOT. @{name} END
CALL 2 SELE @{name} END
COEFF 1 1 1.0
COEFF 2 2 {periodic_coeff}
END
"""
    summary = (
        f"CHARMM BLOCK: periodic external MM (MM atoms CHARMM VDW; "
        f"ML atoms {periodic_coeff}; ELEC off → ScaFaCoS)"
    )
    _run_block_script(summary, block, verbose=verbose)
    return name


def apply_mlpot_energy_block(
    ml_selection: Any,
    *,
    mm_internal_scale: float = 0.0,
    verbose: bool = False,
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
        if mm_internal_scale > 0.0:
            summary = (
                f"CHARMM BLOCK: MLpot all-ML ({n_total} atoms, "
                f"bonded scale={mm_internal_scale:g}, ELEC/VDW off)"
            )
        else:
            summary = (
                f"CHARMM BLOCK: MLpot all-ML ({n_total} atoms, bonded/ELEC/VDW off)"
            )
        _run_block_script(summary, block, verbose=verbose)
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
    n_mm = n_total - n_ml
    if mm_internal_scale > 0.0:
        summary = (
            f"CHARMM BLOCK: MLpot hybrid ({n_ml} ML + {n_mm} MM atoms, "
            f"bonded scale={mm_internal_scale:g} on ML, ELEC/VDW off on ML)"
        )
    else:
        summary = (
            f"CHARMM BLOCK: MLpot hybrid ({n_ml} ML + {n_mm} MM atoms, "
            "ML bonded/ELEC/VDW off)"
        )
    _run_block_script(summary, block, verbose=verbose)
    return name


def clear_mlpot_energy_block(ml_selection: Any, *, block_tag: str) -> None:
    """Drop stored ML selection used for BLOCK (no-op for ``block_tag=='all'``)."""
    if block_tag == "all":
        return
    try:
        ml_selection.unstore()
    except Exception:
        pass
