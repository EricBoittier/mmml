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


# CHARMM bonded energy terms skipped when every atom is ML (see zero_mlpot_psf_mm_terms).
ALL_ML_SKIPE_BONDED = ("BOND", "ANGL", "UREY", "DIHE", "IMPR", "CDIH")


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

    - Re-reads a **bonded-only** zeroed CGENFF .prm (BOND/ANGL/DIHE/IMPR/UREY-b → 0;
      NONBOND/NBFIX/HBOND omitted so READ PARAM APPEND does not clear exclusion lists).
    - Zeros partial charges on ML atoms (ELEC off; MLpot supplies ML electrostatics).
    - Deletes PSF dihedrals/impropers/CMAP that touch ML atoms
      (:func:`delete_ml_torsion_terms`): the APPEND leaves the other terms of
      multi-term CGenFF dihedrals live. Bonds and angles stay in the PSF (no
      ``delete_connectivity``), so nonbond exclusions are unchanged.

    Hybrid ML+MM may still need legacy BLOCK (``MMML_MLPOT_USE_BLOCK=1``): the
    zeroed APPEND works by atom type, so it also zeroes BOND/ANGL/UREY/IMPR and
    one term of every dihedral of MM molecules that use CGenFF types (multi-term
    dihedrals keep the rest). BLOCK is also needed for
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
        vdw_note = ", CHARMM VDW on" if periodic_external else ""
        summary = (
            f"MLpot zeroed CGENFF: all-ML ({n_total} atoms; "
            f"bonded params zeroed, PSF bonds={n_bond_before}{vdw_note})"
        )
    else:
        tag = ml_selection.store(_ML_BLOCK_NAME)
        n_mm = n_total - n_ml
        vdw_note = ", CHARMM VDW on MM" if periodic_external else ""
        summary = (
            f"MLpot zeroed CGENFF: hybrid ({n_ml} ML + {n_mm} MM; "
            f"ML bonded zeroed, PSF bonds={n_bond_before}{vdw_note})"
        )

    apply_zeroed_cgenff_params(bonded_only=True, verbose=verbose)
    # READ PARAM APPEND zeroes BOND/ANGL/UREY/IMPR but overwrites only one
    # term of each multi-term CGenFF dihedral, so the rest stay live (ETOH:181
    # box: DIHE 170.8 -> 133.8 kcal/mol). ML torsions are therefore deleted
    # from the PSF below, in both cases. All-ML also skips every CHARMM bonded
    # term (covers bonded types the zeroed file lacks, e.g. extra PRMs; SKIPE
    # accumulates, so this composes with the energy policy's SKIPE VDW IMNB).
    # Hybrid: SKIPE would also drop the MM molecules' bonded terms.
    if tag == "all":
        pycharmm.lingo.charmm_script("SKIPE " + " ".join(ALL_ML_SKIPE_BONDED))
        summary += f"; SKIPE {' '.join(ALL_ML_SKIPE_BONDED)}"
    else:
        import warnings

        warnings.warn(
            "MLpot hybrid PSF registration: the zeroed-CGenFF READ PARAM APPEND "
            "works by atom type, so MM molecules that use CGenFF types also lose "
            "BOND/ANGL/UREY/IMPR and keep only part of their multi-term dihedrals "
            "(ML torsions are deleted from the PSF). SHAKE-rigid TIP3 is "
            "unaffected; use --mlpot-use-block for flexible MM molecules.",
            stacklevel=2,
        )

    charges = list(pycharmm.psf.get_charges())
    for idx in ml_indices:
        charges[int(idx)] = 0.0
    pycharmm.psf.set_charge(charges)

    removed = delete_ml_torsion_terms(
        ml_selection, all_ml=(tag == "all"), pycharmm=pycharmm
    )
    if removed is not None:
        summary += (
            f"; deleted ML torsions (DIHE={removed['dihedrals']}, "
            f"IMPR={removed['impropers']}, CMAP={removed['cmaps']})"
        )

    assert_psf_bonds_present(context="MLpot registration (after zeroed CGENFF)")

    from mmml.utils.rich_report import emit_charmm_block

    emit_charmm_block(summary, verbose=verbose)
    if verbose:
        print(summary, flush=True)
    return tag


# PSF term counters (CHARMM ``psf`` module) for the ML torsion deletion check.
# gfortran exports ``__psf_MOD_<name>``; Intel Fortran exports ``psf_mp_<name>_``.
_PSF_TORSION_COUNTERS = {"dihedrals": "nphi", "impropers": "nimphi", "cmaps": "ncrterm"}


def _psf_torsion_counts(pycharmm: Any) -> dict[str, int] | None:
    """Live PSF dihedral/improper/CMAP counts, or None when not readable.

    ``?NPHI`` and friends are only refreshed by ``PSFSUM`` (not by the
    ``pycharmm.psf.delete_*`` API), so read the Fortran module variables.
    """
    import ctypes

    try:
        lib = pycharmm.lib.charmm
        counts: dict[str, int] = {}
        for kind, var in _PSF_TORSION_COUNTERS.items():
            for symbol in (f"__psf_MOD_{var}", f"psf_mp_{var}_"):
                try:
                    counts[kind] = int(ctypes.c_int.in_dll(lib, symbol).value)
                    break
                except ValueError:
                    continue
            else:
                return None
        return counts
    except Exception:
        return None


def delete_ml_torsion_terms(
    ml_selection: Any,
    *,
    all_ml: bool = False,
    pycharmm: Any = None,
) -> dict[str, int] | None:
    """Delete PSF dihedrals, impropers and CMAP terms that touch ML atoms.

    ``READ PARAM APPEND FLEX`` of the zeroed CGenFF file overwrites only the
    last stored dihedral with the same four atom types, whatever its
    multiplicity, so multi-term CGenFF torsions keep their other terms
    (one ETOH: DIHE 3.2335 -> 0.2894 kcal/mol). CMAP is not in the zeroed file
    at all. Deleting the terms from the PSF is exact per atom and leaves MM
    molecules' torsion terms in the PSF.

    Bonds and angles stay in the PSF: CHARMM builds nonbond exclusions and
    1-4 pairs (``MAKINB``) from the bond list only, so VDW/ELEC exclusions are
    unchanged. The deleted terms come back only with a PSF reload;
    ``apply_full_cgenff_params`` restores force constants, not PSF entries.

    Returns the number of terms removed per kind, or None when the PSF
    counters cannot be read. With ``all_ml=True``, raises if any
    dihedral/improper/CMAP term is left.
    """
    if pycharmm is None:
        pycharmm = _import_pycharmm()
    before = _psf_torsion_counts(pycharmm)
    pycharmm.psf.delete_dihedrals(ml_selection, ml_selection)
    pycharmm.psf.delete_impropers(ml_selection, ml_selection)
    pycharmm.psf.delete_cmaps(ml_selection, ml_selection)
    after = _psf_torsion_counts(pycharmm)

    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
        mark_ml_torsions_deleted,
    )

    mark_ml_torsions_deleted()
    if before is None or after is None:
        return None
    if all_ml:
        left = {kind: n for kind, n in after.items() if n}
        if left:
            raise RuntimeError(
                "MLpot registration: all-ML PSF still has torsion terms after "
                f"deleting them on ML atoms ({left}); CHARMM would double-count "
                "torsions on top of the ML potential."
            )
    return {kind: before[kind] - after[kind] for kind in before}


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


class SelectiveBondedBlockUnsupportedUnderMPI(RuntimeError):
    """Selective COEFF BLOCK hangs on MPI-linked libcharmm under ``mpirun``."""


def _assert_selective_block_safe(*, context: str = "") -> None:
    from mmml.interfaces.pycharmmInterface.charmm_mpi import (
        selective_bonded_block_unsafe_under_mpi,
    )

    if selective_bonded_block_unsafe_under_mpi():
        where = f" ({context})" if context else ""
        raise SelectiveBondedBlockUnsupportedUnderMPI(
            "selective COEFF BLOCK hangs on MPI-linked libcharmm under mpirun"
            f"{where}; use --no-bonded-mm-mini or serial python"
        )


def _run_block_script(
    summary: str,
    script: str,
    *,
    verbose: bool = False,
    selective: bool = False,
    context: str = "",
) -> None:
    """Apply a BLOCK script quietly and optionally emit a one-line Python summary."""
    if selective:
        _assert_selective_block_safe(context=context or summary)
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


def apply_bonded_mm_only_block(
    *,
    verbose: bool = False,
    restore_params: bool = True,
    force_restore_params: bool = False,
) -> None:
    """Bonded MM terms only (BOND/ANGL/DIHE); zero VDW/ELEC for geometry recovery.

    **Hang / stall note:** With ``restore_params=True`` (default), this calls
    :func:`apply_full_cgenff_params` before BLOCK.  That issues ``READ PARAM APPEND``
    on ``bonded_par_all36_cgenff.prm`` after ``crystal free`` when PBC is active.
    On solvated periodic systems that step can take a long time (appearing hung at
    ``MMML: crystal free before CGENFF READ PARAM APPEND``).  It is skipped when
    bonded params are already restored unless ``force_restore_params=True``.
    Under ``mpirun`` with MPI-linked libcharmm, selective BLOCK itself may hang —
    see :func:`_assert_selective_block_safe`.
    """
    if restore_params:
        from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import (
            apply_full_cgenff_params,
        )

        apply_full_cgenff_params(verbose=verbose, force=force_restore_params)
    block = """BLOCK
CALL 1 SELE ALL END
COEFF 1 1 1.0 BOND 1.0 ANGL 1.0 DIHEdral 1.0 ELEC 0.0 VDW 0.0
END
"""
    _run_block_script(
        "CHARMM BLOCK: bonded-only (BOND/ANGL/DIHE on, ELEC/VDW off)",
        block,
        verbose=verbose,
        selective=True,
        context="apply_bonded_mm_only_block",
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
        selective=True,
        context="apply_bonded_vdw_recovery_block",
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
