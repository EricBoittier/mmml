"""CHARMM DOMDEC / PyCHARMM API survey for spatial ML halo alignment.

Findings (June 2026)
--------------------

CHARMM Fortran (DOMDEC builds)
  - ``domdec_common`` exports ``q_domdec``, ``q_use_single``
    (``setup/nbonds/helpme_wrapper.F90``).
  - Reciprocal-space nodes use ``q_recip_node`` (``domdec_dr_common``).
  - Coordinate tests: ``testcoord_domdec`` (``setup/image/upimag.F90``).
  - Nonbond and image routines gate on ``q_domdec`` and ``q_recip_node``.

Per-rank atom map access (MMML, no upstream PyCHARMM changes)
  - ``domdec_atoms`` reads ``domdec_common`` scalars and allocatable arrays
    directly from the already-loaded ``libcharmm.so`` via ctypes:

    .. code-block:: python

        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            is_domdec_active,
            get_ndir,
            get_local_atom_indices,
            get_ghost_atom_indices,
        )

  - Returns ``False`` / ``(1,1,1)`` / empty arrays when DOMDEC is compiled out
    (``KEY_DOMDEC=0``) or ``libcharmm.so`` is not loaded yet.

MMML current behavior
  - MLpot paths can call ``disable_charmm_domdec()`` once per process when
    ``MMML_FORCE_DOMDEC_OFF=1`` is set.  The default is to skip ``domdec off``
    because calling it on inactive DOMDEC builds can corrupt OpenMPI pools and
    segfault in ``send_coord_to_recip`` / ``PMPI_Free_mem``.
  - ``charmm_mpi`` handles OpenMPI bootstrap only; no ML force gather/scatter.

Spatial ML phases
  - **Phase 2 (current):** Python ``SpatialDomainGrid`` — 1-D PBC slabs along x,
    monomer COM ownership.  ``n_ranks`` set manually.
  - **Phase 3 (implemented via domdec_atoms):**  ``DomdecAlignedGrid`` reads
    ``ndomx`` from ``domdec_common`` and instantiates ``SpatialDomainGrid``
    with ``n_ranks = ndomx``.  For NDIR N 1 1, the Python slab partitioning
    is *identical* to DOMDEC's spatial partitioning.  When the full array
    symbols are available (``KEY_DOMDEC=1``), ``get_local_atom_indices()``
    and ``get_ghost_atom_indices()`` bypass the COM computation entirely.

Open questions
  1. DOMDEC on + MLpot/JAX: segfaults in multi-rank JAX init under DOMDEC MPI
     pools need investigation before enabling in production.
  2. 3-D NDIR (e.g., NDIR 2 2 2): ``DomdecAlignedGrid`` currently only aligns
     along x when Ny = Nz = 1.
  3. Global energy reduction: per-rank partial energies vs global sum for
     logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class DomdecApiSurvey:
    """Structured summary of DOMDEC API availability for ML spatial decomposition."""

    charmm_fortran_domdec: bool
    pycharmm_domdec_script: bool
    pycharmm_local_atom_api: bool
    pycharmm_ghost_atom_api: bool
    mmml_disable_domdec_for_mlpot: bool
    recommended_phase2_grid: str
    halo_width_formula: str
    open_questions: Sequence[str]


def survey_domdec_api() -> DomdecApiSurvey:
    """Return the current DOMDEC / PyCHARMM capability survey.

    ``pycharmm_local_atom_api`` and ``pycharmm_ghost_atom_api`` are now
    ``True`` because ``domdec_atoms`` reads the Fortran symbols directly via
    ctypes — no upstream PyCHARMM patch required.  Both return empty arrays
    when DOMDEC is not compiled in or not yet active.
    """
    return DomdecApiSurvey(
        charmm_fortran_domdec=True,
        pycharmm_domdec_script=True,
        pycharmm_local_atom_api=True,   # implemented in domdec_atoms.py
        pycharmm_ghost_atom_api=True,   # implemented in domdec_atoms.py
        mmml_disable_domdec_for_mlpot=True,
        recommended_phase2_grid=(
            "DomdecAlignedGrid (Phase 3): reads NDIR from domdec_common, "
            "falls back to SpatialDomainGrid when DOMDEC inactive"
        ),
        halo_width_formula="mm_switch_on + physnet_cutoff + ml_switch_width",
        open_questions=(
            "DOMDEC + MLpot/JAX segfault: multi-rank JAX init under DOMDEC MPI",
            "3-D NDIR support: DomdecAlignedGrid currently aligns on x only",
            "Global energy reduction for logging",
        ),
    )
