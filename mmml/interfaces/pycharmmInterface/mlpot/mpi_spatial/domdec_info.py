"""CHARMM DOMDEC / PyCHARMM API survey for spatial ML halo alignment.

Findings (June 2026)
--------------------

CHARMM Fortran (DOMDEC builds)
  - ``domdec_common`` exports ``q_domdec``, ``q_use_single`` (``setup/nbonds/helpme_wrapper.F90``).
  - Reciprocal-space nodes use ``q_recip_node`` (``domdec_dr_common``).
  - Coordinate tests: ``testcoord_domdec`` (``setup/image/upimag.F90``).
  - Nonbond and image routines gate on ``q_domdec`` and ``q_recip_node``.

PyCHARMM Python surface
  - ``pycharmm.lingo.charmm_script("domdec off")`` — only control exposed today
    (see ``import_pycharmm.disable_charmm_domdec``).
  - No Python API for per-rank local atom counts, ghost atom lists, or domain
    bounds was found in the vendored PyCHARMM tree or MMML wrappers.

MMML current behavior
  - MLpot paths can call ``disable_charmm_domdec()`` once per process when
    ``MMML_FORCE_DOMDEC_OFF=1`` is set. The default is to skip ``domdec off``
    because calling it on inactive DOMDEC builds can itself corrupt OpenMPI
    pools and segfault in ``send_coord_to_recip`` / ``PMPI_Free_mem``.
  - ``charmm_mpi`` handles OpenMPI bootstrap only; no ML force gather/scatter.

Implications for spatial ML (Phase 2–3)
  - **Phase 2:** Build a deterministic Python ``SpatialDomainGrid`` from cubic
    box side and ``n_ranks`` (1-D slabs along *x* with PBC MIC). Halo width
    ``R_halo = mm_switch_on + r_physnet + ε`` (see ``resolve_halo_radius``).
  - **Phase 3:** Replace Python grid with CHARMM DOMDEC metadata when PyCHARMM
    exposes local/ghost atom maps (or ctypes hooks into ``domdec_common``).

Open questions
  1. Can ``domdec`` stream commands return domain bounds without enabling full
     domdec dynamics alongside MLpot?
  2. Monomer ownership: COM-based (matches sparse dimer rule) vs atom-based
     DOMDEC ownership — must agree at boundaries.
  3. Energy reduction: global sum vs per-rank local (logging only).
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
    """Return the current DOMDEC / PyCHARMM capability survey."""
    return DomdecApiSurvey(
        charmm_fortran_domdec=True,
        pycharmm_domdec_script=True,
        pycharmm_local_atom_api=False,
        pycharmm_ghost_atom_api=False,
        mmml_disable_domdec_for_mlpot=True,
        recommended_phase2_grid=(
            "Python SpatialDomainGrid: 1-D PBC slabs along x, monomer COM ownership"
        ),
        halo_width_formula="mm_switch_on + physnet_cutoff + ml_switch_width",
        open_questions=(
            "PyCHARMM exposure of per-rank atom ownership and ghost lists",
            "DOMDEC on with MLpot without segfaults (domdec off remains opt-in)",
            "COM vs atom ownership at domain boundaries",
            "Global energy reduction for logging",
        ),
    )
