"""Live CHARMM read-back tests for ``write_zeroed_psf_ready_prm``.

These tests load the generated .prm file into a live CHARMM session and verify
that VDW and IMNB energy terms are structurally zero — i.e., CHARMM's internal
VDW parameter tables are never populated.

This is the Option-C regression guard for::

    pycharmm_mlpot: error: CHARMM energy policy still non-zero after
    pre-registration remediation: vdw (IMNB=-12.8278)

Test structure
--------------
Each test:
1. Generates the zeroed .prm with ``write_zeroed_psf_ready_prm``.
2. Loads RTF (topology) + zeroed PRM into CHARMM via ``read.rtf`` / ``read.prm``.
3. Reads the committed ACO PSF + PDB fixture.
4. Calls ``ENER`` via pycharmm.
5. Asserts VDW == 0.0 and IMNB == 0.0 from ``energy.get_energy()``.

The tests are skipped when pycharmm / libcharmm is not available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import can_import_pycharmm
from tests.functionality.pycharmmETC._paths import PYCHARMMETC_DIR, workdir_pdb, workdir_psf

pytestmark = [
    pytest.mark.pycharmm,
    pytest.mark.skipif(
        not can_import_pycharmm(),
        reason="pycharmm / libcharmm not available",
    ),
]

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_ACO_PSF = PYCHARMMETC_DIR / "psf" / "aco-1.psf"
_ACO_PDB = PYCHARMMETC_DIR / "pdb" / "aco.pdb"
_TIP3_PSF = PYCHARMMETC_DIR / "psf" / "tip3-1.psf"

# Locate the bundled CGenFF RTF (needed for atom type lookup even when PRM is zeroed).
def _cgenff_rtf() -> str:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_RTF
    return CGENFF_RTF


def _cgenff_prm_src() -> Path:
    from mmml.interfaces.pycharmmInterface.mlpot.cgenff_prm_swap import cgenff_prm_path
    return cgenff_prm_path()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_zeroed_prm(tmp_path: Path, *, note: str = "", include_nonbonded_zeros: bool = False) -> Path:
    """Write the zeroed PSF-ready .prm for the current CGenFF source."""
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import write_zeroed_psf_ready_prm

    src = _cgenff_prm_src()
    if not src.is_file():
        pytest.skip(f"CGenFF .prm source not found: {src}")
    dst = tmp_path / "zeroed_psf_ready.prm"
    write_zeroed_psf_ready_prm(src, dst, note=note, include_nonbonded_zeros=include_nonbonded_zeros)
    return dst


def _load_zeroed_prm_and_psf(zeroed_prm: Path, psf_workdir_name: str, pdb_workdir_name: str) -> None:
    """Read RTF + zeroed PRM + PSF + PDB into the live CHARMM session."""
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_pdb_file,
        read_psf_card_file,
    )

    psf_path = Path(workdir_psf(psf_workdir_name))
    pdb_path = Path(workdir_pdb(pdb_workdir_name))

    with charmm_relaxed_bomlev():
        read.rtf(_cgenff_rtf())
        read.prm(str(zeroed_prm), append=False)
        read_psf_card_file(psf_path)
        read_pdb_file(pdb_path, resid=True)


def _run_ener_and_get_terms() -> dict[str, float]:
    """Run ENER and return the energy component dict."""
    import pycharmm
    import pycharmm.energy as energy

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command

    with charmm_silent_command():
        pycharmm.lingo.charmm_script("ENER")

    df = energy.get_energy()
    row = df.iloc[0].to_dict()
    return {str(k): float(v) for k, v in row.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Tests: ACO (acetone, 10 atoms)
# ---------------------------------------------------------------------------


def test_aco_vdw_is_zero_after_zeroed_prm_read(pycharmm_workdir: Path) -> None:
    """READ PARAM with zeroed PRM: VDW must be exactly 0.0 for ACO."""
    zeroed = _make_zeroed_prm(pycharmm_workdir)
    _load_zeroed_prm_and_psf(zeroed, "aco-1.psf", "aco.pdb")

    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()
    terms = _run_ener_and_get_terms()

    vdw = terms.get("VDW", None)
    assert vdw is not None, f"VDW not found in ENER output; got keys: {list(terms)}"
    assert vdw == pytest.approx(0.0, abs=1e-6), (
        f"VDW={vdw:.6g} kcal/mol after loading zeroed PRM — "
        "NONBONDED section was not fully suppressed"
    )


def test_aco_imnb_is_zero_after_zeroed_prm_read(pycharmm_workdir: Path) -> None:
    """READ PARAM with zeroed PRM: IMNB (image nonbond VDW) must be 0.0 for ACO."""
    zeroed = _make_zeroed_prm(pycharmm_workdir)
    _load_zeroed_prm_and_psf(zeroed, "aco-1.psf", "aco.pdb")

    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()
    terms = _run_ener_and_get_terms()

    imnb = terms.get("IMNB", 0.0)
    assert imnb == pytest.approx(0.0, abs=1e-6), (
        f"IMNB={imnb:.6g} kcal/mol — image VDW not suppressed by zeroed PRM"
    )


def test_aco_bonded_terms_are_zero_after_zeroed_prm_read(pycharmm_workdir: Path) -> None:
    """All bonded terms (BOND, ANGL, DIHE, IMPR) must also be 0.0."""
    zeroed = _make_zeroed_prm(pycharmm_workdir)
    _load_zeroed_prm_and_psf(zeroed, "aco-1.psf", "aco.pdb")

    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()
    terms = _run_ener_and_get_terms()

    for key in ("BOND", "ANGL", "DIHE", "IMPR"):
        val = terms.get(key, 0.0)
        assert val == pytest.approx(0.0, abs=1e-6), (
            f"{key}={val:.6g} kcal/mol — bonded force constants not zeroed in PRM"
        )


def test_aco_elec_nonzero_charges_still_present(pycharmm_workdir: Path) -> None:
    """Electrostatics should still fire (partial charges live in PSF, not PRM).

    This verifies the PRM overlay only zeros force constants and VDW — it does
    NOT zero PSF charges.  ELEC > 0 confirms CHARMM state is valid.
    """
    zeroed = _make_zeroed_prm(pycharmm_workdir)
    _load_zeroed_prm_and_psf(zeroed, "aco-1.psf", "aco.pdb")

    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    setup_default_nbonds()
    terms = _run_ener_and_get_terms()

    elec = terms.get("ELEC", 0.0)
    # ACO has partial charges (O=-0.48, C=+0.40 etc.); ELEC must be non-zero.
    assert elec != pytest.approx(0.0, abs=1e-3), (
        f"ELEC={elec:.6g} — expected non-zero (partial charges live in PSF); "
        "CHARMM session may be in an invalid state"
    )


# ---------------------------------------------------------------------------
# Tests: verify full vs zeroed PRM gives different VDW (sanity check)
# ---------------------------------------------------------------------------


def test_full_prm_gives_nonzero_vdw_for_aco(pycharmm_workdir: Path) -> None:
    """Sanity: loading the *full* (unzeroed) PRM gives VDW != 0 for ACO.

    This ensures the test harness actually exercises VDW — without this
    the zero-VDW tests above could pass trivially on a broken ENER probe.
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_pdb_file,
        read_psf_card_file,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds

    src = _cgenff_prm_src()
    if not src.is_file():
        pytest.skip(f"CGenFF .prm not found: {src}")

    psf_path = Path(workdir_psf("aco-1.psf"))
    pdb_path = Path(workdir_pdb("aco.pdb"))

    with charmm_relaxed_bomlev():
        read.rtf(_cgenff_rtf())
        read.prm(str(src), append=False)
        read_psf_card_file(psf_path)
        read_pdb_file(pdb_path, resid=True)

    setup_default_nbonds()
    terms = _run_ener_and_get_terms()

    vdw = terms.get("VDW", 0.0)
    assert vdw != pytest.approx(0.0, abs=1e-4), (
        f"Full PRM gave VDW={vdw:.6g} — expected non-zero; test harness may be broken"
    )


# ---------------------------------------------------------------------------
# Tests: READ PARAM APPEND (overlay mode, same session as full PRM)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CHARMM READ PARAM APPEND does not overwrite existing atom-type VDW "
        "entries — it only inserts *new* types.  Appending a zeroed NONBONDED "
        "overlay after a full CGenFF PRM load leaves the original epsilons in "
        "the live VDW table.  The correct workflow is to load "
        "write_zeroed_psf_ready_prm() output as the *first* (and only) PRM, "
        "not as an overlay.  This test documents the known limitation."
    ),
)
def test_zeroed_prm_append_after_full_prm_zeros_vdw(pycharmm_workdir: Path) -> None:
    """CHARMM READ PARAM APPEND does NOT zero VDW for pre-existing atom types.

    This test documents the known limitation: CHARMM's APPEND mode only adds
    *new* atom types; it does not overwrite epsilon values for types that are
    already in the parameter table.  The test is expected to fail (xfail).

    Correct usage
    -------------
    Load ``write_zeroed_psf_ready_prm()`` output as the **first** PRM
    (``include_nonbonded_zeros=False``) so CHARMM's VDW table is never populated
    with non-zero values.  See ``test_aco_vdw_is_zero_after_zeroed_prm_read``.
    """
    import pycharmm.read as read

    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_relaxed_bomlev
    from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
        read_pdb_file,
        read_psf_card_file,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import setup_default_nbonds
    from mmml.interfaces.pycharmmInterface.nbonds_config import (
        CGENFF_PRM_BOMLEV,
        read_cgenff_prm,
    )

    src = _cgenff_prm_src()
    if not src.is_file():
        pytest.skip(f"CGenFF .prm not found: {src}")

    zeroed = _make_zeroed_prm(pycharmm_workdir, note="append-mode test", include_nonbonded_zeros=True)
    psf_path = Path(workdir_psf("aco-1.psf"))
    pdb_path = Path(workdir_pdb("aco.pdb"))

    with charmm_relaxed_bomlev(CGENFF_PRM_BOMLEV):
        # Step 1: load full PRM (as happens during PSF build)
        read.rtf(_cgenff_rtf())
        read.prm(str(src), append=False)
        read_psf_card_file(psf_path)
        read_pdb_file(pdb_path, resid=True)

    # Step 2: append zeroed overlay via read_cgenff_prm (handles PBC suspend,
    # flex=True, and correct bomlev automatically — same path as production code).
    read_cgenff_prm(zeroed, append=True)

    # Rebuild nonbond lists after READ PARAM APPEND clears them (PARMIO).
    import pycharmm
    from mmml.interfaces.pycharmmInterface.charmm_levels import charmm_silent_command
    setup_default_nbonds()
    with charmm_silent_command():
        pycharmm.lingo.charmm_script("UPDAte")
    terms = _run_ener_and_get_terms()

    vdw = terms.get("VDW", None)
    assert vdw is not None, f"VDW not in ENER; keys={list(terms)}"
    assert vdw == pytest.approx(0.0, abs=1e-6), (
        f"VDW={vdw:.6g} kcal/mol after READ PARAM APPEND zeroed overlay — "
        "NONBONDED section was not suppressed by append"
    )

    imnb = terms.get("IMNB", 0.0)
    assert imnb == pytest.approx(0.0, abs=1e-6), (
        f"IMNB={imnb:.6g} kcal/mol after READ PARAM APPEND zeroed overlay"
    )
