"""Unit tests for ``write_zeroed_psf_ready_prm`` in charmm_prm_zero.

``write_zeroed_psf_ready_prm`` is the Option-C fix for::

    pycharmm_mlpot: error: CHARMM energy policy still non-zero after
    pre-registration remediation: vdw (IMNB=-12.8278)

Loading the generated file via ``READ PARAM APPEND`` before ``register_mlpot``
means CHARMM never registers VDW parameters, so VDW and IMNBvdw are
structurally zero -- no runtime ``SCALAR VDW SET 0.0`` patch is needed.

Properties verified
-------------------
1. NONBONDED / NBFIX / HBOND sections are **absent** -- loading this file via
   READ PARAM APPEND cannot reinstate any epsilon/Rmin entry.
2. All bonded force constants (Kb, Ktheta, Kub, Vn) are zeroed.
3. Equilibrium geometry (r0, theta0, dihedral phase/multiplicity) is preserved.
4. Header stamp is present (MMML PSF-ready overlay).
5. Optional note is written into the header.
6. The function returns the destination Path.
7. Parent directories are created automatically.
8. Bonded section headers are retained for READ PARAM APPEND safety.
9. Integration tests against the real CGenFF .prm file (skipped if absent).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal .prm text fixture covering all bonded sections + NONBONDED/NBFIX
# ---------------------------------------------------------------------------

_MINI_PRM = """\
* minimal CGenFF fragment for testing
*
BONDS
CT3   CL     300.00     1.7700 ! CCl4
HT    OT     450.00     0.9572 ! TIP3P water

ANGLES
HT   OT   HT     55.00    104.52           ! TIP3P
CG311  CG321  HGA2     33.43    110.10   22.53   2.17900 ! PROT UB

DIHEDRALS
CT3  CT3  CT3  CT3    0.1500 3    0.00 ! alkane

IMPROPERS
CG2O1  CG2R61  OG2D1  NG2S1    20.00   0   0.00 ! amide

NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -
cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5

CT3      0.0       -0.0780     2.0400 ! PROT
HT       0.0       -0.0460     0.2245 ! TIP3P

NBFIX
OT   CT3    -0.1521     3.5368 ! water-alkane

HBOND CUTHB 0.5

END
"""


@pytest.fixture()
def mini_prm(tmp_path: Path) -> Path:
    p = tmp_path / "mini.prm"
    p.write_text(_MINI_PRM, encoding="utf-8")
    return p


@pytest.fixture()
def dst(tmp_path: Path) -> Path:
    return tmp_path / "zeroed_psf_ready.prm"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make(src: Path, dst: Path, **kw) -> tuple[Path, str]:
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_zeroed_psf_ready_prm,
    )

    result = write_zeroed_psf_ready_prm(src, dst, **kw)
    return result, dst.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. VDW sections are absent
# ---------------------------------------------------------------------------


def test_nonbonded_section_absent(mini_prm: Path, dst: Path):
    """NONBONDED header and all its atom rows must not appear."""
    _, text = _make(mini_prm, dst)
    assert "NONBONDED" not in text, "NONBONDED section header found in output"
    assert "nbxmod" not in text.lower()
    assert "cutnb" not in text.lower()


def test_nbfix_section_absent(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "NBFIX" not in text


def test_hbond_section_absent(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "HBOND" not in text
    assert "CUTHB" not in text


def test_vdw_atom_rows_absent(mini_prm: Path, dst: Path):
    """Epsilon and Rmin values from the NONBONDED block must not appear."""
    _, text = _make(mini_prm, dst)
    # CT3 and HT rows from NONBONDED
    assert "-0.0780" not in text
    assert "-0.0460" not in text


def test_nbfix_row_absent(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "-0.1521" not in text
    assert "3.5368" not in text


# ---------------------------------------------------------------------------
# 2. Bonded force constants are zeroed
# ---------------------------------------------------------------------------


def test_bond_kb_zeroed(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "300.00" not in text, "BOND Kb should be zeroed"
    assert "450.00" not in text, "BOND Kb should be zeroed"


def test_angle_k_zeroed(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "55.00" not in text, "ANGLE K should be zeroed"
    assert "33.43" not in text, "ANGLE K should be zeroed"


def test_angle_kub_zeroed(mini_prm: Path, dst: Path):
    """Urey-Bradley force constant (Kub) must be zeroed."""
    _, text = _make(mini_prm, dst)
    assert "22.53" not in text, "UB Kub should be zeroed"


def test_dihedral_vn_zeroed(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "0.1500" not in text, "Dihedral Vn should be zeroed"


def test_improper_k_zeroed(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "20.00" not in text, "Improper K should be zeroed"


# ---------------------------------------------------------------------------
# 3. Equilibrium geometry is preserved
# ---------------------------------------------------------------------------


def test_bond_r0_preserved(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "1.7700" in text, "BOND r0 must be preserved"
    assert "0.9572" in text, "BOND r0 must be preserved"


def test_angle_theta0_preserved(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "104.52" in text, "ANGLE theta0 must be preserved"
    assert "110.10" in text, "ANGLE theta0 must be preserved"


def test_angle_rub_preserved(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "2.17900" in text, "UB Rub must be preserved"


def test_dihedral_multiplicity_and_phase_preserved(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    # multiplicity=3, phase=0.00 -- these must survive
    assert " 3 " in text or text.endswith("3    0.00") or "3    0.00" in text


# ---------------------------------------------------------------------------
# 4. Header stamp
# ---------------------------------------------------------------------------


def test_header_stamp_present(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "MMML PSF-ready overlay" in text


def test_header_mentions_source_file(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert mini_prm.name in text


def test_header_vdw_zero_explanation(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "VDW=0" in text or "IMNB=0" in text


# ---------------------------------------------------------------------------
# 5. Optional note written into header
# ---------------------------------------------------------------------------


def test_custom_note_in_header(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst, note="generated for DCM solvated box run")
    assert "generated for DCM solvated box run" in text


def test_no_note_leaves_header_without_garbage(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    # "None" should never appear in the header
    assert "*  None" not in text


# ---------------------------------------------------------------------------
# 6. Return value
# ---------------------------------------------------------------------------


def test_returns_dst_path(mini_prm: Path, dst: Path):
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_zeroed_psf_ready_prm,
    )

    result = write_zeroed_psf_ready_prm(mini_prm, dst)
    assert result == dst


def test_dst_file_exists_after_call(mini_prm: Path, dst: Path):
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_zeroed_psf_ready_prm,
    )

    write_zeroed_psf_ready_prm(mini_prm, dst)
    assert dst.is_file()


# ---------------------------------------------------------------------------
# 7. Parent directories created automatically
# ---------------------------------------------------------------------------


def test_creates_parent_directories(mini_prm: Path, tmp_path: Path):
    from mmml.interfaces.pycharmmInterface.charmm_prm_zero import (
        write_zeroed_psf_ready_prm,
    )

    nested = tmp_path / "a" / "b" / "c" / "zeroed.prm"
    assert not nested.parent.exists()
    write_zeroed_psf_ready_prm(mini_prm, nested)
    assert nested.is_file()


# ---------------------------------------------------------------------------
# 8. Bonded section headers are retained (required for READ PARAM APPEND)
# ---------------------------------------------------------------------------


def test_bonds_section_header_present(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "BONDS" in text


def test_angles_section_header_present(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "ANGLES" in text


def test_dihedrals_section_header_present(mini_prm: Path, dst: Path):
    _, text = _make(mini_prm, dst)
    assert "DIHEDRALS" in text


# ---------------------------------------------------------------------------
# 9. Real CGenFF .prm file (skipped when file absent, e.g. CI without data)
# ---------------------------------------------------------------------------


def _cgenff_prm() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"


@pytest.mark.skipif(not _cgenff_prm().is_file(), reason="CGenFF .prm not present")
def test_real_cgenff_no_nonbonded_section(tmp_path: Path):
    src = _cgenff_prm()
    dst = tmp_path / "zeroed_psf_ready.prm"
    _, text = _make(src, dst)
    assert "\nNONBONDED" not in text
    assert "\nNBFIX" not in text
    assert "\nHBOND" not in text
    assert "nbxmod" not in text.lower()


@pytest.mark.skipif(not _cgenff_prm().is_file(), reason="CGenFF .prm not present")
def test_real_cgenff_known_bond_kb_zeroed(tmp_path: Path):
    """CG1N1-NG1T1 bond Kb=1053 from ACN should be zeroed; r0=1.18 kept."""
    src = _cgenff_prm()
    dst = tmp_path / "zeroed_psf_ready.prm"
    _, text = _make(src, dst)
    assert "1053" not in text
    assert "1.1800" in text


@pytest.mark.skipif(not _cgenff_prm().is_file(), reason="CGenFF .prm not present")
def test_real_cgenff_header_stamp(tmp_path: Path):
    src = _cgenff_prm()
    dst = tmp_path / "zeroed_psf_ready.prm"
    _, text = _make(src, dst)
    assert "MMML PSF-ready overlay" in text
    assert "par_all36_cgenff.prm" in text
