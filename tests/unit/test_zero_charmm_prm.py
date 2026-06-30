"""Unit tests for scripts/zero_charmm_prm.py."""

from pathlib import Path

from scripts.zero_charmm_prm import bonded_only_prm_text, zero_prm_line, zero_prm_text


def test_zero_bond_keeps_r0():
    line = "CG1N1  NG1T1  1053.00     1.1800 ! ACN"
    out = zero_prm_line(line, "BONDS")
    assert "0.0" in out
    assert "1.1800" in out
    assert "1053.00" not in out


def test_zero_angle_keeps_theta0():
    line = "CG321  CG1N1  NG1T1    21.20    180.00 ! CYU"
    out = zero_prm_line(line, "ANGLES")
    assert "0.0" in out
    assert "180.00" in out
    assert "21.20" not in out


def test_zero_dihedral_keeps_multiplicity_and_phase():
    line = "NG1T1  CG1N1  CG2R61 CG2R61  0.0100 2    0.00 ! CNP2"
    out = zero_prm_line(line, "DIHEDRALS")
    assert "0.0 2" in out or "0.0  2" in out.replace("  ", " ")
    assert "0.0100" not in out


def test_zero_nonbonded_keeps_rmin():
    line = "HGA1     0.0       -0.0450     1.3400 ! alkane"
    out = zero_prm_line(line, "NONBONDED")
    assert "0.0" in out
    assert "1.3400" in out
    assert "-0.0450" not in out


def test_bonded_only_skips_nonbonded():
    line = "HGA1     0.0       -0.0450     1.3400 ! alkane"
    out = zero_prm_line(line, "NONBONDED", skip_sections=frozenset({"NONBONDED"}))
    assert "-0.0450" in out


def test_zero_prm_text_omits_nonbonded_header():
    text = (
        "NONBONDED nbxmod  5 atom cdiel fshift vatom vdistance vfswitch -\n"
        "cutnb 14.0 ctofnb 12.0 ctonnb 10.0 eps 1.0 e14fac 1.0 wmin 1.5\n"
        "\n"
        "HGA1     0.0       -0.0450     1.3400 ! alkane\n"
        "END\n"
    )
    out = zero_prm_text(text)
    assert "nbxmod" not in out
    assert "cutnb" not in out
    assert "HGA1" in out
    assert "0.0" in out
    assert "-0.0450" not in out


def test_zero_prm_text_bonded_only_omits_nonbonded_section():
    text = (
        "BONDS\n"
        "HT    OT    450.0       0.9572\n"
        "NONBONDED nbxmod  5 atom cdiel\n"
        "HGA1     0.0       -0.0450     1.3400\n"
        "END\n"
    )
    out = zero_prm_text(text, bonded_only=True)
    assert "NONBONDED" not in out
    assert "HGA1" not in out
    assert "450.0" not in out


def test_zero_nbfix():
    line = "OG2D1     CLGR1    -0.20        3.40   ! NMA"
    out = zero_prm_line(line, "NBFIX")
    assert "0.0" in out
    assert "3.40" in out
    assert "-0.20" not in out


def test_comments_and_sections_passthrough():
    text = (
        "BONDS\n"
        "HT    OT    450.0       0.9572  ! water\n"
        "! comment\n"
        "\n"
        "ANGLES\n"
        "HT   OT   HT     55.0      104.52\n"
        "END\n"
    )
    out = zero_prm_text(text)
    assert "! comment" in out
    assert "450.0" not in out
    assert "0.9572" in out
    assert "55.0" not in out
    assert "104.52" in out


def test_extract_bonded_only_keeps_constants():
    text = (
        "BONDS\n"
        "HT    OT    450.0       0.9572\n"
        "NONBONDED nbxmod  5 atom cdiel\n"
        "HGA1     0.0       -0.0450     1.3400\n"
        "END\n"
    )
    out = bonded_only_prm_text(text, zero_constants=False)
    assert "NONBONDED" not in out
    assert "HGA1" not in out
    assert "450.0" in out


def test_zero_cgenff_prm_file(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    src = repo / "mmml/data/charmm/par_all36_cgenff.prm"
    if not src.is_file():
        return
    from scripts.zero_charmm_prm import zero_prm_file

    dst = tmp_path / "zeroed.prm"
    zero_prm_file(src, dst)
    text = dst.read_text()
    assert "ZEROED COPY" in text
    assert "1053.00" not in text  # sample bond Kb from ACN
    assert "1.1800" in text  # equilibrium kept
