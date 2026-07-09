"""Unit tests for the general peptide builder, solvator, and QC step."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.peptide_builder import (
    parse_sequence,
    qc_built_system,
)
from tests.conftest import can_import_pycharmm


def test_parse_sequence() -> None:
    # 1. Test standard list of residues
    assert parse_sequence(["ALA", "PHE", "GLY"]) == ["ALA", "PHE", "GLY"]
    assert parse_sequence(["ala", "Phe"]) == ["ALA", "PHE"]

    # 2. Test hyphen-separated string
    assert parse_sequence("ALA-PHE-GLY") == ["ALA", "PHE", "GLY"]
    assert parse_sequence("ala-phe-gly") == ["ALA", "PHE", "GLY"]

    # 3. Test space-separated string
    assert parse_sequence("ALA PHE GLY") == ["ALA", "PHE", "GLY"]
    assert parse_sequence("ala   phe gly") == ["ALA", "PHE", "GLY"]

    # 4. Test 1-letter code string
    assert parse_sequence("AFG") == ["ALA", "PHE", "GLY"]
    assert parse_sequence("afg") == ["ALA", "PHE", "GLY"]

    # 5. Test single 3-letter code
    assert parse_sequence("ALA") == ["ALA"]

    # 6. Test invalid codes
    with pytest.raises(ValueError, match="Could not parse residue sequence"):
        parse_sequence("XYZ")
    with pytest.raises(ValueError, match="Unknown residue code"):
        parse_sequence("ALA-XYZ")


def test_qc_validation_mock() -> None:
    # Create a mock PSF file for testing QC logic
    with tempfile.TemporaryDirectory() as tmpdir:
        psf_path = Path(tmpdir) / "mock.psf"
        
        # 4 atoms: C (heavy), H (hydrogen bonded to C), O (heavy), H (hydrogen bonded to O)
        # Atoms 0, 1 bonded; atoms 2, 3 bonded
        psf_content = """* MOCK PSF FOR TESTING QC
*

       4 !NATOM
       1 PEPT 1    ALA  CA   C       0.100000       12.0110           0
       2 PEPT 1    ALA  HA   H       0.090000        1.0080           0
       3 PEPT 1    ALA  C    C       0.200000       12.0110           0
       4 PEPT 1    ALA  O    O      -0.200000       15.9990           0

       2 !NBOND: bonds
       1       2       3       4
"""
        psf_path.write_text(psf_content, encoding="utf-8")

        # 1. Perfectly reasonable positions
        # C-H bond length = 1.09 Å
        # O-C distance is non-bonded, let's keep it at 3.0 Å
        # C-O bond length = 1.23 Å
        positions = np.array([
            [0.0, 0.0, 0.0],  # C (0)
            [0.0, 0.0, 1.09], # H (1) - bonded to 0
            [3.0, 0.0, 0.0],  # C (2)
            [3.0, 0.0, 1.23], # O (3) - bonded to 2
        ], dtype=np.float64)

        report = qc_built_system(positions, psf_path, check_energy=False)
        assert report.is_valid
        assert len(report.errors) == 0

        # 2. Test missing/placeholder coordinate (9999.0)
        bad_positions_placeholder = positions.copy()
        bad_positions_placeholder[0, 0] = 9999.0
        report = qc_built_system(bad_positions_placeholder, psf_path, check_energy=False)
        assert not report.is_valid
        assert any("placeholder" in err for err in report.errors)

        # 3. Test bad H-X bond length (too long, e.g. 2.0 Å)
        bad_positions_h_bond = positions.copy()
        bad_positions_h_bond[1, 2] = 2.0
        report = qc_built_system(bad_positions_h_bond, psf_path, check_energy=False)
        assert not report.is_valid
        assert any("bond length violation" in err for err in report.errors)
        assert len(report.details["bond_violations"]) == 1

        # 4. Test steric clash (non-bonded atoms 0 and 2 are 0.5 Å apart)
        bad_positions_clash = positions.copy()
        bad_positions_clash[2] = np.array([0.5, 0.0, 0.0]) # C(2) close to C(0)
        # update O(3) to be close to C(2) but not clash with C(0)
        bad_positions_clash[3] = np.array([0.5, 0.0, 1.23])
        report = qc_built_system(bad_positions_clash, psf_path, check_energy=False)
        assert not report.is_valid
        assert any("steric clash" in err for err in report.errors)
        assert len(report.details["steric_clashes"]) >= 1


@pytest.mark.skipif(not can_import_pycharmm(), reason="PyCHARMM is not available")
def test_live_peptide_builder_and_qc(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.peptide_builder import (
        build_peptide_in_charmm,
        solvate_peptide_in_charmm,
    )
    
    assert ensure_pycharmm_loaded()

    # Build trialanine in CHARMM
    peptide = build_peptide_in_charmm(
        "AAA",
        first_patch="ACE",
        last_patch="CT3",
        seg_name="PEPT",
        minimize=True,
        mini_steps=100,
        workdir=tmp_path,
    )

    assert peptide.n_atoms == 42 # trialanine has 42 atoms
    assert peptide.psf_path.is_file()
    assert peptide.pdb_path.is_file()

    # Solvate the peptide
    box = solvate_peptide_in_charmm(
        peptide,
        box_side_A=28.0,
        n_waters=12,
        workdir=tmp_path,
    )

    assert box.n_peptide_atoms == 42
    assert box.n_waters == 12
    assert box.psf_path.is_file()
    assert box.pdb_path.is_file()

    # QC check
    report = qc_built_system(
        box.positions,
        box.psf_path,
        box_side_A=box.box_side_A,
        check_energy=True,
    )

    # The built peptide + water box should be valid under QC checks
    assert report.is_valid, f"QC check failed: {report.errors}"
    assert len(report.errors) == 0
    assert report.details["charmm_energy"] < 1e5


def test_infer_charge_and_spin() -> None:
    from mmml.interfaces.pycharmmInterface.peptide_builder import infer_charge_and_spin_from_psf
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock PSF 1: Neutral, even electrons (Z_sum = 6+1+1+8 = 16)
        psf_path_even = Path(tmpdir) / "mock_even.psf"
        psf_even_content = """* MOCK EVEN PSF
*
       4 !NATOM
       1 PEPT 1    ALA  CA   C       0.100000       12.0110           0
       2 PEPT 1    ALA  HA1  H       0.100000        1.0080           0
       3 PEPT 1    ALA  HA2  H      -0.200000        1.0080           0
       4 PEPT 1    ALA  O    O       0.000000       15.9990           0

       0 !NBOND: bonds
"""
        psf_path_even.write_text(psf_even_content, encoding="utf-8")
        q, s = infer_charge_and_spin_from_psf(psf_path_even)
        assert q == 0
        assert s == 1.0

        # Mock PSF 2: Charged (+1), odd electrons (Z_sum = 16, Charge = 1, N_elec = 15)
        psf_path_odd = Path(tmpdir) / "mock_odd.psf"
        psf_odd_content = """* MOCK ODD PSF
*
       4 !NATOM
       1 PEPT 1    ALA  CA   C       0.400000       12.0110           0
       2 PEPT 1    ALA  HA1  H       0.300000        1.0080           0
       3 PEPT 1    ALA  HA2  H       0.200000        1.0080           0
       4 PEPT 1    ALA  O    O       0.000000       15.9990           0

       0 !NBOND: bonds
"""
        psf_path_odd.write_text(psf_odd_content, encoding="utf-8")
        q, s = infer_charge_and_spin_from_psf(psf_path_odd)
        assert q == 1
        assert s == 2.0


@pytest.mark.skipif(not can_import_pycharmm(), reason="PyCHARMM is not available")
def test_gas_phase_peptide_builder(tmp_path: Path) -> None:
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.peptide_builder import build_peptide_in_charmm
    
    assert ensure_pycharmm_loaded()
    
    # Build deca-alanine in gas phase (unsolvated)
    peptide = build_peptide_in_charmm(
        sequence="ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA",
        minimize=True,
        mini_steps=50,
        workdir=tmp_path,
    )
    
    assert peptide.n_atoms > 0
    assert peptide.psf_path.is_file()
    assert peptide.pdb_path.is_file()

