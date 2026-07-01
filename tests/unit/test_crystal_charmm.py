"""Unit tests for literature CIF → CHARMM crystal supercell builder."""

from __future__ import annotations

import pytest


def test_suggest_supercell_reps_dcm_unit_cell():
    from mmml.interfaces.crystal_charmm import suggest_supercell_reps
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_dcm_crystal_cif

    lengths = metrics_from_cif(default_dcm_crystal_cif()).lengths_a
    reps = suggest_supercell_reps(lengths, min_box_side_a=28.0)
    assert reps == (8, 4, 3)
    scaled = tuple(lengths[i] * reps[i] for i in range(3))
    assert min(scaled) >= 28.0 - 0.01


def test_build_literature_dcm_unit_cell_matches_cif():
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_dcm_crystal_cif

    lit = metrics_from_cif(default_dcm_crystal_cif(), space_group=60)
    result = build_literature_charmm_supercell(
        "dcm",
        supercell_reps=(1, 1, 1),
        min_box_side_a=None,
    )
    assert result.n_molecules == 4
    assert result.residue == "DCM"
    assert result.density_g_cm3 == pytest.approx(lit.density_g_cm3, rel=1e-4)
    for axis in range(3):
        assert result.cell_lengths_a[axis] == pytest.approx(
            lit.lengths_a[axis], rel=1e-3
        )


def test_build_literature_dcm_supercell_density_and_count():
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell

    result = build_literature_charmm_supercell(
        "dcm",
        supercell_reps=(2, 2, 2),
        min_box_side_a=None,
    )
    assert result.n_molecules == 4 * 8
    assert result.density_g_cm3 == pytest.approx(1.976, rel=1e-3)
    assert result.pdb_path.is_file()
    text = result.pdb_path.read_text(encoding="utf-8")
    assert "CRYST1" in text
    assert "DCM" in text
    assert "CL1" in text


def test_build_literature_benzene_supercell_auto_reps():
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell

    result = build_literature_charmm_supercell(
        "benz",
        supercell_reps=None,
        min_box_side_a=28.0,
    )
    assert result.n_molecules == 2 * int(
        result.supercell_reps[0]
        * result.supercell_reps[1]
        * result.supercell_reps[2]
    )
    assert min(result.cell_lengths_a) >= 28.0 - 0.05
    assert result.density_g_cm3 == pytest.approx(1.202, rel=1e-3)


def test_charmm_crystal_metrics_from_preset():
    from mmml.interfaces.crystal_charmm import charmm_crystal_metrics_from_preset
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_dcm_crystal_cif

    lit = metrics_from_cif(default_dcm_crystal_cif(), space_group=60)
    m = charmm_crystal_metrics_from_preset("dcm")
    assert m.label == "make-res+CIF"
    assert m.natoms == lit.natoms
    assert m.density_g_cm3 == pytest.approx(lit.density_g_cm3, rel=1e-4)


def test_build_crystal_literature_cli_parser():
    from mmml.cli.misc.build_crystal import parse_args

    args = parse_args(["--literature", "dcm", "-o", "/tmp/dcm_crystal.pdb"])
    assert args.literature == "dcm"
    assert args.molecule is None


def test_build_crystal_literature_main(tmp_path):
    from mmml.cli.misc.build_crystal import main

    out = tmp_path / "dcm_lit.pdb"
    rc = main(["--literature", "dcm", "--supercell", "1,1,1", "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "CRYST1" in out.read_text(encoding="utf-8")
