"""Unit tests for literature CIF → CHARMM crystal supercell builder."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

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


def test_literature_preset_uses_bundled_template_when_no_explicit_monomer():
    from mmml.interfaces import crystal_charmm

    expected = crystal_charmm.default_make_res_monomer_pdb("DCM").resolve()
    with mock.patch.object(
        crystal_charmm,
        "build_charmm_literature_supercell",
    ) as build:
        crystal_charmm.build_literature_charmm_supercell(
            "dcm",
            supercell_reps=(1, 1, 1),
            min_box_side_a=None,
        )

    assert Path(build.call_args.kwargs["monomer_pdb"]) == expected


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


def test_literature_benzene_pdb_keeps_benz_not_ben():
    """4-char CGenFF RESN must survive PDB export for CHARMM GENERATE."""
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        _residue_sequence_from_pdb,
    )

    result = build_literature_charmm_supercell(
        "benz",
        supercell_reps=(1, 1, 1),
        min_box_side_a=None,
    )
    text = result.pdb_path.read_text(encoding="utf-8")
    assert "BENZ" in text
    assert " BEN " not in text
    seq = _residue_sequence_from_pdb(result.pdb_path)
    assert seq
    assert all(r == "BENZ" for r in seq)


def test_acetone_presets_cover_the_ordered_phases():
    """Allan et al.'s Pbca phases, plus the metastable Cmcm one.

    The 15 kbar phase is deliberately absent: its methyls are rotationally
    disordered, so there is no single set of hydrogen positions to map onto
    CGenFF ACO.
    """
    from mmml.interfaces.crystal_charmm import LITERATURE_CRYSTAL_PRESETS

    acetone = {k: v for k, v in LITERATURE_CRYSTAL_PRESETS.items() if k.startswith("aco")}
    assert set(acetone) == {"aco", "aco5k", "aco110k", "acocmcm"}
    for spec in acetone.values():
        assert spec["residue"] == "ACO"
        assert Path(spec["cif"]()).is_file()
    assert LITERATURE_CRYSTAL_PRESETS["aco"]["space_group"] == 61
    assert LITERATURE_CRYSTAL_PRESETS["acocmcm"]["space_group"] == 63


def test_build_literature_acetone_unit_cell_matches_cif():
    """Mapping 16 molecules onto CHARMM atom names must not move any atom."""
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell
    from mmml.interfaces.crystal_reference import metrics_from_cif
    from mmml.paths import default_acetone_crystal_cif

    lit = metrics_from_cif(default_acetone_crystal_cif("pbca_150k"), space_group=61)
    result = build_literature_charmm_supercell(
        "aco",
        supercell_reps=(1, 1, 1),
        min_box_side_a=None,
    )
    assert result.n_molecules == 16
    assert result.residue == "ACO"
    assert result.density_g_cm3 == pytest.approx(lit.density_g_cm3, rel=1e-4)
    for axis in range(3):
        assert result.cell_lengths_a[axis] == pytest.approx(lit.lengths_a[axis], rel=1e-3)

    text = result.pdb_path.read_text(encoding="utf-8")
    assert "CRYST1" in text
    assert " ACO " in text


def test_acetone_lattice_energy_survives_the_charmm_name_mapping():
    """A geometry-preserving rename must leave the lattice energy untouched.

    Atom names are assigned by Hungarian matching against the make-res template,
    which is exactly the kind of step that can permute atoms between molecules
    without anything looking wrong.
    """
    from ase.io import read

    from mmml.analysis.acetone_crystal import read_acetone_phase
    from mmml.analysis.lattice_energy import crystal_lattice_energy
    from mmml.interfaces.crystal_charmm import build_literature_charmm_supercell

    direct = read_acetone_phase("pbca_150k")
    mapped = read(
        str(
            build_literature_charmm_supercell(
                "aco", supercell_reps=(1, 1, 1), min_box_side_a=None
            ).pdb_path
        )
    )

    energies = [
        crystal_lattice_energy(
            atoms.get_positions(), atoms.get_atomic_numbers(), atoms.cell.array, cutoff_A=10.0
        ).e_lattice
        for atoms in (direct, mapped)
    ]
    # Not exact: a PDB stores coordinates to 0.001 A, which is worth about a
    # thousandth of a kcal/mol here. A permutation error would cost far more.
    assert energies[1] == pytest.approx(energies[0], abs=5e-3)


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


def test_build_crystal_box_size_and_write_charmm_parser():
    from mmml.cli.misc.build_crystal import effective_min_box_side_a, parse_args

    args = parse_args(
        [
            "--literature",
            "benz",
            "--box-size",
            "30",
            "--write-charmm",
            "-o",
            "/tmp/benz30.extxyz",
        ]
    )
    assert args.box_size == pytest.approx(30.0)
    assert args.write_charmm is True
    assert effective_min_box_side_a(args) == pytest.approx(30.0)

    args_alias = parse_args(
        ["--literature", "benz", "--side-length", "32.5", "-o", "/tmp/b.pdb"]
    )
    assert args_alias.box_size == pytest.approx(32.5)
    assert effective_min_box_side_a(args_alias) == pytest.approx(32.5)


def test_effective_min_box_side_defaults_to_min_box_side():
    from mmml.cli.misc.build_crystal import effective_min_box_side_a, parse_args

    args = parse_args(["--literature", "dcm", "-o", "/tmp/dcm.pdb"])
    assert args.box_size is None
    assert effective_min_box_side_a(args) == pytest.approx(args.min_box_side)


def test_build_crystal_literature_uses_box_size_for_auto_reps():
    from mmml.cli.misc import build_crystal as bc
    from mmml.interfaces.crystal_charmm import CharmmLiteratureCrystalResult

    fake = CharmmLiteratureCrystalResult(
        atoms=object(),
        pdb_path=Path("/tmp/fake.pdb"),
        residue="BENZ",
        supercell_reps=(3, 3, 3),
        n_molecules=54,
        density_g_cm3=1.2,
        cell_lengths_a=(30.0, 31.0, 32.0),
        cell_angles_deg=(90.0, 90.0, 90.0),
        monomer_pdb=Path("/tmp/benz.pdb"),
    )
    with mock.patch.object(
        bc, "build_literature_charmm_supercell", return_value=fake
    ) as build:
        with mock.patch.object(bc, "write_ase_structure"):
            with mock.patch.object(bc, "_maybe_write_charmm", return_value=0):
                rc = bc.main(
                    [
                        "--literature",
                        "benz",
                        "--box-size",
                        "30",
                        "-o",
                        "/tmp/benz30.extxyz",
                    ]
                )
    assert rc == 0
    assert build.call_args.kwargs["min_box_side_a"] == pytest.approx(30.0)
    assert build.call_args.kwargs["supercell_reps"] is None


def test_build_crystal_write_charmm_calls_helper(tmp_path):
    from mmml.cli.misc import build_crystal as bc
    from mmml.interfaces.crystal_charmm import (
        CharmmLiteratureCrystalResult,
        CrystalCharmmTopologyPaths,
    )

    pdb = tmp_path / "lit.pdb"
    pdb.write_text("CRYST1\nEND\n", encoding="utf-8")
    out = tmp_path / "benz30.extxyz"
    fake = CharmmLiteratureCrystalResult(
        atoms=mock.Mock(
            cell=mock.Mock(cellpar=lambda: (30.0, 31.0, 32.0, 90.0, 90.0, 90.0))
        ),
        pdb_path=pdb,
        residue="BENZ",
        supercell_reps=(2, 2, 2),
        n_molecules=16,
        density_g_cm3=1.2,
        cell_lengths_a=(30.0, 31.0, 32.0),
        cell_angles_deg=(90.0, 90.0, 90.0),
        monomer_pdb=tmp_path / "benz.pdb",
    )
    topo = CrystalCharmmTopologyPaths(
        pdb=tmp_path / "benz30.pdb",
        psf=tmp_path / "benz30.psf",
        crd=tmp_path / "benz30.crd",
        box_json=tmp_path / "benz30_box.json",
    )
    with mock.patch.object(bc, "build_literature_charmm_supercell", return_value=fake):
        with mock.patch.object(bc, "write_ase_structure"):
            with mock.patch.object(
                bc, "write_crystal_charmm_topology", return_value=topo
            ) as write_topo:
                rc = bc.main(
                    [
                        "--literature",
                        "benz",
                        "--box-size",
                        "35",
                        "--write-charmm",
                        "--supercell",
                        "2,2,2",
                        "-o",
                        str(out),
                    ]
                )
    assert rc == 0
    assert write_topo.call_args.args[0] == pdb
    assert write_topo.call_args.kwargs["side_length_A"] == pytest.approx(35.0)
    assert write_topo.call_args.kwargs["residue"] == "BENZ"
    assert write_topo.call_args.kwargs["n_molecules"] == 16


def test_charmm_side_rejects_box_smaller_than_edges():
    from mmml.cli.misc.build_crystal import _charmm_side_length_a, parse_args

    args = parse_args(
        ["--literature", "benz", "--box-size", "20", "-o", "/tmp/b.extxyz"]
    )
    with pytest.raises(ValueError, match="smaller than the largest supercell"):
        _charmm_side_length_a(args, (25.0, 26.0, 27.0))


def test_build_crystal_literature_main(tmp_path):
    from mmml.cli.misc.build_crystal import main

    out = tmp_path / "dcm_lit.pdb"
    rc = main(["--literature", "dcm", "--supercell", "1,1,1", "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "CRYST1" in out.read_text(encoding="utf-8")
