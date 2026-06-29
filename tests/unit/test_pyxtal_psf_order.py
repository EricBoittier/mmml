"""Unit tests for PyXtal ASE -> PSF coordinate mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


def test_assign_ase_cluster_to_psf_order_homogeneous(tmp_path):
    from mmml.interfaces.pyxtal_placement import assign_ase_cluster_to_psf_order

    tmpl_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tmpl_names = ["C1", "H1", "H2", "H3"]
    tmpl_z = np.array([6, 1, 1, 1], dtype=int)
    residue_geometries = {
        "DCM": (tmpl_pos, tmpl_names, tmpl_z),
    }

    mol_a = tmpl_pos + np.array([0.0, 0.0, 0.0])
    mol_b = tmpl_pos + np.array([5.0, 0.0, 0.0])
    positions = np.vstack([mol_a, mol_b])
    symbols = ["C", "H", "H", "H", "C", "H", "H", "H"]
    atoms = Atoms(symbols=symbols, positions=positions, cell=np.diag([20.0, 20.0, 20.0]), pbc=True)

    psf_names = tmpl_names + tmpl_names
    atoms_per_list = [4, 4]
    ordered_residue_names = ["DCM", "DCM"]

    out = assign_ase_cluster_to_psf_order(
        atoms,
        psf_atom_names=psf_names,
        atoms_per_list=atoms_per_list,
        ordered_residue_names=ordered_residue_names,
        residue_geometries=residue_geometries,
        pdb_path=tmp_path / "map.pdb",
    )

    assert out.shape == (8, 3)
    np.testing.assert_allclose(out[:4], mol_a)
    np.testing.assert_allclose(out[4:], mol_b)


def test_assign_ase_cluster_trims_excess_atoms(tmp_path):
    from mmml.interfaces.pyxtal_placement import assign_ase_cluster_to_psf_order

    tmpl_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tmpl_names = ["C1", "H1", "H2", "H3"]
    tmpl_z = np.array([6, 1, 1, 1], dtype=int)
    residue_geometries = {"DCM": (tmpl_pos, tmpl_names, tmpl_z)}

    mol_a = tmpl_pos
    mol_b = tmpl_pos + np.array([5.0, 0.0, 0.0])
    positions = np.vstack([mol_a, mol_b])
    symbols = ["C", "H", "H", "H", "C", "H", "H", "H"]
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=np.diag([20.0, 20.0, 20.0]),
        pbc=True,
    )

    out = assign_ase_cluster_to_psf_order(
        atoms,
        psf_atom_names=tmpl_names,
        atoms_per_list=[4],
        ordered_residue_names=["DCM"],
        residue_geometries=residue_geometries,
        pdb_path=tmp_path / "trim.pdb",
        trim_to_composition=True,
    )
    np.testing.assert_allclose(out, mol_a)


def test_resolve_pyxtal_use_and_packmol_exclusion():
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_use
    from mmml.interfaces.pyxtal_placement import resolve_pyxtal_use

    assert resolve_pyxtal_use(composition="MEOH:4", pyxtal=True)
    assert not resolve_pyxtal_use(composition="MEOH:4", pyxtal=False)
    assert not resolve_packmol_use(
        composition="MEOH:4", packmol=None, pyxtal=True
    )
    assert not resolve_packmol_use(composition="MEOH:4", packmol=None, pyxtal=None)
    assert resolve_packmol_use(composition="MEOH:4", packmol=True, pyxtal=None)


def test_validate_pyxtal_cluster_args_rejects_packmol_combo():
    from mmml.interfaces.pyxtal_placement import validate_pyxtal_cluster_args

    with pytest.raises(ValueError, match="Cannot combine"):
        validate_pyxtal_cluster_args(
            composition="MEOH:4",
            pyxtal=True,
            packmol=True,
        )
