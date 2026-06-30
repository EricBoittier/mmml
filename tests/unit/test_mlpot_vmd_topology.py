"""Unit tests for MPI-safe VMD topology export in MLpot setup."""

from __future__ import annotations

from pathlib import Path


def test_save_cluster_topology_for_vmd_avoids_charmm_coor_pdb():
    source = Path("mmml/interfaces/pycharmmInterface/mlpot/setup.py").read_text(
        encoding="utf-8"
    )
    block = source.split("def save_cluster_topology_for_vmd")[1].split("\ndef ")[0]
    assert "coor_pdb" not in block
    assert "_write_vmd_pdb_from_positions" in block
    assert "is_mpi_rank_zero" in block


def test_write_vmd_pdb_helper_uses_ase_not_pycharmm_write():
    source = Path("mmml/interfaces/pycharmmInterface/mlpot/setup.py").read_text(
        encoding="utf-8"
    )
    block = source.split("def _write_vmd_pdb_from_positions")[1].split("\ndef ")[0]
    assert "ase.io.write" in block
    assert "coor_pdb" not in block
    assert "get_Z_from_psf" in block
