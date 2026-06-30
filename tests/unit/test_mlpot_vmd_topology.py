"""Unit tests for MPI-safe VMD topology export in MLpot setup."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest


def test_save_cluster_topology_for_vmd_uses_ase_not_charmm_coor_pdb(tmp_path):
    pytest.importorskip("ase")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    with (
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.sync_charmm_positions",
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.write_charmm_psf",
            return_value=tmp_path / "model.psf",
        ) as mock_psf,
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup._write_vmd_pdb_from_positions",
        ) as mock_pdb,
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.topology_recovery.capture_topology_fingerprint_from_charmm",
        ) as mock_fp,
        mock.patch("pycharmm.psf.get_iblo_inb", return_value=([], [])),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.save_topology_sidecar",
        ),
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
            return_value=(0, 1),
        ),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            save_cluster_topology_for_vmd,
        )

        out = save_cluster_topology_for_vmd(
            tmp_path,
            positions,
            stem="model",
            title="pre-MLpot cluster",
        )

    mock_psf.assert_called_once()
    mock_pdb.assert_called_once_with(
        tmp_path / "model.pdb",
        positions,
        title="pre-MLpot cluster",
    )
    mock_fp.assert_called_once()
    assert out["psf"] == tmp_path / "model.psf"
    assert out["pdb"] == (tmp_path / "model.pdb").resolve()


def test_write_vmd_pdb_from_positions_writes_ase_pdb(tmp_path):
    pytest.importorskip("ase")
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.utils.get_Z_from_psf",
        return_value=np.array([6, 1], dtype=int),
    ):
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            _write_vmd_pdb_from_positions,
        )

        path = _write_vmd_pdb_from_positions(tmp_path / "model.pdb", positions)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "ATOM" in text
