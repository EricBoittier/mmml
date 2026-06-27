"""Tests for rank-0 MLpot MPI bridge."""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot import mpi_bridge


def test_mpi_rank_size_defers_mpi4py_before_init(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge._mpi4py_is_initialized",
        return_value=False,
    ):
        rank, size = mpi_bridge.mpi_rank_size()
    assert (rank, size) == (1, 4)


def test_mpi_rank_size_serial_defaults():
    with mock.patch.dict(os.environ, {}, clear=True):
        rank, size = mpi_bridge.mpi_rank_size()
    assert rank == 0
    assert size == 1


def test_mpi_rank_size_from_openmpi_env():
    env = {
        "OMPI_COMM_WORLD_RANK": "2",
        "OMPI_COMM_WORLD_SIZE": "4",
    }
    import sys

    with mock.patch.dict(os.environ, env, clear=True):
        with mock.patch.dict(sys.modules, {"mpi4py": None}):
            rank, size = mpi_bridge.mpi_rank_size()
    assert rank == 2
    assert size == 4


def test_mpi_rank_size_prefers_launcher_env_when_mpi4py_singleton():
    """CHARMM MPI_Init can leave mpi4py COMM_WORLD at size 1 under mpirun -np N."""
    env = {
        "OMPI_COMM_WORLD_RANK": "2",
        "OMPI_COMM_WORLD_SIZE": "4",
    }
    fake_comm = mock.MagicMock()
    fake_comm.Get_rank.return_value = 0
    fake_comm.Get_size.return_value = 1
    fake_mpi = mock.MagicMock()
    fake_mpi.COMM_WORLD = fake_comm

    with mock.patch.dict(os.environ, env, clear=True):
        with mock.patch.dict("sys.modules", {"mpi4py": fake_mpi}):
            with mock.patch(
                "mpi4py.MPI",
                fake_mpi,
                create=True,
            ):
                rank, size = mpi_bridge.mpi_rank_size()
    assert rank == 2
    assert size == 4


def test_mlpot_runs_on_all_ranks_when_spatial_mpi(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_SPATIAL_MPI", "1")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ):
        assert mpi_bridge.mlpot_runs_on_this_rank() is True


def test_mlpot_runs_on_rank0_only_when_size_gt_1(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_RANK0_BRIDGE", raising=False)
    monkeypatch.delenv("MMML_MLPOT_SPATIAL_MPI", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 4),
    ):
        assert mpi_bridge.mlpot_runs_on_this_rank() is True
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ):
        assert mpi_bridge.mlpot_runs_on_this_rank() is False


def test_mlpot_runs_all_ranks_when_bridge_disabled(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_RANK0_BRIDGE", "0")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(3, 4),
    ):
        assert mpi_bridge.mlpot_runs_on_this_rank() is True


def test_broadcast_mlpot_result_single_rank():
    forces = np.ones((3, 3), dtype=np.float64)
    out_f, out_e = mpi_bridge.broadcast_mlpot_result(forces, 1.5, 3)
    np.testing.assert_allclose(out_f, forces)
    assert out_e == 1.5


try:
    import mpi4py  # noqa: F401

    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False


@pytest.mark.skipif(not HAS_MPI4PY, reason="mpi4py not installed")
def test_broadcast_mlpot_result_mpi4py():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("needs MPI size >= 2")
    n = 4
    if rank == 0:
        forces = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
        energy = 42.0
    else:
        forces = None
        energy = 0.0
    out_f, out_e = mpi_bridge.broadcast_mlpot_result(forces, energy, n, comm=comm)
    if rank == 0:
        np.testing.assert_allclose(out_f, np.arange(n * 3, dtype=np.float64).reshape(n, 3))
    assert out_e == 42.0
