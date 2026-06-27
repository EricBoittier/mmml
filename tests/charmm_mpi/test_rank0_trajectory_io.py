"""Rank-0 trajectory I/O gating for MPI workflows."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from mmml.interfaces.pycharmmInterface.mlpot.dynamics import CharmmTrajectoryFiles
from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import _io_for_stage


def test_gate_charmm_trajectory_io_rank0_keeps_path():
    from mmml.interfaces.pycharmmInterface.mpi_rank_io import gate_charmm_trajectory_io

    io = CharmmTrajectoryFiles(trajectory=Path("/tmp/heat.dcd"))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(0, 4),
    ):
        out = gate_charmm_trajectory_io(io)
    assert out.trajectory == Path("/tmp/heat.dcd")


def test_gate_charmm_trajectory_io_nonzero_clears_path():
    from mmml.interfaces.pycharmmInterface.mpi_rank_io import gate_charmm_trajectory_io

    io = CharmmTrajectoryFiles(trajectory=Path("/tmp/heat.dcd"))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(2, 4),
    ):
        out = gate_charmm_trajectory_io(io)
    assert out.trajectory is None


def test_io_for_stage_gates_dcd_on_nonzero_rank(tmp_path):
    paths = {
        "heat_dcd": tmp_path / "heat.dcd",
        "heat_res": tmp_path / "heat.res",
        "nve_dcd": tmp_path / "nve.dcd",
        "nve_res": tmp_path / "nve.res",
        "equi_dcd": tmp_path / "equi.dcd",
        "equi_res": tmp_path / "equi.res",
        "prod_dcd": tmp_path / "prod.dcd",
        "prod_res": tmp_path / "prod.res",
    }
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(1, 2),
    ):
        io = _io_for_stage("heat", paths)
    assert io.restart_write == paths["heat_res"]
    assert io.trajectory is None


def test_reset_stage_trajectory_skips_nonzero_rank(tmp_path):
    from mmml.interfaces.pycharmmInterface.mlpot.staged_workflow import (
        _reset_stage_trajectory,
    )

    dcd = tmp_path / "heat.dcd"
    dcd.write_text("fake")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(3, 4),
    ):
        _reset_stage_trajectory(dcd)
    assert dcd.is_file()


def test_rank0_trajectory_path_helper():
    from mmml.interfaces.pycharmmInterface.mpi_rank_io import rank0_trajectory_path

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(0, 1),
    ):
        assert rank0_trajectory_path("/tmp/x.dcd") == Path("/tmp/x.dcd")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(2, 4),
    ):
        assert rank0_trajectory_path("/tmp/x.dcd") is None
        assert rank0_trajectory_path(None) is None
