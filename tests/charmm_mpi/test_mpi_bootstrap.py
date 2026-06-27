"""CHARMM MPI bootstrap tests (mocked mpirun re-exec)."""

from __future__ import annotations

import os
from unittest import mock

from mmml.interfaces.pycharmmInterface import charmm_mpi


def test_maybe_rerun_run_pycharmm_subcommand(monkeypatch, tmp_path):
    monkeypatch.delenv("MMML_NO_MPI_RERUN", raising=False)
    mpirun = tmp_path / "mpirun"
    mpirun.write_text("#!/bin/sh\nexit 0\n")
    mpirun.chmod(0o755)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
        return_value=mpirun.resolve(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as mock_run:
        code = charmm_mpi.maybe_rerun_mmml_under_mpirun(
            ["--pdbfile", "x.pdb", "--cell", "40"],
            subcommand="run-pycharmm",
        )
    assert code == 0
    cmd = mock_run.call_args.args[0]
    assert cmd[3:9] == [
        "--mca",
        "pmix",
        "^ext3x",
        "--mca",
        "orte_abort_print_stack",
        "1",
    ]
    assert cmd[10:12] == ["-m", "mmml.cli.__main__"]
    assert cmd[12:14] == ["run-pycharmm", "--pdbfile"]
