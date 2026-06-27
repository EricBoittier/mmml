"""Unit tests for mpi-check and mpi_rank_io."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from mmml.cli.run.mpi_check import run_mpi_check, render_mpi_check_report
from mmml.interfaces.pycharmmInterface import mpi_rank_io


def test_run_mpi_check_missing_charmm_lib(monkeypatch):
    monkeypatch.delenv("CHARMM_LIB_DIR", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._charmm_lib_path",
        return_value=None,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
        return_value=False,
    ):
        report = run_mpi_check()
    assert report.ok is False
    assert any("CHARMM_LIB_DIR" in e for e in report.errors)


def test_run_mpi_check_mpi_linked_warns_without_mpirun(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    mpirun = tmp_path / "mpirun"
    mpirun.write_text("#!/bin/sh\nexit 0\n")
    mpirun.chmod(0o755)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._charmm_lib_path",
        return_value=lib,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
        return_value=mpirun.resolve(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
        return_value=False,
    ):
        report = run_mpi_check()
    assert report.ok is True
    assert any("Not under mpirun" in w for w in report.warnings)


def test_render_mpi_check_report_includes_status():
    from mmml.cli.run.mpi_check import MpiCheckReport

    text = render_mpi_check_report(MpiCheckReport(ok=True, charmm_links_mpi=True))
    assert "OK" in text
    assert "MPI-linked" in text


def test_mpi_rank_io_rank0_only():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(1, 4),
    ):
        assert mpi_rank_io.is_mpi_rank_zero() is False
        assert mpi_rank_io.rank0_only("x") is None


def test_mpi_rank_io_rank0_write_text(tmp_path):
    path = tmp_path / "out.txt"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(0, 1),
    ):
        assert mpi_rank_io.rank0_write_text(path, "hello") is True
    assert path.read_text() == "hello"

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mpi_rank_io.mpi_rank_size",
        return_value=(2, 4),
    ):
        assert mpi_rank_io.rank0_write_text(path, "nope") is False
    assert path.read_text() == "hello"


def test_mpi_check_tier2_flag(monkeypatch, tmp_path):
    from mmml.cli.run.mpi_check import main

    mpirun = tmp_path / "mpirun"
    mpirun.write_text("#!/bin/sh\nexit 0\n")
    mpirun.chmod(0o755)
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._charmm_lib_path",
        return_value=tmp_path / "libcharmm.so",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
        return_value=mpirun.resolve(),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 2),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_name",
        return_value="gpu",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_local_gpu_count",
        return_value=0,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
        return_value=True,
    ):
        assert main(["--tier2"]) == 1


def test_maybe_rerun_liquid_box_subcommand(monkeypatch, tmp_path):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

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
        "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_serial_charmm_mpi_env",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as mock_run:
        code = charmm_mpi.maybe_rerun_mmml_under_mpirun(
            ["--composition", "DCM:10", "-o", "/tmp/x"],
            subcommand="liquid-box",
        )
    assert code == 0
    cmd = mock_run.call_args.args[0]
    assert "liquid-box" in cmd
    assert cmd[cmd.index("liquid-box") + 1] == "--composition"
