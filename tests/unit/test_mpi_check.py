"""Unit tests for mpi-check and mpi_rank_io."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from unittest import mock

import pytest

from mmml.cli.run.mpi_check import run_mpi_check, render_mpi_check_report
from mmml.interfaces.pycharmmInterface import mpi_rank_io


def _healthy_mpi_check_context(monkeypatch, tmp_path):
    """Patch base MPI checks so CLI tests can focus on tier-specific behavior."""
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    mpirun = tmp_path / "mpirun"
    mpirun.write_text("#!/bin/sh\nexit 0\n")
    mpirun.chmod(0o755)
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))

    stack = ExitStack()
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_charmm_mpi_runtime"
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
            return_value=True,
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi._charmm_lib_path",
            return_value=lib,
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
            return_value=True,
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
            return_value=mpirun.resolve(),
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
            return_value=True,
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
            return_value=(0, 2),
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
            return_value=True,
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_libmpi_path",
            return_value=tmp_path / "libmpi.so",
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi4py_libmpi_path",
            return_value=tmp_path / "libmpi.so",
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi4py_openmpi_mismatch",
            return_value=(True, ""),
        )
    )
    stack.enter_context(
        mock.patch(
            "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
            return_value=False,
        )
    )
    return stack


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


def test_run_mpi_check_reports_mpi4py_openmpi_mismatch(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
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
        return_value=(tmp_path / "mpirun").resolve(),
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
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi4py_openmpi_mismatch",
        return_value=(False, "mpi4py is linked to /usr/lib/libmpi.so but libcharmm.so uses /opt/libmpi.so. Rebuild: ./scripts/rebuild_mpi4py_for_charmm.sh"),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy.spatial_mpi_enabled",
        return_value=False,
    ):
        report = run_mpi_check()
    assert report.ok is False
    assert any("rebuild_mpi4py" in e for e in report.errors)


def test_mpi4py_openmpi_mismatch_same_path(tmp_path):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

    libmpi = tmp_path / "libmpi.so"
    libmpi.write_bytes(b"x")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_libmpi_path",
        return_value=libmpi,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.mpi4py_libmpi_path",
        return_value=libmpi,
    ):
        ok, msg = charmm_mpi.mpi4py_openmpi_mismatch()
    assert ok is True
    assert msg == ""


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


def test_mpi_check_prelaunch_strict(monkeypatch, tmp_path):
    import sys

    from mmml.cli.run.mpi_check import main

    fake_mpi = mock.MagicMock()
    fake_mpi.Is_initialized.return_value = True
    monkeypatch.setitem(sys.modules, "mpi4py", mock.MagicMock(MPI=fake_mpi))

    mpirun = tmp_path / "mpirun"
    mpirun.write_text("#!/bin/sh\nexit 0\n")
    mpirun.chmod(0o755)
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
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
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_jax_device_name",
        return_value="gpu",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.jax_device_policy.mlpot_local_gpu_count",
        return_value=1,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.defer_jax_warmup_until_after_mlpot_sd",
        return_value=False,
    ):
        assert main(["--tier2", "--strict", "--prelaunch"]) == 0


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


def test_mpi_check_tier3_informational_passes(monkeypatch, tmp_path, capsys):
    from mmml.cli.run.mpi_check import main

    with _healthy_mpi_check_context(monkeypatch, tmp_path):
        assert main(["--tier3"]) == 0

    text = capsys.readouterr().out
    assert "Production status: BLOCKED" in text
    assert "Check status: survey completed" in text


def test_mpi_check_tier3_strict_fails(monkeypatch, tmp_path):
    from mmml.cli.run.mpi_check import main

    with _healthy_mpi_check_context(monkeypatch, tmp_path):
        assert main(["--tier3", "--strict"]) == 1


def test_mpi_check_tier3_json_reports_blocker(monkeypatch, tmp_path, capsys):
    from mmml.cli.run.mpi_check import main

    with _healthy_mpi_check_context(monkeypatch, tmp_path):
        assert main(["--tier3", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    tier3 = payload["tier3"]
    survey = tier3["survey"]
    assert tier3["ok"] is True
    assert tier3["blocked"] is True
    assert tier3["spike_doc"] == "tests/functionality/mlpot/SPATIAL_MPI_DOMDEC.md"
    assert survey["pycharmm_local_atom_api"] is True
    assert survey["pycharmm_ghost_atom_api"] is True


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


def test_assert_mpi_launcher_for_mlpot_sd_raises_serial(monkeypatch):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

    monkeypatch.delenv("MMML_ALLOW_SERIAL_MPI_CHARMM", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="OpenMPI launch"):
            charmm_mpi.assert_mpi_launcher_for_mlpot_sd(context="MLpot SD minimize")


def test_assert_mpi_launcher_for_mlpot_sd_allows_mpirun(monkeypatch):
    from mmml.interfaces.pycharmmInterface import charmm_mpi

    monkeypatch.delenv("MMML_ALLOW_SERIAL_MPI_CHARMM", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        charmm_mpi.assert_mpi_launcher_for_mlpot_sd(context="MLpot SD minimize")
