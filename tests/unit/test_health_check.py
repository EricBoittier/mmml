"""Unit tests for mmml health-check."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mmml.cli.run.health_check import (
    check_checkpoint,
    check_core,
    check_jax,
    render_health_report,
    run_health_check,
)


def test_check_core_ok():
    check = check_core()
    assert check.ok is True
    assert check.name == "core"


def test_check_jax_cpu_only(monkeypatch):
    class _Dev:
        def __str__(self) -> str:
            return "CpuDevice(id=0)"

    with mock.patch("jax.devices", return_value=[_Dev()]), mock.patch(
        "jax.default_backend", return_value="cpu"
    ):
        check = check_jax(require_gpu=False)
    assert check.ok is True
    assert check.details["cuda_visible"] is False


def test_check_jax_require_gpu_fails(monkeypatch):
    class _Dev:
        def __str__(self) -> str:
            return "CpuDevice(id=0)"

    with mock.patch("jax.devices", return_value=[_Dev()]), mock.patch(
        "jax.default_backend", return_value="cpu"
    ):
        check = check_jax(require_gpu=True)
    assert check.ok is False


def test_check_checkpoint_missing():
    check = check_checkpoint(Path("/no/such/checkpoint.json"))
    assert check.ok is False
    assert check.errors


def test_run_health_check_only_core():
    report = run_health_check(only=["core"])
    assert report.ok is True
    assert [c.name for c in report.checks] == ["core"]


def test_run_health_check_unknown_only():
    with pytest.raises(ValueError, match="Unknown check"):
        run_health_check(only=["nope"])


def test_render_health_report():
    report = run_health_check(only=["core"])
    text = render_health_report(report)
    assert "interface health check" in text
    assert "[OK] core" in text


def test_main_json_exit_code(monkeypatch):
    from mmml.cli.run import health_check

    with mock.patch(
        "mmml.cli.run.health_check.run_health_check",
        return_value=run_health_check(only=["core"]),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_serial_charmm_mpi_env",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.maybe_rerun_mmml_under_mpirun",
        return_value=None,
    ):
        assert health_check.main(["--json", "--only", "core"]) == 0


def test_main_reruns_under_mpirun(monkeypatch, tmp_path):
    from mmml.cli.run import health_check
    from mmml.interfaces.pycharmmInterface import charmm_mpi

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
            ["--only", "core"],
            subcommand="health-check",
        )
    assert code == 0
    cmd = mock_run.call_args.args[0]
    assert "health-check" in cmd
