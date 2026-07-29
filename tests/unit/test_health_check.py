"""Unit tests for mmml health-check."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from mmml.cli.run.health_check import (
    check_checkpoint,
    check_core,
    check_gpu_quantum,
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


def _fake_cupy(version: str):
    import types

    mod = types.ModuleType("cupy")
    mod.__version__ = version
    return mod


def test_gpu_quantum_noop_without_cupy(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _reject_import("cupy"),
    )
    check = check_gpu_quantum()
    assert check.ok is True
    assert check.warnings == []
    assert check.details["cupy"] is None


def _reject_import(blocked: str):
    import builtins

    real = builtins.__import__

    def _imp(name, *args, **kwargs):
        if name == blocked:
            raise ImportError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    return _imp


def test_gpu_quantum_warns_on_cupy14_with_gpu4pyscf(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy("14.1.1"))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "gpu4pyscf" else None,
    )
    check = check_gpu_quantum()
    # a warning, not a hard failure (non-strict health-check stays ok)
    assert check.ok is True
    assert check.warnings, "expected a cupy>=14 + gpu4pyscf warning"
    text = " ".join(check.warnings)
    assert "cuTENSOR" in text and "836" in text
    assert check.details["cupy"] == "14.1.1"
    assert check.details["gpu4pyscf_installed"] is True


def test_gpu_quantum_ok_on_cupy13(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy("13.6.0"))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "gpu4pyscf" else None,
    )
    check = check_gpu_quantum()
    assert check.ok is True
    assert check.warnings == []
    assert check.details["cupy"] == "13.6.0"
