"""Tests for OpenMPI / DOMDEC CHARMM MPI bootstrap."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from mmml.interfaces.pycharmmInterface import charmm_mpi


def test_charmm_lib_links_mpi_detects_ldd(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    charmm_mpi.charmm_lib_links_mpi.cache_clear()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="libmpi.so.40 => /lib/libmpi.so.40"),
    ):
        assert charmm_mpi.charmm_lib_links_mpi() is True
    charmm_mpi.charmm_lib_links_mpi.cache_clear()


def test_scrub_stale_openmpi_env_when_charmm_mpi_linked(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        removed = charmm_mpi.scrub_stale_openmpi_env()
    assert removed >= 1
    assert "OMPI_COMM_WORLD_SIZE" not in os.environ


def test_ensure_mpi_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("MMML_NO_MPI_INIT", "1")
    assert charmm_mpi.ensure_mpi_for_charmm_domdec() is True


def test_serial_domdec_charmm_does_not_python_init_mpi(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    monkeypatch.delenv("MMML_MPI_PY_INIT", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._init_mpi_thread_multiple",
    ) as py_init:
        assert charmm_mpi.ensure_mpi_for_charmm_domdec() is True
        py_init.assert_not_called()


def test_revalidate_mpi_after_cuda_ok_when_not_needed(monkeypatch):
    monkeypatch.setenv("MMML_NO_MPI_INIT", "1")
    assert charmm_mpi.revalidate_mpi_after_cuda() is True


def test_revalidate_mpi_after_cuda_trusts_mpirun_without_mpi4py(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ), mock.patch(
        "mmml.utils.jax_gpu_warmup.sync_jax_gpu_before_charmm",
    ) as mock_sync:
        assert charmm_mpi.revalidate_mpi_after_cuda(phase="test") is True
    mock_sync.assert_called_once()


def test_charmm_mpirun_path_from_ldd(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    bindir = tmp_path / "openmpi" / "bin"
    bindir.mkdir(parents=True)
    mpirun = bindir / "mpirun"
    mpirun.write_text("#!/bin/sh\n")
    mpirun.chmod(0o755)
    libdir = tmp_path / "openmpi" / "lib"
    libdir.mkdir(parents=True)
    (libdir / "libmpi.so.40").symlink_to("/dev/null")

    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    charmm_mpi.charmm_lib_links_mpi.cache_clear()
    charmm_mpi.charmm_mpirun_path.cache_clear()
    ldd_out = f"libmpi.so.40 => {libdir / 'libmpi.so.40'}"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=ldd_out),
    ):
        found = charmm_mpi.charmm_mpirun_path()
    assert found == mpirun.resolve()
    charmm_mpi.charmm_mpirun_path.cache_clear()
    charmm_mpi.charmm_lib_links_mpi.cache_clear()


    charmm_mpi.charmm_mpirun_path.cache_clear()
    charmm_mpi.charmm_lib_links_mpi.cache_clear()


def test_charmm_mpirun_path_prefers_built_openmpi_over_distro(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    built = tmp_path / "opt" / "openmpi-5" / "build"
    built_bin = built / "bin"
    built_bin.mkdir(parents=True)
    built_lib = built / "lib"
    built_lib.mkdir(parents=True)
    mpirun = built_bin / "mpirun"
    mpirun.write_text("#!/bin/sh\n")
    mpirun.chmod(0o755)
    built_mpi = built_lib / "libmpi.so.40"
    built_mpi.symlink_to("/dev/null")

    usr_mpi = tmp_path / "usr" / "lib" / "libmpi.so.40"
    usr_mpi.parent.mkdir(parents=True)
    usr_mpi.symlink_to("/dev/null")

    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    charmm_mpi.charmm_mpirun_path.cache_clear()
    ldd_out = (
        f"libmpi.so.40 => {usr_mpi}\n"
        f"libmpi.so.40 => {built_mpi}\n"
    )
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=ldd_out),
    ):
        found = charmm_mpi.charmm_mpirun_path()
    assert found == mpirun.resolve()
    charmm_mpi.charmm_mpirun_path.cache_clear()


def test_charmm_mpirun_path_prefers_ldd_over_distro_openmpi_root(monkeypatch, tmp_path):
    """``OPENMPI_ROOT=/usr`` must not beat ``libmpi`` from ``ldd libcharmm.so``."""
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    built = tmp_path / "opt" / "openmpi-5" / "build"
    built_bin = built / "bin"
    built_bin.mkdir(parents=True)
    built_lib = built / "lib"
    built_lib.mkdir(parents=True)
    mpirun = built_bin / "mpirun"
    mpirun.write_text("#!/bin/sh\n")
    mpirun.chmod(0o755)
    built_mpi = built_lib / "libmpi.so.40"
    built_mpi.symlink_to("/dev/null")

    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    monkeypatch.setenv("OPENMPI_ROOT", "/usr")
    charmm_mpi.charmm_mpirun_path.cache_clear()
    ldd_out = f"libmpi.so.40 => {built_mpi}\n"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=ldd_out),
    ):
        found = charmm_mpi.charmm_mpirun_path()
    assert found == mpirun.resolve()
    charmm_mpi.charmm_mpirun_path.cache_clear()


def test_charmm_mpirun_path_from_openmpi_root(monkeypatch, tmp_path):
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    prefix = tmp_path / "openmpi-prefix"
    bindir = prefix / "bin"
    bindir.mkdir(parents=True)
    mpirun = bindir / "mpirun"
    mpirun.write_text("#!/bin/sh\n")
    mpirun.chmod(0o755)

    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    monkeypatch.setenv("OPENMPI_ROOT", str(prefix))
    charmm_mpi.charmm_mpirun_path.cache_clear()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=""),
    ):
        found = charmm_mpi.charmm_mpirun_path()
    assert found == mpirun.resolve()
    charmm_mpi.charmm_mpirun_path.cache_clear()


def test_charmm_mpirun_path_falls_back_to_path_mpirun_for_debian_layout(
    monkeypatch, tmp_path
):
    """Ubuntu multiarch: libmpi in lib/x86_64-linux-gnu, mpirun only on PATH."""
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"stub")
    on_path = tmp_path / "usr" / "bin" / "mpirun"
    on_path.parent.mkdir(parents=True)
    on_path.write_text("#!/bin/sh\n")
    on_path.chmod(0o755)
    libmpi = tmp_path / "usr" / "lib" / "x86_64-linux-gnu" / "libmpi.so.40"
    libmpi.parent.mkdir(parents=True)
    libmpi.symlink_to("/dev/null")

    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    monkeypatch.delenv("OPENMPI_ROOT", raising=False)
    monkeypatch.delenv("MMML_MPIRUN", raising=False)
    charmm_mpi.charmm_mpirun_path.cache_clear()
    charmm_mpi.charmm_lib_links_mpi.cache_clear()
    ldd_out = f"libmpi.so.40 => {libmpi}\n"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout=ldd_out),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.shutil.which",
        return_value=str(on_path),
    ):
        found = charmm_mpi.charmm_mpirun_path()
    assert found == on_path.resolve()
    charmm_mpi.charmm_mpirun_path.cache_clear()
    charmm_mpi.charmm_lib_links_mpi.cache_clear()


def test_recover_mpi_never_finalizes(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_comm_valid",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ):
        assert charmm_mpi.recover_mpi_for_charmm_after_jax(phase="test") is True


def test_prepare_serial_charmm_mpi_env_pins_omp_threads(monkeypatch):
    monkeypatch.delenv("MMML_NO_CHARMM_OMP_PIN", raising=False)
    monkeypatch.delenv("MMML_CHARMM_OMP_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_charmm_mpi_runtime",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ):
        charmm_mpi.prepare_serial_charmm_mpi_env()
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_prepare_serial_charmm_mpi_env_uses_explicit_thread_budget(monkeypatch):
    monkeypatch.delenv("MMML_NO_CHARMM_OMP_PIN", raising=False)
    monkeypatch.setenv("MMML_CHARMM_OMP_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    monkeypatch.delenv("MMML_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("MMML_NO_JAX_COMPILE_THREADS", "1")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_charmm_mpi_runtime",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ):
        charmm_mpi.prepare_serial_charmm_mpi_env()
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "8"
    assert os.environ["MMML_JAX_COMPILE_THREADS"] == "8"
    assert os.environ["MMML_NO_JAX_COMPILE_THREADS"] == "0"


def test_prepare_serial_charmm_mpi_env_preserves_explicit_blas_threads(monkeypatch):
    monkeypatch.delenv("MMML_NO_CHARMM_OMP_PIN", raising=False)
    monkeypatch.setenv("MMML_CHARMM_OMP_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.prepare_charmm_mpi_runtime",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.scrub_stale_openmpi_env",
        return_value=0,
    ):
        charmm_mpi.prepare_serial_charmm_mpi_env()
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_configure_mpi4py_charmm_owned_init(monkeypatch):
    pytest.importorskip("mpi4py")
    monkeypatch.delenv("MMML_MPI_PY_INIT", raising=False)
    monkeypatch.delenv("MMML_DEFER_MPI4PY_PACKAGE_IMPORT", raising=False)
    charmm_mpi._mpi4py_charmm_configured = False
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        charmm_mpi.configure_mpi4py_charmm_owned_init()
    import mpi4py

    assert mpi4py.rc.initialize is False
    assert mpi4py.rc.finalize is False


def test_configure_mpi4py_deferred_package_import(monkeypatch):
    monkeypatch.delenv("MMML_MPI_PY_INIT", raising=False)
    monkeypatch.setenv("MMML_DEFER_MPI4PY_PACKAGE_IMPORT", "1")
    charmm_mpi._mpi4py_charmm_configured = False
    import sys

    saved = sys.modules.pop("mpi4py", None)
    try:
        with mock.patch(
            "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
            return_value=True,
        ):
            charmm_mpi.configure_mpi4py_charmm_owned_init()
        assert "mpi4py" not in sys.modules
        assert os.environ.get("MPI4PY_RC_INITIALIZE") == "false"
    finally:
        if saved is not None:
            sys.modules["mpi4py"] = saved


def test_ensure_charmm_mpi_initialized_idempotent():
    charmm_mpi._charmm_mpi_bootstrapped = False
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.configure_mpi4py_charmm_owned_init",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.import_pycharmm.init_vacuum_charmm_state_mpi",
    ) as mock_init:
        charmm_mpi.ensure_charmm_mpi_initialized()
        charmm_mpi.ensure_charmm_mpi_initialized()
    assert mock_init.call_count == 1
    assert charmm_mpi._charmm_mpi_bootstrapped is True


def test_mpi_charmm_script_all_ranks_under_mpirun():
    calls: list[str] = []

    def _fake_script(script: str, **kwargs) -> None:
        calls.append(script)

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_script_barrier",
    ) as mock_barrier, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=_fake_script,
    ):
        charmm_mpi.mpi_charmm_script("read psf card name foo.psf")

    assert calls == ["read psf card name foo.psf"]
    assert mock_barrier.call_count == 0


def test_mpi_charmm_script_barriers_both():
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_script_barrier",
    ) as mock_barrier, mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
    ):
        charmm_mpi.mpi_charmm_script("read psf", barriers="both")

    assert mock_barrier.call_count == 2


def test_mpi_charmm_script_rank0_drive_skips_nonroot():
    calls: list[str] = []

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_script_barrier",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=lambda s, **kw: calls.append(s),
    ):
        charmm_mpi.mpi_charmm_script("skip", rank0_drive=True)

    assert calls == []


def test_mpi_charmm_script_rank0_drive_calls_on_root():
    calls: list[str] = []

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_script_barrier",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=lambda s, **kw: calls.append(s),
    ):
        charmm_mpi.mpi_charmm_script("skip", rank0_drive=True)

    assert calls == ["skip"]


def test_mpi_charmm_script_serial_calls_directly():
    calls: list[str] = []

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=lambda s, **kw: calls.append(s),
    ):
        charmm_mpi.mpi_charmm_script("crystal free")

    assert calls == ["crystal free"]


def test_init_vacuum_charmm_deferred_under_mpirun():
    path = (
        Path(__file__).resolve().parents[2]
        / "mmml/interfaces/pycharmmInterface/import_pycharmm.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "def init_vacuum_charmm_state_mpi() -> None:" in source
    assert "if PYCHARMM_AVAILABLE and not _under_mpirun():" in source
    assert "MPI.COMM_WORLD.Barrier()" in source


def test_maybe_rerun_md_system_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("MMML_NO_MPI_RERUN", "1")
    assert charmm_mpi.maybe_rerun_md_system_under_mpirun(["md-system", "--help"]) is None


def test_maybe_rerun_md_system_skips_under_mpirun(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_RERUN", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        assert charmm_mpi.maybe_rerun_md_system_under_mpirun(["md-system"]) is None


def test_maybe_rerun_md_system_invokes_mpirun(monkeypatch, tmp_path):
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
        code = charmm_mpi.maybe_rerun_md_system_under_mpirun(
            ["md-system", "--backend", "pycharmm"]
        )
    assert code == 0
    mock_run.assert_called_once()
    cmd = mock_run.call_args.args[0]
    assert str(mpirun.resolve()) == cmd[0]
    assert cmd[1:3] == ["-np", "1"]
    assert "orte_abort_print_stack" in cmd
    idx_m = cmd.index("-m")
    assert cmd[idx_m : idx_m + 2] == ["-m", "mmml.cli.__main__"]
    assert cmd[idx_m + 2 :] == ["md-system", "--backend", "pycharmm"]
    env = mock_run.call_args.kwargs.get("env")
    assert env is not None
    assert str(mpirun.parent) in env["PATH"].split(os.pathsep)


def test_maybe_rerun_md_system_prepends_subcommand(monkeypatch, tmp_path):
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
        code = charmm_mpi.maybe_rerun_md_system_under_mpirun(
            ["--config", "dcm_test.yaml", "--run-all"]
        )
    assert code == 0
    cmd = mock_run.call_args.args[0]
    assert cmd[1:3] == ["-np", "1"]
    assert "orte_abort_print_stack" in cmd
    idx_m = cmd.index("-m")
    assert cmd[idx_m : idx_m + 2] == ["-m", "mmml.cli.__main__"]
    assert cmd[idx_m + 2 :] == ["md-system", "--config", "dcm_test.yaml", "--run-all"]


def test_mpi_mpirun_extra_args_includes_detected_shmem(tmp_path, monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.delenv("MMML_NO_MPI_MCA_PREFIX", raising=False)
    mca = tmp_path / "lib" / "openmpi"
    mca.mkdir(parents=True)
    (mca / "mca_shmem_mmap.so").write_bytes(b"")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_install_prefix",
        return_value=tmp_path,
    ):
        args = charmm_mpi.mpi_mpirun_extra_args()
    assert args[:3] == ["--mca", "pmix", "^ext3x"]
    assert ["--mca", "mca_base_component_path", str(mca)] in [
        args[i : i + 3] for i in range(len(args) - 2)
    ]
    assert ["--mca", "shmem", "mmap"] in [args[i : i + 3] for i in range(len(args) - 2)]


def test_mpi_mpirun_extra_args_abort_stack_by_default(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.delenv("MMML_MPI_VERBOSE", raising=False)
    monkeypatch.setenv("MMML_NO_MPI_MCA_PREFIX", "1")
    for var in ("LD_LIBRARY_PATH", "OPENMPI_ROOT", "EBROOTOPENMPI", "CHARMM_LIB_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert charmm_mpi.mpi_mpirun_extra_args() == [
        "--mca",
        "orte_abort_print_stack",
        "1",
    ]


def test_mpi_mpirun_extra_args_forwards_ld_library_path(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_LD_PATH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/openmpi/lib")
    args = charmm_mpi.mpi_mpirun_extra_args()
    assert "-x" in args
    assert args[args.index("-x") + 1] == "LD_LIBRARY_PATH"


def test_preload_openmpi_mpi_libraries_global(tmp_path, monkeypatch):
    charmm_mpi._mpi_libs_preloaded = False
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "libmpi.so.40").write_bytes(b"x")
    (lib_dir / "libmpi_usempi_ignore_tkr.so.40").write_bytes(b"x")
    (lib_dir / "libmpi_usempif08.so.40").write_bytes(b"x")
    monkeypatch.delenv("MMML_NO_MPI_LD_PATH", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_mpi_library_paths_for_preload",
        return_value=(
            lib_dir / "libmpi.so.40",
            lib_dir / "libmpi_usempi_ignore_tkr.so.40",
            lib_dir / "libmpi_usempif08.so.40",
        ),
    ), mock.patch("mmml.interfaces.pycharmmInterface.charmm_mpi.ctypes.CDLL") as cdll:
        charmm_mpi._preload_openmpi_mpi_libraries_global()
    assert cdll.call_count == 3
    assert charmm_mpi._mpi_libs_preloaded is True
    charmm_mpi._mpi_libs_preloaded = False


def test_openmpi_mpi_library_candidates_glob(tmp_path, monkeypatch):
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()
    charmm_mpi.openmpi_install_prefix.cache_clear()
    charmm_mpi.charmm_mpirun_path.cache_clear()
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    prefix = tmp_path / "openmpi"
    olib = prefix / "lib"
    olib.mkdir(parents=True)
    (olib / "libmpi.so.40").write_bytes(b"x")
    (olib / "libmpi_usempi_ignore_tkr.so.40").write_bytes(b"x")
    (olib / "libmpi_usempif08.so.40").write_bytes(b"x")
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._run_ldd",
        return_value=(
            "libmpi.so.40 => not found\n"
            "libmpi_usempi_ignore_tkr.so.40 => not found\n"
        ),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
        return_value=prefix / "bin" / "mpirun",
    ):
        (prefix / "bin").mkdir(exist_ok=True)
        (prefix / "bin" / "mpirun").write_bytes(b"")
        paths = charmm_mpi.openmpi_mpi_library_paths_for_preload()
    names = {p.name for p in paths}
    assert "libmpi.so.40" in names
    assert "libmpi_usempi_ignore_tkr.so.40" in names
    assert "libmpi_usempif08.so.40" in names
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()
    charmm_mpi.openmpi_install_prefix.cache_clear()
    charmm_mpi.charmm_mpirun_path.cache_clear()


def test_mpi_mpirun_extra_args_verbose(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.delenv("MMML_NO_MPI_MCA_PREFIX", raising=False)
    monkeypatch.setenv("MMML_MPI_VERBOSE", "1")
    args = charmm_mpi.mpi_mpirun_extra_args()
    assert args[:3] == ["--mca", "pmix", "^ext3x"]
    assert "orte_abort_print_stack" in args
    assert "plm_base_verbose" in args


def test_mpi_openmpi_static_shmem_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("MMML_NO_MPI_MCA_PREFIX", raising=False)
    monkeypatch.delenv("OMPI_MCA_shmem", raising=False)
    monkeypatch.delenv("MMML_MCA_SHMEM", raising=False)
    monkeypatch.delenv("OMPI_MCA_mca_base_component_path", raising=False)
    monkeypatch.delenv("OMPI_MCA_component_path", raising=False)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libopen-pal.so").write_bytes(b"")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_install_prefix",
        return_value=tmp_path,
    ):
        charmm_mpi.mpi_openmpi_install_env_defaults()
    assert os.environ["OMPI_MCA_shmem"] == "mmap"
    assert "OMPI_MCA_mca_base_component_path" not in os.environ


def test_openmpi_mca_component_dir_ignores_dbg_msgq_only_openmpi(tmp_path):
    prefix = tmp_path / "prefix"
    ompi = prefix / "lib" / "openmpi"
    ompi.mkdir(parents=True)
    (ompi / "libompi_dbg_msgq.so").write_bytes(b"")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_install_prefix",
        return_value=prefix,
    ):
        assert charmm_mpi.openmpi_mca_component_dir() is None


def test_mpi_openmpi_install_env_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("MMML_NO_MPI_MCA_PREFIX", raising=False)
    monkeypatch.delenv("OPAL_PREFIX", raising=False)
    monkeypatch.delenv("OMPI_MCA_pmix", raising=False)
    monkeypatch.delenv("OMPI_MCA_shmem", raising=False)
    monkeypatch.delenv("OMPI_MCA_component_path", raising=False)
    monkeypatch.delenv("MMML_OPAL_PREFIX", raising=False)
    mca = tmp_path / "lib" / "openmpi"
    mca.mkdir(parents=True)
    (mca / "mca_shmem_mmap.so").write_bytes(b"")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_install_prefix",
        return_value=tmp_path,
    ):
        charmm_mpi.mpi_openmpi_install_env_defaults()
    assert "OPAL_PREFIX" not in os.environ
    assert os.environ["OMPI_MCA_pmix"] == "^ext3x"
    assert os.environ["OMPI_MCA_shmem"] == "mmap"
    assert os.environ["OMPI_MCA_component_path"] == str(mca)


def test_mpi_openmpi_install_env_defaults_opal_prefix_when_complete(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_MCA_PREFIX", raising=False)
    monkeypatch.delenv("OPAL_PREFIX", raising=False)
    prefix = Path("/opt/openmpi-5.0.5/install")
    share = prefix / "share" / "openmpi"
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.openmpi_install_prefix",
        return_value=prefix,
    ), mock.patch.object(
        Path,
        "is_dir",
        autospec=True,
        side_effect=lambda path: path == share,
    ):
        charmm_mpi.mpi_openmpi_install_env_defaults()
    assert os.environ["OPAL_PREFIX"] == str(prefix)


def test_mpi_diagnostic_env_defaults(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.delenv("OMPI_MCA_orte_abort_print_stack", raising=False)
    charmm_mpi.mpi_diagnostic_env_defaults()
    assert os.environ["OMPI_MCA_orte_abort_print_stack"] == "1"


def test_explain_mpi_crash_prints_for_sigsegv(capsys):
    charmm_mpi.explain_mpi_crash(139, argv0="mmml md-system")
    err = capsys.readouterr().err
    assert "SIGSEGV" in err
    assert "Sphinx" in err
    assert "rebuild_charmm_mlpot.sh --debug" in err


def test_defer_jax_warmup_until_after_mlpot_sd_mpi_mpirun(monkeypatch):
    monkeypatch.delenv("MMML_NO_DEFER_JAX_WARMUP", raising=False)
    monkeypatch.delenv("MMML_DEFER_JAX_WARMUP_UNTIL_AFTER_SD", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ):
        assert charmm_mpi.defer_jax_warmup_until_after_mlpot_sd() is True


def test_defer_jax_warmup_until_after_mlpot_sd_serial(monkeypatch):
    monkeypatch.delenv("MMML_NO_DEFER_JAX_WARMUP", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=False,
    ):
        assert charmm_mpi.defer_jax_warmup_until_after_mlpot_sd() is False


def test_parse_otool_mpi_library_dirs():
    otool_out = (
        "/tmp/libcharmm.dylib:\n"
        "\t@rpath/libcharmm.dylib (compatibility version 0.0.0, current version 0.0.0)\n"
        "\t/opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib (compatibility version 81.0.0, current version 81.7.0)\n"
    )
    dirs = charmm_mpi._parse_ldd_mpi_library_dirs(otool_out)
    assert len(dirs) == 1
    assert "open-mpi" in dirs[0] and dirs[0].endswith("/lib")


def test_charmm_mpi_library_dirs_fallback_when_ldd_not_found(tmp_path, monkeypatch):
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()
    charmm_mpi.openmpi_install_prefix.cache_clear()
    charmm_mpi.charmm_mpirun_path.cache_clear()
    lib = tmp_path / "libcharmm.so"
    lib.write_bytes(b"x")
    prefix = tmp_path / "openmpi"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "libmpi.so.40").write_bytes(b"x")
    (prefix / "bin").mkdir()
    (prefix / "bin" / "mpirun").write_bytes(b"#!/bin/sh\n")
    (prefix / "bin" / "mpirun").chmod(0o755)
    monkeypatch.setenv("CHARMM_LIB_DIR", str(tmp_path))
    monkeypatch.delenv("MMML_NO_CHARMM_MPI", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._run_ldd",
        return_value="libmpi.so.40 => not found\n",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpirun_path",
        return_value=prefix / "bin" / "mpirun",
    ):
        dirs = charmm_mpi.charmm_mpi_library_dirs()
    assert str((prefix / "lib").resolve()) in dirs
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()
    charmm_mpi.openmpi_install_prefix.cache_clear()
    charmm_mpi.charmm_mpirun_path.cache_clear()


def test_mpi_library_path_export_uses_dyld_on_darwin(monkeypatch):
    monkeypatch.setattr(charmm_mpi, "_IS_DARWIN", True, raising=False)
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_mpi_library_dirs",
        return_value=("/opt/homebrew/opt/open-mpi/lib",),
    ):
        export = charmm_mpi.mpi_library_path_export()
    assert export.startswith("export DYLD_LIBRARY_PATH=")
    charmm_mpi.charmm_mpi_library_dirs.cache_clear()


def test_mmml_charmm_mpirun_dispatches_native_executable(tmp_path):
    """Native CHARMM must not be routed through the mmml CLI (MMML_BIN is always set)."""
    import subprocess
    import textwrap

    repo = Path(__file__).resolve().parents[2]
    mpirun_sh = repo / "scripts" / "mmml-charmm-mpirun.sh"
    if not mpirun_sh.is_file():
        pytest.skip("scripts/mmml-charmm-mpirun.sh missing")

    fake_charmm = tmp_path / "charmm"
    fake_charmm.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            echo "native-charmm-ok $*"
            """
        )
    )
    fake_charmm.chmod(0o755)

    fake_mpirun = tmp_path / "fake-mpirun"
    fake_mpirun.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -np|--mca|-x) shift; shift; continue ;;
              esac
              break
            done
            exec "$@"
            """
        )
    )
    fake_mpirun.chmod(0o755)

    env = os.environ.copy()
    env["MMML_MPIRUN"] = str(fake_mpirun)
    env["MMML_MPI_NP"] = "1"
    env["MMML_MPI_ORPHAN_CLEANUP_QUIET"] = "1"
    env.pop("MMML_BIN", None)

    proc = subprocess.run(
        ["bash", str(mpirun_sh), str(fake_charmm), "-i", "run.inp", "-o", "run.out"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert "Unknown command" not in combined
    assert f" {fake_charmm} -i run.inp -o run.out" in combined
    assert f"mmml {fake_charmm}" not in combined


def test_bootstrap_charmm_step_invokes_script_without_barrier():
    calls: list[str] = []

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(2, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=lambda s, **kw: calls.append(s),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_natom_count",
        return_value=100,
    ):
        n = charmm_mpi.bootstrap_charmm_step("read_psf", "read psf card name foo.psf\n")

    assert n == 100
    assert calls == ["read psf card name foo.psf\n"]


def test_bootstrap_topology_mpi_psf_crd_steps_all_ranks(tmp_path):
    psf = tmp_path / "x.psf"
    crd = tmp_path / "x.crd"
    prm = tmp_path / "x.prm"
    psf.write_text(
        "2 !NATOM\n"
        "1 DCM 1 DCM 1 C1 0 1\n"
        "2 DCM 1 DCM 1 H1 0 1\n",
        encoding="utf-8",
    )
    crd.write_text("* coords\n*\n2\n", encoding="utf-8")
    prm.write_text("MASS -1 C1 12.0 C\nMASS -1 H1 1.0 H\n", encoding="utf-8")

    calls: list[str] = []
    natom_seq = iter([0, 0, 0, 2])

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(3, 4),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._bootstrap_rank_local_staging_enabled",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.sync_bootstrap_ranks",
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._invoke_charmm_script",
        side_effect=lambda s, **kw: calls.append(s.strip()),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_natom_count",
        side_effect=lambda: next(natom_seq, 2),
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_comm_valid",
        return_value=False,
    ):
        n = charmm_mpi.bootstrap_topology_mpi(
            psf,
            crd,
            prm_path=prm,
            mode="psf-crd",
        )

    assert n == 2
    assert len(calls) == 5
    assert calls[0].startswith("read rtf card name")
    assert "read psf card name" in calls[2]
    assert "read coor card name" in calls[3]


def test_bootstrap_topology_mpi_invalid_mode(tmp_path):
    psf = tmp_path / "x.psf"
    crd = tmp_path / "x.crd"
    psf.write_text("1 !NATOM\n1 DCM 1 DCM 1 C1 0 1\n", encoding="utf-8")
    crd.write_text("*\n*\n1\n", encoding="utf-8")

    with mock.patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        return_value=(0, 1),
    ), pytest.raises(ValueError, match="unsupported bootstrap mode"):
        charmm_mpi.bootstrap_topology_mpi(psf, crd, mode="bad-mode")


def test_stage_topology_files_for_rank_copies_to_uuid_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    psf = tmp_path / "a.psf"
    crd = tmp_path / "a.crd"
    prm = tmp_path / "a.prm"
    psf.write_text("2 !NATOM\n1 DCM 1 DCM 1 C1 0 1\n2 DCM 1 DCM 1 H1 0 1\n", encoding="utf-8")
    crd.write_text("* coords\n*\n2\n", encoding="utf-8")
    prm.write_text("MASS -1 C1 12.0 C\nMASS -1 H1 1.0 H\n", encoding="utf-8")

    staged = charmm_mpi.prepare_rank_local_bootstrap_paths(
        psf=psf,
        crd=crd,
        prm=prm,
        rank=2,
        size=4,
    )

    assert staged["rtf"].is_file()
    assert staged["psf"].parent == staged["crd"].parent == staged["rtf"].parent
    assert staged["staging_dir"].name.startswith("rank2_")
    assert staged["rtf"].read_text(encoding="utf-8").startswith("* MMML MPI bootstrap")
