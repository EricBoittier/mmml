"""Tests for OpenMPI / DOMDEC CHARMM MPI bootstrap."""

from __future__ import annotations

import os
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
    assert cmd[1:6] == ["-np", "1", "--mca", "orte_abort_print_stack", "1"]
    assert cmd[7:9] == ["-m", "mmml.cli.__main__"]
    assert cmd[9:] == ["md-system", "--backend", "pycharmm"]
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
        "mmml.interfaces.pycharmmInterface.charmm_mpi.subprocess.run",
        return_value=mock.Mock(returncode=0),
    ) as mock_run:
        code = charmm_mpi.maybe_rerun_md_system_under_mpirun(
            ["--config", "dcm_test.yaml", "--run-all"]
        )
    assert code == 0
    cmd = mock_run.call_args.args[0]
    assert cmd[1:6] == ["-np", "1", "--mca", "orte_abort_print_stack", "1"]
    assert cmd[7:9] == ["-m", "mmml.cli.__main__"]
    assert cmd[9:] == ["md-system", "--config", "dcm_test.yaml", "--run-all"]


def test_mpi_mpirun_extra_args_abort_stack_by_default(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.delenv("MMML_MPI_VERBOSE", raising=False)
    assert charmm_mpi.mpi_mpirun_extra_args() == [
        "--mca",
        "orte_abort_print_stack",
        "1",
    ]


def test_mpi_mpirun_extra_args_verbose(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_ABORT_STACK", raising=False)
    monkeypatch.setenv("MMML_MPI_VERBOSE", "1")
    args = charmm_mpi.mpi_mpirun_extra_args()
    assert args[:3] == ["--mca", "orte_abort_print_stack", "1"]
    assert "plm_base_verbose" in args


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
