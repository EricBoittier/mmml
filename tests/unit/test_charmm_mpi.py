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
    ):
        assert charmm_mpi.revalidate_mpi_after_cuda(phase="test") is True


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
