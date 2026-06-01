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
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ) as needs:
        assert charmm_mpi.ensure_mpi_for_charmm_domdec() is False
        needs.assert_not_called()


def test_revalidate_mpi_after_cuda_ok_when_not_needed(monkeypatch):
    monkeypatch.setenv("MMML_NO_MPI_INIT", "1")
    assert charmm_mpi.revalidate_mpi_after_cuda() is True


def test_revalidate_mpi_after_cuda_trusts_mpirun_without_mpi4py(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_comm_valid",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ):
        assert charmm_mpi.revalidate_mpi_after_cuda(phase="test") is True


def test_recover_mpi_hard_resets_after_jax_for_domdec_charmm(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    monkeypatch.delenv("MMML_NO_MPI_HARD_RESET", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._hard_reset_mpi",
        return_value=True,
    ) as hard_reset:
        assert charmm_mpi.recover_mpi_for_charmm_after_jax(phase="test") is True
        hard_reset.assert_called_once()


def test_revalidate_mpi_after_cuda_serial_requires_mpi4py(monkeypatch):
    monkeypatch.delenv("MMML_NO_MPI_INIT", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._needs_mpi_setup",
        return_value=True,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi_comm_valid",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._under_mpirun",
        return_value=False,
    ), mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi._mpi4py_available",
        return_value=False,
    ):
        assert charmm_mpi.revalidate_mpi_after_cuda(phase="test") is False
