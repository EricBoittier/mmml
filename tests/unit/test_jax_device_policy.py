"""Tests for MLpot JAX device policy with MPI-linked CHARMM."""

from __future__ import annotations

from unittest import mock


def test_mlpot_defaults_to_cpu_for_mpi_charmm(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_DEVICE", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        from mmml.interfaces.pycharmmInterface import jax_device_policy

        assert jax_device_policy.mlpot_jax_device_name() == "cpu"
        assert jax_device_policy.apply_mlpot_jax_platform_env(quiet=True) == "cpu"
        assert __import__("os").environ.get("JAX_PLATFORMS") == "cpu"


def test_mlpot_gpu_override(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "gpu")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        from mmml.interfaces.pycharmmInterface import jax_device_policy

        assert jax_device_policy.mlpot_jax_device_name() == "gpu"
