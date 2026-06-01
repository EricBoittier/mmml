"""Tests for MLpot JAX device policy with MPI-linked CHARMM."""

from __future__ import annotations

from unittest import mock


def test_mlpot_defaults_to_gpu_for_mpi_charmm(monkeypatch):
    monkeypatch.delenv("MMML_MLPOT_DEVICE", raising=False)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        from mmml.interfaces.pycharmmInterface import jax_device_policy

        assert jax_device_policy.mlpot_jax_device_name() == "gpu"
        assert jax_device_policy.apply_mlpot_jax_platform_env(quiet=True) == "gpu"
        assert __import__("os").environ.get("JAX_PLATFORMS") == "gpu"


def test_mlpot_cpu_override(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    with mock.patch(
        "mmml.interfaces.pycharmmInterface.charmm_mpi.charmm_lib_links_mpi",
        return_value=True,
    ):
        from mmml.interfaces.pycharmmInterface import jax_device_policy

        assert jax_device_policy.mlpot_jax_device_name() == "cpu"


def test_mlpot_jax_compilation_cache_default(monkeypatch, tmp_path):
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("MMML_NO_JAX_COMPILATION_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from mmml.interfaces.pycharmmInterface import jax_device_policy

    cache = jax_device_policy.apply_mlpot_jax_compilation_cache_env(quiet=True)
    assert cache == tmp_path / "mmml" / "jax-compilation-cache"
    assert cache.is_dir()
    assert __import__("os").environ["JAX_COMPILATION_CACHE_DIR"] == str(cache)


def test_mlpot_jax_compilation_cache_respects_override(monkeypatch, tmp_path):
    override = tmp_path / "custom-jax-cache"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(override))
    from mmml.interfaces.pycharmmInterface import jax_device_policy

    cache = jax_device_policy.apply_mlpot_jax_compilation_cache_env(quiet=True)
    assert cache == override
