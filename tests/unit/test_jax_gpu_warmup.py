"""Tests for JAX GPU warmup and CUDA toolchain PATH setup."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest import mock

from mmml.utils import jax_gpu_warmup


def test_find_bundled_ptxas_dir_from_site_packages(tmp_path, monkeypatch):
    ptxas_dir = tmp_path / "nvidia" / "cu13" / "bin"
    ptxas_dir.mkdir(parents=True)
    (ptxas_dir / "ptxas").write_bytes(b"stub")
    monkeypatch.setattr(
        jax_gpu_warmup,
        "_site_package_roots",
        lambda: [tmp_path],
    )
    jax_gpu_warmup.find_bundled_ptxas_dir.cache_clear()
    assert jax_gpu_warmup.find_bundled_ptxas_dir() == ptxas_dir.resolve()
    jax_gpu_warmup.find_bundled_ptxas_dir.cache_clear()


def test_ensure_jax_cuda_toolchain_prepends_path(tmp_path, monkeypatch):
    ptxas_dir = tmp_path / "nvidia" / "cu13" / "bin"
    ptxas_dir.mkdir(parents=True)
    ptxas_bin = ptxas_dir / "ptxas"
    ptxas_bin.write_bytes(b"#!/bin/sh\necho stub\n")
    ptxas_bin.chmod(0o755)

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(
        jax_gpu_warmup,
        "find_bundled_ptxas_dir",
        lambda: ptxas_dir.resolve(),
    )
    assert jax_gpu_warmup.ensure_jax_cuda_toolchain() is True
    assert str(ptxas_dir.resolve()) in os.environ["PATH"].split(os.pathsep)
    assert shutil.which("ptxas") == str(ptxas_bin.resolve())


def test_find_bundled_nvidia_lib_dirs_from_site_packages(tmp_path, monkeypatch):
    cudnn_dir = tmp_path / "nvidia" / "cudnn" / "lib"
    cu13_dir = tmp_path / "nvidia" / "cu13" / "lib"
    cudnn_dir.mkdir(parents=True)
    cu13_dir.mkdir(parents=True)
    (cudnn_dir / "libcudnn.so.9").write_bytes(b"stub")
    (cu13_dir / "libcublasLt.so.13").write_bytes(b"stub")
    monkeypatch.setattr(
        jax_gpu_warmup,
        "_site_package_roots",
        lambda: [tmp_path],
    )
    jax_gpu_warmup.find_bundled_nvidia_lib_dirs.cache_clear()
    assert jax_gpu_warmup.find_bundled_nvidia_lib_dirs() == [
        cudnn_dir.resolve(),
        cu13_dir.resolve(),
    ]
    jax_gpu_warmup.find_bundled_nvidia_lib_dirs.cache_clear()


def test_ensure_jax_cuda_runtime_libs_prefers_bundled_cudnn(tmp_path, monkeypatch):
    cudnn_dir = tmp_path / "nvidia" / "cudnn" / "lib"
    cudnn_dir.mkdir(parents=True)
    (cudnn_dir / "libcudnn.so.9").write_bytes(b"stub")
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    monkeypatch.setattr(
        jax_gpu_warmup,
        "find_bundled_nvidia_lib_dirs",
        lambda: [cudnn_dir.resolve()],
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{system_dir}{os.pathsep}/usr/lib")
    bundled = jax_gpu_warmup.ensure_jax_cuda_runtime_libs(quiet=True)
    assert bundled == [str(cudnn_dir.resolve())]
    parts = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    assert parts[0] == str(cudnn_dir.resolve())
    assert str(system_dir) in parts[1:]


def test_ensure_jax_cuda_toolchain_required_raises_when_missing(monkeypatch):
    monkeypatch.setattr(jax_gpu_warmup, "find_bundled_ptxas_dir", lambda: None)
    with mock.patch("mmml.utils.jax_gpu_warmup.shutil.which", return_value=None):
        try:
            jax_gpu_warmup.ensure_jax_cuda_toolchain(required=True)
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "ptxas not found" in str(exc)
        assert raised
