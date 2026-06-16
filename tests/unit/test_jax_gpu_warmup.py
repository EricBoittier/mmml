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


def test_jax_compile_timers_log_passes(capsys, monkeypatch):
    monkeypatch.setenv("MMML_JAX_COMPILE_TIMERS", "1")
    jax_gpu_warmup.reset_jax_compile_timers()
    calls = {"n": 0}

    def _run_once() -> int:
        calls["n"] += 1
        return calls["n"]

    jax_gpu_warmup.run_jax_warmup_passes("test_kernel", 2, _run_once, block=lambda _: None)
    out = capsys.readouterr().out
    assert "JAX compile timer [test_kernel] pass 1 (compile+run):" in out
    assert "JAX compile timer [test_kernel] pass 2 (run):" in out
    assert "compile≈" in out
    jax_gpu_warmup.maybe_log_jax_compile_timers()
    summary = capsys.readouterr().out
    assert "JAX compile timers — estimated compile=" in summary
    assert "test_kernel:" in summary


def test_jax_compile_timers_disabled_by_default(capsys, monkeypatch):
    monkeypatch.delenv("MMML_JAX_COMPILE_TIMERS", raising=False)
    monkeypatch.delenv("MMML_MLPOT_PROFILE", raising=False)
    jax_gpu_warmup.reset_jax_compile_timers()
    jax_gpu_warmup.run_jax_warmup_passes("silent", 2, lambda: 1, block=lambda _: None)
    assert "JAX compile timer" not in capsys.readouterr().out


def test_jax_compile_timers_follow_mlpot_profile(capsys, monkeypatch):
    monkeypatch.delenv("MMML_JAX_COMPILE_TIMERS", raising=False)
    monkeypatch.setenv("MMML_MLPOT_PROFILE", "1")
    jax_gpu_warmup.reset_jax_compile_timers()
    jax_gpu_warmup.run_jax_warmup_passes("profiled", 1, lambda: 0, block=lambda _: None)
    assert "JAX compile timer [profiled]" in capsys.readouterr().out


def test_warmup_hybrid_spherical_cutoff_skips_duplicate_key(monkeypatch):
    jax_gpu_warmup.reset_hybrid_spherical_warmup_cache()
    monkeypatch.setattr(jax_gpu_warmup, "_jax_warmup_backend", lambda: "cpu")
    calls = {"n": 0}

    def _calc(**_kwargs):
        calls["n"] += 1
        return type("R", (), {"energy": 0.0, "forces": None})()

    import numpy as np

    pos = np.zeros((4, 3))
    z = np.zeros(4, dtype=int)
    kw = dict(
        atomic_numbers=z,
        positions=pos,
        n_monomers=2,
        cutoff_params=object(),
        doML=True,
        doMM=True,
        doML_dimer=True,
    )
    jax_gpu_warmup.warmup_hybrid_spherical_cutoff(_calc, **kw)
    jax_gpu_warmup.warmup_hybrid_spherical_cutoff(_calc, **kw)
    assert calls["n"] == 2
