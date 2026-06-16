"""Tests for JAX compile-time thread bump (CHARMM OMP pin stays at 1 outside warmup)."""

from __future__ import annotations

import os


def test_resolve_jax_compile_thread_count_default(monkeypatch):
    from mmml.interfaces.pycharmmInterface import jax_compile_threads

    monkeypatch.delenv("MMML_NO_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.delenv("MMML_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setattr(jax_compile_threads.os, "cpu_count", lambda: 32)
    assert jax_compile_threads.resolve_jax_compile_thread_count() == 16


def test_resolve_jax_compile_thread_count_explicit(monkeypatch):
    from mmml.interfaces.pycharmmInterface import jax_compile_threads

    monkeypatch.delenv("MMML_NO_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "8")
    assert jax_compile_threads.resolve_jax_compile_thread_count() == 8


def test_jax_compile_threads_context_restores_omp(monkeypatch):
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        jax_compile_threads_context,
    )

    monkeypatch.delenv("MMML_NO_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "4")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")

    with jax_compile_threads_context(quiet=True):
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "4"

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_jax_compile_threads_disabled(monkeypatch):
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        jax_compile_threads_context,
        resolve_jax_compile_thread_count,
    )

    monkeypatch.setenv("MMML_NO_JAX_COMPILE_THREADS", "1")
    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "8")
    assert resolve_jax_compile_thread_count() == 0

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with jax_compile_threads_context(quiet=True) as n:
        assert n == 0
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_apply_jax_compile_xla_flags_appends(monkeypatch):
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        apply_jax_compile_xla_flags,
    )

    monkeypatch.delenv("MMML_NO_JAX_COMPILE_THREADS", raising=False)
    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "6")
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    n = apply_jax_compile_xla_flags(quiet=True)
    assert n == 6
    flags = os.environ["XLA_FLAGS"]
    assert "xla_cpu_multi_thread_eigen=true" in flags
    assert "intra_op_parallelism_threads=6" in flags


def test_sanitize_xla_flags_strips_thread_pool_size(monkeypatch):
    from mmml.interfaces.pycharmmInterface.jax_compile_threads import (
        apply_jax_compile_xla_flags,
        sanitize_xla_flags_env,
    )

    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_cpu_thread_pool_size=16 --xla_gpu_cuda_data_dir=/usr/local/cuda",
    )
    assert sanitize_xla_flags_env(quiet=True) is True
    assert "thread_pool_size" not in os.environ["XLA_FLAGS"]
    assert "xla_gpu_cuda_data_dir" in os.environ["XLA_FLAGS"]

    monkeypatch.setenv("MMML_JAX_COMPILE_THREADS", "4")
    monkeypatch.delenv("MMML_NO_JAX_COMPILE_THREADS", raising=False)
    apply_jax_compile_xla_flags(quiet=True)
    flags = os.environ["XLA_FLAGS"]
    assert "thread_pool_size" not in flags
    assert "intra_op_parallelism_threads=4" in flags
