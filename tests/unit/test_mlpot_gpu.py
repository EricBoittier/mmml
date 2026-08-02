"""Tests for MLpot multi-GPU chunk helpers."""

from __future__ import annotations

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.mlpot_batch_policy import resolve_ml_batch_size
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import (
    effective_ml_gpu_count,
    resolve_ml_gpu_count,
)


def test_chunked_model_apply_preserves_charge_auxiliary() -> None:
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mlpot_gpu import run_chunked_model_apply

    r_chunks = jnp.arange(2 * 2 * 3 * 3, dtype=jnp.float64).reshape(2, 2, 3, 3)
    z_chunks = jnp.ones((2, 2, 3), dtype=jnp.int32)
    n_chunks = jnp.full((2, 2), 3, dtype=jnp.int32)

    def apply_one(r, z, n):
        del z, n
        energy = r[:, 0, 0]
        forces = r.reshape(-1, 3)
        charges = r[..., :1].reshape(-1, 1)
        return energy, forces, charges

    energy, forces, charges = run_chunked_model_apply(
        R_chunks=r_chunks,
        Z_chunks=z_chunks,
        N_chunks=n_chunks,
        n_chunks=2,
        effective_batch_size=3,
        chunk_size=2,
        max_atoms=3,
        n_gpus=1,
        apply_one_chunk=apply_one,
        has_aux=True,
    )

    assert energy.shape == (3,)
    assert forces.shape == (9, 3)
    assert charges.shape == (9, 1)
    np.testing.assert_allclose(charges[:, 0], forces[:, 0])


def test_resolve_ml_gpu_count_explicit():
    assert resolve_ml_gpu_count(3) == 3
    assert resolve_ml_gpu_count(0) == 1


def test_resolve_ml_gpu_count_env(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_N_GPUS", "2")
    assert resolve_ml_gpu_count(None) == 2


def test_effective_ml_gpu_count_clamps(monkeypatch):
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy.mlpot_local_gpu_count",
        lambda: 4,
    )
    assert effective_ml_gpu_count(8, n_chunks=2) == 2
    assert effective_ml_gpu_count(2, n_chunks=10) == 2
    assert effective_ml_gpu_count(2, n_chunks=1) == 1


def test_resolve_ml_batch_size_cpu_default(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "cpu")
    assert resolve_ml_batch_size(90, None) == 64


def test_resolve_ml_batch_size_gpu_default(monkeypatch):
    monkeypatch.setenv("MMML_MLPOT_DEVICE", "gpu")
    assert resolve_ml_batch_size(90, None) == 256
    assert resolve_ml_batch_size(25, None) == 256
