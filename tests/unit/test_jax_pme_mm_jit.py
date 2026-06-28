"""jax-pme MM wrapper must run under jax.jit (hybrid spherical_cutoff path)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.long_range_backend import LongRangeInteractionResult
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    _box_length_from_cell_jax,
    _wrap_mm_fn_with_jax_pme_coulomb,
)


def _fake_pme(pos, coefficients, *, box_length_A, method, sr_cutoff_A, exponent, prefactor):
    del coefficients, box_length_A, method, sr_cutoff_A, exponent, prefactor
    n = pos.shape[0]
    return LongRangeInteractionResult(
        energy_kcalmol=2.0,
        forces_kcalmol_A=np.full((n, 3), 0.5, dtype=np.float64),
    )


def _lj_only_mm_fn(positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.asarray(1.0, dtype=positions.dtype), jnp.zeros_like(positions)


def _lj_only_mm_fn_dynamic(
    positions: jnp.ndarray,
    pair_idx: jnp.ndarray,
    pair_mask: jnp.ndarray,
    box_override=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del pair_idx, pair_mask, box_override
    return jnp.asarray(1.0, dtype=positions.dtype), jnp.zeros_like(positions)


def test_box_length_from_cell_jax_handles_vector_and_matrix() -> None:
    assert float(_box_length_from_cell_jax(jnp.array([32.0, 32.0, 32.0]))) == 32.0
    assert float(_box_length_from_cell_jax(jnp.eye(3) * 40.0)) == 40.0


def test_wrap_mm_fn_jax_pme_static_path_under_jit(monkeypatch) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_jax_pme_power_law",
        _fake_pme,
    )
    wrapped = _wrap_mm_fn_with_jax_pme_coulomb(
        _lj_only_mm_fn,
        charges_np=np.zeros(5, dtype=np.float64),
        pbc_cell=np.array([32.0, 32.0, 32.0], dtype=np.float64),
        method="ewald",
        sr_cutoff_A=6.0,
        dynamic=False,
    )
    pos = jnp.zeros((5, 3), dtype=jnp.float32)
    energy, forces = jax.jit(wrapped)(pos)
    assert float(energy) == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(forces), 0.5, rtol=0, atol=1e-6)


def test_wrap_mm_fn_jax_pme_dynamic_path_under_jit(monkeypatch) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.long_range_backend.compute_jax_pme_power_law",
        _fake_pme,
    )
    wrapped = _wrap_mm_fn_with_jax_pme_coulomb(
        _lj_only_mm_fn_dynamic,
        charges_np=np.zeros(4, dtype=np.float64),
        pbc_cell=np.array([28.0, 28.0, 28.0], dtype=np.float64),
        method="pme",
        sr_cutoff_A=6.0,
        dynamic=True,
    )
    pos = jnp.zeros((4, 3), dtype=jnp.float32)
    pair_idx = jnp.zeros((2, 2), dtype=jnp.int32)
    pair_mask = jnp.ones((2,), dtype=jnp.bool_)
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)

    @jax.jit
    def eval_mm(p, b):
        return wrapped(p, pair_idx, pair_mask, box_override=b)

    energy, forces = eval_mm(pos, box)
    assert float(energy) == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(forces), 0.5, rtol=0, atol=1e-6)
