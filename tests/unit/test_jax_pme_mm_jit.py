"""jax-pme MM wrapper must run under jax.jit (hybrid spherical_cutoff path)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    _box_length_from_cell_jax,
    _wrap_mm_fn_with_jax_pme_coulomb,
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


def test_box_length_from_cell_jax_preserves_cell_dtype() -> None:
    f32_cell = jnp.array([32.0, 32.0, 32.0], dtype=jnp.float32)
    assert _box_length_from_cell_jax(f32_cell).dtype == jnp.float32
    if jax.config.read("jax_enable_x64"):
        f64_cell = jnp.array([32.0, 32.0, 32.0], dtype=jnp.float64)
        assert _box_length_from_cell_jax(f64_cell).dtype == jnp.float64


def _record_box_len_callback(box_len_dtype: list[jnp.dtype]):
    def _callback(
        positions: jnp.ndarray,
        box_length_A: jnp.ndarray,
        *,
        charges_np: np.ndarray,
        method: str,
        sr_cutoff_A: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del charges_np, method, sr_cutoff_A
        box_len_dtype.append(box_length_A.dtype)
        n = positions.shape[0]
        return (
            jnp.asarray(2.0, dtype=positions.dtype),
            jnp.full((n, 3), 0.5, dtype=positions.dtype),
        )

    return _callback


@pytest.mark.parametrize("pos_dtype", [jnp.float32, jnp.float64])
def test_wrap_mm_fn_static_box_length_matches_positions_dtype(
    monkeypatch, pos_dtype: jnp.dtype
) -> None:
    if pos_dtype == jnp.float64 and not jax.config.read("jax_enable_x64"):
        pytest.skip("float64 positions require jax_enable_x64")
    recorded: list[jnp.dtype] = []
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._jax_pme_coulomb_pure_callback",
        _record_box_len_callback(recorded),
    )
    wrapped = _wrap_mm_fn_with_jax_pme_coulomb(
        _lj_only_mm_fn,
        charges_np=np.zeros(5, dtype=np.float64),
        pbc_cell=np.array([32.0, 32.0, 32.0], dtype=np.float64),
        method="ewald",
        sr_cutoff_A=6.0,
        dynamic=False,
    )
    pos = jnp.zeros((5, 3), dtype=pos_dtype)
    energy, forces = jax.jit(wrapped)(pos)
    assert float(energy) == pytest.approx(3.0)
    assert recorded == [pos_dtype]


def _fake_coulomb_callback(
    positions: jnp.ndarray,
    box_length_A: jnp.ndarray,
    *,
    charges_np: np.ndarray,
    method: str,
    sr_cutoff_A: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del charges_np, method, sr_cutoff_A, box_length_A
    n = positions.shape[0]
    return (
        jnp.asarray(2.0, dtype=positions.dtype),
        jnp.full((n, 3), 0.5, dtype=positions.dtype),
    )


def test_wrap_mm_fn_jax_pme_static_path_under_jit(monkeypatch) -> None:
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._jax_pme_coulomb_pure_callback",
        _fake_coulomb_callback,
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
        "mmml.interfaces.pycharmmInterface.mm_energy_forces._jax_pme_coulomb_pure_callback",
        _fake_coulomb_callback,
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
