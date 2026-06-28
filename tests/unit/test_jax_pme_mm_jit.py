"""jax-pme MM wrapper must run under jax.jit (hybrid spherical_cutoff path)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
    HybridJaxPmeCorrectionResult,
    HybridJaxPmeMmResult,
)
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    _box_length_from_cell_jax,
    _wrap_mm_fn_with_jax_pme_coulomb,
)

_OFFSETS = np.array([0, 3, 5], dtype=np.int64)
_MONOMER_ID = np.array([0, 0, 0, 1, 1], dtype=np.int32)
_LAMBDA = np.array([1.0, 1.0], dtype=np.float64)


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


def _wrap_kwargs(*, dynamic: bool) -> dict:
    return dict(
        charges_np=np.zeros(5, dtype=np.float64),
        pbc_cell=np.array([32.0, 32.0, 32.0], dtype=np.float64),
        method="ewald",
        sr_cutoff_A=6.0,
        dynamic=dynamic,
        monomer_offsets=_OFFSETS,
        monomer_id_np=_MONOMER_ID,
        lambda_monomer=_LAMBDA,
        ml_switch_width=1.0,
        mm_switch_on=12.0,
        mm_switch_width=1.0,
        complementary_handoff=True,
        mm_r_min=None,
    )


def _fake_hybrid_lr(pos_np: np.ndarray) -> HybridJaxPmeMmResult:
    n = pos_np.shape[0]
    coulomb = HybridJaxPmeCorrectionResult(
        energy_kcalmol=2.0,
        forces_kcalmol_A=np.full((n, 3), 0.5, dtype=np.float64),
        energy_intra_kcalmol=0.0,
        energy_mic_cross_kcalmol=0.0,
        switch_scale=1.0,
    )
    return HybridJaxPmeMmResult(
        energy_kcalmol=2.0,
        forces_kcalmol_A=coulomb.forces_kcalmol_A,
        coulomb=coulomb,
        dispersion=None,
    )


def test_box_length_from_cell_jax_handles_vector_and_matrix() -> None:
    assert float(_box_length_from_cell_jax(jnp.array([32.0,  32.0, 32.0]))) == 32.0
    assert float(_box_length_from_cell_jax(jnp.eye(3) * 40.0)) == 40.0


def test_box_length_from_cell_jax_preserves_cell_dtype() -> None:
    f32_cell = jnp.array([32.0, 32.0, 32.0], dtype=jnp.float32)
    assert _box_length_from_cell_jax(f32_cell).dtype == jnp.float32
    if jax.config.read("jax_enable_x64"):
        f64_cell = jnp.array([32.0, 32.0, 32.0], dtype=jnp.float64)
        assert _box_length_from_cell_jax(f64_cell).dtype == jnp.float64


def test_wrap_mm_fn_jax_pme_static_path_under_jit(monkeypatch) -> None:
    def _fake(*args, **kwargs):
        del kwargs
        return _fake_hybrid_lr(np.asarray(args[0], dtype=np.float64))

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.hybrid_jax_pme_mm_lr_correction",
        _fake,
    )
    wrapped = _wrap_mm_fn_with_jax_pme_coulomb(_lj_only_mm_fn, **_wrap_kwargs(dynamic=False))
    pos = jnp.zeros((5, 3), dtype=jnp.float32)
    energy, forces = jax.jit(wrapped)(pos)
    assert float(energy) == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(forces), 0.5, rtol=0, atol=1e-6)


def test_wrap_mm_fn_jax_pme_dynamic_path_under_jit(monkeypatch) -> None:
    def _fake(*args, **kwargs):
        del kwargs
        return _fake_hybrid_lr(np.asarray(args[0], dtype=np.float64))

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.hybrid_jax_pme_mm_lr_correction",
        _fake,
    )
    wrapped = _wrap_mm_fn_with_jax_pme_coulomb(
        _lj_only_mm_fn_dynamic, **_wrap_kwargs(dynamic=True)
    )
    pos = jnp.zeros((5, 3), dtype=jnp.float32)
    pair_idx = jnp.zeros((2, 2), dtype=jnp.int32)
    pair_mask = jnp.ones((2,), dtype=jnp.bool_)
    box = jnp.array([30.0, 30.0, 30.0], dtype=jnp.float32)

    @jax.jit
    def eval_mm(p, b):
        return wrapped(p, pair_idx, pair_mask, box_override=b)

    energy, forces = eval_mm(pos, box)
    assert float(energy) == pytest.approx(3.0)
    np.testing.assert_allclose(np.asarray(forces), 0.5, rtol=0, atol=1e-6)
