"""Q⁰ train path: hybrid_forward uses isolated A/B charges, not AB."""

from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.hybrid_energy import hybrid_forward

jax.config.update("jax_enable_x64", True)

SIG = jnp.array([3.5, 2.5], dtype=jnp.float64)
EPS = jnp.array([0.1, 0.05], dtype=jnp.float64)
KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)


def _dimer_batch():
    # A: atoms 0–1, B: atoms 2–3 (far COM so s_ML ~ 0).
    pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    z = jnp.array([6, 1, 6, 1], dtype=jnp.int32)
    mid = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    n = 4
    idx = np.arange(n)
    dst, src = np.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = dst != src
    return {
        "R": pos,
        "Z": z,
        "mol_id": mid.reshape(1, n),
        "cgenff_charge": jnp.array([[0.1, -0.1, 0.2, -0.2]], dtype=jnp.float64),
        "cgenff_type_idx": jnp.array([[0, 1, 0, 1]], dtype=jnp.int32),
        "atom_mask": jnp.ones(n, dtype=jnp.float64),
        "batch_mask": jnp.asarray(keep, dtype=jnp.float64),
        "dst_idx": jnp.asarray(dst, dtype=jnp.int32),
        "src_idx": jnp.asarray(src, dtype=jnp.int32),
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def test_hybrid_forward_q0_uses_monomer_not_ab_charges():
    """Q⁰ must assemble from out_a/out_b; distinct from AB (Q¹) charges."""
    batch = _dimer_batch()
    calls = {"n": 0}

    def model_apply(params, **kwargs):
        calls["n"] += 1
        am = jnp.asarray(kwargs["atom_mask"])
        # Distinct charge patterns per forward (AB / A / B).
        if float(jnp.sum(am)) > 3.5:
            q = jnp.array([9.0, 9.0, 9.0, 9.0])  # AB — must NOT be used for q0
        elif float(am[0]) > 0.5:
            q = jnp.array([0.5, -0.5, 0.0, 0.0])  # A
        else:
            q = jnp.array([0.0, 0.0, 0.3, -0.3])  # B
        e = -1.0 * jnp.sum(kwargs["positions"] * kwargs["positions"])
        # Fake forces via grad of e w.r.t. positions (unused for this assert).
        f = -2.0 * kwargs["positions"]
        return {
            "energy": jnp.array([e]),
            "forces": f,
            "charges": q,
        }

    captured = {}

    def _capture_apply(mode, q_c, q_ml, mol_id, **kwargs):
        captured["q_ml"] = np.asarray(q_ml)
        from mmml.models.mm_charge_mode import apply_mm_charge_mode as real

        return real(mode, q_c, q_ml, mol_id, **kwargs)

    with mock.patch(
        "mmml.models.hybrid_energy.apply_mm_charge_mode", side_effect=_capture_apply
    ), mock.patch(
        "mmml.models.hybrid_energy.cgenff_mm_energy",
        return_value=jnp.array(0.0),
    ), mock.patch(
        "mmml.models.hybrid_energy.inter_monomer_wall_energy",
        return_value=jnp.array(0.0),
    ):
        hybrid_forward(
            model_apply,
            {},
            batch,
            1,
            SIG,
            EPS,
            **KW,
            complementary_handoff=True,
            mm_charge_mode="q0",
            short_range_wall=False,
            include_lj=False,
        )

    assert calls["n"] == 3  # AB, A, B
    assert captured["q_ml"] is not None
    assert np.allclose(captured["q_ml"], [[0.5, -0.5, 0.3, -0.3]], atol=1e-12)
    assert not np.allclose(captured["q_ml"], [[9.0, 9.0, 9.0, 9.0]])
