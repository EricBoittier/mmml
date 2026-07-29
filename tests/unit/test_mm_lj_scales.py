"""Per-type MM LJ σ/ε scales for hybrid train + MD ATC remapping."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.mm_lj_scales import (
    MM_LJ_EPSILON_SCALE_KEY,
    MM_LJ_SIGMA_SCALE_KEY,
    apply_mm_lj_scales,
    attach_mm_lj_scales,
    load_mm_lj_scales_sidecar,
    mm_lj_scales_metadata,
    resolve_md_lj_scales,
    scales_to_atc,
    split_mm_lj_scale_params,
    write_mm_lj_scales_into_hybrid_mm_json,
)


def test_apply_unit_scales_identity():
    sig = jnp.array([3.5, 2.1])
    eps = jnp.array([0.08, 0.02])
    s2, e2 = apply_mm_lj_scales(sig, eps, jnp.ones(2), jnp.ones(2))
    np.testing.assert_allclose(s2, sig)
    np.testing.assert_allclose(e2, eps)


def test_apply_scales_and_include_lj_off():
    sig = jnp.array([3.5, 2.1])
    eps = jnp.array([0.08, 0.02])
    s2, e2 = apply_mm_lj_scales(
        sig, eps, jnp.array([1.1, 0.9]), jnp.array([2.0, 0.5]), include_lj=False
    )
    np.testing.assert_allclose(s2, sig * jnp.array([1.1, 0.9]))
    np.testing.assert_allclose(e2, jnp.zeros(2))


def test_attach_and_split_params():
    base = {"params": {"w": jnp.array(1.0)}}
    attached = attach_mm_lj_scales(base, 3)
    assert attached[MM_LJ_SIGMA_SCALE_KEY].shape == (3,)
    assert attached[MM_LJ_EPSILON_SCALE_KEY].shape == (3,)
    model, sig, eps = split_mm_lj_scale_params(attached)
    assert MM_LJ_SIGMA_SCALE_KEY not in model
    assert "params" in model
    np.testing.assert_allclose(sig, jnp.ones(3))
    np.testing.assert_allclose(eps, jnp.ones(3))


def test_scales_to_atc_by_name():
    ep, sig = scales_to_atc(
        ["CG2O1", "HGR52", "DEFAULT"],
        [1.1, 0.9, 1.0],
        [2.0, 0.5, 1.0],
        ["HGR52", "CG2O1", "OTHER"],
    )
    np.testing.assert_allclose(sig, [0.9, 1.1, 1.0])
    np.testing.assert_allclose(ep, [0.5, 2.0, 1.0])


def test_hybrid_mm_json_round_trip(tmp_path: Path):
    path = tmp_path / "hybrid_mm.json"
    write_mm_lj_scales_into_hybrid_mm_json(
        path,
        type_names=["A", "B"],
        sigma_scale=[1.2, 0.8],
        epsilon_scale=[1.5, 0.5],
    )
    raw = json.loads(path.read_text())
    assert raw["learn_mm_lj_scales"] is True
    loaded = load_mm_lj_scales_sidecar(path)
    assert loaded is not None
    np.testing.assert_allclose(loaded["mm_lj_sigma_scale"], [1.2, 0.8])
    ep, sig = resolve_md_lj_scales(
        scales_file=path,
        atc_names=["B", "A"],
    )
    assert ep is not None and sig is not None
    np.testing.assert_allclose(sig, [0.8, 1.2])
    np.testing.assert_allclose(ep, [0.5, 1.5])


def test_metadata_without_scales():
    meta = mm_lj_scales_metadata(learn_mm_lj_scales=False)
    assert meta == {"learn_mm_lj_scales": False}


def test_hybrid_forward_unit_scales_match_baseline():
    from mmml.models.hybrid_energy import hybrid_forward

    # Reuse fixtures from test_hybrid_energy via a mid-range dimer with MM on.
    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)

    def fake_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
        e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
        f = jnp.zeros_like(positions)
        return {
            "energy": e.reshape(batch_size, 1),
            "forces": f,
        }

    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [6.5, 0, 0], [7.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.array([-0.3, 0.3, -0.3, 0.3])
    atom_mask = jnp.ones(n, dtype=jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src).astype(jnp.float32)
    batch = {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": chg.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep,
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }

    base = hybrid_forward(fake_apply, {}, batch, 1, SIG, EPS, **KW)
    ones = hybrid_forward(
        fake_apply,
        {},
        batch,
        1,
        SIG,
        EPS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=jnp.ones(2),
        mm_lj_epsilon_scale=jnp.ones(2),
        **KW,
    )
    np.testing.assert_allclose(base["energy"], ones["energy"], rtol=1e-6)
    np.testing.assert_allclose(base["e_mm"], ones["e_mm"], rtol=1e-6)

    scaled = hybrid_forward(
        fake_apply,
        {},
        batch,
        1,
        SIG,
        EPS,
        learn_mm_lj_scales=True,
        mm_lj_sigma_scale=jnp.array([1.2, 0.8]),
        mm_lj_epsilon_scale=jnp.array([1.5, 0.5]),
        **KW,
    )
    assert not np.allclose(base["e_mm"], scaled["e_mm"])


def test_lj_scale_gradients_nonzero():
    from mmml.models.hybrid_energy import hybrid_forward

    SIG = jnp.array([3.6527, 2.3876])
    EPS = jnp.array([0.0780, 0.0240])
    KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)

    def fake_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                   batch_segments, batch_size, batch_mask, atom_mask):
        e = jnp.sum(atom_mask) * jnp.asarray(-1.0)
        return {
            "energy": e.reshape(batch_size, 1),
            "forces": jnp.zeros_like(positions),
        }

    n = 4
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0, 0], [6.5, 0, 0], [7.5, 0, 0]], dtype=jnp.float32
    )
    mid = jnp.array([0, 0, 1, 1])
    tidx = jnp.array([0, 1, 0, 1])
    chg = jnp.array([-0.3, 0.3, -0.3, 0.3])
    atom_mask = jnp.ones(n, dtype=jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src).astype(jnp.float32)
    batch = {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": chg.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep,
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }

    def loss(scales):
        sig_s, eps_s = scales
        out = hybrid_forward(
            fake_apply,
            {},
            batch,
            1,
            SIG,
            EPS,
            learn_mm_lj_scales=True,
            mm_lj_sigma_scale=sig_s,
            mm_lj_epsilon_scale=eps_s,
            **KW,
        )
        return jnp.sum(out["energy"])

    g_sig, g_eps = jax.grad(loss)((jnp.ones(2), jnp.ones(2)))
    assert np.isfinite(g_sig).all() and np.isfinite(g_eps).all()
    assert float(jnp.sum(jnp.abs(g_sig)) + jnp.sum(jnp.abs(g_eps))) > 0.0
