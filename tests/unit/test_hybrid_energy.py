"""Hybrid ML/MM assembly: E = (1-s)(E_A+E_B) + s*E_AB + E_MM, with consistent forces.

The taper applies to the dimer *interaction*, never to the total: the monomers'
intramolecular energy is always present.  Getting this wrong made a
well-separated dimer's energy collapse toward 0 against a ~-43 eV reference
(train energy MAE pinned at ~43); ``test_far_dimer_keeps_the_monomer_energy``
is the regression test for exactly that.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIG = jnp.array([3.6527, 2.3876])
EPS = jnp.array([0.0780, 0.0240])
KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)

MID = jnp.array([0, 0, 1, 1, -1])
TIDX = jnp.array([0, 1, 0, 1, -1])
CHG = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])

E_ATOM = -10.0
PAIR_K = -2.0


def _fake_model_apply(params, *, atomic_numbers, positions, dst_idx, src_idx,
                      batch_segments, batch_size, batch_mask, atom_mask):
    """Analytic stand-in: per-atom term + distance-dependent pair term.

    Respects atom_mask/batch_mask exactly as the real model does for padding, so
    restricting the masks to one monomer yields that monomer's energy alone.
    """

    def energy_fn(pos):
        e_atom = E_ATOM * jnp.cos(pos[:, 0]) * atom_mask
        e_per = jax.ops.segment_sum(e_atom, batch_segments, num_segments=batch_size)
        d = pos[dst_idx] - pos[src_idx]
        r = jnp.sqrt(jnp.maximum(jnp.sum(d * d, axis=-1), 1e-12))
        e_pair = PAIR_K * jnp.exp(-r) * batch_mask
        e_per = e_per + jax.ops.segment_sum(
            e_pair, batch_segments[dst_idx], num_segments=batch_size
        )
        return jnp.sum(e_per), e_per

    (_, e_per), grad = jax.value_and_grad(energy_fn, has_aux=True)(positions)
    return {"energy": e_per.reshape(batch_size, 1), "forces": -grad}


def _batch(sep, *, monomer=False):
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0.2, 0], [sep, 0, 0], [sep + 1.0, 0.2, 0], [0, 0, 0]]
    )
    mid = jnp.array([0, 0, -1, -1, -1]) if monomer else MID
    tidx = jnp.array([0, 1, -1, -1, -1]) if monomer else TIDX
    n = 5
    atom_mask = (mid >= 0).astype(jnp.float32)
    idx = jnp.arange(n)
    dst, src = jnp.meshgrid(idx, idx, indexing="ij")
    dst, src = dst.reshape(-1), src.reshape(-1)
    keep = (dst != src) & (atom_mask[dst] > 0) & (atom_mask[src] > 0)
    return {
        "R": pos,
        "Z": jnp.array([6, 1, 6, 1, 0]),
        "mol_id": mid.reshape(1, n),
        "cgenff_type_idx": tidx.reshape(1, n),
        "cgenff_charge": CHG.reshape(1, n),
        "atom_mask": atom_mask,
        "batch_mask": keep.astype(jnp.float32),
        "dst_idx": dst,
        "src_idx": src,
        "batch_segments": jnp.zeros(n, dtype=jnp.int32),
    }


def _f(x):
    """First scalar of a batch-of-1 array."""
    return float(np.asarray(x).reshape(-1)[0])


def _run(batch):
    from mmml.models.hybrid_energy import hybrid_forward

    return hybrid_forward(_fake_model_apply, {}, batch, 1, SIG, EPS, **KW)


def _plain(batch):
    return _fake_model_apply(
        {},
        atomic_numbers=batch["Z"],
        positions=batch["R"],
        dst_idx=batch["dst_idx"],
        src_idx=batch["src_idx"],
        batch_segments=batch["batch_segments"],
        batch_size=1,
        batch_mask=batch["batch_mask"],
        atom_mask=batch["atom_mask"],
    )


def _e_monomers(batch):
    """E_ML(A) + E_ML(B), each evaluated with the other monomer masked out.

    Includes the intra-monomer pair terms, so this is the true 'monomers alone'
    energy -- not just the per-atom contributions.
    """
    from mmml.models.hybrid_energy import _monomer_restricted_masks

    total = 0.0
    for which in (0, 1):
        am, bm = _monomer_restricted_masks(batch, which)
        out = _fake_model_apply(
            {},
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=1,
            batch_mask=bm,
            atom_mask=am,
        )
        total += _f(out["energy"])
    return total


def test_far_dimer_keeps_the_monomer_energy():
    """REGRESSION: at r past the tail, E_total -> E_A + E_B, NOT ~0.

    Scaling the *total* ML energy collapsed this toward 0 while the reference
    stayed at the monomer scale -- an unfittable target (train MAE ~= 43).
    """
    b = _batch(15.0)
    out = _run(b)
    assert _f(out["ml_scale"]) == pytest.approx(0.0)
    assert _f(out["e_mm"]) == pytest.approx(0.0, abs=1e-12)
    assert _f(out["energy"]) == pytest.approx(_e_monomers(b), rel=1e-5)
    assert abs(_f(out["energy"])) > 1.0        # emphatically not ~0


def test_close_dimer_is_the_full_ml_dimer_energy():
    """s = 1 -> E_total = E_AB (interaction entirely ML)."""
    b = _batch(4.0)
    out = _run(b)
    assert _f(out["ml_scale"]) == pytest.approx(1.0)
    assert _f(out["e_mm"]) == pytest.approx(0.0, abs=1e-12)
    assert _f(out["energy"]) == pytest.approx(_f(_plain(b)["energy"]), rel=1e-5)


def test_monomer_is_pure_ml():
    """No mol_id==1 -> E_B=0, E_A=E_AB, dE_ML=0 -> E_total = E_AB."""
    b = _batch(15.0, monomer=True)
    out = _run(b)
    assert _f(out["ml_scale"]) == pytest.approx(1.0)
    assert _f(out["e_mm"]) == 0.0
    assert _f(out["energy"]) == pytest.approx(_f(_plain(b)["energy"]), rel=1e-5)


def test_handoff_interpolates_the_interaction_only():
    b = _batch(7.2)
    out = _run(b)
    s = _f(out["ml_scale"])
    assert 0.0 < s < 1.0
    expect = (1 - s) * _e_monomers(b) + s * _f(_plain(b)["energy"]) + _f(out["e_mm"])
    assert _f(out["energy"]) == pytest.approx(expect, rel=1e-5)


def test_forces_match_autodiff_of_the_assembled_energy():
    """The gate: F_total == -dE_total/dR, including the ds/dR product-rule term."""
    from mmml.models.hybrid_energy import hybrid_forward

    for sep in (4.0, 6.8, 7.2, 7.9, 9.0, 14.0):
        b = _batch(sep)

        def e_total(pos, _b=b):
            bb = dict(_b)
            bb["R"] = pos
            return jnp.sum(
                hybrid_forward(_fake_model_apply, {}, bb, 1, SIG, EPS, **KW)["energy"]
            )

        out = _run(b)
        ref_f = -jax.grad(e_total)(b["R"])
        ref_f = jnp.where((MID >= 0)[:, None], ref_f, 0.0)

        assert bool(jnp.all(jnp.isfinite(out["forces"]))), f"NaN at sep={sep}"
        assert np.allclose(
            np.asarray(out["forces"]), np.asarray(ref_f), rtol=1e-4, atol=1e-6
        ), sep


def test_padding_carries_no_force():
    out = _run(_batch(7.0))
    assert np.allclose(np.asarray(out["forces"])[4], 0.0)


def test_monomer_restricted_masks_isolate_each_monomer():
    from mmml.models.hybrid_energy import _monomer_restricted_masks

    b = _batch(6.0)
    am, bm = _monomer_restricted_masks(b, 0)
    assert list(np.asarray(am).astype(int)) == [1, 1, 0, 0, 0]
    dst, src = np.asarray(b["dst_idx"]), np.asarray(b["src_idx"])
    kept = np.asarray(bm) > 0
    assert set(dst[kept].tolist()) <= {0, 1} and set(src[kept].tolist()) <= {0, 1}
