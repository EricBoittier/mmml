"""Hybrid ML/MM assembly: E_total = s(R)*E_ML + E_MM, with consistent forces."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SIG = jnp.array([3.6527, 2.3876])
EPS = jnp.array([0.0780, 0.0240])
KW = dict(mm_switch_on=8.0, mm_switch_width=5.0, ml_switch_width=1.5)


def _dimer(sep):
    pos = jnp.array(
        [[0.0, 0, 0], [1.0, 0.2, 0], [sep, 0, 0], [sep + 1.0, 0.2, 0], [0, 0, 0]]
    )
    tidx = jnp.array([0, 1, 0, 1, -1])
    mid = jnp.array([0, 0, 1, 1, -1])
    q = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])
    return pos, tidx, mid, q


def _call(pos, tidx, mid, q, e_ml, f_ml):
    from mmml.models.hybrid_energy import hybrid_energy_forces

    return hybrid_energy_forces(e_ml, f_ml, pos, tidx, mid, q, SIG, EPS, **KW)


def test_ml_fully_on_inside_handoff_gives_pure_ml_energy():
    pos, tidx, mid, q = _dimer(4.0)  # r_com < 6.5
    e_ml = jnp.float64(-5.0)
    f_ml = jnp.zeros((5, 3))
    out = _call(pos, tidx, mid, q, e_ml, f_ml)
    assert float(out.ml_scale) == pytest.approx(1.0)
    assert float(out.e_mm) == pytest.approx(0.0, abs=1e-12)
    assert float(out.energy) == pytest.approx(-5.0)  # E_total == E_ML


def test_beyond_the_tail_gives_zero_energy():
    """ML off (>=8) and MM tail closed (>=13) -> nothing left."""
    pos, tidx, mid, q = _dimer(15.0)
    out = _call(pos, tidx, mid, q, jnp.float64(-5.0), jnp.zeros((5, 3)))
    assert float(out.ml_scale) == pytest.approx(0.0)
    assert float(out.e_mm) == pytest.approx(0.0, abs=1e-12)
    assert float(out.energy) == pytest.approx(0.0, abs=1e-12)


def test_handoff_region_mixes_both():
    pos, tidx, mid, q = _dimer(7.2)  # inside 6.5-8.0
    out = _call(pos, tidx, mid, q, jnp.float64(-5.0), jnp.zeros((5, 3)))
    assert 0.0 < float(out.ml_scale) < 1.0
    assert float(out.e_mm) != 0.0
    assert float(out.energy) == pytest.approx(float(out.ml_scale) * -5.0 + float(out.e_mm))


def test_forces_match_direct_autodiff_of_the_total_energy():
    """The gate: F_total must equal -dE_total/dR, including the switch gradient.

    Uses a position-dependent E_ML so that s(R)*E_ML(R) genuinely needs the
    product rule; the naive `scale * F_ML` shortcut fails this.
    """
    from mmml.models.hybrid_energy import hybrid_energy_forces, ml_scale_from_positions
    from mmml.models.cgenff_mm import cgenff_mm_energy

    tidx = jnp.array([0, 1, 0, 1, -1])
    mid = jnp.array([0, 0, 1, 1, -1])
    q = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])

    # A smooth, position-dependent stand-in for the ML model.
    def e_ml_fn(pos):
        return -3.0 * jnp.sum(jnp.cos(pos[:4, 0])) - 0.5 * jnp.sum(pos[:4] ** 2)

    def e_total_fn(pos):
        s = ml_scale_from_positions(pos, mid, mm_switch_on=8.0, ml_switch_width=1.5)
        e_mm = cgenff_mm_energy(pos, tidx, mid, q, SIG, EPS, **KW)
        return s * e_ml_fn(pos) + e_mm

    for sep in (5.0, 6.8, 7.2, 7.9, 9.0, 12.0):
        pos = jnp.array(
            [[0.0, 0, 0], [1.0, 0.2, 0], [sep, 0, 0], [sep + 1.0, 0.2, 0], [0, 0, 0]]
        )
        e_ml = e_ml_fn(pos)
        f_ml = -jax.grad(e_ml_fn)(pos)

        out = hybrid_energy_forces(e_ml, f_ml, pos, tidx, mid, q, SIG, EPS, **KW)
        ref_e = e_total_fn(pos)
        ref_f = -jax.grad(e_total_fn)(pos)
        ref_f = jnp.where((mid >= 0)[:, None], ref_f, 0.0)

        assert bool(jnp.all(jnp.isfinite(out.forces))), f"NaN force at sep={sep}"
        assert float(out.energy) == pytest.approx(float(ref_e), rel=1e-8, abs=1e-10), sep
        assert np.allclose(np.asarray(out.forces), np.asarray(ref_f), rtol=1e-6, atol=1e-8), sep


def test_naive_scaled_forces_would_be_wrong_in_the_handoff():
    """Guard: the switch-gradient term is not negligible where it matters."""
    from mmml.models.hybrid_energy import hybrid_energy_forces

    pos, tidx, mid, q = _dimer(7.2)
    e_ml = jnp.float64(-5.0)
    f_ml = jnp.zeros((5, 3))  # zero ML force isolates the switch-gradient term
    out = hybrid_energy_forces(e_ml, f_ml, pos, tidx, mid, q, SIG, EPS, **KW)
    # scale*F_ML would be exactly zero here; the real force is not.
    assert float(jnp.max(jnp.abs(out.forces))) > 1e-6


def test_padding_carries_no_force():
    pos, tidx, mid, q = _dimer(7.0)
    out = _call(pos, tidx, mid, q, jnp.float64(-2.0), jnp.ones((5, 3)))
    assert np.allclose(np.asarray(out.forces)[4], 0.0)


def test_monomer_keeps_ml_fully_on_and_no_mm():
    from mmml.models.hybrid_energy import hybrid_energy_forces

    pos = jnp.array([[0.0, 0, 0], [1.0, 0, 0], [0, 0, 0]])
    tidx = jnp.array([0, 1, -1]); mid = jnp.array([0, 0, -1])
    q = jnp.array([-0.3, 0.3, 0.0])
    out = hybrid_energy_forces(jnp.float64(-7.0), jnp.zeros((3, 3)), pos, tidx, mid, q, SIG, EPS, **KW)
    assert float(out.ml_scale) == pytest.approx(1.0)
    assert float(out.e_mm) == 0.0
    assert float(out.energy) == pytest.approx(-7.0)


def test_vmap_over_a_batch():
    from mmml.models.hybrid_energy import hybrid_energy_forces

    tidx = jnp.array([0, 1, 0, 1, -1]); mid = jnp.array([0, 0, 1, 1, -1])
    q = jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0])
    batch_pos = jnp.stack([_dimer(s)[0] for s in (5.0, 7.2, 9.0)])
    e_ml = jnp.array([-5.0, -4.0, -3.0])
    f_ml = jnp.zeros((3, 5, 3))

    f = lambda p, e, fm: hybrid_energy_forces(e, fm, p, tidx, mid, q, SIG, EPS, **KW).energy
    out = jax.vmap(f)(batch_pos, e_ml, f_ml)
    assert out.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_apply_hybrid_mm_to_output_respects_the_batch_layout():
    """prepare_batches_jit flattens R/F but leaves cgenff fields (batch, natoms)."""
    from mmml.models.hybrid_energy import apply_hybrid_mm_to_output

    batch_size, natoms = 3, 5
    pos = jnp.stack([_dimer(s)[0] for s in (4.0, 7.2, 15.0)])   # (3,5,3)
    tidx = jnp.tile(jnp.array([0, 1, 0, 1, -1]), (batch_size, 1))
    mid = jnp.tile(jnp.array([0, 0, 1, 1, -1]), (batch_size, 1))
    q = jnp.tile(jnp.array([-0.3, 0.15, -0.3, 0.15, 0.0]), (batch_size, 1))

    batch = {                       # R flat, cgenff fields per-structure
        "R": pos.reshape(batch_size * natoms, 3),
        "cgenff_type_idx": tidx,
        "mol_id": mid,
        "cgenff_charge": q,
    }
    output = {                      # E as (batch,1), F flat -- as the model returns
        "energy": jnp.array([[-5.0], [-4.0], [-3.0]]),
        "forces": jnp.zeros((batch_size * natoms, 3)),
    }

    out = apply_hybrid_mm_to_output(
        output, batch, batch_size, SIG, EPS, **KW
    )
    # shapes preserved
    assert out["energy"].shape == output["energy"].shape
    assert out["forces"].shape == output["forces"].shape
    assert bool(jnp.all(jnp.isfinite(out["forces"])))

    # regimes: 4.0 -> ML only, 7.2 -> handoff, 15.0 -> everything off
    assert float(out["ml_scale"][0]) == pytest.approx(1.0)
    assert 0.0 < float(out["ml_scale"][1]) < 1.0
    assert float(out["ml_scale"][2]) == pytest.approx(0.0)
    assert float(out["energy"][0, 0]) == pytest.approx(-5.0)   # pure ML
    assert float(out["energy"][2, 0]) == pytest.approx(0.0, abs=1e-12)  # nothing left
