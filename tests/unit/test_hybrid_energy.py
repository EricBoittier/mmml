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


def test_hybrid_forward_never_passes_cgenff_to_the_model():
    """Guard against double-counting the MM term.

    The Spooky model has its own in-model CGenFF VdW, gated on
    cgenff_type_idx/cgenff_master_sigmas/cgenff_master_epsilons -- all of which
    default to None. hybrid_forward adds E_MM itself, so it must NOT hand those
    to the model, or MM would be counted twice. Pin the forward's kwargs.
    """
    from mmml.models.hybrid_energy import hybrid_forward

    seen = []

    def spy_apply(params, **kw):
        seen.append(set(kw))
        n = kw["positions"].shape[0]
        bs = kw["batch_size"]
        return {
            "energy": jnp.zeros((bs, 1)),
            "forces": jnp.zeros((n, 3)),
        }

    hybrid_forward(spy_apply, {}, _batch(7.0), 1, SIG, EPS, **KW)

    assert seen, "model was never called"
    forbidden = {"cgenff_type_idx", "cgenff_master_sigmas", "cgenff_master_epsilons"}
    for kw in seen:
        assert not (kw & forbidden), f"hybrid_forward leaked MM args to the model: {kw & forbidden}"
    # and it really did run the three forwards (AB, A, B)
    assert len(seen) == 3


# --------------------------------------------------------------------------
# Opt-in: learnt charges as a CORRECTION to the CGenFF charges.
# --------------------------------------------------------------------------

def _charged_model(dq):
    """Fake model that predicts a fixed per-atom charge `dq`."""
    def apply(params, **kw):
        n = kw["positions"].shape[0]
        bs = kw["batch_size"]
        base = _fake_model_apply(params, **kw)
        base["charges"] = jnp.asarray(dq)[:n]
        return base
    return apply


def test_charge_correction_is_off_by_default():
    """Default MM electrostatics uses the CGenFF charges alone."""
    from mmml.models.hybrid_energy import hybrid_forward

    b = _batch(9.0)
    dq = jnp.array([0.2, -0.1, 0.3, -0.05, 0.0])
    off = hybrid_forward(_charged_model(dq), {}, b, 1, SIG, EPS, **KW)
    on = hybrid_forward(
        _charged_model(dq), {}, b, 1, SIG, EPS,
        mm_charge_mode="fixed_plus_latent", **KW,
    )
    # in the MM tail the correction must actually change E_MM
    assert _f(off["e_mm"]) != pytest.approx(_f(on["e_mm"]))


def test_latent_mode_changes_e_mm_vs_fixed():
    """Mode B replaces CGenFF charges; energy must differ from Mode A."""
    from mmml.models.hybrid_energy import hybrid_forward

    b = _batch(9.0)
    dq = jnp.array([0.2, -0.1, 0.3, -0.05, 0.0])
    fixed = hybrid_forward(_charged_model(dq), {}, b, 1, SIG, EPS, **KW)
    latent = hybrid_forward(
        _charged_model(dq), {}, b, 1, SIG, EPS,
        mm_charge_mode="latent", **KW,
    )
    combo = hybrid_forward(
        _charged_model(dq), {}, b, 1, SIG, EPS,
        mm_charge_mode="fixed_plus_latent", **KW,
    )
    assert _f(fixed["e_mm"]) != pytest.approx(_f(latent["e_mm"]))
    assert _f(latent["e_mm"]) != pytest.approx(_f(combo["e_mm"]))


def test_ewald_monomer_ml_plus_mm_fixed_and_latent():
    """Monomer ML + MM(Ewald): well-separated dimer so s(r_com) -> 0, both
    mm_charge_mode legs that don't need a liquid box (fixed, latent).

    This is the fast/mocked regression for
    ``examples/hybrid_mm_charges/monomer_ml_mm_ewald_example.py`` (see
    ``docs/hybrid-mm-charges.md``): past the ML->MM handoff tail, the switched
    ML-dimer correction vanishes on its own, so ``E_total`` is effectively
    "ML monomers + MM" regardless of ``lr_solver``. Locks in that both charge
    modes actually run with the native Ewald solver and give distinct, finite
    ``e_mm``.
    """
    from mmml.models.hybrid_energy import hybrid_forward

    sep = KW["mm_switch_on"] + KW["mm_switch_width"] + 10.0  # well past the tail
    b = _batch(sep)
    dq = jnp.array([0.2, -0.1, 0.3, -0.05, 0.0])
    ewald_kw = dict(lr_solver="ewald", include_lj=False, pme_box_length=30.0)

    fixed = hybrid_forward(_charged_model(dq), {}, b, 1, SIG, EPS, **KW, **ewald_kw)
    latent = hybrid_forward(
        _charged_model(dq), {}, b, 1, SIG, EPS,
        mm_charge_mode="latent", **KW, **ewald_kw,
    )

    for out in (fixed, latent):
        assert np.isfinite(_f(out["energy"]))
        assert np.isfinite(_f(out["e_mm"]))
        # Past the handoff tail the dimer switch is fully off.
        assert _f(out["ml_scale"]) == pytest.approx(0.0, abs=1e-8)

    assert _f(fixed["e_mm"]) != pytest.approx(_f(latent["e_mm"]))


def test_charge_correction_requires_a_charge_head():
    """A model without charges=True must fail loudly, not silently no-op."""
    from mmml.models.hybrid_energy import hybrid_forward

    with pytest.raises(ValueError, match="charges=True"):
        hybrid_forward(
            _fake_model_apply, {}, _batch(9.0), 1, SIG, EPS,
            mm_charge_mode="fixed_plus_latent", **KW,
        )
    with pytest.raises(ValueError, match="charges=True"):
        hybrid_forward(
            _fake_model_apply, {}, _batch(9.0), 1, SIG, EPS,
            mm_charge_mode="latent", **KW,
        )


def test_correction_is_projected_net_zero_per_monomer():
    """The invariant: q_cgenff + dq stays neutral on EVERY monomer.

    Unprojected, a net monomer charge turns the far-field MM electrostatics into
    monopole-monopole (~1/r) instead of dipole-dipole (~1/r^3).
    """
    from mmml.models.cgenff_mm import neutralize_per_monomer

    mol_id = jnp.array([0, 0, 1, 1, -1])
    dq = jnp.array([0.7, 0.1, -0.4, 0.2, 99.0])   # net charge on both monomers
    out = neutralize_per_monomer(dq, mol_id)
    assert float(jnp.sum(out[:2])) == pytest.approx(0.0, abs=1e-12)   # monomer A
    assert float(jnp.sum(out[2:4])) == pytest.approx(0.0, abs=1e-12)  # monomer B
    assert float(out[4]) == 0.0                                       # padding untouched
    # the *shape* of the correction survives (only the mean is removed)
    assert float(out[0] - out[1]) == pytest.approx(float(dq[0] - dq[1]))


def test_uniform_correction_is_a_no_op():
    """A constant shift carries no information -> projected away entirely."""
    from mmml.models.cgenff_mm import neutralize_per_monomer

    mol_id = jnp.array([0, 0, 1, 1, -1])
    out = neutralize_per_monomer(jnp.array([0.5, 0.5, 0.5, 0.5, 0.0]), mol_id)
    assert np.allclose(np.asarray(out), 0.0, atol=1e-12)


def test_corrected_charges_keep_the_dimer_neutral_in_the_mm_term():
    """End-to-end: with the correction on, monomers stay neutral."""
    from mmml.models.cgenff_mm import neutralize_per_monomer

    b = _batch(9.0)
    dq = jnp.array([0.4, -0.1, 0.25, 0.05, 7.0])
    q_eff = b["cgenff_charge"][0] + neutralize_per_monomer(dq, b["mol_id"][0])
    for sel in (slice(0, 2), slice(2, 4)):
        # cgenff_charge in this fixture is -0.3/+0.15 per monomer = -0.15, so we
        # assert the *correction* added no net charge, not that q_eff sums to 0.
        added = float(jnp.sum(q_eff[sel] - b["cgenff_charge"][0][sel]))
        assert added == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# The hybrid settings are CONFIG, not data: they must survive jit.
# --------------------------------------------------------------------------

def _cfg(**over):
    from mmml.models.hybrid_energy import HybridMMConfig

    kw = dict(
        master_sigmas=tuple(float(x) for x in SIG),
        master_epsilons=tuple(float(x) for x in EPS),
        **KW,
    )
    kw.update(over)
    return HybridMMConfig.coerce(kw)


def test_charge_correction_survives_jit():
    """Regression: `if charge_correction:` on a traced bool raised
    TracerBoolConversionError and killed the first hybrid training run.

    Only a test that goes through jit can catch this -- the eager tests all
    passed while training died on step 1.
    """
    from mmml.models.physnetjax.physnetjax.training.evalstep import _eval_forward

    dq = jnp.array([0.2, -0.1, 0.3, -0.05, 0.0])
    model = _charged_model(dq)
    b = _batch(9.0)

    fn = jax.jit(
        lambda batch, cfg: _eval_forward(model, {}, batch, 1, cfg),
        static_argnums=(1,),
    )
    e_off = _f(fn(b, _cfg())["energy"])
    e_on = _f(fn(b, _cfg(mm_charge_mode="fixed_plus_latent"))["energy"])
    e_latent = _f(fn(b, _cfg(mm_charge_mode="latent"))["energy"])
    assert np.isfinite(e_off) and np.isfinite(e_on) and np.isfinite(e_latent)
    assert e_off != pytest.approx(e_on)
    assert e_off != pytest.approx(e_latent)


def test_config_is_hashable_so_it_can_be_a_static_argument():
    """A dict cannot be static (unhashable) -- that is why the flags traced."""
    assert hash(_cfg()) == hash(_cfg())
    assert hash(_cfg(mm_charge_mode="fixed_plus_latent")) != hash(_cfg())
    assert hash(_cfg(mm_charge_mode="latent")) != hash(_cfg())
    assert _cfg() == _cfg()


def test_config_coerces_a_plain_kwargs_dict():
    """The CLI builds a dict; coerce() is the jit-boundary adapter."""
    from mmml.models.hybrid_energy import HybridMMConfig

    d = dict(master_sigmas=SIG, master_epsilons=EPS, charge_correction=True, **KW)
    cfg = HybridMMConfig.coerce(d)
    assert cfg.charge_correction is True
    assert cfg.mm_charge_mode == "fixed_plus_latent"
    assert cfg.mm_switch_on == 8.0
    assert np.allclose(np.asarray(cfg.kwargs()["master_sigmas"]), np.asarray(SIG))
    assert HybridMMConfig.coerce(None) is None
    assert HybridMMConfig.coerce(cfg) is cfg

    latent = HybridMMConfig.coerce(
        dict(master_sigmas=SIG, master_epsilons=EPS, mm_charge_mode="latent", **KW)
    )
    assert latent.mm_charge_mode == "latent"
    assert latent.charge_correction is False
