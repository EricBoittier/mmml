"""Unit tests for the fluctuating-charge electrostatic embedding term."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.energy.registry import EnergyContext
from mmml.md.energy.terms.ml_mm_elec import COULOMB_KCAL, MLMMElectrostaticTerm
from mmml.md.system import FFParams, MolecularSystem

KCAL_TO_EV = 1.0 / 23.060549


class _FakeModel:
    """Charges that depend on geometry, so dq/dR is non-zero and testable.

    q_0 = -alpha * d(0,1) and q_1 = +alpha * d(0,1): the "solute" polarises as
    its two atoms separate, mimicking the ion-pair formation the real model
    describes.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def apply(self, params, *, atomic_numbers, positions, **kwargs):
        import jax.numpy as jnp

        d = jnp.sqrt(jnp.sum((positions[1] - positions[0]) ** 2) + 1e-12)
        q = jnp.stack([-self.alpha * d, self.alpha * d])
        return {"charges": q}


def _system(solute_q=(0.0, 0.0), box=None, n_solvent=1):
    """2-atom ML solute at the origin, MM point charges along +x."""
    solute = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    solvent = np.array([[5.0 + 3.0 * k, 0.0, 0.0] for k in range(n_solvent)])
    R = np.concatenate([solute, solvent])
    n = R.shape[0]
    charges = np.concatenate([np.asarray(solute_q), np.full(n_solvent, 0.5)])
    ff = FFParams(
        charges=charges,
        epsilon=np.zeros(n),
        rmin_half=np.ones(n),
        at_codes=np.zeros(n, dtype=np.int32),
        exclusions=np.array([[0, 1]], dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=R,
        Z=np.array([17, 7] + [8] * n_solvent),
        box=box,
        mol_id=np.array([0, 0] + list(range(1, n_solvent + 1)), dtype=np.int32),
        monomer_indices=[np.array([0, 1])] + [np.array([2 + k]) for k in range(n_solvent)],
        ff_params=ff,
    )


def _pairs(system):
    """All intermolecular pairs, as the driver's neighbor_fn would supply."""
    n = system.n_atoms
    mol = np.asarray(system.mol_id)
    i, j = np.triu_indices(n, 1)
    keep = mol[i] != mol[j]
    return {
        "pair_i": np.asarray(i[keep], dtype=np.int32),
        "pair_j": np.asarray(j[keep], dtype=np.int32),
        "pair_mask": np.ones(int(keep.sum()), dtype=np.int8),
    }


def _make(system, **kw):
    ctx = EnergyContext(model=_FakeModel(), params={}, options={"ml_atoms": [0, 1]})
    term = MLMMElectrostaticTerm(ml_atoms=[0, 1], **kw)
    return term.make(system, ctx)


def test_energy_matches_hand_computed_coulomb():
    import jax.numpy as jnp

    system = _system()
    # Well inside the switch-on radius so switch() == 1, and damping off, so the
    # value is exactly the bare Coulomb sum.
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0, damping_sigma_A=0.0)
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    got = float(fns.jax_energy_fn(jnp.asarray(system.R), **kw))

    alpha, d = 0.1, 1.0
    q = np.array([-alpha * d, alpha * d])  # sums to zero already
    expected = COULOMB_KCAL * (q[0] * 0.5 / 5.0 + q[1] * 0.5 / 4.0) * KCAL_TO_EV
    assert got == pytest.approx(expected, rel=1e-9)


def test_damping_matches_erf_form():
    """Default damping is erf(r/sigma)/r, applied per pair."""
    import jax.numpy as jnp
    from scipy.special import erf

    system = _system()
    sigma = 1.0
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0, damping_sigma_A=sigma)
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    got = float(fns.jax_energy_fn(jnp.asarray(system.R), **kw))

    alpha, d = 0.1, 1.0
    q = np.array([-alpha * d, alpha * d])
    expected = (
        COULOMB_KCAL
        * (
            q[0] * 0.5 * erf(5.0 / sigma) / 5.0
            + q[1] * 0.5 * erf(4.0 / sigma) / 4.0
        )
        * KCAL_TO_EV
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_damping_bounds_the_energy_at_short_range():
    """The reason damping exists: a bare 1/r diverges as an MM atom closes in.

    MM hydrogens have almost no Lennard-Jones core, so nothing else stops a
    solvent proton from reaching an ML atom carrying a large negative charge.
    Undamped, the pair energy grows without limit; damped, it saturates.
    """
    import jax.numpy as jnp

    system = _system()
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    bare = _make(system, switch_on_A=20.0, cutoff_A=25.0, damping_sigma_A=0.0)
    damped = _make(system, switch_on_A=20.0, cutoff_A=25.0, damping_sigma_A=1.0)

    def at(sep, fns):
        R = np.array(system.R, dtype=float)
        R[2] = R[0] + np.array([sep, 0.0, 0.0])  # walk the MM atom onto atom 0
        return abs(float(fns.jax_energy_fn(jnp.asarray(R), **kw)))

    # erf(r/sigma)/r -> 2/(sigma*sqrt(pi)) as r -> 0, so the collapsing pair's
    # contribution cannot exceed this however close the MM atom gets. The other
    # pair is bounded too, at a separation that stays near its original value.
    q_max, q_mm = 0.1, 0.5
    bound = COULOMB_KCAL * q_max * q_mm * 2.0 / (1.0 * np.sqrt(np.pi)) * KCAL_TO_EV

    for sep in (0.1, 0.01, 1e-6):
        assert np.isfinite(at(sep, damped))
        assert at(sep, damped) < 4.0 * bound
    # Undamped, the same approach runs away.
    assert at(1e-6, bare) > 1e4 * bound


def test_negative_damping_sigma_is_rejected():
    with pytest.raises(ValueError, match="damping_sigma_A"):
        MLMMElectrostaticTerm(ml_atoms=[0, 1], damping_sigma_A=-1.0)


def test_forces_include_the_charge_derivative():
    """The point of the term: dq/dR must contribute, not just d(1/r)/dR."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    system = _system()
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0)
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    R = jnp.asarray(system.R)

    full = np.asarray(jax.grad(lambda r: fns.jax_energy_fn(r, **kw))(R))

    # Same energy but with the charges frozen at their current value: the
    # difference is exactly the dq/dR contribution.
    q_fixed = np.asarray(fns.jax_energy_fn.ml_charges(R))

    def frozen(r):
        d0 = jnp.linalg.norm(r[2] - r[0])
        d1 = jnp.linalg.norm(r[2] - r[1])
        return (
            COULOMB_KCAL * (q_fixed[0] * 0.5 / d0 + q_fixed[1] * 0.5 / d1) * KCAL_TO_EV
        )

    frozen_grad = np.asarray(jax.grad(frozen)(R))
    assert not np.allclose(full, frozen_grad, atol=1e-8), (
        "gradients match the fixed-charge result, so dq/dR is being dropped"
    )
    # The frozen model has no charge response on the solute pair coordinate.
    assert abs(full[0, 0] - frozen_grad[0, 0]) > 1e-6


def test_numerical_gradient_agrees():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    system = _system()
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0)
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}

    def e(r):
        return float(fns.jax_energy_fn(jnp.asarray(r), **kw))

    analytic = np.asarray(jax.grad(lambda r: fns.jax_energy_fn(r, **kw))(jnp.asarray(system.R)))
    h = 1e-6
    for atom in (0, 1, 2):
        for comp in range(3):
            plus = system.R.copy(); plus[atom, comp] += h
            minus = system.R.copy(); minus[atom, comp] -= h
            num = (e(plus) - e(minus)) / (2 * h)
            assert num == pytest.approx(analytic[atom, comp], abs=1e-6)


def test_q0_mode_enforces_neutrality():
    import jax.numpy as jnp

    class _Biased(_FakeModel):
        def apply(self, params, *, atomic_numbers, positions, **kwargs):
            out = super().apply(params, atomic_numbers=atomic_numbers, positions=positions)
            return {"charges": out["charges"] + 0.3}  # net +0.6

    system = _system()
    ctx = EnergyContext(model=_Biased(), params={}, options={"ml_atoms": [0, 1]})
    raw = MLMMElectrostaticTerm(ml_atoms=[0, 1], charge_mode="raw").make(system, ctx)
    q0 = MLMMElectrostaticTerm(ml_atoms=[0, 1], charge_mode="q0").make(system, ctx)
    R = jnp.asarray(system.R)
    assert float(jnp.sum(raw.jax_energy_fn.ml_charges(R))) == pytest.approx(0.6)
    assert float(jnp.sum(q0.jax_energy_fn.ml_charges(R))) == pytest.approx(0.0, abs=1e-12)


def test_double_counting_guard():
    """Non-zero MM charges on the solute must be refused, not silently added."""
    system = _system(solute_q=(0.3, -0.3))
    with pytest.raises(ValueError, match="double-count"):
        _make(system)


def test_only_cross_pairs_contribute():
    """ML-ML pairs belong to the model; MM-MM pairs belong to mm_nonbonded."""
    import jax.numpy as jnp

    system = _system(n_solvent=2)
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0)
    pairs = _pairs(system)
    kw = {k: jnp.asarray(v) for k, v in pairs.items()}
    with_all = float(fns.jax_energy_fn(jnp.asarray(system.R), **kw))

    # Adding an ML-ML pair and an MM-MM pair must not change the energy.
    extra_i = np.concatenate([pairs["pair_i"], [0, 2]]).astype(np.int32)
    extra_j = np.concatenate([pairs["pair_j"], [1, 3]]).astype(np.int32)
    extra_m = np.concatenate([pairs["pair_mask"], [1, 1]]).astype(np.int8)
    with_extra = float(fns.jax_energy_fn(
        jnp.asarray(system.R),
        pair_i=jnp.asarray(extra_i), pair_j=jnp.asarray(extra_j),
        pair_mask=jnp.asarray(extra_m),
    ))
    assert with_extra == pytest.approx(with_all, rel=1e-12)


def test_masked_pairs_are_ignored_and_finite():
    import jax
    import jax.numpy as jnp

    system = _system()
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0)
    pairs = _pairs(system)
    live = float(fns.jax_energy_fn(
        jnp.asarray(system.R),
        **{k: jnp.asarray(v) for k, v in pairs.items()},
    ))
    # Pad with a self-pair (zero distance) that is masked off: must be ignored
    # and must not produce a NaN gradient.
    pi = np.concatenate([pairs["pair_i"], [0]]).astype(np.int32)
    pj = np.concatenate([pairs["pair_j"], [0]]).astype(np.int32)
    pm = np.concatenate([pairs["pair_mask"], [0]]).astype(np.int8)
    kw = dict(pair_i=jnp.asarray(pi), pair_j=jnp.asarray(pj), pair_mask=jnp.asarray(pm))
    padded = float(fns.jax_energy_fn(jnp.asarray(system.R), **kw))
    assert padded == pytest.approx(live, rel=1e-12)
    g = np.asarray(jax.grad(lambda r: fns.jax_energy_fn(r, **kw))(jnp.asarray(system.R)))
    assert np.all(np.isfinite(g))


def test_switch_turns_the_interaction_off_beyond_the_cutoff():
    import jax.numpy as jnp

    system = _system()
    fns = _make(system, switch_on_A=1.0, cutoff_A=2.0)  # solvent at 5 A is outside
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    assert float(fns.jax_energy_fn(jnp.asarray(system.R), **kw)) == pytest.approx(0.0)


def test_minimum_image_is_used_under_pbc():
    import jax.numpy as jnp

    box = np.eye(3) * 10.0
    system = _system(box=box)
    # MM charge at x=5 is 5 A from atom 0 directly, or 5 A the other way: with a
    # 10 A box the two are degenerate, so shift it to make MIC observable.
    R = np.asarray(system.R).copy()
    R[2, 0] = 9.5  # 9.5 A direct, 0.5 A by minimum image
    import dataclasses

    system = dataclasses.replace(system, R=R)
    fns = _make(system, switch_on_A=3.0, cutoff_A=4.0)
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    e = float(fns.jax_energy_fn(jnp.asarray(R), **kw))
    # Without MIC the pair is beyond the cutoff and the energy would be zero.
    assert abs(e) > 1e-6


def test_term_is_registered():
    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy.registry import available_terms, get_term

    assert "ml_mm_elec" in available_terms()
    assert get_term("ml_mm_elec") is MLMMElectrostaticTerm


def test_construction_validation():
    with pytest.raises(ValueError, match="switch_on_A"):
        MLMMElectrostaticTerm(ml_atoms=[0], switch_on_A=12.0, cutoff_A=10.0)
    with pytest.raises(ValueError, match="charge_mode"):
        MLMMElectrostaticTerm(ml_atoms=[0], charge_mode="bogus")


def test_solute_is_unfolded_across_the_periodic_boundary():
    """A solute split by wrapping must reach the model as one molecule.

    The integrator wraps coordinates into the primary cell, so a solute sitting
    on a box face returns with its atoms on opposite sides. Passed through
    as-is, every atom falls outside every other atom's cutoff, the model's graph
    disconnects and its charges stop summing to the total. The charges must not
    depend on which periodic image each atom is reported in.
    """
    import jax.numpy as jnp

    box = np.diag([10.0, 10.0, 10.0])
    system = _system(box=box)
    fns = _make(system, switch_on_A=20.0, cutoff_A=25.0)

    intact = np.array(system.R, dtype=float)
    q_intact = np.asarray(fns.jax_energy_fn.ml_charges(jnp.asarray(intact)))

    # Same geometry, but atom 1 reported in the neighbouring image: the bond is
    # unchanged physically, while the raw separation becomes 9 A instead of 1 A.
    wrapped = intact.copy()
    wrapped[1, 0] -= 10.0
    q_wrapped = np.asarray(fns.jax_energy_fn.ml_charges(jnp.asarray(wrapped)))

    assert np.allclose(q_intact, q_wrapped, atol=1e-10), (
        f"charges changed when the solute was wrapped: {q_intact} vs {q_wrapped}"
    )


def test_charge_clip_bounds_the_embedding_charges():
    """Charges are capped, and untouched inside the cap.

    The cap exists to break a feedback loop, not to reshape normal operation:
    a hard clip has unit gradient inside its bound, so physical charges and
    their dq/dR are unaffected, and only the runaway is caught.
    """
    import jax.numpy as jnp

    system = _system()
    # _FakeModel gives q = +-alpha*d(0,1); stretch the pair so |q| exceeds 1.
    far = np.array(system.R, dtype=float)
    far[1, 0] = 15.0                      # d(0,1) = 15 -> |q| = 1.5 unclipped
    clipped = _make(system, switch_on_A=20.0, cutoff_A=25.0, charge_clip=1.0)
    unclipped = _make(system, switch_on_A=20.0, cutoff_A=25.0, charge_clip=None)

    q_un = np.asarray(unclipped.jax_energy_fn.ml_charges(jnp.asarray(far)))
    q_cl = np.asarray(clipped.jax_energy_fn.ml_charges(jnp.asarray(far)))
    assert np.abs(q_un).max() > 1.0, "fixture should exceed the cap"
    assert np.abs(q_cl).max() <= 1.0 + 1e-9

    # Neutrality survives the clip: it is applied before the q0 correction,
    # because a net charge in a periodic box is the worse error.
    assert q_cl.sum() == pytest.approx(0.0, abs=1e-9)

    # Inside the cap, clipping changes nothing.
    R = jnp.asarray(system.R)
    a = np.asarray(clipped.jax_energy_fn.ml_charges(R))
    b = np.asarray(unclipped.jax_energy_fn.ml_charges(R))
    assert np.allclose(a, b, atol=1e-12)


def test_charge_clip_rejects_nonpositive():
    with pytest.raises(ValueError, match="charge_clip"):
        MLMMElectrostaticTerm(ml_atoms=[0, 1], charge_clip=0.0)


def test_charge_gradient_can_be_frozen():
    """With charge_gradient=False the force is the fixed-charge one.

    This is deliberately an approximation, so the test pins exactly what it
    changes: the energy is identical (charges are still computed and still
    enter it), and the gradient becomes the one a fixed-charge model would
    give. That is the whole content of the option.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    system = _system()
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    R = jnp.asarray(system.R)
    live = _make(system, switch_on_A=20.0, cutoff_A=25.0, charge_gradient=True)
    frozen = _make(system, switch_on_A=20.0, cutoff_A=25.0, charge_gradient=False)

    # Same energy either way.
    assert float(frozen.jax_energy_fn(R, **kw)) == pytest.approx(
        float(live.jax_energy_fn(R, **kw)), rel=1e-12
    )

    # The frozen gradient equals the hand-built fixed-charge gradient.
    q_fixed = np.asarray(live.jax_energy_fn.ml_charges(R))

    def fixed_charge_energy(r):
        d0 = jnp.linalg.norm(r[2] - r[0])
        d1 = jnp.linalg.norm(r[2] - r[1])
        from jax.scipy.special import erf

        return (
            COULOMB_KCAL
            * (
                q_fixed[0] * 0.5 * erf(d0 / 1.0) / d0
                + q_fixed[1] * 0.5 * erf(d1 / 1.0) / d1
            )
            * KCAL_TO_EV
        )

    g_frozen = np.asarray(jax.grad(lambda r: frozen.jax_energy_fn(r, **kw))(R))
    g_fixed = np.asarray(jax.grad(fixed_charge_energy)(R))
    assert np.allclose(g_frozen, g_fixed, atol=1e-8)

    # And it differs from the full gradient, i.e. something real was dropped.
    g_live = np.asarray(jax.grad(lambda r: live.jax_energy_fn(r, **kw))(R))
    assert not np.allclose(g_live, g_frozen, atol=1e-8)


def test_ewald_cross_term_matches_direct_lattice_sum():
    """The Ewald path must reproduce the true periodic solute-solvent energy.

    Checked against a brute-force lattice sum over image cells, which is the
    quantity Ewald is a fast evaluation of. A cutoff ("mic") result is included
    to show what is being fixed: it is missing the long-range tail entirely.
    """
    import jax.numpy as jnp

    import dataclasses

    L = 12.0
    box = np.diag([L, L, L])
    # The system must be NET NEUTRAL: a charged periodic cell has no
    # convergent lattice sum, and Ewald silently applies a neutralising
    # background, so the two would differ for reasons unrelated to the
    # implementation. Two solvent atoms carrying +-0.5 against a solute that
    # already sums to zero.
    system = _system(box=box, n_solvent=2)
    q = np.asarray(system.ff_params.charges, dtype=float).copy()
    q[2], q[3] = 0.5, -0.5
    system = dataclasses.replace(
        system, ff_params=dataclasses.replace(system.ff_params, charges=q)
    )
    assert abs(q.sum()) < 1e-12
    kw = {k: jnp.asarray(v) for k, v in _pairs(system).items()}
    R = jnp.asarray(system.R)

    ew = _make(system, cutoff_A=5.0, switch_on_A=4.0,
               damping_sigma_A=0.0, charge_clip=None, lr_solver="ewald")
    mic = _make(system, cutoff_A=5.0, switch_on_A=4.0,
                damping_sigma_A=0.0, charge_clip=None, lr_solver="mic")

    q_ml = np.asarray(ew.jax_energy_fn.ml_charges(R))
    q_mm = np.asarray(system.ff_params.charges)
    pos = np.asarray(system.R)

    # Direct lattice sum of the solute-solvent (cross) energy.
    n_img = 12
    total = 0.0
    for a in (0, 1):                      # ML atoms
        for b in (2, 3):                  # MM atoms
            for ix in range(-n_img, n_img + 1):
                for iy in range(-n_img, n_img + 1):
                    for iz in range(-n_img, n_img + 1):
                        shift = np.array([ix, iy, iz]) * L
                        d = np.linalg.norm(pos[b] + shift - pos[a])
                        total += q_ml[a] * q_mm[b] / d
    direct = COULOMB_KCAL * total * KCAL_TO_EV

    got_ew = float(ew.jax_energy_fn(R, **kw))
    got_mic = float(mic.jax_energy_fn(R, **kw))

    # Ewald should land close to the lattice sum; the cutoff result should not.
    assert abs(got_ew - direct) < 0.15 * abs(direct), (
        f"ewald {got_ew:.4f} vs direct lattice sum {direct:.4f}"
    )
    assert abs(got_mic - direct) > abs(got_ew - direct), (
        "the cutoff result should be further from the lattice sum than Ewald; "
        f"mic {got_mic:.4f}, ewald {got_ew:.4f}, direct {direct:.4f}"
    )


def test_ewald_requires_a_box():
    import pytest as _pytest

    system = _system(box=None)
    with _pytest.raises(ValueError, match="periodic box"):
        _make(system, lr_solver="ewald")


def test_lr_solver_is_validated():
    with pytest.raises(ValueError, match="lr_solver"):
        MLMMElectrostaticTerm(ml_atoms=[0, 1], lr_solver="pme")
