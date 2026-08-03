"""``MLMMPolarisationTerm``: induction of the ML solute by the MM charges.

``E_pol = -1/2 sum_i alpha_i |E_i|^2``. Three properties are asserted here
because they are what the term claims about itself, and each is checkable
without re-deriving the formula:

* it is **always stabilising** (the energy is a negative-definite quadratic);
* it scales as the field **squared**, so it is near zero for a neutral reactant
  and grows as charge separates -- the reason it was expected to supply a
  *differential* solvent effect;
* Thole damping **removes the attractive singularity**. Undamped, ``-alpha|E|^2``
  diverges downward as an MM atom approaches, which is a well the integrator
  falls into. Damped, the field vanishes at contact, so the energy goes to zero
  from below rather than to minus infinity -- see
  ``test_energy_vanishes_at_contact_instead_of_diverging``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from mmml.md.energy import EnergyContext, available_terms  # noqa: E402
from mmml.md.energy.terms import MLMMPolarisationTerm  # noqa: E402
from mmml.md.system import FFParams, MolecularSystem  # noqa: E402

ML_IDX = [0, 1]          # C, Cl -- the "solute"
MM_IDX = [2, 3, 4]       # one water -- the "solvent"


def _system(mm_charges=(-0.834, 0.417, 0.417), sep: float = 4.0, box=None):
    """Two ML atoms on the x axis, one MM water displaced along +x by ``sep``."""
    pos = np.array([
        [0.0, 0.0, 0.0],          # C   (ML)
        [1.8, 0.0, 0.0],          # Cl  (ML)
        [1.8 + sep, 0.0, 0.0],    # O   (MM)
        [1.8 + sep + 0.96, 0.0, 0.0],
        [1.8 + sep, 0.96, 0.0],
    ])
    n = 5
    charges = np.array([0.0, 0.0, *mm_charges])
    ff = FFParams(
        charges=charges,
        epsilon=np.full(n, 0.1),
        rmin_half=np.full(n, 1.5),
        at_codes=np.arange(n, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=pos,
        Z=np.array([6, 17, 8, 1, 1], dtype=np.int32),
        box=box,
        mol_id=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        ff_params=ff,
    )


def _all_pairs(n: int = 5):
    i, j = np.triu_indices(n, 1)
    return {
        "pair_i": jnp.asarray(i.astype(np.int32)),
        "pair_j": jnp.asarray(j.astype(np.int32)),
        "pair_mask": jnp.ones(i.shape[0], dtype=jnp.int8),
    }


def _energy(system=None, *, term=None, pairs=None, R=None, **kw) -> float:
    system = _system() if system is None else system
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX) if term is None else term
    fn = term.make(system, EnergyContext()).jax_energy_fn
    pairs = _all_pairs(system.n_atoms) if pairs is None else pairs
    R = jnp.asarray(system.R if R is None else R)
    return float(fn(R, **pairs, **kw))


# ------------------------------------------------------------------ registry


def test_registered_under_ml_mm_pol():
    assert "ml_mm_pol" in available_terms()


def test_requests_the_shared_intermolecular_neighbour_family():
    req = MLMMPolarisationTerm(ml_atoms=ML_IDX).neighbor_request(_system())
    assert req.kind == "intermolecular"
    assert req.cutoff_A == pytest.approx(12.0)


# --------------------------------------------------------------- the physics


def test_is_always_stabilising():
    """``-1/2 alpha |E|^2`` cannot be positive for any configuration."""
    rng = np.random.default_rng(0)
    base = _system()
    for _ in range(12):
        R = base.R + rng.normal(scale=0.8, size=base.R.shape)
        assert _energy(base, R=R) <= 0.0


def test_is_zero_when_the_environment_carries_no_charge():
    """No field, no induction. This is the term's own zero point."""
    assert _energy(_system(mm_charges=(0.0, 0.0, 0.0))) == pytest.approx(0.0, abs=1e-14)


def test_scales_as_the_square_of_the_environment_charge():
    """Field-squared, not field: doubling every MM charge quadruples the energy.

    This is why the term is near zero for the neutral reactant and grows as the
    ion pair forms.
    """
    single = _energy(_system(mm_charges=(-0.834, 0.417, 0.417)))
    double = _energy(_system(mm_charges=(-1.668, 0.834, 0.834)))
    assert single < 0.0
    assert double == pytest.approx(4.0 * single, rel=1e-9)


def test_falls_off_as_the_environment_recedes():
    energies = [_energy(_system(sep=s)) for s in (3.0, 5.0, 8.0, 12.0)]
    assert all(e <= 0.0 for e in energies)
    # Monotonically weaker (less negative) with distance.
    assert all(b > a for a, b in zip(energies, energies[1:]))
    assert energies[-1] == pytest.approx(0.0, abs=1e-4)


def _one_on_one(r: float):
    """One ML atom and one MM charge at separation ``r``, and nothing else.

    The damping acts pair by pair, so isolating a single pair is the only way
    to see it: in a polyatomic environment the other partners keep contributing
    while one of them closes in.
    """
    ff = FFParams(
        charges=np.array([0.0, -1.0]),
        epsilon=np.full(2, 0.1),
        rmin_half=np.full(2, 1.5),
        at_codes=np.arange(2, dtype=np.int32),
        exclusions=np.empty((0, 2), dtype=np.int32),
        e14_pairs=np.empty((0, 2), dtype=np.int32),
    )
    return MolecularSystem(
        R=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
        Z=np.array([17, 8], dtype=np.int32),
        box=None,
        mol_id=np.array([0, 1], dtype=np.int32),
        ff_params=ff,
    )


def test_energy_vanishes_at_contact_instead_of_diverging():
    """The reason Thole damping is not cosmetic.

    Undamped, the field of a point charge goes as 1/r^2 and ``-alpha|E|^2`` as
    1/r^4 -- an attractive singularity the integrator falls into. Damped,
    ``1 - exp(-a u^3)`` goes as r^3 at short range, so the field goes as r and
    the energy as r^2: it turns over and returns to zero at contact. There is
    no well.
    """
    pairs = _all_pairs(2)
    term = MLMMPolarisationTerm(ml_atoms=[0])
    seps = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
    energies = [
        _energy(_one_on_one(r), term=term, pairs=pairs) for r in seps
    ]

    assert all(np.isfinite(e) for e in energies), "damping failed to bound the field"
    assert all(e <= 0.0 for e in energies)

    deepest = int(np.argmin(energies))
    assert deepest not in (0, len(seps) - 1), (
        f"expected a turnover, got a monotone profile: {energies}"
    )
    # Shallower at contact than at the minimum, by orders of magnitude.
    assert abs(energies[-1]) < abs(energies[deepest]) * 1e-2


def test_undamped_would_diverge_where_the_damped_term_does_not():
    """Mutation check: without damping this profile is monotone, not turning over.

    Guards against the turnover above being an artifact of the geometry rather
    than of the damping.
    """
    pairs = _all_pairs(2)
    seps = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05])
    damped = np.array([
        _energy(_one_on_one(r), term=MLMMPolarisationTerm(ml_atoms=[0]), pairs=pairs)
        for r in seps
    ])
    # -1/2 alpha (q/r^2)^2, the undamped form, in the same arbitrary scale.
    undamped = -((1.0 / seps**2) ** 2)

    assert int(np.argmin(undamped)) == len(seps) - 1, "undamped must be worst at contact"
    assert int(np.argmin(damped)) != len(seps) - 1


def test_gradient_is_finite_at_contact():
    """A finite energy is not enough; the integrator uses the force."""
    system = _system(sep=0.05)
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX)
    fn = term.make(system, EnergyContext()).jax_energy_fn
    pairs = _all_pairs(system.n_atoms)

    grad = jax.grad(lambda R: fn(R, **pairs))(jnp.asarray(system.R))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_masked_pairs_do_not_poison_the_gradient():
    """Padded slots are weighted zero, and 0 * NaN is NaN if r is not guarded."""
    system = _system()
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX)
    fn = term.make(system, EnergyContext()).jax_energy_fn

    live = _all_pairs(system.n_atoms)
    n_pad = 8
    padded = {
        "pair_i": jnp.concatenate([live["pair_i"], jnp.zeros(n_pad, jnp.int32)]),
        "pair_j": jnp.concatenate([live["pair_j"], jnp.zeros(n_pad, jnp.int32)]),
        "pair_mask": jnp.concatenate([live["pair_mask"], jnp.zeros(n_pad, jnp.int8)]),
    }

    R = jnp.asarray(system.R)
    e_live = float(fn(R, **live))
    e_padded = float(fn(R, **padded))
    assert e_padded == pytest.approx(e_live, rel=1e-12)

    grad = np.asarray(jax.grad(lambda x: fn(x, **padded))(R))
    assert np.all(np.isfinite(grad))


def test_only_ml_mm_pairs_contribute():
    """ML-ML is the model's own business; MM-MM belongs to mm_nonbonded."""
    system = _system()
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX)
    fn = term.make(system, EnergyContext()).jax_energy_fn
    R = jnp.asarray(system.R)

    cross = [(a, b) for a in ML_IDX for b in MM_IDX]
    only_cross = {
        "pair_i": jnp.asarray([a for a, _ in cross], dtype=jnp.int32),
        "pair_j": jnp.asarray([b for _, b in cross], dtype=jnp.int32),
        "pair_mask": jnp.ones(len(cross), dtype=jnp.int8),
    }
    assert float(fn(R, **only_cross)) == pytest.approx(
        float(fn(R, **_all_pairs(system.n_atoms))), rel=1e-12
    )


# ----------------------------------------------------------------- knobs


def test_elec_scale_ramps_the_term_linearly():
    full = _energy()
    assert _energy(elec_scale=0.5) == pytest.approx(0.5 * full, rel=1e-9)
    assert _energy(elec_scale=0.0) == pytest.approx(0.0, abs=1e-14)


def test_scale_multiplies_the_contribution():
    full = _energy()
    scaled = _energy(term=MLMMPolarisationTerm(ml_atoms=ML_IDX, scale=0.25))
    assert scaled == pytest.approx(0.25 * full, rel=1e-9)


def test_alpha_by_charge_deepens_the_well_for_a_developing_anion():
    """Chloride is roughly 1.6x the polarisability of covalent Cl.

    The interpolation is a stand-in for volume-scaled polarisabilities; what is
    asserted is only its direction and its end points.
    """
    system = _system()
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX, alpha_by_charge=True)

    neutral = _energy(system, term=term, ml_charges=jnp.asarray([0.0, 0.0]))
    anionic = _energy(system, term=term, ml_charges=jnp.asarray([0.0, -1.0]))
    partial = _energy(system, term=term, ml_charges=jnp.asarray([0.0, -0.5]))

    assert anionic < partial < neutral < 0.0
    # q = 0 must reproduce the plain neutral-alpha result exactly.
    assert neutral == pytest.approx(_energy(system, term=term), rel=1e-12)


def test_alpha_by_charge_off_ignores_the_charges():
    system = _system()
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX, alpha_by_charge=False)
    assert _energy(system, term=term, ml_charges=jnp.asarray([0.0, -1.0])) == (
        pytest.approx(_energy(system, term=term, ml_charges=jnp.asarray([0.0, 0.0])),
                      rel=1e-12)
    )


def test_alpha_scaling_is_clamped_so_an_overshoot_cannot_inflate_it():
    """A transient |q| > 1 must not grow alpha without bound."""
    system = _system()
    term = MLMMPolarisationTerm(ml_atoms=ML_IDX, alpha_by_charge=True)
    at_one = _energy(system, term=term, ml_charges=jnp.asarray([0.0, -1.0]))
    beyond = _energy(system, term=term, ml_charges=jnp.asarray([0.0, -3.0]))
    assert beyond == pytest.approx(at_one, rel=1e-12)


def test_minimum_image_is_applied_when_a_box_is_given():
    """A partner across the boundary must be seen at its nearest image."""
    side = 12.0
    box = np.diag([side] * 3)
    near = _system(sep=2.0, box=box)
    # Same water, pushed one full box length away: identical under MIC.
    wrapped = _system(sep=2.0, box=box)
    R = np.array(wrapped.R)
    R[MM_IDX] += np.array([side, 0.0, 0.0])

    assert _energy(near) == pytest.approx(_energy(wrapped, R=R), rel=1e-9)


# ----------------------------------------------------------------- failures


def test_refuses_to_build_without_the_solute_indices():
    with pytest.raises(ValueError, match="ml_atoms"):
        MLMMPolarisationTerm().make(_system(), EnergyContext())


def test_takes_the_solute_indices_from_the_context_when_not_given():
    term = MLMMPolarisationTerm()
    fns = term.make(_system(), EnergyContext(options={"ml_atoms": ML_IDX}))
    assert fns.jax_energy_fn is not None


def test_refuses_to_build_without_mm_charges():
    base = _system()
    system = MolecularSystem(
        R=base.R, Z=base.Z, box=None, mol_id=base.mol_id, ff_params=None
    )
    with pytest.raises(ValueError, match="ff_params"):
        MLMMPolarisationTerm(ml_atoms=ML_IDX).make(system, EnergyContext())


def test_names_the_elements_it_has_no_polarisability_for():
    system = _system()
    z = np.array(system.Z)
    z[1] = 79  # gold, deliberately absent from ALPHA_A3
    broken = MolecularSystem(
        R=system.R, Z=z, box=None, mol_id=system.mol_id, ff_params=system.ff_params
    )
    with pytest.raises(ValueError, match="79"):
        MLMMPolarisationTerm(ml_atoms=ML_IDX).make(broken, EnergyContext())


def test_requires_the_intermolecular_pair_list():
    system = _system()
    fn = MLMMPolarisationTerm(ml_atoms=ML_IDX).make(system, EnergyContext()).jax_energy_fn
    with pytest.raises(ValueError, match="pair list"):
        fn(jnp.asarray(system.R))
