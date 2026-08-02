"""Unit tests for the rxncoor reaction-coordinate umbrella term."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.energy.registry import EnergyContext
from mmml.md.energy.terms.rxncoor import ReactionCoordinateBiasTerm
from mmml.md.restraints import LinearDistanceCV
from mmml.md.system import MolecularSystem

# Cl(0), N(1), C(2) collinear: r(C-Cl) = 1.8, r(C-N) = 3.0, xi = -1.2
_R = np.array([[0.0, 0.0, 0.0], [4.8, 0.0, 0.0], [1.8, 0.0, 0.0]], dtype=np.float64)
_CV = LinearDistanceCV.difference(minuend=(2, 0), subtrahend=(2, 1))


def _system(box=None):
    return MolecularSystem(
        R=_R,
        Z=np.array([17, 7, 6]),
        box=box,
        mol_id=np.array([0, 1, 0], dtype=np.int32),
    )


def test_bias_vanishes_at_the_target_and_grows_away_from_it():
    import jax.numpy as jnp

    term = ReactionCoordinateBiasTerm(cv=_CV, target=-1.2, k_ev_per_A2=10.0)
    fns = term.make(_system(), EnergyContext())
    energy = float(fns.jax_energy_fn(jnp.asarray(_R)))
    assert energy == pytest.approx(0.0, abs=1e-9)

    off = ReactionCoordinateBiasTerm(cv=_CV, target=-0.2, k_ev_per_A2=10.0)
    e_off = float(off.make(_system(), EnergyContext()).jax_energy_fn(jnp.asarray(_R)))
    assert e_off == pytest.approx(0.5 * 10.0 * 1.0**2)


def test_it_can_be_built_from_pairs_and_coefficients():
    import jax.numpy as jnp

    term = ReactionCoordinateBiasTerm(
        pairs=[[2, 0], [2, 1]], coefficients=[1.0, -1.0], target=-1.2, k_ev_per_A2=4.0
    )
    assert term.cv == _CV
    e = float(term.make(_system(), EnergyContext()).jax_energy_fn(jnp.asarray(_R)))
    assert e == pytest.approx(0.0, abs=1e-9)


def test_default_target_is_the_starting_cv_value():
    """An unspecified target must restrain at the built geometry, not at zero."""
    import jax.numpy as jnp

    fns = ReactionCoordinateBiasTerm(cv=_CV, k_ev_per_A2=10.0).make(
        _system(), EnergyContext()
    )
    assert fns.jax_energy_fn.target == pytest.approx(-1.2)
    assert float(fns.jax_energy_fn(jnp.asarray(_R))) == pytest.approx(0.0, abs=1e-9)


def test_lambda_t_moves_the_window_center():
    import jax.numpy as jnp

    fns = ReactionCoordinateBiasTerm(cv=_CV, target=-1.2, k_ev_per_A2=10.0).make(
        _system(), EnergyContext()
    )
    moved = float(fns.jax_energy_fn(jnp.asarray(_R), lambda_t=-0.2))
    assert moved == pytest.approx(0.5 * 10.0 * 1.0**2)


def test_forces_match_the_analytic_cv_gradient():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    target, k = -0.5, 8.0
    fns = ReactionCoordinateBiasTerm(cv=_CV, target=target, k_ev_per_A2=k).make(
        _system(), EnergyContext()
    )
    autodiff = -np.asarray(jax.grad(fns.jax_energy_fn)(jnp.asarray(_R)))

    value = _CV.value_numpy(_R)
    grad = np.asarray(_CV.gradient_batched(jnp.asarray(_R), 3, 1))[0]
    expected = -k * (value - target) * grad
    np.testing.assert_allclose(autodiff, expected, rtol=1e-9, atol=1e-9)


def test_bias_exerts_no_net_force():
    import jax
    import jax.numpy as jnp

    fns = ReactionCoordinateBiasTerm(cv=_CV, target=0.5, k_ev_per_A2=10.0).make(
        _system(), EnergyContext()
    )
    forces = -np.asarray(jax.grad(fns.jax_energy_fn)(jnp.asarray(_R)))
    np.testing.assert_allclose(forces.sum(axis=0), np.zeros(3), atol=1e-9)


def test_periodic_solute_uses_the_minimum_image():
    """A solute straddling the boundary must not report a box-length distance."""
    import jax.numpy as jnp

    box = np.diag([10.0, 10.0, 10.0])
    wrapped = np.array([[0.5, 0.0, 0.0], [4.0, 0.0, 0.0], [9.5, 0.0, 0.0]])
    system = MolecularSystem(
        R=wrapped,
        Z=np.array([17, 7, 6]),
        box=box,
        mol_id=np.array([0, 1, 0], dtype=np.int32),
    )
    # MIC: r(C-Cl) = |0.5 - 9.5| -> 1.0 ; r(C-N) = |4.0 - 9.5| -> 4.5 ; xi = -3.5
    term = ReactionCoordinateBiasTerm(cv=_CV, target=-3.5, k_ev_per_A2=10.0)
    energy = float(term.make(system, EnergyContext()).jax_energy_fn(jnp.asarray(wrapped)))
    assert energy == pytest.approx(0.0, abs=1e-6)


def test_gas_and_solvated_paths_agree_on_the_same_coordinate():
    """rxncoor must reproduce the packed sampler's bias, or profiles won't compare."""
    import jax.numpy as jnp

    from mmml.umbrella.energy import packed_bias_energies_nd

    target, k = -0.4, 6.505
    fns = ReactionCoordinateBiasTerm(cv=_CV, target=target, k_ev_per_A2=k).make(
        _system(), EnergyContext()
    )
    solvated = float(fns.jax_energy_fn(jnp.asarray(_R)))
    gas = float(
        packed_bias_energies_nd(jnp.asarray(_R), 3, (_CV,), ((target,),), ((k,),))[0]
    )
    assert solvated == pytest.approx(gas, rel=1e-12)


def test_construction_validation():
    with pytest.raises(ValueError, match="cv .* or pairs"):
        ReactionCoordinateBiasTerm()
    with pytest.raises(ValueError, match="non-negative"):
        ReactionCoordinateBiasTerm(cv=_CV, k_ev_per_A2=-1.0)


def test_out_of_range_atom_index_is_caught_at_build_time():
    term = ReactionCoordinateBiasTerm(
        pairs=[[0, 9], [0, 1]], coefficients=[1.0, -1.0], target=0.0
    )
    with pytest.raises(ValueError, match="atom index 9"):
        term.make(_system(), EnergyContext())


def test_term_is_registered():
    import mmml.md.energy.terms  # noqa: F401
    from mmml.md.energy.registry import available_terms, get_term

    assert "rxncoor" in available_terms()
    assert get_term("rxncoor") is ReactionCoordinateBiasTerm


def _sum_cv():
    """r(C-Cl) + r(C-N): the direction the xi bias exerts no force along."""
    return LinearDistanceCV.from_spec(
        {"pairs": [(2, 0), (2, 1)], "coefficients": [1.0, 1.0]}
    )


def test_wall_is_silent_inside_its_band():
    """Flat-bottomed means the sampled ensemble inside the band is untouched.

    If the wall contributed anything where sampling actually happens it would
    bias the profile MBAR reconstructs, which is the whole reason for a
    flat-bottom rather than a harmonic restraint on the sum.
    """
    import jax.numpy as jnp
    from mmml.md.restraints import FlatBottomWall

    # The fixture geometry has sum = 1.8 + 3.0 = 4.8, inside [3.8, 5.8].
    plain = ReactionCoordinateBiasTerm(cv=_CV, target=-1.2, k_ev_per_A2=10.0)
    walled = ReactionCoordinateBiasTerm(
        cv=_CV, target=-1.2, k_ev_per_A2=10.0,
        walls=[FlatBottomWall(cv=_sum_cv(), lower=3.8, upper=5.8, k=10.0)],
    )
    ctx = EnergyContext()
    R = jnp.asarray(_R)
    a = float(plain.make(_system(), ctx).jax_energy_fn(R))
    b = float(walled.make(_system(), ctx).jax_energy_fn(R))
    assert b == pytest.approx(a, abs=1e-12)


def test_wall_penalises_the_degenerate_branch_the_bias_cannot_see():
    """Same xi, dissociated geometry: the bias is blind, the wall is not.

    Moving both partners outward by the same amount leaves xi exactly unchanged,
    so the umbrella reports itself perfectly satisfied while the methyl has left
    both of them. This is the configuration that drove the solvated runs off the
    training manifold.
    """
    import jax.numpy as jnp
    from mmml.md.restraints import FlatBottomWall

    stretched = _R.copy()
    stretched[0, 0] -= 1.0   # Cl further from C
    stretched[1, 0] += 1.0   # N further from C
    xi_before = float(_CV.value_numpy(_R))
    xi_after = float(_CV.value_numpy(stretched))
    assert xi_after == pytest.approx(xi_before, abs=1e-12), "xi must be unchanged"

    ctx = EnergyContext()
    plain = ReactionCoordinateBiasTerm(cv=_CV, target=xi_before, k_ev_per_A2=10.0)
    walled = ReactionCoordinateBiasTerm(
        cv=_CV, target=xi_before, k_ev_per_A2=10.0,
        walls=[FlatBottomWall(cv=_sum_cv(), lower=3.8, upper=5.8, k=10.0)],
    )
    R = jnp.asarray(stretched)
    # The bias alone sees nothing wrong with the dissociated geometry.
    assert float(plain.make(_system(), ctx).jax_energy_fn(R)) == pytest.approx(0.0, abs=1e-12)
    # sum is now 2.8 + 4.0 = 6.8, which is 1.0 A past the wall.
    expected = 0.5 * 10.0 * (6.8 - 5.8) ** 2
    assert float(walled.make(_system(), ctx).jax_energy_fn(R)) == pytest.approx(
        expected, rel=1e-9
    )


def test_bond_retention_wall_catches_what_the_sum_wall_cannot():
    """The methyl detaching from both partners, at unchanged xi.

    A bound on the sum has to depend on xi -- large where one bond is long,
    small at the transition state -- so one global value cannot cover the range.
    min(r1, r2) has no such dependence: it is the statement that the group stays
    bonded to something, and it holds at every xi in the training set.
    """
    import jax.numpy as jnp
    from mmml.md.restraints import BondRetentionWall

    ctx = EnergyContext()
    wall = BondRetentionWall(pairs=((2, 0), (2, 1)), r_max=2.35, k=10.0)
    term = ReactionCoordinateBiasTerm(
        cv=_CV, target=-1.2, k_ev_per_A2=10.0, walls=[wall]
    )
    fn = term.make(_system(), ctx).jax_energy_fn

    # Fixture geometry: r(C-Cl) = 1.8 -- bonded, so the wall is silent.
    assert float(fn(jnp.asarray(_R))) == pytest.approx(0.0, abs=1e-12)

    # Both partners pushed out by 1 A. xi is unchanged, so the bias still reads
    # zero, but now min(r) = 2.8 A and the methyl is bonded to neither.
    detached = _R.copy()
    detached[0, 0] -= 1.0
    detached[1, 0] += 1.0
    assert float(_CV.value_numpy(detached)) == pytest.approx(
        float(_CV.value_numpy(_R)), abs=1e-12
    )
    expected = 0.5 * 10.0 * (2.8 - 2.35) ** 2
    assert float(fn(jnp.asarray(detached))) == pytest.approx(expected, rel=1e-9)


def test_bond_retention_wall_rejects_a_single_pair():
    from mmml.md.restraints import BondRetentionWall

    with pytest.raises(ValueError, match="at least two competing pairs"):
        BondRetentionWall(pairs=((2, 0),), r_max=2.35)


def test_angle_wall_confines_the_attack_channel():
    """xi is blind to the attack angle; the wall is not.

    Swinging the leaving group round to the side leaves both distances -- and
    therefore xi -- untouched, so the umbrella reports itself satisfied while
    the system has left the backside-attack channel entirely.
    """
    import jax.numpy as jnp
    from mmml.md.restraints import AngleWall

    # Cl(0), N(1), C(2) with C between them: N-C-Cl = 180 deg.
    collinear = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.3], [0.0, 0.0, 1.8]])
    wall = AngleWall(atoms=(1, 2, 0), theta_min_deg=130.0, k=50.0)
    # cos is clamped just inside +-1 so the gradient stays finite at exactly
    # 180 deg, which costs ~0.003 deg of accuracy there. Irrelevant next to a
    # 130 deg bound; the alternative is a NaN force at perfect collinearity.
    assert wall.theta_deg_numpy(collinear) == pytest.approx(180.0, abs=0.01)
    assert float(wall.energy(jnp.asarray(collinear))) == pytest.approx(0.0, abs=1e-12)

    # Swing Cl to 70 deg about C, keeping r(C-Cl) and r(C-N) identical.
    th = np.radians(70.0)
    bent = collinear.copy()
    bent[0] = collinear[2] + 1.8 * np.array([np.sin(th), 0.0, np.cos(th)])
    assert np.linalg.norm(bent[2] - bent[0]) == pytest.approx(
        np.linalg.norm(collinear[2] - collinear[0]), abs=1e-9
    ), "the test must not change r(C-Cl)"
    assert wall.theta_deg_numpy(bent) == pytest.approx(70.0, abs=1e-6)
    deficit = np.radians(130.0 - 70.0)
    assert float(wall.energy(jnp.asarray(bent))) == pytest.approx(
        0.5 * 50.0 * deficit ** 2, rel=1e-9
    )


def test_angle_wall_rejects_bad_atom_specs():
    from mmml.md.restraints import AngleWall

    with pytest.raises(ValueError, match="three distinct atoms"):
        AngleWall(atoms=(1, 2, 2))
    with pytest.raises(ValueError, match="theta_min_deg"):
        AngleWall(atoms=(1, 2, 0), theta_min_deg=200.0)
