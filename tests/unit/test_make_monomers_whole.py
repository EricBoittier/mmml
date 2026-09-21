"""make_monomers_whole: split molecules are rejoined, forces pass through, the cell stays differentiable."""

import jax
import jax.numpy as jnp
import numpy as np

from mmml.interfaces.pycharmmInterface.mmml_calculator import make_monomers_whole

L = 10.0
CELL = jnp.eye(3) * L
# Two 3-atom molecules; molecule 0 straddles the x face, one atom wrapped to the far side.
WHOLE = jnp.array([[9.6, 5.0, 5.0], [10.4, 5.0, 5.0], [10.0, 5.8, 5.0], [3.0, 3.0, 3.0], [3.9, 3.0, 3.0], [3.0, 3.9, 3.0]])
ANCHOR = np.array([0, 0, 0, 3, 3, 3], dtype=np.int32)


def _atom_wrapped(pos):
    return jnp.mod(pos, L)


def test_rejoins_split_molecule_to_same_geometry() -> None:
    split = _atom_wrapped(WHOLE)
    assert float(jnp.abs(split[1, 0] - split[0, 0])) > L / 2  # really split
    rejoined = make_monomers_whole(split, CELL, ANCHOR)
    np.testing.assert_allclose(np.asarray(rejoined - rejoined[ANCHOR]), np.asarray(WHOLE - WHOLE[ANCHOR]), atol=1e-12)


def _toy_energy(pos, cell):
    """In-molecule Cartesian energy (like an ML monomer term): sum of squared bond vectors to the anchor."""
    p = make_monomers_whole(pos, cell, ANCHOR)
    return jnp.sum((p - p[ANCHOR]) ** 2)


def test_energy_and_forces_invariant_to_atom_wrapping() -> None:
    split = _atom_wrapped(WHOLE)
    e_w, f_w = jax.value_and_grad(_toy_energy)(WHOLE, CELL)
    e_s, f_s = jax.value_and_grad(_toy_energy)(split, CELL)
    np.testing.assert_allclose(float(e_s), float(e_w), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(f_s), np.asarray(f_w), atol=1e-12)


def test_strain_derivative_matches_finite_difference_on_split_input() -> None:
    split = _atom_wrapped(WHOLE)
    g_cell = jax.grad(_toy_energy, argnums=1)(split, CELL)
    g_pos = jax.grad(_toy_energy)(split, CELL)
    w_autodiff = float(jnp.sum(split * g_pos) + jnp.trace(g_cell @ CELL.T))  # dE/deps
    h = 1e-6
    w_fd = (float(_toy_energy(split * (1 + h), CELL * (1 + h))) - float(_toy_energy(split * (1 - h), CELL * (1 - h)))) / (2 * h)
    np.testing.assert_allclose(w_autodiff, w_fd, rtol=1e-6)


def test_evaluate_hybrid_ev_invariant_to_atom_wrapping() -> None:
    """The host path (COM rewrap, pair lists, candidates) also sees rejoined molecules."""
    from types import SimpleNamespace

    from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import DecomposedMlpotCalculator

    def raw_geometry_energy(**kw):
        # In-molecule geometry straight from the positions the forward receives (no rejoin here).
        p = kw["positions"].reshape(2, 3, 3)
        return SimpleNamespace(energy=jnp.sum((p - p[:, :1]) ** 2), forces=jnp.zeros_like(kw["positions"]))

    calc = DecomposedMlpotCalculator(raw_geometry_energy, CutoffParameters(), 2, np.ones(6, dtype=int), cell=L, do_mm=False)
    calc._parent_model = SimpleNamespace(_forward_cache_key=None, _spherical_forward_fn=None)
    centred = WHOLE - L / 2  # CHARMM frame [-L/2, L/2]
    e_whole, _, _ = calc.evaluate_hybrid_ev(np.asarray(centred), L)
    e_split, _, _ = calc.evaluate_hybrid_ev(np.asarray(jnp.mod(centred + L / 2, L) - L / 2), L)
    assert abs(e_split - e_whole) < 1e-9
