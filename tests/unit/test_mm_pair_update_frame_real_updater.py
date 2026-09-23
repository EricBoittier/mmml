"""Frame contract on the real vesin MM pair updater (not a fake).

``build_mm_energy_forces_fn(fractional_coordinates=True)`` is what
``setup_calculator(ensemble="npt")`` builds; md-system's JAX-MD path used to
build it Cartesian even for NpT. ``refresh_mm_pairs`` reads the frame from the
updater, so the same positions must give the same pairs, energy and forces
whichever frame the updater was built for, from Cartesian or fractional input.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    have_vesin,
    mm_pair_updater_expects_fractional,
    refresh_mm_pairs,
)

from tests.unit.test_mm_pair_list_completeness import ON, L, _box, _build

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(not have_vesin(), reason="vesin not installed")

BOX = np.array([L, L, L])


def _pairs(pair_idx, pair_mask) -> set[tuple[int, int]]:
    keep = np.asarray(pair_mask) > 0
    return {(min(a, b), max(a, b)) for a, b in np.asarray(pair_idx)[keep]}


def test_real_updater_records_its_frame():
    R = _box(ON + 0.3)
    _, cart = _build(R)
    _, frac = _build(R, fractional_coordinates=True)
    assert mm_pair_updater_expects_fractional(cart) is False
    assert mm_pair_updater_expects_fractional(frac) is True


@pytest.mark.parametrize("input_frame", ["cartesian", "fractional"])
def test_same_pairs_energy_forces_for_both_updater_frames(input_frame):
    R = _box(ON + 0.3, seed=1)
    R = R - BOX * np.floor(R / BOX)  # NpT integrator state lives in the primary cell
    positions = R if input_frame == "cartesian" else R / BOX
    is_cart = input_frame == "cartesian"
    mm_cart, cart = _build(R)
    mm_frac, frac = _build(R, fractional_coordinates=True)

    pc = refresh_mm_pairs(cart, positions, BOX, positions_are_cartesian=is_cart)
    pf = refresh_mm_pairs(frac, positions, BOX, positions_are_cartesian=is_cart)
    assert _pairs(*pc) == _pairs(*pf)
    assert len(_pairs(*pc)) > 0

    e_c, f_c = mm_cart(jnp.asarray(R), *pc)
    e_f, f_f = mm_frac(jnp.asarray(R), *pf)
    np.testing.assert_allclose(float(e_c), float(e_f), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(f_c), np.asarray(f_f), rtol=1e-9, atol=1e-10)


def test_fractional_state_straight_into_cartesian_updater_gives_wrong_pairs():
    """The #231 failure. With vesin every atom looks within 1 A (all pairs);
    the jax-md cell list returned none (the 0-pair NpT runs). Wrong either way."""
    R = _box(ON + 0.3, seed=2)
    R = R - BOX * np.floor(R / BOX)
    _, cart = _build(R)
    right = _pairs(*refresh_mm_pairs(cart, R, BOX, positions_are_cartesian=True))
    wrong = _pairs(*cart(R / BOX, box=BOX))
    assert wrong != right
