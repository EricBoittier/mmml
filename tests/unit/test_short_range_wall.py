"""Short-range inter-monomer wall: a safety net that must not touch the fit.

The hybrid handoff switches the MM LJ wall OFF below 6.5 A and hands close range
to the ML model, which has no repulsive prior outside its data -- a liquid
acetone run reached a 0.28 A atom-atom contact. This wall catches that, and must
be identically zero everywhere the training data lives (closest sampled
inter-monomer contact: 1.971 A).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmml.models.short_range_wall import (
    DEFAULT_WALL_R_ON_A,
    inter_monomer_wall_energy,
    pair_wall_energy,
)

R_DATA_MIN = 1.971  # closest inter-monomer atom contact in out_combined_dedup train


def _f(x):
    return float(np.asarray(x).reshape(-1)[0]) if np.asarray(x).ndim else float(x)


def test_wall_is_exactly_zero_over_all_sampled_distances():
    """The load-bearing property: it cannot perturb the fitted PES."""
    r = jnp.linspace(R_DATA_MIN, 20.0, 400)
    assert np.all(np.asarray(pair_wall_energy(r)) == 0.0)


def test_wall_is_zero_at_and_above_r_on_and_positive_below():
    assert _f(pair_wall_energy(jnp.array([DEFAULT_WALL_R_ON_A]))) == 0.0
    assert _f(pair_wall_energy(jnp.array([DEFAULT_WALL_R_ON_A + 1e-6]))) == 0.0
    assert _f(pair_wall_energy(jnp.array([DEFAULT_WALL_R_ON_A - 0.1]))) > 0.0


def test_wall_is_repulsive_and_monotonic_decreasing_in_r():
    r = jnp.linspace(0.2, DEFAULT_WALL_R_ON_A, 60)
    e = np.asarray(pair_wall_energy(r))
    assert np.all(np.diff(e) <= 1e-9), "wall must never attract"
    assert np.all(e >= 0.0)


def test_wall_actually_stops_the_observed_overlap():
    """0.28 A was the real failure. The wall must be enormous there."""
    assert _f(pair_wall_energy(jnp.array([0.2817]))) > 100.0
    assert _f(pair_wall_energy(jnp.array([1.0]))) > 5.0


def test_wall_diverges_rather_than_saturating():
    """A bare cubic saturates at a finite height an energetic atom tunnels through."""
    e = [_f(pair_wall_energy(jnp.array([r]))) for r in (0.4, 0.2, 0.1, 0.05)]
    assert e == sorted(e), "must keep rising as r -> 0"
    assert e[-1] > 1e3


def test_force_is_continuous_across_the_onset():
    """C2 at r_on: no force discontinuity where it switches on."""
    g = jax.grad(lambda r: pair_wall_energy(r).sum())
    below = _f(g(jnp.array([DEFAULT_WALL_R_ON_A - 1e-4])))
    above = _f(g(jnp.array([DEFAULT_WALL_R_ON_A + 1e-4])))
    assert above == 0.0
    assert abs(below) < 1e-3, f"force jumps to {below} at the onset"


# ---------------------------------------------------------------- structures --

MID = jnp.array([0, 0, 1, 1, -1])


def _pos(sep):
    return jnp.array([[0.0, 0, 0], [1.0, 0, 0], [sep, 0, 0], [sep + 1.0, 0, 0], [0, 0, 0]])


def test_no_wall_between_well_separated_monomers():
    assert _f(inter_monomer_wall_energy(_pos(8.0), MID)) == 0.0


def test_wall_fires_when_monomers_interpenetrate():
    assert _f(inter_monomer_wall_energy(_pos(1.0), MID)) > 0.0


def test_intramonomer_bonds_are_never_walled():
    """Bonded atoms live at 1.0-1.5 A, far inside r_on -- not the wall's business."""
    # atoms 0-1 are 1.0 A apart within monomer 0; 2-3 likewise within monomer 1
    assert _f(inter_monomer_wall_energy(_pos(50.0), MID)) == 0.0


def test_padding_is_excluded():
    """Padding sits at the origin, coincident with real atoms of monomer 0."""
    pos = _pos(50.0)  # padding row (index 4) is at [0,0,0] == atom 0's position
    assert _f(inter_monomer_wall_energy(pos, MID)) == 0.0


def test_monomer_only_structure_has_no_wall():
    mono = jnp.array([0, 0, 0, 0, -1])
    assert _f(inter_monomer_wall_energy(_pos(1.0), mono)) == 0.0


def test_forces_are_finite_with_coincident_padding():
    """0 * NaN = NaN: masked pairs must leave the singularity before the sqrt."""
    g = jax.grad(lambda p: inter_monomer_wall_energy(p, MID))(_pos(1.0))
    assert np.isfinite(np.asarray(g)).all()


def test_wall_pushes_monomers_apart():
    """Sign check: the force must separate them, not pull them together."""
    g = np.asarray(jax.grad(lambda p: inter_monomer_wall_energy(p, MID))(_pos(1.2)))
    # dE/dx on atom 1 (right edge of monomer A) must be positive -> force -dE/dx
    # points in -x, away from monomer B, which lies at +x.
    assert g[1, 0] > 0.0
    assert g[2, 0] < 0.0
