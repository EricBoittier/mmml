"""Tests for the padded-term capacity/overflow helpers and dtype policy."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.md.energy.capacity import (
    INDEX_DTYPE,
    MASK_DTYPE,
    CapacityOverflow,
    check_capacity,
    pad_indices,
    shell_capacity,
)


def test_shell_capacity_scales_with_cutoff_volume():
    # water number density ~0.0334 molecules / Å³
    rho = 0.0334
    small = shell_capacity(6.0, rho, headroom=1.0)
    big = shell_capacity(12.0, rho, headroom=1.0)
    # doubling the cutoff radius -> ~8x the shell volume
    assert big == pytest.approx(8 * small, rel=0.05)
    # headroom multiplies the estimate
    assert shell_capacity(6.0, rho, headroom=1.5) > small
    # never below the floor
    assert shell_capacity(0.1, rho, minimum=8) == 8


def test_shell_capacity_rejects_bad_inputs():
    with pytest.raises(ValueError):
        shell_capacity(-1.0, 0.03)
    with pytest.raises(ValueError):
        shell_capacity(6.0, 0.0)
    with pytest.raises(ValueError):
        shell_capacity(6.0, 0.03, headroom=0.5)


def test_check_capacity_raises_by_default():
    check_capacity(10, 10, "pairs")  # exactly full is OK
    with pytest.raises(CapacityOverflow):
        check_capacity(11, 10, "pairs")


def test_check_capacity_warn_and_ignore():
    with pytest.warns(UserWarning):
        check_capacity(11, 10, "pairs", on_overflow="warn")
    check_capacity(11, 10, "pairs", on_overflow="ignore")  # no raise, no warn


def test_pad_indices_dtypes_and_mask():
    padded, mask = pad_indices([5, 6, 7], capacity=8)
    assert padded.dtype == INDEX_DTYPE
    assert mask.dtype == MASK_DTYPE
    assert padded.shape == (8,) and mask.shape == (8,)
    assert list(padded[:3]) == [5, 6, 7]
    assert list(mask) == [1, 1, 1, 0, 0, 0, 0, 0]
    # padded slots use the fill value (default atom 0)
    assert list(padded[3:]) == [0, 0, 0, 0, 0]


def test_pad_indices_overflow():
    with pytest.raises(CapacityOverflow):
        pad_indices([1, 2, 3], capacity=2)


def test_int8_mask_matches_float_mask_numerically():
    # the dtype-policy claim: an int8 0/1 mask promotes to f64 on multiply and is
    # numerically identical to a float mask.
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from mmml.md.energy import EnergyContext
    from mmml.md.energy.terms import RepulsiveCoreVdwTerm
    from mmml.md.system import MolecularSystem

    rng = np.random.default_rng(7)
    n_core, n_groups = 3, 4
    n_atoms = n_core + n_groups * 3
    system = MolecularSystem(
        R=rng.uniform(0, 10, size=(n_atoms, 3)),
        Z=np.ones(n_atoms, int),
        box=np.diag([10.0, 10.0, 10.0]),
        mol_id=np.arange(n_atoms),
    )
    groups = np.arange(n_core, n_atoms).reshape(n_groups, 3)
    eps = rng.uniform(0.02, 0.2, size=n_atoms)
    rmin = rng.uniform(0.8, 2.0, size=n_atoms)
    fn = RepulsiveCoreVdwTerm(n_core, groups, eps, rmin, 4.0, 1.5).make(
        system, EnergyContext()
    ).jax_energy_fn

    R = jnp.asarray(system.R)
    slots = jnp.arange(n_groups, dtype=jnp.int32)
    e_int8 = float(fn(R, active_group_slots=slots, active_group_mask=jnp.ones(n_groups, dtype=jnp.int8)))
    e_f64 = float(fn(R, active_group_slots=slots, active_group_mask=jnp.ones(n_groups, dtype=jnp.float64)))
    assert e_int8 == pytest.approx(e_f64, rel=0, abs=0)


# --- pair_capacity ----------------------------------------------------------
#
# The pair estimate used to be `n_atoms * shell_capacity(...)`, which counts
# every unordered pair twice. That factor was not a safety margin -- it was a
# double-count that made `headroom` mean twice what it said. These pin the
# corrected behaviour so it cannot drift back.


def test_pair_capacity_is_half_the_shell_sum_before_headroom():
    """One atom's shell x atoms / 2, because the builders emit j > i only."""
    from mmml.md.energy.capacity import pair_capacity, shell_capacity

    n, cutoff, rho = 4000, 6.0, 0.03
    per_atom = shell_capacity(cutoff, rho, headroom=1.0, minimum=1)
    got = pair_capacity(n, cutoff, rho, headroom=1.0)
    assert got == pytest.approx(n * per_atom / 2, rel=1e-9)


def test_pair_capacity_scales_with_headroom():
    from mmml.md.energy.capacity import pair_capacity

    n, cutoff, rho = 4000, 6.0, 0.03
    one = pair_capacity(n, cutoff, rho, headroom=1.0)
    assert pair_capacity(n, cutoff, rho, headroom=3.0) == pytest.approx(3 * one, rel=1e-6)


def test_pair_capacity_never_exceeds_the_pairs_that_can_exist():
    """The shell estimate assumes an unbounded medium; the box is the truth."""
    from mmml.md.energy.capacity import pair_capacity

    # Tiny box, huge cutoff: the sphere estimate is wildly impossible.
    got = pair_capacity(50, 100.0, 0.1, headroom=3.0)
    assert got == 50 * 49 // 2


def test_pair_capacity_discounts_intramolecular_pairs():
    from mmml.md.energy.capacity import pair_capacity

    sizes = np.full(10, 3)  # 10 molecules of 3 atoms
    got = pair_capacity(30, 100.0, 0.1, mol_sizes=sizes, headroom=3.0)
    assert got == 30 * 29 // 2 - 10 * 3


def test_pair_capacity_rejects_headroom_below_one():
    from mmml.md.energy.capacity import pair_capacity

    with pytest.raises(ValueError, match="headroom"):
        pair_capacity(100, 6.0, 0.03, headroom=0.5)


def test_pair_capacity_default_headroom_covers_a_dense_excursion():
    """PAIR_HEADROOM is sized against a 3.6x density spike, not equilibrium.

    Measured worst requirement over 300-10 800 atoms was 2.50x the mean-field
    estimate; the default must stay above that with room to spare.
    """
    from mmml.md.energy.capacity import PAIR_HEADROOM, pair_capacity

    assert PAIR_HEADROOM >= 2.75, "below the measured worst case (2.50x) plus margin"

    n, cutoff, rho = 4000, 12.0, 0.1
    mean_field = pair_capacity(n, cutoff, rho, headroom=1.0)
    default = pair_capacity(n, cutoff, rho)
    assert default >= 2.5 * mean_field
