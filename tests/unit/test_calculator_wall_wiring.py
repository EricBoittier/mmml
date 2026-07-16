"""The MD calculator must apply the SAME short-range wall as training.

Training has the wall; MD did not, and a liquid acetone NVT run collapsed by
~5000 eV (150 -> 705 K) at dt=0.25 fs because the MM r^-12 wall is tapered off
below 6.5 A and the ML model has no repulsive prior outside its data.

These pin the wiring without needing a live CHARMM session.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


def test_calculator_exposes_the_wall_flag_and_defaults_on():
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    sig = inspect.signature(setup_calculator)
    assert "short_range_wall" in sig.parameters
    assert sig.parameters["short_range_wall"].default is True


def test_calculator_imports_the_shared_wall_not_a_copy():
    """Single source of truth: training and MD must evaluate the same function."""
    import mmml.interfaces.pycharmmInterface.mmml_calculator as calc
    from mmml.models.short_range_wall import pair_wall_energy

    assert calc.pair_wall_energy is pair_wall_energy


def test_wall_is_not_gated_on_doMM():
    """The MM taper switching LJ off at close range is the hole the wall fills.

    If the wall sat inside the `if doMM:` block it would vanish exactly where it
    is needed.
    """
    src = inspect.getsource(
        __import__(
            "mmml.interfaces.pycharmmInterface.mmml_calculator", fromlist=["x"]
        ).setup_calculator
    )
    i_wall = src.index("if short_range_wall:")
    # the nearest preceding `if doMM` block must have already closed: the wall
    # line sits at the same indentation as the doMM block itself.
    line = src[src.rindex("\n", 0, i_wall) + 1 : i_wall]
    assert line == "        ", f"wall is nested (indent={len(line)}), expected top-level in the fn"


def test_wall_energy_matches_training_for_a_close_contact():
    """The number the calculator adds must equal training's wall term."""
    import jax.numpy as jnp

    from mmml.models.cgenff_mm import monomer_centroids  # noqa: F401  (import sanity)
    from mmml.models.short_range_wall import (
        inter_monomer_wall_energy,
        pair_wall_energy,
    )

    # two atoms from different monomers at 0.28 A -- the observed overlap
    r = jnp.array([0.2817])
    assert float(pair_wall_energy(r)[0]) > 100.0

    pos = jnp.array([[0.0, 0, 0], [0.2817, 0, 0], [50.0, 0, 0]])
    mol = jnp.array([0, 1, -1])
    e = float(inter_monomer_wall_energy(pos, mol))
    assert e == pytest.approx(float(pair_wall_energy(r)[0]), rel=1e-6)
