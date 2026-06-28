"""Unit tests for :mod:`mmml.interfaces.pycharmmInterface.mm_system_energy`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    charmm_switch_factor,
    excluded_pairs_from_psf_bonds,
    fully_excluded_pairs,
    one_four_pairs_from_bonds,
)


def test_charmm_switch_factor_endpoints() -> None:
    settings = CharmmNbondSettings(cutnb=14.0, ctonnb=10.0, ctofnb=12.0)
    below = float(charmm_switch_factor(jnp.asarray(settings.c2onnb * 0.5), settings))
    at_on = float(charmm_switch_factor(jnp.asarray(settings.c2onnb), settings))
    at_off = float(charmm_switch_factor(jnp.asarray(settings.c2ofnb), settings))
    above = float(charmm_switch_factor(jnp.asarray(settings.c2ofnb * 1.1), settings))
    assert below == pytest.approx(1.0)
    assert at_on == pytest.approx(1.0)
    assert at_off == pytest.approx(0.0, abs=1e-6)
    assert above == pytest.approx(0.0)


def test_fully_excluded_pairs_from_iblo_inb() -> None:
    # Two atoms, atom 1 excludes atom 2 (CHARMM 1-based INB entry).
    iblo = [1, 2]
    inb = [2]
    pairs = fully_excluded_pairs(iblo, inb, natom=2)
    assert pairs == frozenset({(0, 1)})


def test_fully_excluded_pairs_empty_inb() -> None:
    assert fully_excluded_pairs([0, 0, 0], [], natom=3) == frozenset()


def test_excluded_pairs_from_psf_bonds_chain() -> None:
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    pairs = excluded_pairs_from_psf_bonds(bonds)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) in pairs
    assert (1, 3) in pairs
    assert (0, 3) not in pairs


def test_one_four_pairs_from_bonds_chain() -> None:
    # Linear 4-atom chain: 0-1-2-3 => one 1-4 pair (0,3).
    bonds = np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    pairs = one_four_pairs_from_bonds(bonds, natom=4)
    assert (0, 3) in pairs
