"""Unit tests for JAX CMAP gating vs PyCHARMM reference."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_components
from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import charmm_cmap_is_active


def test_charmm_cmap_is_active_false_for_zero():
    assert not charmm_cmap_is_active({"cmap": 0.0, "bond": 1.0})


def test_charmm_cmap_is_active_true_for_nonzero():
    assert charmm_cmap_is_active({"cmap": 0.5})


def test_bonded_energy_components_include_cmap_gate():
    from jax_md.mm_forcefields.base import BondedParameters, Topology

    topology = Topology(
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        torsions=jnp.zeros((0, 4), dtype=jnp.int32),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        cmap_atoms=jnp.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32),
        cmap_map_idx=jnp.asarray([0], dtype=jnp.int32),
    )
    bonded = BondedParameters(
        bond_k=jnp.zeros(0),
        bond_r0=jnp.zeros(0),
        angle_k=jnp.zeros(0),
        angle_theta0=jnp.zeros(0),
        torsion_k=jnp.zeros(0),
        torsion_n=jnp.zeros(0, dtype=jnp.int32),
        torsion_phase=jnp.zeros(0),
        improper_k=jnp.zeros(0),
        improper_n=jnp.zeros(0, dtype=jnp.int32),
        improper_gamma=jnp.zeros(0),
        cmap_maps=jnp.ones((1, 16), dtype=jnp.float64),
    )
    pos = jnp.zeros((8, 3), dtype=jnp.float64)
    with_cmap = bonded_energy_components(
        pos, topology, bonded, include_cmap=True
    )
    without_cmap = bonded_energy_components(
        pos, topology, bonded, include_cmap=False
    )
    assert float(without_cmap["cmap"]) == pytest.approx(0.0)
    assert float(without_cmap["total"]) == pytest.approx(
        float(without_cmap["bond"])
        + float(without_cmap["angle"])
        + float(without_cmap["urey"])
        + float(without_cmap["torsion"])
        + float(without_cmap["improper"])
    )
