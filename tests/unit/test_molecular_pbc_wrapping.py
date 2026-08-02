"""Regression tests for molecular versus atom-wise PBC wrapping.

These tests encode the representation contract used by the production ASE and
trajectory paths: periodic images may translate a molecule, but must never
change its internal Cartesian geometry.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mmml.cli.run.ase_runner import wrap_positions_for_pbc
from mmml.cli.run.shared import _wrap_frame_by_monomer
from mmml.interfaces.pycharmmInterface.cell_list import _wrap_groups_np


BOX = 10.0
CELL = np.eye(3) * BOX
OFFSETS = np.array([0, 3, 6], dtype=np.int32)


def _boundary_waters() -> np.ndarray:
    """Two intact waters whose atoms straddle opposite primary-cell faces."""
    return np.array(
        [
            [9.80, 5.00, 5.00],
            [10.76, 5.00, 5.00],
            [9.56, 5.93, 5.00],
            [-0.20, 2.00, 2.00],
            [0.76, 2.00, 2.00],
            [-0.44, 2.93, 2.00],
        ],
        dtype=np.float64,
    )


def _internal_vectors(positions: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [positions[start + 1 : end] - positions[start] for start, end in zip(OFFSETS[:-1], OFFSETS[1:])]
    )


def test_molecular_wrap_preserves_internal_water_geometry() -> None:
    positions = _boundary_waters()
    wrapped = _wrap_groups_np(positions, CELL, OFFSETS)

    np.testing.assert_allclose(
        _internal_vectors(wrapped), _internal_vectors(positions), atol=1e-12
    )
    for start, end in zip(OFFSETS[:-1], OFFSETS[1:]):
        frac_center = wrapped[start:end].mean(axis=0) / BOX
        assert np.all((frac_center >= 0.0) & (frac_center < 1.0))


def test_atomwise_wrap_is_not_geometry_preserving_for_boundary_water() -> None:
    positions = _boundary_waters()
    atom_wrapped = np.mod(positions, BOX)

    error = np.linalg.norm(
        _internal_vectors(atom_wrapped) - _internal_vectors(positions), axis=1
    )
    assert error.max() >= BOX - 1e-12


def test_ase_production_fallback_uses_molecular_wrap() -> None:
    positions = _boundary_waters()
    hybrid_calc = SimpleNamespace(pbc_map=None, do_pbc_map=False)

    wrapped = wrap_positions_for_pbc(
        positions,
        cell=BOX,
        hybrid_calc=hybrid_calc,
        monomer_offsets=OFFSETS,
    )

    expected = _wrap_groups_np(positions, CELL, OFFSETS)
    np.testing.assert_allclose(wrapped, expected, atol=1e-12)
    np.testing.assert_allclose(
        _internal_vectors(wrapped), _internal_vectors(positions), atol=1e-12
    )


def test_trajectory_output_wrap_uses_same_molecular_contract() -> None:
    positions = _boundary_waters()
    wrapped = _wrap_frame_by_monomer(positions, CELL, OFFSETS)
    expected = _wrap_groups_np(positions, CELL, OFFSETS)

    np.testing.assert_allclose(wrapped, expected, atol=1e-12)
    np.testing.assert_allclose(
        _internal_vectors(wrapped), _internal_vectors(positions), atol=1e-12
    )


def test_jax_pbc_mapper_preserves_internal_geometry_and_is_image_invariant() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from mmml.interfaces.pycharmmInterface.pbc_prep_factory import make_pbc_mapper

    positions = _boundary_waters()
    mol_id = np.repeat(np.arange(2, dtype=np.int32), 3)
    mapper = make_pbc_mapper(jnp.asarray(CELL), jnp.asarray(mol_id))

    mapped = np.asarray(mapper(jnp.asarray(positions)))
    image_shifted = positions.copy()
    image_shifted[:3] += np.array([BOX, -BOX, 2.0 * BOX])
    mapped_image = np.asarray(mapper(jnp.asarray(image_shifted)))

    np.testing.assert_allclose(
        _internal_vectors(mapped), _internal_vectors(positions), atol=2e-6
    )
    np.testing.assert_allclose(mapped_image, mapped, atol=2e-6)
