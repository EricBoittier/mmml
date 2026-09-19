"""Vectorized CHARMM callback copies (profile hotspots #5 and #9)."""

from __future__ import annotations

import ctypes

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.callback_buffers import (
    stack_charmm_xyz,
    subtract_forces_from_charmm_grad,
)
from mmml.utils.geometry_checks import wrap_monomers_primary_cell


def test_stack_charmm_xyz_matches_list_transpose() -> None:
    n = 12
    x = (ctypes.c_double * n)(*range(n))
    y = (ctypes.c_double * n)(*[10 + i for i in range(n)])
    z = (ctypes.c_double * n)(*[20 + i for i in range(n)])
    got = stack_charmm_xyz(x, y, z, n)
    ref = np.array([x[:n], y[:n], z[:n]], dtype=np.float64).T
    np.testing.assert_array_equal(got, ref)


def test_subtract_forces_from_charmm_grad_matches_python_loop() -> None:
    n = 8
    existing = np.linspace(0.25, 2.0, n)
    dx = (ctypes.c_double * n)(*existing)
    dy = (ctypes.c_double * n)(*(-existing))
    dz = (ctypes.c_double * n)(*(2.0 * existing))
    dx_ref = (ctypes.c_double * n)(*existing)
    dy_ref = (ctypes.c_double * n)(*(-existing))
    dz_ref = (ctypes.c_double * n)(*(2.0 * existing))
    forces = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    subtract_forces_from_charmm_grad(dx, dy, dz, forces, n)
    for i in range(n):
        dx_ref[i] -= forces[i, 0]
        dy_ref[i] -= forces[i, 1]
        dz_ref[i] -= forces[i, 2]
    np.testing.assert_allclose([dx[i] for i in range(n)], [dx_ref[i] for i in range(n)])
    np.testing.assert_allclose([dy[i] for i in range(n)], [dy_ref[i] for i in range(n)])
    np.testing.assert_allclose([dz[i] for i in range(n)], [dz_ref[i] for i in range(n)])


def test_subtract_forces_from_python_lists() -> None:
    """Unit tests and some hosts pass lists; as_array must not silently no-op."""
    n = 3
    dx = [10.0, 20.0, 30.0]
    dy = [1.0, 2.0, 3.0]
    dz = [-1.0, -2.0, -3.0]
    forces = np.ones((n, 3), dtype=np.float64)
    subtract_forces_from_charmm_grad(dx, dy, dz, forces, n)
    assert dx == [9.0, 19.0, 29.0]
    assert dy == [0.0, 1.0, 2.0]
    assert dz == [-2.0, -3.0, -4.0]


def test_subtract_forces_does_not_overwrite_existing_grad() -> None:
    """A memmove of ``-F`` would drop CHARMM's already-resident contributions."""
    n = 4
    dx = (ctypes.c_double * n)(*[10.0, 20.0, 30.0, 40.0])
    dy = (ctypes.c_double * n)(*[1.0, 2.0, 3.0, 4.0])
    dz = (ctypes.c_double * n)(*[-1.0, -2.0, -3.0, -4.0])
    forces = np.ones((n, 3), dtype=np.float64)
    subtract_forces_from_charmm_grad(dx, dy, dz, forces, n)
    np.testing.assert_allclose([dx[i] for i in range(n)], [9.0, 19.0, 29.0, 39.0])
    np.testing.assert_allclose([dy[i] for i in range(n)], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose([dz[i] for i in range(n)], [-2.0, -3.0, -4.0, -5.0])


def test_wrap_monomers_primary_cell_matches_per_molecule_loop() -> None:
    rng = np.random.default_rng(1)
    n_mol, apm = 20, 9
    L = 26.0
    pos = rng.uniform(-L, 2 * L, (n_mol * apm, 3))
    offs = np.arange(0, n_mol * apm + 1, apm)
    got = wrap_monomers_primary_cell(pos, offs, np.diag([L] * 3))
    ref = pos.copy()
    for mi in range(n_mol):
        s, e = int(offs[mi]), int(offs[mi + 1])
        com = ref[s:e].mean(axis=0)
        shift = -np.floor(com / L) * L
        ref[s:e] += shift
    np.testing.assert_allclose(got, ref)


def test_wrap_monomers_primary_cell_is_noop_when_com_in_box() -> None:
    n_mol, apm, L = 4, 3, 26.0
    pos = np.zeros((n_mol * apm, 3))
    for mi in range(n_mol):
        pos[mi * apm : (mi + 1) * apm] = (mi + 0.5, 1.0, 2.0)
    offs = np.arange(0, n_mol * apm + 1, apm)
    got = wrap_monomers_primary_cell(pos, offs, np.diag([L] * 3))
    np.testing.assert_allclose(got, pos)
