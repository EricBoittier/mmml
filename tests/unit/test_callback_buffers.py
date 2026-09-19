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
    dx = (ctypes.c_double * n)(*[0.0] * n)
    dy = (ctypes.c_double * n)(*[0.0] * n)
    dz = (ctypes.c_double * n)(*[0.0] * n)
    dx_ref = (ctypes.c_double * n)(*[0.0] * n)
    dy_ref = (ctypes.c_double * n)(*[0.0] * n)
    dz_ref = (ctypes.c_double * n)(*[0.0] * n)
    forces = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    subtract_forces_from_charmm_grad(dx, dy, dz, forces, n)
    for i in range(n):
        dx_ref[i] -= forces[i, 0]
        dy_ref[i] -= forces[i, 1]
        dz_ref[i] -= forces[i, 2]
    np.testing.assert_allclose([dx[i] for i in range(n)], [dx_ref[i] for i in range(n)])
    np.testing.assert_allclose([dy[i] for i in range(n)], [dy_ref[i] for i in range(n)])
    np.testing.assert_allclose([dz[i] for i in range(n)], [dz_ref[i] for i in range(n)])


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
