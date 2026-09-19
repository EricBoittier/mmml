"""Partial-ML (ml_resnames) slice handling in DecomposedMlpotCalculator."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("jax")

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
    DecomposedMlpotCalculator,
    DecomposedMlpotModel,
)


def test_resolve_ml_callback_slice_all_ml_default():
    z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        atoms_per_monomer=[4, 5],
        do_mm=False,
    )
    idx = calc._resolve_ml_callback_slice(9)
    np.testing.assert_array_equal(idx, np.arange(9))


def test_resolve_ml_callback_slice_partial_requires_indices():
    z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        atoms_per_monomer=[4, 5],
        do_mm=False,
    )
    with pytest.raises(RuntimeError, match="ml_atom_indices"):
        calc._resolve_ml_callback_slice(45)


def test_resolve_ml_callback_slice_partial_ok():
    z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=int)
    ml_idx = np.arange(9, dtype=int)
    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        atoms_per_monomer=[4, 5],
        do_mm=False,
        ml_atom_indices=ml_idx,
    )
    np.testing.assert_array_equal(calc._resolve_ml_callback_slice(45), ml_idx)


def test_get_pycharmm_calculator_stores_ml_atom_indices():
    z = np.zeros(9, dtype=int)
    model = DecomposedMlpotModel(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        atoms_per_monomer=[4, 5],
    )
    calc = model.get_pycharmm_calculator(ml_atom_indices=list(range(9)))
    np.testing.assert_array_equal(calc._ml_atom_indices, np.arange(9))
    np.testing.assert_array_equal(model._ml_atom_indices, np.arange(9))


def test_calculate_charmm_partial_ml_scatters_forces(monkeypatch):
    """Full-system CHARMM coords + 9-atom model → evaluate ML slice only."""
    z = np.array([7, 1, 1, 1, 6, 17, 1, 1, 1], dtype=int)
    n_full = 20
    ml_idx = np.arange(9, dtype=int)

    forces_ml = np.arange(27, dtype=np.float64).reshape(9, 3) * 0.01  # eV/Å

    def _fake_forward(*_a, **_k):
        import jax.numpy as jnp

        return jnp.asarray(1.5), jnp.asarray(forces_ml)

    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        2,
        z,
        atoms_per_monomer=[4, 5],
        do_mm=False,
        ml_atom_indices=ml_idx,
        cell=30.0,
    )
    calc._get_spherical_forward_fn = MagicMock(return_value=_fake_forward)
    calc._resolve_mm_pairs = MagicMock(return_value=(None, None, False))
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mlpot_runs_on_this_rank",
        lambda: True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.broadcast_mlpot_result",
        lambda forces, e, n: (forces, e),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        lambda: (0, 1),
    )

    xyz = np.zeros((n_full, 3), dtype=np.float64)
    xyz[ml_idx] = np.arange(27, dtype=np.float64).reshape(9, 3) * 0.1
    x = xyz[:, 0].tolist()
    y = xyz[:, 1].tolist()
    zc = xyz[:, 2].tolist()
    dx = [0.0] * n_full
    dy = [0.0] * n_full
    dz = [0.0] * n_full

    e = calc.calculate_charmm(
        n_full,
        0,
        n_full,
        [0] * n_full,
        x,
        y,
        zc,
        dx,
        dy,
        dz,
        0,
        0,
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    assert e == pytest.approx(1.5 * calc.ev2kcal)
    # Forces only on ML atoms (CHARMM convention: dx -= F)
    for i in range(n_full):
        if i < 9:
            assert dx[i] == pytest.approx(-forces_ml[i, 0] * calc.ev2kcal)
        else:
            assert dx[i] == 0.0
    assert calc.last_ml_forces is not None
    assert calc.last_ml_forces.shape == (n_full, 3)
    np.testing.assert_allclose(calc.last_ml_forces[9:], 0.0)


def test_calculate_charmm_ctypes_accumulates_into_resident_grad(monkeypatch):
    """CHARMM ctypes dx/dy/dz already hold MM contribs; callback must subtract F."""
    import ctypes

    z = np.array([8, 1, 1], dtype=int)
    n = 3
    forces_ev = np.array(
        [[0.10, -0.20, 0.30], [0.40, 0.50, -0.60], [-0.70, 0.80, 0.90]],
        dtype=np.float64,
    )
    resident = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )

    def _fake_forward(*_a, **_k):
        import jax.numpy as jnp

        return jnp.asarray(-3.25), jnp.asarray(forces_ev)

    calc = DecomposedMlpotCalculator(
        MagicMock(),
        CutoffParameters(),
        1,
        z,
        atoms_per_monomer=[3],
        do_mm=False,
    )
    calc._get_spherical_forward_fn = MagicMock(return_value=_fake_forward)
    calc._resolve_mm_pairs = MagicMock(return_value=(None, None, False))
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mlpot_runs_on_this_rank",
        lambda: True,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.broadcast_mlpot_result",
        lambda forces, e, n_atoms: (forces, e),
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_bridge.mpi_rank_size",
        lambda: (0, 1),
    )

    x = (ctypes.c_double * n)(0.0, 1.0, 2.0)
    y = (ctypes.c_double * n)(0.1, 1.1, 2.1)
    zc = (ctypes.c_double * n)(0.2, 1.2, 2.2)
    dx = (ctypes.c_double * n)(*resident[:, 0])
    dy = (ctypes.c_double * n)(*resident[:, 1])
    dz = (ctypes.c_double * n)(*resident[:, 2])

    e = calc.calculate_charmm(
        n, 0, n, [0] * n, x, y, zc, dx, dy, dz, 0, 0, [], [], [], [], [], [], []
    )
    assert e == pytest.approx(-3.25 * calc.ev2kcal)
    scale = float(calc.ev2kcal)
    for i in range(n):
        assert dx[i] == pytest.approx(resident[i, 0] - forces_ev[i, 0] * scale)
        assert dy[i] == pytest.approx(resident[i, 1] - forces_ev[i, 1] * scale)
        assert dz[i] == pytest.approx(resident[i, 2] - forces_ev[i, 2] * scale)
        # A memmove of -F would have dropped the resident 1..9 values.
        assert abs(dx[i] - (-forces_ev[i, 0] * scale)) > 0.5
