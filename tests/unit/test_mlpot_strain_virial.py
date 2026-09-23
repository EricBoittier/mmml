"""MLpot strain-virial pieces: box-differentiable dimer lattice shift and the CHARMM correction tensor."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


def test_dimer_lattice_shift_value_and_gradients() -> None:
    from mmml.interfaces.pycharmmInterface.mmml_calculator import _dimer_lattice_shift
    from mmml.interfaces.pycharmmInterface.pbc_utils_jax import mic_displacement

    cell = jnp.eye(3) * 20.0
    com_a = jnp.array([1.0, 2.0, 3.0])
    com_b = jnp.array([18.5, 2.5, -16.0])  # needs a lattice shift in x and z
    shift = _dimer_lattice_shift(com_a, com_b, cell)
    ref = com_a + mic_displacement(com_a, com_b, cell) - com_b
    np.testing.assert_allclose(np.asarray(shift), np.asarray(ref), atol=1e-12)
    np.testing.assert_allclose(np.asarray(shift), [-20.0, 0.0, 20.0], atol=1e-12)
    # zero position gradient (piecewise-constant image count), non-zero cell gradient
    g_pos = jax.grad(lambda b: jnp.sum(_dimer_lattice_shift(com_a, b, cell)))(com_b)
    assert float(jnp.max(jnp.abs(g_pos))) == 0.0
    g_cell = jax.grad(lambda h: _dimer_lattice_shift(com_a, com_b, h)[0])(cell)
    assert float(g_cell[0, 0]) == pytest.approx(-1.0)


def test_virial_correction_tensor_lattice_and_rewrap_terms() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.strain_virial import virial_correction_kcal

    cell = np.eye(3) * 32.0
    G = np.diag([0.5, -1.0, 2.0])  # dE/dcell, eV/A
    F = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    x_charmm = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x_eval = x_charmm + np.array([[32.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # atom 0 rewrapped by +L in x
    c = virial_correction_kcal(dE_dcell_eV=G, cell=cell, forces_eV_A=F, positions_eval=x_eval,
                               positions_charmm=x_charmm, ev_to_kcal=1.0)
    lattice = -(G @ cell.T).T
    rewrap = np.zeros((3, 3)); rewrap[0, 0] = 32.0 * 1.0  # sum_i dx_ia F_ib
    np.testing.assert_allclose(c, lattice + rewrap)
    np.testing.assert_allclose(virial_correction_kcal(dE_dcell_eV=G, cell=cell, forces_eV_A=F, positions_eval=x_charmm,
                                                      positions_charmm=x_charmm, ev_to_kcal=23.06), lattice * 23.06)


def test_strain_virial_scope_only_for_live_barostat(monkeypatch) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot import strain_virial as sv

    monkeypatch.delenv(sv.STRAIN_VIRIAL_ENV, raising=False)
    monkeypatch.setattr(sv, "require_charmm_virial_hook", lambda: None)
    assert not sv.strain_virial_enabled()
    with sv.cpt_strain_virial_scope({"cpt": True, "pmass": 0}) as active:  # constant-volume CPT heat
        assert not active and not sv.strain_virial_enabled()
    with sv.cpt_strain_virial_scope({"cpt": True, "pmass": 523}) as active:
        assert active and sv.strain_virial_enabled()
    assert not sv.strain_virial_enabled()
    monkeypatch.setenv(sv.STRAIN_VIRIAL_ENV, "1")
    assert sv.strain_virial_enabled()
