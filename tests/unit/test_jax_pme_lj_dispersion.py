"""Unit tests for jax-pme r^-6 LJ dispersion in long_range_backend."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.long_range_backend import (
    compute_jax_pme_lj_dispersion,
    per_atom_jax_pme_c6_sqrt,
    per_atom_jax_pme_c6_sqrt_for_atoms,
    scale_per_atom_coefficients_by_monomer_lambda,
)
from tests.functionality.long_range._common import have_jax_pme_package

pytestmark = pytest.mark.skipif(
    not have_jax_pme_package(),
    reason="jax-pme not installed",
)


def test_per_atom_c6_sqrt_geometric():
    ep = 0.1
    sig = 3.5
    expected = np.sqrt(2.0 * ep * sig**6)
    out = per_atom_jax_pme_c6_sqrt(
        np.array([-ep, -ep]),
        np.array([sig, sig]),
    )
    np.testing.assert_allclose(out, [expected, expected], rtol=1e-12)


def test_lambda_scales_c6_sqrt_pair_product():
    coef = np.array([2.0, 3.0])
    mid = np.array([0, 1], dtype=np.int32)
    lam = np.array([4.0, 9.0])
    scaled = scale_per_atom_coefficients_by_monomer_lambda(coef, mid, lam)
    np.testing.assert_allclose(scaled, [4.0, 9.0])
    assert float(scaled[0] * scaled[1]) == pytest.approx(float(coef[0] * coef[1] * 6.0))


def test_jax_pme_lj_dispersion_two_atom_dimer():
    ep = 0.1
    sig = 3.5
    r = 5.0
    L = 100.0
    c6_ij = 2.0 * ep * (2.0 * sig) ** 6
    direct = -c6_ij / r**6
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    c6_sqrt = np.array([np.sqrt(c6_ij), np.sqrt(c6_ij)])
    out = compute_jax_pme_lj_dispersion(
        pos,
        c6_sqrt,
        box_length_A=L,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    np.testing.assert_allclose(out.energy_kcalmol, direct, rtol=1e-3)


@pytest.mark.parametrize("method", ["ewald", "pme", "p3m"])
def test_jax_pme_lj_methods_agree_ewald_reference(method: str):
    ep = 0.15
    sig = 3.2
    r = 4.5
    L = 80.0
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    c6_sqrt = per_atom_jax_pme_c6_sqrt_for_atoms(
        np.array([-ep, -ep]),
        np.array([sig, sig]),
    )
    ref = compute_jax_pme_lj_dispersion(
        pos, c6_sqrt, box_length_A=L, method="ewald", sr_cutoff_A=6.0
    )
    test = compute_jax_pme_lj_dispersion(
        pos, c6_sqrt, box_length_A=L, method=method, sr_cutoff_A=6.0
    )
    np.testing.assert_allclose(test.energy_kcalmol, ref.energy_kcalmol, rtol=5e-2)
