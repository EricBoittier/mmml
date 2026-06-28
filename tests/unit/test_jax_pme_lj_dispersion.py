"""Unit tests for jax-pme r^-6 LJ dispersion in long_range_backend."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
    hybrid_jax_pme_lj_dispersion_correction,
    intra_monomer_jax_pme_lj_dispersion,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    compute_jax_pme_lj_dispersion,
    per_atom_jax_pme_c6_sqrt,
    per_atom_jax_pme_c6_sqrt_for_atoms,
    scale_per_atom_coefficients_by_monomer_lambda,
    warmup_jax_pme_hybrid_host,
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


@pytest.mark.parametrize("method", ["ewald", "pme"])
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


def test_intra_monomer_lj_dispersion_sums_per_monomer():
    """Intra r^-6 term is the sum of single-monomer jax-pme evaluations."""
    ep = 0.1
    sig = 3.5
    r = 5.0
    L = 100.0
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    c6_sqrt = per_atom_jax_pme_c6_sqrt_for_atoms(
        np.array([-ep, -ep]),
        np.array([sig, sig]),
    )
    offsets = np.array([0, 1, 2], dtype=np.int64)
    intra = intra_monomer_jax_pme_lj_dispersion(
        pos,
        c6_sqrt,
        offsets,
        box_length_A=L,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    e0 = compute_jax_pme_lj_dispersion(
        pos[0:1], c6_sqrt[0:1], box_length_A=L, method="ewald", sr_cutoff_A=6.0
    )
    e1 = compute_jax_pme_lj_dispersion(
        pos[1:2], c6_sqrt[1:2], box_length_A=L, method="ewald", sr_cutoff_A=6.0
    )
    np.testing.assert_allclose(
        intra.energy_kcalmol,
        e0.energy_kcalmol + e1.energy_kcalmol,
        rtol=1e-10,
    )


def test_hybrid_lj_dispersion_is_full_minus_intra_scaled():
    """MM r^-6 uses scale * (E_pme - E_intra), not the full periodic sum."""
    ep = 0.1
    sig = 3.5
    r = 5.0
    L = 100.0
    pos = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
    c6_sqrt = per_atom_jax_pme_c6_sqrt_for_atoms(
        np.array([-ep, -ep]),
        np.array([sig, sig]),
    )
    offsets = np.array([0, 1, 2], dtype=np.int64)
    full = compute_jax_pme_lj_dispersion(
        pos, c6_sqrt, box_length_A=L, method="ewald", sr_cutoff_A=6.0
    )
    intra = intra_monomer_jax_pme_lj_dispersion(
        pos,
        c6_sqrt,
        offsets,
        box_length_A=L,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    switch_scale = 1.0
    corr = hybrid_jax_pme_lj_dispersion_correction(
        pos,
        c6_sqrt,
        offsets,
        box_length_A=L,
        method="ewald",
        sr_cutoff_A=6.0,
        switch_scale=switch_scale,
    )
    expected_unscaled = full.energy_kcalmol - intra.energy_kcalmol
    np.testing.assert_allclose(
        corr.energy_kcalmol,
        switch_scale * expected_unscaled,
        rtol=1e-10,
    )
    expected_f = full.forces_kcalmol_A - intra.forces_kcalmol_A
    np.testing.assert_allclose(
        corr.forces_kcalmol_A,
        switch_scale * expected_f,
        rtol=1e-8,
    )


def test_zero_c6_skips_lj_jax_pme_calls(monkeypatch):
    """Zero scaled C6 returns a zero dispersion correction without jax-pme."""

    def _raise(*args, **kwargs):
        raise AssertionError("LJ jax-pme should not be called for zero C6")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.compute_jax_pme_lj_dispersion",
        _raise,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.intra_monomer_jax_pme_lj_dispersion",
        _raise,
    )
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    offsets = np.array([0, 1, 2], dtype=np.int64)
    corr = hybrid_jax_pme_lj_dispersion_correction(
        pos,
        np.zeros(2, dtype=np.float64),
        offsets,
        box_length_A=40.0,
        method="ewald",
        sr_cutoff_A=6.0,
        switch_scale=0.5,
    )
    assert corr.energy_kcalmol == pytest.approx(0.0)
    assert corr.switch_scale == pytest.approx(0.5)
    np.testing.assert_allclose(corr.forces_kcalmol_A, 0.0)


def test_hybrid_warmup_counts_unique_intra_shapes(monkeypatch):
    calls: list[tuple[int, int]] = []

    def _fake(positions, coefficients, **kwargs):
        calls.append((int(kwargs["exponent"]), int(np.asarray(positions).shape[0])))
        from mmml.interfaces.pycharmmInterface.long_range_backend import LongRangeInteractionResult

        return LongRangeInteractionResult(
            energy_kcalmol=0.0,
            forces_kcalmol_A=np.zeros_like(np.asarray(positions, dtype=np.float64)),
        )

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.long_range_backend.warmup_jax_pme_power_law_host",
        _fake,
    )
    pos = np.zeros((5, 3), dtype=np.float64)
    charges = np.ones(5, dtype=np.float64)
    c6 = np.ones(5, dtype=np.float64)
    offsets = np.array([0, 2, 5], dtype=np.int64)
    counts = warmup_jax_pme_hybrid_host(
        pos,
        charges,
        offsets,
        box_length_A=30.0,
        method="ewald",
        sr_cutoff_A=6.0,
        c6_sqrt=c6,
    )
    assert counts == {
        "coulomb_full": 1,
        "coulomb_intra": 2,
        "dispersion_full": 1,
        "dispersion_intra": 2,
    }
    assert calls == [(1, 5), (1, 2), (1, 3), (6, 5), (6, 2), (6, 3)]

