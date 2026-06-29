"""Unit tests for hybrid jax-pme Coulomb intra-monomer subtraction."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
    _mm_switch_scales,
    hybrid_jax_pme_coulomb_correction,
    hybrid_jax_pme_mm_lr_correction,
    intra_monomer_jax_pme_coulomb,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import compute_jax_pme_coulomb
from tests.functionality.long_range._common import (
    have_jax_pme_package,
    ion_dimer_system,
)

pytestmark = pytest.mark.skipif(
    not have_jax_pme_package(),
    reason="jax-pme not installed",
)


def test_intra_monomer_jax_pme_sums_per_monomer():
    """Intra term is the sum of single-monomer jax-pme evaluations."""
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    intra = intra_monomer_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    e0 = compute_jax_pme_coulomb(
        system.positions_A[0:1],
        system.charges_e[0:1],
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    e1 = compute_jax_pme_coulomb(
        system.positions_A[1:2],
        system.charges_e[1:2],
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    np.testing.assert_allclose(
        intra.energy_kcalmol,
        e0.energy_kcalmol + e1.energy_kcalmol,
        rtol=1e-10,
    )


def test_hybrid_correction_is_full_minus_intra_scaled():
    """MM Coulomb uses scale * (E_pme - E_intra), not the full periodic sum."""
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    cell = np.eye(3) * system.box_length_A

    full = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    intra = intra_monomer_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    corr = hybrid_jax_pme_coulomb_correction(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        pbc_cell=cell,
        mm_switch_on=6.0,
        mm_switch_width=2.0,
        ml_switch_width=1.0,
        complementary_handoff=True,
    )
    expected_unscaled = full.energy_kcalmol - intra.energy_kcalmol
    np.testing.assert_allclose(
        corr.energy_kcalmol,
        corr.switch_scale * expected_unscaled,
        rtol=1e-10,
    )
    expected_f = full.forces_kcalmol_A - intra.forces_kcalmol_A
    np.testing.assert_allclose(
        corr.forces_kcalmol_A,
        corr.switch_scale * expected_f,
        rtol=1e-8,
    )


def test_intra_is_nonzero_for_periodic_single_ions():
    """Each monomer in PBC carries a non-zero self/image jax-pme energy."""
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    intra = intra_monomer_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    assert abs(intra.energy_kcalmol) > 1e-6


def test_switch_scale_vanishes_for_distant_monomers():
    """COM switching zeros MM Coulomb when monomers are outside the MM window."""
    system = ion_dimer_system(separation_A=20.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    cell = np.eye(3) * system.box_length_A
    scales = _mm_switch_scales(
        system.positions_A,
        offsets,
        ml_switch_width=1.0,
        mm_switch_on=12.0,
        mm_switch_width=1.0,
        complementary_handoff=True,
        pbc_cell=cell,
        mm_r_min=None,
    )
    assert float(np.max(scales)) < 1e-6
    corr = hybrid_jax_pme_coulomb_correction(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        pbc_cell=cell,
        mm_switch_on=12.0,
        mm_switch_width=1.0,
        ml_switch_width=1.0,
        complementary_handoff=True,
    )
    assert abs(corr.energy_kcalmol) < 1e-3


def test_full_box_differs_from_cross_monomer_only():
    """Full-box jax-pme differs from full minus intra (intra term is material)."""
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    full = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    intra = intra_monomer_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
    )
    cross_only = full.energy_kcalmol - intra.energy_kcalmol
    assert cross_only != pytest.approx(full.energy_kcalmol, rel=1e-3)


def test_zero_charges_skip_jax_pme_calls(monkeypatch):
    """Zero scaled charges return zero correction without full or intra solves."""

    def _raise(*args, **kwargs):
        raise AssertionError("jax-pme should not be called for zero charges")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.compute_jax_pme_coulomb",
        _raise,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.intra_monomer_jax_pme_coulomb",
        _raise,
    )
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    corr = hybrid_jax_pme_coulomb_correction(
        system.positions_A,
        np.zeros_like(system.charges_e),
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        pbc_cell=np.eye(3) * system.box_length_A,
    )
    assert corr.energy_kcalmol == pytest.approx(0.0)
    np.testing.assert_allclose(corr.forces_kcalmol_A, 0.0)


def test_zero_coulomb_still_reuses_switch_scale_for_dispersion(monkeypatch):
    """The combined correction can skip Coulomb while retaining dispersion scale."""

    def _raise(*args, **kwargs):
        raise AssertionError("Coulomb jax-pme should not be called for zero charges")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.compute_jax_pme_coulomb",
        _raise,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.intra_monomer_jax_pme_coulomb",
        _raise,
    )
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    out = hybrid_jax_pme_mm_lr_correction(
        system.positions_A,
        np.zeros_like(system.charges_e),
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        c6_sqrt=np.zeros_like(system.charges_e),
        pbc_cell=np.eye(3) * system.box_length_A,
    )
    assert out.energy_kcalmol == pytest.approx(0.0)
    np.testing.assert_allclose(out.forces_kcalmol_A, 0.0)


def test_hybrid_coulomb_only_skips_dispersion(monkeypatch):
    """Coulomb-only mode avoids the r^-6 full and intra jax-pme calls."""

    def _fake_coulomb(*args, **kwargs):
        positions = np.asarray(args[0], dtype=np.float64)
        from mmml.interfaces.pycharmmInterface.long_range_backend import LongRangeInteractionResult

        return LongRangeInteractionResult(
            energy_kcalmol=1.0,
            forces_kcalmol_A=np.ones_like(positions),
        )

    def _raise_dispersion(*args, **kwargs):
        raise AssertionError("dispersion jax-pme should be skipped")

    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.hybrid_jax_pme_coulomb_correction",
        _fake_coulomb,
    )
    monkeypatch.setattr(
        "mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb.hybrid_jax_pme_lj_dispersion_correction",
        _raise_dispersion,
    )
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    out = hybrid_jax_pme_mm_lr_correction(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        c6_sqrt=np.ones_like(system.charges_e),
        pbc_cell=np.eye(3) * system.box_length_A,
        include_dispersion=False,
    )
    assert out.energy_kcalmol == pytest.approx(1.0)
    assert out.dispersion is None
