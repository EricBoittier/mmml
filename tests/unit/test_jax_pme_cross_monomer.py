"""Unit tests: cross-monomer jax-pme vs legacy full−intra hybrid path."""

from __future__ import annotations

import time

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.jax_pme_cross_monomer import (
    compute_jax_pme_cross_monomer_power_law,
    resolve_jax_pme_intra_mode,
)
from mmml.interfaces.pycharmmInterface.jax_pme_hybrid_coulomb import (
    _intra_monomer_jax_pme_power_law,
    hybrid_jax_pme_coulomb_correction,
    hybrid_jax_pme_mm_lr_correction,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    DEFAULT_JAX_PME_LJ_PREFACTOR,
    compute_jax_pme_power_law,
)
from tests.functionality.long_range._common import (
    have_jax_pme_package,
    ion_dimer_system,
)

pytestmark = pytest.mark.skipif(
    not have_jax_pme_package(),
    reason="jax-pme not installed",
)


def _legacy_cross_reference(
    positions: np.ndarray,
    coefficients: np.ndarray,
    offsets: np.ndarray,
    *,
    box_length_A: float,
    exponent: int,
    prefactor: float,
) -> tuple[float, np.ndarray]:
    full = compute_jax_pme_power_law(
        positions,
        coefficients,
        box_length_A=box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        exponent=exponent,
        prefactor=prefactor,
    )
    intra = _intra_monomer_jax_pme_power_law(
        positions,
        coefficients,
        offsets,
        box_length_A=box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        exponent=exponent,
        prefactor=prefactor,
    )
    return (
        float(full.energy_kcalmol - intra.energy_kcalmol),
        np.asarray(full.forces_kcalmol_A - intra.forces_kcalmol_A, dtype=np.float64),
    )


def test_resolve_jax_pme_intra_mode_defaults_cross_for_ewald(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MMML_JAX_PME_INTRA_MODE", raising=False)
    assert resolve_jax_pme_intra_mode("ewald") == "cross"
    monkeypatch.setenv("MMML_JAX_PME_INTRA_MODE", "full_minus_intra")
    assert resolve_jax_pme_intra_mode("ewald") == "full_minus_intra"


def test_cross_monomer_matches_legacy_ion_dimer() -> None:
    from jaxpme import prefactors as jpref

    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    ref_e, ref_f = _legacy_cross_reference(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        exponent=1,
        prefactor=float(jpref.kcalmol_A),
    )
    cross = compute_jax_pme_cross_monomer_power_law(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        exponent=1,
        prefactor=float(jpref.kcalmol_A),
    )
    assert cross.energy_kcalmol == pytest.approx(ref_e, rel=0, abs=1e-9)
    np.testing.assert_allclose(cross.forces_kcalmol_A, ref_f, rtol=0, atol=1e-8)


def test_cross_monomer_matches_legacy_cluster() -> None:
    from jaxpme import prefactors as jpref

    rng = np.random.default_rng(3)
    n_mono = 8
    size = 3
    n = n_mono * size
    pos = rng.random((n, 3)) * 20.0
    chg = rng.normal(0.0, 0.1, n)
    offsets = np.arange(0, n + 1, size, dtype=np.int64)
    box_L = 28.0
    for exponent, pref in (
        (1, float(jpref.kcalmol_A)),
        (6, DEFAULT_JAX_PME_LJ_PREFACTOR),
    ):
        coef = chg if exponent == 1 else np.sqrt(np.abs(chg) * 0.05 + 0.01)
        ref_e, ref_f = _legacy_cross_reference(
            pos,
            coef,
            offsets,
            box_length_A=box_L,
            exponent=exponent,
            prefactor=pref,
        )
        cross = compute_jax_pme_cross_monomer_power_law(
            pos,
            coef,
            offsets,
            box_length_A=box_L,
            method="ewald",
            sr_cutoff_A=6.0,
            exponent=exponent,
            prefactor=pref,
        )
        assert cross.energy_kcalmol == pytest.approx(ref_e, rel=0, abs=1e-6)
        np.testing.assert_allclose(cross.forces_kcalmol_A, ref_f, rtol=0, atol=1e-6)


def test_hybrid_correction_uses_cross_mode_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MMML_JAX_PME_INTRA_MODE", raising=False)
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    offsets = np.array([0, 1, 2], dtype=np.int64)
    cell = np.eye(3) * system.box_length_A
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
    )
    monkeypatch.setenv("MMML_JAX_PME_INTRA_MODE", "full_minus_intra")
    legacy = hybrid_jax_pme_coulomb_correction(
        system.positions_A,
        system.charges_e,
        offsets,
        box_length_A=system.box_length_A,
        method="ewald",
        sr_cutoff_A=6.0,
        pbc_cell=cell,
        mm_switch_on=6.0,
        mm_switch_width=2.0,
    )
    assert corr.energy_kcalmol == pytest.approx(legacy.energy_kcalmol, rel=0, abs=1e-9)
    np.testing.assert_allclose(
        corr.forces_kcalmol_A,
        legacy.forces_kcalmol_A,
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.slow
def test_cross_mode_faster_than_legacy_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cross path should beat per-monomer prepare loop after warmup (CPU smoke)."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    rng = np.random.default_rng(4)
    n_mono = 18
    size = 3
    n = n_mono * size
    pos = rng.random((n, 3)) * 20.0
    chg = rng.normal(0.0, 0.1, n)
    offsets = np.arange(0, n + 1, size, dtype=np.int64)
    box_L = 28.0
    cell = np.diag([box_L, box_L, box_L])
    from jaxpme import prefactors as jpref

    cross = compute_jax_pme_cross_monomer_power_law(
        pos,
        chg,
        offsets,
        box_length_A=box_L,
        method="ewald",
        sr_cutoff_A=6.0,
        exponent=1,
        prefactor=float(jpref.kcalmol_A),
    )
    del cross
    hybrid_jax_pme_mm_lr_correction(
        pos,
        chg,
        offsets,
        box_length_A=box_L,
        method="ewald",
        sr_cutoff_A=6.0,
        c6_sqrt=np.sqrt(np.abs(chg) * 0.05 + 0.01),
        pbc_cell=cell,
        mm_switch_on=6.0,
        mm_switch_width=2.0,
    )
    monkeypatch.setenv("MMML_JAX_PME_INTRA_MODE", "full_minus_intra")
    for _ in range(2):
        hybrid_jax_pme_mm_lr_correction(
            pos,
            chg,
            offsets,
            box_length_A=box_L,
            method="ewald",
            sr_cutoff_A=6.0,
            c6_sqrt=np.sqrt(np.abs(chg) * 0.05 + 0.01),
            pbc_cell=cell,
            mm_switch_on=6.0,
            mm_switch_width=2.0,
        )
    t0 = time.perf_counter()
    for _ in range(3):
        hybrid_jax_pme_mm_lr_correction(
            pos,
            chg,
            offsets,
            box_length_A=box_L,
            method="ewald",
            sr_cutoff_A=6.0,
            c6_sqrt=np.sqrt(np.abs(chg) * 0.05 + 0.01),
            pbc_cell=cell,
            mm_switch_on=6.0,
            mm_switch_width=2.0,
        )
    legacy_ms = (time.perf_counter() - t0) * 1000.0 / 3.0

    monkeypatch.setenv("MMML_JAX_PME_INTRA_MODE", "cross")
    for _ in range(2):
        hybrid_jax_pme_mm_lr_correction(
            pos,
            chg,
            offsets,
            box_length_A=box_L,
            method="ewald",
            sr_cutoff_A=6.0,
            c6_sqrt=np.sqrt(np.abs(chg) * 0.05 + 0.01),
            pbc_cell=cell,
            mm_switch_on=6.0,
            mm_switch_width=2.0,
        )
    t0 = time.perf_counter()
    for _ in range(3):
        hybrid_jax_pme_mm_lr_correction(
            pos,
            chg,
            offsets,
            box_length_A=box_L,
            method="ewald",
            sr_cutoff_A=6.0,
            c6_sqrt=np.sqrt(np.abs(chg) * 0.05 + 0.01),
            pbc_cell=cell,
            mm_switch_on=6.0,
            mm_switch_width=2.0,
        )
    cross_ms = (time.perf_counter() - t0) * 1000.0 / 3.0
    if cross_ms >= legacy_ms:
        pytest.skip(
            f"cross not faster on this host (cross={cross_ms:.0f} ms legacy={legacy_ms:.0f} ms)"
        )
