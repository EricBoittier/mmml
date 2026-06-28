"""Pytest suite for long-range Coulomb backends (MIC, jax-pme, ScaFaCoS).

Patterns follow jax-pme ``tests/test_ewald.py`` (Madelung crystals, method
cross-checks, analytic dimers).  ScaFaCoS tests skip when ``libfcs`` is absent.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.functionality.long_range._common import (
    DEFAULT_MM_COULOMB_CUTOFF_A,
    cscl_crystal,
    have_jax_pme_package,
    have_scafacos_library,
    ion_dimer_system,
    jax_pme_coulomb_energy_forces,
    mic_coulomb_energy_forces,
    madelung_constant,
    nacl_cubic,
    random_neutral_cluster,
    scafacos_coulomb_energy_forces,
    scafacos_integration_enabled,
)
from mmml.interfaces.pycharmmInterface.long_range_backend import (
    CHARMM_COULOMB_KCAL,
    have_jax_pme,
    pick_lr_solver,
)

jax_pme = pytest.importorskip("jax")  # ensure jax present for MIC tests


class TestBackendSelection:
    def test_have_jax_pme_detects_jaxpme(self):
        if have_jax_pme_package():
            assert have_jax_pme() is True

    def test_pick_lr_solver_mic_when_no_externals(self, monkeypatch):
        monkeypatch.delenv("MMML_LR_SOLVER", raising=False)
        monkeypatch.setenv("SCAFACOS_LIB", "/nonexistent/libfcs.so")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "mmml.interfaces.pycharmmInterface.long_range_backend.have_scafacos",
                lambda: False,
            )
            mp.setattr(
                "mmml.interfaces.pycharmmInterface.long_range_backend.have_jax_pme",
                lambda: False,
            )
            assert pick_lr_solver("auto") == "mic"


@pytest.mark.parametrize("separation_A", [5.0, 8.0, 12.0])
def test_mic_dimer_matches_analytic_in_large_box(separation_A: float):
    system = ion_dimer_system(separation_A=separation_A, box_length_A=60.0)
    result = mic_coulomb_energy_forces(system, cutoff_A=None)
    e_ref = -CHARMM_COULOMB_KCAL / separation_A
    f_ref = CHARMM_COULOMB_KCAL / separation_A**2
    np.testing.assert_allclose(result.energy_kcalmol, e_ref, rtol=1e-10)
    np.testing.assert_allclose(result.forces_kcalmol_A[0, 0], f_ref, rtol=1e-8)
    np.testing.assert_allclose(result.forces_kcalmol_A[1, 0], -f_ref, rtol=1e-8)


@pytest.mark.parametrize(
    ("system_factory", "method", "rtol"),
    [
        (lambda: cscl_crystal(box_length_A=1.0), "ewald", 4e-6),
        (lambda: cscl_crystal(box_length_A=1.0), "pme", 9e-4),
        (lambda: cscl_crystal(box_length_A=1.0), "p3m", 9e-4),
        (lambda: nacl_cubic(box_length_A=2.0), "ewald", 4e-6),
    ],
)
def test_jax_pme_madelung(system_factory, method, rtol):
    if not have_jax_pme_package():
        pytest.skip("jax-pme not installed")
    system = system_factory()
    result = jax_pme_coulomb_energy_forces(system, method=method, sr_cutoff_A=6.0)
    m = madelung_constant(result, system)
    np.testing.assert_allclose(m, system.madelung_ref, rtol=rtol)


@pytest.mark.parametrize("method", ["ewald", "pme", "p3m"])
def test_jax_pme_methods_agree_on_cscl(method: str):
    if not have_jax_pme_package():
        pytest.skip("jax-pme not installed")
    system = cscl_crystal(box_length_A=10.0)
    ref = jax_pme_coulomb_energy_forces(system, method="ewald", sr_cutoff_A=6.0)
    test = jax_pme_coulomb_energy_forces(system, method=method, sr_cutoff_A=6.0)
    np.testing.assert_allclose(test.energy_kcalmol, ref.energy_kcalmol, rtol=2e-3)


def test_mic_truncated_underestimates_vs_jax_pme():
    if not have_jax_pme_package():
        pytest.skip("jax-pme not installed")
    system = random_neutral_cluster(n_atoms=8, box_length_A=12.0, seed=3)
    mic_trunc = mic_coulomb_energy_forces(system, cutoff_A=DEFAULT_MM_COULOMB_CUTOFF_A)
    ewald = jax_pme_coulomb_energy_forces(system, method="ewald", sr_cutoff_A=6.0)
    assert abs(mic_trunc.energy_kcalmol) <= abs(ewald.energy_kcalmol) * 1.01


def test_mic_vs_jax_pme_large_box_dimer():
    if not have_jax_pme_package():
        pytest.skip("jax-pme not installed")
    system = ion_dimer_system(separation_A=6.0, box_length_A=50.0)
    mic = mic_coulomb_energy_forces(system, cutoff_A=None)
    ewald = jax_pme_coulomb_energy_forces(system, method="ewald", sr_cutoff_A=6.0)
    np.testing.assert_allclose(mic.energy_kcalmol, ewald.energy_kcalmol, rtol=0.05)


@pytest.mark.skipif(
    not scafacos_integration_enabled(),
    reason="Set MMML_SCAFACOS_TESTS=1 and SCAFACOS_LIB for ScaFaCoS integration",
)
@pytest.mark.parametrize("method", ["ewald", "p3m"])
def test_scafacos_vs_jax_pme_cscl(method: str):
    if not have_jax_pme_package():
        pytest.skip("jax-pme not installed")
    system = cscl_crystal(box_length_A=10.0)
    ref = jax_pme_coulomb_energy_forces(system, method="ewald", sr_cutoff_A=6.0)
    method = os.environ.get("SCAFACOS_METHOD", method)
    out = scafacos_coulomb_energy_forces(system, method=method)
    np.testing.assert_allclose(out.energy_kcalmol, ref.energy_kcalmol, rtol=5e-3)
    np.testing.assert_allclose(out.forces_kcalmol_A, ref.forces_kcalmol_A, rtol=5e-2)
