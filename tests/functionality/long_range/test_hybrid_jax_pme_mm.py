"""Hybrid / full-system MM with jax-pme Coulomb (LJ from pairs, elec from jax-pme)."""

from __future__ import annotations

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.long_range_backend import compute_jax_pme_coulomb
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    NonbondedSystemData,
    nonbonded_energy_and_forces,
)
from tests.functionality.long_range._common import (
    cscl_crystal,
    have_jax_pme_package,
    ion_dimer_system,
    jax_pme_coulomb_energy_forces,
)

pytestmark = pytest.mark.skipif(
    not have_jax_pme_package(),
    reason="jax-pme not installed",
)


def _cscl_nonbonded_data() -> tuple[np.ndarray, NonbondedSystemData, np.ndarray]:
    system = cscl_crystal(box_length_A=10.0)
    nbond = NonbondedSystemData(
        charges=system.charges_e,
        at_codes=np.zeros(system.positions_A.shape[0], dtype=np.int32),
        epsilon=np.zeros(system.positions_A.shape[0], dtype=np.float64),
        rmin=np.zeros(system.positions_A.shape[0], dtype=np.float64),
        excluded_pairs=frozenset(),
        e14_pairs=frozenset(),
    )
    cell = np.eye(3, dtype=np.float64) * system.box_length_A
    settings = CharmmNbondSettings(cutnb=12.0, ctonnb=10.0, ctofnb=12.0)
    return system.positions_A, nbond, cell, settings, system


@pytest.mark.parametrize("method", ["ewald", "pme", "p3m"])
def test_mm_system_nonbonded_jax_pme_elec(method: str):
    positions, nbond, cell, settings, system = _cscl_nonbonded_data()
    components, forces = nonbonded_energy_and_forces(
        positions,
        nbond,
        cell,
        settings,
        lr_solver="jax_pme",
        jax_pme_method=method,
    )
    ref = jax_pme_coulomb_energy_forces(system, method=method)  # type: ignore[arg-type]
    np.testing.assert_allclose(float(components["vdw"]), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(components["elec"]), ref.energy_kcalmol, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(forces), ref.forces_kcalmol_A, rtol=2e-3)


def test_mm_system_lj_unchanged_with_jax_pme_elec():
    """LJ from pair loop must not depend on lr_solver (zero LJ on CsCl test)."""
    positions, nbond, cell, settings, _ = _cscl_nonbonded_data()
    nbond_lj = NonbondedSystemData(
        charges=nbond.charges,
        at_codes=nbond.at_codes,
        epsilon=np.array([-0.1, -0.1], dtype=np.float64),
        rmin=np.array([2.0, 2.0], dtype=np.float64),
        excluded_pairs=nbond.excluded_pairs,
        e14_pairs=nbond.e14_pairs,
    )
    mic_comp, _ = nonbonded_energy_and_forces(
        positions, nbond_lj, cell, settings, lr_solver="mic"
    )
    pme_comp, _ = nonbonded_energy_and_forces(
        positions,
        nbond_lj,
        cell,
        settings,
        lr_solver="jax_pme",
        jax_pme_method="ewald",
    )
    np.testing.assert_allclose(float(mic_comp["vdw"]), float(pme_comp["vdw"]), rtol=1e-10)
    assert float(mic_comp["elec"]) != float(pme_comp["elec"])


def test_ion_dimer_jax_pme_via_mm_nonbonded():
    system = ion_dimer_system(separation_A=6.0, box_length_A=40.0)
    nbond = NonbondedSystemData(
        charges=system.charges_e,
        at_codes=np.zeros(2, dtype=np.int32),
        epsilon=np.zeros(2, dtype=np.float64),
        rmin=np.zeros(2, dtype=np.float64),
        excluded_pairs=frozenset(),
        e14_pairs=frozenset(),
    )
    cell = np.eye(3) * system.box_length_A
    settings = CharmmNbondSettings(cutnb=14.0, ctonnb=10.0, ctofnb=12.0)
    comp, forces = nonbonded_energy_and_forces(
        system.positions_A,
        nbond,
        cell,
        settings,
        lr_solver="jax_pme",
        jax_pme_method="ewald",
    )
    ref = compute_jax_pme_coulomb(
        system.positions_A,
        system.charges_e,
        box_length_A=system.box_length_A,
        method="ewald",
    )
    np.testing.assert_allclose(float(comp["elec"]), ref.energy_kcalmol, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(forces), ref.forces_kcalmol_A, rtol=1e-3)
