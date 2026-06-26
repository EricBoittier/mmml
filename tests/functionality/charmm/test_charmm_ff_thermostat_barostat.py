"""Live PyCHARMM tests for barostat, thermostat, and forces (CGENFF only, no MLpot)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import can_import_pycharmm


pytestmark = pytest.mark.skipif(
    not can_import_pycharmm(),
    reason="pycharmm / libcharmm not available",
)


def test_tip3_ener_force_finite_grms_and_forces(tip3_charmm_ff):
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_grms_after_ener_force,
        charmm_total_forces_kcalmol_A,
    )

    grms = charmm_grms_after_ener_force(silent=True)
    forces = charmm_total_forces_kcalmol_A()
    assert np.isfinite(grms)
    assert grms >= 0.0
    assert forces.shape == (len(tip3_charmm_ff), 3)
    assert np.all(np.isfinite(forces))


def test_tip3_forces_are_negative_energy_gradient(tip3_charmm_ff):
    import pycharmm.coor as coor
    import pycharmm.lingo as lingo

    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        charmm_total_forces_kcalmol_A,
    )

    lingo.charmm_script("ENER FORCE")
    grad_df = coor.get_forces()
    gradient = np.column_stack(
        [
            grad_df["dx"].to_numpy(dtype=float),
            grad_df["dy"].to_numpy(dtype=float),
            grad_df["dz"].to_numpy(dtype=float),
        ]
    )
    forces = charmm_total_forces_kcalmol_A()
    np.testing.assert_allclose(forces, -gradient, rtol=1e-5, atol=1e-8)


def test_tip3_hoover_cpt_nvt_short_dynamics_completes(tip3_charmm_ff):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        build_hoover_heat_dynamics,
        run_dynamics,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        ensure_charmm_crystal_for_cpt,
    )

    # Small cubic cell around a single TIP3 — CPT NVT at fixed volume (pmass=0).
    ensure_charmm_crystal_for_cpt(30.0, quiet=True)
    kw = build_hoover_heat_dynamics(
        temp=300.0,
        firstt=300.0,
        finalt=300.0,
        use_pbc=True,
        tmass=100,
        duration_ps=0.0004,
        timestep_ps=0.0002,
    )
    kw["nstep"] = 2
    kw["nsavc"] = 1
    kw["start"] = True
    kw["iasvel"] = 1
    kw["restart"] = False
    kw["iunrea"] = -1
    run_dynamics(kw)

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_energy_is_finite,
    )

    assert charmm_dynamics_energy_is_finite()


def test_tip3_cpt_npt_barostat_keywords_accepted(tip3_charmm_ff):
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        build_cpt_equilibration_dynamics,
        run_dynamics,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.pbc_env import (
        ensure_charmm_crystal_for_cpt,
    )

    ensure_charmm_crystal_for_cpt(30.0, quiet=True)
    kw = build_cpt_equilibration_dynamics(
        temp=300.0,
        pmass=2,
        tmass=20,
        pref=1.0,
        duration_ps=0.0004,
        timestep_ps=0.0002,
    )
    kw["nstep"] = 2
    kw["nsavc"] = 1
    kw["start"] = True
    kw["iasvel"] = 1
    kw["restart"] = False
    kw["iunrea"] = -1
    run_dynamics(kw)

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_energy_is_finite,
    )

    assert charmm_dynamics_energy_is_finite()
