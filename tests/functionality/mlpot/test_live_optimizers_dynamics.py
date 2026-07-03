"""Live GPU/CHARMM validation for MLpot minimizers (SD, FIRE, BFGS) and short dynamics.

Run on a reserved GPU compute node::

    export MMML_CKPT=/path/to/dcm1_params.json
    export JAX_ENABLE_X64=1
    ./scripts/run_pycharmm_pytest_gpu.sh live -q
    # or the full live file:
    ./scripts/run_pycharmm_pytest_gpu.sh \\
      tests/functionality/mlpot/test_live_optimizers_dynamics.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _live_helpers import (
    TIMESTEP_PS,
    can_import_pycharmm,
    max_displacement,
    positions_for_resids,
    resolve_live_checkpoint,
    run_short_sd,
    setup_aco_mlpot,
    setup_dcm_mlpot,
    translate_resid_and_sync,
)

pytestmark = [
    pytest.mark.skipif(not can_import_pycharmm(), reason="pycharmm not available"),
]


@pytest.fixture(scope="module")
def live_ckpt() -> Path:
    ckpt = resolve_live_checkpoint()
    if ckpt is None:
        pytest.skip("No PhysNet checkpoint (set MMML_CKPT)")
    return ckpt


def test_sd_minimization_lowers_hybrid_grms(live_ckpt: Path) -> None:
    """CHARMM SD pass 1 under MLpot should reduce hybrid GRMS on a relaxed dimer."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        measure_hybrid_charmm_grms,
    )

    ctx, _z, r, _n = setup_aco_mlpot(live_ckpt, n_molecules=2, spacing=4.0)
    try:
        before = float(measure_hybrid_charmm_grms(ctx).hybrid)
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=30,
                nprint=30,
                reference_positions=r,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=False,
                show_energy=False,
                verbose=False,
            )
        )
        after = float(measure_hybrid_charmm_grms(ctx).hybrid)
    finally:
        ctx.unset()

    assert np.isfinite(before) and before > 0.0
    assert after < before * 0.95 or after < 5.0


def test_hybrid_fire_bfgs_pre_sd_lowers_grms(live_ckpt: Path) -> None:
    """ASE FIRE + BFGS on the hybrid calculator should recover from a geometry stress spike."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        measure_hybrid_charmm_grms,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import sync_charmm_positions

    ctx, _z, r, _n = setup_aco_mlpot(live_ckpt, n_molecules=4, spacing=4.0)
    try:
        stressed = np.asarray(r, dtype=float).copy()
        stressed[:, 2] += 0.15
        sync_charmm_positions(stressed)
        before = float(measure_hybrid_charmm_grms(ctx).hybrid)
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=15,
                nprint=15,
                reference_positions=stressed,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=True,
                calculator_minimize_steps=60,
                calculator_fire_steps=60,
                show_energy=False,
                verbose=False,
            )
        )
        after = float(measure_hybrid_charmm_grms(ctx).hybrid)
    finally:
        ctx.unset()

    assert before > 10.0
    assert after < before * 0.8


def test_cons_fix_pass2_freezes_fixed_monomer(live_ckpt: Path) -> None:
    """SD pass 2 with cons_fix should not move atoms on the fixed resid."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        MinimizeWithMlpotConfig,
        minimize_with_mlpot,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        setup_cons_fix_for_resids,
        turn_off_cons_fix,
    )

    ctx, _z, r, _n = setup_aco_mlpot(live_ckpt, n_molecules=4, spacing=4.0)
    try:
        translate_resid_and_sync([2], (0.5, 0.0, 0.0))
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=25,
                nprint=25,
                reference_positions=r,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=False,
                show_energy=False,
                verbose=False,
            )
        )
        anchor = positions_for_resids([1])
        setup_cons_fix_for_resids([1])
        run_short_sd(nstep=25)
        turn_off_cons_fix()
        after = positions_for_resids([1])
    finally:
        ctx.unset()

    assert max_displacement(anchor, after) < 1e-4


def test_nve_short_run_stays_finite(live_ckpt: Path, tmp_path: Path) -> None:
    """Mini + short vacuum NVE should finish with finite coordinates and energies."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        MinimizeWithMlpotConfig,
        build_nve_dynamics,
        minimize_with_mlpot,
        run_dynamics_with_io,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_state_is_finite,
    )

    ctx, z, r, _n = setup_aco_mlpot(live_ckpt, n_molecules=2, spacing=4.0)
    res_path = tmp_path / "nve_live.res"
    dcd_path = tmp_path / "nve_live.dcd"
    nstep = 40
    try:
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=20,
                nprint=20,
                reference_positions=r,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=False,
                show_energy=False,
                verbose=False,
            )
        )
        kw = build_nve_dynamics(
            timestep_ps=TIMESTEP_PS,
            duration_ps=nstep * TIMESTEP_PS,
            save_interval_ps=TIMESTEP_PS * 4,
            restart=False,
            temp=300.0,
            nprint=nstep,
            echeck=500.0,
            use_pbc=False,
        )
        kw.update(new=True, start=True, nstep=nstep, nsavc=4)
        run_dynamics_with_io(
            kw,
            CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path),
            overlap_context="live nve finite",
        )
        assert charmm_dynamics_state_is_finite()
    finally:
        ctx.unset()

    assert res_path.is_file() and res_path.stat().st_size > 0
    assert dcd_path.is_file() and dcd_path.stat().st_size > 0


def test_nve_cons_fix_holds_fixed_monomer(live_ckpt: Path, tmp_path: Path) -> None:
    """``cons_fix`` during NVE should keep the fixed monomer fixed."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        MinimizeWithMlpotConfig,
        build_nve_dynamics,
        minimize_with_mlpot,
        run_dynamics_with_io,
        select_by_resids,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
        setup_cons_fix_for_resids,
        turn_off_cons_fix,
    )

    ctx, _z, r, _n = setup_aco_mlpot(live_ckpt, n_molecules=4, spacing=4.0)
    fix_sel = select_by_resids([1])
    res_path = tmp_path / "nve_fix.res"
    dcd_path = tmp_path / "nve_fix.dcd"
    nstep = 30
    try:
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                fixed_ml_selection=fix_sel,
                nstep=20,
                nprint=20,
                reference_positions=r,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=False,
                show_energy=False,
                verbose=False,
            )
        )
        anchor = positions_for_resids([1])
        setup_cons_fix_for_resids([1])
        kw = build_nve_dynamics(
            timestep_ps=TIMESTEP_PS,
            duration_ps=nstep * TIMESTEP_PS,
            save_interval_ps=TIMESTEP_PS * 4,
            restart=False,
            temp=300.0,
            nprint=nstep,
            echeck=500.0,
            use_pbc=False,
        )
        kw.update(new=True, start=True, nstep=nstep, nsavc=4)
        run_dynamics_with_io(
            kw,
            CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path),
            overlap_context="live nve cons_fix",
        )
        turn_off_cons_fix()
        after = positions_for_resids([1])
    finally:
        ctx.unset()

    assert max_displacement(anchor, after) < 1e-3


def test_dcm_dimer_mini_and_nve_smoke(live_ckpt: Path, tmp_path: Path) -> None:
    """DCM:2 smoke with the production dcm1 checkpoint (when available)."""
    from mmml.interfaces.pycharmmInterface.mlpot import (
        CharmmTrajectoryFiles,
        MinimizeWithMlpotConfig,
        build_nve_dynamics,
        minimize_with_mlpot,
        run_dynamics_with_io,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation import (
        charmm_dynamics_state_is_finite,
    )

    ctx, _z, r, _n = setup_dcm_mlpot(live_ckpt, n_molecules=2, spacing=4.0)
    res_path = tmp_path / "dcm_nve.res"
    dcd_path = tmp_path / "dcm_nve.dcd"
    nstep = 25
    try:
        minimize_with_mlpot(
            MinimizeWithMlpotConfig(
                nstep=20,
                nprint=20,
                reference_positions=r,
                pyCModel=ctx.pyCModel,
                mlpot_ctx=ctx,
                skip_if_crd_exists=False,
                calculator_pre_minimize=True,
                calculator_minimize_steps=40,
                show_energy=False,
                verbose=False,
            )
        )
        kw = build_nve_dynamics(
            timestep_ps=TIMESTEP_PS,
            duration_ps=nstep * TIMESTEP_PS,
            save_interval_ps=TIMESTEP_PS * 5,
            restart=False,
            temp=300.0,
            nprint=nstep,
            echeck=500.0,
            use_pbc=False,
        )
        kw.update(new=True, start=True, nstep=nstep, nsavc=5)
        run_dynamics_with_io(
            kw,
            CharmmTrajectoryFiles(restart_write=res_path, trajectory=dcd_path),
            overlap_context="live dcm dimer smoke",
        )
        assert charmm_dynamics_state_is_finite()
    finally:
        ctx.unset()
