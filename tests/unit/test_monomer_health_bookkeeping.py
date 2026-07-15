"""Unit tests for per-monomer health bookkeeping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
    LEVEL_BAD,
    LEVEL_OK,
    LEVEL_WARN,
    MonomerHealthBaseline,
    MonomerHealthConfig,
    MonomerHealthEntry,
    MonomerHealthReport,
    _classify_component,
    _per_monomer_velocity_stats,
    audit_monomer_health,
    emit_monomer_health_dot_matrix,
    monomer_health_config_from_args,
    resolve_monomer_offsets_for_ctx,
    select_flagged_bad_by_highest_grms,
    select_systemic_velocity_warn_by_highest_grms,
)


def test_resolve_monomer_offsets_uses_composition_for_mixed_system() -> None:
    ctx = SimpleNamespace(
        atoms_per_monomer=None,
        pyCModel=None,
        workflow_args=SimpleNamespace(
            composition="MEOH:1,TIP3:1",
            _cluster_atoms_per_list=None,
        ),
    )
    offsets = resolve_monomer_offsets_for_ctx(ctx, n_monomers=2, n_atoms=9)
    assert offsets is not None
    np.testing.assert_array_equal(offsets, [0, 6, 9])


def test_classify_component_absolute_bad() -> None:
    level, reasons = _classify_component(
        20000.0,
        1000.0,
        warn_ratio=3.0,
        bad_ratio=6.0,
        warn_abs=5000.0,
        bad_abs=15000.0,
        name="|v|",
    )
    assert level == LEVEL_BAD
    assert reasons


def test_classify_component_ratio_alone_does_not_warn() -> None:
    """Tiny baseline + large ratio must not flag when abs floors are not met."""
    level, reasons = _classify_component(
        4000.0,
        1.0,
        warn_ratio=3.0,
        bad_ratio=6.0,
        warn_abs=50000.0,
        bad_abs=100000.0,
        name="GRMS",
        baseline_floor=12.5,
        ratio_requires_abs_warn=True,
    )
    assert level == LEVEL_OK
    assert not reasons


def test_classify_component_ratio_annotates_when_abs_warn_met() -> None:
    level, reasons = _classify_component(
        40.0,
        5.0,
        warn_ratio=2.5,
        bad_ratio=5.0,
        warn_abs=30.0,
        bad_abs=80.0,
        name="GRMS",
        baseline_floor=7.5,
        ratio_requires_abs_warn=True,
    )
    assert level == LEVEL_WARN
    assert any("abs" in r for r in reasons)
    assert any("ratio" in r for r in reasons)


def test_per_monomer_velocity_stats() -> None:
    offsets = np.array([0, 2, 4], dtype=int)
    vel = np.array(
        [
            [1000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [3000.0, 4000.0, 0.0],
        ],
        dtype=float,
    )
    rms, vmax = _per_monomer_velocity_stats(vel, offsets)
    assert vmax[0] == pytest.approx(1000.0)
    assert vmax[1] == pytest.approx(5000.0)
    assert rms[1] > rms[0]


def test_monomer_health_config_from_args() -> None:
    args = SimpleNamespace(
        no_dynamics_monomer_health=False,
        dynamics_monomer_health_debug=True,
        no_dynamics_monomer_template_restore=False,
        no_dynamics_monomer_jax_after_restore=False,
        dynamics_monomer_health_max_restore=2,
        dynamics_monomer_velocity_warn_ratio=3.0,
        dynamics_monomer_velocity_bad_ratio=6.0,
        dynamics_monomer_velocity_warn_akma=5000.0,
        dynamics_monomer_velocity_bad_akma=15000.0,
        dynamics_monomer_velocity_warn_recover_fraction=0.8,
        dynamics_monomer_force_warn_ratio=2.5,
        dynamics_monomer_force_bad_ratio=5.0,
        dynamics_monomer_energy_warn_ratio=2.0,
        dynamics_monomer_energy_bad_ratio=4.0,
        quiet=True,
    )
    cfg = monomer_health_config_from_args(args)
    assert cfg.enabled
    assert cfg.debug_dot_matrix
    assert cfg.max_restore_per_check == 2
    assert cfg.velocity_warn_recover_fraction == pytest.approx(0.8)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.collect_monomer_health_metrics"
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.resolve_monomer_offsets_for_ctx",
    return_value=np.array([0, 2, 4], dtype=int),
)
def test_audit_monomer_health_flags_bad_monomer(
    _offsets: MagicMock,
    collect_metrics: MagicMock,
) -> None:
    ctx = SimpleNamespace(
        _monomer_health_baseline=MonomerHealthBaseline(
            velocity_rms_akma=np.array([100.0, 100.0]),
            velocity_max_akma=np.array([200.0, 200.0]),
            hybrid_grms_kcalmol_A=np.array([5.0, 5.0]),
            charmm_grms_kcalmol_A=np.array([3.0, 3.0]),
        ),
        workflow_args=SimpleNamespace(residue="DCM"),
        atoms_per_monomer=[2, 2],
    )
    collect_metrics.return_value = (
        np.array([100.0, 100.0]),
        np.array([200.0, 20000.0]),
        np.array([5.0, 40.0]),
        np.array([3.0, 30.0]),
    )
    report = audit_monomer_health(
        ctx,
        MonomerHealthConfig(),
        n_monomers=2,
        global_step=100,
        overlap_config=None,
    )
    assert report is not None
    assert 1 in report.flagged_bad
    assert report.entries[1].velocity_level == LEVEL_BAD
    assert report.entries[1].geometry_level == LEVEL_OK
    assert report.entries[1].needs_template_restore is False


def test_select_flagged_bad_by_highest_grms() -> None:
    report = MonomerHealthReport(
        entries=(
            MonomerHealthEntry(
                index=0,
                label="DCM",
                velocity_rms_akma=None,
                velocity_max_akma=None,
                hybrid_grms_kcalmol_A=30.0,
                charmm_grms_kcalmol_A=20.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_BAD,
                energy_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=1,
                label="DCM",
                velocity_rms_akma=None,
                velocity_max_akma=None,
                hybrid_grms_kcalmol_A=90.0,
                charmm_grms_kcalmol_A=40.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_BAD,
                energy_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=2,
                label="DCM",
                velocity_rms_akma=None,
                velocity_max_akma=None,
                hybrid_grms_kcalmol_A=60.0,
                charmm_grms_kcalmol_A=80.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_BAD,
                energy_level=LEVEL_OK,
            ),
        ),
        flagged_bad=(0, 1, 2),
        flagged_warn=(),
        baseline_recorded=False,
    )

    assert select_flagged_bad_by_highest_grms(report, max_select=2) == (1, 2)


def test_select_systemic_velocity_warn_by_highest_grms() -> None:
    report = MonomerHealthReport(
        entries=(
            MonomerHealthEntry(
                index=0,
                label="DCM",
                velocity_rms_akma=6000.0,
                velocity_max_akma=9000.0,
                hybrid_grms_kcalmol_A=5.0,
                charmm_grms_kcalmol_A=4.0,
                velocity_level=LEVEL_WARN,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=1,
                label="DCM",
                velocity_rms_akma=7000.0,
                velocity_max_akma=11000.0,
                hybrid_grms_kcalmol_A=8.0,
                charmm_grms_kcalmol_A=6.0,
                velocity_level=LEVEL_WARN,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=2,
                label="DCM",
                velocity_rms_akma=100.0,
                velocity_max_akma=200.0,
                hybrid_grms_kcalmol_A=90.0,
                charmm_grms_kcalmol_A=80.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
            ),
        ),
        flagged_bad=(),
        flagged_warn=(0, 1),
        baseline_recorded=False,
    )

    assert select_systemic_velocity_warn_by_highest_grms(
        report,
        min_fraction=0.5,
    ) == (1, 0)
    assert not select_systemic_velocity_warn_by_highest_grms(
        report,
        min_fraction=0.8,
    )


def test_emit_monomer_health_dot_matrix_plain(capsys: pytest.CaptureFixture[str]) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        MonomerHealthEntry,
        MonomerHealthReport,
    )

    report = MonomerHealthReport(
        entries=(
            MonomerHealthEntry(
                index=0,
                label="DCM",
                velocity_rms_akma=1.0,
                velocity_max_akma=2.0,
                hybrid_grms_kcalmol_A=3.0,
                charmm_grms_kcalmol_A=2.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_WARN,
                energy_level=LEVEL_BAD,
                geometry_level=LEVEL_BAD,
                reasons=("extent 20.0 Å > 12.0 Å",),
            ),
        ),
        flagged_bad=(0,),
        flagged_warn=(),
        baseline_recorded=False,
    )
    with patch("mmml.utils.rich_report.rich_enabled", return_value=False):
        emit_monomer_health_dot_matrix(report, context="test", quiet=False)
    out = capsys.readouterr().out
    assert "DCM" in out
    assert "geometry" in out
    assert "G O R" in out or "G" in out


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.sync_charmm_velocities_akma"
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.velocities_are_pathological",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
    return_value=np.ones(4),
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.dynamics_validation.read_restart_velocities"
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping._current_velocities_akma"
)
def test_restore_monomer_velocities_splices_template_slice(
    current_vel: MagicMock,
    read_restart: MagicMock,
    _masses: MagicMock,
    _pathological: MagicMock,
    sync_vel: MagicMock,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        restore_monomer_velocities_from_template,
    )

    current_vel.return_value = np.array(
        [
            [50000.0, 0.0, 0.0],
            [50000.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    read_restart.return_value = np.array(
        [
            [200.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    offsets = np.array([0, 2, 4], dtype=int)
    ok = restore_monomer_velocities_from_template(
        SimpleNamespace(workflow_args=SimpleNamespace(temperature=100.0)),
        (0,),
        offsets=offsets,
        template_source="/tmp/baseline.res",
        verbose=False,
    )
    assert ok
    synced = sync_vel.call_args[0][0]
    assert synced[0, 0] == pytest.approx(200.0)
    assert synced[1, 0] == pytest.approx(300.0)
    assert synced[2, 0] == pytest.approx(100.0)


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.restore_flagged_monomers_from_template",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.redraw_monomer_velocities",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.resolve_monomer_offsets_for_ctx",
    return_value=np.array([0, 2, 4, 6], dtype=int),
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.audit_monomer_health"
)
@patch("pycharmm.coor.get_natom", return_value=6)
def test_maybe_intervene_monomer_health_recovers_systemic_velocity_warn(
    _natom: MagicMock,
    audit: MagicMock,
    _offsets: MagicMock,
    redraw: MagicMock,
    restore_template: MagicMock,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        maybe_intervene_monomer_health,
    )

    audit.return_value = MonomerHealthReport(
        entries=(
            MonomerHealthEntry(
                index=0,
                label="DCM",
                velocity_rms_akma=7000.0,
                velocity_max_akma=9000.0,
                hybrid_grms_kcalmol_A=2.0,
                charmm_grms_kcalmol_A=1.0,
                velocity_level=LEVEL_WARN,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
                geometry_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=1,
                label="DCM",
                velocity_rms_akma=8000.0,
                velocity_max_akma=12000.0,
                hybrid_grms_kcalmol_A=4.0,
                charmm_grms_kcalmol_A=1.0,
                velocity_level=LEVEL_WARN,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
                geometry_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=2,
                label="DCM",
                velocity_rms_akma=9000.0,
                velocity_max_akma=13000.0,
                hybrid_grms_kcalmol_A=3.0,
                charmm_grms_kcalmol_A=1.0,
                velocity_level=LEVEL_WARN,
                force_level=LEVEL_OK,
                energy_level=LEVEL_OK,
                geometry_level=LEVEL_OK,
            ),
        ),
        flagged_bad=(),
        flagged_warn=(0, 1, 2),
        baseline_recorded=False,
    )
    ctx = MagicMock(workflow_args=SimpleNamespace(temperature=10.0))
    overlap = SimpleNamespace(
        n_monomers=3,
        monomer_health=MonomerHealthConfig(verbose=False),
    )

    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
    ) as invalidate:
        recovered = maybe_intervene_monomer_health(
            ctx,
            overlap,
            context="NVE",
            global_step=100,
        )
    assert recovered
    assert recovered.velocities_redrawn
    assert not recovered.geometry_restored
    redraw.assert_called_once()
    restore_template.assert_not_called()
    invalidate.assert_called()


@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping._run_per_monomer_jax_on_indices"
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.restore_flagged_monomers_from_template",
    return_value=True,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.redraw_monomer_velocities",
    return_value=False,
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.resolve_monomer_offsets_for_ctx",
    return_value=np.array([0, 2, 4], dtype=int),
)
@patch(
    "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.audit_monomer_health"
)
@patch("pycharmm.coor.get_natom", return_value=4)
def test_maybe_intervene_templates_only_geometry_bad(
    _natom: MagicMock,
    audit: MagicMock,
    _offsets: MagicMock,
    redraw: MagicMock,
    restore_template: MagicMock,
    jax_mini: MagicMock,
) -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        maybe_intervene_monomer_health,
    )

    audit.return_value = MonomerHealthReport(
        entries=(
            MonomerHealthEntry(
                index=0,
                label="TIP3",
                velocity_rms_akma=50000.0,
                velocity_max_akma=80000.0,
                hybrid_grms_kcalmol_A=90.0,
                charmm_grms_kcalmol_A=90.0,
                velocity_level=LEVEL_BAD,
                force_level=LEVEL_BAD,
                energy_level=LEVEL_OK,
                geometry_level=LEVEL_OK,
            ),
            MonomerHealthEntry(
                index=1,
                label="TIP3",
                velocity_rms_akma=100.0,
                velocity_max_akma=200.0,
                hybrid_grms_kcalmol_A=5.0,
                charmm_grms_kcalmol_A=5.0,
                velocity_level=LEVEL_OK,
                force_level=LEVEL_OK,
                energy_level=LEVEL_BAD,
                geometry_level=LEVEL_BAD,
                reasons=("extent 20.0 Å > 12.0 Å",),
            ),
        ),
        flagged_bad=(0, 1),
        flagged_warn=(),
        baseline_recorded=False,
    )
    ctx = MagicMock(workflow_args=SimpleNamespace(temperature=200.0))
    overlap = SimpleNamespace(
        n_monomers=2,
        monomer_health=MonomerHealthConfig(
            verbose=False,
            template_restore_requires_geometry=True,
            per_monomer_jax_after_restore=True,
        ),
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.dynamics.invalidate_mlpot_calculator_caches",
    ):
        ok = maybe_intervene_monomer_health(
            ctx, overlap, context="HEAT", global_step=500
        )
    assert ok
    assert ok.geometry_restored
    assert not ok.velocities_redrawn  # redraw mock returns False
    restore_template.assert_called_once()
    restored_idx = restore_template.call_args[0][1]
    assert restored_idx == (1,)
    redraw.assert_called_once()
    redraw_idx = redraw.call_args[0][1]
    assert 0 in redraw_idx
    assert 1 not in redraw_idx
    jax_mini.assert_called_once()


def test_maybe_rebaseline_heat_once() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        maybe_rebaseline_monomer_health_after_heat_velocities,
    )

    ctx = SimpleNamespace()
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping.record_monomer_health_baseline",
        return_value=object(),
    ) as rec:
        assert maybe_rebaseline_monomer_health_after_heat_velocities(
            ctx, n_monomers=4, context="HEAT chunk", global_step=250
        )
        assert maybe_rebaseline_monomer_health_after_heat_velocities(
            ctx, n_monomers=4, context="HEAT chunk", global_step=500
        ) is False
    assert rec.call_count == 1


def test_com_unwrap_flags_rigid_flyoff() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        MonomerHealthConfig,
        _update_com_unwrap_state,
        flag_geometry_problem_monomers,
    )

    ctx = SimpleNamespace(_monomer_com_unwrap_reset=True)
    cell = np.diag([30.0, 30.0, 30.0])
    coms0 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    _update_com_unwrap_state(ctx, coms0, cell, reset_baseline=True)
    # Incremental unwrap across the cell (IMAGE would show small wrapped jump).
    coms1 = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    for _ in range(20):
        coms1 = coms1.copy()
        coms1[0, 0] = (coms1[0, 0] + 1.0) % 30.0
        _update_com_unwrap_state(ctx, coms1, cell, reset_baseline=False)
    drift = np.linalg.norm(
        ctx._monomer_com_unwrap_state["unwrapped"][0]
        - ctx._monomer_com_unwrap_state["baseline_unwrapped"][0]
    )
    assert drift > 15.0

    offsets = np.array([0, 3, 6], dtype=int)
    pos = np.zeros((6, 3), dtype=float)
    pos[0] = coms1[0]
    pos[1] = coms1[0] + [0.1, 0.0, 0.0]
    pos[2] = coms1[0] + [0.0, 0.1, 0.0]
    pos[3] = coms1[1]
    pos[4] = coms1[1] + [0.1, 0.0, 0.0]
    pos[5] = coms1[1] + [0.0, 0.1, 0.0]
    overlap = SimpleNamespace(
        max_monomer_extent_A=0.0,
        intra_min_distance_A=0.0,
        use_pbc=True,
        fallback_box_side_A=30.0,
        intra_exclude_1_3=True,
    )
    with (
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.setup.get_charmm_positions_array",
            return_value=pos,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.overlap_guard._overlap_cell",
            return_value=cell,
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities.charmm_masses_amu",
            return_value=np.ones(6),
        ),
        patch(
            "mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping._flag_bond_stretch_monomers",
            return_value={},
        ),
    ):
        flagged = flag_geometry_problem_monomers(
            ctx,
            overlap,
            offsets=offsets,
            health_config=MonomerHealthConfig(com_flyoff_A=15.0),
        )
    assert 0 in flagged
    assert any("COM drift" in r for r in flagged[0])
    assert 1 not in flagged


def test_bond_stretch_flags_geometry() -> None:
    from mmml.interfaces.pycharmmInterface.mlpot.monomer_health_bookkeeping import (
        _flag_bond_stretch_monomers,
    )

    offsets = np.array([0, 3], dtype=int)
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    pos = ref.copy()
    pos[1, 0] = 3.0  # 3× stretch of first bond
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.monomer_geometry_limits.psf_bond_pairs_0based",
        return_value=[(0, 1), (0, 2)],
    ):
        flagged = _flag_bond_stretch_monomers(
            pos,
            offsets,
            stretch_factor=1.75,
            stretch_abs_A=2.5,
            ref_positions=ref,
        )
    assert 0 in flagged
    assert any("bond" in r for r in flagged[0])
