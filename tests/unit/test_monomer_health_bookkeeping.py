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
    select_flagged_bad_by_highest_grms,
    select_systemic_velocity_warn_by_highest_grms,
)


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


def test_classify_component_ratio_warn() -> None:
    level, reasons = _classify_component(
        4000.0,
        1000.0,
        warn_ratio=3.0,
        bad_ratio=6.0,
        warn_abs=50000.0,
        bad_abs=100000.0,
        name="GRMS",
    )
    assert level == LEVEL_WARN
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
    )
    assert report is not None
    assert 1 in report.flagged_bad
    assert report.entries[1].velocity_level == LEVEL_BAD


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
                reasons=("MM ratio 5.0× baseline",),
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

    assert recovered is True
    redraw.assert_called_once()
    assert redraw.call_args.args[1] == (1, 2, 0)
    invalidate.assert_called_once_with(ctx)
