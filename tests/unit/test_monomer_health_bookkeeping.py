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
    _classify_component,
    _per_monomer_velocity_stats,
    audit_monomer_health,
    emit_monomer_health_dot_matrix,
    monomer_health_config_from_args,
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
