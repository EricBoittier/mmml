"""Unit tests for JAX vs CHARMM recovery parity dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.jax_charmm_parity_report import (
    RecoveryMmParityMetrics,
    collect_recovery_mm_parity_metrics,
    emit_recovery_mm_parity_dashboard,
)


def test_collect_recovery_mm_parity_metrics_computes_deltas():
    ctx = MagicMock(use_pbc=False)
    positions = np.zeros((3, 3))
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.jax_charmm_parity_report._charmm_recovery_reference",
        return_value=({"bonded": 10.0, "vdw": 2.0, "total": 12.0}, np.ones((3, 3))),
    ), patch(
        "mmml.interfaces.pycharmmInterface.mlpot.jax_charmm_parity_report._jax_recovery_reference",
        return_value=({"bonded": 10.0, "vdw": 2.0, "total": 12.0}, np.ones((3, 3))),
    ):
        metrics = collect_recovery_mm_parity_metrics(ctx, positions)
    assert metrics is not None
    assert metrics.within_tolerance is True
    assert metrics.delta_energy_kcal == 0.0
    assert metrics.force_rms_delta == 0.0


def test_emit_recovery_mm_parity_dashboard_calls_rich():
    metrics = RecoveryMmParityMetrics(
        jax_bonded_kcal=1.0,
        charmm_bonded_kcal=1.0,
        jax_vdw_kcal=2.0,
        charmm_vdw_kcal=2.0,
        jax_total_kcal=3.0,
        charmm_total_kcal=3.0,
        delta_energy_kcal=0.0,
        force_rms_delta=0.01,
        force_max_delta=0.02,
        n_atoms_compared=10,
        within_tolerance=True,
    )
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.jax_charmm_parity_report.emit_dashboard",
    ) as dashboard:
        emit_recovery_mm_parity_dashboard(metrics, context="test")
    dashboard.assert_called_once()
    assert dashboard.call_args.kwargs["border_style"] == "green"
