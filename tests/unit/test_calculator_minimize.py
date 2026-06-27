"""Hybrid calculator pre-minimize guards (BFGS spike abort, best-frame restore)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
    HybridCalculatorMinimizeConfig,
    _BestMinimizationFrame,
    _run_hybrid_calculator_bfgs,
    annotate_ase_optimizer_log_line,
    hybrid_calculator_mini_eligible,
    should_abort_bfgs_fmax,
    spike_fmax_limit_ev_a,
)


def test_spike_fmax_limit_uses_factor_and_floor():
    assert spike_fmax_limit_ev_a(2.27, factor=4.0, floor_ev_a=15.0) == pytest.approx(15.0)
    assert spike_fmax_limit_ev_a(5.0, factor=4.0, floor_ev_a=15.0) == pytest.approx(20.0)


def test_annotate_ase_optimizer_log_line_adds_kcal_mol():
    from mmml.data.units import EV_TO_KCAL_MOL

    header = "      Step     Time          Energy          fmax"
    annotated_header = annotate_ase_optimizer_log_line(header)
    assert "Energy[eV (kcal/mol)]" in annotated_header

    step = "FIRE:    0 15:37:38     -930.270153     1489.849493\n"
    annotated_step = annotate_ase_optimizer_log_line(step)
    kcal = -930.270153 * EV_TO_KCAL_MOL
    assert f"-930.270153 ({kcal:.2f})" in annotated_step
    assert "1489.849493" in annotated_step


def test_should_abort_bfgs_fmax_running_best_spike():
    assert should_abort_bfgs_fmax(
        1470.0,
        spike_limit_ev_a=4248.0,
        best_fmax_ev_a=827.0,
        absolute_ceiling_ev_a=500.0,
        running_spike_factor=1.5,
    )
    assert not should_abort_bfgs_fmax(
        900.0,
        spike_limit_ev_a=4248.0,
        best_fmax_ev_a=827.0,
        absolute_ceiling_ev_a=2000.0,
        running_spike_factor=1.5,
    )


def test_hybrid_calculator_mini_eligible_respects_grms_cap():
    assert not hybrid_calculator_mini_eligible(
        1428.0,
        grms_limit=50.0,
        diag_kind="geometry_stress",
        grms_hot=True,
        user_hot=False,
    )
    assert hybrid_calculator_mini_eligible(
        9.8,
        grms_limit=50.0,
        diag_kind="geometry_stress",
        grms_hot=False,
        user_hot=False,
    )


def test_best_minimization_frame_tracks_lowest_fmax():
    atoms = MagicMock()
    atoms.get_forces.side_effect = [
        np.array([[10.0, 0.0, 0.0]]),
        np.array([[6.0, 0.0, 0.0]]),
        np.array([[900.0, 0.0, 0.0]]),
    ]
    atoms.get_positions.side_effect = [
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.full((1, 3), 99.0),
    ]
    frame = _BestMinimizationFrame(atoms)
    frame.record("initial")
    frame.record("step_1")
    frame.record("step_2")
    assert frame.best_force_label == "step_1"
    assert frame.best_force_fmax == pytest.approx(6.0)
    frame.restore_best_force()
    atoms.set_positions.assert_called_once()
    np.testing.assert_array_equal(atoms.set_positions.call_args.args[0], np.ones((1, 3)))


def test_run_hybrid_calculator_bfgs_stops_on_spike_without_stopiteration():
    atoms = MagicMock()
    atoms.get_forces.side_effect = [
        np.array([[2.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0]]),
        np.array([[60.0, 0.0, 0.0]]),
        np.array([[60.0, 0.0, 0.0]]),
    ]
    atoms.get_positions.return_value = np.zeros((1, 3))

    class FakeOpt:
        def __init__(self, *args, **kwargs) -> None:
            self._n = 0
            self._callbacks = []

        def attach(self, fn, interval=1):
            self._callbacks.append(fn)

        def get_number_of_steps(self):
            return self._n

        def irun(self, fmax, steps):
            for _ in range(2):
                self._n += 1
                for fn in self._callbacks:
                    fn()
                yield False

    config = HybridCalculatorMinimizeConfig(
        max_steps=50,
        quiet_bfgs=True,
        use_bfgs_line_search=False,
        verbose=True,
    )

    with patch("ase.optimize.BFGS", FakeOpt):
        opt, best_frame, stopped = _run_hybrid_calculator_bfgs(
            atoms,
            config,
            context_prefix="Test",
            initial_fmax_ev_a=2.0,
        )

    assert stopped is True
    assert best_frame.best_force_fmax == pytest.approx(2.0)
    assert opt.get_number_of_steps() == 2
