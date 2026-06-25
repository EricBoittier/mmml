"""Hybrid calculator pre-minimize guards (BFGS spike abort, best-frame restore)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize import (
    HybridCalculatorMinimizeConfig,
    _BestMinimizationFrame,
    _run_hybrid_calculator_bfgs,
    spike_fmax_limit_ev_a,
)


def test_spike_fmax_limit_uses_factor_and_floor():
    assert spike_fmax_limit_ev_a(2.27, factor=4.0, floor_ev_a=15.0) == pytest.approx(15.0)
    assert spike_fmax_limit_ev_a(5.0, factor=4.0, floor_ev_a=15.0) == pytest.approx(20.0)


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


def test_run_hybrid_calculator_bfgs_aborts_on_spike_and_restores_best():
  atoms = MagicMock()
  forces = [
      np.array([[2.0, 0.0, 0.0]]),
      np.array([[6.0, 0.0, 0.0]]),
      np.array([[900.0, 0.0, 0.0]]),
  ]
  atoms.get_forces.side_effect = forces
  atoms.get_positions.side_effect = [
      np.zeros((1, 3)),
      np.ones((1, 3)),
      np.full((1, 3), 99.0),
  ]

  class FakeOpt:
      def __init__(self, *args, **kwargs) -> None:
          self._n = 0
          self.attach_calls = []

      def attach(self, fn, interval=1):
          self.attach_calls.append(fn)

      def get_number_of_steps(self):
          return self._n

      def run(self, fmax, steps):
          for fn in self.attach_calls:
              if fn.__name__ == "_record_step":
                  self._n += 1
                  fn()
              elif fn.__name__ == "_abort_on_spike":
                  fn()

  config = HybridCalculatorMinimizeConfig(
      max_steps=50,
      quiet_bfgs=True,
      use_bfgs_line_search=False,
      fmax_spike_factor=4.0,
      fmax_spike_floor_ev_a=15.0,
  )

  with patch(
      "mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize.BFGS",
      FakeOpt,
      create=True,
  ), patch(
      "mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize.BfgsOptimizer",
      FakeOpt,
      create=True,
  ):
      import mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize as cm

      with patch.object(cm, "BfgsOptimizer", FakeOpt, create=True):
          pass

  # Patch at import site inside _run_hybrid_calculator_bfgs
  fake_module = MagicMock()
  fake_module.BFGS = FakeOpt

  with patch.dict(
      "sys.modules",
      {"ase.optimize": fake_module, "ase.optimize.bfgslinesearch": fake_module},
  ):
      import importlib

      import mmml.interfaces.pycharmmInterface.mlpot.calculator_minimize as cm_mod

      importlib.reload(cm_mod)
      opt, best_frame, stopped = cm_mod._run_hybrid_calculator_bfgs(
          atoms,
          config,
          context_prefix="Test",
          initial_fmax_ev_a=2.0,
      )

  assert stopped is True
  assert best_frame.best_force_fmax == pytest.approx(2.0)
