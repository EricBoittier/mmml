"""Post-mortem trace for umbrella windows that abort non-finite.

A blown-up window used to leave only a step number: the driver's progress line
prints ``step N/80000`` with no energy or temperature, and the checkpoint it
writes is entirely NaN. The energy/kinetic series and the last few geometries
are the only evidence of what went wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mmml.md.drivers import NonFiniteStateError
from mmml.umbrella.hybrid import save_failure_trace


def _error(n_frames: int = 6, n_atoms: int = 4) -> NonFiniteStateError:
    frames = [np.full((n_atoms, 3), float(i)) for i in range(n_frames)]
    frames[-1][:] = np.nan
    energies = [-10.0 + i for i in range(n_frames)]
    energies[-1] = float("nan")
    return NonFiniteStateError(
        "umbrella_hybrid_w021: non-finite state at step 600/80000 (E=nan, K=nan).",
        step=600,
        n_steps=80000,
        positions=frames,
        energies=energies,
        kinetic_energies=[1.0] * n_frames,
    )


def test_non_finite_error_is_a_runtime_error():
    """Existing ``except RuntimeError`` handlers must keep catching it."""
    assert issubclass(NonFiniteStateError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise _error()


def test_trace_records_the_full_energy_series(tmp_path: Path):
    path = save_failure_trace(tmp_path, 21, _error())
    assert path is not None and path.name == "w021.trace.npz"
    data = np.load(path, allow_pickle=False)
    assert data["energies"].shape == (6,)
    assert data["kinetic_energies"].shape == (6,)
    assert int(data["step"]) == 600
    assert int(data["n_steps"]) == 80000
    assert int(data["n_frames_recorded"]) == 6
    np.testing.assert_allclose(data["energies"][:5], [-10.0, -9.0, -8.0, -7.0, -6.0])


def test_trace_keeps_only_the_tail_of_the_trajectory(tmp_path: Path):
    """A window dying at step 52000 must not dump hundreds of megabytes."""
    path = save_failure_trace(tmp_path, 7, _error(n_frames=260), keep_frames=10)
    data = np.load(path, allow_pickle=False)
    assert data["positions_tail"].shape == (10, 4, 3)
    assert int(data["n_frames_recorded"]) == 260
    # The tail must end at the blow-up, not start at the beginning of the run.
    assert np.isnan(data["positions_tail"][-1]).all()
    np.testing.assert_allclose(data["positions_tail"][0], np.full((4, 3), 250.0))


def test_trace_written_next_to_the_window_checkpoint(tmp_path: Path):
    from mmml.umbrella.hybrid_windows import window_npz_path

    path = save_failure_trace(tmp_path, 21, _error())
    assert path.parent == window_npz_path(tmp_path, 21).parent


def test_no_trace_when_the_error_carries_nothing(tmp_path: Path):
    """Plain RuntimeErrors from elsewhere must not produce an empty file."""
    assert save_failure_trace(tmp_path, 3, RuntimeError("boom")) is None
    assert not (tmp_path / "windows").exists()


def test_trace_survives_a_failure_before_any_frame_was_recorded(tmp_path: Path):
    exc = NonFiniteStateError(
        "died early", step=200, n_steps=80000, positions=[], energies=[-1.0]
    )
    path = save_failure_trace(tmp_path, 28, exc)
    data = np.load(path, allow_pickle=False)
    assert data["positions_tail"].size == 0
    assert data["energies"].tolist() == [-1.0]
