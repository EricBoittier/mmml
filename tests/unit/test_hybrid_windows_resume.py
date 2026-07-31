"""Per-window hybrid umbrella checkpoints and resume selection."""

from __future__ import annotations

import numpy as np

from mmml.umbrella.hybrid_windows import (
    bootstrap_windows_from_snapshots,
    load_all_window_arrays,
    load_window_checkpoint,
    save_window_checkpoint,
    select_windows_to_run,
    window_is_ok,
)
from mmml.umbrella.io import save_snapshots


def _ok_arrays(t=4, n=3):
    pos = np.zeros((t, n, 3), dtype=np.float64)
    pos[:, 0, 0] = np.linspace(1.0, 2.0, t)
    cv = np.linspace(-1.0, 1.0, t)
    e = np.linspace(-10.0, -9.0, t)
    return pos, cv, e, e - 0.1


def test_save_load_ok_checkpoint(tmp_path):
    pos, cv, e, u = _ok_arrays()
    save_window_checkpoint(
        tmp_path,
        2,
        status="ok",
        positions=pos,
        cv=cv,
        energies=e,
        energies_unbiased=u,
        xi0=0.5,
        k_ev_A2=6.5,
    )
    chk = load_window_checkpoint(tmp_path, 2)
    assert window_is_ok(chk)
    assert chk["window"] == 2
    assert chk["xi0"] == 0.5
    np.testing.assert_allclose(chk["positions"], pos)


def test_select_windows_resume_skips_ok_retries_failed(tmp_path):
    pos, cv, e, u = _ok_arrays()
    save_window_checkpoint(
        tmp_path, 0, status="ok", positions=pos, cv=cv, energies=e,
        energies_unbiased=u, xi0=-1.0, k_ev_A2=6.5,
    )
    save_window_checkpoint(
        tmp_path, 1, status="failed", positions=np.full_like(pos, np.nan),
        cv=np.full(4, np.nan), energies=np.full(4, np.nan),
        energies_unbiased=np.full(4, np.nan), xi0=0.0, k_ev_A2=6.5,
        fail_reason="nan",
    )
    # window 2 missing
    to_run, ok = select_windows_to_run(
        3, tmp_path, resume=True, resume_failed=True
    )
    assert ok == [0]
    assert to_run == [1, 2]

    to_run2, _ = select_windows_to_run(
        3, tmp_path, resume=True, resume_failed=False
    )
    assert to_run2 == [2]


def test_select_only_windows(tmp_path):
    to_run, ok = select_windows_to_run(
        5, tmp_path, resume=False, only_windows=(1, 3)
    )
    assert to_run == [1, 3]
    assert ok == []


def test_bootstrap_from_aggregated_snapshots(tmp_path):
    k, t, n = 3, 5, 2
    positions = np.zeros((k, t, n, 3))
    positions[0, :, 0, 0] = 1.0
    positions[1, :, 0, 0] = 2.0
    positions[2] = np.nan
    energies = np.ones((k, t))
    energies[2] = np.nan
    e_unb = energies - 0.5
    cv = np.zeros((k, t, 1))
    save_snapshots(
        tmp_path / "umbrella_snapshots.npz",
        positions=positions,
        Z=np.ones(n, dtype=np.int32),
        atom_i=0,
        atom_j=1,
        xi0=np.array([-1.0, 0.0, 1.0]),
        k_ev_A2=np.full(3, 6.5),
        temperature_K=300.0,
        dt_fs=0.25,
        cv_traj=cv,
        extra={
            "energies_ev": energies,
            "energies_unbiased_ev": e_unb,
            "failed_windows": np.array([2], dtype=np.int32),
        },
    )
    written = bootstrap_windows_from_snapshots(tmp_path, n_windows=3)
    assert written == [0, 1, 2]
    assert window_is_ok(load_window_checkpoint(tmp_path, 0))
    assert not window_is_ok(load_window_checkpoint(tmp_path, 2))

    pos, cv2, ene, unb, failed, reasons = load_all_window_arrays(
        tmp_path, 3, n_frames=t, n_atoms=n
    )
    assert failed == [2]
    assert pos.shape == (3, t, n, 3)
    assert np.all(np.isfinite(pos[0]))
