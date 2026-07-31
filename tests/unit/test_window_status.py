"""Failed-window triage for the hybrid umbrella Snakemake campaign.

A failed window writes a NaN ``wXXX.npz`` and exits 0, so Snakemake counts it as
done; the file has to be deleted before the window will be redone. And deleting
it is not enough on its own, because a whole-campaign ``--resume`` refills any
missing window from ``umbrella_snapshots.npz`` as a failed placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = (
    Path(__file__).resolve().parents[1].parent
    / "workflows"
    / "hybrid_umbrella_windows"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))

from window_status import (  # noqa: E402
    files_to_reset,
    scan_windows,
    window_log_relax_steps,
)

from mmml.umbrella.hybrid_windows import save_window_checkpoint  # noqa: E402


def _window(out: Path, wid: int, *, status: str, relax: int | None = None):
    n_frames, n_atoms = 3, 4
    finite = status == "ok"
    pos = np.zeros((n_frames, n_atoms, 3)) if finite else np.full((n_frames, n_atoms, 3), np.nan)
    cv = np.linspace(0.0, 1.0, n_frames) if finite else np.full(n_frames, np.nan)
    ene = np.zeros(n_frames) if finite else np.full(n_frames, np.nan)
    save_window_checkpoint(
        out,
        wid,
        status=status,
        positions=pos,
        cv=cv,
        energies=ene,
        energies_unbiased=ene.copy(),
        xi0=-1.3 + 0.1 * wid,
        k_ev_A2=6.505,
        fail_reason=None if finite else "non-finite state at step 2600/80000",
    )
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tail = "" if relax is None else f"  relax_steps={relax}  seed_max|F|_ML=3.10\n"
    (logs / f"window_w{wid:03d}.log").write_text(
        f"  window {wid + 1}/30  xi0=0.300  k=6.505  nsteps=80000\n{tail}",
        encoding="utf-8",
    )


def test_scan_reports_ok_failed_and_missing(tmp_path: Path):
    _window(tmp_path, 0, status="ok", relax=300)
    _window(tmp_path, 1, status="failed", relax=300)
    reports = scan_windows(tmp_path, [0, 1, 2])
    assert [r.status for r in reports] == ["ok", "failed", "missing"]
    assert reports[0].finite == pytest.approx(1.0)
    assert reports[1].finite == pytest.approx(0.0)
    assert "non-finite" in reports[1].fail_reason
    assert [r.needs_rerun for r in reports] == [False, True, True]


def test_scan_demotes_ok_window_carrying_nans(tmp_path: Path):
    """A stale checkpoint can claim ok while holding NaNs; trust the data."""
    save_window_checkpoint(
        tmp_path,
        0,
        status="ok",
        positions=np.zeros((2, 3, 3)),
        cv=np.array([np.nan, 0.5]),
        energies=np.zeros(2),
        energies_unbiased=np.zeros(2),
        xi0=0.3,
        k_ev_A2=6.505,
    )
    assert scan_windows(tmp_path, [0])[0].status == "failed"


def test_relax_steps_read_from_log(tmp_path: Path):
    _window(tmp_path, 4, status="ok", relax=137)
    assert window_log_relax_steps(tmp_path, 4) == 137


def test_relax_steps_none_for_prefix_run(tmp_path: Path):
    _window(tmp_path, 5, status="ok", relax=None)
    assert window_log_relax_steps(tmp_path, 5) is None
    assert window_log_relax_steps(tmp_path, 99) is None


def test_reset_failed_drops_only_failed_windows(tmp_path: Path):
    _window(tmp_path, 0, status="ok", relax=300)
    _window(tmp_path, 1, status="failed", relax=300)
    doomed = files_to_reset(scan_windows(tmp_path, [0, 1]), tmp_path)
    names = {p.name for p in doomed}
    assert "w001.npz" in names
    assert "w000.npz" not in names


def test_reset_removes_aggregate_that_would_restore_the_window(tmp_path: Path):
    """Without this, bootstrap_windows_from_snapshots undoes the deletion."""
    _window(tmp_path, 1, status="failed", relax=300)
    (tmp_path / "umbrella_snapshots.npz").write_bytes(b"")
    (tmp_path / "umbrella_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mbar").mkdir()
    (tmp_path / "mbar" / "status.json").write_text("{}", encoding="utf-8")
    names = {p.name for p in files_to_reset(scan_windows(tmp_path, [1]), tmp_path)}
    assert {"w001.npz", "umbrella_snapshots.npz", "umbrella_summary.json", "status.json"} == names


def test_reset_keeps_aggregate_when_no_window_is_dropped(tmp_path: Path):
    _window(tmp_path, 0, status="ok", relax=300)
    (tmp_path / "umbrella_snapshots.npz").write_bytes(b"")
    assert files_to_reset(scan_windows(tmp_path, [0]), tmp_path) == []


def test_reset_unrelaxed_targets_ok_windows_without_relaxation(tmp_path: Path):
    _window(tmp_path, 0, status="ok", relax=300)
    _window(tmp_path, 1, status="ok", relax=None)
    reports = scan_windows(tmp_path, [0, 1])
    assert files_to_reset(reports, tmp_path, reset_failed=True) == []
    names = {
        p.name
        for p in files_to_reset(reports, tmp_path, reset_failed=True, reset_unrelaxed=True)
    }
    assert "w001.npz" in names
    assert "w000.npz" not in names
