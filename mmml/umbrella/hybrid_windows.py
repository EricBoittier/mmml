"""Per-window checkpoints for hybrid umbrella resume.

Each window is written to ``output_dir/windows/wXXX.npz`` as soon as it finishes
(ok or failed). A later ``--resume`` run skips finished ok windows and re-runs
missing / failed ones, then reassembles ``umbrella_snapshots.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mmml.umbrella.io import SNAPSHOTS_NPZ, load_snapshots

WINDOWS_SUBDIR = "windows"


def windows_dir(output_dir: Path) -> Path:
    return Path(output_dir) / WINDOWS_SUBDIR


def window_npz_path(output_dir: Path, wid: int) -> Path:
    return windows_dir(output_dir) / f"w{int(wid):03d}.npz"


def save_window_checkpoint(
    output_dir: Path,
    wid: int,
    *,
    status: str,
    positions: np.ndarray,
    cv: np.ndarray,
    energies: np.ndarray,
    energies_unbiased: np.ndarray,
    xi0: float,
    k_ev_A2: float,
    fail_reason: str | None = None,
) -> Path:
    """Atomic-ish write of one window result (ok or failed)."""
    path = window_npz_path(output_dir, wid)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy appends ``.npz`` unless the path already ends with it — keep the
    # temp name ``*.tmp.npz`` so replace() finds the file we wrote.
    tmp = path.with_name(path.stem + ".tmp.npz")
    payload: dict[str, Any] = {
        "status": np.asarray(status),
        "window": np.int32(wid),
        "positions": np.asarray(positions, dtype=np.float64),
        "cv": np.asarray(cv, dtype=np.float64),
        "energies": np.asarray(energies, dtype=np.float64),
        "energies_unbiased": np.asarray(energies_unbiased, dtype=np.float64),
        "xi0": np.float64(xi0),
        "k_ev_A2": np.float64(k_ev_A2),
    }
    if fail_reason is not None:
        payload["fail_reason"] = np.asarray(fail_reason)
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return path


def load_window_checkpoint(output_dir: Path, wid: int) -> dict[str, Any] | None:
    path = window_npz_path(output_dir, wid)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {key: data[key] for key in data.files}
    status = out["status"]
    out["status"] = str(status.item() if getattr(status, "ndim", 0) == 0 else status)
    out["window"] = int(np.asarray(out["window"]).reshape(-1)[0])
    out["xi0"] = float(np.asarray(out["xi0"]).reshape(-1)[0])
    out["k_ev_A2"] = float(np.asarray(out["k_ev_A2"]).reshape(-1)[0])
    if "fail_reason" in out:
        fr = out["fail_reason"]
        out["fail_reason"] = str(fr.item() if getattr(fr, "ndim", 0) == 0 else fr)
    return out


def window_is_ok(chk: dict[str, Any] | None) -> bool:
    if chk is None or chk.get("status") != "ok":
        return False
    pos = np.asarray(chk["positions"])
    ene = np.asarray(chk["energies"])
    return bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(ene)))


def bootstrap_windows_from_snapshots(
    output_dir: Path,
    *,
    n_windows: int | None = None,
) -> list[int]:
    """Split an existing ``umbrella_snapshots.npz`` into per-window files.

    Returns the list of window indices written. Used so an interrupted run that
    only has the aggregated NPZ can still ``--resume``.
    """
    snap_path = Path(output_dir) / SNAPSHOTS_NPZ
    if not snap_path.is_file():
        return []
    snap = load_snapshots(snap_path)
    positions = np.asarray(snap["positions"], dtype=np.float64)
    k = int(positions.shape[0]) if n_windows is None else int(n_windows)
    k = min(k, int(positions.shape[0]))
    cv_traj = snap.get("cv_traj")
    energies = snap.get("energies_ev")
    e_unb = snap.get("energies_unbiased_ev")
    xi0 = np.asarray(snap["xi0"], dtype=np.float64)
    k_arr = np.asarray(snap["k_ev_A2"], dtype=np.float64)
    failed = set()
    if "failed_windows" in snap:
        failed.update(int(x) for x in np.asarray(snap["failed_windows"]).reshape(-1))
    written: list[int] = []
    for wid in range(k):
        if window_npz_path(output_dir, wid).is_file():
            continue
        pos = positions[wid]
        if energies is not None:
            ene = np.asarray(energies[wid], dtype=np.float64)
        else:
            ene = np.full(pos.shape[0], np.nan)
        if e_unb is not None:
            unb = np.asarray(e_unb[wid], dtype=np.float64)
        else:
            unb = ene.copy()
        if cv_traj is not None:
            cv = np.asarray(cv_traj[wid], dtype=np.float64).reshape(-1)
        else:
            cv = np.full(pos.shape[0], np.nan)
        ok = wid not in failed and np.all(np.isfinite(pos)) and np.all(np.isfinite(ene))
        save_window_checkpoint(
            output_dir,
            wid,
            status="ok" if ok else "failed",
            positions=pos,
            cv=cv,
            energies=ene,
            energies_unbiased=unb,
            xi0=float(xi0[wid]) if wid < len(xi0) else float("nan"),
            k_ev_A2=float(k_arr[wid]) if wid < len(k_arr) else float("nan"),
            fail_reason=None if ok else "imported from aggregated snapshots as failed/non-finite",
        )
        written.append(wid)
    return written


def select_windows_to_run(
    k_windows: int,
    output_dir: Path,
    *,
    resume: bool,
    resume_failed: bool = True,
    only_windows: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(to_run, already_ok)`` window index lists.

    - ``resume=False``: run every window in ``only_windows`` (or all).
    - ``resume=True``: skip checkpoints with ``status=ok``; re-run missing
      windows; re-run ``status=failed`` when ``resume_failed``.
    """
    if only_windows is not None and len(only_windows) > 0:
        candidates = sorted({int(w) for w in only_windows})
        for w in candidates:
            if w < 0 or w >= k_windows:
                raise ValueError(
                    f"window index {w} out of range for n_windows={k_windows}"
                )
    else:
        candidates = list(range(k_windows))

    if not resume:
        return candidates, []

    to_run: list[int] = []
    already_ok: list[int] = []
    for wid in candidates:
        chk = load_window_checkpoint(output_dir, wid)
        if window_is_ok(chk):
            already_ok.append(wid)
            continue
        if chk is not None and chk.get("status") == "failed" and not resume_failed:
            already_ok.append(wid)  # keep failed placeholder; do not rerun
            continue
        to_run.append(wid)
    return to_run, already_ok


def load_all_window_arrays(
    output_dir: Path,
    k_windows: int,
    *,
    n_frames: int,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, str]]:
    """Assemble ``(K, T, …)`` arrays from per-window checkpoints."""
    positions = np.full((k_windows, n_frames, n_atoms, 3), np.nan, dtype=np.float64)
    cv = np.full((k_windows, n_frames), np.nan, dtype=np.float64)
    energies = np.full((k_windows, n_frames), np.nan, dtype=np.float64)
    e_unb = np.full((k_windows, n_frames), np.nan, dtype=np.float64)
    failed: list[int] = []
    reasons: dict[int, str] = {}
    for wid in range(k_windows):
        chk = load_window_checkpoint(output_dir, wid)
        if chk is None:
            failed.append(wid)
            reasons[wid] = "missing window checkpoint"
            continue
        pos = np.asarray(chk["positions"], dtype=np.float64)
        c = np.asarray(chk["cv"], dtype=np.float64).reshape(-1)
        e = np.asarray(chk["energies"], dtype=np.float64).reshape(-1)
        u = np.asarray(chk["energies_unbiased"], dtype=np.float64).reshape(-1)
        t = min(n_frames, pos.shape[0], c.shape[0], e.shape[0], u.shape[0])
        positions[wid, :t] = pos[:t]
        cv[wid, :t] = c[:t]
        energies[wid, :t] = e[:t]
        e_unb[wid, :t] = u[:t]
        if not window_is_ok(chk):
            failed.append(wid)
            reasons[wid] = str(chk.get("fail_reason") or chk.get("status") or "failed")
    return positions, cv, energies, e_unb, failed, reasons
