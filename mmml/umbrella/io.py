"""Snapshot / summary I/O for umbrella sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOTS_NPZ = "umbrella_snapshots.npz"
SUMMARY_JSON = "umbrella_summary.json"
BIN_MINIMA_TRAJ = "umbrella_bin_minima.traj"


def save_snapshots(
    path: Path,
    *,
    positions: np.ndarray,
    Z: np.ndarray,
    atom_i: int,
    atom_j: int,
    xi0: np.ndarray,
    k_ev_A2: np.ndarray,
    temperature_K: float,
    dt_fs: float,
    cv_traj: np.ndarray | None = None,
    checkpoint: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``umbrella_snapshots.npz``.

    ``positions`` shape: ``(K, N_frames, N_atoms, 3)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "positions": np.asarray(positions, dtype=np.float64),
        "Z": np.asarray(Z, dtype=np.int32),
        "atom_i": np.int32(atom_i),
        "atom_j": np.int32(atom_j),
        "xi0": np.asarray(xi0, dtype=np.float64),
        "k_ev_A2": np.asarray(k_ev_A2, dtype=np.float64),
        "temperature_K": np.float64(temperature_K),
        "dt_fs": np.float64(dt_fs),
    }
    if cv_traj is not None:
        payload["cv_traj"] = np.asarray(cv_traj, dtype=np.float64)
    if checkpoint is not None:
        payload["checkpoint"] = np.asarray(checkpoint)
    if extra:
        for key, value in extra.items():
            payload[key] = value
    np.savez_compressed(path, **payload)
    return path


def load_snapshots(path: Path) -> dict[str, Any]:
    """Load snapshot NPZ into a plain dict of arrays / scalars."""
    data = np.load(Path(path), allow_pickle=True)
    out: dict[str, Any] = {key: data[key] for key in data.files}
    out["atom_i"] = int(np.asarray(out["atom_i"]).reshape(-1)[0])
    out["atom_j"] = int(np.asarray(out["atom_j"]).reshape(-1)[0])
    out["temperature_K"] = float(np.asarray(out["temperature_K"]).reshape(-1)[0])
    out["dt_fs"] = float(np.asarray(out["dt_fs"]).reshape(-1)[0])
    if "checkpoint" in out:
        ck = out["checkpoint"]
        out["checkpoint"] = str(ck.item() if getattr(ck, "ndim", 0) == 0 else ck)
    if "cv_spec" in out:
        spec = out["cv_spec"]
        out["cv_spec"] = json.loads(
            str(spec.item() if getattr(spec, "ndim", 0) == 0 else spec)
        )
    if "fail_reasons" in out:
        fr = out["fail_reasons"]
        out["fail_reasons"] = json.loads(
            str(fr.item() if getattr(fr, "ndim", 0) == 0 else fr)
        )
    return out


def write_summary(path: Path, summary: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_mbar_into_summary(run_dir: Path, mbar_block: dict[str, Any]) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    summary_path = run_dir / SUMMARY_JSON
    if summary_path.is_file():
        summary = load_summary(summary_path)
    else:
        summary = {}
    summary["mbar"] = mbar_block
    return write_summary(summary_path, summary)
