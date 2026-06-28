"""Trajectory diagnostics (RDF, VACF) from campaign DCDs and handoff NPZ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def find_dcd_files(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    dcds = [p for p in out_dir.rglob("*.dcd") if p.is_file() and p.stat().st_size > 64]
    dcds.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dcds


def find_handoff_npz(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    files = [p for p in out_dir.rglob("handoff/state.npz") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _pairwise_distances(pos: np.ndarray) -> np.ndarray:
    """Upper-triangle pairwise distances for one frame (N,3)."""
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    iu = np.triu_indices(len(pos), k=1)
    return dist[iu]


def compute_rdf_g(
    positions_frames: list[np.ndarray],
    *,
    r_max: float = 12.0,
    n_bins: int = 120,
) -> dict[str, Any]:
    """Simple center-of-mass-agnostic all-pairs RDF from trajectory frames."""
    if not positions_frames:
        return {"n_frames": 0, "bins_A": [], "g_r": []}

    edges = np.linspace(0.0, r_max, n_bins + 1, dtype=np.float64)
    hist = np.zeros(n_bins, dtype=np.float64)
    n_samples = 0
    for pos in positions_frames:
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2:
            continue
        dists = _pairwise_distances(pos)
        hist += np.histogram(dists, bins=edges)[0]
        n_samples += 1

    if n_samples == 0:
        return {"n_frames": 0, "bins_A": [], "g_r": []}

    hist /= float(n_samples)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = float(np.mean([len(f) for f in positions_frames])) / (
        (4.0 / 3.0) * np.pi * r_max**3
    )
    norm = density * shell_vol * n_samples
    g_r = np.divide(hist, norm, out=np.zeros_like(hist), where=norm > 0)
    return {
        "n_frames": n_samples,
        "r_max_A": r_max,
        "bins_A": centers.tolist(),
        "g_r": g_r.tolist(),
        "peak_r_A": float(centers[int(np.argmax(g_r))]) if g_r.size else None,
    }


def compute_vacf_from_velocities(vel: np.ndarray) -> dict[str, Any]:
    """Normalized VACF for (N,3) velocity snapshot (single-frame proxy)."""
    v = np.asarray(vel, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2:
        return {"n_atoms": int(v.shape[0]) if v.ndim == 2 else 0, "vacf": []}
    v_centered = v - v.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(v_centered, axis=1)
    return {
        "n_atoms": int(v.shape[0]),
        "speed_mean_A_ps": float(np.mean(norms)),
        "speed_std_A_ps": float(np.std(norms)),
        "vacf": [1.0],
    }


def compute_vacf_time_series(
    velocities_frames: list[np.ndarray],
    *,
    max_lag: int | None = None,
) -> dict[str, Any]:
    """VACF from per-frame COM velocity time series."""
    if len(velocities_frames) < 2:
        return {"n_frames": len(velocities_frames), "lags": [], "vacf": []}

    v = np.stack([np.mean(np.asarray(f, dtype=np.float64), axis=0) for f in velocities_frames])
    n = len(v)
    max_lag = min(max_lag or n // 2, n - 1)
    v0 = v - v.mean(axis=0, keepdims=True)
    vacf = []
    for lag in range(max_lag + 1):
        a = v0[: n - lag]
        b = v0[lag:]
        num = float(np.sum(a * b))
        den = float(np.sum(v0 * v0)) + 1e-30
        vacf.append(num / den)
    return {"n_frames": n, "lags": list(range(max_lag + 1)), "vacf": vacf}


def load_dcd_frames(
    dcd_path: Path,
    *,
    stride: int = 1,
    max_frames: int = 200,
) -> list[np.ndarray]:
    try:
        from ase.io import read
    except ImportError:
        return []

    try:
        traj = read(str(dcd_path), index=":", format="dcd")
    except Exception:
        try:
            traj = read(str(dcd_path), index=":")
        except Exception:
            return []

    if not isinstance(traj, list):
        traj = [traj]
    frames: list[np.ndarray] = []
    for i, atoms in enumerate(traj):
        if i % stride != 0:
            continue
        frames.append(np.asarray(atoms.get_positions(), dtype=np.float64))
        if len(frames) >= max_frames:
            break
    return frames


def analyze_cell_trajectories(
    out_dir: Path,
    *,
    stride: int = 5,
    max_frames: int = 100,
) -> dict[str, Any]:
    """Collect RDF/VACF/handoff stats for one cell output directory."""
    result: dict[str, Any] = {"out_dir": str(out_dir), "dcd_files": [], "handoffs": []}

    dcds = find_dcd_files(out_dir)
    result["dcd_files"] = [str(p.relative_to(out_dir)) for p in dcds[:8]]

    if dcds:
        frames = load_dcd_frames(dcds[0], stride=stride, max_frames=max_frames)
        result["trajectory"] = {
            "source": str(dcds[0].relative_to(out_dir)),
            "n_frames_read": len(frames),
            "rdf": compute_rdf_g(frames),
        }
    else:
        result["trajectory"] = {"n_frames_read": 0, "rdf": {"n_frames": 0}}

    for hp in find_handoff_npz(out_dir)[:3]:
        rec: dict[str, Any] = {"path": str(hp.relative_to(out_dir))}
        try:
            data = np.load(hp, allow_pickle=False)
            rec["keys"] = list(data.files)
            if "velocities" in data:
                rec["velocity_stats"] = compute_vacf_from_velocities(data["velocities"])
            if "positions" in data:
                pos = np.asarray(data["positions"], dtype=np.float64)
                rec["n_atoms"] = int(pos.shape[0])
        except Exception as exc:
            rec["error"] = str(exc)
        result["handoffs"].append(rec)

    return result


def plot_rdf_png(rdf: dict[str, Any], path: Path, *, title: str) -> bool:
    bins = rdf.get("bins_A") or []
    g_r = rdf.get("g_r") or []
    if len(bins) < 2 or len(g_r) < 2:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bins, g_r, color="#005384", lw=1.0)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_trajectory_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
