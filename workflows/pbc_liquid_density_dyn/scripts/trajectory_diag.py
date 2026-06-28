"""Trajectory diagnostics (pair-type RDF, VACF) from CHARMM DCD + PSF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mmml.utils.dcd_reader import read_dcd_trajectory
from mmml.utils.psf_reader import read_psf_atom_types


def pair_label(type_a: str, type_b: str) -> str:
    a, b = sorted([str(type_a), str(type_b)])
    return f"{a}__{b}"


def find_dcd_files(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    dcds = [p for p in out_dir.rglob("*.dcd") if p.is_file() and p.stat().st_size > 64]
    dcds.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dcds


def find_psf_files(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    psfs = [p for p in out_dir.rglob("model.psf") if p.is_file()]
    psfs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return psfs


def find_handoff_npz(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    files = [p for p in out_dir.rglob("handoff/state.npz") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def box_lengths_from_handoff(out_dir: Path) -> np.ndarray | None:
    for hp in find_handoff_npz(out_dir):
        try:
            data = np.load(hp, allow_pickle=False)
            if "cell" not in data:
                continue
            cell = np.asarray(data["cell"], dtype=np.float64)
            if cell.shape == (3, 3):
                return np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=np.float64)
        except (OSError, ValueError):
            continue
    return None


def _mic_distances(pos_a: np.ndarray, pos_b: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image distances between two coordinate sets."""
    d = pos_a[:, None, :] - pos_b[None, :, :]
    box = np.asarray(box, dtype=np.float64).reshape(1, 1, 3)
    d -= box * np.round(d / box)
    return np.linalg.norm(d, axis=-1)


def _type_pair_indices(atom_types: np.ndarray) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Map sorted (type_a, type_b) -> (idx_a, idx_b) index arrays."""
    types = np.asarray(atom_types, dtype=str)
    unique = sorted(set(types.tolist()))
    by_type = {t: np.flatnonzero(types == t) for t in unique}
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for i, ta in enumerate(unique):
        for tb in unique[i:]:
            out[(ta, tb)] = (by_type[ta], by_type[tb])
    return out


def compute_pair_rdf(
    positions_frames: np.ndarray,
    atom_types: np.ndarray,
    *,
    box_lengths: np.ndarray | None = None,
    r_max: float = 12.0,
    n_bins: int = 120,
) -> dict[str, Any]:
    """Partial g(r) for every unordered MM atom-type pair."""
    pos_all = np.asarray(positions_frames, dtype=np.float64)
    if pos_all.ndim != 3 or pos_all.shape[0] == 0:
        return {"n_frames": 0, "pairs": {}, "atom_types": []}

    types = np.asarray(atom_types, dtype=str)
    if types.shape[0] != pos_all.shape[1]:
        raise ValueError(
            f"PSF atom count ({types.shape[0]}) != DCD atom count ({pos_all.shape[1]})"
        )

    pair_idx = _type_pair_indices(types)
    edges = np.linspace(0.0, r_max, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    n_frames = pos_all.shape[0]

    pairs_out: dict[str, Any] = {}
    for (ta, tb), (idx_a, idx_b) in pair_idx.items():
        label = pair_label(ta, tb)
        hist = np.zeros(n_bins, dtype=np.float64)
        n_pair_samples = 0
        same = ta == tb
        for frame in range(n_frames):
            pos = pos_all[frame]
            if box_lengths is None:
                span = pos.max(axis=0) - pos.min(axis=0)
                box = np.maximum(span * 1.05, 1.0)
            else:
                box = np.asarray(box_lengths, dtype=np.float64)

            pa = pos[idx_a]
            pb = pos[idx_b]
            if same:
                if len(idx_a) < 2:
                    continue
                dists = _mic_distances(pa, pb, box)
                iu = np.triu_indices(len(idx_a), k=1)
                d = dists[iu]
                n_pair_samples += d.size
            else:
                dists = _mic_distances(pa, pb, box)
                d = dists.ravel()
                n_pair_samples += d.size
            if d.size:
                hist += np.histogram(d, bins=edges)[0]

        if n_pair_samples == 0:
            continue

        if box_lengths is not None:
            box_ref = np.asarray(box_lengths, dtype=np.float64)
        else:
            span = np.ptp(pos_all[0], axis=0)
            box_ref = np.maximum(span * 1.05, 1.0)
        volume = float(np.prod(box_ref))
        rho = len(types) / volume
        norm = (n_pair_samples / n_frames) * shell_vol * rho
        g_r = np.divide(hist / max(n_frames, 1), norm, out=np.zeros_like(hist), where=norm > 0)
        peak_i = int(np.argmax(g_r)) if g_r.size else 0
        pairs_out[label] = {
            "type_a": ta,
            "type_b": tb,
            "n_frames": n_frames,
            "n_pairs_sampled": int(n_pair_samples),
            "bins_A": centers.tolist(),
            "g_r": g_r.tolist(),
            "peak_r_A": float(centers[peak_i]) if g_r.size else None,
            "peak_g": float(g_r[peak_i]) if g_r.size else None,
        }

    return {
        "n_frames": n_frames,
        "n_atoms": int(pos_all.shape[1]),
        "atom_types": sorted(set(types.tolist())),
        "n_type_pairs": len(pairs_out),
        "pairs": pairs_out,
        "r_max_A": r_max,
    }


def compute_vacf_from_velocities(vel: np.ndarray) -> dict[str, Any]:
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


def load_dcd_trajectory(
    dcd_path: Path,
    *,
    stride: int = 1,
    max_frames: int = 200,
    require_complete: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    pos, meta = read_dcd_trajectory(
        dcd_path,
        max_frames=max_frames,
        frame_stride=max(1, int(stride)),
        require_complete=require_complete,
    )
    return pos, meta


def analyze_cell_trajectories(
    out_dir: Path,
    *,
    stride: int = 5,
    max_frames: int = 100,
    r_max: float = 12.0,
    n_bins: int = 120,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "out_dir": str(out_dir),
        "dcd_files": [],
        "psf_files": [],
        "handoffs": [],
    }

    dcds = find_dcd_files(out_dir)
    psfs = find_psf_files(out_dir)
    result["dcd_files"] = [str(p.relative_to(out_dir)) for p in dcds[:8]]
    result["psf_files"] = [str(p.relative_to(out_dir)) for p in psfs[:4]]

    if dcds and psfs:
        try:
            atom_types = read_psf_atom_types(psfs[0])
            pos, meta = load_dcd_trajectory(
                dcds[0], stride=stride, max_frames=max_frames, require_complete=False
            )
            box = box_lengths_from_handoff(out_dir)
            pair_rdf = compute_pair_rdf(
                pos,
                atom_types,
                box_lengths=box,
                r_max=r_max,
                n_bins=n_bins,
            )
            result["trajectory"] = {
                "source_dcd": str(dcds[0].relative_to(out_dir)),
                "source_psf": str(psfs[0].relative_to(out_dir)),
                "dcd_meta": meta,
                "box_lengths_A": box.tolist() if box is not None else None,
                "pair_rdf": pair_rdf,
            }
        except Exception as exc:
            result["trajectory"] = {"error": str(exc), "n_frames_read": 0}
    else:
        result["trajectory"] = {
            "n_frames_read": 0,
            "pair_rdf": {"n_frames": 0, "pairs": {}},
            "missing": {
                "dcd": not bool(dcds),
                "psf": not bool(psfs),
            },
        }

    for hp in find_handoff_npz(out_dir)[:3]:
        rec: dict[str, Any] = {"path": str(hp.relative_to(out_dir))}
        try:
            data = np.load(hp, allow_pickle=False)
            rec["keys"] = list(data.files)
            if "velocities" in data:
                rec["velocity_stats"] = compute_vacf_from_velocities(data["velocities"])
            if "positions" in data:
                rec["n_atoms"] = int(np.asarray(data["positions"]).shape[0])
        except Exception as exc:
            rec["error"] = str(exc)
        result["handoffs"].append(rec)

    return result


def plot_pair_rdf_png(
    pair_rdf: dict[str, Any],
    path: Path,
    *,
    title: str,
    max_panels: int = 6,
) -> bool:
    pairs: dict[str, Any] = pair_rdf.get("pairs") or {}
    if not pairs:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ranked = sorted(
        pairs.items(),
        key=lambda kv: float(kv[1].get("peak_g") or 0.0),
        reverse=True,
    )[:max_panels]
    n = len(ranked)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=11)
    for ax, (label, rec) in zip(axes.ravel(), ranked, strict=False):
        bins = rec.get("bins_A") or []
        g_r = rec.get("g_r") or []
        ax.plot(bins, g_r, lw=0.9, color="#005384")
        ax.set_title(label.replace("__", "–"), fontsize=9)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def plot_rdf_png(rdf: dict[str, Any], path: Path, *, title: str) -> bool:
    """Plot pair RDF panels (alias for collect pipeline)."""
    if "pairs" in rdf:
        return plot_pair_rdf_png(rdf, path, title=title)
    bins = rdf.get("bins_A") or []
    g_r = rdf.get("g_r") or []
    if len(bins) < 2:
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
