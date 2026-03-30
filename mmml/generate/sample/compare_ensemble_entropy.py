#!/usr/bin/env python
"""
Compare gzip-normalized complexity and Shannon / JS / KL / MMD / Wasserstein
metrics between multi-frame XYZ ensembles (e.g. benchmark cc vs mesh dimers).

Pipeline: COM-center (mass-weighted), optional Kabsch alignment to a reference,
fixed-topology Z-matrix from a reference frame, inner-averaged SOAP, then
entropy-style metrics per file and optional pairwise cc vs mesh at each scale.

Optional (--geom-histograms): for each XYZ, density-normalized histograms of all
pairwise distances by element pair (C-C, C-H, …) and all vertex angles i–j–k
by type key center(neighbor1_neighbor2), plus overlay plots for cross-run comparison.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms as ASEAtoms
from ase.io import read as ase_read
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance

from mmml.interfaces.chemcoordInterface.interface import patch_chemcoord_for_pandas3

patch_chemcoord_for_pandas3()

import chemcoord as cc  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SPECIES = ["H", "C", "O"]
DEFAULT_R_CUT = 15.0
DEFAULT_N_MAX = 7
DEFAULT_L_MAX = 3
DEFAULT_SIGMA = 0.5


def make_plots(
    rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    """Generate summary PNG plots from per-file and pairwise metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning("matplotlib not available, skipping plots: %s", e)
        return

    df = pd.DataFrame(rows)
    if not df.empty:
        names = [Path(p).name for p in df["path"]]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

        axes[0, 0].bar(names, df["bits_per_frame_cart"], label="cart")
        axes[0, 0].bar(names, df["bits_per_frame_zmat"], alpha=0.7, label="zmat")
        axes[0, 0].bar(names, df["bits_per_frame_soap"], alpha=0.5, label="soap")
        axes[0, 0].set_title("Bits per frame (gzip)")
        axes[0, 0].tick_params(axis="x", rotation=40, labelsize=8)
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(names, df["shannon_pca2d_soap"], marker="o")
        axes[0, 1].set_title("Shannon entropy (SOAP PCA-2D)")
        axes[0, 1].tick_params(axis="x", rotation=40, labelsize=8)

        axes[1, 0].plot(names, df["shannon_cc_dist_1d"], marker="o", color="tab:green")
        axes[1, 0].set_title("Shannon entropy (C-C distance)")
        axes[1, 0].tick_params(axis="x", rotation=40, labelsize=8)

        axes[1, 1].bar(names, df["n_frames"], color="tab:orange")
        axes[1, 1].set_title("Frames per file")
        axes[1, 1].tick_params(axis="x", rotation=40, labelsize=8)

        out1 = outdir / "ensemble_entropy_per_file.png"
        fig.savefig(out1, dpi=180)
        plt.close(fig)
        print(f"Wrote {out1}")

    if pair_rows:
        pf = pd.DataFrame(pair_rows).sort_values("scale")
        fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

        axes2[0, 0].plot(pf["scale"], pf["js_pca2d"], marker="o")
        axes2[0, 0].set_title("JS divergence vs scale")
        axes2[0, 0].set_xlabel("scale")

        axes2[0, 1].plot(pf["scale"], pf["mmd_rbf"], marker="o", color="tab:purple")
        axes2[0, 1].set_title("MMD (RBF) vs scale")
        axes2[0, 1].set_xlabel("scale")

        axes2[1, 0].plot(pf["scale"], pf["w1_pca_dim0"], marker="o", label="PCA dim0")
        axes2[1, 0].plot(pf["scale"], pf["w1_pca_dim1"], marker="o", label="PCA dim1")
        axes2[1, 0].set_title("Wasserstein-1 vs scale")
        axes2[1, 0].set_xlabel("scale")
        axes2[1, 0].legend(fontsize=8)

        axes2[1, 1].plot(pf["scale"], pf["kl_pca2d_ab"], marker="o", label="KL(cc||mesh)")
        axes2[1, 1].plot(pf["scale"], pf["kl_pca2d_ba"], marker="o", label="KL(mesh||cc)")
        axes2[1, 1].set_title("KL divergence vs scale")
        axes2[1, 1].set_xlabel("scale")
        axes2[1, 1].legend(fontsize=8)

        out2 = outdir / "ensemble_entropy_pairwise.png"
        fig2.savefig(out2, dpi=180)
        plt.close(fig2)
        print(f"Wrote {out2}")


def _geom_pair_key(sym_a: str, sym_b: str) -> str:
    a, b = sorted((sym_a, sym_b))
    return f"{a}-{b}"


def _geom_angle_key(sym_center: str, sym_i: str, sym_k: str) -> str:
    a, b = sorted((sym_i, sym_k))
    return f"{sym_center}({a}_{b})"


def _angle_degrees(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    v1 = p_i - p_j
    v2 = p_k - p_j
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-14 or n2 < 1e-14:
        return float("nan")
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def collect_geometry_samples(
    frames: list[Any],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    All 2-body distances (unordered pairs) and 3-body angles (vertex at middle index
    pattern: every distinct pair of neighbors about each center atom).

    Keys: distance 'C-H', 'C-C', ...; angle 'C(C_H)', 'O(C_C)', ... (center first).
    """
    dist_by: dict[str, list[float]] = {}
    ang_by: dict[str, list[float]] = {}

    for atoms in frames:
        syms = atoms.get_chemical_symbols()
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        n = len(syms)
        if n < 2:
            continue

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                key = _geom_pair_key(syms[i], syms[j])
                dist_by.setdefault(key, []).append(d)

        if n < 3:
            continue
        for j in range(n):
            others = [i for i in range(n) if i != j]
            for a_idx in range(len(others)):
                for b_idx in range(a_idx + 1, len(others)):
                    i = others[a_idx]
                    k = others[b_idx]
                    ang = _angle_degrees(pos[i], pos[j], pos[k])
                    if not np.isfinite(ang):
                        continue
                    key = _geom_angle_key(syms[j], syms[i], syms[k])
                    ang_by.setdefault(key, []).append(ang)

    return dist_by, ang_by


def _subsample_frames(
    frames: list[Any], max_frames: int | None, rng: np.random.Generator
) -> list[Any]:
    if max_frames is None or len(frames) <= max_frames:
        return frames
    idx = rng.choice(len(frames), size=max_frames, replace=False)
    return [frames[i] for i in sorted(idx)]


def plot_geom_histograms_for_file(
    stem: str,
    dist_by: dict[str, list[float]],
    ang_by: dict[str, list[float]],
    out_dir: Path,
    *,
    bins_dist: int,
    bins_angle: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning("matplotlib not available, skipping geom histograms: %s", e)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for name, values in sorted(dist_by.items()):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.hist(arr, bins=bins_dist, density=True, color="steelblue", alpha=0.85)
        ax.set_title(f"{stem}: distances {name} (n={arr.size})")
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Probability density")
        fig.savefig(out_dir / f"dist_{name.replace('-', '_')}.png", dpi=160)
        plt.close(fig)

    for name, values in sorted(ang_by.items()):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        safe = (
            name.replace("(", "_")
            .replace(")", "")
            .replace("__", "_")
            .replace("/", "_")
        )
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.hist(arr, bins=bins_angle, range=(0.0, 180.0), density=True, color="darkorange", alpha=0.85)
        ax.set_title(f"{stem}: angles {name} (n={arr.size})")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Probability density")
        fig.savefig(out_dir / f"angle_{safe}.png", dpi=160)
        plt.close(fig)


def plot_geom_histograms_overlay(
    per_file: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]],
    out_root: Path,
    *,
    bins_dist: int,
    bins_angle: int,
) -> None:
    """One figure per distance/angle key; normalized densities for comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        logger.warning("matplotlib not available, skipping overlay plots: %s", e)
        return

    overlay = out_root / "geom_histograms_overlay"
    overlay.mkdir(parents=True, exist_ok=True)

    all_dist_keys: set[str] = set()
    all_ang_keys: set[str] = set()
    for _d, a in per_file.values():
        all_dist_keys.update(_d.keys())
        all_ang_keys.update(a.keys())

    for key in sorted(all_dist_keys):
        pooled: list[float] = []
        for _stem, (db, _ab) in per_file.items():
            pooled.extend(db.get(key, []))
        pool_arr = np.asarray(pooled, dtype=np.float64)
        pool_arr = pool_arr[np.isfinite(pool_arr)]
        if pool_arr.size == 0:
            continue
        lo, hi = float(np.min(pool_arr)), float(np.max(pool_arr))
        span = max(hi - lo, 1e-9)
        pad = max(0.02 * span, 1e-4)
        rlo, rhi = lo - pad, hi + pad

        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for stem, (db, _ab) in sorted(per_file.items()):
            vals = db.get(key)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            ax.hist(
                arr,
                bins=bins_dist,
                range=(rlo, rhi),
                density=True,
                histtype="step",
                linewidth=1.6,
                label=f"{stem} (n={arr.size})",
            )
        ax.set_title(f"Distance {key} (density-normalized, shared range)")
        ax.set_xlabel("Distance (Å)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7, loc="best")
        fig.savefig(overlay / f"dist_{key.replace('-', '_')}.png", dpi=160)
        plt.close(fig)

    for key in sorted(all_ang_keys):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for stem, (_db, ab) in sorted(per_file.items()):
            vals = ab.get(key)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            ax.hist(
                arr,
                bins=bins_angle,
                range=(0.0, 180.0),
                density=True,
                histtype="step",
                linewidth=1.6,
                label=f"{stem} (n={arr.size})",
            )
        safe = (
            key.replace("(", "_")
            .replace(")", "")
            .replace("__", "_")
            .replace("/", "_")
        )
        ax.set_title(f"Angle {key} (density-normalized)")
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7, loc="best")
        fig.savefig(overlay / f"angle_{safe}.png", dpi=160)
        plt.close(fig)

    print(f"Wrote overlay histograms under {overlay}")


def run_geometry_histograms(
    paths: list[Path],
    outdir: Path,
    *,
    max_frames: int | None,
    rng: np.random.Generator,
    bins_dist: int,
    bins_angle: int,
) -> None:
    per_file: dict[str, tuple[dict[str, list[float]], dict[str, list[float]]]] = {}
    base = outdir / "geom_histograms"

    for path in paths:
        if not path.exists():
            continue
        frames = load_xyz_frames(path)
        frames = _subsample_frames(frames, max_frames, rng)
        if not frames:
            logger.warning("Skipping geom histograms for empty/unreadable: %s", path)
            continue
        stem = path.stem
        dist_by, ang_by = collect_geometry_samples(frames)
        per_file[stem] = (dist_by, ang_by)
        plot_geom_histograms_for_file(
            stem,
            dist_by,
            ang_by,
            base / stem,
            bins_dist=bins_dist,
            bins_angle=bins_angle,
        )
        print(f"Wrote geometry histograms under {base / stem}")

    if len(per_file) > 1:
        plot_geom_histograms_overlay(
            per_file,
            outdir,
            bins_dist=bins_dist,
            bins_angle=bins_angle,
        )


def load_xyz_frames(path: Path) -> list[Any]:
    """Load all frames as ASE Atoms; empty or missing files return []."""
    if not path.exists() or path.stat().st_size == 0:
        if path.exists() and path.stat().st_size == 0:
            logger.warning("Skipping empty file: %s", path)
        return []
    try:
        traj = ase_read(str(path), index=":")
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return []

    if isinstance(traj, list):
        return traj
    return [traj]


def ase_to_chemcoord(atoms: Any) -> cc.Cartesian:
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    df = pd.DataFrame(
        {
            "atom": sym,
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
        }
    )
    return cc.Cartesian(df)


def com_center_positions(atoms: Any) -> np.ndarray:
    """Mass-weighted COM shift; returns new positions (n, 3)."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    masses = np.asarray(atoms.get_masses(), dtype=np.float64)
    com = np.sum(pos * masses[:, None], axis=0) / np.sum(masses)
    return pos - com


def kabsch_align_to_reference(
    positions: np.ndarray, ref_positions: np.ndarray
) -> np.ndarray:
    """Rotate positions (same atom order) to best align with ref_positions."""
    rot, _ = Rotation.align_vectors(ref_positions, positions)
    return rot.apply(positions)


def construction_table_from_zmat(zmat: Any) -> pd.DataFrame:
    c_table = zmat.loc[:, ["b", "a", "d"]].copy()
    for col in ("b", "a", "d"):
        c_table[col] = c_table[col].astype(object)
    return c_table


def zmat_internal_values(
    cart: cc.Cartesian, c_table: pd.DataFrame
) -> np.ndarray:
    zm = cart.get_zmat(c_table)
    vals = np.asarray(
        zm.loc[:, ["bond", "angle", "dihedral"]].to_numpy(dtype=np.float64),
        dtype=np.float64,
    ).ravel()
    return vals


def gzip_compressed_size_bytes(payload: bytes, level: int = 9) -> int:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as gz:
        gz.write(payload)
    return buf.tell()


def bytes_float64(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr, dtype=np.float64).tobytes()


def shannon_entropy_from_probs(p: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > eps]
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p + eps)))


def inverse_participation_ratio(p: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > eps]
    p = p / np.sum(p)
    return float(1.0 / np.sum(p**2))


def pca_2d_joint(
    X1: np.ndarray, X2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Joint PCA on stacked (X1, X2); returns Z1, Z2 (n x 2), mean, V[:, :2]."""
    X = np.vstack([X1, X2])
    mu = X.mean(axis=0)
    Xc = X - mu
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    v = vt.T[:, :2]
    Z1 = (X1 - mu) @ v
    Z2 = (X2 - mu) @ v
    return Z1, Z2, mu, v


def histogram2d_probs(
    Z: np.ndarray,
    bins: int,
    *,
    range_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple]:
    """Return normalized 2D histogram probabilities and (xedges, yedges)."""
    x, y = Z[:, 0], Z[:, 1]
    if range_xy is None:
        h, xe, ye = np.histogram2d(x, y, bins=bins)
    else:
        h, xe, ye = np.histogram2d(x, y, bins=bins, range=range_xy)
    p = h.astype(np.float64).ravel()
    s = p.sum()
    if s <= 0:
        return p, (xe, ye)
    return p / s, (xe, ye)


def joint_pca_histogram_entropy(
    Z: np.ndarray, bins: int = 32
) -> tuple[float, float]:
    """Single-sample entropy and IPR on 2D PCA of Z (standardized columns)."""
    if len(Z) < 2:
        return np.nan, np.nan
    mu = Z.mean(axis=0)
    sig = Z.std(axis=0) + 1e-12
    Zs = (Z - mu) / sig
    _, _, vt = np.linalg.svd(Zs, full_matrices=False)
    v = vt.T[:, :2]
    Z2 = Zs @ v
    p, _ = histogram2d_probs(Z2, bins=bins)
    return shannon_entropy_from_probs(p), inverse_participation_ratio(p)


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("KL: p and q must have the same shape")
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    return float(np.sum(p * np.log2((p + eps) / (q + eps))))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def pairwise_soap_metrics(
    soap_a: np.ndarray,
    soap_b: np.ndarray,
    *,
    bins: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Subsample to equal N, joint PCA + shared 2D histogram; JS, KL, MMD, W1."""
    n = min(len(soap_a), len(soap_b))
    if n < 2:
        return {
            "js_pca2d": np.nan,
            "kl_pca2d_ab": np.nan,
            "kl_pca2d_ba": np.nan,
            "mmd_rbf": np.nan,
            "w1_pca_dim0": np.nan,
            "w1_pca_dim1": np.nan,
            "n_subsample": float(n),
        }

    ia = rng.choice(len(soap_a), size=n, replace=False)
    ib = rng.choice(len(soap_b), size=n, replace=False)
    Xa = soap_a[ia].astype(np.float64)
    Xb = soap_b[ib].astype(np.float64)

    X = np.vstack([Xa, Xb])
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-12
    Xa_s = (Xa - mu) / sig
    Xb_s = (Xb - mu) / sig

    Z1, Z2, _, _ = pca_2d_joint(Xa_s, Xb_s)

    # Shared range for histograms
    zmin = np.minimum(Z1.min(axis=0), Z2.min(axis=0))
    zmax = np.maximum(Z1.max(axis=0), Z2.max(axis=0))
    pad = 1e-5 + 0.1 * (zmax - zmin)
    range_xy = ((zmin[0] - pad[0], zmax[0] + pad[0]), (zmin[1] - pad[1], zmax[1] + pad[1]))

    pa, _ = histogram2d_probs(Z1, bins=bins, range_xy=range_xy)
    pb, _ = histogram2d_probs(Z2, bins=bins, range_xy=range_xy)
    pa = pa / (pa.sum() + 1e-15)
    pb = pb / (pb.sum() + 1e-15)

    js = jensen_shannon_divergence(pa, pb)
    kl_ab = kl_divergence(pa, pb)
    kl_ba = kl_divergence(pb, pa)

    # MMD with RBF kernel (median heuristic on pairwise distances)
    gamma = mmd_rbf_gamma(Xa_s, Xb_s, rng=rng)
    mmd = mmd_rbf_value(Xa_s, Xb_s, gamma=gamma)

    w0 = wasserstein_distance(Z1[:, 0], Z2[:, 0])
    w1 = wasserstein_distance(Z1[:, 1], Z2[:, 1])

    return {
        "js_pca2d": float(js),
        "kl_pca2d_ab": float(kl_ab),
        "kl_pca2d_ba": float(kl_ba),
        "mmd_rbf": float(mmd),
        "w1_pca_dim0": float(w0),
        "w1_pca_dim1": float(w1),
        "n_subsample": float(n),
    }


def mmd_rbf_gamma(X: np.ndarray, Y: np.ndarray, *, rng: np.random.Generator) -> float:
    """Median heuristic for RBF kernel width."""
    Z = np.vstack([X, Y])
    n = min(len(Z), 500)
    idx = rng.choice(len(Z), size=n, replace=False)
    Zs = Z[idx]
    d2 = np.sum((Zs[:, None, :] - Zs[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(n, k=1)
    med = np.median(np.sqrt(d2[iu]))
    if med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med**2)


def mmd_rbf_value(X: np.ndarray, Y: np.ndarray, *, gamma: float) -> float:
    """Unbiased MMD^2 approximation (average diagonal removed)."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return np.nan

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))

    xx = rbf(X, X)
    yy = rbf(Y, Y)
    xy = rbf(X, Y)
    xx = (np.sum(xx) - np.trace(xx)) / (n * (n - 1))
    yy = (np.sum(yy) - np.trace(yy)) / (m * (m - 1))
    xy_mean = np.mean(xy)
    return float(xx + yy - 2 * xy_mean)


def inter_fragment_c_c_distance(positions: np.ndarray) -> float:
    """First C atom (index 0) to second C (index 6) for 12-atom methanol dimer."""
    if len(positions) < 12:
        return float("nan")
    return float(np.linalg.norm(positions[0] - positions[6]))


def _make_soap(
    species: list[str],
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
):
    from dscribe.descriptors import SOAP

    return SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        average="inner",
    )


def process_ensemble(
    path: Path,
    *,
    c_table: pd.DataFrame,
    ref_positions_com: np.ndarray | None,
    kabsch: bool,
    soap_engine: Any | None,
    max_frames: int | None,
    rng: np.random.Generator,
    hist_bins: int,
) -> dict[str, Any]:
    frames = load_xyz_frames(path)
    if not frames:
        z = np.zeros((0, 0), dtype=np.float64)
        return {
            "path": str(path),
            "n_frames": 0,
            "zmat_failures": 0,
            "gzip_bytes_cart": 0,
            "gzip_bytes_zmat": 0,
            "gzip_bytes_soap": 0,
            "bits_per_frame_cart": np.nan,
            "bits_per_frame_zmat": np.nan,
            "bits_per_frame_soap": np.nan,
            "shannon_pca2d_soap": np.nan,
            "ipr_pca2d_soap": np.nan,
            "shannon_cc_dist_1d": np.nan,
            "soap_dim": 0,
            "soap_array": z,
        }

    if max_frames is not None and len(frames) > max_frames:
        idx = rng.choice(len(frames), size=max_frames, replace=False)
        frames = [frames[i] for i in sorted(idx)]

    n_frames = len(frames)

    cart_bytes = bytearray()
    zmat_bytes = bytearray()
    soap_rows: list[np.ndarray] = []
    cc_dists: list[float] = []
    zmat_fail = 0

    for atoms in frames:
        pos = com_center_positions(atoms)
        if kabsch and ref_positions_com is not None:
            pos = kabsch_align_to_reference(pos, ref_positions_com)

        cart_bytes.extend(bytes_float64(pos.ravel()))

        cart = ase_to_chemcoord(atoms)
        cart.loc[:, ["x", "y", "z"]] = pos
        try:
            zv = zmat_internal_values(cart, c_table)
            zmat_bytes.extend(bytes_float64(zv))
        except Exception as e:
            zmat_fail += 1
            logger.debug("Z-matrix failed for frame in %s: %s", path, e)

        cc_dists.append(inter_fragment_c_c_distance(pos))

        if soap_engine is not None:
            pos = atoms.get_positions()
            masses = np.asarray(atoms.get_masses(), dtype=np.float64)
            com = np.sum(pos * masses[:, None], axis=0) / np.sum(masses)
            pos_c = pos - com
            if kabsch and ref_positions_com is not None:
                pos_c = kabsch_align_to_reference(pos_c, ref_positions_com)

            a = ASEAtoms(
                symbols=atoms.get_chemical_symbols(),
                positions=pos_c,
            )
            desc = soap_engine.create(a)
            soap_rows.append(np.asarray(desc, dtype=np.float64).ravel())

    soap_arr = np.vstack(soap_rows) if soap_rows else np.zeros((0, 0))

    gzip_cart = gzip_compressed_size_bytes(bytes(cart_bytes))
    gzip_zmat = gzip_compressed_size_bytes(bytes(zmat_bytes)) if zmat_bytes else 0
    gzip_soap = (
        gzip_compressed_size_bytes(bytes_float64(soap_arr.ravel()))
        if soap_arr.size
        else 0
    )

    bits_per_frame_cart = (gzip_cart * 8.0) / n_frames if n_frames else np.nan
    bits_per_frame_zmat = (gzip_zmat * 8.0) / n_frames if n_frames else np.nan
    bits_per_frame_soap = (gzip_soap * 8.0) / n_frames if n_frames else np.nan

    # Shannon H on 2D PCA of standardized SOAP (single ensemble)
    sh_pca2d: float | np.floating = np.nan
    ipr_pca2d: float | np.floating = np.nan
    if soap_arr.size and len(soap_arr) >= 2:
        sh_pca2d, ipr_pca2d = joint_pca_histogram_entropy(soap_arr, bins=hist_bins)

    # 1D C–C distance entropy on 50 bins
    sh_cc_h = np.nan
    if len(cc_dists) >= 2:
        h, _ = np.histogram(cc_dists, bins=50)
        p = h.astype(np.float64)
        p = p[p > 0]
        p = p / p.sum()
        sh_cc_h = shannon_entropy_from_probs(p)

    return {
        "path": str(path),
        "n_frames": n_frames,
        "zmat_failures": zmat_fail,
        "gzip_bytes_cart": gzip_cart,
        "gzip_bytes_zmat": gzip_zmat,
        "gzip_bytes_soap": gzip_soap,
        "bits_per_frame_cart": bits_per_frame_cart,
        "bits_per_frame_zmat": bits_per_frame_zmat,
        "bits_per_frame_soap": bits_per_frame_soap,
        "shannon_pca2d_soap": float(sh_pca2d) if np.isfinite(sh_pca2d) else np.nan,
        "ipr_pca2d_soap": float(ipr_pca2d) if np.isfinite(ipr_pca2d) else np.nan,
        "shannon_cc_dist_1d": float(sh_cc_h) if np.isfinite(sh_cc_h) else np.nan,
        "soap_dim": int(soap_arr.shape[1]) if soap_arr.size else 0,
        "soap_array": soap_arr,
    }


def parse_scale_from_name(path: Path) -> float | None:
    m = re.search(r"scale_([\d.]+)\.xyz$", path.name, re.I)
    if not m:
        return None
    return float(m.group(1))


def find_reference_xyz(paths: list[Path], explicit: Path | None) -> Path:
    if explicit is not None and explicit.exists():
        return explicit
    for p in sorted(paths):
        if p.name.startswith("cc_scale_") and "1.000" in p.name:
            return p
    for p in paths:
        if p.name.startswith("cc_scale_"):
            return p
    raise FileNotFoundError("No reference XYZ for Z-matrix topology; pass --reference.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare ensemble entropy metrics (gzip, SOAP PCA, pairwise JS/MMD)."
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Multi-frame XYZ files (default: benchmark_out/cc*.xyz and mesh*.xyz).",
    )
    p.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference XYZ (first frame defines Z-matrix construction table).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for per-ensemble and pairwise CSVs.",
    )
    p.add_argument(
        "--pairwise",
        action="store_true",
        help="Write pairwise cc vs mesh metrics per matched scale.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Random subsample each file to at most this many frames.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for subsampling and pairwise draws.",
    )
    p.add_argument(
        "--kabsch",
        action="store_true",
        help="Kabsch-align each COM-centered frame to COM-centered reference.",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="SOAP Gaussian width.",
    )
    p.add_argument(
        "--r-cut",
        type=float,
        default=DEFAULT_R_CUT,
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=DEFAULT_N_MAX,
    )
    p.add_argument(
        "--l-max",
        type=int,
        default=DEFAULT_L_MAX,
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=32,
        help="2D PCA histogram bins for Shannon / JS.",
    )
    p.add_argument(
        "--pairwise-bins",
        type=int,
        default=32,
        help="Bins for pairwise JS/KL on 2D PCA of SOAP.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Write summary PNG plots to outdir.",
    )
    p.add_argument(
        "--geom-histograms",
        action="store_true",
        help=(
            "For each XYZ: density-normalized histograms of all pairwise distances "
            "by element-pair type and all vertex angles by (center, neighbor) types. "
            "Also writes overlay plots under geom_histograms_overlay/."
        ),
    )
    p.add_argument(
        "--geom-bins-dist",
        type=int,
        default=80,
        metavar="N",
        help="Histogram bins for distance plots (default: 80).",
    )
    p.add_argument(
        "--geom-bins-angle",
        type=int,
        default=72,
        metavar="N",
        help="Histogram bins for angle plots (default: 72, 2.5° per bin).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.inputs:
        paths = [p.resolve() for p in args.inputs]
    else:
        here = Path(__file__).resolve().parent / "benchmark_out"
        paths = sorted(here.glob("cc_scale_*.xyz")) + sorted(here.glob("mesh_scale_*.xyz"))
        paths = [p.resolve() for p in paths]

    if not paths:
        raise SystemExit("No input XYZ files.")

    ref_path = find_reference_xyz(paths, args.reference)
    ref_frames = load_xyz_frames(ref_path)
    if not ref_frames:
        raise SystemExit(f"Reference has no frames: {ref_path}")

    ref_atoms = ref_frames[0]
    ref_pos = com_center_positions(ref_atoms)
    ref_cart = ase_to_chemcoord(ref_atoms)
    ref_cart.loc[:, ["x", "y", "z"]] = ref_pos
    ref_zmat = ref_cart.get_zmat()
    c_table = construction_table_from_zmat(ref_zmat)

    try:
        soap_engine = _make_soap(
            DEFAULT_SPECIES,
            args.r_cut,
            args.n_max,
            args.l_max,
            args.sigma,
        )
    except ImportError as e:
        raise SystemExit(
            "dscribe is required for SOAP metrics. Install with: pip install -e '.[quantum]'"
        ) from e

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, Any]] = []
    results_by_path: dict[str, dict[str, Any]] = {}

    for path in paths:
        if not path.exists():
            logger.warning("Missing file: %s", path)
            continue
        r = process_ensemble(
            path,
            c_table=c_table,
            ref_positions_com=ref_pos if args.kabsch else None,
            kabsch=args.kabsch,
            soap_engine=soap_engine,
            max_frames=args.max_frames,
            rng=rng,
            hist_bins=args.hist_bins,
        )
        soap_arr = r.pop("soap_array", None)
        pkey = str(path.resolve())
        results_by_path[pkey] = r
        if soap_arr is not None:
            results_by_path[pkey]["_soap_array"] = soap_arr

        row = {k: v for k, v in r.items() if k != "soap_array"}
        rows.append(row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    per_csv = args.outdir / "ensemble_entropy_per_file.csv"
    fieldnames = [
        "path",
        "n_frames",
        "zmat_failures",
        "gzip_bytes_cart",
        "gzip_bytes_zmat",
        "gzip_bytes_soap",
        "bits_per_frame_cart",
        "bits_per_frame_zmat",
        "bits_per_frame_soap",
        "shannon_pca2d_soap",
        "ipr_pca2d_soap",
        "shannon_cc_dist_1d",
        "soap_dim",
    ]
    with per_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {per_csv}")

    pair_rows: list[dict[str, Any]] = []
    if args.pairwise:
        by_scale: dict[float, dict[str, Path]] = {}
        for p in paths:
            sc = parse_scale_from_name(p)
            if sc is None:
                continue
            by_scale.setdefault(sc, {})["cc" if p.name.startswith("cc_") else "mesh"] = p

        for scale, d in sorted(by_scale.items()):
            cc_p = d.get("cc")
            mesh_p = d.get("mesh")
            if cc_p is None or mesh_p is None:
                continue
            rc = results_by_path.get(str(cc_p.resolve()))
            rm = results_by_path.get(str(mesh_p.resolve()))
            if not rc or not rm:
                continue
            if rc.get("n_frames", 0) == 0 or rm.get("n_frames", 0) == 0:
                logger.warning("Skipping pairwise scale=%s: empty ensemble.", scale)
                continue
            sa = np.asarray(rc.get("_soap_array"))
            sb = np.asarray(rm.get("_soap_array"))
            if sa.size == 0 or sb.size == 0:
                continue
            pm = pairwise_soap_metrics(
                sa,
                sb,
                bins=args.pairwise_bins,
                rng=rng,
            )
            pair_rows.append(
                {
                    "scale": scale,
                    "cc_path": str(cc_p),
                    "mesh_path": str(mesh_p),
                    "n_cc": rc["n_frames"],
                    "n_mesh": rm["n_frames"],
                    **pm,
                }
            )

        pair_csv = args.outdir / "ensemble_entropy_pairwise.csv"
        if pair_rows:
            keys = list(pair_rows[0].keys())
            with pair_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in pair_rows:
                    w.writerow(row)
            print(f"Wrote {pair_csv}")
        else:
            print("No pairwise rows (need matching cc/mesh pairs with SOAP data).")

    if args.plot:
        make_plots(rows=rows, pair_rows=pair_rows, outdir=args.outdir)

    if args.geom_histograms:
        run_geometry_histograms(
            paths,
            args.outdir,
            max_frames=args.max_frames,
            rng=rng,
            bins_dist=args.geom_bins_dist,
            bins_angle=args.geom_bins_angle,
        )


if __name__ == "__main__":
    main()
