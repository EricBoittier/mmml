#!/usr/bin/env python3
"""Matplotlib 2D energy surfaces for one dimer-pair scan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# (npz_key, display label, colormap)
ENERGY_PANELS: tuple[tuple[str, str, str], ...] = (
    ("charmm_ENER_kcal", "CHARMM ENER", "viridis"),
    ("xtb_energy_kcal", "xTB GFN2", "plasma"),
    ("orca_mp2_energy_kcal", "ORCA MP2", "cividis"),
)


def _relative_kcal(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    if out.size == 0 or np.all(np.isnan(out)):
        return out
    ref = np.nanmin(out)
    out = out - ref
    return out


def plot_pair_scan_npz(
    npz_path: Path,
    output_png: Path,
    *,
    title: str | None = None,
    dpi: int = 120,
) -> bool:
    """Plot energy panels; return False if NPZ missing."""
    if not npz_path.is_file():
        return False

    data = np.load(npz_path, allow_pickle=False)
    d01 = np.asarray(data["d01_grid"], dtype=float)
    d02 = np.asarray(data["d02_grid"], dtype=float)
    label = title or str(data.get("label", npz_path.parent.name))

    panels = [(k, lab, cmap) for k, lab, cmap in ENERGY_PANELS if k in data]
    if not panels:
        return False

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    D1, D2 = np.meshgrid(d01, d02, indexing="ij")

    for ax, (key, lab, cmap) in zip(axes, panels, strict=True):
        z = _relative_kcal(data[key])
        im = ax.pcolormesh(D1, D2, z, shading="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label="ΔE (kcal/mol)")
        ax.set_xlabel("d01 (Å)")
        ax.set_ylabel("d02 (Å)")
        ax.set_title(lab)
        ax.set_aspect("equal")

    fig.suptitle(label, fontsize=11, y=1.02)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pending_pair(
    output_png: Path,
    *,
    pair_tag: str,
    label: str,
    dpi: int = 100,
) -> None:
    """Placeholder figure when scan NPZ is not available yet."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.35,
        f"{pair_tag}\n(scan pending)",
        ha="center",
        va="center",
        fontsize=10,
        color="0.45",
        transform=ax.transAxes,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="#f5f5f5")
    plt.close(fig)
