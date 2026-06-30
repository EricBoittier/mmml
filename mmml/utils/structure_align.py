"""Kabsch alignment and ASE overlays for NPZ training structures."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize.plot import plot_atoms


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return rotation matrix R such that mobile @ R best matches target (both Nx3)."""
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Expected matching (N, 3) arrays, got {mobile.shape} and {target.shape}")

    covariance = mobile.T @ target
    rotation, _, weights = np.linalg.svd(covariance)
    if np.linalg.det(rotation) * np.linalg.det(weights) < 0.0:
        rotation[:, -1] *= -1.0
    return rotation @ weights


def align_positions(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Center both frames and rotate mobile onto target."""
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mobile_center = mobile - mobile.mean(axis=0)
    target_center = target - target.mean(axis=0)
    rotation = kabsch_rotation(mobile_center, target_center)
    return mobile_center @ rotation


def structure_rmsd(mobile: np.ndarray, target: np.ndarray) -> float:
    aligned = align_positions(mobile, target)
    target_center = target - target.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((aligned - target_center) ** 2, axis=1))))


def load_npz_structure(npz_path: Path, index: int) -> Atoms:
    """Load one padded NPZ frame as ASE ``Atoms`` (trimmed to ``N[index]``)."""
    data = np.load(npz_path, allow_pickle=True)
    n_samples = int(data["R"].shape[0])
    if index < 0 or index >= n_samples:
        raise IndexError(f"Sample index {index} out of range (0..{n_samples - 1})")

    natoms_real = int(data["N"][index])
    z_full = np.asarray(data["Z"][index], dtype=np.int32).reshape(-1)
    r_full = np.asarray(data["R"][index], dtype=np.float64).reshape(-1, 3)
    return Atoms(numbers=z_full[:natoms_real], positions=r_full[:natoms_real])


def _default_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_aligned_structures(
    *,
    npz_path: Path,
    indices: Sequence[int],
    labels: Sequence[str] | None = None,
    reference_index: int | None = None,
    out_path: Path | None = None,
    title: str | None = None,
    rotation: str = "90x,0y,0z",
    radii: float = 0.35,
    reference_color: str = "#333333",
    show: bool = False,
) -> plt.Figure:
    """Align NPZ structures to a reference and superimpose with ASE ``plot_atoms``."""
    if not indices:
        raise ValueError("indices must not be empty")

    unique_indices = list(dict.fromkeys(int(i) for i in indices))
    if reference_index is None:
        reference_index = unique_indices[0]
    reference_index = int(reference_index)

    reference = load_npz_structure(npz_path, reference_index)
    ref_positions = np.asarray(reference.get_positions(), dtype=np.float64)

    if labels is None:
        labels = [f"idx {idx}" for idx in unique_indices]
    elif len(labels) != len(unique_indices):
        raise ValueError(f"labels length {len(labels)} != indices length {len(unique_indices)}")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_atoms(
        Atoms(numbers=reference.get_atomic_numbers(), positions=ref_positions),
        ax=ax,
        rotation=rotation,
        radii=radii,
        colors=[reference_color] * len(reference),
    )

    palette = _default_palette(len(unique_indices))
    legend_handles: list[plt.Line2D] = []
    ref_label = next((lab for idx, lab in zip(unique_indices, labels) if idx == reference_index), f"idx {reference_index}")
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=reference_color, markersize=8, label=f"ref {ref_label}")
    )

    for color, idx, label in zip(palette, unique_indices, labels):
        if idx == reference_index:
            continue
        atoms = load_npz_structure(npz_path, idx)
        if len(atoms) != len(reference):
            raise ValueError(
                f"Structure {idx} has {len(atoms)} atoms but reference {reference_index} has {len(reference)}"
            )
        if not np.array_equal(atoms.get_atomic_numbers(), reference.get_atomic_numbers()):
            raise ValueError(
                f"Atomic numbers differ between structure {idx} and reference {reference_index}"
            )

        aligned = align_positions(np.asarray(atoms.get_positions()), ref_positions)
        rmsd = structure_rmsd(np.asarray(atoms.get_positions()), ref_positions)
        plot_atoms(
            Atoms(numbers=atoms.get_atomic_numbers(), positions=aligned),
            ax=ax,
            rotation=rotation,
            radii=radii * 0.85,
            colors=[color] * len(atoms),
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=f"{label} (RMSD {rmsd:.3f} Å)",
            )
        )

    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    ax.set_title(title or "Aligned NPZ structures")
    fig.tight_layout()

    if out_path is not None:
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    elif out_path is None:
        plt.close(fig)

    return fig


def select_structure_indices(
    *,
    suspect_samples: Sequence[dict[str, object]],
    global_outliers: Sequence[dict[str, object]],
    max_structures: int = 8,
) -> list[int]:
    """Merge suspect and global outlier indices, preserving priority order."""
    ordered: list[int] = []
    seen: set[int] = set()
    for row in suspect_samples:
        idx = int(row["index"])
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    for row in global_outliers:
        idx = int(row["index"])
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered[:max_structures]
