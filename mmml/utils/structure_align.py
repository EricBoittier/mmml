"""Kabsch alignment and ASE structure panels for NPZ training structures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.visualize.plot import plot_atoms

BOND_FACTOR = 1.2


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


def _bond_cutoff(z_i: int, z_j: int) -> float:
    return BOND_FACTOR * (float(covalent_radii[z_i]) + float(covalent_radii[z_j]))


def bond_lengths_per_element(atoms: Atoms) -> dict[int, list[float]]:
    """Collect bonded pair lengths keyed by participating atomic number."""
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    by_element: dict[int, list[float]] = defaultdict(list)

    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            length = float(np.linalg.norm(positions[i] - positions[j]))
            if length > _bond_cutoff(int(numbers[i]), int(numbers[j])):
                continue
            by_element[int(numbers[i])].append(length)
            by_element[int(numbers[j])].append(length)
    return dict(by_element)


def _element_label(z: int) -> str:
    if 0 < z < len(chemical_symbols):
        return chemical_symbols[z]
    return f"Z{z}"


def _prepare_aligned_atoms(
    npz_path: Path,
    indices: Sequence[int],
    reference_index: int,
) -> tuple[list[Atoms], list[float]]:
    reference = load_npz_structure(npz_path, reference_index)
    ref_positions = np.asarray(reference.get_positions(), dtype=np.float64)
    ref_numbers = reference.get_atomic_numbers()

    aligned_atoms: list[Atoms] = []
    rmsds: list[float] = []
    for idx in indices:
        atoms = load_npz_structure(npz_path, idx)
        if len(atoms) != len(reference):
            raise ValueError(
                f"Structure {idx} has {len(atoms)} atoms but reference {reference_index} has {len(reference)}"
            )
        if not np.array_equal(atoms.get_atomic_numbers(), ref_numbers):
            raise ValueError(
                f"Atomic numbers differ between structure {idx} and reference {reference_index}"
            )
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        if idx == reference_index:
            aligned_positions = positions
            rmsd = 0.0
        else:
            aligned_positions = align_positions(positions, ref_positions)
            rmsd = structure_rmsd(positions, ref_positions)
        aligned_atoms.append(Atoms(numbers=atoms.get_atomic_numbers(), positions=aligned_positions))
        rmsds.append(rmsd)
    return aligned_atoms, rmsds


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
    show: bool = False,
) -> plt.Figure:
    """Plot Kabsch-aligned structures side by side with bond-length histograms below."""
    if not indices:
        raise ValueError("indices must not be empty")

    unique_indices = list(dict.fromkeys(int(i) for i in indices))
    if reference_index is None:
        reference_index = unique_indices[0]
    reference_index = int(reference_index)

    if labels is None:
        labels = [f"idx {idx}" for idx in unique_indices]
    elif len(labels) != len(unique_indices):
        raise ValueError(f"labels length {len(labels)} != indices length {len(unique_indices)}")

    aligned_atoms, rmsds = _prepare_aligned_atoms(npz_path, unique_indices, reference_index)
    elements = sorted({int(z) for atoms in aligned_atoms for z in atoms.get_atomic_numbers()})
    n_struct = len(unique_indices)
    n_elem = max(len(elements), 1)
    n_cols = max(n_struct, n_elem)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.4 * n_cols, 7.5),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    if n_cols == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for col in range(n_cols):
        if col >= n_struct:
            axes[0, col].axis("off")
            continue
        atoms = aligned_atoms[col]
        plot_atoms(atoms, ax=axes[0, col], rotation=rotation, radii=radii)
        subtitle = labels[col]
        if rmsds[col] > 0.0:
            subtitle = f"{subtitle}\nRMSD {rmsds[col]:.3f} Å"
        if unique_indices[col] == reference_index:
            subtitle = f"{subtitle}\n(reference)"
        axes[0, col].set_title(subtitle, fontsize=10)

    bond_data_by_structure = [bond_lengths_per_element(atoms) for atoms in aligned_atoms]
    hist_colors = plt.get_cmap("tab10")

    for col in range(n_cols):
        ax = axes[1, col]
        if col >= n_elem:
            ax.axis("off")
            continue
        z = elements[col]
        symbol = _element_label(z)
        series: list[np.ndarray] = []
        series_labels: list[str] = []
        for struct_i, bonds_by_elem in enumerate(bond_data_by_structure):
            lengths = bonds_by_elem.get(z, [])
            if not lengths:
                continue
            series.append(np.asarray(lengths, dtype=np.float64))
            short_label = labels[struct_i].split("\n", maxsplit=1)[0]
            if len(short_label) > 22:
                short_label = short_label[:19] + "..."
            series_labels.append(short_label)

        if not series:
            ax.text(0.5, 0.5, "no bonded pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{symbol} bonds")
            ax.set_xlabel("length (Å)")
            continue

        bins = np.linspace(
            min(arr.min() for arr in series),
            max(arr.max() for arr in series),
            12,
        )
        if bins[0] == bins[-1]:
            bins = np.linspace(bins[0] - 0.05, bins[-1] + 0.05, 12)

        for struct_i, (arr, lab) in enumerate(zip(series, series_labels)):
            ax.hist(
                arr,
                bins=bins,
                alpha=0.55,
                color=hist_colors(struct_i % 10),
                label=lab,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(f"{symbol} bond lengths")
        ax.set_xlabel("length (Å)")
        ax.set_ylabel("count")
        if len(series) > 1:
            ax.legend(fontsize=7, loc="best")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
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
