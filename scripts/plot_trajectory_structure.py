#!/usr/bin/env python3
"""Generate structural-distribution plots from an ASE-readable trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from mmml.utils.plotting.trajectory_structure import (
    element_pair_rdfs,
    internal_coordinate_distributions,
    water_tetrahedrality,
)


def plot_rdfs(radii, rdfs, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for axis, (pair, rdf) in zip(axes.flat, sorted(rdfs.items())):
        axis.plot(radii, rdf)
        axis.set_title(pair)
        axis.set_ylabel("g(r)")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("r (Å)")
    figure.suptitle("Element-pair radial distribution functions")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_internal(data, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    groups = ((data.bonds, "Bond lengths", "Å", 40), (data.angles, "Angles", "degrees", 45),
              (data.dihedrals, "Dihedrals", "degrees", 72))
    for axis, (coordinates, title, unit, bins) in zip(axes, groups):
        all_values = np.concatenate(list(coordinates.values()))
        axis.hist(all_values, bins=bins, density=True, alpha=0.8)
        axis.set_title(f"{title} ({len(coordinates)} coordinates)")
        axis.set_xlabel(unit)
        axis.set_ylabel("Probability density")
        axis.grid(alpha=0.2)
    figure.suptitle("Peptide internal degrees of freedom")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_tetrahedrality(values, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-0.5, 1.0, 61)
    for key, color in (("near", "#E64B35"), ("bulk", "#3C5488")):
        axis.hist(values[key], bins=bins, density=True, histtype="step", linewidth=2,
                  label=f"{key} (n={len(values[key]):,})", color=color)
    axis.set_xlabel("Tetrahedral order parameter q")
    axis.set_ylabel("Probability density")
    axis.set_title("Water tetrahedrality near peptide and in bulk")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--solute-atoms", type=int, default=22)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--rdf-bins", type=int, default=160)
    parser.add_argument("--near-cutoff", type=float, default=5.0)
    parser.add_argument("--bulk-cutoff", type=float, default=8.0)
    args = parser.parse_args()
    frames = read(args.trajectory, index=":")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    radii, rdfs = element_pair_rdfs(frames, r_max=args.r_max, bins=args.rdf_bins)
    internal = internal_coordinate_distributions(frames, range(args.solute_atoms))
    tetrahedrality = water_tetrahedrality(
        frames, peptide_indices=range(args.solute_atoms), near_cutoff=args.near_cutoff,
        bulk_cutoff=args.bulk_cutoff,
    )
    plot_rdfs(radii, rdfs, args.output_dir / "test4_element_pair_rdfs.png")
    plot_internal(internal, args.output_dir / "test4_internal_coordinates.png")
    plot_tetrahedrality(tetrahedrality, args.output_dir / "test4_water_tetrahedrality.png")
    np.savez(args.output_dir / "test4_element_pair_rdfs.npz", radii=radii, **rdfs)
    np.savez(args.output_dir / "test4_internal_coordinates.npz", **{
        f"bond_{index}": value for index, value in enumerate(internal.bonds.values())
    }, **{f"angle_{index}": value for index, value in enumerate(internal.angles.values())},
        **{f"dihedral_{index}": value for index, value in enumerate(internal.dihedrals.values())})
    np.savez(args.output_dir / "test4_water_tetrahedrality.npz", **tetrahedrality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
