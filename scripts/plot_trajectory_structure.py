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
    hydrogen_bond_analysis,
    heavy_atom_pair_distance_umap,
    internal_coordinate_distributions,
    radius_of_gyration_and_diffusion,
    water_tetrahedrality,
)


def plot_rdfs(radii, rdfs, output: Path) -> None:
    import matplotlib.patheffects as path_effects
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    elements = sorted({element for pair in rdfs for element in pair.split("-")})
    size = len(elements)
    figure, axes = plt.subplots(size, size, figsize=(3.2 * size, 3.0 * size), sharex=True)
    line_styles = ("-", "--", "-.", ":")
    for row, element_a in enumerate(elements):
        for column, element_b in enumerate(elements):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            key = "-".join(sorted((element_a, element_b)))
            color_a = np.asarray(jmol_colors[atomic_numbers[element_a]])
            color_b = np.asarray(jmol_colors[atomic_numbers[element_b]])
            pair_color = color_a if element_a == element_b else 0.5 * (color_a + color_b)
            line_style = "-" if row == column else line_styles[(row + column) % len(line_styles)]
            line = axis.plot(
                radii, rdfs[key], color=pair_color, linestyle=line_style, linewidth=2.0
            )[0]
            axis.set_title(key, fontweight="bold")
            if float(np.mean(pair_color)) > 0.88:
                outline = [path_effects.Stroke(linewidth=3.5, foreground="#999999"), path_effects.Normal()]
                line.set_path_effects(outline)
            axis.grid(alpha=0.18)
            if column == 0:
                axis.set_ylabel(f"{element_a}  g(r)")
            if row == size - 1:
                axis.set_xlabel("r (Å)")
            axis.text(
                0.96, 0.92, line_style, transform=axis.transAxes, ha="right", va="top",
                color=pair_color, fontsize=13,
            )
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


def plot_hydrogen_bond_timeseries(data, output: Path) -> None:
    figure, peptide_axis = plt.subplots(figsize=(10, 5.5))
    water_axis = peptide_axis.twinx()
    frame = np.arange(len(data.total_counts))
    peptide_water = peptide_axis.step(
        frame, data.peptide_water_counts, where="mid", color="#00A087",
        label="peptide-water", linewidth=1.6,
    )[0]
    peptide_peptide = peptide_axis.step(
        frame, data.peptide_peptide_counts, where="mid", color="#E64B35",
        label="peptide-peptide", linewidth=1.6,
    )[0]
    water_water = water_axis.plot(
        frame, data.water_water_counts, color="#3C5488", label="water-water",
        linewidth=1.1, alpha=0.8,
    )[0]
    peptide_axis.set_xlabel("Frame")
    peptide_axis.set_ylabel("Peptide H-bond count", color="#008B75")
    water_axis.set_ylabel("Water-water H-bond count", color="#3C5488")
    peptide_axis.tick_params(axis="y", colors="#008B75")
    water_axis.tick_params(axis="y", colors="#3C5488")
    peptide_axis.grid(alpha=0.2)
    peptide_axis.legend(
        handles=(peptide_water, peptide_peptide, water_water), loc="upper right", ncol=3
    )
    peptide_axis.set_title("Hydrogen bonds over time")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hydrogen_bond_network(data, frames, output: Path, *, solute_atoms: int) -> None:
    import networkx as nx

    graph = nx.DiGraph()
    frame = frames[data.representative_frame]
    graph.add_nodes_from(range(len(frame)))
    graph.add_edges_from(data.representative_edges)
    positions_3d = frame.get_positions(wrap=True)
    positions = {index: positions_3d[index, :2] for index in graph}
    symbols = np.asarray(frame.get_chemical_symbols())
    figure, axis = plt.subplots(figsize=(11, 10))
    atom_colors = {"H": "#BDBDBD", "C": "#333333", "N": "#3C5488", "O": "#E64B35"}
    nx.draw_networkx_nodes(
        graph, positions, node_color=[atom_colors.get(symbol, "#888888") for symbol in symbols],
        node_size=[38 if index < solute_atoms else 10 for index in graph],
        alpha=0.20, linewidths=0, ax=axis,
    )
    edge_groups = {"peptide-peptide": [], "peptide-water": [], "water-water": []}
    for edge in graph.edges:
        donor_peptide = edge[0] < solute_atoms
        acceptor_peptide = edge[1] < solute_atoms
        if donor_peptide and acceptor_peptide:
            edge_groups["peptide-peptide"].append(edge)
        elif donor_peptide or acceptor_peptide:
            edge_groups["peptide-water"].append(edge)
        else:
            edge_groups["water-water"].append(edge)
    for label, color, alpha, width in (
        ("water-water", "#4DBBD5", 0.24, 0.45),
        ("peptide-water", "#00A087", 0.9, 1.8),
        ("peptide-peptide", "#E64B35", 1.0, 2.2),
    ):
        nx.draw_networkx_edges(
            graph, positions, edgelist=edge_groups[label], edge_color=color, alpha=alpha,
            width=width, arrows=False, label=f"{label} ({len(edge_groups[label])})", ax=axis,
        )
    peptide_nodes = range(solute_atoms)
    nx.draw_networkx_labels(
        graph, positions, labels={index: f"{symbols[index]}{index}" for index in peptide_nodes},
        font_size=6, ax=axis,
    )
    axis.legend(loc="upper right")
    axis.set_aspect("equal")
    axis.set_axis_off()
    axis.set_title(f"Hydrogen-bond network, representative frame {data.representative_frame}")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_radius_and_diffusion(data, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(data.time_ps, data.radius_of_gyration, color="#00A087")
    axes[0].plot(
        data.time_ps, data.end_to_end_distance, color="#E64B35", alpha=0.85,
        label=f"end-to-end ({data.end_to_end_indices[0]}–{data.end_to_end_indices[1]})",
    )
    axes[0].plot([], [], color="#00A087", label=r"$R_g$")
    axes[0].set_ylabel("Peptide distance (Å)")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.2)
    peptide_msd_axis = axes[1].twinx()
    water_line = axes[1].plot(
        data.time_ps, data.water_oxygen_msd, color="#3C5488", label="water O MSD"
    )[0]
    peptide_line = peptide_msd_axis.plot(
        data.time_ps, data.peptide_com_msd, color="#00A087", alpha=0.8,
        label="peptide COM MSD",
    )[0]
    fit_slice = slice(data.fit_start_frame, None)
    slope = 6.0 * data.diffusion_angstrom2_per_ps
    intercept = np.mean(
        data.water_oxygen_msd[fit_slice] - slope * data.time_ps[fit_slice]
    )
    water_fit = axes[1].plot(
        data.time_ps[fit_slice], slope * data.time_ps[fit_slice] + intercept, "--",
        color="#E64B35", label=f"fit: D={data.diffusion_angstrom2_per_ps:.3g} Å²/ps",
    )[0]
    peptide_slope = 6.0 * data.peptide_diffusion_angstrom2_per_ps
    peptide_intercept = np.mean(
        data.peptide_com_msd[fit_slice] - peptide_slope * data.time_ps[fit_slice]
    )
    peptide_fit = peptide_msd_axis.plot(
        data.time_ps[fit_slice], peptide_slope * data.time_ps[fit_slice] + peptide_intercept,
        ":", color="#F39B7F",
        label=f"peptide fit: D={data.peptide_diffusion_angstrom2_per_ps:.3g} Å²/ps",
    )[0]
    axes[1].set_xlabel(r"Lag time $\Delta t$ (ps)")
    axes[1].set_ylabel(r"Water oxygen MSD (Å$^2$)")
    peptide_msd_axis.set_ylabel(r"Peptide COM MSD (Å$^2$)", color="#008B75")
    peptide_msd_axis.tick_params(axis="y", colors="#008B75")
    axes[1].legend(
        handles=(water_line, water_fit, peptide_line, peptide_fit), ncol=2, fontsize=8
    )
    axes[1].grid(alpha=0.2)
    figure.suptitle("Unwrapped trajectory dynamics")
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_umap(embedding: np.ndarray, time_ps: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    points = axis.scatter(
        embedding[:, 0], embedding[:, 1], c=time_ps, cmap="viridis", s=28, alpha=0.85
    )
    figure.colorbar(points, ax=axis, label=r"Lag time $\Delta t$ (ps)")
    axis.plot(embedding[:, 0], embedding[:, 1], color="#777777", alpha=0.22, linewidth=0.7)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Peptide heavy-atom pair-distance UMAP")
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
    parser.add_argument("--hydrogen-bonds-only", action="store_true")
    parser.add_argument("--dynamics-only", action="store_true")
    parser.add_argument("--timestep-ps", type=float, default=1.0)
    parser.add_argument("--diffusion-fit-start", type=float, default=0.5)
    parser.add_argument("--end-to-end", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--umap-neighbors", type=int, default=20)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--hbond-distance", type=float, default=3.8)
    parser.add_argument("--hbond-angle", type=float, default=135.0)
    args = parser.parse_args()
    frames = read(args.trajectory, index=":")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.hydrogen_bonds_only and not args.dynamics_only:
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
    if not args.dynamics_only:
        hydrogen_bonds = hydrogen_bond_analysis(
            frames, peptide_indices=range(args.solute_atoms), distance_cutoff=args.hbond_distance,
            angle_cutoff_degrees=args.hbond_angle,
        )
        plot_hydrogen_bond_timeseries(hydrogen_bonds, args.output_dir / "test4_hydrogen_bond_timeseries.png")
        plot_hydrogen_bond_network(
            hydrogen_bonds, frames, args.output_dir / "test4_hydrogen_bond_network.png",
            solute_atoms=args.solute_atoms,
        )
        edges = np.asarray(list(hydrogen_bonds.edge_occupancy), dtype=str)
        occupancies = np.asarray(list(hydrogen_bonds.edge_occupancy.values()))
        np.savez(
            args.output_dir / "test4_hydrogen_bonds.npz", total=hydrogen_bonds.total_counts,
            peptide_peptide=hydrogen_bonds.peptide_peptide_counts,
            peptide_water=hydrogen_bonds.peptide_water_counts,
            water_water=hydrogen_bonds.water_water_counts, edges=edges, occupancy=occupancies,
            representative_frame=hydrogen_bonds.representative_frame,
            representative_edges=np.asarray(hydrogen_bonds.representative_edges, dtype=int),
        )
    dynamics = radius_of_gyration_and_diffusion(
        frames, peptide_indices=range(args.solute_atoms), timestep_ps=args.timestep_ps,
        fit_start_fraction=args.diffusion_fit_start,
        end_to_end_indices=tuple(args.end_to_end) if args.end_to_end else None,
    )
    plot_radius_and_diffusion(dynamics, args.output_dir / "test4_radius_gyration_msd.png")
    embedding, pair_distances, heavy_atom_pairs = heavy_atom_pair_distance_umap(
        frames, peptide_indices=range(args.solute_atoms), n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    plot_umap(embedding, dynamics.time_ps, args.output_dir / "test4_heavy_atom_distance_umap.png")
    np.savez(
        args.output_dir / "test4_radius_gyration_msd.npz", time_ps=dynamics.time_ps,
        radius_of_gyration=dynamics.radius_of_gyration,
        end_to_end_distance=dynamics.end_to_end_distance,
        end_to_end_indices=dynamics.end_to_end_indices,
        water_oxygen_msd=dynamics.water_oxygen_msd,
        peptide_com_msd=dynamics.peptide_com_msd,
        diffusion_angstrom2_per_ps=dynamics.diffusion_angstrom2_per_ps,
        diffusion_cm2_per_s=dynamics.diffusion_cm2_per_s,
        peptide_diffusion_angstrom2_per_ps=dynamics.peptide_diffusion_angstrom2_per_ps,
        peptide_diffusion_cm2_per_s=dynamics.peptide_diffusion_cm2_per_s,
        fit_start_frame=dynamics.fit_start_frame,
    )
    np.savez(
        args.output_dir / "test4_heavy_atom_distance_umap.npz", embedding=embedding,
        pair_distances=pair_distances, heavy_atom_pairs=heavy_atom_pairs,
        time_ps=dynamics.time_ps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
