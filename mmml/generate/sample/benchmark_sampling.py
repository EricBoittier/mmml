#!/usr/bin/env python
"""
Benchmark dimer generation with both sampling approaches:
- internal-coordinate chemcoord sampling (sample_dimer_cc)
- mesh-based contact sampling (generate_dimers_mesh)

Scans a list of scale values and reports runtime + number of generated dimers.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

from mmml.generate.sample import sample_cc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark dimer sampling approaches.")
    parser.add_argument(
        "--cc-xyz",
        default="meoh_dimer.xyz",
        help="Input XYZ for chemcoord internal-coordinate sampler (must contain 2 fragments).",
    )
    parser.add_argument(
        "--mesh-xyz",
        default="old/meoh.xyz",
        help="Input XYZ for mesh sampler (single monomer).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.2, 1.4],
        help="Scale values to scan for both methods.",
    )
    parser.add_argument(
        "--max-dimers-mesh",
        type=int,
        default=10000,
        help="Max candidate dimers for the mesh method per scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=sample_cc.DEFAULT_NOISE_SCALE,
        help="Coordinate noise scale applied in both methods.",
    )
    parser.add_argument(
        "--outdir",
        default="benchmark_out",
        help="Directory for generated XYZ files and CSV summary.",
    )
    return parser.parse_args()


def count_xyz_structures(path: Path) -> int:
    """
    Count structures in a multi-structure XYZ file by reading atom-count headers.
    """
    if not path.exists():
        return 0

    n_structures = 0
    with path.open("r") as f:
        while True:
            header = f.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            try:
                n_atoms = int(header)
            except ValueError:
                break
            _ = f.readline()  # comment line
            for _ in range(n_atoms):
                if not f.readline():
                    return n_structures
            n_structures += 1
    return n_structures


def write_cc_xyz(path: Path, frames, atom_symbols) -> None:
    with path.open("w") as f:
        for idx, xyz in enumerate(frames):
            f.write(f"{len(xyz)}\n")
            f.write(f"cc dimer {idx}\n")
            for sym, (x, y, z) in zip(atom_symbols, xyz[["x", "y", "z"]].to_numpy()):
                f.write(f"{sym:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def benchmark_cc(cc_xyz: Path, scales: list[float], outdir: Path, seed: int, noise_scale: float):
    cart = sample_cc.load_molecule_xyz(str(cc_xyz))
    atom_symbols = cart["atom"].tolist()
    results = []

    for i, scale in enumerate(scales):
        t0 = time.perf_counter()
        frames = sample_cc.sample_dimer_cc(
            str(cc_xyz),
            mol_r_scale=scale,
            seed=seed + i,
            noise_scale=noise_scale,
        )
        elapsed = time.perf_counter() - t0
        out_xyz = outdir / f"cc_scale_{scale:.3f}.xyz"
        write_cc_xyz(out_xyz, frames, atom_symbols)
        results.append(
            {
                "method": "cc",
                "scale": scale,
                "seconds": elapsed,
                "n_dimers": len(frames),
                "output_xyz": str(out_xyz),
            }
        )
    return results


def benchmark_mesh(
    mesh_xyz: Path,
    scales: list[float],
    outdir: Path,
    seed: int,
    noise_scale: float,
    max_dimers_mesh: int,
):
    molecule = sample_cc.load_molecule_xyz(str(mesh_xyz))
    eq = sample_cc.symmetrize_molecule(molecule, max_n=25, tolerance=0.3, epsilon=1e-5)
    sym_mol = eq["sym_mol"]
    positions = sym_mol[["x", "y", "z"]].to_numpy()
    radii = sample_cc.vdw_radii_for_cartesian(sym_mol)
    results = []

    for i, scale in enumerate(scales):
        t0 = time.perf_counter()
        atom_meshes = sample_cc.mesh_points_for_atoms(
            positions,
            radii,
            n_radial=10,
            n_angular=10,
            radii_scale=scale,
        )
        mesh_points = sample_cc.unique_mesh_points_from_symmetry(eq, atom_meshes)
        normals = sample_cc.normals_from_nearest_neighbors(mesh_points)
        out_xyz = outdir / f"mesh_scale_{scale:.3f}.xyz"
        sample_cc.generate_dimers_mesh(
            mesh_points=mesh_points,
            normals=normals,
            mol_cart=sym_mol,
            max_dimers=max_dimers_mesh,
            output_xyz=str(out_xyz),
            overlap_tolerance=0.1,
            seed=seed + i,
            noise_scale=noise_scale,
        )
        elapsed = time.perf_counter() - t0
        n_dimers = count_xyz_structures(out_xyz)
        results.append(
            {
                "method": "mesh",
                "scale": scale,
                "seconds": elapsed,
                "n_dimers": n_dimers,
                "output_xyz": str(out_xyz),
            }
        )
    return results


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["method", "scale", "seconds", "n_dimers", "output_xyz"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cc_xyz = Path(args.cc_xyz)
    mesh_xyz = Path(args.mesh_xyz)
    scales = list(args.scales)

    rows = []

    if cc_xyz.exists():
        rows.extend(
            benchmark_cc(
                cc_xyz=cc_xyz,
                scales=scales,
                outdir=outdir,
                seed=args.seed,
                noise_scale=args.noise_scale,
            )
        )
    else:
        print(f"Skipping cc benchmark: missing file {cc_xyz}")

    if mesh_xyz.exists():
        rows.extend(
            benchmark_mesh(
                mesh_xyz=mesh_xyz,
                scales=scales,
                outdir=outdir,
                seed=args.seed,
                noise_scale=args.noise_scale,
                max_dimers_mesh=args.max_dimers_mesh,
            )
        )
    else:
        print(f"Skipping mesh benchmark: missing file {mesh_xyz}")

    if not rows:
        raise SystemExit("No benchmark runs executed; provide valid input files.")

    summary_csv = outdir / "benchmark_summary.csv"
    write_summary_csv(summary_csv, rows)

    print("Benchmark complete.")
    print(f"Summary: {summary_csv}")
    for row in rows:
        print(
            f"{row['method']:4s} scale={row['scale']:.3f} "
            f"n_dimers={row['n_dimers']:6d} time_s={row['seconds']:.3f}"
        )


if __name__ == "__main__":
    main()

