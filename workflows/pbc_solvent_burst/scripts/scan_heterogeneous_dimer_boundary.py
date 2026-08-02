#!/usr/bin/env python3
"""Scan a padded DCM/ACO dimer through the sparse ML activation boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from mmml.interfaces.pycharmmInterface.cutoffs import CutoffParameters
from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator
from mmml.utils.plotting.styles import apply_plot_style, legend_outside


def read_pdb(path: Path, resid: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    zmap = {"H": 1, "C": 6, "O": 8, "CL": 17}
    xyz, numbers = [], []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if resid is not None and int(line[22:26]) != resid:
            continue
        name = line[12:16].strip().upper()
        element = "CL" if name.startswith("CL") else name[0]
        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        numbers.append(zmap[element])
    return np.asarray(xyz, float), np.asarray(numbers, np.int32)


def centered(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0)


def build_calc(checkpoint: Path, z: np.ndarray, r0: np.ndarray, sparse: bool):
    factory = setup_calculator(
        ATOMS_PER_MONOMER=[5, 10, 5], N_MONOMERS=3,
        doML=True, doMM=False, doML_dimer=True,
        model_restart_path=str(checkpoint), MAX_ATOMS_PER_SYSTEM=15,
        ml_sparse_dimers=sparse, ml_max_active_dimers=1,
        mm_switch_on=6.0, ml_switch_width=1.0, cell=32.0,
        ml_compute_dtype="float64", verbose=True,
    )
    cutoff = CutoffParameters(ml_switch_width=1.0, mm_switch_on=6.0, mm_switch_width=4.0)
    calc, _, _ = factory(
        atomic_numbers=z, atomic_positions=r0, n_monomers=3,
        cutoff_params=cutoff, doML=True, doMM=False, doML_dimer=True,
        backprop=False, debug=False, verbose=False,
    )
    return calc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dcm-pdb", type=Path, required=True)
    parser.add_argument("--aco-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npoints", type=int, default=101)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dcm, zd = read_pdb(args.dcm_pdb, resid=1)
    aco, za = read_pdb(args.aco_pdb)
    dcm, aco = centered(dcm), centered(aco)
    # A third monomer forces sparse selection while remaining outside the active shell.
    far = dcm + np.array([13.0, 13.0, 0.0])
    z = np.concatenate([zd, za, zd])
    base = np.concatenate([dcm, aco + np.array([4.5, 0, 0]), far])
    full = build_calc(args.checkpoint, z, base, False)
    sparse = build_calc(args.checkpoint, z, base, True)
    distances = np.linspace(4.5, 7.5, args.npoints)
    result = {"distance_A": distances.tolist()}
    for label, calc in (("full", full), ("sparse", sparse)):
        energies, radial_forces = [], []
        for distance in distances:
            pos = base.copy()
            pos[5:15] = aco + np.array([distance, 0, 0])
            atoms = Atoms(numbers=z, positions=pos, cell=[32, 32, 32], pbc=True)
            atoms.calc = calc
            energies.append(float(atoms.get_potential_energy()))
            radial_forces.append(float(np.sum(atoms.get_forces()[5:15, 0])))
        result[f"energy_{label}_eV"] = energies
        result[f"force_{label}_eV_A"] = radial_forces
    result["minus_dEdr_sparse_eV_A"] = (-np.gradient(result["energy_sparse_eV"], distances)).tolist()
    result["max_energy_delta_eV"] = float(np.max(np.abs(np.asarray(result["energy_sparse_eV"]) - result["energy_full_eV"])))
    result["max_force_delta_eV_A"] = float(np.max(np.abs(np.asarray(result["force_sparse_eV_A"]) - result["force_full_eV_A"])))
    result["max_conservative_error_eV_A"] = float(np.max(np.abs(np.asarray(result["force_sparse_eV_A"]) - result["minus_dEdr_sparse_eV_A"])))
    (args.output_dir / "boundary_scan.json").write_text(json.dumps(result, indent=2) + "\n")

    apply_plot_style("icml")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    axes[0].plot(distances, result["energy_full_eV"], label="full dimer batch")
    axes[0].plot(distances, result["energy_sparse_eV"], "--", label="sparse padded batch")
    axes[0].set_ylabel("Energy (eV)")
    axes[1].plot(distances, result["force_sparse_eV_A"], label="returned sparse force")
    axes[1].plot(distances, result["minus_dEdr_sparse_eV_A"], "--", label=r"$-dE/dR$")
    axes[1].axvspan(5, 6, alpha=.12, label="physical handoff")
    axes[1].axvline(7, color="0.35", linestyle=":", label="buffered mask boundary")
    axes[1].set(xlabel="DCM–ACO COM distance (Å)", ylabel="Radial force (eV/Å)")
    for ax in axes: ax.grid(alpha=.18)
    legend_outside(fig, side="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / "boundary_scan.png", dpi=200, bbox_inches="tight")
    from mmml.utils.rich_report import print_colored_json

    print_colored_json({k: result[k] for k in result if k.startswith("max_")})

if __name__ == "__main__":
    main()
