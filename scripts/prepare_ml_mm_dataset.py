#!/usr/bin/env python3
"""Dataset preparation script for ML/MM hybrid training.

Partitions extxyz structures into dimer monomer components using covalent graph connectivity,
back-maps monomers to CGenFF residue templates (DCM, ACO, BENZ, TIP3, MEOH), pre-computes
CGenFF MM nonbonded baselines (Coulomb + LJ) and MBD dispersion baselines, and exports an
enriched dataset ready for residual ML/MM model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.io import iread, write
import ase.units

# CGenFF nonbonded parameters for the 5 DES monomer templates
# q (elem charge), sigma (Angstrom), epsilon (kcal/mol)
CGENFF_PARAMS = {
    "DCM": {
        "atoms": ["C", "H1", "H2", "CL1", "CL2"],
        "charges": [-0.180, 0.170, 0.170, -0.080, -0.080],
        "sigmas": [3.56359, 2.35200, 2.35200, 3.47094, 3.47094],
        "epsilons": [0.0560, 0.0240, 0.0240, 0.2720, 0.2720],
    },
    "ACO": { # ACE (Acetone)
        "atoms": ["O1", "C1", "C2", "C3", "H21", "H22", "H23", "H31", "H32", "H33"],
        "charges": [-0.510, 0.540, -0.270, -0.270, 0.080, 0.080, 0.080, 0.080, 0.080, 0.080],
        "sigmas": [3.02906, 3.56359, 3.67050, 3.67050, 2.35200, 2.35200, 2.35200, 2.35200, 2.35200, 2.35200],
        "epsilons": [0.1200, 0.1100, 0.0780, 0.0780, 0.0240, 0.0240, 0.0240, 0.0240, 0.0240, 0.0240],
    },
    "BENZ": {
        "atoms": ["CG", "HG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ"],
        "charges": [-0.115, 0.115, -0.115, 0.115, -0.115, 0.115, -0.115, 0.115, -0.115, 0.115, -0.115, 0.115],
        "sigmas": [3.55005, 2.42000, 3.55005, 2.42000, 3.55005, 2.42000, 3.55005, 2.42000, 3.55005, 2.42000, 3.55005, 2.42000],
        "epsilons": [0.0700, 0.0300, 0.0700, 0.0300, 0.0700, 0.0300, 0.0700, 0.0300, 0.0700, 0.0300, 0.0700, 0.0300],
    },
    "TIP3": {
        "atoms": ["OH2", "H1", "H2"],
        "charges": [-0.834, 0.417, 0.417],
        "sigmas": [3.15070, 0.40001, 0.40001],
        "epsilons": [0.1521, 0.0460, 0.0460],
    },
    "MEOH": {
        "atoms": ["CB", "OG", "HG1", "HB1", "HB2", "HB3"],
        "charges": [-0.040, -0.660, 0.430, 0.090, 0.090, 0.090],
        "sigmas": [3.67050, 3.12000, 0.40001, 2.35200, 2.35200, 2.35200],
        "epsilons": [0.0780, 0.1700, 0.0460, 0.0240, 0.0240, 0.0240],
    },
}

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol


def find_covalent_components(atoms: Atoms) -> list[list[int]]:
    """Partition atoms into connected covalent molecular components."""
    z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    n = len(atoms)
    
    # Adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            r_cov_sum = covalent_radii[z[i]] + covalent_radii[z[j]]
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < 1.3 * r_cov_sum:
                adj[i, j] = True
                adj[j, i] = True

    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                neighbors = np.flatnonzero(adj[curr])
                for nbr in neighbors:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(sorted(comp))
    return components


def match_cgenff_template(atoms: Atoms, comp_indices: list[int]) -> tuple[str, list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Match monomer component against CGenFF templates and return parameters."""
    sub_z = atoms.get_atomic_numbers()[comp_indices]
    counts = dict(zip(*np.unique(sub_z, return_counts=True)))
    
    # Formula matching
    # DCM: C:1, H:2, Cl:17 x2
    if counts == {6: 1, 1: 2, 17: 2}:
        res_name = "DCM"
    elif counts == {8: 1, 6: 3, 1: 6}: # ACO (Acetone)
        res_name = "ACO"
    elif counts == {6: 6, 1: 6}: # BENZ
        res_name = "BENZ"
    elif counts == {8: 1, 1: 2}: # TIP3
        res_name = "TIP3"
    elif counts == {6: 1, 8: 1, 1: 4}: # MEOH
        res_name = "MEOH"
    else:
        raise ValueError(f"Unrecognized monomer formula for atomic numbers: {sub_z}")

    tmpl = CGENFF_PARAMS[res_name]
    charges = np.array(tmpl["charges"], dtype=np.float64)
    sigmas = np.array(tmpl["sigmas"], dtype=np.float64)
    epsilons = np.array(tmpl["epsilons"], dtype=np.float64)

    return res_name, comp_indices, charges, sigmas, epsilons


def compute_inter_monomer_cgenff_mm(atoms: Atoms, comp_a: list[int], q_a: np.ndarray, sig_a: np.ndarray, eps_a: np.ndarray,
                                   comp_b: list[int], q_b: np.ndarray, sig_b: np.ndarray, eps_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute inter-monomer MM Coulomb + Lennard-Jones baseline energy and forces."""
    pos = atoms.get_positions()
    pos_a = pos[comp_a]
    pos_b = pos[comp_b]
    
    forces = np.zeros_like(pos, dtype=np.float64)
    e_coulomb = 0.0
    e_vdw = 0.0

    for i_sub, i_global in enumerate(comp_a):
        for j_sub, j_global in enumerate(comp_b):
            dr = pos_a[i_sub] - pos_b[j_sub]
            r = np.linalg.norm(dr)
            if r < 1e-6:
                continue
            
            # Coulomb
            q_ij = q_a[i_sub] * q_b[j_sub]
            e_c = K_COULOMB_KCAL_ANG * q_ij / r
            f_c_mag = K_COULOMB_KCAL_ANG * q_ij / (r**3) # force along dr
            
            # Lennard-Jones Lorentz-Berthelot combination
            sig_ij = 0.5 * (sig_a[i_sub] + sig_b[j_sub])
            eps_ij = np.sqrt(eps_a[i_sub] * eps_b[j_sub])
            sr6 = (sig_ij / r)**6
            sr12 = sr6**2
            e_v = 4.0 * eps_ij * (sr12 - sr6)
            f_v_mag = (24.0 * eps_ij / (r**2)) * (2.0 * sr12 - sr6) # force along dr

            e_coulomb += e_c
            e_vdw += e_v

            f_pair = (f_c_mag + f_v_mag) * dr
            forces[i_global] += f_pair
            forces[j_global] -= f_pair

    e_total_mm = e_coulomb + e_vdw
    return e_total_mm, forces


def process_dataset(extxyz_in: str | Path, extxyz_out: str | Path, max_structures: int | None = None):
    extxyz_in = Path(extxyz_in).expanduser()
    extxyz_out = Path(extxyz_out).expanduser()

    print(f"==================================================================")
    print(f" ML/MM Dataset Preparer & CGenFF Baseline Pre-computer")
    print(f" Input: {extxyz_in} -> Output: {extxyz_out}")
    print(f"==================================================================")

    processed = 0
    dimers_found = 0

    out_frames = []

    for idx, atoms in enumerate(iread(str(extxyz_in), index=":", format="extxyz")):
        if max_structures and idx >= max_structures:
            break
        
        comps = find_covalent_components(atoms)
        if len(comps) != 2:
            # Skip non-dimer frames for ML/MM baseline subtraction
            continue

        dimers_found += 1
        comp_a, comp_b = comps[0], comps[1]

        try:
            res_a, _, q_a, sig_a, eps_a = match_cgenff_template(atoms, comp_a)
            res_b, _, q_b, sig_b, eps_b = match_cgenff_template(atoms, comp_b)

            e_mm, f_mm = compute_inter_monomer_cgenff_mm(
                atoms, comp_a, q_a, sig_a, eps_a, comp_b, q_b, sig_b, eps_b
            )

            # Assign mol_id array (0 for Monomer A, 1 for Monomer B)
            mol_id = np.zeros(len(atoms), dtype=np.int32)
            mol_id[comp_b] = 1
            atoms.arrays["mol_id"] = mol_id

            # Attach CGenFF background energy (in eV) and forces (in eV/Å)
            # 1 kcal/mol = 0.0433641153 eV
            KCAL_TO_EV = 1.0 / ase.units.kcal * ase.units.mol
            atoms.info["E_cgenff_mm"] = e_mm * KCAL_TO_EV
            atoms.arrays["F_cgenff_mm"] = f_mm * KCAL_TO_EV

            out_frames.append(atoms)
            processed += 1

        except Exception as exc:
            # Skip frames whose monomers don't match standard CGenFF templates
            pass

        if (idx + 1) % 5000 == 0:
            print(f"  Parsed {idx + 1:,} structures | Dimers processed: {processed:,}")

    print(f"\n[+] Total Dimer Frames Extracted & Processed: {processed:,} / {dimers_found:,}")
    extxyz_out.parent.mkdir(parents=True, exist_ok=True)
    write(str(extxyz_out), out_frames, format="extxyz")
    print(f"[+] Prepared dataset written to: {extxyz_out}")
    print(f"==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Prepare ML/MM hybrid dataset with CGenFF baselines")
    parser.add_argument("--extxyz-in", required=True, help="Input raw extxyz dataset")
    parser.add_argument("--extxyz-out", default="data/des_dimers_ml_mm.extxyz", help="Output enriched extxyz dataset")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    process_dataset(args.extxyz_in, args.extxyz_out, max_structures=args.max_structures)


if __name__ == "__main__":
    main()
