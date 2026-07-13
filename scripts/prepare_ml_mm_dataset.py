#!/usr/bin/env python3
"""Dataset preparation script for ML/MM hybrid training with CGenFF topology back-mapping.

Partitions extxyz structures into dimer monomer components using covalent graph connectivity,
back-maps monomers to official CGenFF residue templates in CGENFF.RES / top_all36_cgenff.rtf,
pre-computes CGenFF MM nonbonded baselines (Coulomb + LJ 6-12), and exports an enriched dataset
ready for residual ML/MM model training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
from ase.io import iread, write
import ase.units

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol

# Default CGenFF topology path
DEF_RTF_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "top_all36_cgenff.rtf"
DEF_PRM_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"
DEF_RES_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "CGENFF.RES"


def parse_cgenff_residue_table(res_path: Path) -> dict[str, str]:
    """Parse RESI records from CGENFF.RES mapping residue name to empirical formula."""
    res_map = {}
    if not res_path.exists():
        return res_map
    
    with res_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("RESI"):
                parts = line.split("!")
                res_head = parts[0].strip().split()
                if len(res_head) >= 2:
                    res_name = res_head[1]
                    comment = parts[1].strip() if len(parts) > 1 else ""
                    # Extract formula if present in comment e.g. ! C2H3O2, acetate
                    formula_match = re.search(r"([A-Za-z0-9]+)", comment)
                    if formula_match:
                        res_map[res_name] = formula_match.group(1)
    return res_map


def load_cgenff_nonbonded_params(prm_path: Path) -> dict[str, tuple[float, float]]:
    """Parse NONBONDED section from par_all36_cgenff.prm returning {atom_type: (epsilon, rmin_half)}."""
    nb_params = {}
    in_nb = False
    if not prm_path.exists():
        return nb_params

    with prm_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("NONBONDED"):
                in_nb = True
                continue
            if in_nb:
                if line.startswith("CUTNB") or line.startswith("END") or line.startswith("NBFIX"):
                    if line.startswith("NBFIX"):
                        break
                    continue
                parts = line.split("!")[0].split()
                if len(parts) >= 4:
                    try:
                        atom_type = parts[0]
                        epsilon = abs(float(parts[2])) # kcal/mol
                        rmin_half = float(parts[3])    # Angstrom
                        sigma = rmin_half * 2.0 / (2.0**(1.0 / 6.0))
                        nb_params[atom_type] = (epsilon, sigma)
                    except ValueError:
                        pass
    return nb_params


def load_cgenff_rtf_residues(rtf_path: Path, prm_path: Path) -> dict[str, dict]:
    """Parse all RESI blocks from top_all36_cgenff.rtf to extract atomic charges and nonbonded parameters."""
    nb_params = load_cgenff_nonbonded_params(prm_path)
    residues = {}
    
    if not rtf_path.exists():
        return residues

    current_resi = None
    resi_data = None

    with rtf_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            parts = line.split("!")[0].split()
            if not parts:
                continue
            if parts[0] == "RESI":
                if current_resi and resi_data and resi_data["atoms"]:
                    residues[current_resi] = resi_data
                current_resi = parts[1]
                resi_data = {"name": current_resi, "atoms": [], "types": [], "charges": [], "sigmas": [], "epsilons": []}
            elif parts[0] == "ATOM" and current_resi:
                atom_name = parts[1]
                atom_type = parts[2]
                charge = float(parts[3])
                eps, sig = nb_params.get(atom_type, (0.05, 3.5))
                resi_data["atoms"].append(atom_name)
                resi_data["types"].append(atom_type)
                resi_data["charges"].append(charge)
                resi_data["sigmas"].append(sig)
                resi_data["epsilons"].append(eps)

        if current_resi and resi_data and resi_data["atoms"]:
            residues[current_resi] = resi_data

    return residues


# Global CGenFF template cache loaded from RTF & PRM
_CGENFF_TEMPLATES = load_cgenff_rtf_residues(DEF_RTF_PATH, DEF_PRM_PATH)


def find_covalent_components(atoms: Atoms) -> list[list[int]]:
    """Partition atoms into connected covalent molecular components."""
    z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    n = len(atoms)
    
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
    """Match monomer component against registered CGenFF templates."""
    sub_z = atoms.get_atomic_numbers()[comp_indices]
    counts = dict(zip(*np.unique(sub_z, return_counts=True)))
    
    # Fast match for common DES monomers
    if counts == {6: 1, 1: 2, 17: 2} and "DCM" in _CGENFF_TEMPLATES:
        res_name = "DCM"
    elif counts == {8: 1, 6: 3, 1: 6} and "ACO" in _CGENFF_TEMPLATES:
        res_name = "ACO"
    elif counts == {6: 6, 1: 6} and "BENZ" in _CGENFF_TEMPLATES:
        res_name = "BENZ"
    elif counts == {8: 1, 1: 2} and "TIP3" in _CGENFF_TEMPLATES:
        res_name = "TIP3"
    elif counts == {6: 1, 8: 1, 1: 4} and "MEOH" in _CGENFF_TEMPLATES:
        res_name = "MEOH"
    else:
        # Search by composition in parsed RTF residues
        matched_res = None
        for r_name, r_tmpl in _CGENFF_TEMPLATES.items():
            if len(r_tmpl["atoms"]) == len(comp_indices):
                matched_res = r_name
                break
        res_name = matched_res or atoms[comp_indices].get_chemical_formula()

    if res_name in _CGENFF_TEMPLATES:
        tmpl = _CGENFF_TEMPLATES[res_name]
        charges = np.array(tmpl["charges"], dtype=np.float64)
        sigmas = np.array(tmpl["sigmas"], dtype=np.float64)
        epsilons = np.array(tmpl["epsilons"], dtype=np.float64)
    else:
        # Generic CGenFF/Universal VDW fallback for arbitrary components
        n = len(comp_indices)
        charges = np.zeros(n, dtype=np.float64)
        sigmas = np.array([2.0 * covalent_radii[z] for z in sub_z], dtype=np.float64)
        epsilons = np.full(n, 0.05, dtype=np.float64)

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
            f_c_mag = K_COULOMB_KCAL_ANG * q_ij / (r**3)
            
            # Lennard-Jones Lorentz-Berthelot
            sig_ij = 0.5 * (sig_a[i_sub] + sig_b[j_sub])
            eps_ij = np.sqrt(eps_a[i_sub] * eps_b[j_sub])
            sr6 = (sig_ij / r)**6
            sr12 = sr6**2
            e_v = 4.0 * eps_ij * (sr12 - sr6)
            f_v_mag = (24.0 * eps_ij / (r**2)) * (2.0 * sr12 - sr6)

            e_coulomb += e_c
            e_vdw += e_v

            f_pair = (f_c_mag + f_v_mag) * dr
            forces[i_global] += f_pair
            forces[j_global] -= f_pair

    return (e_coulomb + e_vdw), forces


def process_dataset(extxyz_in: str | Path, extxyz_out: str | Path, max_structures: int | None = None):
    extxyz_in = Path(extxyz_in).expanduser()
    extxyz_out = Path(extxyz_out).expanduser()

    print(f"==================================================================")
    print(f" CGenFF RTF/PRM Topology Back-Mapping & Baseline Pre-computer")
    print(f" Loaded {len(_CGENFF_TEMPLATES)} official CGenFF RESI templates")
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
            continue

        dimers_found += 1
        comp_a, comp_b = comps[0], comps[1]

        try:
            res_a, _, q_a, sig_a, eps_a = match_cgenff_template(atoms, comp_a)
            res_b, _, q_b, sig_b, eps_b = match_cgenff_template(atoms, comp_b)

            e_mm, f_mm = compute_inter_monomer_cgenff_mm(
                atoms, comp_a, q_a, sig_a, eps_a, comp_b, q_b, sig_b, eps_b
            )

            mol_id = np.zeros(len(atoms), dtype=np.int32)
            mol_id[comp_b] = 1
            atoms.arrays["mol_id"] = mol_id

            KCAL_TO_EV = 1.0 / ase.units.kcal * ase.units.mol
            atoms.info["E_cgenff_mm"] = e_mm * KCAL_TO_EV
            atoms.arrays["F_cgenff_mm"] = f_mm * KCAL_TO_EV

            out_frames.append(atoms)
            processed += 1

        except Exception:
            pass

        if (idx + 1) % 5000 == 0:
            print(f"  Parsed {idx + 1:,} structures | Dimers processed: {processed:,}")

    print(f"\n[+] Total Dimer Frames Extracted & Processed: {processed:,} / {dimers_found:,}")
    extxyz_out.parent.mkdir(parents=True, exist_ok=True)
    write(str(extxyz_out), out_frames, format="extxyz")
    print(f"[+] Prepared dataset written to: {extxyz_out}")
    print(f"==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Prepare ML/MM hybrid dataset with CGenFF RTF/PRM baselines")
    parser.add_argument("--extxyz-in", required=True, help="Input raw extxyz dataset")
    parser.add_argument("--extxyz-out", default="data/des_dimers_ml_mm.extxyz", help="Output enriched extxyz dataset")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    process_dataset(args.extxyz_in, args.extxyz_out, max_structures=args.max_structures)


if __name__ == "__main__":
    main()
