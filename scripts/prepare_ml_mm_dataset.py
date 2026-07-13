#!/usr/bin/env python3
"""Dataset preparation script for ML/MM hybrid training with CGenFF topology back-mapping.

Fast Orbax cache reader & writer:
Partitions dataset structures into dimer monomer components using covalent graph connectivity,
back-maps monomers to official CGenFF residue templates in CGENFF.RES / top_all36_cgenff.rtf,
pre-computes CGenFF MM nonbonded baselines (Coulomb + LJ 6-12), and exports an enriched dataset
directly into a new Orbax cache and/or extxyz ready for residual ML/MM model training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from ase.data import covalent_radii
from ase.io import iread, write
import ase.units

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol
KCAL_TO_EV = 1.0 / ase.units.kcal * ase.units.mol

# Default CGenFF topology path
DEF_RTF_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "top_all36_cgenff.rtf"
DEF_PRM_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"
DEF_RES_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "CGENFF.RES"


def load_cgenff_nonbonded_params(prm_path: Path) -> dict[str, tuple[float, float]]:
    """Parse NONBONDED section from par_all36_cgenff.prm returning {atom_type: (epsilon, sigma)}."""
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


_CGENFF_TEMPLATES = load_cgenff_rtf_residues(DEF_RTF_PATH, DEF_PRM_PATH)


def find_covalent_components_fast(z: np.ndarray, pos: np.ndarray) -> list[list[int]]:
    """Partition atoms into connected covalent molecular components using fast numpy distance matrix."""
    n = len(z)
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


def match_cgenff_template_fast(z_sub: np.ndarray, comp_indices: list[int]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Match monomer component against registered CGenFF templates."""
    counts = dict(zip(*np.unique(z_sub, return_counts=True)))
    
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
        res_name = "GENERIC"

    if res_name in _CGENFF_TEMPLATES:
        tmpl = _CGENFF_TEMPLATES[res_name]
        charges = np.array(tmpl["charges"], dtype=np.float64)
        sigmas = np.array(tmpl["sigmas"], dtype=np.float64)
        epsilons = np.array(tmpl["epsilons"], dtype=np.float64)
    else:
        n = len(comp_indices)
        charges = np.zeros(n, dtype=np.float64)
        sigmas = np.array([2.0 * covalent_radii[zi] for zi in z_sub], dtype=np.float64)
        epsilons = np.full(n, 0.05, dtype=np.float64)

    return res_name, charges, sigmas, epsilons


def compute_inter_monomer_cgenff_mm_fast(pos: np.ndarray, comp_a: list[int], q_a: np.ndarray, sig_a: np.ndarray, eps_a: np.ndarray,
                                        comp_b: list[int], q_b: np.ndarray, sig_b: np.ndarray, eps_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute inter-monomer MM Coulomb + LJ baseline energy and forces in fast numpy."""
    pos_a = pos[comp_a]
    pos_b = pos[comp_b]
    forces = np.zeros_like(pos, dtype=np.float64)
    
    # Vectorized pairwise differences
    dr = pos_a[:, None, :] - pos_b[None, :, :]  # (N_a, N_b, 3)
    r = np.linalg.norm(dr, axis=-1)              # (N_a, N_b)
    
    q_ij = q_a[:, None] * q_b[None, :]
    sig_ij = 0.5 * (sig_a[:, None] + sig_b[None, :])
    eps_ij = np.sqrt(eps_a[:, None] * eps_b[None, :])
    
    # Coulomb
    e_coulomb = np.sum(K_COULOMB_KCAL_ANG * q_ij / r)
    f_c_mag = K_COULOMB_KCAL_ANG * q_ij / (r**3)
    
    # LJ
    sr6 = (sig_ij / r)**6
    sr12 = sr6**2
    e_vdw = np.sum(4.0 * eps_ij * (sr12 - sr6))
    f_v_mag = (24.0 * eps_ij / (r**2)) * (2.0 * sr12 - sr6)
    
    f_mag = f_c_mag + f_v_mag
    f_vec = dr * f_mag[:, :, None] # (N_a, N_b, 3)
    
    forces[comp_a] += np.sum(f_vec, axis=1)
    forces[comp_b] -= np.sum(f_vec, axis=0)

    e_total_ev = (e_coulomb + e_vdw) * KCAL_TO_EV
    forces_ev = forces * KCAL_TO_EV
    return e_total_ev, forces_ev


def process_orbax_cache(cache_dir: str | Path, output_cache: str | Path, max_structures: int | None = None):
    cache_dir = Path(cache_dir).expanduser()
    output_cache = Path(output_cache).expanduser()

    print(f"==================================================================")
    print(f" Fast Orbax Cache ML/MM Pre-computer & Topology Back-Mapper")
    print(f" Loaded {len(_CGENFF_TEMPLATES)} official CGenFF RESI templates")
    print(f" Source Cache: {cache_dir}")
    print(f" Target Cache: {output_cache}")
    print(f"==================================================================")

    data = ocp.PyTreeCheckpointer().restore(cache_dir)
    
    Z_all = np.asarray(data["Z"]).reshape(-1)
    R_all = np.asarray(data["R"]).reshape(-1, 3)
    F_all = np.asarray(data["F"]).reshape(-1, 3)
    offsets = np.asarray(data["mol_offsets"]).reshape(-1)
    E_all = np.asarray(data["E"]).reshape(-1, 1)
    N_all = np.asarray(data["N"]).reshape(-1, 1)
    Q_all = np.asarray(data["Q"]).reshape(-1, 1)
    S_all = np.asarray(data["S"]).reshape(-1, 1)
    D_all = np.asarray(data["D"]).reshape(-1, 3)

    n_total = len(N_all)
    if max_structures:
        n_total = min(n_total, max_structures)

    print(f"[+] Total Structures in Cache: {n_total:,}")
    t0 = time.time()

    kept_r = []
    kept_z = []
    kept_f = []
    kept_f_cgenff = []
    kept_e = []
    kept_e_cgenff = []
    kept_n = []
    kept_q = []
    kept_s = []
    kept_d = []
    kept_mol_id = []
    kept_offsets = [0]

    dimers_processed = 0

    for i in range(n_total):
        start = offsets[i]
        end = offsets[i + 1]
        z_struct = Z_all[start:end]
        r_struct = R_all[start:end]
        f_struct = F_all[start:end]

        comps = find_covalent_components_fast(z_struct, r_struct)
        if len(comps) != 2:
            continue

        comp_a, comp_b = comps[0], comps[1]
        
        try:
            res_a, q_a, sig_a, eps_a = match_cgenff_template_fast(z_struct[comp_a], comp_a)
            res_b, q_b, sig_b, eps_b = match_cgenff_template_fast(z_struct[comp_b], comp_b)

            e_mm, f_mm = compute_inter_monomer_cgenff_mm_fast(
                r_struct, comp_a, q_a, sig_a, eps_a, comp_b, q_b, sig_b, eps_b
            )

            n_atoms = len(z_struct)
            mol_id = np.zeros(n_atoms, dtype=np.int32)
            mol_id[comp_b] = 1

            kept_r.append(r_struct)
            kept_z.append(z_struct)
            kept_f.append(f_struct)
            kept_f_cgenff.append(f_mm)
            kept_e.append(E_all[i])
            kept_e_cgenff.append(e_mm)
            kept_n.append(n_atoms)
            kept_q.append(Q_all[i])
            kept_s.append(S_all[i])
            kept_d.append(D_all[i])
            kept_mol_id.append(mol_id)
            kept_offsets.append(kept_offsets[-1] + n_atoms)

            dimers_processed += 1

        except Exception:
            pass

        if (i + 1) % 100000 == 0:
            dt = time.time() - t0
            print(f"  Scanned {i + 1:,} structures | Dimers extracted: {dimers_processed:,} ({dt:.1f}s)")

    print(f"\n[+] Total Dimer Structures Prepared: {dimers_processed:,} in {time.time() - t0:.2f}s")

    output_data = {
        "R": np.concatenate(kept_r, axis=0),
        "Z": np.concatenate(kept_z, axis=0),
        "F": np.concatenate(kept_f, axis=0),
        "F_cgenff_mm": np.concatenate(kept_f_cgenff, axis=0),
        "mol_offsets": np.asarray(kept_offsets, dtype=np.int64),
        "E": np.asarray(kept_e, dtype=np.float64).reshape(-1, 1),
        "E_cgenff_mm": np.asarray(kept_e_cgenff, dtype=np.float64).reshape(-1, 1),
        "N": np.asarray(kept_n, dtype=np.int32).reshape(-1, 1),
        "Q": np.asarray(kept_q, dtype=np.float64).reshape(-1, 1),
        "S": np.asarray(kept_s, dtype=np.float64).reshape(-1, 1),
        "D": np.asarray(kept_d, dtype=np.float64).reshape(-1, 3),
        "mol_id": np.concatenate(kept_mol_id, axis=0),
    }
    output_data["metadata_n_structures"] = np.asarray(dimers_processed, dtype=np.int64)
    output_data["metadata_n_atoms_total"] = np.asarray(kept_offsets[-1], dtype=np.int64)
    output_data["metadata_max_atoms"] = np.asarray(max(kept_n), dtype=np.int32)

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed Orbax data cache to: {output_cache}")
    ocp.PyTreeCheckpointer().save(output_cache, output_data, force=True)
    print(f"[+] Prepared dataset successfully saved!")
    print(f"==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Fast Orbax Cache ML/MM dataset preparer")
    parser.add_argument("--cache-dir", required=True, help="Input source Orbax data cache directory")
    parser.add_argument("--output-cache", default="data/orbax_cache_des_ml_mm", help="Output destination Orbax cache directory")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    process_orbax_cache(args.cache_dir, args.output_cache, max_structures=args.max_structures)


if __name__ == "__main__":
    main()
