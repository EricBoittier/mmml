#!/usr/bin/env python3
"""High-performance multi-core dataset preparation script for ML/MM hybrid training.

Multi-threaded Orbax Cache Processor:
- Guarantees sum(cgenff_charges) == target_monomer_charge for exact charge conservation.
- Uses multiprocessing across all available CPU cores to process millions of frames in seconds.
- Stores atomic cgenff_type_idx, cgenff_charge, and mol_id for every atom in the Orbax cache,
  plus master CGenFF nonbonded parameter tables for dynamic JAX graph evaluation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
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

from ase.data import covalent_radii
import ase.units

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol
KCAL_TO_EV = 1.0 / ase.units.kcal * ase.units.mol

DEF_RTF_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "top_all36_cgenff.rtf"
DEF_PRM_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"


def load_cgenff_nonbonded_table(prm_path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Parse NONBONDED section from par_all36_cgenff.prm returning type_map, sigmas, epsilons."""
    nb_map = {}
    sigmas = []
    epsilons = []
    in_nb = False
    
    if not prm_path.exists():
        return {"DEFAULT": 0}, np.array([3.5], dtype=np.float64), np.array([0.05], dtype=np.float64)

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
                        if atom_type not in nb_map:
                            epsilon = abs(float(parts[2])) # kcal/mol
                            rmin_half = float(parts[3])    # Angstrom
                            sigma = rmin_half * 2.0 / (2.0**(1.0 / 6.0))
                            
                            idx = len(nb_map)
                            nb_map[atom_type] = idx
                            sigmas.append(sigma)
                            epsilons.append(epsilon)
                    except ValueError:
                        pass

    if "DEFAULT" not in nb_map:
        idx = len(nb_map)
        nb_map["DEFAULT"] = idx
        sigmas.append(3.5)
        epsilons.append(0.05)

    return nb_map, np.array(sigmas, dtype=np.float64), np.array(epsilons, dtype=np.float64)


def load_cgenff_rtf_residues(rtf_path: Path, nb_map: dict[str, int]) -> tuple[dict[str, dict], dict[tuple[tuple[int, int], ...], list[str]]]:
    """Parse all RESI blocks from top_all36_cgenff.rtf, indexing by name and elemental composition."""
    residues = {}
    composition_map = {}
    if not rtf_path.exists():
        return residues, composition_map

    current_resi = None
    resi_data = None
    default_idx = nb_map.get("DEFAULT", 0)

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
                    comp_key = tuple(sorted(dict(zip(*np.unique(resi_data["z_elements"], return_counts=True))).items()))
                    composition_map.setdefault(comp_key, []).append(current_resi)
                current_resi = parts[1]
                resi_data = {"name": current_resi, "atoms": [], "type_indices": [], "charges": [], "z_elements": []}
            elif parts[0] == "ATOM" and current_resi:
                atom_name = parts[1]
                atom_type = parts[2]
                charge = float(parts[3])
                t_idx = nb_map.get(atom_type, default_idx)
                
                # Infer atomic element from atom_type / atom_name
                first_letter = atom_name[0].upper()
                if first_letter == "C" and len(atom_name) > 1 and atom_name[1].lower() in ("l", "r"):
                    elem_sym = "CL" if atom_name[1].lower() == "l" else "BR"
                elif first_letter == "B" and len(atom_name) > 1 and atom_name[1].lower() == "r":
                    elem_sym = "BR"
                elif first_letter in chemical_symbols:
                    elem_sym = first_letter
                else:
                    elem_sym = atom_type[0].upper()
                
                z_elem = atomic_numbers.get(elem_sym, 6)
                resi_data["atoms"].append(atom_name)
                resi_data["type_indices"].append(t_idx)
                resi_data["charges"].append(charge)
                resi_data["z_elements"].append(z_elem)

        if current_resi and resi_data and resi_data["atoms"]:
            residues[current_resi] = resi_data
            comp_key = tuple(sorted(dict(zip(*np.unique(resi_data["z_elements"], return_counts=True))).items()))
            composition_map.setdefault(comp_key, []).append(current_resi)

    return residues, composition_map


_NB_MAP, _CGENFF_SIGMAS, _CGENFF_EPSILONS = load_cgenff_nonbonded_table(DEF_PRM_PATH)
_CGENFF_RESIDUES, _COMPOSITION_MAP = load_cgenff_rtf_residues(DEF_RTF_PATH, _NB_MAP)


def find_covalent_components_fast(z: np.ndarray, pos: np.ndarray) -> list[list[int]]:
    """Partition atoms into connected covalent molecular components using fast numpy distance matrix."""
    n = len(z)
    dr = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(dr, axis=-1)
    
    r_cov = covalent_radii[z]
    r_sum = 1.3 * (r_cov[:, None] + r_cov[None, :])
    adj = dist < r_sum
    np.fill_diagonal(adj, False)

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


def match_cgenff_template_fast(z_sub: np.ndarray, comp_indices: list[int], target_charge: float = 0.0) -> tuple[str, np.ndarray, np.ndarray]:
    """Match monomer component against CGenFF templates by elemental composition and enforce sum(charges) == target_charge exactly."""
    counts = dict(zip(*np.unique(z_sub, return_counts=True)))
    comp_key = tuple(sorted(counts.items()))
    
    # 1. Fast match for common DES monomers
    if counts == {6: 1, 1: 2, 17: 2} and "DCM" in _CGENFF_RESIDUES:
        res_name = "DCM"
    elif counts == {8: 1, 6: 3, 1: 6} and "ACO" in _CGENFF_RESIDUES:
        res_name = "ACO"
    elif counts == {6: 6, 1: 6} and "BENZ" in _CGENFF_RESIDUES:
        res_name = "BENZ"
    elif counts == {8: 1, 1: 2} and "TIP3" in _CGENFF_RESIDUES:
        res_name = "TIP3"
    elif counts == {6: 1, 8: 1, 1: 4} and "MEOH" in _CGENFF_RESIDUES:
        res_name = "MEOH"
    elif comp_key in _COMPOSITION_MAP:
        # Match from elemental composition index
        res_name = _COMPOSITION_MAP[comp_key][0]
    else:
        raise KeyError(f"No CGenFF RTF template found for elemental composition key: {comp_key}")

    tmpl = _CGENFF_RESIDUES[res_name]
    type_indices = np.array(tmpl["type_indices"], dtype=np.int32)
    charges = np.array(tmpl["charges"], dtype=np.float64)

    # STRICT CHARGE CONSERVATION GUARD:
    # Adjust charges uniformly so sum(charges) matches target_charge exactly (to float64 precision)
    n_atoms = len(charges)
    if n_atoms > 0:
        charge_diff = target_charge - np.sum(charges)
        if abs(charge_diff) > 1e-12:
            charges = charges + (charge_diff / n_atoms)

    return res_name, type_indices, charges


def compute_inter_monomer_cgenff_mm_fast(pos: np.ndarray, comp_a: list[int], t_a: np.ndarray, q_a: np.ndarray,
                                        comp_b: list[int], t_b: np.ndarray, q_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute inter-monomer MM Coulomb + LJ baseline energy and forces in fast numpy."""
    pos_a = pos[comp_a]
    pos_b = pos[comp_b]
    forces = np.zeros_like(pos, dtype=np.float64)
    
    sig_a = _CGENFF_SIGMAS[t_a]
    eps_a = _CGENFF_EPSILONS[t_a]
    sig_b = _CGENFF_SIGMAS[t_b]
    eps_b = _CGENFF_EPSILONS[t_b]

    # Vectorized pairwise differences
    dr = pos_a[:, None, :] - pos_b[None, :, :]  # (N_a, N_b, 3)
    r = np.linalg.norm(dr, axis=-1)              # (N_a, N_b)
    
    q_ij = q_a[:, None] * q_b[None, :]
    sig_ij = 0.5 * (sig_a[:, None] + sig_b[None, :])
    eps_ij = np.sqrt(eps_a[:, None] * eps_b[None, :])
    
    # Coulomb
    r_coulomb = np.maximum(r, 1e-6)
    e_coulomb = np.sum(K_COULOMB_KCAL_ANG * q_ij / r_coulomb)
    f_c_mag = K_COULOMB_KCAL_ANG * q_ij / (r_coulomb**3)
    
    # LJ with soft-core distance clamping at r < 0.8 * sig_ij
    r_vdw = np.maximum(r, 0.8 * sig_ij)
    sr6 = (sig_ij / r_vdw)**6
    sr12 = sr6**2
    e_vdw = np.sum(4.0 * eps_ij * (sr12 - sr6))
    f_v_mag = (24.0 * eps_ij / (r_vdw**2)) * (2.0 * sr12 - sr6)
    
    f_mag = f_c_mag + f_v_mag
    f_vec = dr * f_mag[:, :, None]
    
    forces[comp_a] += np.sum(f_vec, axis=1)
    forces[comp_b] -= np.sum(f_vec, axis=0)

    e_total_ev = (e_coulomb + e_vdw) * KCAL_TO_EV
    forces_ev = forces * KCAL_TO_EV
    return e_total_ev, forces_ev


def process_single_frame(args_tuple):
    """Worker function for parallel frame processing across CPU cores."""
    z_struct, r_struct, f_struct, energy_i, q_i, s_i, d_i = args_tuple
    
    comps = find_covalent_components_fast(z_struct, r_struct)
    if len(comps) != 2:
        return None

    comp_a, comp_b = comps[0], comps[1]
    
    try:
        # Neutral monomer target charge assumption for neutral DES dimers (0.0)
        res_a, t_a, q_a = match_cgenff_template_fast(z_struct[comp_a], comp_a, target_charge=0.0)
        res_b, t_b, q_b = match_cgenff_template_fast(z_struct[comp_b], comp_b, target_charge=0.0)

        e_mm, f_mm = compute_inter_monomer_cgenff_mm_fast(
            r_struct, comp_a, t_a, q_a, comp_b, t_b, q_b
        )

        n_atoms = len(z_struct)
        mol_id = np.zeros(n_atoms, dtype=np.int32)
        mol_id[comp_b] = 1

        cgenff_type = np.zeros(n_atoms, dtype=np.int32)
        cgenff_charge = np.zeros(n_atoms, dtype=np.float64)
        cgenff_type[comp_a] = t_a
        cgenff_type[comp_b] = t_b
        cgenff_charge[comp_a] = q_a
        cgenff_charge[comp_b] = q_b

        return (
            r_struct, z_struct, f_struct, f_mm, energy_i, e_mm,
            n_atoms, q_i, s_i, d_i, mol_id, cgenff_type, cgenff_charge
        )
    except Exception:
        return None


def process_orbax_cache(cache_dir: str | Path, output_cache: str | Path, max_structures: int | None = None, num_workers: int | None = None):
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_cache = Path(output_cache).expanduser().resolve()
    workers = num_workers or min(mp.cpu_count(), 32)
    
    # Use spawn to prevent JAX multithreaded os.fork() deadlocks in Python 3.13
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"==================================================================")
    print(f" Multi-Core Orbax Cache ML/MM Pre-computer ({workers} CPU Workers)")
    print(f" Master Nonbonded Types: {len(_NB_MAP):,} types | Registered RESI: {len(_CGENFF_RESIDUES):,}")
    print(f" Strict Charge Conservation: sum(cgenff_charge) == target_charge (0.0 e)")
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

    print(f"[+] Total Structures to Scan: {n_total:,}")
    t0 = time.time()

    # Generator for multiprocessing
    def frame_generator():
        for i in range(n_total):
            start = offsets[i]
            end = offsets[i + 1]
            yield (
                Z_all[start:end], R_all[start:end], F_all[start:end],
                E_all[i], Q_all[i], S_all[i], D_all[i]
            )

    kept_r, kept_z, kept_f, kept_f_cgenff = [], [], [], []
    kept_e, kept_e_cgenff, kept_n, kept_q = [], [], [], []
    kept_s, kept_d, kept_mol_id = [], [], []
    kept_cgenff_type, kept_cgenff_charge = [], []
    kept_offsets = [0]

    dimers_processed = 0

    with mp.Pool(processes=workers) as pool:
        for res in pool.imap(process_single_frame, frame_generator(), chunksize=5000):
            if res is None:
                continue

            (r_struct, z_struct, f_struct, f_mm, energy_i, e_mm,
             n_atoms, q_i, s_i, d_i, mol_id, cgenff_type, cgenff_charge) = res

            kept_r.append(r_struct)
            kept_z.append(z_struct)
            kept_f.append(f_struct)
            kept_f_cgenff.append(f_mm)
            kept_e.append(energy_i)
            kept_e_cgenff.append(e_mm)
            kept_n.append(n_atoms)
            kept_q.append(q_i)
            kept_s.append(s_i)
            kept_d.append(d_i)
            kept_mol_id.append(mol_id)
            kept_cgenff_type.append(cgenff_type)
            kept_cgenff_charge.append(cgenff_charge)
            kept_offsets.append(kept_offsets[-1] + n_atoms)

            dimers_processed += 1
            if dimers_processed % 500000 == 0:
                dt = time.time() - t0
                rate = dimers_processed / dt
                print(f"  Processed {dimers_processed:,} dimers ({rate:.0f} frames/sec)")

    dt = time.time() - t0
    print(f"\n[+] Total Dimer Structures Prepared: {dimers_processed:,} in {dt:.2f}s ({dimers_processed / dt:.0f} frames/sec)")

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
        "cgenff_type_idx": np.concatenate(kept_cgenff_type, axis=0),
        "cgenff_charge": np.concatenate(kept_cgenff_charge, axis=0),
        "cgenff_master_sigmas": _CGENFF_SIGMAS,
        "cgenff_master_epsilons": _CGENFF_EPSILONS,
    }
    output_data["metadata_n_structures"] = np.asarray(dimers_processed, dtype=np.int64)
    output_data["metadata_n_atoms_total"] = np.asarray(kept_offsets[-1], dtype=np.int64)
    output_data["metadata_max_atoms"] = np.asarray(max(kept_n), dtype=np.int32)

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving enriched Orbax data cache to: {output_cache}")
    ocp.PyTreeCheckpointer().save(output_cache, output_data, force=True)
    print(f"[+] Prepared dataset successfully saved!")
    print(f"==================================================================")


def main():
    parser = argparse.ArgumentParser(description="Multi-Core Orbax Cache ML/MM dataset preparer")
    parser.add_argument("--cache-dir", required=True, help="Input source Orbax data cache directory")
    parser.add_argument("--output-cache", default="data/orbax_cache_des_ml_mm", help="Output destination Orbax cache directory")
    parser.add_argument("--max-structures", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--num-workers", type=int, default=None, help="CPU multiprocessing pool size (default: auto-detect)")
    args = parser.parse_args()

    process_orbax_cache(args.cache_dir, args.output_cache, max_structures=args.max_structures, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
