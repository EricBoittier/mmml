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
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ase.data import covalent_radii, atomic_numbers
import ase.units

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol
# kcal/mol -> eV (0.0433641...). The reciprocal (23.06) is the eV -> kcal/mol factor;
# using it here inflated E_cgenff_mm/F_cgenff_mm by 531.8x.
KCAL_TO_EV = ase.units.kcal / ase.units.mol

DEF_RTF_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "top_all36_cgenff.rtf"
DEF_PRM_PATH = _REPO_ROOT / "mmml" / "data" / "charmm" / "par_all36_cgenff.prm"

# Guard so module-level warnings only print once in the main process, not in every worker.
# Evaluated at import time: spawned pool workers have names like 'SpawnPoolWorker-N'.
_IS_WORKER = mp.current_process().name != "MainProcess"


def load_cgenff_nonbonded_table(prm_path: Path) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Parse NONBONDED section from par_all36_cgenff.prm returning type_map, sigmas, epsilons.
    
    Raises ValueError if epsilon or sigma is exactly 0.0 for any type (sentinel for bad parse).
    """
    nb_map = {}
    sigmas = []
    epsilons = []
    in_nb = False
    
    if not prm_path.exists():
        raise FileNotFoundError(f"CGenFF parameter file not found: {prm_path}")

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
                            epsilon = abs(float(parts[2]))  # kcal/mol
                            rmin_half = float(parts[3])     # Angstrom (Rmin/2)
                            sigma = rmin_half * 2.0 / (2.0 ** (1.0 / 6.0))
                            idx = len(nb_map)
                            nb_map[atom_type] = idx
                            sigmas.append(sigma)
                            epsilons.append(epsilon)
                    except (ValueError, IndexError):
                        # Skip non-atom lines (CUTNB, header lines, etc.)
                        pass

    if not nb_map:
        raise RuntimeError(f"No NONBONDED entries found in {prm_path}")

    # Sentinel DEFAULT type for unmapped types — zero values deliberately visible
    nb_map["DEFAULT"] = len(nb_map)
    sigmas.append(0.0)    # deliberately zero → produces zero LJ → clearly wrong in diagnostics
    epsilons.append(0.0)  # deliberately zero → produces zero LJ → clearly wrong in diagnostics

    # Post-parse sanity check: warn about any non-DEFAULT types with zero epsilon/sigma
    sig_arr = np.array(sigmas, dtype=np.float64)
    eps_arr = np.array(epsilons, dtype=np.float64)
    nb_map["DEFAULT"]
    bad_types = [t for t, i in nb_map.items() if t != "DEFAULT" and (sig_arr[i] == 0.0 or eps_arr[i] == 0.0)]
    if bad_types and not _IS_WORKER:
        print(f"[WARNING] {len(bad_types)} CGenFF atom types parsed with zero sigma or epsilon: {bad_types[:10]}")
        print(f"  (LPH is a lone-pair pseudo-atom with zero LJ by design — expected and harmless)")

    return nb_map, sig_arr, eps_arr


def _parse_mass_element_map(rtf_path: Path) -> dict[str, int]:
    """Parse MASS records from top_all36_cgenff.rtf to get authoritative atom_type -> atomic_number mapping.
    
    MASS line format: MASS  -1  HGA1  1.00800 H ! comment
    The 5th field (index 4) is the element symbol — this is the ground truth.
    If the element field is absent (old RTF format), fall back to mass-based inference.
    """
    _MASS_BY_APPROX = {
        1.008: 1, 4.003: 2, 6.941: 3, 9.012: 4, 10.811: 5, 12.011: 6,
        14.007: 7, 15.999: 8, 18.998: 9, 20.180: 10, 22.990: 11, 24.305: 12,
        26.982: 13, 28.086: 14, 30.974: 15, 32.065: 16, 35.453: 17, 39.948: 18,
        79.904: 35, 126.904: 53,
    }
    type_to_z = {}
    if not rtf_path.exists():
        return type_to_z
    with rtf_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("MASS"):
                continue
            parts = line.split("!")[0].split()
            if len(parts) < 4:
                continue
            atom_type = parts[2]
            try:
                mass = float(parts[3])
            except ValueError:
                continue
            # Prefer explicit element symbol in field 5
            if len(parts) >= 5 and parts[4].isalpha() and len(parts[4]) <= 2:
                elem_sym = parts[4].capitalize()
                z = atomic_numbers.get(elem_sym, 0)
            else:
                # Fall back to mass lookup (rounded to nearest tabulated mass)
                closest = min(_MASS_BY_APPROX.keys(), key=lambda m: abs(m - mass))
                z = _MASS_BY_APPROX[closest] if abs(closest - mass) < 1.0 else 6
            if z > 0:
                type_to_z[atom_type] = z
    return type_to_z


def load_cgenff_rtf_residues(
    rtf_path: Path,
    nb_map: dict[str, int],
    type_to_z: dict[str, int],
) -> tuple[dict[str, dict], dict[tuple[tuple[int, int], ...], list[str]], dict[tuple, list[str]]]:
    """Parse all RESI blocks from top_all36_cgenff.rtf using MASS-record element mapping.
    
    Returns:
        residues: dict[resi_name -> resi_data]
        composition_map: dict[comp_key -> [resi_name, ...]]  (may have collisions for isomers)
        collision_map: dict[comp_key -> [resi_name, ...]]    (only entries with >1 name = isomers)
    """
    residues = {}
    composition_map = {}
    if not rtf_path.exists():
        raise FileNotFoundError(f"CGenFF RTF file not found: {rtf_path}")

    default_idx = nb_map["DEFAULT"]
    current_resi = None
    resi_data = None
    bad_type_count = 0

    def _finalize_resi():
        """Store completed RESI block and index by composition."""
        nonlocal bad_type_count
        if not (current_resi and resi_data and resi_data["atoms"]):
            return
        # Validate: no atom should have the DEFAULT sentinel type index
        for i, (aname, tidx) in enumerate(zip(resi_data["atoms"], resi_data["type_indices"])):
            if tidx == default_idx:
                bad_type_count += 1
        residues[current_resi] = resi_data
        z_arr = np.array(resi_data["z_elements"], dtype=np.int32)
        comp_key = tuple(sorted(dict(zip(*np.unique(z_arr, return_counts=True))).items()))
        composition_map.setdefault(comp_key, []).append(current_resi)

    with rtf_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            parts = line.split("!")[0].split()
            if not parts:
                continue
            if parts[0] == "RESI":
                _finalize_resi()
                current_resi = parts[1]
                resi_data = {
                    "name": current_resi,
                    "atoms": [],
                    "type_indices": [],
                    "charges": [],
                    "z_elements": [],
                    "bonds": [],
                }
            elif parts[0] == "ATOM" and current_resi:
                atom_name = parts[1]
                atom_type = parts[2]
                try:
                    charge = float(parts[3])
                except (ValueError, IndexError):
                    charge = 0.0
                t_idx = nb_map.get(atom_type, default_idx)
                # Use MASS-record authoritative element mapping
                z_elem = type_to_z.get(atom_type, 6)
                resi_data["atoms"].append(atom_name)
                resi_data["type_indices"].append(t_idx)
                resi_data["charges"].append(charge)
                resi_data["z_elements"].append(z_elem)
            elif parts[0] in {"BOND", "DOUBLE", "TRIPLE"} and current_resi:
                # CHARMM lists bonded atom-name pairs after the directive.  Skip
                # cross-residue references (``+N``/``-C``) and TIP3's artificial
                # H-H SHAKE constraint: neither describes the covalent graph used
                # to identify atoms in an isolated geometry.
                names = parts[1:]
                for a, b in zip(names[0::2], names[1::2]):
                    if a.startswith(("+", "-")) or b.startswith(("+", "-")):
                        continue
                    resi_data["bonds"].append((a, b))
        _finalize_resi()

    collision_map = {k: v for k, v in composition_map.items() if len(v) > 1}

    if bad_type_count > 0:
        print(f"[WARNING] {bad_type_count} ATOM records mapped to DEFAULT sentinel type (unmapped atom_type in PRM). "
              f"These atoms will have zero LJ parameters!")
    if collision_map and not _IS_WORKER:
        n_collisions = sum(len(v) for v in collision_map.values())
        print(f"[WARNING] {len(collision_map)} elemental composition keys map to multiple RESI templates ({n_collisions} total). "
              f"Isomer matching via composition alone is ambiguous — SMILES lookup will be used where available.")

    return residues, composition_map, collision_map


_NB_MAP, _CGENFF_SIGMAS, _CGENFF_EPSILONS = load_cgenff_nonbonded_table(DEF_PRM_PATH)
_CGENFF_TYPE_TO_Z = _parse_mass_element_map(DEF_RTF_PATH)
_CGENFF_RESIDUES, _COMPOSITION_MAP, _COLLISION_MAP = load_cgenff_rtf_residues(DEF_RTF_PATH, _NB_MAP, _CGENFF_TYPE_TO_Z)


# ─── Canonical SMILES → CGenFF RESI name lookup for all DES-S66 molecules ──────
# Canonical SMILES generated with RDKit (MolToSmiles(MolFromSmiles(smi))) for each molecule
# in the DES-S66 dataset. Covers all constitutional isomers unambiguously.
DES_SMILES_TO_RESI: dict[str, str] = {
    # Water, noble gases, diatomics
    "O": "TIP3",
    "[H][H]": "TIP3",        # H2 - no template, use TIP3 as placeholder
    "[Ne]": "TIP3",          # noble gases - no LJ params needed
    "[Ar]": "TIP3",
    "[Kr]": "TIP3",
    "[Xe]": "TIP3",
    "N": "AMM1",             # ammonia NH3
    # Alkanes
    "CC": "ETHA",
    "CCC": "PRPA",
    "CCCC": "BUTA",
    "CC(C)C": "IBUT",
    "CCCCC": "PENT",
    "CCCCCC": "HEXA",
    # Cycloalkanes
    "C1CCCC1": "CPEN",
    "C1CCCCC1": "CHX",       # cyclohexane - use CPEN if CHX absent
    # Alkenes and alkynes
    "C=C": "ETHE",
    "CC=C": "PRPE",
    "CC=CC": "BTE2",
    "CCC=C": "BTE1",
    "C#C": "ETHE",           # acetylene - no perfect match, use ETHE as placeholder
    "CC#CC": "DIPE",         # 2-butyne
    "CCC#C": "BTE1",         # 1-butyne
    # Alcohols
    "CO": "MEOH",
    "CCO": "ETOH",
    "CCCO": "PRO2",          # propanol
    "CC(O)C": "PRO2",        # isopropanol
    "OCCCCO": "ETOH",        # 1,4-butanediol - no perfect, use ETOH
    "OC1CCCC1": "PRO2",      # cyclopentanol
    "OC1CCCCC1": "PRO2",     # cyclohexanol
    "Oc1ccccc1": "PHEN",     # phenol
    # Ethers
    "COC": "THF",            # dimethyl ether - use THF
    "CCOC": "THF",           # ethyl methyl ether
    "CCCOC": "THF",          # propyl methyl ether
    "COCOC": "THF",
    "COCOCC": "THF",
    "C1CCCOC1": "THF",       # tetrahydropyran
    "C1CCOCC1": "THF",       # 1,4-dioxane → no match use THF
    "O1CCOCC1": "THF",       # 1,4-dioxane
    "O1COCOC1": "THF",       # 1,3-dioxolane
    "C1CCOCO1": "THF",
    # Aldehydes & ketones
    "CC=O": "AALD",          # acetaldehyde
    "CCC=O": "PALD",         # propanal
    "C=O": "FORM",           # formaldehyde (closest: FORM is formamide, use AALD)
    "CC(C)=O": "ACO",        # acetone
    # Carboxylic acids & esters
    "OC=O": "ACEH",          # formic acid
    "CC(O)=O": "ACEH",       # acetic acid
    "COC(=O)C": "MAS",       # methyl acetate
    "CCOC(=O)C": "ETAC",     # ethyl acetate
    "CCOC=O": "MPRO",        # ethyl formate
    "CNC(=O)C": "NMA",       # N-methylacetamide
    "O=CN(C)C": "DMAM",      # DMF approx
    "CC(=O)N(C)C": "NMA",    # N,N-dimethylacetamide
    "CC(=O)N": "ACEM",       # acetamide
    "CNC=O": "FORM",         # N-methylformamide
    "CCC(=O)N": "PRAM",      # propionamide
    # Halogenated
    "CCl": "CALD",           # chloromethane / methyl chloride
    "ClCCl": "DCM",          # dichloromethane
    "ClC(Cl)Cl": "CHAL",     # chloroform
    "CC(Cl)(Cl)Cl": "CHAL",  # trichloromethane
    "CCCCl": "DCM",          # 1-chloropropane
    "ClCCCl": "DCM",         # 1,2-dichloroethane
    "CBr": "CALD",           # bromomethane
    "CC(Br)Br": "DCM",       # 1,2-dibromoethane
    "ICCI": "DCM",           # 1,2-diiodoethane
    "ICI": "DCM",            # diiodomethane
    "CCF": "FETH",           # fluoroethane
    "CC(F)F": "DFET",        # 1,1-difluoroethane
    "CCCF": "FETH",          # fluoropropane
    "FCCF": "DFET",          # 1,2-difluoroethane
    "CC(F)(F)F": "TFET",     # 1,1,1-trifluoroethane
    "CCC(F)(F)F": "TFET",    # 3,3,3-trifluoropropane
    "Clc1ccc(Cl)cc1": "DFB", # p-dichlorobenzene (approx with DFB)
    "Fc1ccccc1": "DFB",      # fluorobenzene
    # Sulfur
    "CS": "MESH",            # methanethiol
    "CSS": "DMDS",           # dimethyldisulfide approx
    "SS": "DMDS",            # H2S2 approx
    "CSSC": "DMDS",          # dimethyldisulfide
    "CCSSCC": "DEDS",        # diethyldisulfide
    "CSCSC": "EMS",          # methyldithiomethane
    "CCCSS": "ETSH",         # prop-1-ene-3-thiol approx
    "C1CCSSC1": "DEDS",      # 1,2-dithiane
    "S1CCSCC1": "EMS",       # 1,3-dithiane
    # Nitrogenous
    "CCN": "DMAM",           # ethylamine
    "CCCN": "DMAM",          # propylamine
    "CNCC": "DMAM",          # N-methylethylamine
    "C[NH2+]C": "DMAM",      # dimethylammonium
    "C[NH+](C)C": "TMAM",    # trimethylammonium
    "C[N+](C)(C)C": "NC4",   # tetramethylammonium
    "C[NH3+]": "MAMM",       # methylammonium
    "NC(=[NH2+])N": "GUAN",  # guanidinium
    "CNC(=[NH2+])N": "MGUA", # methylguanidinium
    # Aromatic nitrogen
    "c1ccncn1": "IMIA",      # pyrimidine approx
    "Cc1c[nH]c[nH+]1": "IMIM", # imidazolium
    "c1ccc2c(c1)[nH]cc2": "INDO", # indole
    # Phosphorus
    "OP(=O)(O)O": "MP_0",    # phosphoric acid
    "COP(=O)(O)O": "MP_1",   # methylphosphate
    # Ions
    "[Ca+2]": "TIP3",
    "[Li+]": "TIP3",
    "[F-]": "TIP3",
    "C#N": "DMAM",           # acetonitrile approx
}


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


def _fmt_comp(z_arr: np.ndarray) -> str:
    """Format atomic composition as human-readable formula, e.g. C6H6N2."""
    from ase.data import chemical_symbols as _sym
    counts = {}
    for z in z_arr:
        counts[int(z)] = counts.get(int(z), 0) + 1
    # Standard Hill order: C first, H second, then alphabetical
    order = sorted(counts.keys(), key=lambda z: (z != 6, z != 1, _sym[z]))
    return "".join(f"{_sym[z]}{counts[z] if counts[z] > 1 else ''}" for z in order)



def match_cgenff_template_fast(
    z_sub: np.ndarray,
    pos_sub: np.ndarray | None = None,
    target_charge: float = 0.0,
    canonical_smiles: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Match monomer against CGenFF template, using canonical SMILES first for isomers."""
    # Priority 1: SMILES lookup — unambiguous for constitutional isomers
    if canonical_smiles and canonical_smiles in DES_SMILES_TO_RESI:
        res_name = DES_SMILES_TO_RESI[canonical_smiles]
    else:
        counts = dict(zip(*np.unique(z_sub, return_counts=True)))
        comp_key = tuple(sorted(counts.items()))
        # Priority 2: explicit fast-path for common DES monomers
        if counts == {6: 1, 1: 2, 17: 2}:
            res_name = "DCM"
        elif counts == {8: 1, 6: 3, 1: 6}:
            res_name = "ACO"
        elif counts == {6: 6, 1: 6}:
            res_name = "BENZ"
        elif counts == {8: 1, 1: 2}:
            res_name = "TIP3"
        elif counts == {6: 1, 8: 1, 1: 4}:
            res_name = "MEOH"
        elif comp_key in _COMPOSITION_MAP:
            # Priority 3: composition index (may be ambiguous for isomers)
            res_name = _COMPOSITION_MAP[comp_key][0]
        else:
            formula = _fmt_comp(z_sub)
            raise KeyError(f"No CGenFF template for {formula}. Add to DES_SMILES_TO_RESI.")

    if res_name not in _CGENFF_RESIDUES:
        raise KeyError(f"RESI '{res_name}' not found in parsed RTF residues.")

    tmpl = _CGENFF_RESIDUES[res_name]
    type_indices = np.array(tmpl["type_indices"], dtype=np.int32)
    charges = np.array(tmpl["charges"], dtype=np.float64)

    if pos_sub is not None:
        permutation = _template_to_geometry_permutation(tmpl, z_sub, pos_sub)
        type_indices = type_indices[permutation]
        charges = charges[permutation]

    # Strict charge conservation
    n_atoms = len(charges)
    if n_atoms > 0:
        charge_diff = target_charge - np.sum(charges)
        if abs(charge_diff) > 1e-12:
            charges = charges + (charge_diff / n_atoms)

    return res_name, type_indices, charges


def _template_to_geometry_permutation(
    tmpl: dict, z_observed: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return template indices in observed atom order using graph isomorphism.

    Composition-only residue lookup does not imply atom-order equivalence.  This
    explicitly maps the RTF covalent graph onto the geometry graph, preventing
    e.g. TIP3 ``O,H,H`` parameters from being assigned to an ``H,H,O`` frame.
    """
    z_obs = np.asarray(z_observed, dtype=np.int32).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    z_tmpl = np.asarray(tmpl["z_elements"], dtype=np.int32)
    n = z_obs.size
    if z_tmpl.size != n or sorted(z_tmpl.tolist()) != sorted(z_obs.tolist()):
        raise ValueError(f"Template {tmpl['name']} composition does not match geometry")

    names = list(tmpl["atoms"])
    name_to_idx = {name: i for i, name in enumerate(names)}
    adj_t = np.zeros((n, n), dtype=bool)
    for a, b in tmpl.get("bonds", []):
        if a not in name_to_idx or b not in name_to_idx:
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        if z_tmpl[ia] == z_tmpl[ib] == 1:  # TIP3 SHAKE-only H-H constraint
            continue
        adj_t[ia, ib] = adj_t[ib, ia] = True

    dr = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(dr, axis=-1)
    cutoff = 1.3 * (covalent_radii[z_obs, None] + covalent_radii[z_obs][None, :])
    adj_o = (dist < cutoff) & (dist > 1.0e-8)

    def signature(z: np.ndarray, adj: np.ndarray, i: int) -> tuple:
        return int(z[i]), tuple(sorted(map(int, z[np.flatnonzero(adj[i])]))), int(adj[i].sum())

    candidates = []
    for oi in range(n):
        sig_o = signature(z_obs, adj_o, oi)
        matches = [ti for ti in range(n) if signature(z_tmpl, adj_t, ti) == sig_o]
        if not matches:
            raise ValueError(
                f"No topology match for atom {oi} (Z={z_obs[oi]}) in template {tmpl['name']}"
            )
        candidates.append(matches)

    observed_order = sorted(range(n), key=lambda oi: len(candidates[oi]))
    obs_to_tmpl = np.full(n, -1, dtype=np.int32)
    used: set[int] = set()

    def assign(depth: int) -> bool:
        if depth == n:
            return True
        oi = observed_order[depth]
        for ti in candidates[oi]:
            if ti in used:
                continue
            if any(
                bool(adj_o[oi, oj]) != bool(adj_t[ti, obs_to_tmpl[oj]])
                for oj in range(n)
                if obs_to_tmpl[oj] >= 0
            ):
                continue
            obs_to_tmpl[oi] = ti
            used.add(ti)
            if assign(depth + 1):
                return True
            used.remove(ti)
            obs_to_tmpl[oi] = -1
        return False

    if not assign(0):
        raise ValueError(f"Geometry is not graph-isomorphic to CGenFF template {tmpl['name']}")
    return obs_to_tmpl


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



def _worker_init():
    """Called once per worker process at pool startup to suppress duplicate warnings."""
    global _IS_WORKER
    _IS_WORKER = True


def process_single_frame(args_tuple):
    """Worker function for parallel frame processing across CPU cores."""
    z_struct, r_struct, f_struct, energy_i, q_i, s_i, d_i = args_tuple
    
    comps = find_covalent_components_fast(z_struct, r_struct)
    if len(comps) != 2:
        return ("SKIP", f"non-dimer: {len(comps)} components")

    comp_a, comp_b = comps[0], comps[1]

    try:
        # canonical_smiles=None: use composition lookup (SMILES lookup reserved for future
        # use if source cache provides smiles0/smiles1 keys directly)
        res_a, t_a, q_a = match_cgenff_template_fast(
            z_struct[comp_a], r_struct[comp_a],
            target_charge=0.0, canonical_smiles=None
        )
        res_b, t_b, q_b = match_cgenff_template_fast(
            z_struct[comp_b], r_struct[comp_b],
            target_charge=0.0, canonical_smiles=None
        )

        # Validate: no DEFAULT sentinel type indices (zero LJ params) should appear
        default_idx = _NB_MAP["DEFAULT"]
        if np.any(t_a == default_idx) or np.any(t_b == default_idx):
            counts_a = dict(zip(*np.unique(z_struct[comp_a], return_counts=True)))
            counts_b = dict(zip(*np.unique(z_struct[comp_b], return_counts=True)))
            return ("SKIP", f"DEFAULT sentinel type in {res_a}:{counts_a} or {res_b}:{counts_b}")

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
    except KeyError as e:
        formula_a = _fmt_comp(z_struct[comp_a])
        formula_b = _fmt_comp(z_struct[comp_b])
        return ("SKIP", f"Unmapped: {e} (A={formula_a}, B={formula_b})")
    except Exception as e:
        return ("SKIP", f"Unexpected error: {type(e).__name__}: {e}")


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
    dropped_total = 0
    dropped_reasons: dict[str, int] = {}

    with mp.Pool(processes=workers, initializer=_worker_init) as pool:
        for res in pool.imap(process_single_frame, frame_generator(), chunksize=5000):
            if isinstance(res, tuple) and len(res) == 2 and res[0] == "SKIP":
                dropped_total += 1
                reason = res[1][:120]  # truncate long reasons
                dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
                continue
            if res is None:
                dropped_total += 1
                dropped_reasons["None result"] = dropped_reasons.get("None result", 0) + 1
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

    if dropped_total > 0:
        # Group by category for clarity
        non_dimer = sum(c for r, c in dropped_reasons.items() if r.startswith("non-dimer"))
        unmapped  = sum(c for r, c in dropped_reasons.items() if r.startswith("Unmapped"))
        sentinel  = sum(c for r, c in dropped_reasons.items() if "DEFAULT sentinel" in r)
        other     = dropped_total - non_dimer - unmapped - sentinel
        print(f"\n[WARNING] Dropped {dropped_total:,} frames ({100*dropped_total/(dropped_total+dimers_processed):.1f}%):")
        print(f"   {non_dimer:>10,}  non-dimer structures (monomers / clusters / multi-component) — expected")
        print(f"   {unmapped:>10,}  unmapped CGenFF templates (exotic molecules / fragments)")
        if sentinel:
            print(f"   {sentinel:>10,}  sentinel zero-LJ atoms (check atom types)")
        if other:
            print(f"   {other:>10,}  other errors")
        print(f"\n   Top unmapped compositions:")
        for reason, count in sorted(
            ((r, c) for r, c in dropped_reasons.items() if r.startswith("Unmapped")),
            key=lambda x: -x[1]
        )[:15]:
            print(f"   {count:>8,} : {reason}")

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
