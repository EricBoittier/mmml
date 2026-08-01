"""CGenFF atom-type / charge assignment for dimer ML/MM training datasets.

This is the reusable core shared by:

* ``scripts/prepare_ml_mm_dataset.py``   -- Orbax bulk caches (ragged frames)
* ``mmml prepare-mm-dataset``            -- NPZ training splits (dense frames)

The assignment pipeline, per frame:

1. Parse the CGenFF ``NONBONDED`` section (``par_all36_cgenff.prm``) into a
   ``type -> idx`` map plus master ``sigma``/``epsilon`` tables, and the ``RESI``
   blocks (``top_all36_cgenff.rtf``) into per-atom types, charges and the bond
   graph.  This is cached in a :class:`CgenffReference`.
2. Split the frame into covalent components (monomers) by a covalent-radius
   distance cutoff.
3. Match each monomer to a CGenFF ``RESI`` template by composition (with explicit
   fast-paths for the common DES monomers) or canonical SMILES.
4. Reorder the template onto the observed geometry via graph isomorphism so the
   per-atom parameters line up with the actual atom order.
5. Emit per-atom ``cgenff_type_idx`` (index into the master tables),
   ``cgenff_charge`` (rescaled to conserve each monomer's net charge) and
   ``mol_id``, plus an inter-monomer CGenFF MM (LJ + Coulomb) energy/force
   baseline.

``sigma`` follows the conventional ``4*eps [(sig/r)^12 - (sig/r)^6]`` LJ form
(Lorentz-Berthelot combination); CHARMM's ``Rmin/2`` is converted on parse via
``sigma = 2 * (Rmin/2) / 2**(1/6)``.  See :mod:`mmml.models.cgenff_mm` for the
padding convention consumed downstream: ``cgenff_type_idx < 0`` / ``mol_id < 0``
mark padding atoms.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ase.units
import numpy as np
from ase.data import atomic_numbers, covalent_radii

__all__ = [
    "K_COULOMB_KCAL_ANG",
    "KCAL_TO_EV",
    "DEF_PRM_PATH",
    "DEF_RTF_PATH",
    "CgenffReference",
    "FrameAssignment",
    "load_reference",
    "find_covalent_components",
    "match_cgenff_template",
    "assign_frame_cgenff",
    "format_composition",
]

K_COULOMB_KCAL_ANG = 332.06371  # e^2 / Angstrom -> kcal/mol
# kcal/mol -> eV (0.0433641...).  The reciprocal (23.06) is the eV -> kcal/mol
# factor; using it here would inflate E_cgenff_mm / F_cgenff_mm by ~531.8x.
KCAL_TO_EV = ase.units.kcal / ase.units.mol

_DATA_DIR = Path(__file__).resolve().parent / "charmm"
DEF_PRM_PATH = _DATA_DIR / "par_all36_cgenff.prm"
DEF_RTF_PATH = _DATA_DIR / "top_all36_cgenff.rtf"
# CHARMM stream files merged on top of CGenFF, for chemistry CGenFF has no
# template for. The merge is additive, so nothing CGenFF already typed changes.
# See docs/des-so3lr-dimers.md.
#   toppar_water_ions.str            -- monatomic ions (CLA/SOD/POT/LIT/CAL/MG/...)
#   toppar_dum_noble_gases.str       -- HE1/NE1 (and a DUM pseudo-atom, ignored)
#   toppar_noble_gases_literature.str -- AR1/KR1/XE1, **not a CHARMM file**
#
# CHARMM ships no Ar/Kr/Xe residue anywhere, so that last file carries standard
# literature 12-6 parameters instead. They were not fitted alongside CGenFF and
# their cross terms are unvalidated -- noble-gas results are provisional. The
# file's own header records the source values and the conversion. Pass
# extra_toppar=() for the bare CGenFF reference, or drop that one entry to keep
# the CHARMM-only set.
DEF_EXTRA_TOPPAR: tuple[Path, ...] = (
    _DATA_DIR / "toppar_water_ions.str",
    _DATA_DIR / "toppar_dum_noble_gases.str",
    _DATA_DIR / "toppar_noble_gases_literature.str",
)

# Warnings only print once from the main process, not from every spawn worker.
_IS_WORKER = mp.current_process().name != "MainProcess"


# ─── Force-field parsing ───────────────────────────────────────────────────────


def load_cgenff_nonbonded_table(
    prm_path: Path,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Parse the ``NONBONDED`` section returning ``(type_map, sigmas, epsilons)``.

    A ``DEFAULT`` sentinel type with zero sigma/epsilon is appended so unmapped
    atom types produce an obviously-wrong (zero) LJ term instead of a silent
    plausible one.
    """
    nb_map: dict[str, int] = {}
    sigmas: list[float] = []
    epsilons: list[float] = []
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
                            rmin_half = float(parts[3])  # Angstrom (Rmin/2)
                            sigma = rmin_half * 2.0 / (2.0 ** (1.0 / 6.0))
                            nb_map[atom_type] = len(nb_map)
                            sigmas.append(sigma)
                            epsilons.append(epsilon)
                    except (ValueError, IndexError):
                        # Non-atom line (CUTNB, header, etc.)
                        pass

    if not nb_map:
        raise RuntimeError(f"No NONBONDED entries found in {prm_path}")

    # Sentinel DEFAULT type for unmapped types -- zero values deliberately visible.
    nb_map["DEFAULT"] = len(nb_map)
    sigmas.append(0.0)
    epsilons.append(0.0)

    sig_arr = np.array(sigmas, dtype=np.float64)
    eps_arr = np.array(epsilons, dtype=np.float64)
    bad_types = [
        t
        for t, i in nb_map.items()
        if t != "DEFAULT" and (sig_arr[i] == 0.0 or eps_arr[i] == 0.0)
    ]
    if bad_types and not _IS_WORKER:
        print(
            f"[cgenff] {len(bad_types)} atom types parsed with zero sigma or epsilon: "
            f"{bad_types[:10]}"
        )
        print(
            "  (LPH lone pairs and DUM dummy sites have zero LJ by design -- expected)"
        )

    return nb_map, sig_arr, eps_arr


def parse_mass_element_map(rtf_path: Path) -> dict[str, int]:
    """Parse ``MASS`` records -> ``atom_type -> atomic_number``.

    ``MASS -1 HGA1 1.00800 H`` -- the 5th field is the element symbol (ground
    truth).  Falls back to nearest-mass inference for old RTFs without it.
    """
    _MASS_BY_APPROX = {
        1.008: 1, 4.003: 2, 6.941: 3, 9.012: 4, 10.811: 5, 12.011: 6,
        14.007: 7, 15.999: 8, 18.998: 9, 20.180: 10, 22.990: 11, 24.305: 12,
        26.982: 13, 28.086: 14, 30.974: 15, 32.065: 16, 35.450: 17, 39.948: 18,
        39.098: 19, 40.080: 20, 65.370: 30, 79.904: 35, 85.468: 37,
        112.411: 48, 126.904: 53, 132.905: 55, 137.327: 56,
    }
    type_to_z: dict[str, int] = {}
    from_element: set[str] = set()
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
            # Massless pseudo-atoms (DUM in toppar_dum_noble_gases.str, and
            # similar dummy sites elsewhere) are not elements. Without this they
            # fall through the nearest-mass fallback below and land on carbon,
            # which would register a bogus single-carbon composition.
            if mass < 0.5:
                continue
            explicit = len(parts) >= 5 and parts[4].isalpha() and len(parts[4]) <= 2
            if explicit:
                z = atomic_numbers.get(parts[4].capitalize(), 0)
            else:
                # Nearest-mass inference is a fallback, and a lossy one: K
                # (39.098) and Ca (40.08) both sit within 1 amu of argon. A
                # stream file repeats its MASS records without the element
                # column in the parameter section, so never let a guess
                # overwrite a value the topology section stated outright.
                if atom_type in from_element:
                    continue
                closest = min(_MASS_BY_APPROX.keys(), key=lambda m: abs(m - mass))
                z = _MASS_BY_APPROX[closest] if abs(closest - mass) < 1.0 else 6
            if z > 0:
                type_to_z[atom_type] = z
                if explicit:
                    from_element.add(atom_type)
    return type_to_z


def load_cgenff_rtf_residues(
    rtf_path: Path,
    nb_map: dict[str, int],
    type_to_z: dict[str, int],
) -> tuple[dict[str, dict], dict[tuple, list[str]], dict[tuple, list[str]]]:
    """Parse all ``RESI`` blocks -> ``(residues, composition_map, collision_map)``."""
    residues: dict[str, dict] = {}
    composition_map: dict[tuple, list[str]] = {}
    if not rtf_path.exists():
        raise FileNotFoundError(f"CGenFF RTF file not found: {rtf_path}")

    default_idx = nb_map["DEFAULT"]
    current_resi = None
    resi_data = None
    bad_type_count = 0

    def _finalize_resi():
        nonlocal bad_type_count
        if not (current_resi and resi_data and resi_data["atoms"]):
            return
        for tidx in resi_data["type_indices"]:
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
                resi_data["atoms"].append(atom_name)
                resi_data["type_indices"].append(nb_map.get(atom_type, default_idx))
                resi_data["charges"].append(charge)
                resi_data["z_elements"].append(type_to_z.get(atom_type, 6))
            elif parts[0] in {"BOND", "DOUBLE", "TRIPLE"} and current_resi:
                # Skip cross-residue references (+N/-C) and TIP3's H-H SHAKE
                # constraint: neither is part of the intra-monomer covalent graph.
                names = parts[1:]
                for a, b in zip(names[0::2], names[1::2]):
                    if a.startswith(("+", "-")) or b.startswith(("+", "-")):
                        continue
                    resi_data["bonds"].append((a, b))
        _finalize_resi()

    collision_map = {k: v for k, v in composition_map.items() if len(v) > 1}

    if bad_type_count > 0 and not _IS_WORKER:
        print(
            f"[cgenff] {bad_type_count} ATOM records mapped to DEFAULT sentinel type "
            f"(unmapped atom_type in PRM); those atoms have zero LJ parameters."
        )
    return residues, composition_map, collision_map


# ─── Canonical SMILES -> CGenFF RESI name lookup for DES-S66 molecules ─────────
DES_SMILES_TO_RESI: dict[str, str] = {
    "O": "TIP3", "[H][H]": "TIP3", "[Ne]": "TIP3", "[Ar]": "TIP3", "[Kr]": "TIP3",
    "[Xe]": "TIP3", "N": "AMM1",
    "CC": "ETHA", "CCC": "PRPA", "CCCC": "BUTA", "CC(C)C": "IBUT", "CCCCC": "PENT",
    "CCCCCC": "HEXA", "C1CCCC1": "CPEN", "C1CCCCC1": "CHX",
    "C=C": "ETHE", "CC=C": "PRPE", "CC=CC": "BTE2", "CCC=C": "BTE1", "C#C": "ETHE",
    "CC#CC": "DIPE", "CCC#C": "BTE1",
    "CO": "MEOH", "CCO": "ETOH", "CCCO": "PRO2", "CC(O)C": "PRO2", "OCCCCO": "ETOH",
    "OC1CCCC1": "PRO2", "OC1CCCCC1": "PRO2", "Oc1ccccc1": "PHEN",
    "COC": "THF", "CCOC": "THF", "CCCOC": "THF", "COCOC": "THF", "COCOCC": "THF",
    "C1CCCOC1": "THF", "C1CCOCC1": "THF", "O1CCOCC1": "THF", "O1COCOC1": "THF",
    "C1CCOCO1": "THF",
    "CC=O": "AALD", "CCC=O": "PALD", "C=O": "FORM", "CC(C)=O": "ACO",
    "OC=O": "ACEH", "CC(O)=O": "ACEH", "COC(=O)C": "MAS", "CCOC(=O)C": "ETAC",
    "CCOC=O": "MPRO", "CNC(=O)C": "NMA", "O=CN(C)C": "DMAM", "CC(=O)N(C)C": "NMA",
    "CC(=O)N": "ACEM", "CNC=O": "FORM", "CCC(=O)N": "PRAM",
    "CCl": "CALD", "ClCCl": "DCM", "ClC(Cl)Cl": "CHAL", "CC(Cl)(Cl)Cl": "CHAL",
    "CCCCl": "DCM", "ClCCCl": "DCM", "CBr": "CALD", "CC(Br)Br": "DCM", "ICCI": "DCM",
    "ICI": "DCM", "CCF": "FETH", "CC(F)F": "DFET", "CCCF": "FETH", "FCCF": "DFET",
    "CC(F)(F)F": "TFET", "CCC(F)(F)F": "TFET", "Clc1ccc(Cl)cc1": "DFB",
    "Fc1ccccc1": "DFB",
    "CS": "MESH", "CSS": "DMDS", "SS": "DMDS", "CSSC": "DMDS", "CCSSCC": "DEDS",
    "CSCSC": "EMS", "CCCSS": "ETSH", "C1CCSSC1": "DEDS", "S1CCSCC1": "EMS",
    "CCN": "DMAM", "CCCN": "DMAM", "CNCC": "DMAM", "C[NH2+]C": "DMAM",
    "C[NH+](C)C": "TMAM", "C[N+](C)(C)C": "NC4", "C[NH3+]": "MAMM",
    "NC(=[NH2+])N": "GUAN", "CNC(=[NH2+])N": "MGUA",
    "c1ccncn1": "IMIA", "Cc1c[nH]c[nH+]1": "IMIM", "c1ccc2c(c1)[nH]cc2": "INDO",
    "OP(=O)(O)O": "MP_0", "COP(=O)(O)O": "MP_1",
    "[Ca+2]": "TIP3", "[Li+]": "TIP3", "[F-]": "TIP3", "C#N": "DMAM",
}

# Common DES monomers matched directly by elemental composition (fast path).
_COMPOSITION_FAST_PATH: dict[tuple, str] = {
    ((1, 2), (6, 1), (17, 2)): "DCM",
    ((1, 6), (6, 3), (8, 1)): "ACO",
    ((1, 6), (6, 6)): "BENZ",
    ((1, 2), (8, 1)): "TIP3",
    ((1, 4), (6, 1), (8, 1)): "MEOH",
}


# ─── Cached reference ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CgenffReference:
    """Parsed CGenFF force field: nonbonded tables + RESI templates."""

    nb_map: dict[str, int]
    sigmas: np.ndarray  # (n_types,) conventional sigma (Angstrom)
    epsilons: np.ndarray  # (n_types,) |epsilon| (kcal/mol)
    type_to_z: dict[str, int]
    residues: dict[str, dict]
    composition_map: dict[tuple, list[str]]
    collision_map: dict[tuple, list[str]]

    @property
    def default_idx(self) -> int:
        return self.nb_map["DEFAULT"]


@lru_cache(maxsize=4)
def load_reference(
    prm_path: str | Path = DEF_PRM_PATH,
    rtf_path: str | Path = DEF_RTF_PATH,
    extra_toppar: tuple[str | Path, ...] = DEF_EXTRA_TOPPAR,
) -> CgenffReference:
    """Load and cache the CGenFF reference from the given PRM/RTF paths.

    ``extra_toppar`` are CHARMM stream files (``.str``) carrying both topology
    and parameter sections, merged on top of the CGenFF base. The default adds
    ``toppar_water_ions.str``, which is what supplies the monatomic ion
    residues (``CLA``, ``SOD``, ``POT``, ``LIT``, ``CAL``, ``MG``, …) that
    CGenFF alone has no template for.

    The merge is strictly **additive**: an atom type or ``RESI`` already
    defined by CGenFF is left alone, and new compositions are appended to the
    candidate list rather than inserted ahead of it. Assignments that already
    worked therefore cannot change.
    """
    prm = Path(prm_path)
    rtf = Path(rtf_path)
    nb_map, sigmas, epsilons = load_cgenff_nonbonded_table(prm)
    type_to_z = parse_mass_element_map(rtf)

    extra_paths = [Path(p) for p in extra_toppar]
    for extra in extra_paths:
        if not extra.exists():
            raise FileNotFoundError(f"extra toppar not found: {extra}")
        e_nb, e_sig, e_eps = load_cgenff_nonbonded_table(extra)
        for name, idx in e_nb.items():
            if name == "DEFAULT" or name in nb_map:
                continue
            nb_map[name] = len(sigmas)
            sigmas = np.append(sigmas, e_sig[idx])
            epsilons = np.append(epsilons, e_eps[idx])
        for t, z in parse_mass_element_map(extra).items():
            type_to_z.setdefault(t, z)

    residues, composition_map, collision_map = load_cgenff_rtf_residues(
        rtf, nb_map, type_to_z
    )
    for extra in extra_paths:
        e_res, e_comp, _ = load_cgenff_rtf_residues(extra, nb_map, type_to_z)
        idx_to_type = {v: k for k, v in nb_map.items()}
        installed = set()
        for name, tmpl in e_res.items():
            if name in residues:
                continue  # CGenFF wins; TIP3 is defined in both
            # Skip templates built on pseudo-atoms with no element (DUM in
            # toppar_dum_noble_gases.str). load_cgenff_rtf_residues falls back
            # to carbon for an unmapped atom type, which would register RESI DUM
            # as a lone carbon and let it match real single-carbon fragments.
            if any(idx_to_type.get(int(i)) not in type_to_z
                   for i in tmpl["type_indices"]):
                continue
            residues[name] = tmpl
            installed.add(name)
        # Only register compositions for templates we actually installed --
        # otherwise a name CGenFF already owns could be reachable under the
        # stream file's composition while resolving to CGenFF's geometry.
        for comp_key, names in e_comp.items():
            keep = [n for n in names if n in installed]
            if not keep:
                # Do not create the key at all. An empty candidate list makes
                # match_cgenff_template's composition_map[key][0] raise
                # IndexError, which assign_frame_cgenff does not catch -- it
                # only handles KeyError/ValueError -- so a skipped template
                # would crash the run instead of reporting "unmapped".
                continue
            existing = composition_map.setdefault(comp_key, [])
            for name in keep:
                if name not in existing:
                    existing.append(name)
    collision_map = {k: v for k, v in composition_map.items() if len(v) > 1}

    return CgenffReference(
        nb_map=nb_map,
        sigmas=sigmas,
        epsilons=epsilons,
        type_to_z=type_to_z,
        residues=residues,
        composition_map=composition_map,
        collision_map=collision_map,
    )


# ─── Geometry / topology ───────────────────────────────────────────────────────


def find_covalent_components(z: np.ndarray, pos: np.ndarray) -> list[list[int]]:
    """Partition atoms into connected covalent components (BFS over a bond graph)."""
    n = len(z)
    dr = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(dr, axis=-1)
    r_cov = covalent_radii[z]
    adj = dist < 1.3 * (r_cov[:, None] + r_cov[None, :])
    np.fill_diagonal(adj, False)

    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i in visited:
            continue
        comp: list[int] = []
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for nbr in np.flatnonzero(adj[curr]):
                if nbr not in visited:
                    visited.add(int(nbr))
                    queue.append(int(nbr))
        components.append(sorted(comp))
    return components


def format_composition(z_arr: np.ndarray) -> str:
    """Human-readable Hill-order formula, e.g. ``C6H6``."""
    from ase.data import chemical_symbols as _sym

    counts: dict[int, int] = {}
    for z in z_arr:
        counts[int(z)] = counts.get(int(z), 0) + 1
    order = sorted(counts.keys(), key=lambda z: (z != 6, z != 1, _sym[z]))
    return "".join(f"{_sym[z]}{counts[z] if counts[z] > 1 else ''}" for z in order)


def match_cgenff_template(
    ref: CgenffReference,
    z_sub: np.ndarray,
    pos_sub: np.ndarray | None = None,
    target_charge: float | None = None,
    canonical_smiles: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Match one monomer against a CGenFF ``RESI`` template.

    Returns ``(resi_name, type_indices, charges)`` in the observed atom order.
    Priority: canonical SMILES -> composition fast-path -> composition index.

    ``target_charge=None`` (the default) conserves the **template's own** net
    charge. For the neutral CGenFF residues this is 0 and behaves exactly as
    the old hard-coded default did; for the ion residues from
    ``toppar_water_ions.str`` it is what stops a chloride being rescaled to
    neutral. Pass a float to force a specific monomer charge.
    """
    if canonical_smiles and canonical_smiles in DES_SMILES_TO_RESI:
        res_name = DES_SMILES_TO_RESI[canonical_smiles]
    else:
        counts = dict(zip(*np.unique(z_sub, return_counts=True)))
        comp_key = tuple(sorted(counts.items()))
        if comp_key in _COMPOSITION_FAST_PATH:
            res_name = _COMPOSITION_FAST_PATH[comp_key]
        elif comp_key in ref.composition_map:
            res_name = ref.composition_map[comp_key][0]
        else:
            raise KeyError(
                f"No CGenFF template for {format_composition(z_sub)}. "
                "Add it to DES_SMILES_TO_RESI or the composition fast-path."
            )

    if res_name not in ref.residues:
        raise KeyError(f"RESI '{res_name}' not found in parsed RTF residues.")

    tmpl = ref.residues[res_name]
    type_indices = np.array(tmpl["type_indices"], dtype=np.int32)
    charges = np.array(tmpl["charges"], dtype=np.float64)

    if pos_sub is not None:
        permutation = _template_to_geometry_permutation(tmpl, z_sub, pos_sub)
        type_indices = type_indices[permutation]
        charges = charges[permutation]

    n_atoms = len(charges)
    if n_atoms > 0:
        if target_charge is None:
            # The RESI's declared net charge, recovered from its own atoms.
            target_charge = float(np.round(np.sum(charges)))
        charge_diff = target_charge - float(np.sum(charges))
        if abs(charge_diff) > 1e-12:
            charges = charges + (charge_diff / n_atoms)

    return res_name, type_indices, charges


def _template_to_geometry_permutation(
    tmpl: dict, z_observed: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return template indices in observed atom order via graph isomorphism.

    Composition-only lookup does not imply atom-order equivalence.  This maps the
    RTF covalent graph onto the geometry graph, so e.g. TIP3 ``O,H,H`` params are
    never assigned to an ``H,H,O`` frame.
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

    candidates: list[list[int]] = []
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


def compute_inter_monomer_mm(
    ref: CgenffReference,
    pos: np.ndarray,
    comp_a: list[int],
    t_a: np.ndarray,
    q_a: np.ndarray,
    comp_b: list[int],
    t_b: np.ndarray,
    q_b: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Inter-monomer CGenFF MM (Coulomb + LJ) energy and forces, in eV / eV·Å⁻¹."""
    pos_a = pos[comp_a]
    pos_b = pos[comp_b]
    forces = np.zeros_like(pos, dtype=np.float64)

    sig_a, eps_a = ref.sigmas[t_a], ref.epsilons[t_a]
    sig_b, eps_b = ref.sigmas[t_b], ref.epsilons[t_b]

    dr = pos_a[:, None, :] - pos_b[None, :, :]  # (N_a, N_b, 3)
    r = np.linalg.norm(dr, axis=-1)  # (N_a, N_b)

    q_ij = q_a[:, None] * q_b[None, :]
    sig_ij = 0.5 * (sig_a[:, None] + sig_b[None, :])
    eps_ij = np.sqrt(eps_a[:, None] * eps_b[None, :])

    # Coulomb
    r_coulomb = np.maximum(r, 1e-6)
    e_coulomb = np.sum(K_COULOMB_KCAL_ANG * q_ij / r_coulomb)
    f_c_mag = K_COULOMB_KCAL_ANG * q_ij / (r_coulomb**3)

    # LJ with soft-core distance clamp at r < 0.8 * sig_ij
    r_vdw = np.maximum(r, 0.8 * sig_ij)
    sr6 = (sig_ij / r_vdw) ** 6
    sr12 = sr6**2
    e_vdw = np.sum(4.0 * eps_ij * (sr12 - sr6))
    f_v_mag = (24.0 * eps_ij / (r_vdw**2)) * (2.0 * sr12 - sr6)

    f_vec = dr * (f_c_mag + f_v_mag)[:, :, None]
    forces[comp_a] += np.sum(f_vec, axis=1)
    forces[comp_b] -= np.sum(f_vec, axis=0)

    return float((e_coulomb + e_vdw) * KCAL_TO_EV), forces * KCAL_TO_EV


# ─── Single-frame assignment ───────────────────────────────────────────────────


@dataclass
class FrameAssignment:
    """CGenFF assignment for one (unpadded) dimer frame.

    All per-atom arrays are in the frame's original atom order.
    """

    mol_id: np.ndarray  # (n_atoms,) int32, monomer id 0/1
    cgenff_type_idx: np.ndarray  # (n_atoms,) int32, index into master tables
    cgenff_charge: np.ndarray  # (n_atoms,) float64, per-monomer conserved
    e_cgenff_mm: float  # inter-monomer MM energy (eV)
    f_cgenff_mm: np.ndarray  # (n_atoms, 3) inter-monomer MM force (eV/Å)
    res_names: tuple[str, str]  # matched RESI names (comp_a, comp_b)


def assign_frame_cgenff(
    z: np.ndarray,
    r: np.ndarray,
    ref: CgenffReference,
    *,
    compute_mm: bool = True,
    monomer_charges: tuple[float | None, float | None] = (None, None),
) -> tuple[FrameAssignment | None, str | None]:
    """Assign CGenFF types/charges to one unpadded dimer frame.

    Returns ``(assignment, None)`` on success or ``(None, reason)`` when the frame
    is skipped (not exactly two covalent components, unmapped template, or a
    DEFAULT-sentinel atom type slipping through).
    """
    z = np.asarray(z).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3)

    comps = find_covalent_components(z, r)
    if len(comps) != 2:
        return None, f"non-dimer: {len(comps)} components"
    comp_a, comp_b = comps[0], comps[1]

    try:
        res_a, t_a, q_a = match_cgenff_template(
            ref, z[comp_a], r[comp_a], target_charge=monomer_charges[0]
        )
        res_b, t_b, q_b = match_cgenff_template(
            ref, z[comp_b], r[comp_b], target_charge=monomer_charges[1]
        )
    except KeyError as exc:
        fa, fb = format_composition(z[comp_a]), format_composition(z[comp_b])
        return None, f"unmapped: {exc} (A={fa}, B={fb})"
    except ValueError as exc:
        return None, f"topology: {exc}"

    if np.any(t_a == ref.default_idx) or np.any(t_b == ref.default_idx):
        return None, f"DEFAULT sentinel type in {res_a}/{res_b}"

    n_atoms = len(z)
    mol_id = np.zeros(n_atoms, dtype=np.int32)
    mol_id[comp_b] = 1
    cgenff_type = np.zeros(n_atoms, dtype=np.int32)
    cgenff_charge = np.zeros(n_atoms, dtype=np.float64)
    cgenff_type[comp_a] = t_a
    cgenff_type[comp_b] = t_b
    cgenff_charge[comp_a] = q_a
    cgenff_charge[comp_b] = q_b

    if compute_mm:
        e_mm, f_mm = compute_inter_monomer_mm(
            ref, r, comp_a, t_a, q_a, comp_b, t_b, q_b
        )
    else:
        e_mm, f_mm = 0.0, np.zeros((n_atoms, 3), dtype=np.float64)

    return (
        FrameAssignment(
            mol_id=mol_id,
            cgenff_type_idx=cgenff_type,
            cgenff_charge=cgenff_charge,
            e_cgenff_mm=e_mm,
            f_cgenff_mm=f_mm,
            res_names=(res_a, res_b),
        ),
        None,
    )
