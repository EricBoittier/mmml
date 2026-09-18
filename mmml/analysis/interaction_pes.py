"""Rigid cluster interaction PES and molecular many-body leftovers.

Interaction energy (kcal/mol unless noted)::

    E_int(AB)  = E(AB) - E(A) - E(B)
    E_int(ABC) = E(ABC) - E(A) - E(B) - E(C)
    E3         = E_int(ABC) - sum_{I<J} E_int(IJ)

COM distances are mass-weighted. The approach axis is +Z, matching
``mmml.analysis.dimer_molecules``. Geometry builders reuse
``build_rigid_dimer`` / ``assign_mol_id``; evaluation is calculator-agnostic.

Plots live in ``mmml.analysis.interaction_pes_plot`` and consume this JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read as ase_read

from mmml.analysis.dimer_molecules import orient_molecule
from mmml.analysis.dimer_scans import (
    assign_mol_id,
    build_rigid_dimer,
    centered_atoms,
    fragment_index_arrays,
    intermolecular_min_distance,
)
from mmml.data.units import EV_TO_KCAL_MOL

SCHEMA_VERSION = "mmml.interaction_pes/v1"

# 1D COM slice: past PET-MAD xs receptive field (~2 layers × 4.5 Å).
DEFAULT_R_MIN_A = 2.5
DEFAULT_R_MAX_A = 12.0
DEFAULT_N_R_1D = 20

# 2D: COM distance × in-plane rotation of monomer B.
DEFAULT_R_2D_MIN_A = 3.0
DEFAULT_R_2D_MAX_A = 9.0
DEFAULT_N_R_2D = 12
DEFAULT_N_THETA_2D = 12
DEFAULT_THETA_MAX_DEG = 180.0

DEFAULT_N_R_TRIMER = 16
PET_MAD_XS_LAYER_CUTOFF_A = 4.5
PET_MAD_XS_N_GNN_LAYERS = 2
PET_MAD_XS_RECEPTIVE_FIELD_A = PET_MAD_XS_LAYER_CUTOFF_A * PET_MAD_XS_N_GNN_LAYERS

# Prior 3-body residual points (water ~2.85 Å, ethanol ~4.2 Å).
TRIMER_WATER_REF_A = 2.85
TRIMER_ETHANOL_REF_A = 4.2
FAR_FIELD_CONTROL_A = 12.0

OH_BOND_MAX_A = 1.25
COM_VECTOR_MIN_A = 1.0e-8
POSITION_CACHE_DECIMALS = 6

ORIENTATION_HBOND = "hbond"
ORIENTATION_STACKED = "stacked"
ORIENTATIONS = (ORIENTATION_HBOND, ORIENTATION_STACKED)

SYSTEM_WATER = "water"
SYSTEM_ETHANOL = "ethanol"
SYSTEM_ACETONE = "acetone"
DEFAULT_SLICE_SYSTEMS = (SYSTEM_WATER, SYSTEM_ETHANOL)
DEFAULT_SURFACE_SYSTEM = SYSTEM_ETHANOL
DEFAULT_TRIMER_SYSTEMS = (SYSTEM_WATER, SYSTEM_ETHANOL)

DEFAULT_WATER_XYZ = Path("examples") / "orca" / "water_opt" / "water.xyz"
DEFAULT_ETOH_XYZ = Path("examples") / "pet_mad_etoh_pbc" / "etoh.xyz"

CalculatorFactory = Callable[[], Calculator]


def linspace_angstrom(start: float, stop: float, n: int) -> np.ndarray:
    """Inclusive distance grid in Å."""
    if n < 2:
        raise ValueError("distance grid needs n >= 2")
    if stop < start:
        raise ValueError("distance grid requires stop >= start")
    return np.linspace(float(start), float(stop), int(n))


def merge_grid(base: np.ndarray, extra: Sequence[float]) -> np.ndarray:
    """Sorted unique union of a linspace and named control points (Å)."""
    values = np.concatenate([np.asarray(base, dtype=np.float64), np.asarray(extra, dtype=np.float64)])
    return np.unique(np.round(values, 12))


def rotation_matrix_about_axis(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    """Rodrigues rotation by ``angle_rad`` about ``axis``."""
    unit = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm < COM_VECTOR_MIN_A:
        raise ValueError("rotation axis must have non-zero length")
    unit = unit / norm
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    k = np.array(
        [
            [0.0, -unit[2], unit[1]],
            [unit[2], 0.0, -unit[0]],
            [-unit[1], unit[0], 0.0],
        ]
    )
    return c * np.eye(3) + s * k + (1.0 - c) * np.outer(unit, unit)


def _unit_from_com(atoms: Atoms, index: int) -> np.ndarray:
    centered = centered_atoms(atoms, center="com")
    vec = np.asarray(centered.get_positions()[index], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < COM_VECTOR_MIN_A:
        raise ValueError(f"atom {index} is at the COM; cannot form an orientation axis")
    return vec / norm


def hydroxyl_pair(atoms: Atoms) -> tuple[int, int] | None:
    """``(O_index, H_index)`` when a hydroxyl bond shorter than ``OH_BOND_MAX_A`` exists."""
    symbols = atoms.get_chemical_symbols()
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    o_idxs = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    h_idxs = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    if not o_idxs or not h_idxs:
        return None
    best: tuple[float, int, int] | None = None
    for o_idx in o_idxs:
        for h_idx in h_idxs:
            dist = float(np.linalg.norm(pos[h_idx] - pos[o_idx]))
            if best is None or dist < best[0]:
                best = (dist, o_idx, h_idx)
    if best is None or best[0] > OH_BOND_MAX_A:
        return None
    return best[1], best[2]


def orientation_vectors(atoms: Atoms) -> tuple[np.ndarray, np.ndarray, str]:
    """Donor and acceptor unit vectors in the COM frame.

    Hydroxyl molecules use O–H···O. Carbonyls (acetone) use antiparallel C=O.
    """
    oh = hydroxyl_pair(atoms)
    if oh is not None:
        o_idx, h_idx = oh
        return _unit_from_com(atoms, h_idx), _unit_from_com(atoms, o_idx), "hydroxyl"
    symbols = atoms.get_chemical_symbols()
    o_idxs = [i for i, symbol in enumerate(symbols) if symbol == "O"]
    if not o_idxs:
        raise ValueError("need an oxygen to define a dimer orientation axis")
    accept = _unit_from_com(atoms, o_idxs[0])
    return -accept, accept, "carbonyl"


def orient_monomer_pair(monomer: Atoms, orientation: str) -> tuple[Atoms, Atoms]:
    """Return COM-centered A/B copies for a named rigid orientation."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    donor, acceptor, _kind = orientation_vectors(monomer)
    monomer_a = orient_molecule(monomer, acceptor, point_toward_plus_z=True)
    monomer_a = centered_atoms(monomer_a, center="com")
    monomer_b = orient_molecule(monomer, donor, point_toward_plus_z=False)
    monomer_b = centered_atoms(monomer_b, center="com")
    if orientation == ORIENTATION_STACKED:
        # 180° about X: donor that faced A (−Z) now points away (+Z).
        monomer_b = rotate_about_com(monomer_b, np.diag([1.0, -1.0, -1.0]))
    return monomer_a, monomer_b


def rotate_about_com(atoms: Atoms, rotation: np.ndarray) -> Atoms:
    """Rotate ``atoms`` about its COM by a 3×3 matrix."""
    out = atoms.copy()
    com = np.asarray(out.get_center_of_mass(), dtype=np.float64)
    pos = np.asarray(out.get_positions(), dtype=np.float64) - com
    out.set_positions(pos @ np.asarray(rotation, dtype=np.float64).T + com)
    return out


def dimer_at_distance(
    monomer_a: Atoms,
    monomer_b: Atoms,
    distance_angstrom: float,
    *,
    theta_deg: float = 0.0,
) -> Atoms:
    """Place A/B at COM separation ``distance_angstrom`` along +Z; rotate B about Z."""
    rotated_b = rotate_about_com(
        monomer_b,
        rotation_matrix_about_axis((0.0, 0.0, 1.0), np.deg2rad(theta_deg)),
    )
    combined, _fragments = build_rigid_dimer(
        monomer_a,
        rotated_b,
        distance_angstrom=float(distance_angstrom),
        axis=(0.0, 0.0, 1.0),
        center="com",
    )
    return combined


def equilateral_trimer(monomer: Atoms, side_angstrom: float) -> Atoms:
    """Three rigid COM copies on an equilateral triangle of side ``side_angstrom``."""
    template = centered_atoms(monomer, center="com")
    n = len(template)
    copies = [template.copy() for _ in range(3)]
    copies[1].translate(np.array([side_angstrom, 0.0, 0.0]))
    copies[2].translate(np.array([0.5 * side_angstrom, 0.5 * np.sqrt(3.0) * side_angstrom, 0.0]))
    combined = copies[0] + copies[1] + copies[2]
    combined = assign_mol_id(combined, [n, n, n])
    combined.translate(-np.asarray(combined.get_center_of_mass(), dtype=np.float64))
    return combined


def fragment_pair(atoms: Atoms, id_i: int, id_j: int) -> Atoms:
    """Return the two-molecule subsystem with ``mol_id`` in ``{id_i, id_j}``."""
    mol_id = np.asarray(atoms.arrays["mol_id"], dtype=np.int64)
    mask = (mol_id == id_i) | (mol_id == id_j)
    subset = atoms[mask].copy()
    sizes = [int(np.sum(mol_id == id_i)), int(np.sum(mol_id == id_j))]
    return assign_mol_id(subset, sizes)


def atoms_cache_key(atoms: Atoms) -> str:
    """Stable SHA-256 of atomic numbers and rounded positions."""
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    pos = np.round(np.asarray(atoms.get_positions(), dtype=np.float64), POSITION_CACHE_DECIMALS)
    payload = numbers.tobytes() + pos.tobytes()
    return hashlib.sha256(payload).hexdigest()


def evaluate_energy_ev(
    atoms: Atoms,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
) -> float:
    """ASE potential energy in eV, cached by geometry fingerprint."""
    key = atoms_cache_key(atoms)
    if key in cache:
        return float(cache[key])
    frame = atoms.copy()
    frame.calc = calculator_factory()
    energy = float(frame.get_potential_energy())
    if not np.isfinite(energy):
        raise ValueError(f"non-finite energy for {len(frame)}-atom geometry")
    cache[key] = energy
    return energy


def interaction_energy_ev(
    cluster: Atoms,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
) -> tuple[float, float, tuple[float, ...]]:
    """Return ``(E_int, E_cluster, monomer_energies)`` in eV using ``mol_id`` fragments."""
    mol_id = np.asarray(cluster.arrays["mol_id"], dtype=np.int64)
    ids = np.unique(mol_id)
    e_cluster = evaluate_energy_ev(cluster, calculator_factory, cache)
    monomer_energies: list[float] = []
    for mol in ids:
        fragment = cluster[mol_id == mol]
        monomer_energies.append(evaluate_energy_ev(fragment, calculator_factory, cache))
    e_int = e_cluster - float(np.sum(monomer_energies))
    return e_int, e_cluster, tuple(monomer_energies)


def trimer_mbe_ev(
    trimer: Atoms,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
) -> dict[str, float]:
    """Molecular 3-body leftover for a 3-fragment cluster (eV)."""
    mol_id = np.asarray(trimer.arrays["mol_id"], dtype=np.int64)
    ids = tuple(int(v) for v in np.unique(mol_id))
    if len(ids) != 3:
        raise ValueError(f"trimer MBE expects 3 mol_id values, got {ids}")
    e_int, e_abc, monomers = interaction_energy_ev(trimer, calculator_factory, cache)
    pair_ints: list[float] = []
    for i, j in ((ids[0], ids[1]), (ids[0], ids[2]), (ids[1], ids[2])):
        pair = fragment_pair(trimer, i, j)
        e_pair_int, _e_pair, _ = interaction_energy_ev(pair, calculator_factory, cache)
        pair_ints.append(e_pair_int)
    e_pair_sum = float(np.sum(pair_ints))
    e3 = e_int - e_pair_sum
    return {
        "e_abc_ev": e_abc,
        "e_a_ev": monomers[0],
        "e_b_ev": monomers[1],
        "e_c_ev": monomers[2],
        "e_int_ev": e_int,
        "e_pair_sum_ev": e_pair_sum,
        "e3_ev": e3,
        "pair_int_ab_ev": pair_ints[0],
        "pair_int_ac_ev": pair_ints[1],
        "pair_int_bc_ev": pair_ints[2],
    }


def load_monomer_xyz(path: Path | str) -> Atoms:
    """Read an XYZ monomer and COM-center it."""
    atoms = ase_read(str(path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    return centered_atoms(atoms, center="com")


def default_system_monomers(*, include_acetone: bool = True) -> dict[str, Atoms]:
    """Repo water / ethanol XYZ plus the acetone distill monomer when requested."""
    repo = Path(__file__).resolve().parents[1]
    systems = {
        SYSTEM_WATER: load_monomer_xyz(repo / DEFAULT_WATER_XYZ),
        SYSTEM_ETHANOL: load_monomer_xyz(repo / DEFAULT_ETOH_XYZ),
    }
    if include_acetone:
        from mmml.distill.acetone_pool import load_acetone_monomer

        systems[SYSTEM_ACETONE] = centered_atoms(load_acetone_monomer(), center="com")
    return systems


def _ev_to_kcal(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value, dtype=np.float64) * EV_TO_KCAL_MOL


def scan_dimer_slice(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    *,
    orientation: str,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
) -> dict[str, Any]:
    """1D ``E_int(r)`` along COM distance for one rigid orientation."""
    monomer_a, monomer_b = orient_monomer_pair(monomer, orientation)
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    e_int = np.empty(r_vals.shape, dtype=np.float64)
    e_ab = np.empty(r_vals.shape, dtype=np.float64)
    min_contact = np.empty(r_vals.shape, dtype=np.float64)
    e_a = e_b = None
    for i, distance in enumerate(r_vals):
        dimer = dimer_at_distance(monomer_a, monomer_b, float(distance))
        e_int_i, e_ab_i, monomers = interaction_energy_ev(dimer, calculator_factory, cache)
        e_int[i] = e_int_i
        e_ab[i] = e_ab_i
        e_a, e_b = monomers
        idx_a, idx_b = fragment_index_arrays([len(monomer_a), len(monomer_b)])
        min_contact[i] = intermolecular_min_distance(
            dimer.get_positions()[idx_a], dimer.get_positions()[idx_b]
        )
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    well_index = int(np.nanargmin(e_int_kcal))
    far_mask = np.isclose(r_vals, FAR_FIELD_CONTROL_A, atol=1.0e-6)
    far_kcal = float(e_int_kcal[far_mask][0]) if np.any(far_mask) else float(e_int_kcal[-1])
    return {
        "system": system,
        "orientation": orientation,
        "r_angstrom": r_vals.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "e_ab_ev": e_ab.tolist(),
        "e_a_ev": float(e_a if e_a is not None else np.nan),
        "e_b_ev": float(e_b if e_b is not None else np.nan),
        "min_contact_angstrom": min_contact.tolist(),
        "well_r_angstrom": float(r_vals[well_index]),
        "well_kcal_mol": float(e_int_kcal[well_index]),
        "far_field_kcal_mol": far_kcal,
    }


def scan_dimer_surface(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    theta_deg: Sequence[float],
    *,
    orientation: str,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
) -> dict[str, Any]:
    """2D ``E_int(r, theta)``; rows are theta, columns are r."""
    monomer_a, monomer_b = orient_monomer_pair(monomer, orientation)
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    th_vals = np.asarray(theta_deg, dtype=np.float64)
    e_int = np.empty((th_vals.size, r_vals.size), dtype=np.float64)
    min_contact = np.empty_like(e_int)
    idx_a, idx_b = fragment_index_arrays([len(monomer_a), len(monomer_b)])
    for i, theta in enumerate(th_vals):
        for j, distance in enumerate(r_vals):
            dimer = dimer_at_distance(monomer_a, monomer_b, float(distance), theta_deg=float(theta))
            e_int[i, j], _e_ab, _ = interaction_energy_ev(dimer, calculator_factory, cache)
            min_contact[i, j] = intermolecular_min_distance(
                dimer.get_positions()[idx_a], dimer.get_positions()[idx_b]
            )
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    return {
        "system": system,
        "orientation": orientation,
        "r_angstrom": r_vals.tolist(),
        "theta_deg": th_vals.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "min_contact_angstrom": min_contact.tolist(),
        "well_kcal_mol": float(np.nanmin(e_int_kcal)),
    }


def scan_trimer_slice(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    *,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
) -> dict[str, Any]:
    """Equilateral-triangle trimer scan vs side length."""
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    e_int = np.empty(r_vals.shape, dtype=np.float64)
    e_pair = np.empty(r_vals.shape, dtype=np.float64)
    e3 = np.empty(r_vals.shape, dtype=np.float64)
    e_abc = np.empty(r_vals.shape, dtype=np.float64)
    for i, side in enumerate(r_vals):
        trimer = equilateral_trimer(monomer, float(side))
        mbe = trimer_mbe_ev(trimer, calculator_factory, cache)
        e_int[i] = mbe["e_int_ev"]
        e_pair[i] = mbe["e_pair_sum_ev"]
        e3[i] = mbe["e3_ev"]
        e_abc[i] = mbe["e_abc_ev"]
    e3_kcal = np.asarray(_ev_to_kcal(e3), dtype=np.float64)
    peak = int(np.nanargmax(np.abs(e3_kcal)))
    return {
        "system": system,
        "r_angstrom": r_vals.tolist(),
        "e_int_kcal_mol": np.asarray(_ev_to_kcal(e_int), dtype=np.float64).tolist(),
        "e_pair_sum_kcal_mol": np.asarray(_ev_to_kcal(e_pair), dtype=np.float64).tolist(),
        "e3_kcal_mol": e3_kcal.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_pair_sum_ev": e_pair.tolist(),
        "e3_ev": e3.tolist(),
        "e_abc_ev": e_abc.tolist(),
        "e3_peak_r_angstrom": float(r_vals[peak]),
        "e3_peak_kcal_mol": float(e3_kcal[peak]),
        "e3_far_kcal_mol": float(e3_kcal[-1]),
    }


def summarize_campaign(document: Mapping[str, Any]) -> dict[str, Any]:
    """Pull well depths, far-field, and peak ``E3`` into a flat summary."""
    summary: dict[str, Any] = {}
    for slice_row in document.get("dimer_slices", []):
        key = f"{slice_row['system']}_{slice_row['orientation']}"
        summary[f"{key}_well_kcal_mol"] = slice_row["well_kcal_mol"]
        summary[f"{key}_well_r_angstrom"] = slice_row["well_r_angstrom"]
        summary[f"{key}_far_field_kcal_mol"] = slice_row["far_field_kcal_mol"]
    for tri in document.get("trimer_slices", []):
        key = tri["system"]
        summary[f"{key}_trimer_e3_peak_kcal_mol"] = tri["e3_peak_kcal_mol"]
        summary[f"{key}_trimer_e3_peak_r_angstrom"] = tri["e3_peak_r_angstrom"]
        summary[f"{key}_trimer_e3_far_kcal_mol"] = tri["e3_far_kcal_mol"]
    return summary


def run_interaction_pes_campaign(
    *,
    calculator_factory: CalculatorFactory,
    systems: Mapping[str, Atoms],
    r_1d: Sequence[float] | None = None,
    r_2d: Sequence[float] | None = None,
    theta_deg: Sequence[float] | None = None,
    r_trimer: Sequence[float] | None = None,
    slice_systems: Sequence[str] = DEFAULT_SLICE_SYSTEMS,
    surface_system: str = DEFAULT_SURFACE_SYSTEM,
    trimer_systems: Sequence[str] = DEFAULT_TRIMER_SYSTEMS,
    orientations: Sequence[str] = ORIENTATIONS,
    calculator_name: str = "metatomic",
    checkpoint: str | None = None,
    checkpoint_sha256: str | None = None,
    cache: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate dimer slices, one 2D surface, and trimer MBE slices."""
    energy_cache: dict[str, float] = {} if cache is None else cache
    r_1d_vals = np.asarray(
        r_1d if r_1d is not None else linspace_angstrom(DEFAULT_R_MIN_A, DEFAULT_R_MAX_A, DEFAULT_N_R_1D),
        dtype=np.float64,
    )
    r_2d_vals = np.asarray(
        r_2d
        if r_2d is not None
        else linspace_angstrom(DEFAULT_R_2D_MIN_A, DEFAULT_R_2D_MAX_A, DEFAULT_N_R_2D),
        dtype=np.float64,
    )
    theta_vals = np.asarray(
        theta_deg
        if theta_deg is not None
        else linspace_angstrom(0.0, DEFAULT_THETA_MAX_DEG, DEFAULT_N_THETA_2D),
        dtype=np.float64,
    )
    r_tri_vals = np.asarray(
        r_trimer
        if r_trimer is not None
        else merge_grid(
            linspace_angstrom(DEFAULT_R_MIN_A, DEFAULT_R_MAX_A, DEFAULT_N_R_TRIMER),
            (TRIMER_WATER_REF_A, TRIMER_ETHANOL_REF_A, FAR_FIELD_CONTROL_A),
        ),
        dtype=np.float64,
    )
    document: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "energy_definition": "interaction",
        "units": {
            "energy": "kcal/mol",
            "stored_energy": "eV",
            "distance": "angstrom",
            "angle": "degree",
        },
        "pet_receptive_field_angstrom": PET_MAD_XS_RECEPTIVE_FIELD_A,
        "calculator": calculator_name,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "systems": {name: {"n_atoms": int(len(atoms))} for name, atoms in systems.items()},
        "dimer_slices": [],
        "dimer_surfaces": [],
        "trimer_slices": [],
    }
    for name in slice_systems:
        monomer = systems[name]
        for orientation in orientations:
            document["dimer_slices"].append(
                scan_dimer_slice(
                    monomer,
                    r_1d_vals,
                    orientation=orientation,
                    calculator_factory=calculator_factory,
                    cache=energy_cache,
                    system=name,
                )
            )
    if surface_system:
        document["dimer_surfaces"].append(
            scan_dimer_surface(
                systems[surface_system],
                r_2d_vals,
                theta_vals,
                orientation=ORIENTATION_HBOND,
                calculator_factory=calculator_factory,
                cache=energy_cache,
                system=surface_system,
            )
        )
    for name in trimer_systems:
        document["trimer_slices"].append(
            scan_trimer_slice(
                systems[name],
                r_tri_vals,
                calculator_factory=calculator_factory,
                cache=energy_cache,
                system=name,
            )
        )
    document["summary"] = summarize_campaign(document)
    document["n_cached_energies"] = len(energy_cache)
    return document


def dump_interaction_pes_json(document: Mapping[str, Any], path: Path | str) -> Path:
    """Write the campaign document as UTF-8 JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def load_interaction_pes_json(path: Path | str) -> dict[str, Any]:
    """Load a campaign document and check the schema version."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("interaction PES JSON must be an object")
    schema = document.get("schema")
    if schema != SCHEMA_VERSION:
        raise ValueError(f"expected schema {SCHEMA_VERSION!r}, got {schema!r}")
    return document


def sha256_file(path: Path) -> str:
    """SHA-256 of a checkpoint or other binary input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
