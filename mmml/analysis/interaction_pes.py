"""Rigid cluster interaction PES and molecular many-body leftovers.

Interaction energy (kcal/mol unless noted)::

    E_int(AB)  = E(AB) - E(A) - E(B)
    E_int(ABC) = E(ABC) - E(A) - E(B) - E(C)
    E3         = E_int(ABC) - sum_{I<J} E_int(IJ)

Site–site distances (O–O for hydroxyl dimers) replace COM translations.
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

from mmml.analysis.dimer_scans import (
    assign_mol_id,
    centered_atoms,
    fragment_index_arrays,
    intermolecular_min_distance,
)
from mmml.analysis.interaction_pes_geom import (
    DEFAULT_ACCEPTOR_FLAP_DEG,
    MOTIF_CYCLIC,
    MOTIF_LABELS,
    MOTIF_LINEAR,
    ORIENTATION_ACCEPTOR_ACCEPTOR,
    ORIENTATION_CARBONYL,
    ORIENTATION_HBOND,
    ORIENTATION_LABELS,
    ORIENTATION_LINEAR_OH_O,
    ORIENTATION_METHYL,
    ORIENTATION_STACKED,
    PLANE_XZ,
    PLANE_YZ,
    cyclic_hbond_trimer,
    dimer_acceptor_acceptor,
    dimer_dha_deg,
    dimer_for_orientation,
    dimer_oh_o,
    dimer_site_indices,
    equilateral_trimer,
    hydroxyl_pair,
    linear_hbond_trimer,
    orientations_for_system,
    rotate_about_com,
    rotation_matrix_about_axis,
    scan_coordinate_for_orientation,
    site_site_distance,
    trimer_for_motif,
    trimer_oo_distances,
)
from mmml.data.units import EV_TO_KCAL_MOL

SCHEMA_VERSION = "mmml.interaction_pes/v2"

DEFAULT_R_MIN_A = 2.2
DEFAULT_R_MAX_A = 12.0
DEFAULT_N_R_1D = 0  # 0 → piecewise well/far grid
DEFAULT_R_2D_MIN_A = 2.6
DEFAULT_R_2D_MAX_A = 8.0
DEFAULT_N_R_2D = 0
DEFAULT_N_THETA_2D = 13
DEFAULT_THETA_MIN_DEG = 120.0
DEFAULT_THETA_MAX_DEG = 180.0
DEFAULT_N_R_TRIMER = 0
DEFAULT_N_THETA_1D = 13

PET_MAD_XS_LAYER_CUTOFF_A = 4.5
PET_MAD_XS_N_GNN_LAYERS = 2
PET_MAD_XS_RECEPTIVE_FIELD_A = PET_MAD_XS_LAYER_CUTOFF_A * PET_MAD_XS_N_GNN_LAYERS

TRIMER_WATER_REF_A = 2.85
TRIMER_ETHANOL_REF_A = 4.2
FAR_FIELD_CONTROL_A = 12.0
DEFAULT_MIN_CONTACT_PLOT_A = 2.0
TRIMER_E3_PEAK_MAX_ABS_EINT_KCAL = 40.0
WELL_R_WINDOW = (2.4, 6.5)
DEFAULT_RE_FALLBACK_A = {
    "water": 2.90,
    "ethanol": 2.90,
    "acetone": 3.50,
}

OH_BOND_MAX_A = 1.25
COM_VECTOR_MIN_A = 1.0e-8
POSITION_CACHE_DECIMALS = 6

ORIENTATIONS = (ORIENTATION_LINEAR_OH_O, ORIENTATION_ACCEPTOR_ACCEPTOR)

SYSTEM_WATER = "water"
SYSTEM_ETHANOL = "ethanol"
SYSTEM_ACETONE = "acetone"
DEFAULT_SLICE_SYSTEMS = (SYSTEM_WATER, SYSTEM_ETHANOL)
DEFAULT_SURFACE_SYSTEM = SYSTEM_ETHANOL
DEFAULT_ANGULAR_SYSTEMS = (SYSTEM_WATER, SYSTEM_ETHANOL)
DEFAULT_TRIMER_SYSTEMS = (SYSTEM_WATER, SYSTEM_ETHANOL)
DEFAULT_TRIMER_MOTIFS = {
    SYSTEM_WATER: (MOTIF_LINEAR, MOTIF_CYCLIC),
    SYSTEM_ETHANOL: (MOTIF_LINEAR, MOTIF_CYCLIC),
}

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


def _inclusive_step(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((float(stop) - float(start)) / float(step))) + 1
    return np.linspace(float(start), float(stop), n)


def default_r_1d_angstrom() -> np.ndarray:
    """Fine near the well, coarse past 6 Å, with a visible contact limb."""
    wall = _inclusive_step(2.2, 2.4, 0.2)
    well = _inclusive_step(2.6, 4.5, 0.15)
    mid = np.array([4.8, 5.2, 5.7, 6.2])
    far = np.array([7.0, 8.0, 9.0, 10.5, FAR_FIELD_CONTROL_A])
    return merge_grid(np.concatenate([wall, well, mid, far]), ())


def default_r_2d_angstrom() -> np.ndarray:
    well = _inclusive_step(2.6, 4.4, 0.2)
    return merge_grid(well, (5.0, 6.0, 8.0))


def default_dha_deg() -> np.ndarray:
    return linspace_angstrom(DEFAULT_THETA_MIN_DEG, DEFAULT_THETA_MAX_DEG, DEFAULT_N_THETA_2D)


def default_r_trimer_angstrom() -> np.ndarray:
    well = _inclusive_step(2.5, 4.5, 0.2)
    return merge_grid(
        well,
        (TRIMER_WATER_REF_A, TRIMER_ETHANOL_REF_A, 5.5, 6.5, 8.0, 9.0, FAR_FIELD_CONTROL_A),
    )


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


def _ev_to_kcal(value: float | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64) * EV_TO_KCAL_MOL


def _contact_and_sites(cluster: Atoms, orientation: str | None = None) -> tuple[float, float | None, float | None]:
    mol_id = np.asarray(cluster.arrays["mol_id"], dtype=np.int64)
    ids = tuple(int(v) for v in np.unique(mol_id))
    sizes = [int(np.sum(mol_id == mol)) for mol in ids]
    fragments = fragment_index_arrays(sizes)
    pos = np.asarray(cluster.get_positions(), dtype=np.float64)
    contact = intermolecular_min_distance(pos[fragments[0]], pos[fragments[1]])
    site_r = dha = None
    if orientation is not None and len(ids) == 2:
        i, j = dimer_site_indices(cluster, orientation)
        site_r = site_site_distance(cluster, i, j)
        if orientation in (ORIENTATION_LINEAR_OH_O, ORIENTATION_ACCEPTOR_ACCEPTOR):
            try:
                dha = dimer_dha_deg(cluster)
            except ValueError:
                dha = None
    return float(contact), (None if site_r is None else float(site_r)), dha


def scan_dimer_slice(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    *,
    orientation: str,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
) -> dict[str, Any]:
    """1D ``E_int(r)`` along the orientation's site–site distance."""
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    e_int = np.empty(r_vals.shape, dtype=np.float64)
    e_ab = np.empty(r_vals.shape, dtype=np.float64)
    min_contact = np.empty(r_vals.shape, dtype=np.float64)
    site_r = np.empty(r_vals.shape, dtype=np.float64)
    dha = np.full(r_vals.shape, np.nan, dtype=np.float64)
    e_a = e_b = None
    for i, distance in enumerate(r_vals):
        dimer = dimer_for_orientation(monomer, orientation, float(distance))
        e_int_i, e_ab_i, monomers = interaction_energy_ev(dimer, calculator_factory, cache)
        e_int[i] = e_int_i
        e_ab[i] = e_ab_i
        e_a, e_b = monomers
        contact, measured, angle = _contact_and_sites(dimer, orientation)
        min_contact[i] = contact
        site_r[i] = measured if measured is not None else float(distance)
        if angle is not None:
            dha[i] = angle
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    row: dict[str, Any] = {
        "system": system,
        "orientation": orientation,
        "orientation_label": ORIENTATION_LABELS.get(orientation, orientation),
        "scan_coordinate": scan_coordinate_for_orientation(orientation),
        "acceptor_flap_deg": float(DEFAULT_ACCEPTOR_FLAP_DEG)
        if orientation == ORIENTATION_LINEAR_OH_O
        else 0.0,
        "r_angstrom": r_vals.tolist(),
        "site_r_angstrom": site_r.tolist(),
        "dha_deg": dha.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "e_ab_ev": e_ab.tolist(),
        "e_a_ev": float(e_a if e_a is not None else np.nan),
        "e_b_ev": float(e_b if e_b is not None else np.nan),
        "min_contact_angstrom": min_contact.tolist(),
    }
    well_kcal, well_r, far_kcal = _well_from_slice(row)
    row["well_r_angstrom"] = well_r
    row["well_kcal_mol"] = well_kcal
    row["far_field_kcal_mol"] = far_kcal
    return row


def scan_dimer_angular(
    monomer: Atoms,
    theta_deg: Sequence[float],
    *,
    r_angstrom: float,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
    plane: str = PLANE_XZ,
) -> dict[str, Any]:
    """1D ``E_int(θ)`` at fixed O–O, θ = donor–H–acceptor angle."""
    th_vals = np.asarray(theta_deg, dtype=np.float64)
    e_int = np.empty(th_vals.shape, dtype=np.float64)
    dha = np.empty(th_vals.shape, dtype=np.float64)
    min_contact = np.empty(th_vals.shape, dtype=np.float64)
    for i, theta in enumerate(th_vals):
        dimer = dimer_oh_o(monomer, float(r_angstrom), dha_deg=float(theta), plane=plane)
        e_int[i], _e_ab, _ = interaction_energy_ev(dimer, calculator_factory, cache)
        contact, _site, angle = _contact_and_sites(dimer, ORIENTATION_LINEAR_OH_O)
        min_contact[i] = contact
        dha[i] = angle if angle is not None else float(theta)
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    well_i = int(np.nanargmin(e_int_kcal))
    return {
        "system": system,
        "orientation": ORIENTATION_LINEAR_OH_O,
        "angle_name": "donor_h_acceptor",
        "acceptor_flap_deg": float(DEFAULT_ACCEPTOR_FLAP_DEG),
        "plane": plane,
        "r_angstrom": float(r_angstrom),
        "theta_deg": th_vals.tolist(),
        "dha_deg": dha.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "min_contact_angstrom": min_contact.tolist(),
        "well_kcal_mol": float(e_int_kcal[well_i]),
        "well_theta_deg": float(th_vals[well_i]),
    }


def scan_dimer_surface(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    theta_deg: Sequence[float],
    *,
    orientation: str = ORIENTATION_LINEAR_OH_O,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
) -> dict[str, Any]:
    """2D ``E_int(r, θ)`` with θ = donor–H–acceptor angle; rows are θ, columns r."""
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    th_vals = np.asarray(theta_deg, dtype=np.float64)
    e_int = np.empty((th_vals.size, r_vals.size), dtype=np.float64)
    min_contact = np.empty_like(e_int)
    dha = np.empty_like(e_int)
    for i, theta in enumerate(th_vals):
        for j, distance in enumerate(r_vals):
            dimer = dimer_oh_o(monomer, float(distance), dha_deg=float(theta), plane=PLANE_XZ)
            e_int[i, j], _e_ab, _ = interaction_energy_ev(dimer, calculator_factory, cache)
            contact, _site, angle = _contact_and_sites(dimer, orientation)
            min_contact[i, j] = contact
            dha[i, j] = angle if angle is not None else float(theta)
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    well_flat = int(np.nanargmin(e_int_kcal))
    well_i, well_j = np.unravel_index(well_flat, e_int_kcal.shape)
    return {
        "system": system,
        "orientation": orientation,
        "orientation_label": ORIENTATION_LABELS.get(orientation, orientation),
        "scan_coordinate": scan_coordinate_for_orientation(orientation),
        "angle_name": "donor_h_acceptor",
        "acceptor_flap_deg": float(DEFAULT_ACCEPTOR_FLAP_DEG),
        "r_angstrom": r_vals.tolist(),
        "theta_deg": th_vals.tolist(),
        "dha_deg": dha.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "min_contact_angstrom": min_contact.tolist(),
        "well_kcal_mol": float(e_int_kcal[well_i, well_j]),
        "well_r_angstrom": float(r_vals[well_j]),
        "well_theta_deg": float(th_vals[well_i]),
    }


def scan_trimer_slice(
    monomer: Atoms,
    distances_angstrom: Sequence[float],
    *,
    calculator_factory: CalculatorFactory,
    cache: dict[str, float],
    system: str,
    motif: str = MOTIF_CYCLIC,
) -> dict[str, Any]:
    """H-bond trimer scan vs adjacent O–O."""
    r_vals = np.asarray(distances_angstrom, dtype=np.float64)
    e_int = np.empty(r_vals.shape, dtype=np.float64)
    e_pair = np.empty(r_vals.shape, dtype=np.float64)
    e3 = np.empty(r_vals.shape, dtype=np.float64)
    e_abc = np.empty(r_vals.shape, dtype=np.float64)
    min_oo = np.empty(r_vals.shape, dtype=np.float64)
    for i, side in enumerate(r_vals):
        trimer = trimer_for_motif(monomer, motif, float(side))
        mbe = trimer_mbe_ev(trimer, calculator_factory, cache)
        e_int[i] = mbe["e_int_ev"]
        e_pair[i] = mbe["e_pair_sum_ev"]
        e3[i] = mbe["e3_ev"]
        e_abc[i] = mbe["e_abc_ev"]
        min_oo[i] = min(trimer_oo_distances(trimer))
    e3_kcal = np.asarray(_ev_to_kcal(e3), dtype=np.float64)
    e_int_kcal = np.asarray(_ev_to_kcal(e_int), dtype=np.float64)
    pair_kcal = np.asarray(_ev_to_kcal(e_pair), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(e_int_kcal) > 1.0e-8, e3_kcal / e_int_kcal, np.nan)
    peak = int(np.nanargmax(np.abs(e3_kcal)))
    ref_r = TRIMER_WATER_REF_A if system == SYSTEM_WATER else TRIMER_ETHANOL_REF_A
    ref_e3 = _interp_named(r_vals, e3_kcal, ref_r)
    ref_eint = _interp_named(r_vals, e_int_kcal, ref_r)
    return {
        "system": system,
        "motif": motif,
        "motif_label": MOTIF_LABELS.get(motif, motif),
        "scan_coordinate": "O-O",
        "r_angstrom": r_vals.tolist(),
        "min_oo_angstrom": min_oo.tolist(),
        "e_int_kcal_mol": e_int_kcal.tolist(),
        "e_pair_sum_kcal_mol": pair_kcal.tolist(),
        "e3_kcal_mol": e3_kcal.tolist(),
        "e3_over_eint": frac.tolist(),
        "e_int_ev": e_int.tolist(),
        "e_pair_sum_ev": e_pair.tolist(),
        "e3_ev": e3.tolist(),
        "e_abc_ev": e_abc.tolist(),
        "e3_peak_r_angstrom": float(r_vals[peak]),
        "e3_peak_kcal_mol": float(e3_kcal[peak]),
        "e3_far_kcal_mol": float(e3_kcal[-1]),
        "ref_r_angstrom": float(ref_r),
        "ref_e3_kcal_mol": ref_e3,
        "ref_e_int_kcal_mol": ref_eint,
    }


def _interp_named(r_vals: np.ndarray, y: np.ndarray, r_star: float) -> float:
    if r_vals.size == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(r_vals - float(r_star))))
    return float(y[idx])


def mask_clash_energy(
    energy: np.ndarray,
    min_contact_angstrom: np.ndarray,
    *,
    min_contact_A: float = DEFAULT_MIN_CONTACT_PLOT_A,
) -> np.ndarray:
    """Copy of ``energy`` with clash samples set to NaN (optional summaries)."""
    masked = np.asarray(energy, dtype=np.float64).copy()
    contact = np.asarray(min_contact_angstrom, dtype=np.float64)
    masked[contact < float(min_contact_A)] = np.nan
    return masked


def _well_from_slice(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return ``(well_kcal, well_r, far_kcal)`` on the attractive-well window."""
    r_vals = np.asarray(row["r_angstrom"], dtype=np.float64)
    energy = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
    finite = np.isfinite(energy)
    window = finite & (r_vals >= WELL_R_WINDOW[0]) & (r_vals <= WELL_R_WINDOW[1])
    if not np.any(window):
        window = finite
    if not np.any(window):
        return float("nan"), float("nan"), float("nan")
    well_index_local = int(np.nanargmin(energy[window]))
    well_index = int(np.flatnonzero(window)[well_index_local])
    far_mask = np.isclose(r_vals, FAR_FIELD_CONTROL_A, atol=1.0e-6)
    far_vals = energy[far_mask]
    finite_energy = energy[finite]
    far_kcal = float(far_vals[0]) if far_vals.size and np.isfinite(far_vals[0]) else float(finite_energy[-1])
    return float(energy[well_index]), float(r_vals[well_index]), far_kcal


def summarize_campaign(document: Mapping[str, Any]) -> dict[str, Any]:
    """Pull well depths, far-field, and peak ``E3`` into a flat summary."""
    summary: dict[str, Any] = {}
    for slice_row in document.get("dimer_slices", []):
        key = f"{slice_row['system']}_{slice_row['orientation']}"
        well_kcal, well_r, far_kcal = _well_from_slice(slice_row)
        summary[f"{key}_well_kcal_mol"] = well_kcal
        summary[f"{key}_well_r_angstrom"] = well_r
        summary[f"{key}_far_field_kcal_mol"] = far_kcal
    for ang in document.get("dimer_angular", []):
        key = f"{ang['system']}_{ang.get('plane', 'xz')}"
        summary[f"{key}_angular_well_kcal_mol"] = ang["well_kcal_mol"]
        summary[f"{key}_angular_well_theta_deg"] = ang["well_theta_deg"]
        summary[f"{key}_angular_r_angstrom"] = ang["r_angstrom"]
    for surface in document.get("dimer_surfaces", []):
        key = surface["system"]
        summary[f"{key}_surface_well_kcal_mol"] = surface["well_kcal_mol"]
        summary[f"{key}_surface_well_r_angstrom"] = surface.get("well_r_angstrom")
        summary[f"{key}_surface_well_theta_deg"] = surface.get("well_theta_deg")
    for tri in document.get("trimer_slices", []):
        key = f"{tri['system']}_{tri.get('motif', MOTIF_CYCLIC)}"
        e3 = np.asarray(tri["e3_kcal_mol"], dtype=np.float64)
        e_int = np.asarray(tri["e_int_kcal_mol"], dtype=np.float64)
        r_vals = np.asarray(tri["r_angstrom"], dtype=np.float64)
        keep = np.abs(e_int) <= TRIMER_E3_PEAK_MAX_ABS_EINT_KCAL
        if np.any(keep):
            masked = np.where(keep, e3, np.nan)
            peak = int(np.nanargmax(np.abs(masked)))
            summary[f"{key}_trimer_e3_peak_kcal_mol"] = float(e3[peak])
            summary[f"{key}_trimer_e3_peak_r_angstrom"] = float(r_vals[peak])
        else:
            summary[f"{key}_trimer_e3_peak_kcal_mol"] = tri["e3_peak_kcal_mol"]
            summary[f"{key}_trimer_e3_peak_r_angstrom"] = tri["e3_peak_r_angstrom"]
        summary[f"{key}_trimer_e3_far_kcal_mol"] = tri["e3_far_kcal_mol"]
        summary[f"{key}_trimer_ref_r_angstrom"] = tri.get("ref_r_angstrom")
        summary[f"{key}_trimer_ref_e3_kcal_mol"] = tri.get("ref_e3_kcal_mol")
        summary[f"{key}_trimer_ref_e_int_kcal_mol"] = tri.get("ref_e_int_kcal_mol")
    return summary


def _r_e_from_slices(document: Mapping[str, Any], system: str) -> float:
    for row in document.get("dimer_slices", []):
        if row["system"] == system and row["orientation"] == ORIENTATION_LINEAR_OH_O:
            well_r = row.get("well_r_angstrom")
            if well_r is not None and np.isfinite(well_r):
                return float(well_r)
    return float(DEFAULT_RE_FALLBACK_A.get(system, 2.90))


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
    angular_systems: Sequence[str] = DEFAULT_ANGULAR_SYSTEMS,
    trimer_systems: Sequence[str] = DEFAULT_TRIMER_SYSTEMS,
    trimer_motifs: Mapping[str, Sequence[str]] | None = None,
    calculator_name: str = "metatomic",
    checkpoint: str | None = None,
    checkpoint_sha256: str | None = None,
    cache: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate dimer slices, angular cuts, one 2D surface, and trimer MBE."""
    raw_factory = calculator_factory
    holder: list = []

    def calculator_factory() -> Calculator:
        if not holder:
            holder.append(raw_factory())
        return holder[0]

    energy_cache: dict[str, float] = {} if cache is None else cache
    r_1d_vals = np.asarray(r_1d if r_1d is not None else default_r_1d_angstrom(), dtype=np.float64)
    r_2d_vals = np.asarray(r_2d if r_2d is not None else default_r_2d_angstrom(), dtype=np.float64)
    theta_vals = np.asarray(theta_deg if theta_deg is not None else default_dha_deg(), dtype=np.float64)
    r_tri_vals = np.asarray(
        r_trimer if r_trimer is not None else default_r_trimer_angstrom(),
        dtype=np.float64,
    )
    motifs = dict(DEFAULT_TRIMER_MOTIFS if trimer_motifs is None else trimer_motifs)
    document: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "energy_definition": "interaction",
        "energy_formula": "E_int = E(AB) - E(A) - E(B)",
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
        "dimer_angular": [],
        "dimer_surfaces": [],
        "trimer_slices": [],
    }
    for name in slice_systems:
        monomer = systems[name]
        for orientation in orientations_for_system(name):
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
    for name in angular_systems:
        if name not in systems:
            continue
        r_e = _r_e_from_slices(document, name)
        for plane in (PLANE_XZ, PLANE_YZ):
            document["dimer_angular"].append(
                scan_dimer_angular(
                    systems[name],
                    theta_vals,
                    r_angstrom=r_e,
                    calculator_factory=calculator_factory,
                    cache=energy_cache,
                    system=name,
                    plane=plane,
                )
            )
    if surface_system:
        document["dimer_surfaces"].append(
            scan_dimer_surface(
                systems[surface_system],
                r_2d_vals,
                theta_vals,
                orientation=ORIENTATION_LINEAR_OH_O,
                calculator_factory=calculator_factory,
                cache=energy_cache,
                system=surface_system,
            )
        )
    for name in trimer_systems:
        for motif in motifs.get(name, (MOTIF_CYCLIC,)):
            document["trimer_slices"].append(
                scan_trimer_slice(
                    systems[name],
                    r_tri_vals,
                    calculator_factory=calculator_factory,
                    cache=energy_cache,
                    system=name,
                    motif=motif,
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


def dump_interaction_pes_npz(document: Mapping[str, Any], path: Path | str) -> Path:
    """Write numeric arrays from the campaign document as a compressed NPZ."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "pet_receptive_field_angstrom": np.array(
            document.get("pet_receptive_field_angstrom", PET_MAD_XS_RECEPTIVE_FIELD_A)
        ),
    }
    for i, row in enumerate(document.get("dimer_slices", [])):
        prefix = f"slice_{i}_{row['system']}_{row['orientation']}"
        arrays[f"{prefix}_r"] = np.asarray(row["r_angstrom"], dtype=np.float64)
        arrays[f"{prefix}_e_int_kcal"] = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
        arrays[f"{prefix}_contact"] = np.asarray(row["min_contact_angstrom"], dtype=np.float64)
    for i, row in enumerate(document.get("dimer_angular", [])):
        prefix = f"angular_{i}_{row['system']}_{row.get('plane', 'xz')}"
        arrays[f"{prefix}_theta"] = np.asarray(row["theta_deg"], dtype=np.float64)
        arrays[f"{prefix}_e_int_kcal"] = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
    for i, row in enumerate(document.get("dimer_surfaces", [])):
        prefix = f"surface_{i}_{row['system']}"
        arrays[f"{prefix}_r"] = np.asarray(row["r_angstrom"], dtype=np.float64)
        arrays[f"{prefix}_theta"] = np.asarray(row["theta_deg"], dtype=np.float64)
        arrays[f"{prefix}_e_int_kcal"] = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
    for i, row in enumerate(document.get("trimer_slices", [])):
        prefix = f"trimer_{i}_{row['system']}_{row.get('motif', 'cyclic')}"
        arrays[f"{prefix}_r"] = np.asarray(row["r_angstrom"], dtype=np.float64)
        arrays[f"{prefix}_e_int_kcal"] = np.asarray(row["e_int_kcal_mol"], dtype=np.float64)
        arrays[f"{prefix}_e3_kcal"] = np.asarray(row["e3_kcal_mol"], dtype=np.float64)
        arrays[f"{prefix}_pair_kcal"] = np.asarray(row["e_pair_sum_kcal_mol"], dtype=np.float64)
    np.savez_compressed(out, **arrays)
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


# Re-exports used by tests and the CLI.
__all__ = [
    "DEFAULT_ETOH_XYZ",
    "DEFAULT_N_R_1D",
    "DEFAULT_N_R_2D",
    "DEFAULT_N_R_TRIMER",
    "DEFAULT_N_THETA_2D",
    "DEFAULT_R_2D_MAX_A",
    "DEFAULT_R_2D_MIN_A",
    "DEFAULT_R_MAX_A",
    "DEFAULT_R_MIN_A",
    "DEFAULT_SURFACE_SYSTEM",
    "DEFAULT_THETA_MAX_DEG",
    "DEFAULT_THETA_MIN_DEG",
    "DEFAULT_WATER_XYZ",
    "FAR_FIELD_CONTROL_A",
    "ORIENTATION_ACCEPTOR_ACCEPTOR",
    "ORIENTATION_CARBONYL",
    "ORIENTATION_HBOND",
    "ORIENTATION_LINEAR_OH_O",
    "ORIENTATION_METHYL",
    "ORIENTATION_STACKED",
    "PET_MAD_XS_RECEPTIVE_FIELD_A",
    "SCHEMA_VERSION",
    "SYSTEM_ACETONE",
    "SYSTEM_ETHANOL",
    "SYSTEM_WATER",
    "TRIMER_ETHANOL_REF_A",
    "TRIMER_WATER_REF_A",
    "cyclic_hbond_trimer",
    "default_dha_deg",
    "default_r_1d_angstrom",
    "default_r_2d_angstrom",
    "default_r_trimer_angstrom",
    "default_system_monomers",
    "dimer_acceptor_acceptor",
    "dimer_oh_o",
    "dump_interaction_pes_json",
    "dump_interaction_pes_npz",
    "equilateral_trimer",
    "evaluate_energy_ev",
    "hydroxyl_pair",
    "interaction_energy_ev",
    "linear_hbond_trimer",
    "linspace_angstrom",
    "load_interaction_pes_json",
    "load_monomer_xyz",
    "mask_clash_energy",
    "merge_grid",
    "rotate_about_com",
    "rotation_matrix_about_axis",
    "run_interaction_pes_campaign",
    "scan_dimer_angular",
    "scan_dimer_slice",
    "scan_dimer_surface",
    "scan_trimer_slice",
    "sha256_file",
    "summarize_campaign",
    "trimer_mbe_ev",
]
