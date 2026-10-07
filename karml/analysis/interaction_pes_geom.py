"""Chemically named rigid orientations for interaction PES scans.

Internal axes (O–H, HOH bisector, C=O), not COM-to-atom copies. Dimers are
placed by a site–site distance (O–O for hydroxyl; O···C_methyl / C···C for
acetone). The 2D / angular coordinate is the donor–H–acceptor angle, not a
free spin about the approach axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from mmml.analysis.dimer_molecules import rotation_matrix_align_to_z
from mmml.analysis.dimer_scans import assign_mol_id, centered_atoms

OH_BOND_MAX_A = 1.25
CO_BOND_MAX_A = 1.45
COM_VECTOR_MIN_A = 1.0e-8
PLANE_FLATTEN_MIN_A = 1.0e-4

ORIENTATION_LINEAR_OH_O = "linear_oh_o"
ORIENTATION_ACCEPTOR_ACCEPTOR = "acceptor_acceptor"
ORIENTATION_CARBONYL = "carbonyl"
ORIENTATION_METHYL = "methyl"
ORIENTATION_HBOND = ORIENTATION_LINEAR_OH_O
ORIENTATION_STACKED = ORIENTATION_ACCEPTOR_ACCEPTOR

HYDROXYL_ORIENTATIONS = (ORIENTATION_LINEAR_OH_O, ORIENTATION_ACCEPTOR_ACCEPTOR)
ACETONE_ORIENTATIONS = (ORIENTATION_CARBONYL, ORIENTATION_METHYL)

ORIENTATION_LABELS = {
    ORIENTATION_LINEAR_OH_O: r"linear OH$\cdots$O",
    ORIENTATION_ACCEPTOR_ACCEPTOR: "acceptor–acceptor",
    ORIENTATION_CARBONYL: r"C=O acceptor",
    ORIENTATION_METHYL: "methyl–methyl",
}

MOTIF_LINEAR = "linear"
MOTIF_CYCLIC = "cyclic"
TRIMER_MOTIFS = (MOTIF_LINEAR, MOTIF_CYCLIC)
MOTIF_LABELS = {
    MOTIF_LINEAR: "linear H-bond chain",
    MOTIF_CYCLIC: "cyclic H-bond",
}

SCAN_OO = "O-O"
SCAN_O_CMETHYL = "O-C_methyl"
SCAN_CMETHYL_CMETHYL = "C_methyl-C_methyl"

PLANE_XZ = "xz"
PLANE_YZ = "yz"
# Textbook Cs water-dimer acceptor flap (still linear OH···O).
DEFAULT_ACCEPTOR_FLAP_DEG = 57.0


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


def apply_rotation(atoms: Atoms, rotation: np.ndarray, *, origin: Sequence[float] | None = None) -> Atoms:
    """Rotate ``atoms`` by a 3×3 matrix about ``origin`` (default: current origin)."""
    out = atoms.copy()
    pivot = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    pos = np.asarray(out.get_positions(), dtype=np.float64) - pivot
    out.set_positions(pos @ np.asarray(rotation, dtype=np.float64).T + pivot)
    return out


def rotate_about_com(atoms: Atoms, rotation: np.ndarray) -> Atoms:
    """Rotate ``atoms`` about its COM by a 3×3 matrix."""
    com = np.asarray(atoms.get_center_of_mass(), dtype=np.float64)
    return apply_rotation(atoms, rotation, origin=com)


def translate_atom_to(atoms: Atoms, index: int, position: Sequence[float]) -> Atoms:
    """Translate so ``atoms[index]`` sits at ``position``."""
    out = atoms.copy()
    pos = np.asarray(out.get_positions(), dtype=np.float64)
    out.set_positions(pos + (np.asarray(position, dtype=np.float64) - pos[int(index)]))
    return out


def combine_fragments(*molecules: Atoms) -> Atoms:
    """Concatenate rigid copies and tag ``mol_id``."""
    if not molecules:
        raise ValueError("need at least one fragment")
    combined = molecules[0].copy()
    for mol in molecules[1:]:
        combined = combined + mol
    return assign_mol_id(combined, [len(m) for m in molecules])


def indices_of_symbol(atoms: Atoms, symbol: str) -> list[int]:
    return [i for i, name in enumerate(atoms.get_chemical_symbols()) if name == symbol]


def attached_hydrogens(atoms: Atoms, o_idx: int) -> list[int]:
    """Hydrogen indices with O–H shorter than ``OH_BOND_MAX_A``."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    o_pos = pos[int(o_idx)]
    return [
        h
        for h in indices_of_symbol(atoms, "H")
        if float(np.linalg.norm(pos[h] - o_pos)) <= OH_BOND_MAX_A
    ]


def hydroxyl_pair(atoms: Atoms) -> tuple[int, int]:
    """``(O_index, H_index)`` for the shortest O–H contact below ``OH_BOND_MAX_A``.

    Ties keep the lower hydrogen index so equal water O–H bonds are stable.
    """
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    o_idxs = indices_of_symbol(atoms, "O")
    h_idxs = indices_of_symbol(atoms, "H")
    if not o_idxs or not h_idxs:
        raise ValueError("hydroxyl orientation needs oxygen and hydrogen")
    best: tuple[float, int, int] | None = None
    for o_idx in o_idxs:
        for h_idx in h_idxs:
            dist = float(np.linalg.norm(pos[h_idx] - pos[o_idx]))
            if best is None:
                best = (dist, o_idx, h_idx)
                continue
            closer = dist + 1.0e-8 < best[0]
            tie = abs(dist - best[0]) <= 1.0e-8 and h_idx < best[2]
            if closer or tie:
                best = (dist, o_idx, h_idx)
    if best is None or best[0] > OH_BOND_MAX_A:
        raise ValueError("no O–H bond shorter than OH_BOND_MAX_A")
    return best[1], best[2]


def plus_z_hydrogen(atoms: Atoms, o_idx: int) -> int:
    """Attached H with the largest z (the donor H after ``orient_hydroxyl_donor``)."""
    attached = attached_hydrogens(atoms, o_idx)
    if not attached:
        raise ValueError("oxygen has no attached hydrogen")
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    return max(attached, key=lambda h: float(pos[h, 2]))



def flatten_index(atoms: Atoms, o_idx: int, h_idx: int) -> int:
    """A second atom used to pin the molecular plane (other H, else nearest heavy)."""
    symbols = atoms.get_chemical_symbols()
    candidates = [
        i
        for i, symbol in enumerate(symbols)
        if i not in (o_idx, h_idx) and symbol == "H"
    ]
    if not candidates:
        candidates = [i for i, symbol in enumerate(symbols) if i not in (o_idx, h_idx) and symbol != "H"]
    if not candidates:
        raise ValueError("need a third atom to define a molecular plane")
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    o_pos = pos[o_idx]
    return min(candidates, key=lambda i: float(np.linalg.norm(pos[i] - o_pos)))


def oxygen_bisector(atoms: Atoms, o_idx: int) -> np.ndarray:
    """Unit vector from O toward the mean of H atoms attached to it (else O→closest H)."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    attached = attached_hydrogens(atoms, o_idx)
    if not attached:
        attached = [hydroxyl_pair(atoms)[1]]
    vec = np.mean(pos[attached] - pos[int(o_idx)], axis=0)
    norm = float(np.linalg.norm(vec))
    if norm < COM_VECTOR_MIN_A:
        raise ValueError("degenerate oxygen bisector")
    return vec / norm


def carbonyl_pair(atoms: Atoms) -> tuple[int, int]:
    """``(C_index, O_index)`` for the shortest C=O contact."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    o_idxs = indices_of_symbol(atoms, "O")
    c_idxs = indices_of_symbol(atoms, "C")
    if not o_idxs or not c_idxs:
        raise ValueError("carbonyl orientation needs carbon and oxygen")
    best: tuple[float, int, int] | None = None
    for o_idx in o_idxs:
        for c_idx in c_idxs:
            dist = float(np.linalg.norm(pos[c_idx] - pos[o_idx]))
            if best is None or dist < best[0]:
                best = (dist, c_idx, o_idx)
    if best is None or best[0] > CO_BOND_MAX_A:
        raise ValueError("no C=O bond shorter than CO_BOND_MAX_A")
    return best[1], best[2]


def methyl_carbon_indices(atoms: Atoms) -> list[int]:
    """Carbons that are not the carbonyl carbon."""
    c_idxs = indices_of_symbol(atoms, "C")
    carb_c, _o = carbonyl_pair(atoms)
    methyl = [i for i in c_idxs if i != carb_c]
    if not methyl:
        raise ValueError("no methyl carbon distinct from the carbonyl carbon")
    return methyl


def closest_hydrogen(atoms: Atoms, index: int) -> int:
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    h_idxs = indices_of_symbol(atoms, "H")
    if not h_idxs:
        raise ValueError("no hydrogen atoms")
    return min(h_idxs, key=lambda h: float(np.linalg.norm(pos[h] - pos[index])))


def bond_angle_deg(atoms: Atoms, i: int, j: int, k: int) -> float:
    """Angle ``i–j–k`` in degrees (vertex at ``j``)."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < COM_VECTOR_MIN_A or n2 < COM_VECTOR_MIN_A:
        raise ValueError("degenerate bond angle")
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def site_site_distance(atoms: Atoms, i: int, j: int) -> float:
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    return float(np.linalg.norm(pos[j] - pos[i]))


def polar_angle_for_dha(r_oo: float, r_oh: float, dha_deg: float) -> float:
    """Polar angle γ of the acceptor O so O_d–H–O_a equals ``dha_deg``.

    Donor O at the origin, H at ``(0,0,r_oh)``, acceptor O at
    ``r_oo (sin γ, 0, cos γ)``. ``γ = 0`` is linear (DHA = 180°).
    """
    if r_oo <= r_oh + 0.05:
        raise ValueError(f"O–O ({r_oo:.3f} Å) must exceed O–H ({r_oh:.3f} Å)")
    alpha = float(np.deg2rad(dha_deg))
    if alpha <= 0.0 or alpha > np.pi + 1.0e-9:
        raise ValueError(f"DHA angle must be in (0, 180] deg, got {dha_deg}")
    if abs(alpha - np.pi) < 1.0e-8:
        return 0.0
    k = float(np.cos(alpha) ** 2)
    r = float(r_oo)
    d = float(r_oh)
    a = r * r
    b = 2.0 * r * d * (k - 1.0)
    c = d * d * (1.0 - k) - k * r * r
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        if disc >= -1.0e-10:
            disc = 0.0
        else:
            raise ValueError(f"no O–H–O geometry at r={r_oo:.3f} Å, DHA={dha_deg:.1f} deg")
    sqrt_disc = float(np.sqrt(disc))
    roots = [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]
    best: float | None = None
    best_err = np.inf
    for cosine in roots:
        if cosine < -1.0 - 1.0e-8 or cosine > 1.0 + 1.0e-8:
            continue
        cosine = float(np.clip(cosine, -1.0, 1.0))
        gamma = float(np.arccos(cosine))
        o_a = r * np.array([np.sin(gamma), 0.0, cosine])
        h = np.array([0.0, 0.0, d])
        v1 = -h
        v2 = o_a - h
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        recovered = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
        err = abs(recovered - float(dha_deg))
        if err < best_err:
            best_err = err
            best = gamma
    if best is None or best_err > 0.5:
        raise ValueError(f"failed to recover DHA={dha_deg:.1f} deg at r={r_oo:.3f} Å")
    return best


def _align_vector_to_plus_z(atoms: Atoms, vector: np.ndarray) -> Atoms:
    rotation = rotation_matrix_align_to_z(np.asarray(vector, dtype=np.float64))
    return apply_rotation(atoms, rotation)


def _flatten_atom_to_xz(atoms: Atoms, index: int) -> Atoms:
    """Rotate about Z so ``atoms[index]`` lies in the +X half of the XZ plane."""
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)[int(index)]
    xy = pos[:2]
    if float(np.linalg.norm(xy)) < PLANE_FLATTEN_MIN_A:
        return atoms.copy()
    angle = -float(np.arctan2(xy[1], xy[0]))
    return apply_rotation(atoms, rotation_matrix_about_axis((0.0, 0.0, 1.0), angle))


def orient_hydroxyl_donor(monomer: Atoms) -> Atoms:
    """O at the origin, O→H along +Z, molecular plane in XZ."""
    o_idx, h_idx = hydroxyl_pair(monomer)
    out = translate_atom_to(monomer, o_idx, (0.0, 0.0, 0.0))
    pos = np.asarray(out.get_positions(), dtype=np.float64)
    out = _align_vector_to_plus_z(out, pos[h_idx] - pos[o_idx])
    return _flatten_atom_to_xz(out, flatten_index(out, o_idx, h_idx))


def orient_hydroxyl_acceptor(monomer: Atoms) -> Atoms:
    """O at the origin, HOH bisector along +Z (hydrogens away, lone pairs −Z)."""
    o_idx, h_idx = hydroxyl_pair(monomer)
    out = translate_atom_to(monomer, o_idx, (0.0, 0.0, 0.0))
    out = _align_vector_to_plus_z(out, oxygen_bisector(out, o_idx))
    return _flatten_atom_to_xz(out, flatten_index(out, o_idx, h_idx))


def _plane_coords(gamma: float, r_oo: float, plane: str) -> np.ndarray:
    s, c = float(np.sin(gamma)), float(np.cos(gamma))
    if plane == PLANE_XZ:
        return r_oo * np.array([s, 0.0, c])
    if plane == PLANE_YZ:
        return r_oo * np.array([0.0, s, c])
    raise ValueError(f"plane must be {PLANE_XZ!r} or {PLANE_YZ!r}, got {plane!r}")


def _orient_acceptor_at(
    monomer: Atoms,
    o_pos: np.ndarray,
    target: np.ndarray,
    plane_normal: np.ndarray,
) -> Atoms:
    """Place acceptor O at ``o_pos`` with lone pairs facing ``target``."""
    o_idx, h_idx = hydroxyl_pair(monomer)
    out = translate_atom_to(monomer, o_idx, (0.0, 0.0, 0.0))
    away = o_pos - target
    if float(np.linalg.norm(away)) < COM_VECTOR_MIN_A:
        raise ValueError("acceptor oxygen coincides with the donor hydrogen")
    bisector = oxygen_bisector(out, o_idx)
    to_z = rotation_matrix_align_to_z(bisector)
    z_to_away = rotation_matrix_align_to_z(away).T
    out = apply_rotation(out, z_to_away @ to_z)
    out = translate_atom_to(out, o_idx, o_pos)
    ref = flatten_index(out, o_idx, h_idx)
    return _flatten_into_plane(out, o_pos, away, plane_normal, ref)


def _flatten_into_plane(
    atoms: Atoms,
    point: np.ndarray,
    axis: np.ndarray,
    plane_normal: np.ndarray,
    ref_idx: int,
) -> Atoms:
    """Rotate about ``axis`` through ``point`` so ``ref_idx`` lies in the H-bond plane."""
    axis_u = np.asarray(axis, dtype=np.float64)
    nrm = float(np.linalg.norm(axis_u))
    if nrm < COM_VECTOR_MIN_A:
        return atoms.copy()
    axis_u = axis_u / nrm
    normal = np.asarray(plane_normal, dtype=np.float64)
    normal = normal / float(np.linalg.norm(normal))
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)[int(ref_idx)]
    v = pos - np.asarray(point, dtype=np.float64)
    v_perp = v - axis_u * float(np.dot(v, axis_u))
    if float(np.linalg.norm(v_perp)) < PLANE_FLATTEN_MIN_A:
        return atoms.copy()
    target = np.cross(axis_u, normal)
    if float(np.linalg.norm(target)) < PLANE_FLATTEN_MIN_A:
        return atoms.copy()
    target = target / float(np.linalg.norm(target))
    v_u = v_perp / float(np.linalg.norm(v_perp))
    cosine = float(np.clip(np.dot(v_u, target), -1.0, 1.0))
    sine = float(np.dot(axis_u, np.cross(v_u, target)))
    angle = float(np.arctan2(sine, cosine))
    return apply_rotation(atoms, rotation_matrix_about_axis(axis_u, angle), origin=point)


def dimer_oh_o(
    monomer: Atoms,
    r_oo: float,
    *,
    dha_deg: float = 180.0,
    plane: str = PLANE_XZ,
    acceptor_flap_deg: float = DEFAULT_ACCEPTOR_FLAP_DEG,
) -> Atoms:
    """Linear (or bent) OH···O dimer at O–O = ``r_oo`` and O–H–O = ``dha_deg``.

    ``acceptor_flap_deg`` is the Cs water-dimer rotation of the acceptor about
    the axis through acceptor O perpendicular to the donor plane (X). DHA stays
    180° when ``dha_deg=180``.
    """
    donor = orient_hydroxyl_donor(monomer)
    o_d = hydroxyl_pair(donor)[0]
    h_d = plus_z_hydrogen(donor, o_d)
    r_oh = site_site_distance(donor, o_d, h_d)
    gamma = polar_angle_for_dha(float(r_oo), r_oh, float(dha_deg))
    o_a_pos = _plane_coords(gamma, float(r_oo), plane)
    h_pos = np.asarray(donor.get_positions(), dtype=np.float64)[h_d]
    # γ = 0 is on-axis: use the donor-plane flatten so 1D / xz / yz 180° match.
    flat_plane = PLANE_XZ if abs(gamma) < 1.0e-8 else plane
    normal = np.array([0.0, 1.0, 0.0]) if flat_plane == PLANE_XZ else np.array([1.0, 0.0, 0.0])
    acceptor = _orient_acceptor_at(monomer, o_a_pos, h_pos, normal)
    dimer = combine_fragments(donor, acceptor)
    return _apply_acceptor_flap(dimer, float(acceptor_flap_deg))


def _apply_acceptor_flap(dimer: Atoms, flap_deg: float) -> Atoms:
    if abs(float(flap_deg)) < 1.0e-8:
        return dimer
    n_a = int(np.sum(np.asarray(dimer.arrays["mol_id"]) == 0))
    acceptor = dimer[n_a:]
    o_idx = hydroxyl_pair(acceptor)[0]
    origin = np.asarray(acceptor.get_positions(), dtype=np.float64)[o_idx]
    rotated = apply_rotation(
        acceptor,
        rotation_matrix_about_axis((1.0, 0.0, 0.0), np.deg2rad(float(flap_deg))),
        origin=origin,
    )
    return combine_fragments(dimer[:n_a], rotated)


def dimer_acceptor_acceptor(monomer: Atoms, r_oo: float) -> Atoms:
    """O···O approach with both HOH bisectors pointing away (lone pairs facing)."""
    acceptor_a = orient_hydroxyl_acceptor(monomer)
    o_idx = hydroxyl_pair(acceptor_a)[0]
    acceptor_a = apply_rotation(acceptor_a, rotation_matrix_about_axis((1.0, 0.0, 0.0), np.pi))
    acceptor_b = orient_hydroxyl_acceptor(monomer)
    acceptor_b = translate_atom_to(acceptor_b, o_idx, (0.0, 0.0, float(r_oo)))
    return combine_fragments(acceptor_a, acceptor_b)


def _align_bond_from_to(monomer: Atoms, i_from: int, i_to: int, *, plus_z: bool) -> Atoms:
    out = translate_atom_to(monomer, i_from, (0.0, 0.0, 0.0))
    pos = np.asarray(out.get_positions(), dtype=np.float64)
    vec = pos[i_to] - pos[i_from]
    if not plus_z:
        vec = -vec
    out = _align_vector_to_plus_z(out, vec)
    heavy = [i for i, symbol in enumerate(out.get_chemical_symbols()) if i not in (i_from, i_to)]
    if heavy:
        out = _flatten_atom_to_xz(out, heavy[0])
    return out


def dimer_carbonyl_ch_o(monomer: Atoms, r_o_c: float) -> Atoms:
    """C–H···O=C: methyl C–H of B aimed at carbonyl O of A; scan O···C_methyl."""
    c_carb, o_idx = carbonyl_pair(monomer)
    methyl = methyl_carbon_indices(monomer)[0]
    h_idx = closest_hydrogen(monomer, methyl)
    acceptor = _align_bond_from_to(monomer, o_idx, c_carb, plus_z=False)
    # O at origin, C along −Z, oxygen facing +Z.
    donor = _align_bond_from_to(monomer, methyl, h_idx, plus_z=False)
    # Methyl C at origin, H along −Z. Place C at +r so H points at O.
    donor = translate_atom_to(donor, methyl, (0.0, 0.0, float(r_o_c)))
    return combine_fragments(acceptor, donor)


def dimer_methyl_methyl(monomer: Atoms, r_cc: float) -> Atoms:
    """Methyl–methyl control: two methyl carbons along Z, molecules pointing away."""
    methyl = methyl_carbon_indices(monomer)[0]
    c_carb, _o = carbonyl_pair(monomer)
    mol_a = _align_bond_from_to(monomer, methyl, c_carb, plus_z=False)
    # Methyl C at origin, rest of molecule along −Z.
    mol_b = _align_bond_from_to(monomer, methyl, c_carb, plus_z=True)
    mol_b = translate_atom_to(mol_b, methyl, (0.0, 0.0, float(r_cc)))
    return combine_fragments(mol_a, mol_b)


def dimer_for_orientation(monomer: Atoms, orientation: str, distance: float, **kwargs) -> Atoms:
    """Dispatch a named rigid dimer at the orientation's site–site distance."""
    if orientation == ORIENTATION_LINEAR_OH_O:
        return dimer_oh_o(monomer, distance, **kwargs)
    if orientation == ORIENTATION_ACCEPTOR_ACCEPTOR:
        return dimer_acceptor_acceptor(monomer, distance)
    if orientation == ORIENTATION_CARBONYL:
        return dimer_carbonyl_ch_o(monomer, distance)
    if orientation == ORIENTATION_METHYL:
        return dimer_methyl_methyl(monomer, distance)
    raise ValueError(f"unknown orientation {orientation!r}")


def scan_coordinate_for_orientation(orientation: str) -> str:
    if orientation in HYDROXYL_ORIENTATIONS:
        return SCAN_OO
    if orientation == ORIENTATION_CARBONYL:
        return SCAN_O_CMETHYL
    if orientation == ORIENTATION_METHYL:
        return SCAN_CMETHYL_CMETHYL
    raise ValueError(f"unknown orientation {orientation!r}")


def orientations_for_system(system: str) -> tuple[str, ...]:
    if system == "acetone":
        return ACETONE_ORIENTATIONS
    return HYDROXYL_ORIENTATIONS


def dimer_site_indices(dimer: Atoms, orientation: str) -> tuple[int, int]:
    """Atom indices of the scanned site–site pair in a two-fragment dimer."""
    n_a = int(np.sum(np.asarray(dimer.arrays["mol_id"]) == 0))
    mol_a, mol_b = dimer[:n_a], dimer[n_a:]
    if orientation in HYDROXYL_ORIENTATIONS:
        i = hydroxyl_pair(mol_a)[0]
        j = hydroxyl_pair(mol_b)[0] + n_a
        return i, j
    if orientation == ORIENTATION_CARBONYL:
        _c, o_idx = carbonyl_pair(mol_a)
        methyl = methyl_carbon_indices(mol_b)[0] + n_a
        return o_idx, methyl
    if orientation == ORIENTATION_METHYL:
        i = methyl_carbon_indices(mol_a)[0]
        j = methyl_carbon_indices(mol_b)[0] + n_a
        return i, j
    raise ValueError(f"unknown orientation {orientation!r}")


def dimer_dha_deg(dimer: Atoms) -> float:
    """Donor–H–acceptor angle (deg): H of A closest to acceptor O."""
    n_a = int(np.sum(np.asarray(dimer.arrays["mol_id"]) == 0))
    donor = dimer[:n_a]
    o_d = hydroxyl_pair(donor)[0]
    o_a = hydroxyl_pair(dimer[n_a:])[0] + n_a
    o_a_pos = np.asarray(dimer.get_positions(), dtype=np.float64)[o_a]
    d_pos = np.asarray(donor.get_positions(), dtype=np.float64)
    attached = attached_hydrogens(donor, o_d)
    h_d = min(attached, key=lambda h: float(np.linalg.norm(d_pos[h] - o_a_pos)))
    return bond_angle_deg(dimer, o_d, h_d, o_a)


def _rotate_plus_z_to(atoms: Atoms, direction: Sequence[float]) -> Atoms:
    rotation = rotation_matrix_align_to_z(np.asarray(direction, dtype=np.float64)).T
    return apply_rotation(atoms, rotation)


def linear_hbond_trimer(monomer: Atoms, r_oo: float) -> Atoms:
    """A donates to B donates to C along +Z; adjacent O–O = ``r_oo``."""
    o_idx, _h = hydroxyl_pair(monomer)
    copies: list[Atoms] = []
    for i in range(3):
        if i < 2:
            mol = orient_hydroxyl_donor(monomer)
        else:
            mol = orient_hydroxyl_acceptor(monomer)
        mol = translate_atom_to(mol, o_idx, (0.0, 0.0, float(i) * float(r_oo)))
        copies.append(mol)
    return combine_fragments(*copies)


def cyclic_hbond_trimer(monomer: Atoms, r_oo: float) -> Atoms:
    """Each hydroxyl donates to the next O around an equilateral O–O triangle."""
    o_idx, h_idx = hydroxyl_pair(monomer)
    radius = float(r_oo) / np.sqrt(3.0)
    angles = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    copies: list[Atoms] = []
    for i, theta in enumerate(angles):
        o_pos = radius * np.array([np.cos(theta), np.sin(theta), 0.0])
        next_theta = angles[(i + 1) % 3]
        next_o = radius * np.array([np.cos(next_theta), np.sin(next_theta), 0.0])
        direction = next_o - o_pos
        mol = orient_hydroxyl_donor(monomer)
        mol = _rotate_plus_z_to(mol, direction)
        mol = translate_atom_to(mol, o_idx, o_pos)
        copies.append(mol)
    combined = combine_fragments(*copies)
    # Silence unused names for flatten bookkeeping; h_idx documents the donor H.
    _ = h_idx
    return combined


def trimer_for_motif(monomer: Atoms, motif: str, r_oo: float) -> Atoms:
    if motif == MOTIF_LINEAR:
        return linear_hbond_trimer(monomer, r_oo)
    if motif == MOTIF_CYCLIC:
        return cyclic_hbond_trimer(monomer, r_oo)
    raise ValueError(f"unknown trimer motif {motif!r}")


def trimer_oo_distances(trimer: Atoms) -> tuple[float, float, float]:
    """The three intermolecular O–O distances (Å)."""
    mol_id = np.asarray(trimer.arrays["mol_id"], dtype=np.int64)
    ids = tuple(int(v) for v in np.unique(mol_id))
    oxygens: list[np.ndarray] = []
    pos = np.asarray(trimer.get_positions(), dtype=np.float64)
    symbols = trimer.get_chemical_symbols()
    for mol in ids:
        o_local = [i for i, (mid, sym) in enumerate(zip(mol_id, symbols, strict=True)) if mid == mol and sym == "O"]
        if not o_local:
            raise ValueError("trimer fragment has no oxygen")
        oxygens.append(pos[o_local[0]])
    return (
        float(np.linalg.norm(oxygens[0] - oxygens[1])),
        float(np.linalg.norm(oxygens[0] - oxygens[2])),
        float(np.linalg.norm(oxygens[1] - oxygens[2])),
    )


def equilateral_trimer(monomer: Atoms, side_angstrom: float) -> Atoms:
    """Three rigid COM copies on an equilateral triangle (MBE-algebra helper)."""
    template = centered_atoms(monomer, center="com")
    n = len(template)
    copies = [template.copy() for _ in range(3)]
    copies[1].translate(np.array([side_angstrom, 0.0, 0.0]))
    copies[2].translate(np.array([0.5 * side_angstrom, 0.5 * np.sqrt(3.0) * side_angstrom, 0.0]))
    combined = copies[0] + copies[1] + copies[2]
    combined = assign_mol_id(combined, [n, n, n])
    combined.translate(-np.asarray(combined.get_center_of_mass(), dtype=np.float64))
    return combined
