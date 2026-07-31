"""Crystalline dichloromethane: the two measured pressure points, and what holds it together.

Source: M. Podsiadło, K. F. Dziubek and A. Katrusiak, "In situ high-pressure
crystallization and compression of halogen contacts in dichloromethane", *Acta
Crystallogr.* B **61**, 595 (2005), doi:10.1107/S0108768105017374, deposited as
CCDC doi:10.5517/cc9lyjb and redistributed by the Crystallography Open Database.

The paper's conclusion is a claim about which interaction holds the crystal
together::

    "the crystal cohesion forces are dominated by H...Cl interactions rather
     than by Cl...Cl attractions"

That is unusually testable for a structural paper: it is a statement about
energy, not geometry, and
:func:`mmml.analysis.lattice_energy.decompose_lattice_energy_by_element_pair`
answers it directly by splitting the lattice energy over molecule pairs. A
later plane-wave DFT study reached the same conclusion independently --
D. Kurzydłowski, T. Chumak and J. Rogoża, *Crystals* **10**, 920 (2020),
doi:10.3390/cryst10100920 -- finding that for CH2Cl2 "dipole-dipole interactions
and hydrogen bonds are the main factors" with "halogen bonds playing only a
minor role".

Both deposited structures are high-pressure ones. That matters more than it
might seem: a crystal held at 1.33 or 1.63 GPa is compressed onto its repulsive
wall, so its static lattice energy is *not* a cohesive energy and must not be
compared against a sublimation enthalpy. The ambient-pressure structure is
Kawaguchi et al. (1973), which predates CIF deposition and has no openly
licensed coordinates; only its cell was published. :data:`KAWAGUCHI_AMBIENT_CELL`
records that cell so a force-field relaxation to zero pressure has something to
be judged against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmml.analysis.crystal_contacts import (
    Contact,
    collapse_equivalent,
    element_pair_contacts,
    molecular_frames,
)
from mmml.analysis.lattice_energy import SublimationReference

__all__ = [
    "DcmPhase",
    "DCM_CRYSTAL_PHASES",
    "DCM_SUBLIMATION_REFERENCE",
    "KAWAGUCHI_AMBIENT_CELL",
    "AmbientCellReference",
    "classify_halogen_motif",
    "dcm_phase",
    "read_dcm_phase",
    "rebuild_methylene_hydrogens",
    "halogen_contacts",
    "h_cl_contacts",
    "GAS_PHASE_CH_A",
    "GAS_PHASE_HCH_DEG",
]

# Gas-phase CH2Cl2 geometry from microwave spectroscopy; used to rebuild the
# hydrogens, whose X-ray positions are not determined well enough to compare.
GAS_PHASE_CH_A: float = 1.087
GAS_PHASE_HCH_DEG: float = 112.0


@dataclass(frozen=True)
class AmbientCellReference:
    """A published unit cell with no openly available coordinates.

    Enough to judge a relaxed cell against, and not enough to build from. Kept
    as a distinct type from :class:`DcmPhase` so the difference cannot be
    overlooked: there is no CIF behind this one.
    """

    label: str
    space_group: str
    cell_lengths_A: tuple[float, float, float]
    temperature_K: float
    z: int
    citation: str
    note: str

    @property
    def cell_volume_A3(self) -> float:
        a, b, c = self.cell_lengths_A
        return a * b * c


# Kawaguchi, Tanaka, Takeuchi & Watanabé, Bull. Chem. Soc. Jpn. 46, 62 (1973).
# Cell dimensions are facts, not expression, so recording them here carries no
# licensing question; the atomic coordinates are not reproduced.
KAWAGUCHI_AMBIENT_CELL = AmbientCellReference(
    label="Pbcn, ~153 K, ambient pressure",
    space_group="Pbcn",
    cell_lengths_A=(4.249, 8.138, 9.492),
    temperature_K=153.0,
    z=4,
    citation=(
        "Kawaguchi, Tanaka, Takeuchi & Watanabe, Bull. Chem. Soc. Jpn. 46, 62 "
        "(1973), doi:10.1246/bcsj.46.62"
    ),
    note=(
        "The structure the high-pressure phases are isostructural with. "
        "Determined before CIF deposition existed, so only the cell is quoted "
        "here; there is no open coordinate set to build from."
    ),
)


# dH_vap and dH_fus for CH2Cl2, both via the NIST Chemistry WebBook. The
# vaporisation leg is the lowest-temperature entry anchored to a real data range
# (233-313 K); everything closer to the melting point is an extrapolation.
DCM_SUBLIMATION_REFERENCE = SublimationReference(
    dvap_h_kj_mol=30.2,
    dvap_h_temperature_K=248.0,
    dvap_h_source="Ganeff & Jungers (data 233-313 K), via NIST WebBook",
    dfus_h_kj_mol=6.16,
    dfus_h_temperature_K=178.2,
    dfus_h_source="Moseeva et al. 1978 / Domalski & Hearing 1996, via NIST WebBook",
)


@dataclass(frozen=True)
class DcmPhase:
    """One deposited dichloromethane structure."""

    key: str
    cod_id: int
    label: str
    space_group: str
    space_group_number: int
    cell_lengths_A: tuple[float, float, float]
    cell_volume_A3: float
    density_g_cm3: float
    temperature_K: float
    pressure_GPa: float
    z: int
    note: str
    published_contacts: dict[str, float] = field(default_factory=dict)

    @property
    def usable_for_mm(self) -> bool:
        """Both deposited structures have refined, ordered hydrogens."""
        return True

    def cif_path(self):
        from mmml.paths import default_dcm_crystal_cif

        return default_dcm_crystal_cif(self.key)


DCM_CRYSTAL_PHASES: dict[str, DcmPhase] = {
    "pbcn_133gpa": DcmPhase(
        key="pbcn_133gpa",
        cod_id=2100014,
        label="Pbcn, 1.33 GPa, 293 K, single-crystal X-ray",
        space_group="Pbcn",
        space_group_number=60,
        cell_lengths_A=(3.984, 7.863, 9.357),
        cell_volume_A3=293.12,
        density_g_cm3=1.920,
        temperature_K=293.0,
        pressure_GPa=1.33,
        z=4,
        note=(
            "The lower of the two pressure points, grown in situ in a "
            "diamond-anvil cell. Closest of the two to ambient conditions and "
            "so the better starting point for a relaxation to zero pressure."
        ),
    ),
    "pbcn_163gpa": DcmPhase(
        key="pbcn_163gpa",
        cod_id=2100015,
        label="Pbcn, 1.63 GPa, 293 K, single-crystal X-ray",
        space_group="Pbcn",
        space_group_number=60,
        cell_lengths_A=(3.924, 7.793, 9.335),
        cell_volume_A3=285.46,
        density_g_cm3=1.972,
        temperature_K=293.0,
        pressure_GPa=1.63,
        z=4,
        note=(
            "The more compressed point, and the structure the bundled ``dcm`` "
            "build-crystal preset and the liquid-box density tables have always "
            "used. 2.6% smaller in volume than the 1.33 GPa cell."
        ),
    ),
}


def dcm_phase(key: str) -> DcmPhase:
    """Look up one phase, with an error listing the alternatives."""
    k = key.strip().lower()
    try:
        return DCM_CRYSTAL_PHASES[k]
    except KeyError:
        known = ", ".join(DCM_CRYSTAL_PHASES)
        raise KeyError(f"Unknown DCM phase {key!r}; known phases: {known}") from None


def read_dcm_phase(key: str, *, rebuild_hydrogens: bool = False) -> Any:
    """Read one phase's CIF as a full unit cell of :class:`ase.Atoms`.

    ASE applies the deposited symmetry operators, so the four molecules of the
    Pbcn cell come back from an asymmetric unit of half a molecule (the CH2Cl2
    sits on a crystallographic twofold axis).

    ``rebuild_hydrogens`` is off by default so that what is returned is what was
    deposited. Turn it on for anything that compares the two structures with
    each other or feeds them to a force field -- see
    :func:`rebuild_methylene_hydrogens` for why that is not optional.
    """
    from ase.io import read

    atoms = read(str(dcm_phase(key).cif_path()))
    return rebuild_methylene_hydrogens(atoms) if rebuild_hydrogens else atoms


def rebuild_methylene_hydrogens(
    atoms: Any,
    *,
    ch_A: float = GAS_PHASE_CH_A,
    hch_deg: float = GAS_PHASE_HCH_DEG,
) -> Any:
    """Replace every CH2Cl2 hydrogen using the well-determined heavy-atom frame.

    Returns a copy. Normalising the C-H *distance* is not enough for these two
    structures: their refined hydrogens differ in direction as well, by about
    0.3 A in the fractional z coordinate, which is comparable to the quoted
    uncertainty and larger than the compression being studied. X-rays scatter
    from electrons and there are two of them on a hydrogen, so this is expected
    rather than a defect of the refinement.

    The carbon and chlorines are determined to a few thousandths of an Angstrom,
    and CH2Cl2 has C2v symmetry, so the hydrogens follow from them: they sit in
    the plane bisecting the Cl-C-Cl angle, perpendicular to it, opening by
    ``hch_deg``. Only two spectroscopic constants enter, both known far better
    than the diffraction data can place a hydrogen.

    The effect is not cosmetic -- it moves the CGenFF lattice energy by about
    0.2 kcal/mol and it decides the sign of the apparent change in the shortest
    H...Cl contact between the two pressure points.
    """
    mol_id, positions, _ = molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    # Heavy atoms keep the coordinates they were handed, wrapping included; only
    # the hydrogens move, placed relative to their own carbon.
    moved = np.asarray(atoms.get_positions(), dtype=np.float64).copy()
    half_angle = np.radians(float(hch_deg)) / 2.0

    for m in range(int(mol_id.max()) + 1):
        sel = np.flatnonzero(mol_id == m)
        carbons = sel[z[sel] == 6]
        chlorines = sel[z[sel] == 17]
        hydrogens = sel[z[sel] == 1]
        if len(carbons) != 1 or len(chlorines) != 2 or len(hydrogens) != 2:
            raise ValueError(
                f"molecule {m} is not CH2Cl2: {len(carbons)} C, "
                f"{len(chlorines)} Cl, {len(hydrogens)} H"
            )
        c = positions[carbons[0]]
        e1 = positions[chlorines[0]] - c
        e2 = positions[chlorines[1]] - c
        e1 /= np.linalg.norm(e1)
        e2 /= np.linalg.norm(e2)
        bisector = e1 + e2
        bisector /= np.linalg.norm(bisector)
        normal = np.cross(e1, e2)
        normal /= np.linalg.norm(normal)
        # Hydrogens open away from the chlorines, symmetric about the Cl-C-Cl
        # plane -- the C2v arrangement.
        carbon_as_given = moved[carbons[0]]
        for h, sign in zip(hydrogens, (+1.0, -1.0)):
            direction = -bisector * np.cos(half_angle) + sign * normal * np.sin(half_angle)
            moved[h] = carbon_as_given + float(ch_A) * direction

    out = atoms.copy()
    out.set_positions(moved)
    return out


def classify_halogen_motif(theta1_deg: float, theta2_deg: float) -> str:
    """Name a Cl...Cl motif from the two C-Cl...Cl angles.

    The Desiraju & Parthasarathy classification (*J. Am. Chem. Soc.* **111**,
    8725, 1989): Type I is the symmetric geometry with ``theta1 ~ theta2``,
    which arises from close packing and carries no electrostatic preference;
    Type II has ``theta1 ~ 180`` and ``theta2 ~ 90``, putting the electropositive
    sigma-hole of one chlorine against the electron-rich belt of the other, and
    is the genuinely attractive halogen bond.

    Contacts that are neither are reported as ``intermediate`` rather than
    forced into a bin.
    """
    hi, lo = max(float(theta1_deg), float(theta2_deg)), min(
        float(theta1_deg), float(theta2_deg)
    )
    if hi - lo <= 20.0:
        return "I (symmetric, close packing)"
    if hi >= 150.0 and lo <= 120.0:
        return "II (halogen bond)"
    return "intermediate"


def _bonded_carbon(positions: np.ndarray, z: np.ndarray, atom: int) -> int:
    """Index of the carbon bonded to ``atom`` within the same (unwrapped) molecule."""
    carbons = np.flatnonzero(z == 6)
    d = np.linalg.norm(positions[carbons] - positions[atom], axis=1)
    return int(carbons[int(np.argmin(d))])


def halogen_contacts(
    atoms: Any,
    *,
    max_distance_A: float = 4.2,
    tolerance_A: float = 5e-3,
) -> list[Contact]:
    """Intermolecular Cl...Cl contacts with their Desiraju-Parthasarathy type.

    ``angle_deg`` carries the larger of the two C-Cl...Cl angles, which is the
    one that distinguishes a halogen bond from a packing contact.
    """
    from mmml.analysis.lattice_energy import lattice_shift_vectors, molecular_reach_A

    mol_id, positions, cell = molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    cl_idx = np.flatnonzero(z == 17)
    if not len(cl_idx):
        return []
    carbon_of = {int(i): _bonded_carbon(positions, z, int(i)) for i in cl_idx}

    reach = molecular_reach_A(positions, mol_id)
    shifts = lattice_shift_vectors(cell, max_distance_A, reach_A=reach)

    found: list[Contact] = []
    for shift in shifts:
        is_home = not np.any(shift)
        other = positions[cl_idx] + shift
        delta = other[None, :, :] - positions[cl_idx][:, None, :]
        dist = np.linalg.norm(delta, axis=-1)
        keep = dist < max_distance_A
        keep &= (
            mol_id[cl_idx][:, None] != mol_id[cl_idx][None, :]
            if is_home
            else dist > 1e-8
        )
        for i, j in zip(*np.nonzero(keep)):
            ai, aj = int(cl_idx[i]), int(cl_idx[j])
            # theta1 at the first chlorine, theta2 at the second: the C-Cl bond
            # vector against the Cl...Cl contact vector, from each end.
            v_contact = delta[i, j]
            v_c1 = positions[carbon_of[ai]] - positions[ai]
            v_c2 = (positions[carbon_of[aj]] + shift) - other[j]
            theta1 = _angle_between(v_c1, v_contact)
            theta2 = _angle_between(v_c2, -v_contact)
            found.append(
                Contact(
                    distance_A=float(dist[i, j]),
                    mol_i=int(mol_id[ai]),
                    mol_j=int(mol_id[aj]),
                    atom_i=ai,
                    atom_j=aj,
                    angle_deg=max(theta1, theta2),
                    motif=classify_halogen_motif(theta1, theta2),
                )
            )
    return collapse_equivalent(found, tolerance_A)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(
        np.clip(
            np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cos)))


def h_cl_contacts(
    atoms: Any,
    *,
    max_distance_A: float = 3.4,
    tolerance_A: float = 5e-3,
    rebuild_hydrogens: bool = True,
) -> list[Contact]:
    """Intermolecular H...Cl contacts, shortest first.

    ``rebuild_hydrogens`` defaults to on, and should stay on for anything
    quantitative: the two deposited structures refined C-H to 1.01(10) and
    1.13(12) A and disagree on the hydrogen direction by a comparable amount,
    which is larger than the compression between them. See
    :func:`rebuild_methylene_hydrogens`.
    """
    if rebuild_hydrogens:
        atoms = rebuild_methylene_hydrogens(atoms)
    return element_pair_contacts(
        atoms, "H", "Cl", max_distance_A=max_distance_A, tolerance_A=tolerance_A
    )
