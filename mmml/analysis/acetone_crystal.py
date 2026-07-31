"""The five crystal structures of acetone from Allan et al., and how to check them.

Source: D. R. Allan, S. J. Clark, R. M. Ibberson, S. Parsons, C. R. Pulham and
L. Sawyer, "The influence of pressure and temperature on the crystal structure
of acetone", *Chem. Commun.* 1999, 751 (doi:10.1039/a900558g), deposited as
CCDC 182/1197 and redistributed by the Crystallography Open Database.

The paper's argument is structural: acetone packs through dipolar
carbonyl-carbonyl contacts, and all three archetypal motifs of Allen et al.
(*Acta Crystallogr.* B54, 320) appear across its phases. The broad heat-capacity
anomaly near 127 K, unexplained since Kelley's 1929 measurement, is attributed
to those contacts plus C-H...O contacts shortening on cooling. Reproducing those
distances from a built cell is therefore the sharpest available check that the
structure was assembled correctly, which is what :func:`carbonyl_contacts` and
:func:`ch_o_contacts` are for.

The published distances are recorded per phase so a build can be compared
against them without re-reading the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "AcetonePhase",
    "ACETONE_CRYSTAL_PHASES",
    "ACETONE_SUBLIMATION_REFERENCE",
    "Contact",
    "SublimationReference",
    "acetone_phase",
    "read_acetone_phase",
    "carbonyl_contacts",
    "ch_o_contacts",
    "classify_carbonyl_motif",
]


@dataclass(frozen=True)
class SublimationReference:
    """Experimental sublimation enthalpy assembled from a thermodynamic cycle.

    The NIST WebBook lists no direct sublimation measurement for acetone, so
    ``dH_sub(T_fus) = dH_vap(T_fus) + dH_fus(T_fus)`` is the available route.
    Both legs are quoted near the melting point rather than at 298 K, because a
    room-temperature ``dH_vap`` would be compared against a crystal that does
    not exist at that temperature.

    Treat the result as an estimate good to a kJ/mol or so, not a measurement.
    The two legs come from different sources at slightly different temperatures,
    and the heat capacity difference between phases means the number still drifts
    with temperature -- which is the whole subject of the Allan et al. paper.
    """

    dvap_h_kj_mol: float = 32.9
    dvap_h_temperature_K: float = 228.0
    dvap_h_source: str = "Stephenson & Malanowski 1987 (data 178-243 K), via NIST WebBook"
    dfus_h_kj_mol: float = 5.72
    dfus_h_temperature_K: float = 176.6
    dfus_h_source: str = "Kelley 1929 / Domalski & Hearing 1996, via NIST WebBook"

    @property
    def dsub_h_kj_mol(self) -> float:
        return self.dvap_h_kj_mol + self.dfus_h_kj_mol

    @property
    def dsub_h_kcal_mol(self) -> float:
        return self.dsub_h_kj_mol / 4.184


# Kelley's 1929 calorimetry is the same study that first saw the heat-capacity
# anomaly near 127 K which the Allan et al. paper set out to explain.
ACETONE_SUBLIMATION_REFERENCE = SublimationReference()


@dataclass(frozen=True)
class AcetonePhase:
    """One published acetone structure with the numbers needed to check a build."""

    key: str
    cod_id: int
    label: str
    space_group: str
    space_group_number: int
    cell_lengths_A: tuple[float, float, float]
    cell_volume_A3: float
    temperature_K: float
    pressure_kbar: float
    z: int
    ordered_hydrogens: bool
    deuterated: bool
    note: str
    # Published intermolecular distances (Angstrom), keyed by the paper's motif
    # names. Absent keys simply were not quoted for that phase.
    published_contacts: dict[str, float] = field(default_factory=dict)

    @property
    def usable_for_mm(self) -> bool:
        """Whether the deposited hydrogens support a force-field calculation."""
        return self.ordered_hydrogens

    def cif_path(self):
        from mmml.paths import default_acetone_crystal_cif

        return default_acetone_crystal_cif(self.key)


ACETONE_CRYSTAL_PHASES: dict[str, AcetonePhase] = {
    "pbca_5k": AcetonePhase(
        key="pbca_5k",
        cod_id=7110465,
        label="Pbca, 5 K, neutron powder (acetone-d6)",
        space_group="Pbca",
        space_group_number=61,
        cell_lengths_A=(9.16686, 7.53231, 21.24861),
        cell_volume_A3=1467.17,
        temperature_K=5.0,
        pressure_kbar=0.0,
        z=16,
        ordered_hydrogens=True,
        deuterated=True,
        note=(
            "Rietveld refinement against HRPD time-of-flight data; the phase where "
            "the paper reports the dipolar and C-H...O contacts at their shortest."
        ),
        published_contacts={
            "type_i_perpendicular": 3.391,
            "type_i_perpendicular_layer_a": 3.368,
            "type_ii_antiparallel": 3.231,
            "ch_o_between_chains": 2.336,
            "ch_o_within_chains": 2.479,
        },
    ),
    "pbca_110k": AcetonePhase(
        key="pbca_110k",
        cod_id=7110466,
        label="Pbca, 110 K, single-crystal X-ray",
        space_group="Pbca",
        space_group_number=61,
        cell_lengths_A=(9.172, 7.761, 21.66),
        cell_volume_A3=1542.0,
        temperature_K=110.0,
        pressure_kbar=0.0,
        z=16,
        ordered_hydrogens=True,
        deuterated=False,
        note=(
            "Below the heat-capacity anomaly: the paper reports three contacts "
            "'locking in' at shortened distances relative to 150 K."
        ),
        published_contacts={
            "type_i_perpendicular": 3.417,
            "ch_o_between_chains": 2.511,
            "ch_o_within_chains": 2.604,
        },
    ),
    "pbca_150k": AcetonePhase(
        key="pbca_150k",
        cod_id=7110464,
        label="Pbca, 150 K, single-crystal X-ray (stable low-temperature phase)",
        space_group="Pbca",
        space_group_number=61,
        cell_lengths_A=(8.873, 8.000, 22.027),
        cell_volume_A3=1563.5,
        temperature_K=150.0,
        pressure_kbar=0.0,
        z=16,
        ordered_hydrogens=True,
        deuterated=False,
        note=(
            "The stable phase all low-temperature crystallisations yielded. Layers "
            "stack along c; c is roughly twice the high-pressure value because "
            "neighbouring layers are crystallographically independent."
        ),
        published_contacts={
            "type_i_perpendicular": 3.491,
            "type_i_perpendicular_chains": 3.458,
            "type_ii_antiparallel": 3.300,
            "ch_o_between_chains": 2.618,
            "ch_o_within_chains": 2.71,
        },
    ),
    "cmcm_160k": AcetonePhase(
        key="cmcm_160k",
        cod_id=7110463,
        label="Cmcm, 160 K, metastable C-centred phase",
        space_group="Cmcm",
        space_group_number=63,
        cell_lengths_A=(6.514, 5.4159, 10.756),
        cell_volume_A3=379.5,
        temperature_K=160.0,
        pressure_kbar=0.0,
        z=4,
        ordered_hydrogens=True,
        deuterated=False,
        note=(
            "Obtained twice only, by cooling slowly through the melting point; "
            "decomposes to the Pbca phase within hours. The paper's text quotes "
            "165 K for this data set while the deposited CIF records 160 K."
        ),
        published_contacts={"type_iii_sheared_parallel": 3.587},
    ),
    "cmcm_15kbar": AcetonePhase(
        key="cmcm_15kbar",
        cod_id=7110462,
        label="Cmcm, room temperature, 15 kbar",
        space_group="Cmcm",
        space_group_number=63,
        cell_lengths_A=(6.1219, 5.2029, 10.244),
        cell_volume_A3=326.29,
        temperature_K=293.0,
        pressure_kbar=15.0,
        z=4,
        ordered_hydrogens=False,
        deuterated=False,
        note=(
            "Methyl groups are rotationally disordered about the C-C axis, so the "
            "deposited cell carries 12 half-occupancy hydrogens per molecule. Use "
            "it for the packing motif; a force field needs one ordered rotamer "
            "chosen first."
        ),
        published_contacts={"type_iii_sheared_parallel": 3.365},
    ),
}


def acetone_phase(key: str) -> AcetonePhase:
    """Look up one phase, with a helpful error listing the alternatives."""
    k = key.strip().lower()
    try:
        return ACETONE_CRYSTAL_PHASES[k]
    except KeyError:
        known = ", ".join(ACETONE_CRYSTAL_PHASES)
        raise KeyError(f"Unknown acetone phase {key!r}; known phases: {known}") from None


def read_acetone_phase(key: str, *, protiate: bool = True) -> Any:
    """Read one phase's CIF as a full unit cell of :class:`ase.Atoms`.

    ASE applies the deposited symmetry operators, so the Pbca entries come back
    as all 16 molecules rather than the two-molecule asymmetric unit.

    ``protiate`` resets deuterium masses to hydrogen. The 5 K structure was
    refined against neutron data on acetone-d6, and ASE preserves the D masses;
    left alone they inflate the computed density by 10% and would silently
    contaminate any comparison against the protiated phases.
    """
    from ase.data import atomic_masses
    from ase.io import read

    phase = acetone_phase(key)
    atoms = read(str(phase.cif_path()))
    if protiate:
        atoms.set_masses(atomic_masses[atoms.get_atomic_numbers()])
    return atoms


@dataclass(frozen=True)
class Contact:
    """One intermolecular contact between molecules ``i`` and ``j``."""

    distance_A: float
    mol_i: int
    mol_j: int
    atom_i: int
    atom_j: int
    angle_deg: float | None = None
    motif: str | None = None


def _molecular_frames(atoms: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap into molecules; return ``(mol_id, positions, cell)``."""
    from mmml.analysis.lattice_energy import unwrap_molecules

    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    mol_id, positions = unwrap_molecules(
        np.asarray(atoms.get_positions(), dtype=np.float64),
        np.asarray(atoms.get_atomic_numbers(), dtype=int),
        cell,
    )
    return mol_id, positions, cell


def _carbonyl_indices(
    positions: np.ndarray, z: np.ndarray, mol_id: np.ndarray
) -> list[tuple[int, int]]:
    """``(carbonyl_carbon, oxygen)`` index pairs, one per molecule."""
    pairs: list[tuple[int, int]] = []
    for m in range(int(mol_id.max()) + 1):
        sel = np.flatnonzero(mol_id == m)
        oxygens = sel[z[sel] == 8]
        carbons = sel[z[sel] == 6]
        if len(oxygens) != 1 or len(carbons) == 0:
            raise ValueError(f"molecule {m} is not a ketone: {len(oxygens)} O, {len(carbons)} C")
        o = int(oxygens[0])
        d = np.linalg.norm(positions[carbons] - positions[o], axis=1)
        pairs.append((int(carbons[int(np.argmin(d))]), o))
    return pairs


def classify_carbonyl_motif(angle_deg: float) -> str:
    """Name a carbonyl-carbonyl motif from the angle between the two C=O vectors.

    A deliberately coarse stand-in for the full geometric criteria of Allen et
    al. (*Acta Crystallogr.* B54, 320), which also use the C...O=C angles and the
    offset between the dipoles. The angle alone separates the three archetypes
    well enough to label a contact for inspection; it is not a substitute for
    the published classification.
    """
    a = abs(float(angle_deg))
    if a > 135.0:
        return "II (antiparallel)"
    if a < 45.0:
        return "III (sheared-parallel)"
    return "I (perpendicular)"


def carbonyl_contacts(
    atoms: Any,
    *,
    max_distance_A: float = 4.0,
    tolerance_A: float = 5e-3,
) -> list[Contact]:
    """Intermolecular carbonyl C...O contacts, shortest first.

    Each contact is directed: ``C`` of one molecule approaching ``O`` of
    another. Symmetry-equivalent contacts are collapsed to one entry, so the
    list reads like the distances quoted in a crystallographic paper rather
    than a per-atom dump.
    """
    from mmml.analysis.lattice_energy import lattice_shift_vectors, molecular_reach_A

    mol_id, positions, cell = _molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    pairs = _carbonyl_indices(positions, z, mol_id)

    c_idx = np.array([p[0] for p in pairs])
    o_idx = np.array([p[1] for p in pairs])
    co_vec = positions[o_idx] - positions[c_idx]
    co_unit = co_vec / np.linalg.norm(co_vec, axis=1)[:, None]

    reach = molecular_reach_A(positions, mol_id)
    shifts = lattice_shift_vectors(cell, max_distance_A, reach_A=reach)

    found: list[Contact] = []
    for shift in shifts:
        is_home = not np.any(shift)
        delta = (positions[o_idx][None, :, :] + shift) - positions[c_idx][:, None, :]
        dist = np.linalg.norm(delta, axis=-1)
        keep = dist < max_distance_A
        if is_home:
            keep &= ~np.eye(len(pairs), dtype=bool)
        for i, j in zip(*np.nonzero(keep)):
            cos = float(np.clip(np.dot(co_unit[i], co_unit[j]), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(cos)))
            found.append(
                Contact(
                    distance_A=float(dist[i, j]),
                    mol_i=int(i),
                    mol_j=int(j),
                    atom_i=int(c_idx[i]),
                    atom_j=int(o_idx[j]),
                    angle_deg=angle,
                    motif=classify_carbonyl_motif(angle),
                )
            )
    return _collapse_equivalent(found, tolerance_A)


def ch_o_contacts(
    atoms: Any,
    *,
    max_distance_A: float = 3.0,
    tolerance_A: float = 5e-3,
) -> list[Contact]:
    """Intermolecular H...O contacts, shortest first.

    The paper quotes H...O separations rather than C...O for the hydrogen bonds,
    and notes that X-ray and neutron H positions are not directly comparable --
    X-ray C-H distances are systematically short, which is why the 150 K
    contacts are quoted after normalising C-H to 1.08 A. No such normalisation
    is applied here; compare X-ray phases with X-ray phases.
    """
    from mmml.analysis.lattice_energy import lattice_shift_vectors, molecular_reach_A

    mol_id, positions, cell = _molecular_frames(atoms)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    h_idx = np.flatnonzero(z == 1)
    o_idx = np.flatnonzero(z == 8)

    reach = molecular_reach_A(positions, mol_id)
    shifts = lattice_shift_vectors(cell, max_distance_A, reach_A=reach)

    found: list[Contact] = []
    for shift in shifts:
        delta = (positions[o_idx][None, :, :] + shift) - positions[h_idx][:, None, :]
        dist = np.linalg.norm(delta, axis=-1)
        keep = dist < max_distance_A
        different = mol_id[h_idx][:, None] != mol_id[o_idx][None, :]
        keep &= different if not np.any(shift) else np.ones_like(keep)
        for i, j in zip(*np.nonzero(keep)):
            found.append(
                Contact(
                    distance_A=float(dist[i, j]),
                    mol_i=int(mol_id[h_idx[i]]),
                    mol_j=int(mol_id[o_idx[j]]),
                    atom_i=int(h_idx[i]),
                    atom_j=int(o_idx[j]),
                )
            )
    return _collapse_equivalent(found, tolerance_A)


def _collapse_equivalent(contacts: list[Contact], tolerance_A: float) -> list[Contact]:
    """Keep one representative per symmetry-equivalent distance, shortest first."""
    out: list[Contact] = []
    for contact in sorted(contacts, key=lambda c: c.distance_A):
        if any(abs(contact.distance_A - kept.distance_A) < tolerance_A for kept in out):
            continue
        out.append(contact)
    return out
