"""CGENFF bonded topology and parameters for JAX MM calculators.

Loads CHARMM RTF/PRM connectivity and maps atom-type parameters onto a concrete
PSF-like index list.  File parsing reuses :mod:`jax_md.mm_forcefields.io.charmm`
so bonded arrays stay aligned with the jax-md reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM, CGENFF_RTF

try:
    from jax_md.mm_forcefields.base import BondedParameters, Topology
    from jax_md.mm_forcefields.io.charmm import parse_prm, parse_pdb_simple
    from jax_md.mm_forcefields.oplsaa.topology import create_topology
except ImportError as exc:  # pragma: no cover - jax-md is a core dependency
    raise ImportError(
        "jax-md mm_forcefields is required for CGENFF bonded topology loading"
    ) from exc


@dataclass(frozen=True, slots=True)
class CgenffBondedSystem:
    """Bonded topology + parameters for one or more molecules in index order."""

    positions: Array
    topology: Topology
    bonded: BondedParameters
    atom_types: tuple[str, ...]
    charges: Array

    @property
    def n_atoms(self) -> int:
        return int(self.topology.n_atoms)


def default_cgenff_paths() -> tuple[Path, Path]:
    return Path(CGENFF_RTF), Path(CGENFF_PRM)


def extract_residue_rtf(full_rtf: Path | str, residue_name: str) -> str:
    """Return a minimal RTF string for one ``RESI`` template from a full CGENFF RTF."""
    path = Path(full_rtf)
    target = residue_name.strip().upper()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    header: list[str] = []
    residue_lines: list[str] = []
    in_residue = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("MASS"):
            header.append(line)
            continue
        if stripped.startswith("RESI"):
            name = stripped.split()[1]
            if name.upper() == target:
                in_residue = True
                residue_lines = [line]
                continue
            if in_residue:
                break
        elif in_residue:
            if stripped.startswith("RESI") or stripped.startswith("END"):
                break
            residue_lines.append(line)

    if not residue_lines:
        raise ValueError(f"RESI {target!r} not found in {path}")

    return "\n".join(
        [
            "* CGENFF residue extract",
            "*",
            "",
            *header,
            "",
            *residue_lines,
            "",
            "END",
        ]
    )


def _strip_rtf_comment(line: str) -> str:
    if "!" in line:
        return line.split("!", 1)[0].strip()
    return line.strip()


def _parse_residue_connectivity(rtf_source: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str]]]:
    """Parse BOND and IMPR records from a single-residue RTF snippet."""
    bonds: list[tuple[str, str]] = []
    impropers: list[tuple[str, str, str, str]] = []
    for raw in rtf_source.splitlines():
        line = _strip_rtf_comment(raw)
        if not line:
            continue
        if line.startswith("BOND"):
            parts = line.split()[1:]
            for j in range(0, len(parts) - 1, 2):
                bonds.append((parts[j], parts[j + 1]))
        elif line.startswith("IMPR"):
            parts = line.split()[1:]
            if len(parts) >= 4:
                impropers.append((parts[0], parts[1], parts[2], parts[3]))
    return bonds, impropers


def _parse_residue_atoms(rtf_source: str) -> list[tuple[str, str, float]]:
    """Return ``(name, type, charge)`` for ATOM records in one RESI block."""
    atoms: list[tuple[str, str, float]] = []
    in_residue = False
    for raw in rtf_source.splitlines():
        line = _strip_rtf_comment(raw)
        if line.startswith("RESI"):
            in_residue = True
            continue
        if not in_residue:
            continue
        if line.startswith("RESI") or line == "END":
            break
        if line.startswith("ATOM"):
            parts = line.split()
            if len(parts) >= 4:
                atoms.append((parts[1], parts[2], float(parts[3])))
    return atoms


def _infer_angles_and_torsions(
    bonds: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    adjacency: list[set[int]] = [set() for _ in range(n_atoms)]
    for idx1, idx2 in bonds:
        i1 = int(idx1)
        i2 = int(idx2)
        adjacency[i1].add(i2)
        adjacency[i2].add(i1)

    angles: list[list[int]] = []
    for center in range(n_atoms):
        neighbors = sorted(adjacency[center])
        for i, idx1 in enumerate(neighbors):
            for idx2 in neighbors[i + 1 :]:
                angles.append([idx1, center, idx2])

    torsions: list[list[int]] = []
    for idx2, idx3 in bonds:
        i2 = int(idx2)
        i3 = int(idx3)
        for idx1 in adjacency[i2]:
            if idx1 == i3:
                continue
            for idx4 in adjacency[i3]:
                if idx4 == i2:
                    continue
                torsions.append([idx1, i2, i3, idx4])

    angle_arr = (
        np.asarray(angles, dtype=np.int32)
        if angles
        else np.zeros((0, 3), dtype=np.int32)
    )
    torsion_arr = (
        np.asarray(torsions, dtype=np.int32)
        if torsions
        else np.zeros((0, 4), dtype=np.int32)
    )
    return angle_arr, torsion_arr


def _lookup_dihedral_params(
    dihedral_params: dict,
    types: tuple[str, str, str, str],
) -> tuple[float, int, float] | None:
    terms = _lookup_all_dihedral_params(dihedral_params, types)
    return terms[0] if terms else None


def _lookup_all_dihedral_params(
    dihedral_params: dict,
    types: tuple[str, str, str, str],
) -> list[tuple[float, int, float]]:
    key = types
    if key not in dihedral_params or not dihedral_params[key]:
        return []
    out: list[tuple[float, int, float]] = []
    for d in dihedral_params[key]:
        out.append((float(d.k), int(d.n), float(np.radians(d.phase))))
    return out


@dataclass(frozen=True, slots=True)
class PsfConnectivity:
    """Bonded connectivity from a CHARMM PSF EXT file."""

    n_atoms: int
    atom_types: tuple[str, ...]
    charges: Array
    bonds: np.ndarray
    angles: np.ndarray
    torsions: np.ndarray
    impropers: np.ndarray


def _read_psf_section_ints(
    lines: list[str],
    start: int,
    *,
    n_per_entry: int,
) -> tuple[list[int], int]:
    """Read ``count * n_per_entry`` 1-based indices until the next ``!`` section."""
    count = int(lines[start].split()[0])
    needed = count * n_per_entry
    values: list[int] = []
    idx = start + 1
    while len(values) < needed:
        if idx >= len(lines):
            raise ValueError(f"PSF section ended before {needed} integers were read")
        stripped = lines[idx].strip()
        if stripped.startswith("!"):
            break
        values.extend(int(x) for x in stripped.split())
        idx += 1
    if len(values) != needed:
        raise ValueError(f"PSF section expected {needed} integers, found {len(values)}")
    return values, idx


def parse_psf_ext(psf_path: Path | str) -> PsfConnectivity:
    """Parse bonds/angles/dihedrals/impropers from a CHARMM PSF EXT file."""
    lines = Path(psf_path).read_text(encoding="utf-8", errors="replace").splitlines()
    natom = None
    atom_types: list[str] = []
    charges: list[float] = []
    bonds = angles = torsions = impropers = None

    i = 0
    while i < len(lines):
        line = lines[i]
        if "!NATOM" in line:
            natom = int(line.split()[0])
            for j in range(i + 1, i + 1 + natom):
                parts = lines[j].split()
                if len(parts) < 7:
                    raise ValueError(f"Malformed PSF atom line: {lines[j]!r}")
                atom_types.append(parts[5])
                charges.append(float(parts[6]))
            i = i + 1 + natom
            continue
        if "!NBOND" in line:
            raw, i = _read_psf_section_ints(lines, i, n_per_entry=2)
            bonds = np.asarray(raw, dtype=np.int32).reshape(-1, 2) - 1
            continue
        if "!NTHETA" in line:
            raw, i = _read_psf_section_ints(lines, i, n_per_entry=3)
            angles = np.asarray(raw, dtype=np.int32).reshape(-1, 3) - 1
            continue
        if "!NPHI" in line:
            raw, i = _read_psf_section_ints(lines, i, n_per_entry=4)
            torsions = np.asarray(raw, dtype=np.int32).reshape(-1, 4) - 1
            continue
        if "!NIMPHI" in line:
            raw, i = _read_psf_section_ints(lines, i, n_per_entry=4)
            impropers = np.asarray(raw, dtype=np.int32).reshape(-1, 4) - 1
            continue
        i += 1

    if natom is None:
        raise ValueError(f"No !NATOM section in {psf_path}")
    bonds = bonds if bonds is not None else np.zeros((0, 2), dtype=np.int32)
    angles = angles if angles is not None else np.zeros((0, 3), dtype=np.int32)
    torsions = torsions if torsions is not None else np.zeros((0, 4), dtype=np.int32)
    impropers = impropers if impropers is not None else np.zeros((0, 4), dtype=np.int32)
    return PsfConnectivity(
        n_atoms=natom,
        atom_types=tuple(atom_types),
        charges=jnp.asarray(charges, dtype=jnp.float64),
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        impropers=impropers,
    )


def _build_bonded_parameters(
    atom_types: Sequence[str],
    bonds: np.ndarray,
    angles: np.ndarray,
    torsions: np.ndarray,
    impropers: np.ndarray,
    bond_params: dict,
    angle_params: dict,
    dihedral_params: dict,
) -> BondedParameters:
    bond_k = np.zeros(len(bonds), dtype=np.float64)
    bond_r0 = np.zeros(len(bonds), dtype=np.float64)
    for i, (idx1, idx2) in enumerate(bonds):
        key = (atom_types[int(idx1)], atom_types[int(idx2)])
        if key in bond_params:
            bond_k[i] = bond_params[key].k
            bond_r0[i] = bond_params[key].r0

    angle_k = np.zeros(len(angles), dtype=np.float64)
    angle_theta0 = np.zeros(len(angles), dtype=np.float64)
    for i, (idx1, idx2, idx3) in enumerate(angles):
        key = (atom_types[int(idx1)], atom_types[int(idx2)], atom_types[int(idx3)])
        if key in angle_params:
            angle_k[i] = angle_params[key].k
            angle_theta0[i] = np.radians(angle_params[key].theta0)

    torsion_rows: list[list[int]] = []
    torsion_k: list[float] = []
    torsion_n: list[int] = []
    torsion_gamma: list[float] = []
    for idx1, idx2, idx3, idx4 in torsions:
        types = (
            atom_types[int(idx1)],
            atom_types[int(idx2)],
            atom_types[int(idx3)],
            atom_types[int(idx4)],
        )
        terms = _lookup_all_dihedral_params(dihedral_params, types)
        if not terms:
            torsion_rows.append([int(idx1), int(idx2), int(idx3), int(idx4)])
            torsion_k.append(0.0)
            torsion_n.append(0)
            torsion_gamma.append(0.0)
            continue
        for k, n, gamma in terms:
            torsion_rows.append([int(idx1), int(idx2), int(idx3), int(idx4)])
            torsion_k.append(k)
            torsion_n.append(n)
            torsion_gamma.append(gamma)
    torsion_idx = (
        np.asarray(torsion_rows, dtype=np.int32)
        if torsion_rows
        else np.zeros((0, 4), dtype=np.int32)
    )

    improper_rows: list[list[int]] = []
    improper_k: list[float] = []
    improper_n: list[int] = []
    improper_gamma: list[float] = []
    for idx1, idx2, idx3, idx4 in impropers:
        types = (
            atom_types[int(idx1)],
            atom_types[int(idx2)],
            atom_types[int(idx3)],
            atom_types[int(idx4)],
        )
        terms = _lookup_all_dihedral_params(dihedral_params, types)
        if not terms:
            improper_rows.append([int(idx1), int(idx2), int(idx3), int(idx4)])
            improper_k.append(0.0)
            improper_n.append(0)
            improper_gamma.append(0.0)
            continue
        for k, n, gamma in terms:
            improper_rows.append([int(idx1), int(idx2), int(idx3), int(idx4)])
            improper_k.append(k)
            improper_n.append(n)
            improper_gamma.append(gamma)
    improper_idx = (
        np.asarray(improper_rows, dtype=np.int32)
        if improper_rows
        else np.zeros((0, 4), dtype=np.int32)
    )

    return BondedParameters(
        bond_k=jnp.asarray(bond_k),
        bond_r0=jnp.asarray(bond_r0),
        angle_k=jnp.asarray(angle_k),
        angle_theta0=jnp.asarray(angle_theta0),
        torsion_k=jnp.asarray(torsion_k, dtype=jnp.float64),
        torsion_n=jnp.asarray(torsion_n, dtype=np.int32),
        torsion_gamma=jnp.asarray(torsion_gamma, dtype=jnp.float64),
        improper_k=jnp.asarray(improper_k, dtype=jnp.float64),
        improper_n=jnp.asarray(improper_n, dtype=np.int32),
        improper_gamma=jnp.asarray(improper_gamma, dtype=jnp.float64),
        cmap_maps=None,
    ), torsion_idx, improper_idx


def load_cgenff_bonded_from_psf(
    psf_path: Path | str,
    positions: Array | np.ndarray,
    *,
    prm_file: Path | str | None = None,
    molecule_id: Array | None = None,
) -> CgenffBondedSystem:
    """Load CGENFF bonded topology/parameters from a CHARMM PSF EXT file."""
    psf = parse_psf_ext(psf_path)
    prm_path = Path(prm_file) if prm_file is not None else default_cgenff_paths()[1]
    bond_params, angle_params, dihedral_params, _ = parse_prm(str(prm_path))

    bonded, torsions, impropers = _build_bonded_parameters(
        psf.atom_types,
        psf.bonds,
        psf.angles,
        psf.torsions,
        psf.impropers,
        bond_params,
        angle_params,
        dihedral_params,
    )
    if molecule_id is None:
        molecule_id = jnp.zeros(psf.n_atoms, dtype=jnp.int32)

    topology = create_topology(
        n_atoms=psf.n_atoms,
        bonds=jnp.asarray(psf.bonds),
        angles=jnp.asarray(psf.angles),
        torsions=jnp.asarray(torsions),
        impropers=jnp.asarray(impropers),
        molecule_id=jnp.asarray(molecule_id),
    )
    return CgenffBondedSystem(
        positions=jnp.asarray(positions, dtype=jnp.float64),
        topology=topology,
        bonded=bonded,
        atom_types=psf.atom_types,
        charges=psf.charges,
    )


def load_cgenff_bonded_from_charmm_files(
    pdb_file: Path | str,
    *,
    rtf_file: Path | str | None = None,
    prm_file: Path | str | None = None,
    residue_name: str | None = None,
    molecule_id: Array | None = None,
) -> CgenffBondedSystem:
    """Load CGENFF bonded topology/parameters for atoms in a PDB file.

    When ``residue_name`` is set, only that ``RESI`` block is taken from the
    bundled (or supplied) full CGENFF RTF.  Atom order must match the PDB.
    """
    pdb_path = Path(pdb_file)
    prm_path = Path(prm_file) if prm_file is not None else default_cgenff_paths()[1]

    if rtf_file is not None:
        rtf_path = Path(rtf_file)
        rtf_source = rtf_path.read_text(encoding="utf-8", errors="replace")
    elif residue_name is not None:
        rtf_source = extract_residue_rtf(default_cgenff_paths()[0], residue_name)
    else:
        rtf_path = default_cgenff_paths()[0]
        rtf_source = rtf_path.read_text(encoding="utf-8", errors="replace")

    bond_params, angle_params, dihedral_params, _ = parse_prm(str(prm_path))
    pdb_atom_names, positions = parse_pdb_simple(str(pdb_path))

    n_atoms = len(pdb_atom_names)
    atom_records = _parse_residue_atoms(rtf_source)
    if len(atom_records) != n_atoms:
        raise ValueError(
            f"Atom count mismatch: PDB has {n_atoms} atoms, RTF residue has "
            f"{len(atom_records)} atoms"
        )

    atom_types = tuple(rec[1] for rec in atom_records)
    charges = jnp.asarray([rec[2] for rec in atom_records], dtype=jnp.float64)

    rtf_bonds, rtf_impropers = _parse_residue_connectivity(rtf_source)
    bonds_list: list[list[int]] = []
    name_to_idx = {atom_records[i][0]: i for i in range(n_atoms)}
    for atom1_name, atom2_name in rtf_bonds:
        bonds_list.append([name_to_idx[atom1_name], name_to_idx[atom2_name]])
    bonds = (
        np.asarray(bonds_list, dtype=np.int32)
        if bonds_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    angles, torsions = _infer_angles_and_torsions(bonds, n_atoms)

    impropers_list: list[list[int]] = []
    for a1, a2, a3, a4 in rtf_impropers:
        impropers_list.append(
            [name_to_idx[a1], name_to_idx[a2], name_to_idx[a3], name_to_idx[a4]]
        )
    impropers = (
        np.asarray(impropers_list, dtype=np.int32)
        if impropers_list
        else np.zeros((0, 4), dtype=np.int32)
    )

    if molecule_id is None:
        molecule_id = jnp.zeros(n_atoms, dtype=jnp.int32)

    topology = create_topology(
        n_atoms=n_atoms,
        bonds=jnp.asarray(bonds),
        angles=jnp.asarray(angles),
        torsions=jnp.asarray(torsions),
        impropers=jnp.asarray(impropers),
        molecule_id=jnp.asarray(molecule_id),
    )

    bond_k = np.zeros(len(bonds), dtype=np.float64)
    bond_r0 = np.zeros(len(bonds), dtype=np.float64)
    for i, (idx1, idx2) in enumerate(bonds):
        key = (atom_types[int(idx1)], atom_types[int(idx2)])
        if key in bond_params:
            bond_k[i] = bond_params[key].k
            bond_r0[i] = bond_params[key].r0

    angle_k = np.zeros(len(angles), dtype=np.float64)
    angle_theta0 = np.zeros(len(angles), dtype=np.float64)
    for i, (idx1, idx2, idx3) in enumerate(angles):
        key = (atom_types[int(idx1)], atom_types[int(idx2)], atom_types[int(idx3)])
        if key in angle_params:
            angle_k[i] = angle_params[key].k
            angle_theta0[i] = np.radians(angle_params[key].theta0)

    torsion_k = np.zeros(len(torsions), dtype=np.float64)
    torsion_n = np.zeros(len(torsions), dtype=np.int32)
    torsion_gamma = np.zeros(len(torsions), dtype=np.float64)
    for i, (idx1, idx2, idx3, idx4) in enumerate(torsions):
        types = (
            atom_types[int(idx1)],
            atom_types[int(idx2)],
            atom_types[int(idx3)],
            atom_types[int(idx4)],
        )
        params = _lookup_dihedral_params(dihedral_params, types)
        if params is not None:
            torsion_k[i], torsion_n[i], torsion_gamma[i] = params

    improper_k = np.zeros(len(impropers), dtype=np.float64)
    improper_n = np.zeros(len(impropers), dtype=np.int32)
    improper_gamma = np.zeros(len(impropers), dtype=np.float64)
    for i, (idx1, idx2, idx3, idx4) in enumerate(impropers):
        types = (
            atom_types[int(idx1)],
            atom_types[int(idx2)],
            atom_types[int(idx3)],
            atom_types[int(idx4)],
        )
        params = _lookup_dihedral_params(dihedral_params, types)
        if params is not None:
            improper_k[i], improper_n[i], improper_gamma[i] = params

    bonded = BondedParameters(
        bond_k=jnp.asarray(bond_k),
        bond_r0=jnp.asarray(bond_r0),
        angle_k=jnp.asarray(angle_k),
        angle_theta0=jnp.asarray(angle_theta0),
        torsion_k=jnp.asarray(torsion_k),
        torsion_n=jnp.asarray(torsion_n),
        torsion_gamma=jnp.asarray(torsion_gamma),
        improper_k=jnp.asarray(improper_k),
        improper_n=jnp.asarray(improper_n),
        improper_gamma=jnp.asarray(improper_gamma),
        cmap_maps=None,
    )

    return CgenffBondedSystem(
        positions=jnp.asarray(positions, dtype=jnp.float64),
        topology=topology,
        bonded=bonded,
        atom_types=atom_types,
        charges=charges,
    )


def mm_atom_mask_from_indices(n_atoms: int, mm_atom_indices: Sequence[int]) -> Array:
    """Boolean mask: ``True`` for MM atoms, ``False`` for ML atoms."""
    mask = jnp.zeros((n_atoms,), dtype=bool)
    for idx in mm_atom_indices:
        mask = mask.at[int(idx)].set(True)
    return mask


def mm_atom_mask_complement(ml_atom_indices: Sequence[int], n_atoms: int) -> Array:
    """Return MM mask given ML atom indices (0-based)."""
    ml_mask = jnp.zeros((n_atoms,), dtype=bool)
    for idx in ml_atom_indices:
        ml_mask = ml_mask.at[int(idx)].set(True)
    return ~ml_mask


def filter_bonded_topology_for_mm(
    topology: Topology,
    bonded: BondedParameters,
    mm_mask: Array,
) -> tuple[Topology, BondedParameters]:
    """Keep bonded interactions whose atoms are all MM (embedding boundary safe)."""
    mm_np = np.asarray(mm_mask, dtype=bool)

    def _keep_rows(indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.all(mm_np[indices], axis=1)

    bond_keep = _keep_rows(np.asarray(topology.bonds))
    angle_keep = _keep_rows(np.asarray(topology.angles))
    torsion_keep = _keep_rows(np.asarray(topology.torsions))
    improper_keep = _keep_rows(np.asarray(topology.impropers))

    def _slice_params(arr: Array | None, keep: np.ndarray) -> Array | None:
        if arr is None:
            return None
        return jnp.asarray(arr)[keep]

    filtered_topology = Topology(
        n_atoms=topology.n_atoms,
        bonds=jnp.asarray(topology.bonds)[bond_keep],
        angles=jnp.asarray(topology.angles)[angle_keep],
        torsions=jnp.asarray(topology.torsions)[torsion_keep],
        impropers=jnp.asarray(topology.impropers)[improper_keep],
        exclusion_mask=topology.exclusion_mask,
        pair_14_mask=topology.pair_14_mask,
        molecule_id=topology.molecule_id,
        cmap_atoms=topology.cmap_atoms,
        cmap_map_idx=topology.cmap_map_idx,
        exc_pairs=topology.exc_pairs,
        nbfix_atom_type=topology.nbfix_atom_type,
    )

    filtered_bonded = BondedParameters(
        bond_k=_slice_params(bonded.bond_k, bond_keep),
        bond_r0=_slice_params(bonded.bond_r0, bond_keep),
        angle_k=_slice_params(bonded.angle_k, angle_keep),
        angle_theta0=_slice_params(bonded.angle_theta0, angle_keep),
        torsion_k=_slice_params(bonded.torsion_k, torsion_keep),
        torsion_n=_slice_params(bonded.torsion_n, torsion_keep),
        torsion_gamma=_slice_params(bonded.torsion_gamma, torsion_keep),
        improper_k=_slice_params(bonded.improper_k, improper_keep),
        improper_n=_slice_params(bonded.improper_n, improper_keep),
        improper_gamma=_slice_params(bonded.improper_gamma, improper_keep),
        cmap_maps=bonded.cmap_maps,
    )
    return filtered_topology, filtered_bonded


def concat_cgenff_systems(systems: Iterable[CgenffBondedSystem]) -> CgenffBondedSystem:
    """Concatenate multiple non-interacting molecules into one system."""
    systems = list(systems)
    if not systems:
        raise ValueError("concat_cgenff_systems requires at least one system")

    offset = 0
    all_positions: list[Array] = []
    all_bonds: list[np.ndarray] = []
    all_angles: list[np.ndarray] = []
    all_torsions: list[np.ndarray] = []
    all_impropers: list[np.ndarray] = []
    all_mol_id: list[Array] = []
    bond_ks: list[Array] = []
    bond_r0s: list[Array] = []
    angle_ks: list[Array] = []
    angle_theta0s: list[Array] = []
    torsion_ks: list[Array] = []
    torsion_ns: list[Array] = []
    torsion_gammas: list[Array] = []
    improper_ks: list[Array] = []
    improper_ns: list[Array] = []
    improper_gammas: list[Array] = []
    atom_types: list[str] = []
    charges: list[Array] = []

    for mol_idx, system in enumerate(systems):
        n = system.n_atoms
        all_positions.append(system.positions)
        atom_types.extend(system.atom_types)
        charges.append(system.charges)

        def _shift(arr: Array) -> np.ndarray:
            if arr is None or arr.shape[0] == 0:
                return np.zeros((0, arr.shape[-1]), dtype=np.int32)
            shifted = np.asarray(arr, dtype=np.int32) + offset
            return shifted

        all_bonds.append(_shift(system.topology.bonds))
        all_angles.append(_shift(system.topology.angles))
        all_torsions.append(_shift(system.topology.torsions))
        all_impropers.append(_shift(system.topology.impropers))
        all_mol_id.append(jnp.full((n,), mol_idx, dtype=jnp.int32))

        bond_ks.append(system.bonded.bond_k)
        bond_r0s.append(system.bonded.bond_r0)
        angle_ks.append(system.bonded.angle_k)
        angle_theta0s.append(system.bonded.angle_theta0)
        torsion_ks.append(system.bonded.torsion_k)
        torsion_ns.append(system.bonded.torsion_n)
        torsion_gammas.append(system.bonded.torsion_gamma)
        improper_ks.append(system.bonded.improper_k)
        improper_ns.append(system.bonded.improper_n)
        improper_gammas.append(system.bonded.improper_gamma)
        offset += n

    n_atoms = offset
    topology = create_topology(
        n_atoms=n_atoms,
        bonds=jnp.asarray(np.concatenate(all_bonds, axis=0) if all_bonds else np.zeros((0, 2))),
        angles=jnp.asarray(np.concatenate(all_angles, axis=0) if all_angles else np.zeros((0, 3))),
        torsions=jnp.asarray(
            np.concatenate(all_torsions, axis=0) if all_torsions else np.zeros((0, 4))
        ),
        impropers=jnp.asarray(
            np.concatenate(all_impropers, axis=0) if all_impropers else np.zeros((0, 4))
        ),
        molecule_id=jnp.concatenate(all_mol_id),
    )
    bonded = BondedParameters(
        bond_k=jnp.concatenate(bond_ks),
        bond_r0=jnp.concatenate(bond_r0s),
        angle_k=jnp.concatenate(angle_ks),
        angle_theta0=jnp.concatenate(angle_theta0s),
        torsion_k=jnp.concatenate(torsion_ks),
        torsion_n=jnp.concatenate(torsion_ns),
        torsion_gamma=jnp.concatenate(torsion_gammas),
        improper_k=jnp.concatenate(improper_ks),
        improper_n=jnp.concatenate(improper_ns),
        improper_gamma=jnp.concatenate(improper_gammas),
        cmap_maps=None,
    )
    return CgenffBondedSystem(
        positions=jnp.concatenate(all_positions, axis=0),
        topology=topology,
        bonded=bonded,
        atom_types=tuple(atom_types),
        charges=jnp.concatenate(charges),
    )
