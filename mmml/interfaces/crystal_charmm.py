"""Build CHARMM-ready crystal PDBs from literature CIFs + ``make-res`` monomer templates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from mmml.interfaces.pyxtal_placement import (
    _match_atoms_to_template_names,
    _match_molecule_blocks_to_psf_order,
    write_psf_order_mapping_pdb,
)
from mmml.paths import (
    default_benzene_crystal_cif,
    default_dcm_crystal_cif,
)

if TYPE_CHECKING:
    from mmml.interfaces.crystal_reference import CrystalMetrics

MonomerTemplate = tuple[np.ndarray, list[str], np.ndarray]

LITERATURE_CRYSTAL_PRESETS: dict[str, dict[str, Any]] = {
    "dcm": {
        "residue": "DCM",
        "cif": default_dcm_crystal_cif,
        "space_group": 60,
        "monomer_pdb": lambda: default_make_res_monomer_pdb("DCM"),
    },
    "benz": {
        "residue": "BENZ",
        "cif": default_benzene_crystal_cif,
        "space_group": 14,
        "monomer_pdb": lambda: default_make_res_monomer_pdb("BENZ"),
    },
}


def default_make_res_monomer_pdb(residue: str) -> Path:
    """Bundled CHARMM monomer PDB (``make-res``-style atom names)."""
    from mmml.paths import bundled_file

    key = residue.strip().upper()
    name = {"DCM": "dcm", "BENZ": "benz"}.get(key, key.lower())
    return bundled_file("data", "molecules", f"{name}_monomer.pdb")


def resolve_make_res_monomer_pdb(
    residue: str,
    *,
    monomer_pdb: Path | str | None = None,
) -> Path:
    """Prefer explicit path, then cwd ``make-res`` outputs, then bundled template."""
    if monomer_pdb is not None:
        path = Path(monomer_pdb).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"monomer PDB not found: {path}")
        return path
    res = residue.strip().upper()
    for candidate in (
        Path(f"pdb/{res.lower()}.pdb"),
        Path("pdb/initial.pdb"),
        default_make_res_monomer_pdb(res),
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No monomer PDB for residue {res!r}. Run "
        f"'mmml make-res --res {res} --skip-energy-show' or pass --monomer-pdb."
    )


def _parse_pdb_atom_records(pdb_path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    from ase.data import atomic_numbers

    positions: list[list[float]] = []
    names: list[str] = []
    symbols: list[str] = []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        elem = (line[76:78].strip() or name[:1]).strip().title()
        if elem == "Cl":
            elem = "Cl"
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        positions.append([x, y, z])
        names.append(name)
        symbols.append(elem)
    if not positions:
        raise ValueError(f"No ATOM/HETATM records in {pdb_path}")
    z = np.array([atomic_numbers[s] for s in symbols], dtype=int)
    return np.asarray(positions, dtype=float), names, z


def load_monomer_template(pdb_path: Path | str) -> MonomerTemplate:
    """Positions, CHARMM atom names, and atomic numbers from a ``make-res`` PDB."""
    return _parse_pdb_atom_records(Path(pdb_path))


def split_crystal_molecules(
    atoms: Any,
    atoms_per_molecule: int,
    *,
    cutoff_mult: float = 1.08,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split periodic ``ase.Atoms`` into connected molecules (MIC-aware)."""
    from ase.neighborlist import natural_cutoffs, neighbor_list
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n_total = len(atoms)
    n_per = int(atoms_per_molecule)
    if n_per <= 0:
        raise ValueError(f"atoms_per_molecule must be positive, got {n_per}")
    if n_total % n_per != 0:
        raise ValueError(
            f"crystal atom count {n_total} is not divisible by "
            f"monomer size {n_per}"
        )
    cutoffs = natural_cutoffs(atoms, mult=float(cutoff_mult))
    i, j = neighbor_list(
        "ij",
        atoms,
        cutoffs,
        self_interaction=False,
    )
    n = n_total
    data = np.ones(len(i), dtype=int)
    graph = csr_matrix((data, (i, j)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    expected_mols = n_total // n_per
    if int(n_comp) != expected_mols:
        raise RuntimeError(
            f"connectivity split found {n_comp} molecules, expected {expected_mols} "
            f"from {n_total} atoms / {n_per} per monomer"
        )
    positions = np.asarray(atoms.get_positions(), dtype=float)
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for comp in range(int(n_comp)):
        idx = np.where(labels == comp)[0]
        if int(idx.shape[0]) != n_per:
            raise RuntimeError(
                f"molecule component {comp} has {idx.shape[0]} atoms, expected {n_per}"
            )
        order = np.argsort(idx)
        idx = idx[order]
        blocks.append((positions[idx], z[idx]))
    return blocks


def _format_cryst1(atoms: Any) -> str:
    par = atoms.cell.cellpar()
    a, b, c, alpha, beta, gamma = (float(x) for x in par)
    return (
        f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
        f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
    )


def write_charmm_crystal_pdb(
    pdb_path: Path | str,
    *,
    molecule_blocks: Sequence[tuple[np.ndarray, np.ndarray]],
    ordered_residue_names: Sequence[str],
    residue_geometries: dict[str, MonomerTemplate],
    cell_atoms: Any,
) -> Path:
    """Write a crystal PDB with literature cell and CHARMM atom names."""
    out = Path(pdb_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.pdb")
    write_psf_order_mapping_pdb(
        tmp,
        list(molecule_blocks),
        list(ordered_residue_names),
        residue_geometries,
    )
    lines = tmp.read_text(encoding="utf-8").splitlines()
    tmp.unlink(missing_ok=True)
    cryst1 = _format_cryst1(cell_atoms)
    rebuilt: list[str] = []
    for line in lines:
        if line.startswith("CRYST1"):
            rebuilt.append(cryst1)
        elif line.startswith("REMARK"):
            rebuilt.append(
                "REMARK   mmml literature CIF mapped to CHARMM make-res atom names"
            )
        else:
            rebuilt.append(line)
    out.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")
    return out


def build_charmm_crystal_from_cif(
    cif_path: Path | str,
    residue: str,
    *,
    monomer_pdb: Path | str | None = None,
    pdb_out: Path | str | None = None,
) -> tuple[Any, Path, MonomerTemplate]:
    """Map a literature CIF onto CHARMM ``make-res`` atom names.

    Returns ``(ase.Atoms, pdb_path, monomer_template)``.
    """
    from ase import Atoms
    from ase.io import read

    res_key = residue.strip().upper()
    cif = Path(cif_path).expanduser().resolve()
    monomer_path = resolve_make_res_monomer_pdb(res_key, monomer_pdb=monomer_pdb)
    template = load_monomer_template(monomer_path)
    tmpl_pos, tmpl_names, tmpl_z = template

    cell_atoms = read(str(cif))
    if not cell_atoms.pbc.any():
        cell_atoms.pbc = True

    blocks = split_crystal_molecules(cell_atoms, int(tmpl_z.shape[0]))
    ordered_names = [res_key] * len(blocks)
    residue_geometries = {res_key: template}
    ordered_blocks = _match_molecule_blocks_to_psf_order(
        blocks,
        ordered_names,
        residue_geometries,
    )

    pdb_path = Path(pdb_out) if pdb_out is not None else Path(f"pdb/{res_key.lower()}_crystal.pdb")
    write_charmm_crystal_pdb(
        pdb_path,
        molecule_blocks=ordered_blocks,
        ordered_residue_names=ordered_names,
        residue_geometries=residue_geometries,
        cell_atoms=cell_atoms,
    )

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for (pos, z_block), _res in zip(ordered_blocks, ordered_names):
        names = _match_atoms_to_template_names(
            pos,
            z_block,
            tmpl_pos,
            tmpl_names,
            tmpl_z,
        )
        for zi, xyz, _nm in zip(z_block, pos, names):
            from ase.data import chemical_symbols

            symbols.append(chemical_symbols[int(zi)])
            positions.append(np.asarray(xyz, dtype=float))

    out_atoms = Atoms(
        symbols=symbols,
        positions=np.vstack(positions),
        cell=cell_atoms.cell,
        pbc=cell_atoms.pbc,
    )
    return out_atoms, pdb_path.resolve(), template


def build_literature_charmm_crystal(
    preset: str,
    *,
    monomer_pdb: Path | str | None = None,
    pdb_out: Path | str | None = None,
) -> tuple[Any, Path, str]:
    """Shortcut for bundled ``dcm`` / ``benz`` literature presets."""
    key = preset.strip().lower()
    if key not in LITERATURE_CRYSTAL_PRESETS:
        raise ValueError(
            f"unknown literature preset {preset!r}; choose from {sorted(LITERATURE_CRYSTAL_PRESETS)}"
        )
    spec = LITERATURE_CRYSTAL_PRESETS[key]
    residue = str(spec["residue"])
    cif = Path(spec["cif"]())
    atoms, pdb_path, _ = build_charmm_crystal_from_cif(
        cif,
        residue,
        monomer_pdb=monomer_pdb,
        pdb_out=pdb_out,
    )
    return atoms, pdb_path, residue


def charmm_crystal_metrics_from_preset(preset: str) -> "CrystalMetrics":
    """Unit-cell metrics for a literature preset after CHARMM mapping."""
    from mmml.interfaces.crystal_reference import CrystalMetrics, metrics_from_atoms

    atoms, _, _ = build_literature_charmm_crystal(preset)
    sg = int(LITERATURE_CRYSTAL_PRESETS[preset.strip().lower()]["space_group"])
    return metrics_from_atoms(atoms, label="make-res+CIF", space_group=sg)
