"""Build CHARMM-ready crystal supercells from literature CIFs + ``make-res`` monomers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from mmml.interfaces.pyxtal_placement import (
    _match_atoms_to_template_names,
    _match_molecule_blocks_to_psf_order,
    ase_supercell,
    crystal_mass_density_g_cm3,
    parse_supercell_reps,
    write_psf_order_mapping_pdb,
)
from mmml.paths import (
    default_benzene_crystal_cif,
    default_dcm_crystal_cif,
)

if TYPE_CHECKING:
    from mmml.interfaces.crystal_reference import CrystalMetrics

MonomerTemplate = tuple[np.ndarray, list[str], np.ndarray]

# Default minimum box edge (Å) for CHARMM cutnb≈14 — each supercell edge should exceed ~2×cutnb.
DEFAULT_MIN_BOX_SIDE_A = 28.0

LITERATURE_CRYSTAL_PRESETS: dict[str, dict[str, Any]] = {
    "dcm": {
        "residue": "DCM",
        "cif": default_dcm_crystal_cif,
        "space_group": 60,
        "reference_density_g_cm3": 1.972,
    },
    "benz": {
        "residue": "BENZ",
        "cif": default_benzene_crystal_cif,
        "space_group": 14,
        "reference_density_g_cm3": 1.202,
    },
}


@dataclass(frozen=True)
class CharmmLiteratureCrystalResult:
    """Literature unit cell mapped to CHARMM names, tiled for MD."""

    atoms: Any
    pdb_path: Path
    residue: str
    supercell_reps: tuple[int, int, int]
    n_molecules: int
    density_g_cm3: float
    cell_lengths_a: tuple[float, float, float]
    cell_angles_deg: tuple[float, float, float]
    monomer_pdb: Path


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


def suggest_supercell_reps(
    cell_lengths_a: Sequence[float],
    *,
    min_box_side_a: float = DEFAULT_MIN_BOX_SIDE_A,
) -> tuple[int, int, int]:
    """Minimum integer repeats so each cell edge × repeat ≥ *min_box_side_a*."""
    target = float(min_box_side_a)
    if target <= 0.0:
        raise ValueError(f"min_box_side_a must be positive, got {target}")
    reps: list[int] = []
    for length in cell_lengths_a:
        L = float(length)
        if L <= 0.0:
            raise ValueError(f"cell length must be positive, got {L}")
        reps.append(max(1, int(np.ceil(target / L))))
    return (reps[0], reps[1], reps[2])


def _parse_pdb_atom_records(pdb_path: Path) -> MonomerTemplate:
    from ase.data import atomic_numbers

    positions: list[list[float]] = []
    names: list[str] = []
    symbols: list[str] = []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        elem = line[76:78].strip() or line[12:14].strip()[:1]
        elem = elem.title()
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
    """Split periodic ``ase.Atoms`` into connected molecules."""
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
    i, j = neighbor_list("ij", atoms, cutoffs, self_interaction=False)
    graph = csr_matrix((np.ones(len(i), dtype=int), (i, j)), shape=(n_total, n_total))
    n_comp, labels = connected_components(graph, directed=False)
    expected_mols = n_total // n_per
    if int(n_comp) != expected_mols:
        raise RuntimeError(
            f"connectivity split found {n_comp} molecules, expected {expected_mols}"
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
        blocks.append((positions[idx], z[idx]))
    return blocks


def _format_cryst1(atoms: Any) -> str:
    a, b, c, alpha, beta, gamma = (float(x) for x in atoms.cell.cellpar())
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
    """Write a periodic PDB with literature/supercell CRYST1 and CHARMM atom names."""
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
                "REMARK   mmml literature CIF + make-res atom names (simulation supercell)"
            )
        else:
            rebuilt.append(line)
    out.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")
    return out


def _atoms_from_mapped_blocks(
    ordered_blocks: Sequence[tuple[np.ndarray, np.ndarray]],
    ordered_names: Sequence[str],
    template: MonomerTemplate,
    cell_atoms: Any,
) -> Any:
    from ase import Atoms
    from ase.data import chemical_symbols

    tmpl_pos, tmpl_names, tmpl_z = template
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for (pos, z_block), _res in zip(ordered_blocks, ordered_names):
        _match_atoms_to_template_names(pos, z_block, tmpl_pos, tmpl_names, tmpl_z)
        for zi, xyz in zip(z_block, pos):
            symbols.append(chemical_symbols[int(zi)])
            positions.append(np.asarray(xyz, dtype=float))
    return Atoms(
        symbols=symbols,
        positions=np.vstack(positions),
        cell=cell_atoms.cell,
        pbc=cell_atoms.pbc,
    )


def map_cif_to_charmm_blocks(
    cif_path: Path | str,
    residue: str,
    *,
    monomer_pdb: Path | str | None = None,
) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]], list[str], MonomerTemplate, Path]:
    """Read literature CIF and map molecules onto ``make-res`` atom-name templates."""
    from ase.io import read

    res_key = residue.strip().upper()
    cif = Path(cif_path).expanduser().resolve()
    monomer_path = resolve_make_res_monomer_pdb(res_key, monomer_pdb=monomer_pdb)
    template = load_monomer_template(monomer_path)
    _, _, tmpl_z = template

    cell_atoms = read(str(cif))
    if not cell_atoms.pbc.any():
        cell_atoms.pbc = True

    blocks = split_crystal_molecules(cell_atoms, int(tmpl_z.shape[0]))
    ordered_names = [res_key] * len(blocks)
    residue_geometries = {res_key: template}
    ordered_blocks = _match_molecule_blocks_to_psf_order(
        blocks, ordered_names, residue_geometries
    )
    atoms = _atoms_from_mapped_blocks(ordered_blocks, ordered_names, template, cell_atoms)
    return atoms, ordered_blocks, ordered_names, template, monomer_path


def build_charmm_literature_supercell(
    *,
    residue: str,
    cif_path: Path | str,
    supercell_reps: tuple[int, int, int] | None = None,
    min_box_side_a: float | None = DEFAULT_MIN_BOX_SIDE_A,
    monomer_pdb: Path | str | None = None,
    pdb_out: Path | str | None = None,
    target_density_g_cm3: float | None = None,
) -> CharmmLiteratureCrystalResult:
    """Tile a literature CIF (CHARMM atom names) for MD with an appropriate supercell.

    The unit-cell geometry and density come from the CIF. *supercell_reps* tiles
    the cell for simulation box size; when omitted, repeats are chosen from
    *min_box_side_a* (default 28 Å ≈ 2× typical CHARMM ``cutnb``).
    """
    from mmml.interfaces.pyxtal_placement import scale_atoms_cell_to_density

    res_key = residue.strip().upper()
    unit_atoms, _, ordered_names, template, monomer_path = map_cif_to_charmm_blocks(
        cif_path, res_key, monomer_pdb=monomer_pdb
    )
    unit_par = tuple(float(x) for x in unit_atoms.cell.cellpar())
    unit_lengths = unit_par[:3]

    if supercell_reps is None:
        if min_box_side_a is None:
            reps = (1, 1, 1)
        else:
            reps = suggest_supercell_reps(unit_lengths, min_box_side_a=float(min_box_side_a))
    else:
        reps = tuple(int(r) for r in supercell_reps)
        if any(r <= 0 for r in reps):
            raise ValueError(f"supercell repetitions must be positive: {reps}")

    if reps != (1, 1, 1):
        sim_atoms = ase_supercell(unit_atoms, reps)
    else:
        sim_atoms = unit_atoms.copy()

    if target_density_g_cm3 is not None:
        scale_atoms_cell_to_density(sim_atoms, float(target_density_g_cm3))

    n_per = int(template[2].shape[0])
    sim_blocks = split_crystal_molecules(sim_atoms, n_per)
    sim_ordered = [res_key] * len(sim_blocks)
    residue_geometries = {res_key: template}
    sim_blocks = _match_molecule_blocks_to_psf_order(
        sim_blocks, sim_ordered, residue_geometries
    )

    pdb_path = (
        Path(pdb_out)
        if pdb_out is not None
        else Path(f"pdb/{res_key.lower()}_crystal_{reps[0]}x{reps[1]}x{reps[2]}.pdb")
    )
    write_charmm_crystal_pdb(
        pdb_path,
        molecule_blocks=sim_blocks,
        ordered_residue_names=sim_ordered,
        residue_geometries=residue_geometries,
        cell_atoms=sim_atoms,
    )

    par = tuple(float(x) for x in sim_atoms.cell.cellpar())
    n_mol_unit = len(ordered_names)
    n_molecules = n_mol_unit * int(np.prod(reps))

    return CharmmLiteratureCrystalResult(
        atoms=sim_atoms,
        pdb_path=pdb_path.resolve(),
        residue=res_key,
        supercell_reps=reps,
        n_molecules=n_molecules,
        density_g_cm3=float(crystal_mass_density_g_cm3(sim_atoms)),
        cell_lengths_a=(par[0], par[1], par[2]),
        cell_angles_deg=(par[3], par[4], par[5]),
        monomer_pdb=monomer_path,
    )


def build_literature_charmm_supercell(
    preset: str,
    *,
    supercell_reps: tuple[int, int, int] | None = None,
    min_box_side_a: float | None = DEFAULT_MIN_BOX_SIDE_A,
    monomer_pdb: Path | str | None = None,
    pdb_out: Path | str | None = None,
    target_density_g_cm3: float | None = None,
) -> CharmmLiteratureCrystalResult:
    """Shortcut for bundled ``dcm`` / ``benz`` literature presets."""
    key = preset.strip().lower()
    if key not in LITERATURE_CRYSTAL_PRESETS:
        raise ValueError(
            f"unknown literature preset {preset!r}; choose from {sorted(LITERATURE_CRYSTAL_PRESETS)}"
        )
    spec = LITERATURE_CRYSTAL_PRESETS[key]
    return build_charmm_literature_supercell(
        residue=str(spec["residue"]),
        cif_path=Path(spec["cif"]()),
        supercell_reps=supercell_reps,
        min_box_side_a=min_box_side_a,
        monomer_pdb=monomer_pdb,
        pdb_out=pdb_out,
        target_density_g_cm3=target_density_g_cm3,
    )


def charmm_crystal_metrics_from_preset(
    preset: str,
    *,
    supercell_reps: tuple[int, int, int] = (1, 1, 1),
) -> "CrystalMetrics":
    """Unit-cell metrics for literature preset after CHARMM mapping."""
    from mmml.interfaces.crystal_reference import metrics_from_atoms

    result = build_literature_charmm_supercell(
        preset,
        supercell_reps=supercell_reps,
        min_box_side_a=None,
    )
    sg = int(LITERATURE_CRYSTAL_PRESETS[preset.strip().lower()]["space_group"])
    return metrics_from_atoms(result.atoms, label="make-res+CIF", space_group=sg)
