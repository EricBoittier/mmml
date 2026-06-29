"""PyXtal-backed symmetry-aware structure building (optional ``mmml[chem]``).

PyXtal generates crystal structures with space-group symmetry. This module
exports ASE :class:`ase.Atoms` with periodic boundary conditions for downstream
MMML workflows (ASE/JAX-MD optimization, NPZ handoff, PyCHARMM setup).

Install PyXtal with::

    uv sync --extra chem
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PathLike = str | Path


def have_pyxtal() -> bool:
    """Return True when the optional ``pyxtal`` package is importable."""
    try:
        import pyxtal  # noqa: F401

        return True
    except ImportError:
        return False


def require_pyxtal() -> None:
    """Raise ``ImportError`` with install instructions when PyXtal is missing."""
    if not have_pyxtal():
        raise ImportError(
            "PyXtal is not installed. Install with: uv sync --extra chem  "
            "(or pip install 'mmml[chem]')."
        )


@dataclass(frozen=True)
class MolecularCrystalBuildRequest:
    """Inputs for a random molecular crystal build via PyXtal."""

    molecules: list[str]
    stoichiometry: list[int]
    dimension: int = 3
    space_group: int = 14
    factor: float = 1.0
    seed: int | None = None
    molecular: bool = True
    max_attempts: int = 10
    resort: bool = True
    center_only: bool = False

    def __post_init__(self) -> None:
        if len(self.molecules) != len(self.stoichiometry):
            raise ValueError(
                "molecules and stoichiometry must have the same length "
                f"({len(self.molecules)} vs {len(self.stoichiometry)})"
            )
        if not self.molecules:
            raise ValueError("at least one molecule specification is required")
        if any(int(z) <= 0 for z in self.stoichiometry):
            raise ValueError(f"stoichiometry entries must be positive: {self.stoichiometry}")
        if int(self.dimension) not in (0, 1, 2, 3):
            raise ValueError(f"dimension must be 0–3, got {self.dimension}")
        if int(self.space_group) <= 0:
            raise ValueError(f"space_group must be positive, got {self.space_group}")


@dataclass
class MolecularCrystalBuildResult:
    """PyXtal crystal plus exported ASE structure."""

    crystal: Any
    atoms: Any
    attempts: int = 1
    space_group: int = 0
    formula: str = ""


def _import_pyxtal():
    require_pyxtal()
    from pyxtal import pyxtal

    return pyxtal


def load_pyxtal_molecule(spec: str):
    """Load a PyXtal molecule from SMILES, formula, or a structure file path."""
    require_pyxtal()
    from pyxtal.molecule import pyxtal_molecule

    return pyxtal_molecule(spec)


def build_molecular_crystal_random(
    request: MolecularCrystalBuildRequest,
) -> MolecularCrystalBuildResult:
    """Generate a random molecular crystal; retry until PyXtal succeeds."""
    pyxtal_cls = _import_pyxtal()
    rng = np.random.default_rng(request.seed)
    last_error: Exception | None = None

    for attempt in range(1, max(1, int(request.max_attempts)) + 1):
        crystal = pyxtal_cls(molecular=bool(request.molecular))
        try:
            trial_seed = None if request.seed is None else int(rng.integers(0, 2**31 - 1))
            crystal.from_random(
                int(request.dimension),
                int(request.space_group),
                list(request.molecules),
                [int(z) for z in request.stoichiometry],
                float(request.factor),
                trial_seed,
            )
        except Exception as exc:
            last_error = exc
            continue

        atoms = pyxtal_to_ase(
            crystal,
            resort=bool(request.resort),
            center_only=bool(request.center_only),
        )
        sg = int(getattr(getattr(crystal, "group", None), "number", request.space_group))
        formula = str(getattr(crystal, "formula", "") or "")
        return MolecularCrystalBuildResult(
            crystal=crystal,
            atoms=atoms,
            attempts=attempt,
            space_group=sg,
            formula=formula,
        )

    raise RuntimeError(
        f"PyXtal failed to build a molecular crystal after "
        f"{max(1, int(request.max_attempts))} attempt(s) "
        f"(dim={request.dimension}, spg={request.space_group}, "
        f"molecules={request.molecules}, Z={request.stoichiometry})"
    ) from last_error


def pyxtal_to_ase(
    crystal: Any,
    *,
    resort: bool = True,
    center_only: bool = False,
) -> Any:
    """Convert a PyXtal object to ``ase.Atoms`` (preserves cell and PBC)."""
    if not getattr(crystal, "valid", True):
        raise ValueError("PyXtal crystal is not valid; cannot export to ASE")
    if not hasattr(crystal, "to_ase"):
        raise TypeError(f"Expected PyXtal crystal with to_ase(); got {type(crystal)}")
    return crystal.to_ase(resort=bool(resort), center_only=bool(center_only))


def ase_supercell(
    atoms: Any,
    reps: Sequence[int],
) -> Any:
    """Build an ASE supercell along ``reps`` (e.g. ``(2, 2, 2)``)."""
    from ase.build import make_supercell

    if len(reps) != 3:
        raise ValueError(f"reps must have length 3, got {reps}")
    matrix = np.diag([int(r) for r in reps])
    return make_supercell(atoms, matrix)


def write_ase_structure(
    atoms: Any,
    path: PathLike,
    *,
    format: str | None = None,
) -> Path:
    """Write ``atoms`` using ASE I/O."""
    from ase.io import write

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), atoms, format=format)
    return out


def atoms_to_reference_npz(
    atoms: Any,
    path: PathLike,
    *,
    label: str = "pyxtal",
) -> Path:
    """Write a minimal NPZ for ``md_handoff`` / ``build_cluster_from_reference_npz``."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    payload: dict[str, Any] = {
        "R": positions,
        "Z": numbers,
        "source": np.array(label),
    }
    if atoms.pbc.any():
        payload["cell"] = np.asarray(atoms.get_cell(), dtype=np.float64)
        payload["pbc"] = np.asarray(atoms.pbc, dtype=bool)
    np.savez_compressed(out, **payload)
    return out


def parse_supercell_reps(text: str) -> tuple[int, int, int]:
    """Parse ``2,2,2`` or ``2x2x2`` into a supercell repetition tuple."""
    raw = str(text).strip().lower().replace("x", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"supercell must have three integers (e.g. 2,2,2 or 2x2x2), got {text!r}"
        )
    reps = tuple(int(p) for p in parts)
    if any(r <= 0 for r in reps):
        raise ValueError(f"supercell repetitions must be positive: {reps}")
    return reps  # type: ignore[return-value]


def parse_stoichiometry(
    molecules: Sequence[str],
    stoichiometry: Sequence[int] | None,
    z_values: Sequence[int] | None,
) -> list[int]:
    """Resolve per-molecule Z from ``--stoichiometry`` or repeated ``--z``."""
    n = len(molecules)
    if stoichiometry is not None:
        vals = [int(z) for z in stoichiometry]
        if len(vals) != n:
            raise ValueError(
                f"--stoichiometry length {len(vals)} must match molecule count {n}"
            )
        return vals
    if z_values is not None:
        vals = [int(z) for z in z_values]
        if len(vals) == 1 and n > 1:
            return vals * n
        if len(vals) != n:
            raise ValueError(f"--z length {len(vals)} must match molecule count {n}")
        return vals
    return [2] * n


def resolve_pyxtal_use(
    *,
    composition: str | None,
    pyxtal: bool | None = None,
    builder: str | None = None,
) -> bool:
    """Return True when the crystal builder is selected for ``--composition``."""
    if builder is not None:
        return str(builder).strip().lower() == "crystal" and composition is not None
    if pyxtal is False:
        return False
    return bool(pyxtal) and composition is not None


def unique_residue_species(
    composition: Sequence[tuple[str, int]],
) -> list[str]:
    """Unique residue names in first-seen order."""
    seen: list[str] = []
    for residue, _count in composition:
        key = str(residue).upper()
        if key not in seen:
            seen.append(key)
    return seen


def resolve_pyxtal_unit_stoichiometry(
    composition: Sequence[tuple[str, int]],
    pyxtal_stoichiometry: Sequence[int] | None = None,
) -> list[int]:
    """Per-species Z in the PyXtal asymmetric unit (default 2 for each unique residue)."""
    species = unique_residue_species(composition)
    if pyxtal_stoichiometry is not None:
        vals = [int(z) for z in pyxtal_stoichiometry]
        if len(vals) == 1 and len(species) > 1:
            vals = vals * len(species)
        if len(vals) != len(species):
            raise ValueError(
                f"--pyxtal-stoichiometry length {len(vals)} must match "
                f"unique species count {len(species)} ({species})"
            )
        if any(z <= 0 for z in vals):
            raise ValueError(f"--pyxtal-stoichiometry entries must be positive: {vals}")
        return vals
    return [2] * len(species)


def validate_pyxtal_cluster_args(
    *,
    composition: str | None,
    pyxtal: bool | None = None,
    packmol: bool | None = None,
    builder: str | None = None,
) -> None:
    """Raise ``ValueError`` when PyXtal placement flags are inconsistent."""
    if not resolve_pyxtal_use(composition=composition, pyxtal=pyxtal, builder=builder):
        return
    if not composition:
        raise ValueError("PyXtal placement requires --composition (e.g. MEOH:8).")
    if packmol is True:
        raise ValueError(
            "Cannot combine --pyxtal with explicit --packmol. "
            "Use --pyxtal --no-packmol --composition RES:N."
        )


def add_pyxtal_cluster_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags for PyXtal symmetry-aware cluster placement in ``md-system``."""
    group = parser.add_argument_group("PyXtal crystal placement (requires mmml[chem])")
    group.add_argument(
        "--pyxtal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Build --composition with PyXtal (space-group crystal) instead of Packmol/grid. "
            "Requires uv sync --extra chem."
        ),
    )
    group.add_argument(
        "--pyxtal-spg",
        type=int,
        default=14,
        help="International space-group number for PyXtal from_random (default: 14).",
    )
    group.add_argument(
        "--pyxtal-dim",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="PyXtal crystal dimensionality (default: 3).",
    )
    group.add_argument(
        "--pyxtal-factor",
        type=float,
        default=1.0,
        help="PyXtal volume factor passed to from_random (default: 1.0).",
    )
    group.add_argument(
        "--pyxtal-stoichiometry",
        type=int,
        nargs="+",
        default=None,
        metavar="Z",
        help=(
            "Formula units per unique species in the PyXtal unit cell "
            "(default: 2 for each; one value repeats for all species)."
        ),
    )
    group.add_argument(
        "--pyxtal-supercell",
        type=str,
        default=None,
        metavar="NX,NY,NZ",
        help="Supercell expansion after PyXtal build (e.g. 2,2,2). Default: 1,1,1.",
    )
    group.add_argument(
        "--pyxtal-attempts",
        type=int,
        default=20,
        help="Maximum PyXtal from_random retries (default: 20).",
    )
    group.add_argument(
        "--pyxtal-trim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the PyXtal supercell has more molecules than --composition, "
            "keep the first N and warn (default: on)."
        ),
    )
    group.add_argument(
        "--optimize-pyxtal",
        action="store_true",
        help=(
            "Optional ASE pre-relax of the PyXtal structure before CHARMM MM "
            "cluster minimize."
        ),
    )
    group.add_argument(
        "--optimize-pyxtal-emt",
        action="store_true",
        help="Use ASE EMT for --optimize-pyxtal (smoke tests only).",
    )


def _atomic_number_multiset(atomic_numbers: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(z) for z in atomic_numbers))


def _split_ase_atoms_into_molecules(
    atoms: Any,
    atoms_per_list: Sequence[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split ``ase.Atoms`` into per-molecule (positions, Z) blocks."""
    positions = np.asarray(atoms.get_positions(), dtype=float)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    expected = int(np.sum(atoms_per_list))
    if int(positions.shape[0]) != expected:
        raise RuntimeError(
            f"ASE atom count ({positions.shape[0]}) != sum(atoms_per_list) ({expected})"
        )
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for n_per in atoms_per_list:
        end = start + int(n_per)
        blocks.append((positions[start:end], atomic_numbers[start:end]))
        start = end
    return blocks


def _match_atoms_to_template_names(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    template_positions: np.ndarray,
    template_names: Sequence[str],
    template_atomic_numbers: np.ndarray,
) -> list[str]:
    """Map ASE atom rows onto CHARMM template atom names via element-aware assignment."""
    from scipy.optimize import linear_sum_assignment

    n_atoms = int(positions.shape[0])
    if n_atoms != len(template_names) or n_atoms != int(template_atomic_numbers.shape[0]):
        raise ValueError("Template and ASE molecule atom counts differ")

    tmpl_pos = np.asarray(template_positions, dtype=float)
    tmpl_pos = tmpl_pos - tmpl_pos.mean(axis=0)
    pos = np.asarray(positions, dtype=float)
    pos = pos - pos.mean(axis=0)

    cost = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if int(atomic_numbers[i]) != int(template_atomic_numbers[j]):
                cost[i, j] = 1.0e6
            else:
                cost[i, j] = float(np.linalg.norm(pos[i] - tmpl_pos[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    if any(cost[r, c] >= 1.0e5 for r, c in zip(row_ind, col_ind)):
        raise RuntimeError("Could not align ASE molecule atoms to CHARMM template by element")

    names = [""] * n_atoms
    for row, col in zip(row_ind, col_ind):
        names[int(row)] = str(template_names[int(col)])
    return names


def _match_molecule_blocks_to_psf_order(
    molecule_blocks: list[tuple[np.ndarray, np.ndarray]],
    ordered_residue_names: Sequence[str],
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Permute ASE molecule blocks to match PSF residue order."""
    used = [False] * len(molecule_blocks)
    ordered: list[tuple[np.ndarray, np.ndarray]] = []
    for residue in ordered_residue_names:
        key = str(residue).upper()
        tmpl_z = residue_geometries[key][2]
        target = _atomic_number_multiset(tmpl_z)
        n_atoms = int(tmpl_z.shape[0])
        match_idx: int | None = None
        for idx, (_pos, z_block) in enumerate(molecule_blocks):
            if used[idx]:
                continue
            if int(z_block.shape[0]) != n_atoms:
                continue
            if _atomic_number_multiset(z_block) != target:
                continue
            match_idx = idx
            break
        if match_idx is None:
            raise RuntimeError(
                f"Could not match a PyXtal ASE molecule block to PSF residue {key}"
            )
        used[match_idx] = True
        ordered.append(molecule_blocks[match_idx])
    return ordered


def write_psf_order_mapping_pdb(
    pdb_path: Path,
    molecule_blocks: list[tuple[np.ndarray, np.ndarray]],
    ordered_residue_names: Sequence[str],
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
) -> Path:
    """Write a Packmol-style PDB with CHARMM atom names and PSF residue indices."""
    from mmml.interfaces.pycharmmInterface.packmol_placement import _element_symbol

    lines = [
        "REMARK   mmml PyXtal cluster (CHARMM atom names for PSF reordering)",
        "CRYST1   200.000   200.000   200.000  90.00  90.00  90.00 P 1           1",
    ]
    serial = 1
    for mol_idx, ((pos, z_block), residue) in enumerate(
        zip(molecule_blocks, ordered_residue_names), start=1
    ):
        key = str(residue).upper()
        tmpl_pos, tmpl_names, tmpl_z = residue_geometries[key]
        atom_names = _match_atoms_to_template_names(
            pos,
            z_block,
            tmpl_pos,
            tmpl_names,
            tmpl_z,
        )
        resn = key[:3] or "UNK"
        for name, zi, xyz in zip(atom_names, z_block, pos):
            elem = _element_symbol(int(zi))
            x, y, z_coord = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            lines.append(
                f"ATOM  {serial:5d} {name[:4]:>4s} {resn:<3s} A{mol_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z_coord:8.3f}  1.00  0.00          "
                f"{elem:>2s}"
            )
            serial += 1
    lines.append("END")
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pdb_path


def assign_ase_cluster_to_psf_order(
    atoms: Any,
    *,
    psf_atom_names: list[str],
    atoms_per_list: list[int],
    ordered_residue_names: list[str],
    residue_geometries: dict[str, tuple[np.ndarray, list[str], np.ndarray]],
    pdb_path: Path | None = None,
    trim_to_composition: bool = True,
) -> np.ndarray:
    """Map PyXtal ASE coordinates onto CHARMM PSF atom order."""
    from ase import Atoms

    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        assign_packmol_pdb_to_psf_order,
    )

    target_molecules = len(ordered_residue_names)
    expected_atoms = int(np.sum(atoms_per_list))
    n_atoms = len(atoms)
    if n_atoms > expected_atoms:
        if trim_to_composition:
            print(
                f"WARN: PyXtal ASE atom count ({n_atoms}) exceeds composition "
                f"({expected_atoms}); trimming to first {target_molecules} molecule(s)",
                flush=True,
            )
            atoms = Atoms(
                symbols=atoms.get_chemical_symbols()[:expected_atoms],
                positions=atoms.get_positions()[:expected_atoms],
                cell=atoms.cell,
                pbc=atoms.pbc,
            )
        else:
            raise RuntimeError(
                f"PyXtal ASE atom count ({n_atoms}) exceeds composition "
                f"({expected_atoms}). Adjust --pyxtal-stoichiometry and/or "
                f"--pyxtal-supercell, or pass --no-pyxtal-trim."
            )
    elif n_atoms < expected_atoms:
        raise RuntimeError(
            f"PyXtal ASE atom count ({n_atoms}) is below composition "
            f"({expected_atoms}). Adjust --pyxtal-stoichiometry and/or --pyxtal-supercell."
        )

    blocks = _split_ase_atoms_into_molecules(atoms, atoms_per_list)
    ordered_blocks = _match_molecule_blocks_to_psf_order(
        blocks,
        ordered_residue_names,
        residue_geometries,
    )
    out_pdb = pdb_path or Path("pyxtal_cluster_mapping.pdb")
    write_psf_order_mapping_pdb(
        out_pdb,
        ordered_blocks,
        ordered_residue_names,
        residue_geometries,
    )
    return assign_packmol_pdb_to_psf_order(
        out_pdb,
        psf_atom_names,
        atoms_per_list,
    )
