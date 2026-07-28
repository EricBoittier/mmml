"""Rigid internal-coordinate geometry preparation from atom-index DoFs."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read

from .config import DegreeOfFreedom, IcScanConfig
from .grid import ScanPoint, expand_scan_points


def load_structure(path: Path | str) -> Atoms:
    """Load a single molecular structure (first frame if multi-frame)."""

    atoms = read(Path(path).expanduser())
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"no frames found in {path}")
        atoms = atoms[0]
    if not isinstance(atoms, Atoms):
        raise TypeError(f"expected ASE Atoms from {path}, got {type(atoms)!r}")
    return atoms


def measure_dof(atoms: Atoms, dof: DegreeOfFreedom) -> float:
    """Return the current value of a DoF on ``atoms`` (Å or degrees)."""

    indices = dof.atoms
    if dof.kind == "bond":
        return float(atoms.get_distance(indices[0], indices[1]))
    if dof.kind == "angle":
        return float(atoms.get_angle(*indices))
    return float(atoms.get_dihedral(*indices))


def measure_all(atoms: Atoms, dofs: tuple[DegreeOfFreedom, ...]) -> dict[str, float]:
    return {dof.name: measure_dof(atoms, dof) for dof in dofs}


def default_moving_indices(atoms: Atoms, dof: DegreeOfFreedom) -> list[int]:
    """Indices of atoms moved when setting this DoF.

    Defaults follow common ASE practice:
    - bond: move the second atom and everything after it in index order
    - angle / dihedral: move from the third listed atom onward

    Prefer ``indices=`` over boolean ``mask=`` for ``set_distance`` (ASE 3.28
    treats ndarray masks as truthy ambiguously).
    """

    n = len(atoms)
    if dof.mask is not None:
        return list(dof.mask)
    if dof.kind == "bond":
        start = dof.atoms[1]
    else:
        start = dof.atoms[2]
    return list(range(start, n))


def apply_dof(atoms: Atoms, dof: DegreeOfFreedom, value: float) -> None:
    """Set one internal coordinate in-place on ``atoms``."""

    moving = default_moving_indices(atoms, dof)
    indices = dof.atoms
    if dof.kind == "bond":
        atoms.set_distance(indices[0], indices[1], value, fix=0, indices=moving)
    elif dof.kind == "angle":
        atoms.set_angle(*indices, value, indices=moving)
    else:
        atoms.set_dihedral(*indices, value, indices=moving)


def apply_coordinates(
    base: Atoms,
    dofs: tuple[DegreeOfFreedom, ...],
    coordinates: dict[str, float],
) -> Atoms:
    """Return a copy of ``base`` with all configured DoFs set to ``coordinates``."""

    atoms = base.copy()
    dof_map = {dof.name: dof for dof in dofs}
    for name, value in coordinates.items():
        apply_dof(atoms, dof_map[name], float(value))
    return atoms


def prepare_geometries(
    config: IcScanConfig,
    *,
    base: Atoms | None = None,
) -> tuple[Atoms, list[tuple[ScanPoint, Atoms]]]:
    """Build all rigid scan geometries without evaluating energies.

    Returns the reference structure and an ordered list of ``(point, atoms)``.
    """

    if config.geometry_mode != "rigid":
        raise ValueError(f"unsupported geometry_mode: {config.geometry_mode!r}")
    structure = base if base is not None else load_structure(config.structure)
    _validate_indices(structure, config.dofs)
    base_values = measure_all(structure, config.dofs)
    points = expand_scan_points(config, base_values=base_values)
    prepared: list[tuple[ScanPoint, Atoms]] = []
    for point in points:
        atoms = apply_coordinates(structure, config.dofs, point.coordinates)
        measured = measure_all(atoms, config.dofs)
        atoms.info.update(point.to_info())
        for name, value in measured.items():
            atoms.info[f"actual_{name}"] = float(value)
        prepared.append((point, atoms))
    return structure, prepared


def _validate_indices(atoms: Atoms, dofs: tuple[DegreeOfFreedom, ...]) -> None:
    n = len(atoms)
    for dof in dofs:
        for index in dof.atoms:
            if index >= n:
                raise ValueError(
                    f"DoF {dof.name!r} atom index {index} out of range for "
                    f"{n}-atom structure"
                )
        if dof.mask is not None:
            for index in dof.mask:
                if index >= n:
                    raise ValueError(
                        f"DoF {dof.name!r} mask index {index} out of range for "
                        f"{n}-atom structure"
                    )
