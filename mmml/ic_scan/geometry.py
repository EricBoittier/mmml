"""Rigid internal-coordinate geometry preparation from atom-index DoFs."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read

from .config import DegreeOfFreedom, IcScanConfig
from .grid import ScanPoint, expand_scan_points
from .topology import (
    angles_match,
    atoms_on_side,
    circular_delta_deg,
    covalent_bond_graph,
)


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


def default_moving_indices(
    atoms: Atoms,
    dof: DegreeOfFreedom,
    *,
    adjacency: dict[int, set[int]] | None = None,
) -> list[int]:
    """Indices of atoms moved when setting this DoF.

    Prefer an explicit ``dof.mask``. Otherwise use the covalent bond graph:

    - **bond** ``a0–a1``: fragment on the ``a1`` side of the bond
    - **angle** ``a1–a2–a3``: fragment on the ``a3`` side of ``a2–a3``
    - **dihedral** ``a1–a2–a3–a4``: fragment on the ``a3`` side of ``a2–a3``
      (must include ``a4``)

    Index-order fallbacks (``range(a3, n)``) are *not* used: PSF/ASE index order
    is not a topological side of the scanned bond.

    ASE ``set_distance`` takes ``indices=`` (not a boolean mask) on ASE 3.28+.
    """

    if dof.mask is not None:
        return list(dof.mask)
    graph = adjacency if adjacency is not None else covalent_bond_graph(atoms)
    if dof.kind == "bond":
        seed, block = dof.atoms[1], dof.atoms[0]
    elif dof.kind == "angle":
        seed, block = dof.atoms[2], dof.atoms[1]
    else:
        seed, block = dof.atoms[2], dof.atoms[1]
    moving = atoms_on_side(graph, seed=seed, block=block)
    if dof.kind == "dihedral" and dof.atoms[3] not in moving:
        raise ValueError(
            f"DoF {dof.name!r}: topology mask for dihedral {list(dof.atoms)} "
            f"does not include a4={dof.atoms[3]}. Check atom order: ASE rotates "
            f"about a2–a3 with a4 on the a3 side. For NMA amide C–C–N–C use "
            f"CL–C–N–CR = [0, 4, 6, 8] (a4=CR on the N side), not a methyl "
            f"hydrogen as a4 unless the methyl carbon is a3."
        )
    return moving


def validate_moving_indices(dof: DegreeOfFreedom, moving: list[int]) -> None:
    """Raise if an explicit/implicit mask cannot realize the ASE set_* contract."""

    if dof.kind == "bond":
        tip = dof.atoms[1]
        if tip not in moving:
            raise ValueError(
                f"DoF {dof.name!r}: bond mask must include a1={tip} "
                f"(got {moving})"
            )
        return
    if dof.kind == "angle":
        tip = dof.atoms[2]
        if tip not in moving:
            raise ValueError(
                f"DoF {dof.name!r}: angle mask must include a3={tip} "
                f"(got {moving})"
            )
        return
    tip = dof.atoms[3]
    if tip not in moving:
        raise ValueError(
            f"DoF {dof.name!r}: dihedral mask must include a4={tip}. "
            f"ASE only moves atoms listed in mask/indices; if a4 is missing the "
            f"requested torsion is not applied. Got mask={moving}. "
            f"For amide CL–C–N–CR include CR and the N-methyl fragment "
            f"(e.g. [7, 8, 9, 10, 11]), not only the methyl hydrogens."
        )
    axis = {dof.atoms[1], dof.atoms[2]}
    if axis <= set(moving) and len(moving) <= 2:
        raise ValueError(
            f"DoF {dof.name!r}: mask {moving} only covers the central bond; "
            f"include the fragment on the a3 side (at least a4={tip})."
        )


def apply_dof(
    atoms: Atoms,
    dof: DegreeOfFreedom,
    value: float,
    *,
    adjacency: dict[int, set[int]] | None = None,
) -> None:
    """Set one internal coordinate in-place on ``atoms``."""

    moving = default_moving_indices(atoms, dof, adjacency=adjacency)
    validate_moving_indices(dof, moving)
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
    *,
    active_dofs: tuple[str, ...] | None = None,
    atol_deg: float = 1.0,
    atol_bond: float = 1.0e-3,
    max_passes: int = 8,
) -> Atoms:
    """Return a copy of ``base`` with DoFs set to ``coordinates``.

    When several torsions are set (N-D scans), one pass can disturb another.
    Coordinates are re-applied in config order for up to ``max_passes`` until
    every requested value matches within tolerance, then verified.
    """

    atoms = base.copy()
    adjacency = covalent_bond_graph(atoms)
    dof_map = {dof.name: dof for dof in dofs}
    # Apply active DoFs first (scan axes), then any remaining keys (references).
    if active_dofs is not None:
        order = list(active_dofs) + [
            name for name in coordinates if name not in active_dofs
        ]
    else:
        order = [dof.name for dof in dofs if dof.name in coordinates]
        order.extend(name for name in coordinates if name not in order)

    def _matches(dof: DegreeOfFreedom, target: float) -> bool:
        actual = measure_dof(atoms, dof)
        if dof.kind == "bond":
            return abs(actual - target) <= atol_bond
        return angles_match(actual, target, atol_deg=atol_deg)

    for _ in range(max_passes):
        for name in order:
            apply_dof(atoms, dof_map[name], float(coordinates[name]), adjacency=adjacency)
        if all(
            _matches(dof_map[name], float(coordinates[name])) for name in order
        ):
            break
    else:
        problems = []
        for name in order:
            dof = dof_map[name]
            target = float(coordinates[name])
            actual = measure_dof(atoms, dof)
            if not _matches(dof, target):
                if dof.kind == "bond":
                    problems.append(
                        f"{name}: requested {target:.6g} Å, got {actual:.6g} Å"
                    )
                else:
                    problems.append(
                        f"{name}: requested {target:.4g}°, got {actual:.4g}° "
                        f"(Δ={circular_delta_deg(actual, target):+.4g}°)"
                    )
        raise ValueError(
            "failed to realize requested internal coordinates after "
            f"{max_passes} passes:\n  - "
            + "\n  - ".join(problems)
            + "\nCheck dihedral atom order (a4 on the a3 side of a2–a3) and "
            "mask (must include a4 and that fragment). See docs/ic-scan-design.md."
        )
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
    adjacency = covalent_bond_graph(structure)
    for dof in config.dofs:
        moving = default_moving_indices(structure, dof, adjacency=adjacency)
        validate_moving_indices(dof, moving)
    base_values = measure_all(structure, config.dofs)
    points = expand_scan_points(config, base_values=base_values)
    prepared: list[tuple[ScanPoint, Atoms]] = []
    for point in points:
        atoms = apply_coordinates(
            structure,
            config.dofs,
            point.coordinates,
            active_dofs=point.active_dofs,
        )
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
