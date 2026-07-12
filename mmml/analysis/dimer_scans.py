"""Small ASE helpers for rigid molecular dimer scans.

The utilities here intentionally stay calculator-agnostic.  They produce
deterministic dimer geometries with fragment metadata that can be consumed by
learned multipole, MBD, xTB, SpookyNet, CHARMM/CGenFF, or hybrid calculators.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator


@dataclass(frozen=True)
class DimerGeometry:
    """One rigid dimer geometry and its scan metadata."""

    pair: tuple[str, str]
    distance_angstrom: float
    atoms: Atoms
    fragments: tuple[np.ndarray, np.ndarray]


def molecule_pair_labels(
    labels: Sequence[str],
    *,
    include_homodimers: bool = True,
    include_heterodimers: bool = True,
) -> list[tuple[str, str]]:
    """Return molecule pairs in deterministic upper-triangular order."""

    if not include_homodimers and not include_heterodimers:
        return []

    unique_labels = list(dict.fromkeys(labels))
    if include_homodimers and include_heterodimers:
        return list(combinations_with_replacement(unique_labels, 2))
    if include_homodimers:
        return [(label, label) for label in unique_labels]
    return list(combinations(unique_labels, 2))


def normalized_vector(vector: Sequence[float], *, name: str = "vector") -> np.ndarray:
    """Return a unit vector, raising a clear error for zero-length input."""

    unit = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return unit / norm


def geometric_centroid(atoms: Atoms) -> np.ndarray:
    """Return the unweighted coordinate centroid in Å."""

    if len(atoms) == 0:
        raise ValueError("atoms must contain at least one atom")
    return np.asarray(atoms.get_positions(), dtype=np.float64).mean(axis=0)


def centered_atoms(atoms: Atoms, *, center: str = "centroid") -> Atoms:
    """Return a copy translated so the selected center is at the origin."""

    centered = atoms.copy()
    if center == "centroid":
        origin = geometric_centroid(centered)
    elif center == "com":
        origin = np.asarray(centered.get_center_of_mass(), dtype=np.float64)
    else:
        raise ValueError("center must be 'centroid' or 'com'")
    centered.translate(-origin)
    return centered


def assign_mol_id(
    atoms: Atoms,
    fragment_sizes: Sequence[int],
    *,
    array_name: str = "mol_id",
) -> Atoms:
    """Return a copy with a per-atom integer fragment/molecule ID array."""

    total = sum(int(size) for size in fragment_sizes)
    if total != len(atoms):
        raise ValueError(
            f"fragment sizes sum to {total}, but atoms contains {len(atoms)} atoms"
        )

    tagged = atoms.copy()
    mol_ids = np.concatenate(
        [
            np.full(int(size), fragment_index, dtype=np.int64)
            for fragment_index, size in enumerate(fragment_sizes)
        ]
    )
    tagged.arrays[array_name] = mol_ids
    return tagged


def fragment_index_arrays(fragment_sizes: Sequence[int]) -> tuple[np.ndarray, ...]:
    """Return contiguous fragment index arrays for combined ASE systems."""

    starts = np.cumsum([0, *[int(size) for size in fragment_sizes[:-1]]])
    return tuple(
        np.arange(start, start + int(size), dtype=np.int64)
        for start, size in zip(starts, fragment_sizes, strict=True)
    )


def build_rigid_dimer(
    monomer_a: Atoms,
    monomer_b: Atoms,
    *,
    distance_angstrom: float,
    axis: Sequence[float] = (1.0, 0.0, 0.0),
    center: str = "centroid",
    mol_id_array: str = "mol_id",
) -> tuple[Atoms, tuple[np.ndarray, np.ndarray]]:
    """Place two rigid monomers at a fixed center-to-center separation.

    The returned geometry has monomer A centered at ``-0.5 * distance * axis``
    and monomer B centered at ``+0.5 * distance * axis``.  This keeps the dimer
    midpoint at the origin and makes scan coordinates easy to compare.
    """

    direction = normalized_vector(axis, name="axis")
    monomer_a_centered = centered_atoms(monomer_a, center=center)
    monomer_b_centered = centered_atoms(monomer_b, center=center)
    monomer_a_centered.translate(-0.5 * float(distance_angstrom) * direction)
    monomer_b_centered.translate(0.5 * float(distance_angstrom) * direction)

    combined = monomer_a_centered + monomer_b_centered
    combined = assign_mol_id(
        combined,
        [len(monomer_a_centered), len(monomer_b_centered)],
        array_name=mol_id_array,
    )
    fragments = fragment_index_arrays([len(monomer_a_centered), len(monomer_b_centered)])
    return combined, (fragments[0], fragments[1])


def distance_scan_geometries(
    monomer_a: Atoms,
    monomer_b: Atoms,
    distances_angstrom: Iterable[float],
    *,
    pair: tuple[str, str] = ("A", "B"),
    axis: Sequence[float] = (1.0, 0.0, 0.0),
    center: str = "centroid",
    mol_id_array: str = "mol_id",
) -> Iterator[DimerGeometry]:
    """Yield rigid dimer geometries over a center-to-center distance scan."""

    for distance_angstrom in distances_angstrom:
        atoms, fragments = build_rigid_dimer(
            monomer_a,
            monomer_b,
            distance_angstrom=float(distance_angstrom),
            axis=axis,
            center=center,
            mol_id_array=mol_id_array,
        )
        yield DimerGeometry(
            pair=pair,
            distance_angstrom=float(distance_angstrom),
            atoms=atoms,
            fragments=fragments,
        )


def evaluate_scan(
    geometries: Iterable[DimerGeometry],
    calculator_factory,
) -> list[dict[str, float | str]]:
    """Evaluate a set of dimer geometries with an ASE calculator factory."""

    rows: list[dict[str, float | str]] = []
    for geometry in geometries:
        atoms = geometry.atoms.copy()
        atoms.calc = calculator_factory()
        energy_ev = float(atoms.get_potential_energy())
        rows.append(
            {
                "molecule_a": geometry.pair[0],
                "molecule_b": geometry.pair[1],
                "distance_angstrom": geometry.distance_angstrom,
                "energy_ev": energy_ev,
                "energy_kcal_mol": energy_ev * 23.060548867,
            }
        )
    return rows


def make_xtb_calculator(
    *,
    method: str = "GFN2-xTB",
    **kwargs,
) -> Calculator:
    """Create an ASE xTB calculator when the optional ``xtb`` or ``tblite`` package exists."""

    try:
        from xtb.ase.calculator import XTB
        return XTB(method=method, **kwargs)
    except ModuleNotFoundError:
        try:
            from tblite.ase import TBLite
            tblite_method = method.lower().replace("-xtb", "")
            return TBLite(method=tblite_method, **kwargs)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Neither xTB nor tblite ASE support is installed. Install one of "
                "the optional packages (xtb-python or tblite) in the runtime environment."
            ) from exc

