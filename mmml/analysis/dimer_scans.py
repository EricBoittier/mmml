"""Small ASE helpers for rigid molecular dimer scans.

The utilities here intentionally stay calculator-agnostic.  They produce
deterministic dimer geometries with fragment metadata that can be consumed by
learned multipole, MBD, xTB, SpookyNet, CHARMM/CGenFF, or hybrid calculators.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import shutil

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
    offset_angstrom: float = 0.0


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
    """Return a copy translated so the selected center is at the origin.

    *center* may be ``'centroid'`` (default), ``'com'`` (centre of mass), or
    ``'none'`` to skip centring (returns a plain copy).
    """

    centered = atoms.copy()
    if center == "none":
        return centered
    if center == "centroid":
        origin = geometric_centroid(centered)
    elif center == "com":
        origin = np.asarray(centered.get_center_of_mass(), dtype=np.float64)
    else:
        raise ValueError("center must be 'centroid', 'com', or 'none'")
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


def build_rigid_dimer_2d(
    monomer_a: Atoms,
    monomer_b: Atoms,
    *,
    distance_angstrom: float,
    offset_angstrom: float = 0.0,
    axis: Sequence[float] = (1.0, 0.0, 0.0),
    transverse_axis: Sequence[float] = (0.0, 1.0, 0.0),
    center: str = "centroid",
    mol_id_array: str = "mol_id",
) -> tuple[Atoms, tuple[np.ndarray, np.ndarray]]:
    """Place two rigid monomers at a fixed separation distance and transverse offset.

    Pass ``center='none'`` when *monomer_a* and *monomer_b* are already centred
    and pre-oriented (e.g. from :func:`mmml.analysis.dimer_molecules.orient_molecule`).
    """
    direction = normalized_vector(axis, name="axis")
    trans_direction = normalized_vector(transverse_axis, name="transverse_axis")

    monomer_a_centered = centered_atoms(monomer_a, center=center)
    monomer_b_centered = centered_atoms(monomer_b, center=center)
    
    # Translate A and B along the separation axis
    monomer_a_centered.translate(-0.5 * float(distance_angstrom) * direction)
    monomer_b_centered.translate(0.5 * float(distance_angstrom) * direction)
    
    # Apply transverse displacement/offset to monomer B
    monomer_b_centered.translate(float(offset_angstrom) * trans_direction)
    
    combined = monomer_a_centered + monomer_b_centered
    combined = assign_mol_id(
        combined,
        [len(monomer_a_centered), len(monomer_b_centered)],
        array_name=mol_id_array,
    )
    fragments = fragment_index_arrays([len(monomer_a_centered), len(monomer_b_centered)])
    return combined, (fragments[0], fragments[1])


def distance_scan_geometries_2d(
    monomer_a: Atoms,
    monomer_b: Atoms,
    distances_angstrom: Iterable[float],
    offsets_angstrom: Iterable[float],
    *,
    pair: tuple[str, str] = ("A", "B"),
    axis: Sequence[float] = (1.0, 0.0, 0.0),
    transverse_axis: Sequence[float] = (0.0, 1.0, 0.0),
    center: str = "centroid",
    mol_id_array: str = "mol_id",
) -> Iterator[DimerGeometry]:
    """Yield rigid dimer geometries over a center-to-center distance and offset scan."""
    for offset_angstrom in offsets_angstrom:
        for distance_angstrom in distances_angstrom:
            atoms, fragments = build_rigid_dimer_2d(
                monomer_a,
                monomer_b,
                distance_angstrom=float(distance_angstrom),
                offset_angstrom=float(offset_angstrom),
                axis=axis,
                transverse_axis=transverse_axis,
                center=center,
                mol_id_array=mol_id_array,
            )
            yield DimerGeometry(
                pair=pair,
                distance_angstrom=float(distance_angstrom),
                offset_angstrom=float(offset_angstrom),
                atoms=atoms,
                fragments=fragments,
            )


def min_fragment_contact_distance(
    atoms: Atoms, fragments: tuple[np.ndarray, np.ndarray]
) -> float:
    """Closest atom-atom distance between the two dimer fragments (Å).

    ``distance_angstrom`` in a scan is measured between each monomer's
    chemically-motivated anchor point, not its centroid or van der Waals
    surface — so for bulky/asymmetric molecules a nominal "close" scan
    distance can put atoms on opposite fragments on top of each other. This
    is the actual physical separation to check before trusting energies near
    the bottom of a distance grid.
    """
    pos = atoms.get_positions()
    idx_a, idx_b = fragments
    dmat = np.linalg.norm(pos[idx_a][:, None, :] - pos[idx_b][None, :, :], axis=-1)
    return float(dmat.min())


def find_safe_min_distance(
    monomer_a: Atoms,
    monomer_b: Atoms,
    *,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    transverse_axis: Sequence[float] = (0.0, 1.0, 0.0),
    min_contact: float = 1.5,
    search_range: tuple[float, float] = (1.5, 8.0),
    step: float = 0.1,
) -> float:
    """Smallest on-axis (offset=0) centre-to-centre distance clearing *min_contact*.

    A scan's ``distance_angstrom`` is anchor-to-anchor, not atom-to-atom, so a
    single fixed distance floor is either unsafe for bulky/asymmetric pairs
    (atoms overlapping) or wasteful for compact ones (many scanned points
    sitting deep in an already-known-clashing region). This does a cheap
    geometry-only sweep (no energy evaluation) to find where fragment atoms
    actually stop overlapping, so a scan grid can be anchored per pair.
    Returns *search_range[1]* if no distance in range clears the threshold.
    """
    d = search_range[0]
    while d <= search_range[1] + 1e-9:
        atoms, fragments = build_rigid_dimer_2d(
            monomer_a, monomer_b,
            distance_angstrom=d, offset_angstrom=0.0,
            axis=axis, transverse_axis=transverse_axis, center="none",
        )
        if min_fragment_contact_distance(atoms, fragments) >= min_contact:
            return float(d)
        d += step
    return float(search_range[1])


def evaluate_scan(
    geometries: Iterable[DimerGeometry],
    calculator_factory,
) -> list[dict[str, float | str]]:
    """Evaluate a set of dimer geometries with an ASE calculator factory."""

    rows: list[dict[str, float | str]] = []
    for geometry in geometries:
        atoms = geometry.atoms.copy()
        atoms.calc = calculator_factory()
        try:
            energy_ev = float(atoms.get_potential_energy())
            row = {
                "molecule_a": geometry.pair[0],
                "molecule_b": geometry.pair[1],
                "distance_angstrom": geometry.distance_angstrom,
                "offset_angstrom": geometry.offset_angstrom,
                "energy_ev": energy_ev,
                "energy_kcal_mol": energy_ev * 23.060548867,
                "min_contact_angstrom": min_fragment_contact_distance(
                    geometry.atoms, geometry.fragments
                ),
            }
            if hasattr(atoms.calc, "results") and "pair_energies_by_component" in atoms.calc.results:
                comp_list = atoms.calc.results["pair_energies_by_component"]
                if comp_list:
                    for k, val_ev in comp_list[0].items():
                        if k != "pair":
                            row[f"comp_{k}_ev"] = val_ev
                            row[f"comp_{k}_kcal_mol"] = val_ev * 23.060548867
            rows.append(row)
        except Exception as e:
            print(f"    Warning: calculation failed at {geometry.distance_angstrom} Å: {e}")
    return rows


def evaluate_scan_monomer_decomposed(
    geometries: Iterable[DimerGeometry],
    calculator_factory,
) -> list[dict[str, float | str]]:
    """Evaluate a scan with the dimer/monomer energy decomposition.

    For each geometry, computes the dimer energy ``E_dimer`` and the two
    isolated-monomer energies ``Ea``, ``Eb`` (monomer atoms taken at their
    dimer-geometry positions, using ``geometry.fragments``) with the same
    calculator. Reports the interaction energy ``E_int = E_dimer - Ea - Eb``
    alongside the reconstructed total ``E_int + Ea + Eb`` (== ``E_dimer``) as
    ``energy_ev`` / ``energy_kcal_mol``, so downstream consumers get both the
    absolute energy and its monomer decomposition (``comp_Ea_ev``,
    ``comp_Eb_ev``, ``comp_Eint_ev`` and ``_kcal_mol`` counterparts).
    """

    rows: list[dict[str, float | str]] = []
    component_keys = (
        "neural_energy",
        "electrostatics_energy",
        "cgenff_vdw_energy",
        "zbl_repulsion_energy",
        "mbd_energy",
    )
    for geometry in geometries:
        atoms = geometry.atoms.copy()
        idx_a, idx_b = geometry.fragments
        atoms_a = geometry.atoms[idx_a].copy()
        atoms_b = geometry.atoms[idx_b].copy()
        try:
            atoms.calc = calculator_factory()
            e_dimer_ev = float(atoms.get_potential_energy())
            components_dimer = {
                key: float(atoms.calc.results.get(key, 0.0)) for key in component_keys
            }

            atoms_a.calc = calculator_factory()
            e_a_ev = float(atoms_a.get_potential_energy())
            components_a = {
                key: float(atoms_a.calc.results.get(key, 0.0)) for key in component_keys
            }

            atoms_b.calc = calculator_factory()
            e_b_ev = float(atoms_b.get_potential_energy())
            components_b = {
                key: float(atoms_b.calc.results.get(key, 0.0)) for key in component_keys
            }

            e_int_ev = e_dimer_ev - e_a_ev - e_b_ev
            e_hybrid_ev = e_int_ev + e_a_ev + e_b_ev

            row: dict[str, float | str] = {
                    "molecule_a": geometry.pair[0],
                    "molecule_b": geometry.pair[1],
                    "distance_angstrom": geometry.distance_angstrom,
                    "offset_angstrom": geometry.offset_angstrom,
                    "energy_ev": e_hybrid_ev,
                    "energy_kcal_mol": e_hybrid_ev * 23.060548867,
                    "comp_Ea_ev": e_a_ev,
                    "comp_Ea_kcal_mol": e_a_ev * 23.060548867,
                    "comp_Eb_ev": e_b_ev,
                    "comp_Eb_kcal_mol": e_b_ev * 23.060548867,
                    "comp_Eint_ev": e_int_ev,
                    "comp_Eint_kcal_mol": e_int_ev * 23.060548867,
                    "min_contact_angstrom": min_fragment_contact_distance(
                        geometry.atoms, geometry.fragments
                    ),
                }
            for key in component_keys:
                component_int = (
                    components_dimer[key] - components_a[key] - components_b[key]
                )
                row[f"comp_Eint_{key}_ev"] = component_int
                row[f"comp_Eint_{key}_kcal_mol"] = component_int * 23.060548867
            rows.append(row)
        except Exception as e:
            print(f"    Warning: calculation failed at {geometry.distance_angstrom} Å: {e}")
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
            return TBLite(method=method, **kwargs)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Neither xTB nor tblite ASE support is installed. Install one of "
                "the optional packages (xtb-python or tblite) in the runtime environment."
            ) from exc


def make_dftb3_d4_calculator(
    *,
    slako_dir: str | Path,
    workdir: str | Path,
    command: str = "dftb+",
) -> Calculator:
    """Create an ASE-backed DFTB3-D4 calculator for molecular dimers.

    This mirrors the DFTB+ recipe's DFTB3/3ob-3-1 + D4 setup.  DFTB+ itself
    and the 3ob Slater--Koster files are external runtime assets: keeping them
    optional avoids making the regular scan environment depend on a Fortran
    executable or a large parameter-data download.

    ``workdir`` is deliberately explicit because the DFTB+ file-I/O interface
    writes ``dftb_in.hsd``, ``geo_end.gen``, and result files on every call.
    The campaign evaluates one geometry at a time, so one reusable directory
    is sufficient and prevents those transient files from polluting the repo.
    """

    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError(
            f"DFTB+ executable {command!r} was not found on PATH. "
            "Install DFTB+ or pass its executable with --dftb-command."
        )

    slako_path = Path(slako_dir).expanduser().resolve()
    if not slako_path.is_dir():
        raise FileNotFoundError(
            f"DFTB3 Slater--Koster directory does not exist: {slako_path}. "
            "Point --dftb-sk-dir at the 3ob-3-1 directory."
        )

    # Fail before the scan when the minimal element set used in this campaign
    # cannot be represented.  3ob-3-1 also supplies Cl, needed by DCM.
    required = ("H-H.skf", "C-C.skf", "O-O.skf", "Cl-Cl.skf")
    missing = [name for name in required if not (slako_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{slako_path} is not a complete 3ob-3-1 directory; missing: "
            + ", ".join(missing)
        )

    from ase.calculators.dftb import Dftb

    scratch = Path(workdir).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    # The DFTD4 constants are the DFTB+ recipe values for DFTB3/3ob-3-1.
    return Dftb(
        label=str(scratch / "dftb3_d4"),
        command=f"{executable} > PREFIX.out",
        slako_dir=str(slako_path),
        Hamiltonian_SCC="Yes",
        Hamiltonian_SCCTolerance=1.0e-8,
        Hamiltonian_ThirdOrderFull="Yes",
        Hamiltonian_HCorrection_="Damping",
        Hamiltonian_HCorrection_Exponent=4.0,
        Hamiltonian_Dispersion_="DFTD4",
        Hamiltonian_Dispersion_s6=1.0,
        Hamiltonian_Dispersion_s9=0.0,
        Hamiltonian_Dispersion_s8=0.4727337,
        Hamiltonian_Dispersion_a1=0.5467502,
        Hamiltonian_Dispersion_a2=4.4955068,
        ParserOptions_ParserVersion=10,
    )
