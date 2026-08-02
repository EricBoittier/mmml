"""Validate monomer internal geometry against the templates a builder placed.

Packmol places rigid copies of a CHARMM-minimized monomer, so every monomer in a
freshly packed cluster has the template's internal geometry. A cluster MM
minimization relaxes bonds/angles and rotates torsions, but it must never break
the covalent skeleton. A broken CHARMM/pycharmm build can silently return
scrambled coordinates instead (atoms of different monomers landing on the same
point, GRMS reported as exactly 0.0 before and after), and those coordinates are
otherwise cached and reused.

Only 1-2 and 1-3 distances are compared: they are governed by the stiff bond and
angle terms, so they are near-invariant under a genuine minimization, while 1-4+
distances change by an Angstrom or more from a single legitimate torsion
rotation. Comparing distances (not coordinates) makes the check invariant to the
rigid rotation/translation Packmol applied.

Positions must have whole monomers (this runs before any PBC wrapping).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# Largest change in a 1-2/1-3 distance (Å) accepted between the placed monomer
# template and the CHARMM-minimized cluster. Genuine SD+ABNR relaxation of a
# packed liquid moves these by a few hundredths of an Å (see
# ``scripts/validate_packmol_monomer_geometry.py``); scrambled coordinates move
# them by more than an Å.
DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A = 0.35

# Set to 0 (or a negative value) to disable the check; set to a float to override.
MONOMER_INTERNAL_DEVIATION_ENV = "MMML_MAX_MONOMER_INTERNAL_DEVIATION_A"

# Covalent-radii sum scale used to detect bonds in the monomer template.
DEFAULT_BOND_SCALE = 1.25

_FALLBACK_COVALENT_RADII_A: dict[int, float] = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}


@dataclass(frozen=True)
class MonomerInternalDistortion:
    """Worst 1-2/1-3 distance change of one monomer versus its template."""

    monomer: int
    residue: str
    atom_i: int
    atom_j: int
    template_distance_A: float
    distance_A: float
    deviation_A: float


@dataclass(frozen=True)
class MonomerInternalGeometryReport:
    """Outcome of :func:`scan_monomer_internal_geometry`."""

    max_deviation_A: float
    worst: MonomerInternalDistortion | None
    n_monomers_checked: int
    n_monomers_skipped: int
    n_pairs_checked: int

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst
        return {
            "max_deviation_A": float(self.max_deviation_A),
            "n_monomers_checked": int(self.n_monomers_checked),
            "n_monomers_skipped": int(self.n_monomers_skipped),
            "n_pairs_checked": int(self.n_pairs_checked),
            "worst": None
            if worst is None
            else {
                "monomer": int(worst.monomer),
                "residue": str(worst.residue),
                "atom_i": int(worst.atom_i),
                "atom_j": int(worst.atom_j),
                "template_distance_A": float(worst.template_distance_A),
                "distance_A": float(worst.distance_A),
                "deviation_A": float(worst.deviation_A),
            },
        }


def resolve_max_monomer_internal_deviation_A(
    override: float | None = None,
) -> float:
    """Threshold (Å) for the post-minimize monomer check; ``0.0`` disables it."""
    if override is not None:
        value = float(override)
        return value if value > 0.0 else 0.0
    raw = os.environ.get(MONOMER_INTERNAL_DEVIATION_ENV)
    if raw is None or not str(raw).strip():
        return float(DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A)
    try:
        value = float(str(raw).strip())
    except ValueError:
        return float(DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A)
    return value if value > 0.0 else 0.0


def _covalent_radii_A(numbers: np.ndarray) -> np.ndarray:
    try:
        from ase.data import covalent_radii

        table = np.asarray(covalent_radii, dtype=float)
        radii = np.array(
            [
                float(table[z]) if 0 <= int(z) < len(table) else 0.0
                for z in numbers
            ],
            dtype=float,
        )
    except ImportError:
        radii = np.array(
            [float(_FALLBACK_COVALENT_RADII_A.get(int(z), 0.0)) for z in numbers],
            dtype=float,
        )
    # Unknown elements: a mid-range radius keeps them bonded to their neighbours
    # rather than silently dropping them from the skeleton.
    radii[radii <= 0.0] = 0.9
    return radii


def covalent_skeleton_pairs(
    coords: np.ndarray,
    numbers: Sequence[int] | np.ndarray,
    *,
    bond_scale: float = DEFAULT_BOND_SCALE,
) -> np.ndarray:
    """1-2 and 1-3 atom-index pairs of a monomer template, shape ``(n_pairs, 2)``.

    Empty when the template has no detectable bond (single atoms, monatomic ions).
    """
    pos = np.asarray(coords, dtype=float)
    z = np.asarray(numbers, dtype=int)
    n_atoms = int(pos.shape[0])
    if n_atoms < 2 or z.shape[0] != n_atoms:
        return np.empty((0, 2), dtype=int)

    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    radii = _covalent_radii_A(z)
    cutoff = float(bond_scale) * (radii[:, None] + radii[None, :])
    bonded = (dist < cutoff) & ~np.eye(n_atoms, dtype=bool)

    pairs: set[tuple[int, int]] = set()
    neighbors = [sorted(int(j) for j in np.flatnonzero(bonded[i])) for i in range(n_atoms)]
    for i in range(n_atoms):
        for a_idx, j in enumerate(neighbors[i]):
            if i < j:
                pairs.add((i, j))
            # 1-3: two neighbours of the same central atom i.
            for k in neighbors[i][a_idx + 1 :]:
                pairs.add((j, k) if j < k else (k, j))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(sorted(pairs), dtype=int)


def _template_coords_and_numbers(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Accept ``(coords, numbers)`` or ``(coords, atom_names, numbers)`` templates."""
    if isinstance(value, (tuple, list)) and len(value) == 3:
        coords, _atom_names, numbers = value
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        coords, numbers = value
    else:
        raise TypeError(
            "monomer template must be (coords, numbers) or (coords, atom_names, numbers)"
        )
    return np.asarray(coords, dtype=float), np.asarray(numbers, dtype=int)


def _monomer_offsets(atoms_per_list: Sequence[int]) -> np.ndarray:
    counts = np.asarray([int(n) for n in atoms_per_list], dtype=int)
    return np.concatenate([[0], np.cumsum(counts)]).astype(int)


def scan_monomer_internal_geometry(
    positions: np.ndarray,
    atoms_per_list: Sequence[int],
    *,
    residue_names: Sequence[str],
    templates: Mapping[str, Any],
    bond_scale: float = DEFAULT_BOND_SCALE,
) -> tuple[np.ndarray, MonomerInternalGeometryReport]:
    """Per-monomer max 1-2/1-3 distance deviation (Å) versus the placed template.

    Returns ``(deviations, report)``; ``deviations[i]`` is NaN for monomers that
    could not be checked (unknown residue, atom-count mismatch, no bonds).
    """
    pos = np.asarray(positions, dtype=float)
    offsets = _monomer_offsets(atoms_per_list)
    n_monomers = int(len(offsets) - 1)
    deviations = np.full(n_monomers, np.nan, dtype=float)
    if n_monomers <= 0:
        return deviations, MonomerInternalGeometryReport(
            max_deviation_A=0.0,
            worst=None,
            n_monomers_checked=0,
            n_monomers_skipped=0,
            n_pairs_checked=0,
        )
    if len(residue_names) != n_monomers:
        raise ValueError(
            f"residue_names has {len(residue_names)} entries for {n_monomers} monomers"
        )

    # Bond detection and template distances are per residue type, not per monomer.
    cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for key, value in templates.items():
        coords, numbers = _template_coords_and_numbers(value)
        pairs = covalent_skeleton_pairs(coords, numbers, bond_scale=bond_scale)
        if pairs.size == 0:
            cache[str(key).upper()] = (pairs, np.empty(0, dtype=float), int(len(coords)))
            continue
        ref = np.linalg.norm(
            coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=-1
        )
        cache[str(key).upper()] = (pairs, ref, int(len(coords)))

    worst: MonomerInternalDistortion | None = None
    max_deviation = 0.0
    n_checked = 0
    n_skipped = 0
    n_pairs_checked = 0
    for mi in range(n_monomers):
        residue = str(residue_names[mi]).upper()
        entry = cache.get(residue)
        start, end = int(offsets[mi]), int(offsets[mi + 1])
        if entry is None:
            n_skipped += 1
            continue
        pairs, ref, n_template_atoms = entry
        if pairs.size == 0 or (end - start) != n_template_atoms:
            n_skipped += 1
            continue
        chunk = pos[start:end]
        dist = np.linalg.norm(chunk[pairs[:, 0]] - chunk[pairs[:, 1]], axis=-1)
        delta = np.abs(dist - ref)
        n_checked += 1
        n_pairs_checked += int(pairs.shape[0])
        worst_idx = int(np.argmax(delta))
        deviations[mi] = float(delta[worst_idx])
        if float(delta[worst_idx]) > max_deviation:
            max_deviation = float(delta[worst_idx])
            worst = MonomerInternalDistortion(
                monomer=mi,
                residue=residue,
                atom_i=int(pairs[worst_idx, 0]),
                atom_j=int(pairs[worst_idx, 1]),
                template_distance_A=float(ref[worst_idx]),
                distance_A=float(dist[worst_idx]),
                deviation_A=float(delta[worst_idx]),
            )

    return deviations, MonomerInternalGeometryReport(
        max_deviation_A=float(max_deviation),
        worst=worst,
        n_monomers_checked=n_checked,
        n_monomers_skipped=n_skipped,
        n_pairs_checked=n_pairs_checked,
    )


def assert_monomer_internal_geometry(
    positions: np.ndarray,
    atoms_per_list: Sequence[int],
    *,
    residue_names: Sequence[str],
    templates: Mapping[str, Any],
    max_deviation_A: float = DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A,
    bond_scale: float = DEFAULT_BOND_SCALE,
    context: str = "monomer internal geometry",
) -> MonomerInternalGeometryReport:
    """Raise ``RuntimeError`` when any monomer's covalent skeleton is distorted.

    ``max_deviation_A <= 0`` measures without enforcing.
    """
    pos = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(pos)):
        n_bad = int(np.sum(~np.isfinite(pos)))
        raise RuntimeError(f"{context}: non-finite coordinates ({n_bad} value(s))")

    deviations, report = scan_monomer_internal_geometry(
        pos,
        atoms_per_list,
        residue_names=residue_names,
        templates=templates,
        bond_scale=bond_scale,
    )
    limit = float(max_deviation_A)
    if limit <= 0.0 or report.worst is None:
        return report
    if report.max_deviation_A <= limit:
        return report

    n_over = int(np.sum(np.nan_to_num(deviations, nan=-1.0) > limit))
    w = report.worst
    raise RuntimeError(
        f"{context}: {n_over}/{report.n_monomers_checked} monomer(s) have a distorted "
        f"covalent skeleton after minimization "
        f"(max 1-2/1-3 distance change {report.max_deviation_A:.3f} Å > {limit:.3f} Å). "
        f"Worst: monomer {w.monomer + 1} ({w.residue}) atoms {w.atom_i + 1}/{w.atom_j + 1}, "
        f"{w.template_distance_A:.3f} Å in the placed template → {w.distance_A:.3f} Å. "
        "Minimization does not break bonds — this usually means the CHARMM/pycharmm "
        "build returned garbage coordinates. Compare the pre-minimize Packmol PDB "
        "against the minimized coordinates to confirm, then rebuild CHARMM "
        f"(scripts/rebuild_charmm_mlpot.sh). Set {MONOMER_INTERNAL_DEVIATION_ENV} "
        "to override the threshold."
    )


# A relaxed cluster sits at O(1-100) kcal/mol of electrostatics per atom. Even a
# bad steric clash stays orders of magnitude below this. Exceeding it means atoms
# have been pulled onto each other, which only happens when the repulsive wall is
# missing.
COLLAPSED_ELEC_PER_ATOM_KCAL = 1.0e4


def charmm_collapsed_nonbonded_hint(n_atoms: int) -> str:
    """Extra diagnosis for a failed geometry gate: did the nonbonded table die?

    Keyed on the electrostatic energy, not VDW. A wiped NONBONDED table does not
    necessarily zero *every* VDW pair -- the parameters carried by the last
    appended file survive, so e.g. MEOH (CG331/HGA3, present in the bundled
    ``par_ch3cl.prm``) still reports a plausible-looking VDW while TIP3 (OT/HT)
    reports zero. Measured on a MEOH:4 cluster: VDW 0.171 with the table wiped
    versus -1.638 healthy, which no threshold can separate. The collapse shows up
    unambiguously in ELEC instead: -9.1e6 kcal/mol against 43.1 healthy.

    Lives here rather than next to its caller so that importing it does not pull
    in ``import_pycharmm``, which binds ``pycharmm`` at import time and captures
    ``None`` while the session is cold -- poisoning any live-CHARMM test
    collected afterwards. ``pycharmm.energy`` is imported per call instead.

    Returns "" when nothing looks wrong, or when CHARMM cannot be read. Never
    raises: this only decorates an error that is already being raised.
    """
    n_atoms = int(n_atoms)
    if n_atoms < 2:
        return ""
    try:
        import pycharmm.energy as energy

        elec = float(energy.get_elec())
    except Exception:  # noqa: BLE001 - diagnosis must not mask the real error
        return ""
    if abs(elec) < COLLAPSED_ELEC_PER_ATOM_KCAL * n_atoms:
        return ""
    return (
        f"CHARMM also reports ELEC={elec:.4g} kcal/mol for {n_atoms} atoms "
        f"({elec / n_atoms:.4g} per atom), which is not a physical relaxed "
        "structure -- the minimization ran without a full van der Waals wall, so "
        "electrostatics pulled atoms onto each other. The usual cause is CHARMM's "
        "NONBONDED table being wiped by a repeated CGenFF parameter read; see "
        "tests/functionality/charmm/test_charmm_param_read_contract.py and the "
        "saved READ PARAM APPEND flag in setup/charmm/source/api/api_read.F90."
    )
