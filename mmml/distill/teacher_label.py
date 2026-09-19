"""Label geometries with an ASE teacher (PET-MAD or a dummy calculator).

Default ``energy_mode=interaction`` writes the MMML hybrid pieces:

* monomers: ``E = E_teacher - E_ref`` (``E_ref`` is the first ``pdb_eq`` monomer)
* dimers: unswitched ``E = E(AB) - E(A) - E(B)`` and matching forces

Do **not** bake ``ml_switch_scale`` into the labels; MLpot applies the handoff
at MD time. Units: energy eV, forces eV/Å (ASE / metatomic).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase.calculators.calculator import Calculator

from mmml.distill.acetone_pool import Geometry
from mmml.interfaces.calculators.ase_fragment_hybrid import evaluate_whole_system

ENERGY_MODE_INTERACTION = "interaction"
ENERGY_MODE_TOTAL = "total"
ENERGY_MODES = (ENERGY_MODE_INTERACTION, ENERGY_MODE_TOTAL)


@dataclass
class LabeledSample:
    geometry: Geometry
    energy_eV: float
    forces_ev_per_angstrom: np.ndarray
    energy_total_eV: float
    energy_int_eV: float | None


class TeacherEvaluator(Protocol):
    """Batched teacher: ``(numbers, positions)`` list → ``(E eV, F eV/Å)`` list."""

    def evaluate(
        self, structures: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[float, np.ndarray]]: ...


class AseTeacher:
    """One ASE ``Calculator`` call per structure (reference / tests)."""

    def __init__(self, calculator: Calculator) -> None:
        self.calculator = calculator

    def evaluate(
        self, structures: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[float, np.ndarray]]:
        out = []
        for numbers, positions in structures:
            res = evaluate_whole_system(self.calculator, numbers, positions)
            out.append(
                (float(res.energy_ev), np.asarray(res.forces_ev_per_angstrom, dtype=np.float64))
            )
        return out


def _as_teacher(teacher: Calculator | TeacherEvaluator) -> TeacherEvaluator:
    return teacher if hasattr(teacher, "evaluate") else AseTeacher(teacher)  # type: ignore[arg-type]


def _split_dimer(geo: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_a, n_b = geo.atoms_per_monomer
    z = geo.numbers
    r = geo.positions
    return z[:n_a], r[:n_a], z[n_a : n_a + n_b], r[n_a : n_a + n_b]


def label_geometries(
    teacher: Calculator | TeacherEvaluator,
    geometries: list[Geometry],
    *,
    energy_mode: str = ENERGY_MODE_INTERACTION,
) -> list[LabeledSample]:
    """Evaluate the teacher on each geometry. ``energy_mode`` selects the stored E/F.

    ``teacher`` is an ASE calculator (one call per structure) or anything with
    ``evaluate(structures)`` such as :class:`BatchedMetatomicTeacher`. All
    monomers, dimers and dimer fragments go to the teacher in one request.
    """
    mode = str(energy_mode).strip().lower()
    if mode not in ENERGY_MODES:
        raise ValueError(f"energy_mode must be one of {ENERGY_MODES}, got {energy_mode!r}")
    for geo in geometries:
        if geo.kind not in ("monomer", "dimer"):
            raise ValueError(f"unsupported geometry kind {geo.kind!r}")

    # Flat request: monomer -> [whole]; dimer -> [AB, A, B] (fragments only
    # needed for interaction labels, but E_int is stored in both modes).
    structures: list[tuple[np.ndarray, np.ndarray]] = []
    slots: list[int] = []
    for geo in geometries:
        slots.append(len(structures))
        structures.append((geo.numbers, geo.positions))
        if geo.kind == "dimer":
            z_a, r_a, z_b, r_b = _split_dimer(geo)
            structures.extend([(z_a, r_a), (z_b, r_b)])
    results = _as_teacher(teacher).evaluate(structures)
    if len(results) != len(structures):
        raise RuntimeError(f"teacher returned {len(results)} results for {len(structures)}")

    e_ref = 0.0
    if mode == ENERGY_MODE_INTERACTION:
        for geo, slot in zip(geometries, slots):
            if geo.kind == "monomer" and geo.source == "pdb_eq":
                e_ref = results[slot][0]
                break

    labeled: list[LabeledSample] = []
    for geo, slot in zip(geometries, slots):
        e_tot, f_tot = results[slot]
        if geo.kind == "monomer":
            energy = e_tot if mode == ENERGY_MODE_TOTAL else (e_tot - e_ref)
            labeled.append(
                LabeledSample(
                    geometry=geo,
                    energy_eV=float(energy),
                    forces_ev_per_angstrom=np.asarray(f_tot, dtype=np.float64),
                    energy_total_eV=float(e_tot),
                    energy_int_eV=None,
                )
            )
            continue
        (e_a, f_a), (e_b, f_b) = results[slot + 1], results[slot + 2]
        n_a = f_a.shape[0]
        e_int = float(e_tot - e_a - e_b)
        f_int = np.concatenate([f_tot[:n_a] - f_a, f_tot[n_a:] - f_b], axis=0)
        if mode == ENERGY_MODE_TOTAL:
            energy, forces = float(e_tot), f_tot
        else:
            energy, forces = e_int, f_int
        labeled.append(
            LabeledSample(
                geometry=geo,
                energy_eV=float(energy),
                forces_ev_per_angstrom=np.asarray(forces, dtype=np.float64),
                energy_total_eV=float(e_tot),
                energy_int_eV=e_int,
            )
        )
    return labeled
