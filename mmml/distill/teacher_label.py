"""Label geometries with an ASE teacher (PET-MAD or a dummy calculator).

Default ``energy_mode=interaction`` writes the MMML hybrid pieces:

* monomers: ``E = E_teacher - E_ref`` (``E_ref`` is the first ``pdb_eq`` monomer)
* dimers: unswitched ``E = E(AB) - E(A) - E(B)`` and matching forces

Do **not** bake ``ml_switch_scale`` into the labels; MLpot applies the handoff
at MD time. Units: energy eV, forces eV/Å (ASE / metatomic).
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _eval(calc: Calculator, numbers: np.ndarray, positions: np.ndarray) -> tuple[float, np.ndarray]:
    out = evaluate_whole_system(calc, numbers, positions)
    return float(out.energy_ev), np.asarray(out.forces_ev_per_angstrom, dtype=np.float64)


def _split_dimer(geo: Geometry) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_a, n_b = geo.atoms_per_monomer
    z = geo.numbers
    r = geo.positions
    return z[:n_a], r[:n_a], z[n_a : n_a + n_b], r[n_a : n_a + n_b]


def label_geometries(
    calculator: Calculator,
    geometries: list[Geometry],
    *,
    energy_mode: str = ENERGY_MODE_INTERACTION,
) -> list[LabeledSample]:
    """Evaluate the teacher on each geometry. ``energy_mode`` selects the stored E/F."""
    mode = str(energy_mode).strip().lower()
    if mode not in ENERGY_MODES:
        raise ValueError(f"energy_mode must be one of {ENERGY_MODES}, got {energy_mode!r}")

    e_ref = 0.0
    if mode == ENERGY_MODE_INTERACTION:
        for geo in geometries:
            if geo.kind == "monomer" and geo.source == "pdb_eq":
                e_ref, _ = _eval(calculator, geo.numbers, geo.positions)
                break
    labeled: list[LabeledSample] = []
    for geo in geometries:
        if geo.kind == "monomer":
            e_tot, f_tot = _eval(calculator, geo.numbers, geo.positions)
            energy = e_tot if mode == ENERGY_MODE_TOTAL else (e_tot - e_ref)
            labeled.append(
                LabeledSample(
                    geometry=geo,
                    energy_eV=float(energy),
                    forces_ev_per_angstrom=f_tot,
                    energy_total_eV=float(e_tot),
                    energy_int_eV=None,
                )
            )
            continue
        if geo.kind != "dimer":
            raise ValueError(f"unsupported geometry kind {geo.kind!r}")
        z_a, r_a, z_b, r_b = _split_dimer(geo)
        e_ab, f_ab = _eval(calculator, geo.numbers, geo.positions)
        e_a, f_a = _eval(calculator, z_a, r_a)
        e_b, f_b = _eval(calculator, z_b, r_b)
        e_int = float(e_ab - e_a - e_b)
        f_int = np.concatenate([f_ab[: z_a.shape[0]] - f_a, f_ab[z_a.shape[0] :] - f_b], axis=0)
        if mode == ENERGY_MODE_TOTAL:
            energy, forces = float(e_ab), f_ab
        else:
            energy, forces = e_int, f_int
        labeled.append(
            LabeledSample(
                geometry=geo,
                energy_eV=float(energy),
                forces_ev_per_angstrom=np.asarray(forces, dtype=np.float64),
                energy_total_eV=float(e_ab),
                energy_int_eV=e_int,
            )
        )
    return labeled
