"""Calculator-neutral monomer/dimer ML hybrid (the MMML ML/MM fragment scheme).

Used by metatomic (and any other ASE calculator) to reproduce the PhysNet
MLpot split without JAX:

* ``do_ml``: isolated-monomer energies (PBC off).
* ``do_ml_dimer``: close-pair interaction ``E(AB) - E(A) - E(B)``, with exact
  MIC wrap of monomer B (not differentiated) and the canonical ML handoff
  ``ml_switch_scale(r_com)``.
* Optional intermolecular MM is supplied by the caller (JAX MM or CHARMM),
  not computed here.

Units: energy eV, forces eV/Å, positions Å.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    GAMMA_ON,
)

METATOMIC_EVAL_MODES = ("fragments", "whole_system")
_SWITCH_DEN_FLOOR = 1.0e-12


def monomer_offsets_from_counts(atoms_per_monomer: Sequence[int]) -> np.ndarray:
    """Inclusive-exclusive offsets; length is ``n_monomers + 1``."""
    offsets = [0]
    for n in atoms_per_monomer:
        offsets.append(offsets[-1] + int(n))
    return np.asarray(offsets, dtype=np.int32)


def _cell_as_matrix(cell: np.ndarray | float | None) -> np.ndarray | None:
    if cell is None:
        return None
    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 0:
        side = float(arr)
        if side <= 0.0:
            return None
        return np.eye(3, dtype=np.float64) * side
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"cell must be scalar, (3,), or (3, 3); got shape {arr.shape}")


def wrap_dimer_monomer_b_numpy(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Exact MIC lattice shift of monomer B COM relative to A (MD-safe, not dE/dshift)."""
    cell_m = _cell_as_matrix(cell)
    if cell_m is None:
        return np.asarray(pos_b, dtype=np.float64)
    ra = np.asarray(pos_a, dtype=np.float64)
    rb = np.asarray(pos_b, dtype=np.float64)
    com_a = ra.mean(axis=0)
    com_b = rb.mean(axis=0)
    delta = com_b - com_a
    # Row-vector convention (ASE / pbc_utils_jax.frac_coords): S = solve(cell.T, r).
    frac = np.linalg.solve(cell_m.T, delta)
    frac -= np.round(frac)
    mic = frac @ cell_m
    shift = (com_a + mic) - com_b
    return rb + shift


def numpy_ml_switch_scale(
    r_com: float,
    *,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    ml_switch_width: float = DEFAULT_ML_SWITCH_WIDTH,
    gamma: float = GAMMA_ON,
) -> float:
    """NumPy twin of :func:`ml_switch_scale` (1 inside ML region, 0 past ``mm_switch_on``)."""
    value, _deriv = numpy_ml_switch_scale_and_deriv(
        r_com,
        mm_switch_on=mm_switch_on,
        ml_switch_width=ml_switch_width,
        gamma=gamma,
    )
    return value


def numpy_ml_switch_scale_and_deriv(
    r_com: float,
    *,
    mm_switch_on: float,
    ml_switch_width: float,
    gamma: float = GAMMA_ON,
) -> tuple[float, float]:
    """Return ``(s, ds/dr)`` for the ML dimer handoff."""
    r = float(r_com)
    x0 = float(mm_switch_on) - float(ml_switch_width)
    x1 = float(mm_switch_on)
    width = x1 - x0
    if abs(width) < _SWITCH_DEN_FLOOR:
        s = 0.0 if r >= x1 else 1.0
        return s, 0.0
    t = (r - x0) / width
    if t <= 0.0:
        return 1.0, 0.0
    if t >= 1.0:
        return 0.0, 0.0
    t_g = t ** float(gamma)
    # smoothstep01 = 6t^5 - 15t^4 + 10t^3; d/dt = 30 t^2 (t-1)^2
    sharp = t_g * t_g * t_g * (10.0 + t_g * (-15.0 + 6.0 * t_g))
    dsharp_dtg = 30.0 * t_g * t_g * (t_g - 1.0) * (t_g - 1.0)
    dtg_dt = float(gamma) * (t ** (float(gamma) - 1.0)) if t > 0.0 else 0.0
    dt_dr = 1.0 / width
    ds_dr = -dsharp_dtg * dtg_dt * dt_dr
    return float(1.0 - sharp), float(ds_dr)


def _eval_atoms(
    atoms: Atoms,
    calculator: Calculator,
) -> tuple[float, np.ndarray]:
    eval_atoms = atoms.copy()
    eval_atoms.calc = calculator
    energy = float(eval_atoms.get_potential_energy())
    forces = np.asarray(eval_atoms.get_forces(), dtype=np.float64)
    return energy, forces


def _slice_atoms(
    numbers: np.ndarray,
    positions: np.ndarray,
    start: int,
    stop: int,
) -> Atoms:
    return Atoms(
        numbers=np.asarray(numbers[start:stop], dtype=int),
        positions=np.asarray(positions[start:stop], dtype=np.float64),
        pbc=False,
    )


@dataclass(frozen=True)
class FragmentHybridResult:
    """Hybrid ML fragment energy/forces in eV / eV/Å."""

    energy_ev: float
    forces_ev_per_angstrom: np.ndarray
    n_monomers_evaluated: int
    n_dimers_evaluated: int
    eval_mode: str


def evaluate_whole_system(
    calculator: Calculator,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    *,
    cell: np.ndarray | float | None = None,
) -> FragmentHybridResult:
    """One ASE evaluation on the full ML selection."""
    numbers = np.asarray(atomic_numbers, dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    atoms = Atoms(numbers=numbers, positions=pos)
    cell_m = _cell_as_matrix(cell)
    if cell_m is not None:
        atoms.set_cell(cell_m)
        atoms.set_pbc(True)
    energy, forces = _eval_atoms(atoms, calculator)
    return FragmentHybridResult(
        energy_ev=energy,
        forces_ev_per_angstrom=forces,
        n_monomers_evaluated=1,
        n_dimers_evaluated=0,
        eval_mode="whole_system",
    )


def evaluate_fragment_hybrid(
    calculator: Calculator,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    atoms_per_monomer: Sequence[int],
    *,
    do_ml: bool = True,
    do_ml_dimer: bool = True,
    cell: np.ndarray | float | None = None,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    ml_switch_width: float = DEFAULT_ML_SWITCH_WIDTH,
) -> FragmentHybridResult:
    """Monomer sum plus switched dimer interaction (MMML ML/MM scheme, ML part)."""
    numbers = np.asarray(atomic_numbers, dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    per = [int(n) for n in atoms_per_monomer]
    offsets = monomer_offsets_from_counts(per)
    n_atoms = int(offsets[-1])
    if numbers.shape[0] != n_atoms or pos.shape[0] != n_atoms:
        raise ValueError(
            f"atom count {pos.shape[0]} != sum(atoms_per_monomer)={n_atoms}"
        )
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    energy = 0.0
    n_monomers_eval = 0
    n_dimers_eval = 0
    monomer_energies: list[float] = [0.0] * len(per)
    monomer_forces: list[np.ndarray] = []

    if do_ml:
        for i, n_i in enumerate(per):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            frag = _slice_atoms(numbers, pos, start, stop)
            e_i, f_i = _eval_atoms(frag, calculator)
            energy += e_i
            forces[start:stop] += f_i
            monomer_energies[i] = e_i
            monomer_forces.append(f_i)
            n_monomers_eval += 1
    else:
        monomer_forces = [
            np.zeros((n_i, 3), dtype=np.float64) for n_i in per
        ]

    if do_ml_dimer and len(per) > 1:
        cell_m = _cell_as_matrix(cell)
        n_mon = len(per)
        for i in range(n_mon):
            start_i, stop_i = int(offsets[i]), int(offsets[i + 1])
            pos_i = pos[start_i:stop_i]
            for j in range(i + 1, n_mon):
                start_j, stop_j = int(offsets[j]), int(offsets[j + 1])
                pos_j = pos[start_j:stop_j]
                if cell_m is not None:
                    pos_j_eval = wrap_dimer_monomer_b_numpy(pos_i, pos_j, cell_m)
                else:
                    pos_j_eval = pos_j
                com_i = pos_i.mean(axis=0)
                com_j = pos_j_eval.mean(axis=0)
                delta = com_j - com_i
                r_com = float(np.linalg.norm(delta))
                scale, dscale_dr = numpy_ml_switch_scale_and_deriv(
                    r_com,
                    mm_switch_on=mm_switch_on,
                    ml_switch_width=ml_switch_width,
                )
                if scale == 0.0 and dscale_dr == 0.0:
                    continue
                dimer = Atoms(
                    numbers=np.concatenate([numbers[start_i:stop_i], numbers[start_j:stop_j]]),
                    positions=np.concatenate([pos_i, pos_j_eval], axis=0),
                    pbc=False,
                )
                e_ab, f_ab = _eval_atoms(dimer, calculator)
                if cell_m is None and do_ml:
                    e_i = monomer_energies[i]
                    f_i = monomer_forces[i]
                    e_j = monomer_energies[j]
                    f_j = monomer_forces[j]
                else:
                    if not do_ml:
                        frag_i = _slice_atoms(numbers, pos, start_i, stop_i)
                        e_i, f_i = _eval_atoms(frag_i, calculator)
                    else:
                        e_i = monomer_energies[i]
                        f_i = monomer_forces[i]
                    frag_j = Atoms(
                        numbers=numbers[start_j:stop_j],
                        positions=pos_j_eval,
                        pbc=False,
                    )
                    e_j, f_j = _eval_atoms(frag_j, calculator)
                e_int = e_ab - e_i - e_j
                n_i = int(per[i])
                f_int_i = f_ab[:n_i] - f_i
                f_int_j = f_ab[n_i:] - f_j
                energy += scale * e_int
                forces[start_i:stop_i] += scale * f_int_i
                forces[start_j:stop_j] += scale * f_int_j
                if dscale_dr != 0.0 and r_com > _SWITCH_DEN_FLOOR:
                    rhat = delta / r_com
                    # F = -d(s E)/dR includes -E ds/dR; ds/dR_a = ds/dr * (-rhat/n_a)
                    coeff = -e_int * dscale_dr
                    forces[start_i:stop_i] += coeff * (-rhat) / float(n_i)
                    forces[start_j:stop_j] += coeff * (rhat) / float(per[j])
                n_dimers_eval += 1

    return FragmentHybridResult(
        energy_ev=float(energy),
        forces_ev_per_angstrom=forces,
        n_monomers_evaluated=n_monomers_eval,
        n_dimers_evaluated=n_dimers_eval,
        eval_mode="fragments",
    )


class AseFragmentHybridCalculator(Calculator):
    """ASE calculator: fragment ML hybrid, optional intermolecular MM calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        calculator_factory: Callable[[], Calculator],
        atoms_per_monomer: Sequence[int],
        *,
        eval_mode: str = "fragments",
        do_ml: bool = True,
        do_ml_dimer: bool = True,
        mm_calculator: Calculator | None = None,
        mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
        ml_switch_width: float = DEFAULT_ML_SWITCH_WIDTH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mode = str(eval_mode).strip().lower()
        if mode not in METATOMIC_EVAL_MODES:
            raise ValueError(
                f"eval_mode must be one of {METATOMIC_EVAL_MODES}; got {eval_mode!r}"
            )
        self._factory = calculator_factory
        self._calc = calculator_factory()
        self.atoms_per_monomer = [int(n) for n in atoms_per_monomer]
        self.eval_mode = mode
        self.do_ml = bool(do_ml)
        self.do_ml_dimer = bool(do_ml_dimer)
        self.mm_calculator = mm_calculator
        self.mm_switch_on = float(mm_switch_on)
        self.ml_switch_width = float(ml_switch_width)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("Atoms object is required")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64)
        cell = np.asarray(atoms.cell.array, dtype=np.float64) if atoms.pbc.any() else None
        if self.eval_mode == "whole_system":
            result = evaluate_whole_system(
                self._calc, numbers, positions, cell=cell
            )
        else:
            result = evaluate_fragment_hybrid(
                self._calc,
                numbers,
                positions,
                self.atoms_per_monomer,
                do_ml=self.do_ml,
                do_ml_dimer=self.do_ml_dimer,
                cell=cell,
                mm_switch_on=self.mm_switch_on,
                ml_switch_width=self.ml_switch_width,
            )
        energy = result.energy_ev
        forces = result.forces_ev_per_angstrom
        if self.mm_calculator is not None:
            mm_atoms = atoms.copy()
            mm_atoms.calc = self.mm_calculator
            energy += float(mm_atoms.get_potential_energy())
            forces = forces + np.asarray(mm_atoms.get_forces(), dtype=np.float64)
        self.results = {"energy": float(energy), "forces": forces}
