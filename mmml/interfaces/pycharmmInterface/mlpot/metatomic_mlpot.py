"""Metatomic CHARMM MLpot: USER callback + MMML fragment ML/MM scheme.

PyCHARMM only requires ``get_pycharmm_calculator()`` and Fortran-shaped
``calculate_charmm`` (kcal/mol, forces into ``dx/dy/dz``). This adapter fills
that slot with an ASE metatomic model.

ML/MM scheme (``eval_mode=fragments``):
  USER += isolated-monomer metatomic + switched dimer interaction
  (``E(AB)-E(A)-E(B)``). Optional JAX MM from an MM-only ``setup_calculator``
  spherical_fn is added when ``do_mm`` is true.

All-ML (``eval_mode=whole_system``): one metatomic evaluation on the ML
selection; CHARMM ELEC/VDW should be zeroed by the existing jax_mic energy
policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator

from mmml.data.units import EV_TO_KCAL_MOL
from mmml.interfaces.calculators.ase_fragment_hybrid import (
    METATOMIC_EVAL_MODES,
    FragmentHybridResult,
    evaluate_fragment_hybrid,
    evaluate_whole_system,
)
from mmml.interfaces.calculators.metatomic import (
    is_metatomic_checkpoint,
    load_metatomic_calculator,
)
from mmml.interfaces.pycharmmInterface.cutoffs import (
    CutoffParameters,
    cutoff_parameters_from_args,
)

MmEnergyFn = Callable[..., tuple[float, np.ndarray]]


def resolve_metatomic_eval_mode(
    args: Any | None = None,
    *,
    explicit: str | None = None,
) -> str:
    """``fragments`` (ML/MM scheme) or ``whole_system`` (all-ML USER)."""
    raw = explicit
    if raw is None and args is not None:
        raw = getattr(args, "metatomic_eval_mode", None)
    mode = str(raw or "fragments").strip().lower().replace("-", "_")
    if mode not in METATOMIC_EVAL_MODES:
        raise ValueError(
            f"metatomic_eval_mode must be one of {METATOMIC_EVAL_MODES}; got {raw!r}"
        )
    return mode


class MetatomicMlpotCalculator:
    """CHARMM ``calculate_charmm`` adapter around a metatomic ASE calculator."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        atomic_numbers: np.ndarray,
        atoms_per_monomer: Sequence[int],
        eval_mode: str = "fragments",
        do_ml: bool = True,
        do_ml_dimer: bool = True,
        do_mm: bool = False,
        mm_energy_forces_fn: MmEnergyFn | None = None,
        get_update_fn: Any | None = None,
        cutoff_params: CutoffParameters | None = None,
        cell: float | bool = False,
        ml_atom_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self._calc = calculator
        self.atomic_numbers = np.asarray(
            [int(x) for x in atomic_numbers], dtype=np.int32
        )
        self._atoms_per_monomer = [int(n) for n in atoms_per_monomer]
        self.eval_mode = resolve_metatomic_eval_mode(explicit=eval_mode)
        self.do_ml = bool(do_ml)
        self.do_ml_dimer = bool(do_ml_dimer)
        self.do_mm = bool(do_mm)
        self._mm_energy_forces_fn = mm_energy_forces_fn
        self._get_update_fn = get_update_fn
        self.cutoff_params = cutoff_params or CutoffParameters()
        self._cell = float(cell) if cell else False
        if ml_atom_indices is None:
            self._ml_atom_indices: np.ndarray | None = None
        else:
            self._ml_atom_indices = np.asarray(ml_atom_indices, dtype=int).reshape(-1)
        self.last_ml_forces: np.ndarray | None = None
        self._last_ml_forces: np.ndarray | None = None
        self.n_monomers = len(self._atoms_per_monomer)

    def _resolve_ml_slice(self, n_charmm: int) -> np.ndarray:
        expected = int(self.atomic_numbers.shape[0])
        stored = self._ml_atom_indices
        if stored is not None:
            return np.asarray(stored, dtype=int).reshape(-1)
        if int(n_charmm) == expected:
            return np.arange(expected, dtype=int)
        raise RuntimeError(
            f"Metatomic MLpot: CHARMM Natom={int(n_charmm)} != n_ml={expected} "
            "and ml_atom_indices was not set."
        )

    def _evaluate_ml(
        self,
        pos_ml: np.ndarray,
        box_side: float | None,
    ) -> FragmentHybridResult:
        cell = box_side if box_side is not None and box_side > 0.0 else None
        if self.eval_mode == "whole_system":
            return evaluate_whole_system(
                self._calc,
                self.atomic_numbers,
                pos_ml,
                cell=cell,
            )
        return evaluate_fragment_hybrid(
            self._calc,
            self.atomic_numbers,
            pos_ml,
            self._atoms_per_monomer,
            do_ml=self.do_ml,
            do_ml_dimer=self.do_ml_dimer,
            cell=cell,
            mm_switch_on=float(self.cutoff_params.mm_switch_on),
            ml_switch_width=float(self.cutoff_params.ml_switch_width),
        )

    def _evaluate_mm(
        self,
        pos_ml: np.ndarray,
        box_side: float | None,
    ) -> tuple[float, np.ndarray]:
        n = int(pos_ml.shape[0])
        zeros = np.zeros((n, 3), dtype=np.float64)
        if not self.do_mm or self._mm_energy_forces_fn is None:
            return 0.0, zeros
        mm_fn = self._mm_energy_forces_fn
        update_fn = self._get_update_fn
        if update_fn is not None:
            box = None
            if box_side is not None:
                box = np.asarray([box_side, box_side, box_side], dtype=np.float64)
            pair_idx, pair_mask = (
                update_fn(pos_ml, box=box) if box is not None else update_fn(pos_ml)
            )
            energy, forces = mm_fn(pos_ml, pair_idx, pair_mask)
        else:
            energy, forces = mm_fn(pos_ml)
        return float(np.asarray(energy).reshape(-1)[0]), np.asarray(forces, dtype=np.float64)

    def calculate_charmm(
        self,
        Natom: int,
        Ntrans: int,
        Natim: int,
        idxp,
        x,
        y,
        z,
        dx,
        dy,
        dz,
        Nmlp: int,
        Nmlmmp: int,
        idxi,
        idxj,
        idxjp,
        idxu,
        idxv,
        idxup,
        idxvp,
    ) -> float:
        """CHARMM USER energy in kcal/mol; accumulate kcal/mol/Å into ``dx/dy/dz``."""
        del Ntrans, Natim, idxp, Nmlp, Nmlmmp, idxi, idxj, idxjp, idxu, idxv, idxup, idxvp
        n = int(Natom)
        pos_full = np.array([x[:n], y[:n], z[:n]], dtype=np.float64).T
        ml_idx = self._resolve_ml_slice(n)
        pos_ml = pos_full[ml_idx]
        box_side = float(self._cell) if self._cell else None
        ml = self._evaluate_ml(pos_ml, box_side)
        e_mm, f_mm = self._evaluate_mm(pos_ml, box_side)
        energy_ev = ml.energy_ev + e_mm
        forces_ev = ml.forces_ev_per_angstrom + f_mm
        energy_kcal = float(energy_ev) * EV_TO_KCAL_MOL
        forces_kcal = np.asarray(forces_ev, dtype=np.float64) * EV_TO_KCAL_MOL
        self.last_ml_forces = forces_kcal
        self._last_ml_forces = forces_kcal
        for local_i, atom_i in enumerate(ml_idx):
            ai = int(atom_i)
            dx[ai] -= float(forces_kcal[local_i, 0])
            dy[ai] -= float(forces_kcal[local_i, 1])
            dz[ai] -= float(forces_kcal[local_i, 2])
        return energy_kcal


class MetatomicMlpotModel:
    """``pycharmm.MLpot`` handle: ``get_pycharmm_calculator`` contract."""

    def __init__(
        self,
        calculator: Calculator,
        *,
        atomic_numbers: np.ndarray,
        atoms_per_monomer: Sequence[int],
        eval_mode: str = "fragments",
        do_ml: bool = True,
        do_ml_dimer: bool = True,
        do_mm: bool = False,
        mm_energy_forces_fn: MmEnergyFn | None = None,
        get_update_fn: Any | None = None,
        cutoff_params: CutoffParameters | None = None,
        cell: float | bool = False,
        checkpoint: Path | None = None,
    ) -> None:
        self._calc = calculator
        self._atomic_numbers = np.asarray(atomic_numbers, dtype=int)
        self._atoms_per_monomer = [int(n) for n in atoms_per_monomer]
        self._eval_mode = resolve_metatomic_eval_mode(explicit=eval_mode)
        self._do_ml = bool(do_ml)
        self._do_ml_dimer = bool(do_ml_dimer)
        self._do_mm = bool(do_mm)
        self._mm_energy_forces_fn = mm_energy_forces_fn
        self._get_update_fn = get_update_fn
        self._cutoff_params = cutoff_params or CutoffParameters()
        self._cell = float(cell) if cell else False
        self._ml_atom_indices: np.ndarray | None = None
        self._registered_calculator: MetatomicMlpotCalculator | None = None
        self.checkpoint = checkpoint
        self._n_monomers = len(self._atoms_per_monomer)
        self._last_ml_forces: np.ndarray | None = None

    def set_cell(self, side: float | bool) -> None:
        """Update the cubic MIC cell on the model and any registered calculator."""
        self._cell = float(side) if side else False
        calc = self._registered_calculator
        if calc is not None:
            calc._cell = self._cell

    def get_pycharmm_calculator(
        self,
        ml_atom_indices=None,
        ml_atomic_numbers=None,
        **kwargs,
    ) -> MetatomicMlpotCalculator:
        del kwargs
        if ml_atom_indices is not None:
            self._ml_atom_indices = np.asarray(ml_atom_indices, dtype=int).reshape(-1)
        numbers = self._atomic_numbers
        if ml_atomic_numbers is not None:
            numbers = np.asarray(ml_atomic_numbers, dtype=int)
        calc = MetatomicMlpotCalculator(
            self._calc,
            atomic_numbers=numbers,
            atoms_per_monomer=self._atoms_per_monomer,
            eval_mode=self._eval_mode,
            do_ml=self._do_ml,
            do_ml_dimer=self._do_ml_dimer,
            do_mm=self._do_mm,
            mm_energy_forces_fn=self._mm_energy_forces_fn,
            get_update_fn=self._get_update_fn,
            cutoff_params=self._cutoff_params,
            cell=self._cell,
            ml_atom_indices=self._ml_atom_indices,
        )
        self._registered_calculator = calc
        return calc


def _maybe_build_mm_only_spherical(
    *,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    atomic_numbers: np.ndarray,
    checkpoint: Path,
    cell: float | bool,
    cutoff_params: CutoffParameters,
    args: Any | None,
    do_mm: bool,
    verbose: bool,
) -> tuple[Any | None, Any | None]:
    """MM-only ``setup_calculator`` spherical_fn; ML stays in the ASE adapter."""
    if not do_mm:
        return None, None
    from mmml.interfaces.pycharmmInterface.calculator_utils import unpack_factory_result
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    factory = setup_calculator(
        list(atoms_per_monomer),
        N_MONOMERS=int(n_monomers),
        doML=False,
        doMM=True,
        doML_dimer=False,
        cell=cell,
        verbose=verbose,
        model_restart_path=checkpoint,
        ml_potential_mode="metatomic",
        mm_atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        ml_switch_width=cutoff_params.ml_switch_width,
        mm_switch_on=cutoff_params.mm_switch_on,
        mm_switch_width=cutoff_params.mm_switch_width,
        complementary_handoff=cutoff_params.complementary_handoff,
        hybrid_hamiltonian=cutoff_params.hybrid_hamiltonian,
        jax_mm_spoof_psf=(
            getattr(args, "jax_mm_spoof_psf", None) if args is not None else None
        ),
    )
    z = np.asarray(atomic_numbers, dtype=int)
    r0 = np.zeros((len(z), 3), dtype=np.float64)
    import jax.numpy as jnp

    _, spherical_fn, get_update_fn = unpack_factory_result(
        factory(
            atomic_numbers=jnp.asarray(z),
            atomic_positions=jnp.asarray(r0),
            n_monomers=int(n_monomers),
            cutoff_params=cutoff_params,
            doML=False,
            doMM=True,
            doML_dimer=False,
            backprop=False,
            create_ase_calculator=False,
        )
    )
    return spherical_fn, get_update_fn


def _mm_fn_from_spherical(spherical_fn: Any, cutoff_params: CutoffParameters) -> MmEnergyFn:
    def mm_fn(positions, pair_idx=None, pair_mask=None):
        import jax.numpy as jnp

        kwargs: dict[str, Any] = dict(
            positions=jnp.asarray(positions),
            doML=False,
            doMM=True,
            doML_dimer=False,
            cutoff_params=cutoff_params,
        )
        if pair_idx is not None:
            kwargs["mm_pair_idx"] = pair_idx
            kwargs["mm_pair_mask"] = pair_mask
        out = spherical_fn(**kwargs)
        energy = float(np.asarray(out.energy).reshape(-1)[0])
        forces = np.asarray(out.forces, dtype=np.float64)
        return energy, forces

    return mm_fn


def build_metatomic_mlpot_model(
    checkpoint: Path | str,
    atomic_numbers: np.ndarray,
    atoms_per_monomer: Sequence[int],
    n_monomers: int,
    *,
    cell: float | bool = False,
    verbose: bool = False,
    args: Any | None = None,
    calculator: Calculator | None = None,
    eval_mode: str | None = None,
    do_ml: bool = True,
    do_ml_dimer: bool = True,
    do_mm: bool = True,
) -> MetatomicMlpotModel:
    """Build a CHARMM-registerable metatomic MLpot model (ASE ML + optional JAX MM)."""
    ckpt = Path(checkpoint).expanduser().resolve()
    z = np.asarray(atomic_numbers, dtype=int)
    per = [int(x) for x in atoms_per_monomer]
    if len(per) != int(n_monomers):
        raise ValueError(
            f"atoms_per_monomer length {len(per)} != n_monomers={n_monomers}"
        )
    mode = resolve_metatomic_eval_mode(args, explicit=eval_mode)
    calc = calculator if calculator is not None else load_metatomic_calculator(ckpt)
    cutoff_params = (
        cutoff_parameters_from_args(args) if args is not None else CutoffParameters()
    )
    mm_fn: MmEnergyFn | None = None
    get_update_fn = None
    if do_mm:
        try:
            spherical_fn, get_update_fn = _maybe_build_mm_only_spherical(
                atoms_per_monomer=per,
                n_monomers=int(n_monomers),
                atomic_numbers=z,
                checkpoint=ckpt,
                cell=cell,
                cutoff_params=cutoff_params,
                args=args,
                do_mm=True,
                verbose=verbose,
            )
            if spherical_fn is not None:
                mm_fn = _mm_fn_from_spherical(spherical_fn, cutoff_params)
        except Exception as exc:
            if verbose:
                print(
                    f"Metatomic MLpot: JAX MM spherical_fn unavailable ({exc!r}); "
                    "USER term is ML-only. Keep CHARMM ELEC/VDW or pass do_mm=False.",
                    flush=True,
                )
            mm_fn = None
            get_update_fn = None
    if verbose:
        print(
            f"Metatomic MLpot: eval_mode={mode} do_ml={do_ml} "
            f"do_ml_dimer={do_ml_dimer} do_mm={do_mm and mm_fn is not None} "
            f"checkpoint={ckpt}",
            flush=True,
        )
    return MetatomicMlpotModel(
        calc,
        atomic_numbers=z,
        atoms_per_monomer=per,
        eval_mode=mode,
        do_ml=do_ml,
        do_ml_dimer=do_ml_dimer,
        do_mm=bool(do_mm and mm_fn is not None),
        mm_energy_forces_fn=mm_fn,
        get_update_fn=get_update_fn,
        cutoff_params=cutoff_params,
        cell=cell,
        checkpoint=ckpt,
    )


def should_use_metatomic_mlpot(
    checkpoint: Path | str | None,
    args: Any | None = None,
) -> bool:
    """True when CLI/checkpoint selects the metatomic CHARMM MLpot path."""
    mode = str(getattr(args, "ml_potential_mode", "") or "").strip().lower() if args else ""
    if mode in {"metatomic", "metatensor"}:
        return True
    return is_metatomic_checkpoint(checkpoint)
