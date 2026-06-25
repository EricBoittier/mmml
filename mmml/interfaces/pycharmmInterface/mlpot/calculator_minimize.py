"""Hybrid JAX calculator minimization on CHARMM coordinates before MLpot SD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    mlpot_hybrid_grms_from_calculator,
    mlpot_spherical_energy_forces_ev_angstrom,
)


@dataclass
class HybridCalculatorMinimizeConfig:
    """ASE BFGS on the full hybrid calculator (synced to CHARMM coords)."""

    max_steps: int = 200
    fmax_ev_a: float = 0.05
    bfgs_maxstep: float = 0.05
    verbose: bool = True
    quiet_bfgs: bool = False
    use_bfgs_line_search: bool = True
    fmax_spike_factor: float = 4.0
    fmax_spike_floor_ev_a: float = 15.0


def spike_fmax_limit_ev_a(
    initial_fmax_ev_a: float,
    *,
    factor: float,
    floor_ev_a: float,
) -> float:
    """Abort BFGS when max|F| exceeds this limit (guards catastrophic steps)."""
    if not np.isfinite(initial_fmax_ev_a) or initial_fmax_ev_a <= 0.0:
        return float(floor_ev_a)
    return float(max(initial_fmax_ev_a * float(factor), float(floor_ev_a)))


class _BestMinimizationFrame:
    """Track lowest-fmax geometry during ASE minimization (see jaxmd pre-min)."""

    def __init__(self, atoms: Any) -> None:
        self.atoms = atoms
        self.best_force_positions: np.ndarray | None = None
        self.best_force_fmax = float("inf")
        self.best_force_label = ""

    def record(self, label: str) -> None:
        try:
            fmax = float(np.abs(self.atoms.get_forces()).max())
        except Exception:
            return
        if not np.isfinite(fmax):
            return
        positions = np.asarray(self.atoms.get_positions(), dtype=np.float64).copy()
        if fmax < self.best_force_fmax:
            self.best_force_positions = positions
            self.best_force_fmax = fmax
            self.best_force_label = label

    def restore_best_force(self) -> float:
        if self.best_force_positions is not None:
            self.atoms.set_positions(self.best_force_positions)
        return float(np.abs(self.atoms.get_forces()).max())


def _hybrid_mlpot_ase_calculator_class():
    import ase.calculators.calculator as ase_calc

    class HybridMlpotAseCalculator(ase_calc.Calculator):
        """Minimal ASE calculator wrapping ``spherical_fn`` for BFGS pre-minimization."""

        implemented_properties = ["energy", "forces"]

        def __init__(self, mlpot_ctx: Any) -> None:
            super().__init__()
            self.mlpot_ctx = mlpot_ctx
            pyCModel = getattr(mlpot_ctx, "pyCModel", None)
            if pyCModel is None:
                raise RuntimeError("HybridMlpotAseCalculator requires mlpot_ctx.pyCModel")
            self._pyCModel = pyCModel
            self.use_pbc = bool(getattr(mlpot_ctx, "use_pbc", False))
            box = getattr(mlpot_ctx, "cubic_box_side_A", None)
            if box is None:
                box = getattr(mlpot_ctx, "charmm_cubic_box_side_A", None)
            self._box_A = float(box) if box is not None else None

        def calculate(
            self,
            atoms,
            properties,
            system_changes,
        ) -> None:
            super().calculate(atoms, properties, system_changes)
            pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            evald = mlpot_spherical_energy_forces_ev_angstrom(
                self._pyCModel,
                positions=pos,
                use_pbc=self.use_pbc,
                box_A=self._box_A,
            )
            if evald is None:
                raise RuntimeError(
                    "hybrid spherical_fn evaluation failed during calculator mini"
                )
            energy_ev, forces_ev = evald
            self.results["energy"] = float(energy_ev)
            self.results["forces"] = np.asarray(forces_ev, dtype=np.float64)

    return HybridMlpotAseCalculator


def _promote_mlpot_jax_for_calculator_mini(mlpot_ctx: Any, *, verbose: bool) -> None:
    pyCModel = getattr(mlpot_ctx, "pyCModel", None)
    promote = getattr(pyCModel, "promote_jax_factory_to_gpu", None)
    if callable(promote):
        promote()
        if verbose:
            print(
                "Pre-SD hybrid calculator minimize: promoted JAX factory to GPU",
                flush=True,
            )
        return
    from mmml.utils.jax_gpu_warmup import ensure_xla_gpu_warmed

    ensure_xla_gpu_warmed(force=True)


def _run_hybrid_calculator_bfgs(
    atoms: Any,
    config: HybridCalculatorMinimizeConfig,
    *,
    context_prefix: str,
    initial_fmax_ev_a: float,
) -> tuple[Any, _BestMinimizationFrame, bool]:
    """Run guarded ASE BFGS; returns (optimizer, best_frame, stopped_on_spike)."""
    if config.use_bfgs_line_search:
        from ase.optimize.bfgslinesearch import BFGSLineSearch as BfgsOptimizer
    else:
        from ase.optimize import BFGS as BfgsOptimizer

    best_frame = _BestMinimizationFrame(atoms)
    best_frame.record("initial")
    spike_limit = spike_fmax_limit_ev_a(
        initial_fmax_ev_a,
        factor=config.fmax_spike_factor,
        floor_ev_a=config.fmax_spike_floor_ev_a,
    )
    stopped_on_spike = False

    def _record_step() -> None:
        best_frame.record(f"step_{opt.get_number_of_steps()}")

    def _abort_on_spike() -> None:
        nonlocal stopped_on_spike
        fmax = float(np.abs(atoms.get_forces()).max())
        if not np.isfinite(fmax) or fmax > spike_limit:
            stopped_on_spike = True
            if config.verbose:
                fmax_txt = f"{fmax:.4f}" if np.isfinite(fmax) else "non-finite"
                print(
                    f"{context_prefix} hybrid calculator BFGS: "
                    f"fmax spike {fmax_txt} > {spike_limit:.4f} eV/Å; stopping",
                    flush=True,
                )
            raise StopIteration

    opt = BfgsOptimizer(
        atoms,
        logfile=None if config.quiet_bfgs else "-",
        maxstep=float(config.bfgs_maxstep),
    )
    opt.attach(_record_step, interval=1)
    opt.attach(_abort_on_spike, interval=1)
    try:
        opt.run(fmax=float(config.fmax_ev_a), steps=int(config.max_steps))
    except StopIteration:
        pass
    best_frame.record("final")
    return opt, best_frame, stopped_on_spike


def minimize_hybrid_calculator_before_sd(
    mlpot_ctx: Any,
    config: HybridCalculatorMinimizeConfig,
    *,
    context_prefix: str = "Pre-SD",
) -> float:
    """Relax CHARMM coordinates with ASE BFGS on the hybrid calculator.

    Returns hybrid GRMS (kcal/mol/Å) after minimization (watchdog baseline).
    """
    import ase

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
        sync_charmm_lists_after_mini,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    z = getattr(mlpot_ctx, "ml_Z", None)
    if z is None:
        raise RuntimeError("mlpot_ctx.ml_Z required for hybrid calculator minimize")

    _promote_mlpot_jax_for_calculator_mini(mlpot_ctx, verbose=config.verbose)
    pos0 = get_charmm_positions_array()
    grms0 = mlpot_hybrid_grms_from_calculator(mlpot_ctx, positions=pos0)
    if config.verbose:
        grms_txt = f"{grms0:.4f}" if grms0 is not None and np.isfinite(grms0) else "?"
        opt_name = "BFGSLineSearch" if config.use_bfgs_line_search else "BFGS"
        print(
            f"{context_prefix} hybrid calculator {opt_name}: start GRMS={grms_txt} "
            f"kcal/mol/Å (max {config.max_steps} steps, fmax={config.fmax_ev_a} eV/Å)",
            flush=True,
        )

    atoms = ase.Atoms(numbers=np.asarray(z, dtype=int), positions=pos0)
    atoms.calc = _hybrid_mlpot_ase_calculator_class()(mlpot_ctx)
    initial_fmax = float(np.abs(atoms.get_forces()).max())
    opt, best_frame, stopped_on_spike = _run_hybrid_calculator_bfgs(
        atoms,
        config,
        context_prefix=context_prefix,
        initial_fmax_ev_a=initial_fmax,
    )
    restored_fmax = best_frame.restore_best_force()
    if config.verbose and (
        stopped_on_spike
        or (
            best_frame.best_force_label
            and best_frame.best_force_label not in ("initial", "final")
        )
    ):
        print(
            f"{context_prefix} hybrid calculator BFGS: restored best-force frame "
            f"({best_frame.best_force_label}, fmax={best_frame.best_force_fmax:.6f} eV/Å)",
            flush=True,
        )

    sync_charmm_positions(np.asarray(atoms.get_positions(), dtype=np.float64))
    sync_charmm_lists_after_mini(quiet=True)
    invalidate_mlpot_calculator_caches(mlpot_ctx)
    mlpot_ctx.reregister_mlpot(verbose=False)
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_grms_after_ener_force

    charmm_grms_after_ener_force()

    grms1 = mlpot_hybrid_grms_from_calculator(mlpot_ctx)
    if grms1 is None or not np.isfinite(grms1):
        raise RuntimeError("hybrid GRMS unavailable after calculator minimize")
    if config.verbose:
        fmax = float(np.abs(atoms.get_forces()).max())
        spike_note = " (spike abort)" if stopped_on_spike else ""
        print(
            f"{context_prefix} hybrid calculator BFGS done{spike_note}: "
            f"GRMS {grms_txt} -> {grms1:.4f} kcal/mol/Å, "
            f"fmax={fmax:.6f} eV/Å, steps={opt.get_number_of_steps()}",
            flush=True,
        )
    mlpot_ctx.sd_watchdog_baseline_grms = float(grms1)
    return float(grms1)
