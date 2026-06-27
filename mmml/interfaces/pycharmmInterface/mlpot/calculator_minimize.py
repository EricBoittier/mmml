"""Hybrid JAX calculator minimization on CHARMM coordinates before MLpot SD."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, TextIO

import numpy as np

from mmml.data.units import EV_TO_KCAL_MOL, format_energy_ev_kcal

from mmml.interfaces.pycharmmInterface.mlpot.cli_common import (
    mlpot_hybrid_grms_from_calculator,
    mlpot_spherical_energy_forces_ev_angstrom,
)


@dataclass(frozen=True)
class HybridMinimizeResult:
    """Hybrid calculator mini outcome (GRMS in kcal/mol/Å + whether ASE ran)."""

    grms: float
    ran: bool


def coerce_hybrid_minimize_result(
    result: HybridMinimizeResult | tuple[float, bool] | float,
) -> HybridMinimizeResult:
    """Normalize mini return values (supports legacy bare-float mocks)."""
    if isinstance(result, HybridMinimizeResult):
        return result
    if isinstance(result, tuple):
        return HybridMinimizeResult(grms=float(result[0]), ran=bool(result[1]))
    return HybridMinimizeResult(grms=float(result), ran=True)


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
    fmax_absolute_ceiling_ev_a: float = 500.0
    fmax_running_spike_factor: float = 1.5
    max_initial_fmax_ev_a: float = 100.0
    max_start_grms_kcalmol_A: float | None = None
    stall_patience_steps: int = 15
    stall_improvement_ratio: float = 0.99


@dataclass
class HybridCalculatorFireConfig:
    """ASE FIRE on the full hybrid calculator (synced to CHARMM coords)."""

    max_steps: int = 200
    fmax_ev_a: float = 0.05
    fire_maxstep: float = 0.2
    verbose: bool = True
    fmax_spike_factor: float = 4.0
    fmax_spike_floor_ev_a: float = 15.0
    fmax_absolute_ceiling_ev_a: float = 500.0
    fmax_running_spike_factor: float = 1.5
    max_initial_fmax_ev_a: float = 100.0
    max_start_grms_kcalmol_A: float | None = None
    energy_spike_ev: float = 50.0
    energy_absolute_ceiling_ev: float = 1.0e4
    stall_patience_steps: int = 15
    stall_improvement_ratio: float = 0.99


def hybrid_calculator_mini_eligible(
    hybrid_grms: float,
    *,
    grms_limit: float | None,
    diag_kind: str,
    grms_hot: bool,
    user_hot: bool,
) -> bool:
    """Whether ASE hybrid BFGS is appropriate (bonded recovery should run first when False)."""
    max_start = float(grms_limit) if grms_limit is not None else 50.0
    if float(hybrid_grms) > max_start:
        return False
    return bool(grms_hot or diag_kind == "geometry_stress" or user_hot)


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


def should_abort_bfgs_fmax(
    current_fmax_ev_a: float,
    *,
    spike_limit_ev_a: float,
    best_fmax_ev_a: float,
    absolute_ceiling_ev_a: float,
    running_spike_factor: float,
) -> bool:
    """Return True when the current BFGS step should be rolled back and stopped."""
    if not np.isfinite(current_fmax_ev_a):
        return True
    if current_fmax_ev_a > float(absolute_ceiling_ev_a):
        return True
    if (
        np.isfinite(best_fmax_ev_a)
        and best_fmax_ev_a > 0.0
        and current_fmax_ev_a > float(best_fmax_ev_a) * float(running_spike_factor)
    ):
        return True
    return current_fmax_ev_a > float(spike_limit_ev_a)


def should_abort_fire_step(
    *,
    current_fmax_ev_a: float,
    current_energy_ev: float,
    spike_limit_ev_a: float,
    best_fmax_ev_a: float,
    best_energy_ev: float,
    initial_energy_ev: float,
    absolute_fmax_ceiling_ev_a: float,
    running_spike_factor: float,
    energy_spike_ev: float,
    energy_absolute_ceiling_ev: float,
) -> bool:
    """Return True when the current FIRE step should be rolled back and stopped."""
    if should_abort_bfgs_fmax(
        current_fmax_ev_a,
        spike_limit_ev_a=spike_limit_ev_a,
        best_fmax_ev_a=best_fmax_ev_a,
        absolute_ceiling_ev_a=absolute_fmax_ceiling_ev_a,
        running_spike_factor=running_spike_factor,
    ):
        return True
    if not np.isfinite(current_energy_ev):
        return True
    if current_energy_ev > float(energy_absolute_ceiling_ev):
        return True
    if (
        np.isfinite(initial_energy_ev)
        and current_energy_ev > float(initial_energy_ev) + float(energy_spike_ev)
    ):
        return True
    if (
        np.isfinite(best_energy_ev)
        and current_energy_ev > float(best_energy_ev) + float(energy_spike_ev)
    ):
        return True
    return False


def resolve_adaptive_fire_maxstep(
    *,
    configured_maxstep: float,
    initial_fmax_ev_a: float,
    initial_grms_kcalmol_A: float | None,
    grms_soft_cap_kcalmol_A: float = 30.0,
) -> float:
    """Shrink FIRE max displacement when starting from a rough hybrid landscape."""
    maxstep = float(configured_maxstep)
    if np.isfinite(initial_fmax_ev_a) and initial_fmax_ev_a > 0.0:
        maxstep = min(maxstep, max(0.01, 0.5 / float(initial_fmax_ev_a)))
    if (
        initial_grms_kcalmol_A is not None
        and np.isfinite(initial_grms_kcalmol_A)
        and float(initial_grms_kcalmol_A) > float(grms_soft_cap_kcalmol_A)
    ):
        scale = float(grms_soft_cap_kcalmol_A) / float(initial_grms_kcalmol_A)
        maxstep = min(maxstep, max(0.01, maxstep * scale))
    return float(max(0.01, maxstep))


_ASE_OPTIMIZER_STEP_RE = re.compile(
    r"^(\S+:\s+\d+\s+\d+:\d+:\d+\s+)"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(\s+)"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def annotate_ase_optimizer_log_line(line: str) -> str:
    """Add kcal/mol beside eV energies in ASE FIRE/BFGS optimizer log lines."""
    if "Energy" in line and "fmax" in line and "Step" in line:
        return line.replace("Energy", "Energy[eV (kcal/mol)]", 1)

    stripped = line.strip()
    if not stripped or stripped.startswith("---"):
        return line

    match = _ASE_OPTIMIZER_STEP_RE.match(stripped)
    if match is None:
        return line

    prefix, e_str, mid, fmax_str = match.groups()
    try:
        e_ev = float(e_str)
    except ValueError:
        return line
    e_kcal = e_ev * EV_TO_KCAL_MOL
    lead = line[: len(line) - len(line.lstrip())]
    body = f"{prefix}{e_ev:.6f} ({e_kcal:.2f}){mid}{fmax_str}"
    newline = "\n" if line.endswith("\n") else ""
    return f"{lead}{body}{newline}"


class _DualUnitAseOptimizerLog:
    """Stream wrapper that annotates ASE optimizer energy columns with kcal/mol."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        self._buf = ""

    def write(self, data: str) -> int:
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._stream.write(annotate_ase_optimizer_log_line(line + "\n"))
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self._stream.write(annotate_ase_optimizer_log_line(self._buf))
            self._buf = ""
        self._stream.flush()


def ase_optimizer_dual_unit_logfile(stream: TextIO | None = None) -> _DualUnitAseOptimizerLog:
    """Logfile target for ASE optimizers: energy column shows eV and kcal/mol."""
    return _DualUnitAseOptimizerLog(stream)


class _BestMinimizationFrame:
    """Track lowest-fmax geometry during ASE minimization (see jaxmd pre-min)."""

    def __init__(self, atoms: Any) -> None:
        self.atoms = atoms
        self.best_force_positions: np.ndarray | None = None
        self.best_force_fmax = float("inf")
        self.best_force_label = ""
        self.best_energy_positions: np.ndarray | None = None
        self.best_energy_ev = float("inf")
        self.best_energy_label = ""

    def record(self, label: str) -> None:
        try:
            fmax = float(np.abs(self.atoms.get_forces()).max())
        except Exception:
            fmax = float("inf")
        try:
            energy = float(self.atoms.get_potential_energy())
        except Exception:
            energy = float("inf")
        if not np.isfinite(fmax) and not np.isfinite(energy):
            return
        positions = np.asarray(self.atoms.get_positions(), dtype=np.float64).copy()
        if np.isfinite(fmax) and fmax < self.best_force_fmax:
            self.best_force_positions = positions
            self.best_force_fmax = fmax
            self.best_force_label = label
        if np.isfinite(energy) and energy < self.best_energy_ev:
            self.best_energy_positions = positions
            self.best_energy_ev = energy
            self.best_energy_label = label

    def restore_best_force(self) -> float:
        if self.best_force_positions is not None:
            self.atoms.set_positions(self.best_force_positions)
        if np.isfinite(self.best_force_fmax):
            return float(self.best_force_fmax)
        return float(np.abs(self.atoms.get_forces()).max())

    def restore_best_energy(self) -> float:
        if self.best_energy_positions is not None:
            self.atoms.set_positions(self.best_energy_positions)
        if np.isfinite(self.best_energy_ev):
            return float(self.best_energy_ev)
        return float(self.atoms.get_potential_energy())

    def restore_best(self, *, prefer_energy: bool = False) -> None:
        if prefer_energy and self.best_energy_positions is not None:
            self.restore_best_energy()
            return
        self.restore_best_force()


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
    steps_since_improvement = 0
    prev_best_fmax = float("inf")

    def _record_step() -> None:
        nonlocal steps_since_improvement, prev_best_fmax
        best_frame.record(f"step_{opt.get_number_of_steps()}")
        if best_frame.best_force_fmax < prev_best_fmax * float(config.stall_improvement_ratio):
            steps_since_improvement = 0
            prev_best_fmax = float(best_frame.best_force_fmax)
        else:
            steps_since_improvement += 1

    def _check_guards() -> None:
        nonlocal stopped_on_spike
        if stopped_on_spike:
            return
        fmax = float(np.abs(atoms.get_forces()).max())
        if should_abort_bfgs_fmax(
            fmax,
            spike_limit_ev_a=spike_limit,
            best_fmax_ev_a=best_frame.best_force_fmax,
            absolute_ceiling_ev_a=config.fmax_absolute_ceiling_ev_a,
            running_spike_factor=config.fmax_running_spike_factor,
        ):
            stopped_on_spike = True
            if config.verbose:
                fmax_txt = f"{fmax:.4f}" if np.isfinite(fmax) else "non-finite"
                print(
                    f"{context_prefix} hybrid calculator BFGS: "
                    f"fmax guard triggered ({fmax_txt} eV/Å); stopping",
                    flush=True,
                )
            return
        if (
            steps_since_improvement >= int(config.stall_patience_steps)
            and opt.get_number_of_steps() >= int(config.stall_patience_steps)
            and best_frame.best_force_fmax > float(config.fmax_ev_a) * 10.0
        ):
            stopped_on_spike = True
            if config.verbose:
                print(
                    f"{context_prefix} hybrid calculator BFGS: "
                    f"stalled at fmax={best_frame.best_force_fmax:.4f} eV/Å "
                    f"for {steps_since_improvement} steps; stopping",
                    flush=True,
                )

    opt = BfgsOptimizer(
        atoms,
        logfile=None if config.quiet_bfgs else ase_optimizer_dual_unit_logfile(),
        maxstep=float(config.bfgs_maxstep),
    )
    opt.attach(_record_step, interval=1)
    opt.attach(_check_guards, interval=1)
    for converged in opt.irun(
        fmax=float(config.fmax_ev_a),
        steps=int(config.max_steps),
    ):
        if stopped_on_spike or converged:
            break
    best_frame.record("final")
    return opt, best_frame, stopped_on_spike


def _run_hybrid_calculator_fire(
    atoms: Any,
    config: HybridCalculatorFireConfig,
    *,
    context_prefix: str,
    initial_fmax_ev_a: float,
    initial_energy_ev: float,
    initial_grms_kcalmol_A: float | None,
) -> tuple[Any, _BestMinimizationFrame, bool]:
    """Run guarded ASE FIRE; returns (optimizer, best_frame, stopped_on_spike)."""
    from ase.optimize.fire import FIRE

    best_frame = _BestMinimizationFrame(atoms)
    best_frame.record("initial")
    spike_limit = spike_fmax_limit_ev_a(
        initial_fmax_ev_a,
        factor=config.fmax_spike_factor,
        floor_ev_a=config.fmax_spike_floor_ev_a,
    )
    fire_maxstep = resolve_adaptive_fire_maxstep(
        configured_maxstep=config.fire_maxstep,
        initial_fmax_ev_a=initial_fmax_ev_a,
        initial_grms_kcalmol_A=initial_grms_kcalmol_A,
    )
    stopped_on_spike = False
    steps_since_improvement = 0
    prev_best_fmax = float("inf")

    def _record_step() -> None:
        nonlocal steps_since_improvement, prev_best_fmax
        best_frame.record(f"step_{opt.get_number_of_steps()}")
        if best_frame.best_force_fmax < prev_best_fmax * float(config.stall_improvement_ratio):
            steps_since_improvement = 0
            prev_best_fmax = float(best_frame.best_force_fmax)
        else:
            steps_since_improvement += 1

    def _check_guards() -> None:
        nonlocal stopped_on_spike
        if stopped_on_spike:
            return
        try:
            fmax = float(np.abs(atoms.get_forces()).max())
            energy = float(atoms.get_potential_energy())
        except Exception:
            stopped_on_spike = True
            if config.verbose:
                print(
                    f"{context_prefix} hybrid calculator FIRE: "
                    "non-finite energy/forces; stopping",
                    flush=True,
                )
            return
        if should_abort_fire_step(
            current_fmax_ev_a=fmax,
            current_energy_ev=energy,
            spike_limit_ev_a=spike_limit,
            best_fmax_ev_a=best_frame.best_force_fmax,
            best_energy_ev=best_frame.best_energy_ev,
            initial_energy_ev=initial_energy_ev,
            absolute_fmax_ceiling_ev_a=config.fmax_absolute_ceiling_ev_a,
            running_spike_factor=config.fmax_running_spike_factor,
            energy_spike_ev=config.energy_spike_ev,
            energy_absolute_ceiling_ev=config.energy_absolute_ceiling_ev,
        ):
            stopped_on_spike = True
            if config.verbose:
                fmax_txt = f"{fmax:.4f}" if np.isfinite(fmax) else "non-finite"
                energy_txt = f"{energy:.4f}" if np.isfinite(energy) else "non-finite"
                print(
                    f"{context_prefix} hybrid calculator FIRE: "
                    f"guard triggered (fmax={fmax_txt} eV/Å, E={energy_txt} eV); stopping",
                    flush=True,
                )
            return
        if (
            steps_since_improvement >= int(config.stall_patience_steps)
            and opt.get_number_of_steps() >= int(config.stall_patience_steps)
            and best_frame.best_force_fmax > float(config.fmax_ev_a) * 10.0
        ):
            stopped_on_spike = True
            if config.verbose:
                print(
                    f"{context_prefix} hybrid calculator FIRE: "
                    f"stalled at fmax={best_frame.best_force_fmax:.4f} eV/Å "
                    f"for {steps_since_improvement} steps; stopping",
                    flush=True,
                )

    if config.verbose and fire_maxstep < float(config.fire_maxstep) - 1e-9:
        print(
            f"{context_prefix} hybrid calculator FIRE: "
            f"adaptive maxstep {fire_maxstep:.4f} Å "
            f"(configured {float(config.fire_maxstep):.4f})",
            flush=True,
        )

    opt = FIRE(
        atoms,
        logfile=None if not config.verbose else ase_optimizer_dual_unit_logfile(),
        maxstep=float(fire_maxstep),
    )
    opt.attach(_record_step, interval=1)
    opt.attach(_check_guards, interval=1)
    for converged in opt.irun(
        fmax=float(config.fmax_ev_a),
        steps=int(config.max_steps),
    ):
        if stopped_on_spike or converged:
            break
    best_frame.record("final")
    return opt, best_frame, stopped_on_spike


def minimize_hybrid_calculator_before_sd(
    mlpot_ctx: Any,
    config: HybridCalculatorMinimizeConfig,
    *,
    context_prefix: str = "Pre-SD",
) -> HybridMinimizeResult:
    """Relax CHARMM coordinates with ASE BFGS on the hybrid calculator."""
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
    max_start = config.max_start_grms_kcalmol_A
    if max_start is None:
        max_start = 50.0
    if grms0 is not None and np.isfinite(grms0) and float(grms0) > float(max_start):
        if config.verbose:
            print(
                f"{context_prefix}: skip calculator BFGS "
                f"(GRMS {float(grms0):.1f} > {float(max_start):.1f} kcal/mol/Å); "
                "run bonded-MM recovery first",
                flush=True,
            )
        mlpot_ctx.sd_watchdog_baseline_grms = float(grms0)
        return HybridMinimizeResult(grms=float(grms0), ran=False)
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
    if initial_fmax > float(config.max_initial_fmax_ev_a):
        if config.verbose:
            print(
                f"{context_prefix}: skip calculator BFGS "
                f"(initial fmax {initial_fmax:.1f} > "
                f"{float(config.max_initial_fmax_ev_a):.1f} eV/Å); "
                "run bonded-MM recovery first",
                flush=True,
            )
        mlpot_ctx.sd_watchdog_baseline_grms = (
            float(grms0) if grms0 is not None and np.isfinite(grms0) else None
        )
        if mlpot_ctx.sd_watchdog_baseline_grms is None:
            raise RuntimeError("hybrid GRMS unavailable before calculator minimize")
        return HybridMinimizeResult(
            grms=float(mlpot_ctx.sd_watchdog_baseline_grms),
            ran=False,
        )
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
        try:
            e_final = float(atoms.get_potential_energy())
            energy_txt = format_energy_ev_kcal(e_final)
        except Exception:
            energy_txt = "?"
        print(
            f"{context_prefix} hybrid calculator BFGS done{spike_note}: "
            f"GRMS {grms_txt} -> {grms1:.4f} kcal/mol/Å, "
            f"E={energy_txt}, fmax={fmax:.6f} eV/Å, "
            f"steps={opt.get_number_of_steps()}",
            flush=True,
        )
    mlpot_ctx.sd_watchdog_baseline_grms = float(grms1)
    return HybridMinimizeResult(grms=float(grms1), ran=True)


def minimize_hybrid_calculator_fire_before_sd(
    mlpot_ctx: Any,
    config: HybridCalculatorFireConfig | None = None,
    *,
    max_steps: int = 200,
    fmax_ev_a: float = 0.05,
    fire_maxstep: float = 0.2,
    verbose: bool = True,
    max_start_grms_kcalmol_A: float | None = None,
    context_prefix: str = "Pre-SD",
) -> HybridMinimizeResult:
    """Relax CHARMM coordinates with guarded ASE FIRE on the hybrid calculator."""
    import ase

    from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
        invalidate_mlpot_calculator_caches,
        sync_charmm_lists_after_mini,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        sync_charmm_positions,
    )

    if config is None:
        config = HybridCalculatorFireConfig(
            max_steps=int(max_steps),
            fmax_ev_a=float(fmax_ev_a),
            fire_maxstep=float(fire_maxstep),
            verbose=bool(verbose),
            max_start_grms_kcalmol_A=max_start_grms_kcalmol_A,
        )

    z = getattr(mlpot_ctx, "ml_Z", None)
    if z is None:
        raise RuntimeError("mlpot_ctx.ml_Z required for hybrid calculator FIRE")

    _promote_mlpot_jax_for_calculator_mini(mlpot_ctx, verbose=config.verbose)
    pos0 = get_charmm_positions_array()
    grms0 = mlpot_hybrid_grms_from_calculator(mlpot_ctx, positions=pos0)
    max_start = config.max_start_grms_kcalmol_A
    if max_start is None:
        max_start = 50.0
    if grms0 is not None and np.isfinite(grms0) and float(grms0) > float(max_start):
        if config.verbose:
            print(
                f"{context_prefix}: skip calculator FIRE "
                f"(GRMS {float(grms0):.1f} > {float(max_start):.1f} kcal/mol/Å)",
                flush=True,
            )
        return HybridMinimizeResult(grms=float(grms0), ran=False)

    atoms = ase.Atoms(numbers=np.asarray(z, dtype=int), positions=pos0)
    atoms.calc = _hybrid_mlpot_ase_calculator_class()(mlpot_ctx)
    initial_fmax = float(np.abs(atoms.get_forces()).max())
    initial_energy = float(atoms.get_potential_energy())
    if initial_fmax > float(config.max_initial_fmax_ev_a):
        if config.verbose:
            print(
                f"{context_prefix}: skip calculator FIRE "
                f"(initial fmax {initial_fmax:.1f} > "
                f"{float(config.max_initial_fmax_ev_a):.1f} eV/Å)",
                flush=True,
            )
        if grms0 is None or not np.isfinite(grms0):
            raise RuntimeError("hybrid GRMS unavailable before calculator FIRE")
        return HybridMinimizeResult(grms=float(grms0), ran=False)

    if config.verbose:
        grms_txt = f"{grms0:.4f}" if grms0 is not None and np.isfinite(grms0) else "?"
        print(
            f"{context_prefix} hybrid calculator FIRE: start GRMS={grms_txt} "
            f"kcal/mol/Å (max {config.max_steps} steps, fmax={config.fmax_ev_a} eV/Å)",
            flush=True,
        )

    opt, best_frame, stopped_on_spike = _run_hybrid_calculator_fire(
        atoms,
        config,
        context_prefix=context_prefix,
        initial_fmax_ev_a=initial_fmax,
        initial_energy_ev=initial_energy,
        initial_grms_kcalmol_A=grms0,
    )
    best_frame.restore_best(prefer_energy=stopped_on_spike)
    if config.verbose and (
        stopped_on_spike
        or (
            best_frame.best_force_label
            and best_frame.best_force_label not in ("initial", "final")
        )
    ):
        print(
            f"{context_prefix} hybrid calculator FIRE: restored best frame "
            f"({best_frame.best_force_label}, fmax={best_frame.best_force_fmax:.6f} eV/Å, "
            f"E={best_frame.best_energy_ev:.6f} eV)",
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
        raise RuntimeError("hybrid GRMS unavailable after calculator FIRE")
    if config.verbose:
        fmax = float(np.abs(atoms.get_forces()).max())
        grms_txt = f"{grms0:.4f}" if grms0 is not None and np.isfinite(grms0) else "?"
        spike_note = " (spike abort)" if stopped_on_spike else ""
        try:
            e_final = float(atoms.get_potential_energy())
            energy_txt = format_energy_ev_kcal(e_final)
        except Exception:
            energy_txt = "?"
        print(
            f"{context_prefix} hybrid calculator FIRE done{spike_note}: "
            f"GRMS {grms_txt} -> {grms1:.4f} kcal/mol/Å, "
            f"E={energy_txt}, fmax={fmax:.6f} eV/Å, "
            f"steps={opt.get_number_of_steps()}",
            flush=True,
        )
    mlpot_ctx.sd_watchdog_baseline_grms = float(grms1)
    return HybridMinimizeResult(grms=float(grms1), ran=True)
