"""ASE geometry optimization helpers for PyXtal-built structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

OptimizerName = Literal["bfgs", "fire", "lbfgs"]


@dataclass
class AseOptimizeResult:
    atoms: Any
    optimizer: str
    n_steps: int
    fmax_ev_a: float
    energy_ev: float | None = None


def _optimizer_class(name: OptimizerName):
    from ase.optimize import BFGS, LBFGS
    from ase.optimize.fire import FIRE

    key = str(name).strip().lower()
    if key == "bfgs":
        return BFGS
    if key == "fire":
        return FIRE
    if key == "lbfgs":
        return LBFGS
    raise ValueError(f"Unknown ASE optimizer {name!r}; expected bfgs, fire, or lbfgs")


def attach_emt_calculator(atoms: Any) -> Any:
    """Attach ASE EMT (lightweight smoke-test calculator)."""
    from ase.calculators.emt import EMT

    atoms.calc = EMT()
    return atoms


def optimize_ase_atoms(
    atoms: Any,
    *,
    calculator: Any | None = None,
    use_emt: bool = False,
    optimizer: OptimizerName = "bfgs",
    fmax_ev_a: float = 0.05,
    max_steps: int = 200,
    fix_cell: bool = False,
    logfile: str | None = "-",
) -> AseOptimizeResult:
    """Relax ``atoms`` with ASE (atomic positions only; cell is not relaxed)."""
    working = atoms.copy()

    if calculator is not None:
        working.calc = calculator
    elif use_emt:
        attach_emt_calculator(working)
    elif working.calc is None:
        raise ValueError(
            "optimize_ase_atoms requires a calculator: pass calculator=..., "
            "set atoms.calc, or use use_emt=True for a quick EMT smoke test."
        )

    if not fix_cell and working.pbc.any():
        print(
            "WARN: ASE cell relaxation is not enabled in this helper; "
            "optimizing atomic positions only (fix_cell=False ignored).",
            flush=True,
        )

    opt_cls = _optimizer_class(optimizer)
    opt = opt_cls(working, logfile=logfile)
    opt.run(fmax=float(fmax_ev_a), steps=int(max_steps))
    forces = np.asarray(working.get_forces(), dtype=float)
    fmax = float(np.abs(forces).max()) if forces.size else float("inf")
    energy = None
    try:
        energy = float(working.get_potential_energy())
    except Exception:
        pass
    return AseOptimizeResult(
        atoms=working,
        optimizer=str(optimizer),
        n_steps=int(getattr(opt, "nsteps", max_steps)),
        fmax_ev_a=fmax,
        energy_ev=energy,
    )
