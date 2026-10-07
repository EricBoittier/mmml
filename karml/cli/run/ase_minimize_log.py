"""Compact progress logging for ASE BFGS/FIRE (replaces ASE's verbose step table)."""

from __future__ import annotations

import sys
from typing import Any, TextIO

import numpy as np


def ase_atomic_fmax(atoms) -> float:
    """ASE-style fmax: max atomic force norm (eV/Å)."""
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.size == 0:
        return float("nan")
    return float(np.linalg.norm(forces, axis=1).max())


def resolve_ase_optimizer_logfile(args: Any) -> str | None:
    """Return ASE ``logfile`` target.

    - ``quiet_bfgs``: no ASE table
    - ``verbose_bfgs``: full ASE table on stdout (``"-"``)
    - default: no ASE table; use :class:`CompactAseOptimizerLog` instead
    """
    if bool(getattr(args, "quiet_bfgs", False)):
        return None
    if bool(getattr(args, "verbose_bfgs", False)):
        return "-"
    return None


def resolve_ase_log_every(args: Any, *, n_steps: int | None = None) -> int:
    """Steps between compact log lines (default ~10 lines per run)."""
    explicit = getattr(args, "bfgs_log_every", None)
    if explicit is not None:
        return max(1, int(explicit))
    if n_steps is not None and int(n_steps) > 0:
        return max(1, int(n_steps) // 10)
    return 10


class CompactAseOptimizerLog:
    """Attach to an ASE optimizer; print sparse ``step / E / fmax`` lines."""

    def __init__(
        self,
        label: str,
        *,
        every: int = 10,
        max_steps: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.label = str(label)
        self.every = max(1, int(every))
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.stream = stream if stream is not None else sys.stdout
        self._last_printed = -1

    def attach(self, opt) -> None:
        def _callback() -> None:
            atoms = opt.atoms
            n = int(getattr(opt, "nsteps", 0) or 0)
            # ASE increments nsteps after the step; step 0 is the initial force eval.
            if n != 0 and n % self.every != 0 and (
                self.max_steps is None or n < int(self.max_steps)
            ):
                return
            if n == self._last_printed:
                return
            self._last_printed = n
            try:
                energy = float(atoms.get_potential_energy())
                fmax = ase_atomic_fmax(atoms)
            except Exception:
                return
            if self.max_steps is not None:
                step_s = f"{n:>4d}/{self.max_steps}"
            else:
                step_s = f"{n:>4d}"
            print(
                f"{self.label}: step {step_s}  E={energy:12.6f} eV  fmax={fmax:.6f} eV/Å",
                file=self.stream,
                flush=True,
            )

        opt.attach(_callback, interval=1)


def attach_compact_ase_optimizer_log(
    opt,
    args: Any,
    *,
    label: str,
    max_steps: int | None = None,
) -> None:
    """Attach compact logging when not quiet/verbose (ASE table already disabled)."""
    if bool(getattr(args, "quiet_bfgs", False)):
        return
    if bool(getattr(args, "verbose_bfgs", False)):
        return
    every = resolve_ase_log_every(args, n_steps=max_steps)
    CompactAseOptimizerLog(label, every=every, max_steps=max_steps).attach(opt)
