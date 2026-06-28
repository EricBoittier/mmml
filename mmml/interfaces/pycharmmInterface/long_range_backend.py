"""Long-range electrostatic backend selection for hybrid ML/MM potentials.

MMML's default JAX MM path uses minimum-image Coulomb truncated at the switched-MM
outer radius (~13 Å by default).  Optional backends can supply k-space corrections:

* ``mic`` — truncated MIC Coulomb only (default)
* ``jax_pme`` — JAX-native PME via the ``jax-pme`` dependency
* ``scafacos`` — ScaFaCoS ``libfcs`` (PME / P³M / P²NFFT / …)

Selection mirrors ``nl_backend.py``: CLI/YAML may pass an explicit name; otherwise
``MMML_LR_SOLVER`` and ``auto`` pick the first available backend.

See ``mlpot/LONG_RANGE_ELECTROSTATICS.md`` for how these layers interact with
CHARMM IMAGE lists and MLpot BLOCK terms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

LrSolverName = Literal["auto", "mic", "jax_pme", "scafacos"]

CHARMM_COULOMB_KCAL = 332.063711


def have_jax_pme() -> bool:
    try:
        from jaxpme import Ewald, P3M, PME  # noqa: F401

        return True
    except ImportError:
        return False


def have_scafacos() -> bool:
    try:
        from mmml.interfaces.scafacosInterface.scafacos_session import have_scafacos as _have

        return _have()
    except Exception:
        return False


def resolve_lr_solver(name: str | None = None) -> LrSolverName:
    """Resolve solver from argument, ``MMML_LR_SOLVER`` env, or ``auto``."""
    raw = (name or os.environ.get("MMML_LR_SOLVER", "auto")).strip().lower()
    if raw in ("auto", "mic", "jax_pme", "scafacos"):
        return raw  # type: ignore[return-value]
    raise ValueError(
        f"lr_solver must be auto|mic|jax_pme|scafacos; got {name!r}"
    )


def pick_lr_solver(requested: str | None = None) -> LrSolverName:
    """Choose the active long-range electrostatic backend."""
    name = resolve_lr_solver(requested)
    if name == "auto":
        if have_scafacos():
            return "scafacos"
        if have_jax_pme():
            return "jax_pme"
        return "mic"
    if name == "scafacos" and not have_scafacos():
        if have_jax_pme():
            return "jax_pme"
        return "mic"
    if name == "jax_pme" and not have_jax_pme():
        if have_scafacos():
            return "scafacos"
        return "mic"
    return name


@dataclass(frozen=True)
class LongRangeCoulombResult:
    energy_kcalmol: float
    forces_kcalmol_A: np.ndarray


class LongRangeCoulombSolver(Protocol):
    """Protocol for k-space / full-range Coulomb supplements."""

    name: str

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        """Return total electrostatic energy and forces for all atoms."""


class MicOnlySolver:
    """Placeholder: pair-listed MIC Coulomb remains in ``mm_energy_forces.py``."""

    name = "mic"

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        raise NotImplementedError(
            "mic backend evaluates Coulomb inside build_mm_energy_forces_fn; "
            "no separate long-range pass"
        )


class ScaFaCoSLongRangeSolver:
    name = "scafacos"

    def __init__(
        self,
        *,
        method: str | None = None,
        parameters: dict[str, str | float | int] | None = None,
    ) -> None:
        self._method = (method or os.environ.get("SCAFACOS_METHOD", "p2nfft")).strip()
        self._parameters = dict(parameters or {})

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        from mmml.interfaces.scafacosInterface.scafacos_session import compute_scafacos_coulomb

        result = compute_scafacos_coulomb(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            method=self._method,
            parameters=self._parameters or None,
        )
        return LongRangeCoulombResult(
            energy_kcalmol=result.energy_kcalmol,
            forces_kcalmol_A=result.forces_kcalmol_A,
        )


def create_lr_solver(requested: str | None = None) -> LongRangeCoulombSolver:
    """Instantiate the resolved long-range Coulomb backend."""
    chosen = pick_lr_solver(requested)
    if chosen == "scafacos":
        return ScaFaCoSLongRangeSolver()
    if chosen == "jax_pme":
        raise NotImplementedError(
            "jax_pme long-range backend is reserved; use MMML_LR_SOLVER=scafacos "
            "or mic until jax-pme wiring lands in mm_energy_forces.py"
        )
    return MicOnlySolver()


def describe_lr_solver(requested: str | None = None) -> str:
    """Human-readable summary for logs (chosen backend + availability)."""
    chosen = pick_lr_solver(requested)
    parts = [f"lr_solver={chosen}"]
    parts.append(f"scafacos={'yes' if have_scafacos() else 'no'}")
    parts.append(f"jax_pme={'yes' if have_jax_pme() else 'no'}")
    return ", ".join(parts)
