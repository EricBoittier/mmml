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
JaxPmeMethod = Literal["ewald", "pme", "p3m"]

CHARMM_COULOMB_KCAL = 332.063711
DEFAULT_JAX_PME_SR_CUTOFF_A = 6.0


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


def resolve_jax_pme_method(method: str | None = None) -> JaxPmeMethod:
    """Resolve jax-pme method from argument or ``JAX_PME_METHOD`` env."""
    raw = (method or os.environ.get("JAX_PME_METHOD", "ewald")).strip().lower()
    if raw in ("ewald", "pme", "p3m"):
        return raw  # type: ignore[return-value]
    raise ValueError(f"jax_pme method must be ewald|pme|p3m; got {method!r}")


def box_length_from_cell(cell: np.ndarray) -> float:
    """Cubic orthorhombic edge length (Å) from a 3-vector or 3×3 cell."""
    arr = np.asarray(cell, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.shape[0] == 3:
            return float(arr[0])
        return float(arr.reshape(-1)[0])
    return float(arr[0, 0])


def compute_jax_pme_coulomb(
    positions_A: np.ndarray,
    charges_e: np.ndarray,
    *,
    box_length_A: float,
    method: JaxPmeMethod | str = "ewald",
    sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
) -> LongRangeCoulombResult:
    """Full periodic Coulomb via jax-pme (Ewald / PME / P3M)."""
    from ase import Atoms
    from jaxpme import Ewald, P3M, PME
    from jaxpme import prefactors as jpref

    method_name = resolve_jax_pme_method(str(method))
    pos = np.asarray(positions_A, dtype=np.float64)
    chg = np.asarray(charges_e, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (n_atoms, 3); got {pos.shape}")
    if chg.shape[0] != pos.shape[0]:
        raise ValueError(f"charges length {chg.shape[0]} != n_atoms {pos.shape[0]}")
    L = float(box_length_A)
    atoms = Atoms(positions=pos, cell=np.eye(3) * L, pbc=True)
    calc_map = {"ewald": Ewald, "pme": PME, "p3m": P3M}
    calc = calc_map[method_name](prefactor=jpref.kcalmol_A)
    smearing = float(sr_cutoff_A) / 5.0
    mesh_spacing = smearing / 8.0
    lr_wavelength = smearing / 2.0
    if method_name == "ewald":
        inputs = calc.prepare(
            atoms,
            chg,
            cutoff=float(sr_cutoff_A),
            smearing=smearing,
            lr_wavelength=lr_wavelength,
        )
    else:
        inputs = calc.prepare(
            atoms,
            chg,
            cutoff=float(sr_cutoff_A),
            mesh_spacing=mesh_spacing,
            smearing=smearing,
        )
    energy, forces = calc.energy_forces(*inputs)
    return LongRangeCoulombResult(
        energy_kcalmol=float(energy),
        forces_kcalmol_A=np.asarray(forces, dtype=np.float64),
    )


class JaxPmeLongRangeSolver:
    """jax-pme Ewald / PME / P3M Coulomb backend."""

    name = "jax_pme"

    def __init__(
        self,
        *,
        method: str | None = None,
        sr_cutoff_A: float = DEFAULT_JAX_PME_SR_CUTOFF_A,
    ) -> None:
        self._method = resolve_jax_pme_method(method)
        self._sr_cutoff_A = float(sr_cutoff_A)

    @property
    def method(self) -> JaxPmeMethod:
        return self._method

    def compute(
        self,
        positions_A: np.ndarray,
        charges_e: np.ndarray,
        *,
        box_length_A: float,
    ) -> LongRangeCoulombResult:
        return compute_jax_pme_coulomb(
            positions_A,
            charges_e,
            box_length_A=box_length_A,
            method=self._method,
            sr_cutoff_A=self._sr_cutoff_A,
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
        return JaxPmeLongRangeSolver()
    return MicOnlySolver()


def describe_lr_solver(requested: str | None = None) -> str:
    """Human-readable summary for logs (chosen backend + availability)."""
    chosen = pick_lr_solver(requested)
    parts = [f"lr_solver={chosen}"]
    parts.append(f"scafacos={'yes' if have_scafacos() else 'no'}")
    parts.append(f"jax_pme={'yes' if have_jax_pme() else 'no'}")
    return ", ".join(parts)
