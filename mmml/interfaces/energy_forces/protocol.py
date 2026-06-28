"""Unified protocol for models and QC backends that provide energy and forces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from ase import Atoms


class ProviderKind(str, Enum):
    """Known energy/forces provider families."""

    PHYSNET = "physnet"
    SPOOKY_PHYSNET = "spooky_physnet"
    JOINT_PHYSNET_DCMNET = "joint_physnet_dcmnet"
    EFIELD_PHYSNET = "efield_physnet"
    DCMNET = "dcmnet"
    PYSCF = "pyscf"
    ORCA = "orca"
    XTB = "xtb"
    MOLPRO = "molpro"
    ASE = "ase"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can compute and how it should be used."""

    kind: ProviderKind
    supports_energy: bool = True
    supports_forces: bool = True
    supports_dipole: bool = False
    supports_charges: bool = False
    supports_esp: bool = False
    supports_external_field: bool = False
    supports_decomposed_ml: bool = False
    """True when usable in hybrid CHARMM monomer/dimer MLpot (PhysNet family)."""
    supports_batch: bool = True
    notes: str = ""


@runtime_checkable
class EnergyForcesProvider(Protocol):
    """Evaluate energies and forces for one or more ASE structures.

    Canonical interface for ML checkpoints (PhysNet, joint, E-field), ASE
    calculators, and supplementary QC backends (PySCF, ORCA, xTB, Molpro).
    """

    name: str
    method_label: str
    energy_unit: str
    force_unit: str
    capabilities: ProviderCapabilities

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]: ...


# Backward-compatible alias used by cross-check tooling.
QCEvaluator = EnergyForcesProvider
