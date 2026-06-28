"""Unified energy and forces providers for ML models and QC backends.

Use :func:`build_provider` to construct any supported provider from a
:class:`ProviderSpec`. ML checkpoints (PhysNet, SpookyPhysNet, joint
PhysNet+DCMNet, EFieldPhysNet) and supplementary QC backends (PySCF, ORCA,
xTB, Molpro) share the same :class:`EnergyForcesProvider` protocol.

Hybrid CHARMM monomer/dimer MLpot requires
:attr:`ProviderCapabilities.supports_decomposed_ml` (PhysNet family only).
Use :func:`assert_hybrid_ml_compatible` before building decomposed MLpot.
"""

from mmml.interfaces.energy_forces.adapters import AseCalculatorProvider
from mmml.interfaces.energy_forces.ml import (
    assert_hybrid_ml_compatible,
    build_ml_ase_provider,
    build_ml_provider,
    capabilities_for_kind,
    detect_model_kind,
)
from mmml.interfaces.energy_forces.protocol import (
    EnergyForcesProvider,
    ProviderCapabilities,
    ProviderKind,
    QCEvaluator,
)
from mmml.interfaces.energy_forces.registry import build_provider, provider_from_dict
from mmml.interfaces.energy_forces.spec import ProviderSpec

__all__ = [
    "AseCalculatorProvider",
    "EnergyForcesProvider",
    "ProviderCapabilities",
    "ProviderKind",
    "ProviderSpec",
    "QCEvaluator",
    "assert_hybrid_ml_compatible",
    "build_ml_ase_provider",
    "build_ml_provider",
    "build_provider",
    "capabilities_for_kind",
    "detect_model_kind",
    "provider_from_dict",
]
