"""Factory for unified energy/forces providers (ML + QC + ASE)."""

from __future__ import annotations

from typing import Any

from mmml.interfaces.energy_forces.adapters import AseCalculatorProvider, QCEvaluatorAdapter
from mmml.interfaces.energy_forces.ml import build_ml_provider, capabilities_for_kind
from mmml.interfaces.energy_forces.protocol import EnergyForcesProvider, ProviderKind
from mmml.interfaces.energy_forces.spec import ProviderSpec
from mmml.interfaces.qc_backends.molpro import build_molpro_backend
from mmml.interfaces.qc_backends.orca_qm import build_orca_backend
from mmml.interfaces.qc_backends.pyscf_backend import build_pyscf_backend
from mmml.interfaces.qc_backends.xtb import build_xtb_backend

_QC_BUILDERS = {
    "pyscf": build_pyscf_backend,
    "orca": build_orca_backend,
    "orca_qm": build_orca_backend,
    "xtb": build_xtb_backend,
    "molpro": build_molpro_backend,
}

_KIND_BY_NAME = {
    "pyscf": ProviderKind.PYSCF,
    "orca": ProviderKind.ORCA,
    "orca_qm": ProviderKind.ORCA,
    "xtb": ProviderKind.XTB,
    "molpro": ProviderKind.MOLPRO,
}


def build_provider(spec: ProviderSpec) -> EnergyForcesProvider:
    """Instantiate any supported energy/forces provider."""
    name = spec.name.lower()
    options = dict(spec.options)

    if name in ("ml", "physnet", "checkpoint", "joint", "efield"):
        return build_ml_provider(options)

    if name == "ase":
        calculator = options.pop("calculator", None)
        if calculator is None:
            raise ValueError("ASE provider requires 'calculator' in options.")
        return AseCalculatorProvider(calculator, **options)

    builder = _QC_BUILDERS.get(name)
    if builder is None:
        supported = sorted(set(_QC_BUILDERS) | {"ml", "physnet", "checkpoint", "ase"})
        raise ValueError(f"Unknown provider {spec.name!r}. Supported: {', '.join(supported)}")

    evaluator = builder(options)
    kind = _KIND_BY_NAME.get(name, ProviderKind.UNKNOWN)
    return QCEvaluatorAdapter(evaluator, capabilities=capabilities_for_kind(kind))


def provider_from_dict(entry: dict[str, Any]) -> ProviderSpec:
    """Parse a YAML/JSON provider entry."""
    name = entry.get("name")
    if not name:
        raise ValueError("Each provider entry must include 'name'.")
    options = {k: v for k, v in entry.items() if k != "name"}
    return ProviderSpec(name=str(name), options=options)
