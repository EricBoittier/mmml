"""Factory for supplementary QC backends."""

from __future__ import annotations

from typing import Any

from mmml.interfaces.energy_forces.registry import build_provider, provider_from_dict
from mmml.interfaces.energy_forces.spec import ProviderSpec
from mmml.interfaces.qc_backends.protocol import BackendSpec, QCEvaluator


def build_backend(spec: BackendSpec) -> QCEvaluator:
    """Instantiate a backend from a BackendSpec."""
    return build_provider(ProviderSpec(name=spec.name, options=dict(spec.options)))


def backend_from_dict(entry: dict[str, Any]) -> BackendSpec:
    """Parse a YAML/JSON backend entry."""
    spec = provider_from_dict(entry)
    return BackendSpec(name=spec.name, options=dict(spec.options))
