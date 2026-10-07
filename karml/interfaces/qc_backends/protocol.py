"""Protocol and configuration for supplementary QC backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from karml.interfaces.energy_forces.protocol import EnergyForcesProvider, QCEvaluator

__all__ = ["BackendSpec", "EnergyForcesProvider", "QCEvaluator"]


@dataclass(frozen=True)
class BackendSpec:
    """Configuration for one cross-check backend."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        method = self.options.get("method") or self.options.get("xc") or self.options.get("functional")
        basis = self.options.get("basis")
        parts = [self.name]
        if method:
            parts.append(str(method))
        if basis:
            parts.append(str(basis))
        return "/".join(parts)
