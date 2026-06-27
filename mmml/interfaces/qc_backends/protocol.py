"""Protocol and configuration for supplementary QC backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from ase import Atoms


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


@runtime_checkable
class QCEvaluator(Protocol):
    """Evaluate energies and forces for a batch of ASE structures."""

    name: str
    method_label: str
    energy_unit: str
    force_unit: str

    def evaluate_batch(
        self,
        frames: list[Atoms],
        *,
        properties: frozenset[str],
    ) -> dict[str, np.ndarray]: ...
