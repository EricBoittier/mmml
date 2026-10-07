"""Configuration for constructing energy/forces providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderSpec:
    """Configuration for :func:`build_provider`."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        method = (
            self.options.get("method")
            or self.options.get("xc")
            or self.options.get("functional")
            or self.options.get("checkpoint")
        )
        basis = self.options.get("basis")
        parts = [self.name]
        if method:
            parts.append(str(method))
        if basis:
            parts.append(str(basis))
        return "/".join(str(p) for p in parts)
