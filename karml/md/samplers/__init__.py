"""Samplers: propagators that are a peer of MD drivers, not a driver mode.

MD is the default sampler; rigid-body sampling is an alternative that moves
whole monomers as rigid bodies (COM translation + unit-quaternion rotation;
decision, §10) via MC moves or constrained rigid MD. A sampler reuses the same
:class:`~karml.md.system.MolecularSystem` and
:class:`~karml.md.energy.registry.HybridEnergy`; only the propagator differs, so
rigid sampling composes with any energy term and any backend without touching
the drivers.

The concrete :class:`RigidBodySampler` lives in ``karml/md/samplers/rigid.py``
(kept lazy so ``import karml.md.samplers`` needs no jax).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from karml.md.config import RunConfig
from karml.md.energy.registry import HybridEnergy
from karml.md.results import Trajectory
from karml.md.samplers.rigid import RigidBodySampler
from karml.md.system import MolecularSystem

__all__ = ["Sampler", "RigidBodySampler"]


@runtime_checkable
class Sampler(Protocol):
    """Generate configurations for ``system`` scored by ``energy``."""

    name: str

    def run(
        self,
        system: MolecularSystem,
        energy: HybridEnergy,
        config: RunConfig,
    ) -> Trajectory:
        ...
