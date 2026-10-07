"""Samplers: propagators that are a peer of MD drivers, not a driver mode.

MD is the default sampler; rigid-body sampling is an alternative that moves
whole monomers as rigid bodies (COM translation + unit-quaternion rotation;
decision, §10) via MC moves or constrained rigid MD. A sampler reuses the same
:class:`~mmml.md.system.MolecularSystem` and
:class:`~mmml.md.energy.registry.HybridEnergy`; only the propagator differs, so
rigid sampling composes with any energy term and any backend without touching
the drivers.

The concrete :class:`RigidBodySampler` lives in ``mmml/md/samplers/rigid.py``
(kept lazy so ``import mmml.md.samplers`` needs no jax).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mmml.md.config import RunConfig
from mmml.md.energy.registry import HybridEnergy
from mmml.md.results import Trajectory
from mmml.md.samplers.rigid import RigidBodySampler
from mmml.md.system import MolecularSystem

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
