"""Drivers: integrator engines that propagate a system under a hybrid energy.

The maintained unified driver is currently ``JaxmdDriver``. Proposed
``AseDriver``, ``CharmmDriver``, and ``ApoCharmmDriver`` names appear in the
architecture design but are not implementations in this package.

The ``on_overlap`` hook is the explicit, impure escape hatch for CHARMM
repair/minimize (decision, §10) so energy terms stay pure. Concrete drivers
migrate here from ``karml.cli.run.md_pbc_suite`` (``ase.py``, ``jaxmd.py``,
``pycharmm_mlpot.py``) and, for apocharmm, from a new pybind11 driver.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from karml.md.config import EnsembleSpec
from karml.md.energy.registry import HybridEnergy
from karml.md.results import Trajectory
from karml.md.system import MolecularSystem

__all__ = ["Driver", "JaxmdDriver", "NonFiniteStateError"]


@runtime_checkable
class Driver(Protocol):
    """Propagate ``system`` under ``energy`` for the given ``ensemble``."""

    name: str

    def run(
        self,
        system: MolecularSystem,
        energy: HybridEnergy,
        ensemble: EnsembleSpec,
        *,
        on_overlap: Callable[..., Any] | None = None,
    ) -> Trajectory:
        ...


# Safe eager export: the implementation itself keeps jax/jax-md imports lazy.
from karml.md.drivers.jaxmd import JaxmdDriver, NonFiniteStateError  # noqa: E402
