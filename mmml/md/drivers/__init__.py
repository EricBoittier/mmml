"""Drivers: integrator engines that propagate a system under a hybrid energy.

One driver per engine — ``AseDriver``, ``JaxmdDriver``, ``CharmmDriver``,
``ApoCharmmDriver`` — each consuming a :class:`~mmml.md.energy.registry.HybridEnergy`
and an :class:`~mmml.md.config.EnsembleSpec` and producing a
:class:`~mmml.md.results.Trajectory`.

The ``on_overlap`` hook is the explicit, impure escape hatch for CHARMM
repair/minimize (decision, §10) so energy terms stay pure. Concrete drivers
migrate here from ``mmml.cli.run.md_pbc_suite`` (``ase.py``, ``jaxmd.py``,
``pycharmm_mlpot.py``) and, for apocharmm, from a new pybind11 driver.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from mmml.md.config import EnsembleSpec
from mmml.md.energy.registry import HybridEnergy
from mmml.md.results import Trajectory
from mmml.md.system import MolecularSystem

__all__ = ["Driver"]


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
