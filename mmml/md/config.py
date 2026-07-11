"""Unified run configuration for the ``md-system`` / ``cg_jaxmd`` stack.

``RunConfig`` is the single internal representation both front-ends lower to
(see ``docs/md-cg-unification-design.md``, §5, constraint 7): the ``md-system``
argparse CLI and the ``cg_jaxmd`` Snakemake JSON. ``EnsembleSpec`` captures the
thermodynamic ensemble, orthogonal to the energy definition (constraint 6).

Scaffolding only — the argparse/JSON lowering adapters land in later steps and
will live alongside the existing ``mmml.cli.run.md_config`` helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mmml.md.system import SystemSpec

__all__ = ["EnsembleSpec", "RunConfig"]


@dataclass(frozen=True)
class EnsembleSpec:
    """Thermodynamic ensemble + integration parameters.

    ``ensemble`` is one of ``{"min", "nve", "nvt", "npt"}``; ``space`` is
    ``{"free", "pbc"}``. ``thermostat`` / ``barostat`` name the specific
    integrator (e.g. ``"nhc"``, ``"langevin"``, ``"langevin_piston"``) so the
    same spec maps onto ASE, jax-md, PyCHARMM, and apocharmm drivers.
    """

    ensemble: str = "nve"
    space: str = "pbc"
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    dt_fs: float = 1.0
    n_steps: int = 0
    thermostat: str | None = None
    barostat: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """One run: what to build, how to score it, and how to propagate it.

    Target of both the ``md-system`` CLI and the ``cg_jaxmd`` Snakemake JSON.
    ``terms`` selects registered energy terms by name (see
    :mod:`mmml.md.energy.registry`); ``backend`` selects the driver engine and
    ``sampler`` selects MD vs. rigid-body sampling.
    """

    system: SystemSpec
    terms: tuple[str, ...] = ()
    ensemble: EnsembleSpec = field(default_factory=EnsembleSpec)
    backend: str = "auto"          # auto | ase | jaxmd | pycharmm | apocharmm
    sampler: str = "md"            # md | rigid
    checkpoint: Path | None = None
    output_dir: Path | None = None
    seed: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)
