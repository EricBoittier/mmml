"""Unified MD architecture: builders, energy terms, drivers, and samplers.

Shared layer that the ``md-system`` CLI and the ``cg_jaxmd`` workflow both lower
onto. See ``docs/md-cg-unification-design.md`` for the full schema and the
decision ledger.

This package is dependency-light on import: jax, ASE, and PyCHARMM are pulled in
lazily by the concrete implementations, not by these protocol/dataclass seams.
"""

from __future__ import annotations

from mmml.md.assemble import (
    assemble_and_run,
    available_builders,
    build_hybrid_energy,
    build_system,
    get_builder,
)
from mmml.md.config import EnsembleSpec, RunConfig
from mmml.md.lowering import (
    runconfig_from_cg_config,
    runconfig_from_md_system_args,
    terms_from_cg_config,
)
from mmml.md.results import Trajectory
from mmml.md.system import FFParams, MolecularSystem, SystemSpec

__all__ = [
    "FFParams",
    "MolecularSystem",
    "SystemSpec",
    "EnsembleSpec",
    "RunConfig",
    "Trajectory",
    "assemble_and_run",
    "available_builders",
    "build_hybrid_energy",
    "build_system",
    "get_builder",
    "runconfig_from_cg_config",
    "runconfig_from_md_system_args",
    "terms_from_cg_config",
]
