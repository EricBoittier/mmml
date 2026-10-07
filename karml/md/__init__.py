"""Unified MD architecture: builders, energy terms, drivers, and samplers.

Shared layer that the ``md-system`` CLI and the ``cg_jaxmd`` workflow both lower
onto. See ``docs/md-cg-unification-design.md`` for the full schema and the
decision ledger.

This package is dependency-light on import: jax, ASE, and PyCHARMM are pulled in
lazily by the concrete implementations, not by these protocol/dataclass seams.
"""

from __future__ import annotations

from karml.md.assemble import (
    assemble_and_run,
    available_builders,
    build_hybrid_energy,
    build_system,
    get_builder,
)
from karml.md.config import EnsembleSpec, RunConfig
from karml.md.interactions import InteractionPolicy, compile_interaction_policy
from karml.md.temperature import TemperatureSchedule, parse_temperature_schedule
from karml.md.lowering import (
    runconfig_from_cg_config,
    runconfig_from_md_system_args,
    terms_from_cg_config,
)
from karml.md.neighbors import make_intermolecular_neighbor_fn
from karml.md.results import Trajectory
from karml.md.samplers import RigidBodySampler
from karml.md.system import FFParams, MolecularSystem, SystemSpec

__all__ = [
    "FFParams",
    "MolecularSystem",
    "SystemSpec",
    "EnsembleSpec",
    "RunConfig",
    "InteractionPolicy",
    "compile_interaction_policy",
    "TemperatureSchedule",
    "parse_temperature_schedule",
    "Trajectory",
    "assemble_and_run",
    "available_builders",
    "build_hybrid_energy",
    "build_system",
    "get_builder",
    "runconfig_from_cg_config",
    "runconfig_from_md_system_args",
    "terms_from_cg_config",
    "RigidBodySampler",
    "make_intermolecular_neighbor_fn",
]
