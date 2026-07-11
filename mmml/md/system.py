"""Backend-agnostic molecular system and force-field state.

Shared topology layer for the unified ``md-system`` / ``cg_jaxmd`` architecture
(see ``docs/md-cg-unification-design.md``, §5-6). ``FFParams`` and
``MolecularSystem`` are the immutable artifacts every layer above the builders
reads; ``SystemSpec`` is the declarative input a :class:`SystemBuilder` consumes.

Decision A (§10): CHARMM force-field state — charges, LJ tables, exclusions,
e14 / vdw14 — is resolved **once by the builder** and carried on
``MolecularSystem.ff_params``. Energy terms read it; none re-derive it at
runtime. This scaffolding is intentionally logic-free: the real builders that
populate these dataclasses migrate here from ``mmml.interfaces.pycharmmInterface``
and ``mmml.cli.run.md_pbc_suite`` in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["FFParams", "MolecularSystem", "SystemSpec"]


@dataclass(frozen=True)
class FFParams:
    """Fully-resolved CHARMM force-field state, built once and carried as data.

    Sourced from the PSF + CHARMM parameters at *build* time and immutable
    thereafter (decision A, §10). Energy terms consume these arrays rather than
    recomputing exclusions / 1-4 lists / LJ tables inline as ``cg_jaxmd``
    currently does.
    """

    charges: np.ndarray            # (N,) partial charges
    lj_eps: np.ndarray             # LJ epsilon (per-atom or type-indexed)
    lj_sigma: np.ndarray           # LJ sigma (per-atom or type-indexed)
    lj_type_index: np.ndarray      # (N,) index into the LJ tables
    exclusions: np.ndarray         # (M, 2) excluded pair list
    e14_pairs: np.ndarray          # (K, 2) 1-4 pair list
    e14_scale: np.ndarray          # (K,) per-pair electrostatic 1-4 scaling
    vdw14: np.ndarray              # 1-4 LJ params / scaling
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MolecularSystem:
    """Immutable, backend-agnostic topology + coordinates.

    Emitted by a :class:`SystemBuilder`; read by energy terms, drivers, and
    samplers. Lowers to backend-specific containers (ASE ``Atoms``, jax-md
    arrays, apocharmm ``CharmmContext``) at the driver boundary.
    """

    R: np.ndarray                              # (N, 3) positions
    Z: np.ndarray                              # (N,) atomic numbers
    box: np.ndarray | None                     # (3, 3) cell, or None for free space
    mol_id: np.ndarray                         # (N,) molecule membership
    monomer_indices: list[np.ndarray] = field(default_factory=list)
    water_indices: list[np.ndarray] = field(default_factory=list)
    psf_path: Path | None = None
    ff_params: FFParams | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.R.shape[0])

    @property
    def is_periodic(self) -> bool:
        return self.box is not None


@dataclass(frozen=True)
class SystemSpec:
    """Declarative description of a system to build.

    The input to :meth:`SystemBuilder.build`. ``builder`` selects the backend
    (``"packmol"``, ``"pyxtal"``, ``"peptide_water"``, ``"template_pdb"``);
    ``params`` carries builder-specific options (composition, box sizing,
    template paths, ...).
    """

    builder: str
    composition: str | None = None
    n_molecules: int | None = None
    box_size: float | None = None
    template_pdb: Path | None = None
    seed: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)
