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


def _pairs_to_array(pairs) -> np.ndarray:
    """Convert a set/iterable of ``(i, j)`` pairs to a sorted ``(M, 2)`` int32 array."""
    ordered = sorted(tuple(int(x) for x in p) for p in pairs)
    if not ordered:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(ordered, dtype=np.int32).reshape(-1, 2)


@dataclass(frozen=True)
class FFParams:
    """Fully-resolved CHARMM force-field state, built once and carried as data.

    Mirrors ``NonbondedSystemData`` (mm_system_energy) field-for-field: this is
    the canonical form the builder resolves once from the PSF + CHARMM parameters
    and every energy term reads (decision A, §10). Terms consume these arrays
    rather than recomputing exclusions / 1-4 lists / LJ tables inline as
    ``cg_jaxmd`` currently does. See ``docs/hybrid-mlmm-decomposition.md`` §2.
    """

    charges: np.ndarray            # (N,) partial charges          ← nbdata.charges
    epsilon: np.ndarray            # (N,) LJ epsilon (kcal/mol)    ← nbdata.epsilon
    rmin_half: np.ndarray          # (N,) LJ Rmin/2 (Å)            ← nbdata.rmin
    at_codes: np.ndarray           # (N,) nonbonded type code      ← nbdata.at_codes
    exclusions: np.ndarray         # (M, 2) 1-2/1-3 excluded pairs ← nbdata.excluded_pairs
    e14_pairs: np.ndarray          # (K, 2) 1-4 pairs              ← nbdata.e14_pairs
    psf_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_nonbonded_system_data(cls, nbdata: Any) -> "FFParams":
        """Build from a ``NonbondedSystemData`` (or any object exposing its fields).

        Duck-typed so ``mmml.md.system`` needs no jax/CHARMM import. CHARMM's
        ``rmin`` field is the per-atom *Rmin/2* half-value → ``rmin_half``; the
        exclusion and 1-4 ``frozenset``s become sorted ``(*, 2)`` index arrays.
        """
        return cls(
            charges=np.asarray(nbdata.charges, dtype=np.float64),
            epsilon=np.asarray(nbdata.epsilon, dtype=np.float64),
            rmin_half=np.asarray(nbdata.rmin, dtype=np.float64),
            at_codes=np.asarray(nbdata.at_codes, dtype=np.int32),
            exclusions=_pairs_to_array(nbdata.excluded_pairs),
            e14_pairs=_pairs_to_array(nbdata.e14_pairs),
            psf_path=getattr(nbdata, "psf_path", None),
        )


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
