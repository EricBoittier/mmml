"""Spatial domain grid and halo masks for MPI ML decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy import (
    mic_displacement_numpy,
)

DEFAULT_PHYSNET_CUTOFF_A: float = 6.0


def resolve_halo_radius(
    *,
    mm_switch_on: float = DEFAULT_MM_SWITCH_ON,
    physnet_cutoff: float = DEFAULT_PHYSNET_CUTOFF_A,
    ml_switch_width: float = DEFAULT_ML_SWITCH_WIDTH,
    monomer_extent_A: float = 0.0,
) -> float:
    """Halo width for domain ∪ halo dimer selection (Å)."""
    return float(mm_switch_on) + float(physnet_cutoff) + float(ml_switch_width) + float(
        monomer_extent_A
    )


def halo_radius_from_cutoffs(
    cutoff_params: object,
    *,
    physnet_cutoff: float = DEFAULT_PHYSNET_CUTOFF_A,
    monomer_extent_A: float = 0.0,
) -> float:
    """Build halo radius from a :class:`~mmml.interfaces.pycharmmInterface.cutoffs.CutoffParameters`."""
    mm_switch_on = float(getattr(cutoff_params, "mm_switch_on", DEFAULT_MM_SWITCH_ON))
    ml_switch_width = float(
        getattr(cutoff_params, "ml_switch_width", DEFAULT_ML_SWITCH_WIDTH)
    )
    return resolve_halo_radius(
        mm_switch_on=mm_switch_on,
        physnet_cutoff=physnet_cutoff,
        ml_switch_width=ml_switch_width,
        monomer_extent_A=monomer_extent_A,
    )


def compute_monomer_coms(
    positions: np.ndarray,
    n_monomers: int,
    atoms_per_monomer: Union[int, Sequence[int]],
) -> np.ndarray:
    """Monomer centers of mass, shape ``(n_monomers, 3)``."""
    pos = np.asarray(positions, dtype=np.float64)
    if isinstance(atoms_per_monomer, int):
        apm = int(atoms_per_monomer)
        return pos.reshape(int(n_monomers), apm, 3).mean(axis=1)
    per = [int(x) for x in atoms_per_monomer]
    if len(per) != int(n_monomers):
        raise ValueError(f"atoms_per_monomer length {len(per)} != n_monomers {n_monomers}")
    offsets = np.zeros(len(per) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(per)
    coms = np.zeros((len(per), 3), dtype=np.float64)
    for mi, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        coms[mi] = pos[a:b].mean(axis=0)
    return coms


@dataclass(frozen=True)
class SpatialDomainGrid:
    """1-D domain decomposition along *x* with cubic PBC (Phase 2 default)."""

    box_side_A: float
    n_ranks: int
    halo_radius_A: float

    def _wrap_x(self, x: np.ndarray) -> np.ndarray:
        side = float(self.box_side_A)
        return np.mod(np.asarray(x, dtype=np.float64), side)

    def rank_for_com(self, com: np.ndarray) -> int:
        """Owning rank for a monomer COM (slab along x)."""
        return rank_for_com(com, self.box_side_A, self.n_ranks)

    def slab_bounds(self, rank: int) -> tuple[float, float]:
        """Inclusive-exclusive x bounds ``[lo, hi)`` for ``rank`` in ``[0, n_ranks)``."""
        side = float(self.box_side_A)
        n = int(self.n_ranks)
        r = int(rank)
        if n <= 0 or r < 0 or r >= n:
            raise ValueError(f"invalid rank {r} for n_ranks={n}")
        width = side / n
        return r * width, (r + 1) * width

    def extended_bounds(self, rank: int) -> tuple[float, float]:
        """Domain ∪ halo bounds along x (may wrap; returned unwrapped span)."""
        lo, hi = self.slab_bounds(rank)
        h = float(self.halo_radius_A)
        return lo - h, hi + h

    def monomers_in_extended_domain(
        self,
        coms: np.ndarray,
        rank: int,
    ) -> np.ndarray:
        """Boolean mask over monomers visible to ``rank`` (domain ∪ halo along x)."""
        return monomers_in_extended_domain(coms, rank, self.box_side_A, self.n_ranks, self.halo_radius_A)

    def owned_monomer_mask(self, coms: np.ndarray, rank: int) -> np.ndarray:
        """Monomers owned by ``rank`` (no halo)."""
        x = self._wrap_x(coms[:, 0])
        lo, hi = self.slab_bounds(rank)
        return (x >= lo) & (x < hi)


def rank_for_com(com: np.ndarray, box_side_A: float, n_ranks: int) -> int:
    """Map COM to owning rank (1-D slabs along x, PBC wrap on x)."""
    side = float(box_side_A)
    n = max(1, int(n_ranks))
    x = float(np.mod(com[0], side))
    idx = int(np.floor(x / side * n))
    return min(max(idx, 0), n - 1)


def monomers_in_extended_domain(
    coms: np.ndarray,
    rank: int,
    box_side_A: float,
    n_ranks: int,
    halo_radius_A: float,
) -> np.ndarray:
    """True where monomer COM is inside rank's x-slab ± halo (minimum-image along x)."""
    coms = np.asarray(coms, dtype=np.float64)
    side = float(box_side_A)
    n = max(1, int(n_ranks))
    r = int(rank)
    width = side / n
    lo = r * width - float(halo_radius_A)
    hi = (r + 1) * width + float(halo_radius_A)

    x = np.mod(coms[:, 0], side)

    def in_interval(xv: np.ndarray, a: float, b: float) -> np.ndarray:
        """Interval along x with PBC wrap (1-D MIC)."""
        if a <= 0 and b >= side:
            return np.ones(xv.shape[0], dtype=bool)
        # Shift so interval is contiguous
        rel = np.mod(xv - a, side)
        length = b - a
        if length <= 0:
            length += side
        return rel < length

    return in_interval(x, lo, hi)


def dimer_com_mic(
    com_a: np.ndarray,
    com_b: np.ndarray,
    cell: Optional[np.ndarray],
) -> np.ndarray:
    """Dimer COM as midpoint using MIC vector between monomer COMs."""
    d = mic_displacement_numpy(com_a, com_b, cell)
    return com_a + 0.5 * d
