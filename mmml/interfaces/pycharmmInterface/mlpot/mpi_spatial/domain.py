"""Spatial domain grid and halo masks for MPI ML decomposition.

Phase 2: ``SpatialDomainGrid`` — 1-D x-slab partitioning from ``box_side_A``
and a manually supplied ``n_ranks``.

Phase 3: ``DomdecAlignedGrid`` — reads ``ndomx`` from ``domdec_common`` via
ctypes (see ``domdec_atoms``), instantiates ``SpatialDomainGrid`` with
``n_ranks = ndomx``, and — when the full Fortran arrays are available —
exposes ``get_local_atom_indices()`` / ``get_ghost_atom_indices()`` to skip
the COM computation entirely.

Design note on alignment
------------------------
With NDIR N 1 1 (all splits along *x*), CHARMM assigns rank ``r`` to the
x-domain ``[r/N · Lx, (r+1)/N · Lx)``, identical to ``SpatialDomainGrid``
with ``n_ranks = N``.  The two decompositions are therefore *already aligned*
— the only new code needed is to auto-read ``N`` from Fortran rather than
requiring the caller to hard-code it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Phase 3 — DomdecAlignedGrid
# ---------------------------------------------------------------------------

@dataclass
class DomdecAlignedGrid:
    """Phase 3 spatial grid that auto-reads NDIR from CHARMM's ``domdec_common``.

    On construction (or on each call to ``refresh()``) it queries:

    - ``is_domdec_active()`` → whether to use NDIR or fall back to ``n_ranks``
    - ``get_ndir()`` → ``(Nx, Ny, Nz)``

    For NDIR ``N 1 1`` the inner ``SpatialDomainGrid`` is created with
    ``n_ranks = N``, giving an **identical** x-slab partitioning to DOMDEC.

    For non-1D NDIR (``Ny > 1`` or ``Nz > 1``), a ``ValueError`` is raised
    unless ``allow_nd=True``, in which case only the x-axis is used and a
    warning is emitted.

    Fast-path atom index access
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``use_ctypes_arrays=True`` (default) and the Fortran array symbols are
    present, ``get_local_atom_indices()`` and ``get_ghost_atom_indices()`` read
    directly from ``iiml`` / ``iimf`` without any COM computation::

        grid = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=12.0)
        local = grid.get_local_atom_indices()   # np.int32, 0-based
        ghost = grid.get_ghost_atom_indices()   # np.int32, 0-based

    Fallback
    ~~~~~~~~
    When DOMDEC is inactive, symbols are absent, or ``KEY_DOMDEC=0``, the grid
    falls back to ``n_ranks_fallback`` (default 1, equivalent to Tier 2 serial
    MLpot on a single rank).
    """

    box_side_A: float
    halo_radius_A: float
    n_ranks_fallback: int = 1
    allow_nd: bool = False
    use_ctypes_arrays: bool = True

    # Internal state — populated by _build()
    _grid: SpatialDomainGrid = field(init=False, repr=False)
    _ndir: tuple[int, int, int] = field(init=False, repr=False)
    _domdec_active: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        """(Re)query DOMDEC state and rebuild the inner ``SpatialDomainGrid``."""
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ndir,
            is_domdec_active,
        )

        self._domdec_active = is_domdec_active()
        if self._domdec_active:
            nx, ny, nz = get_ndir()
            if (ny > 1 or nz > 1) and not self.allow_nd:
                raise ValueError(
                    f"DomdecAlignedGrid: NDIR {nx} {ny} {nz} has >1 axis along y or z. "
                    "Only NDIR N 1 1 is supported unless allow_nd=True. "
                    "Use allow_nd=True to fall back to x-axis partitioning."
                )
            if ny > 1 or nz > 1:
                import warnings
                warnings.warn(
                    f"DomdecAlignedGrid: NDIR {nx} {ny} {nz} — using x-axis only "
                    f"(n_ranks={nx}). y/z decomposition is ignored.",
                    stacklevel=3,
                )
            n_ranks = max(1, nx)
            self._ndir = (nx, ny, nz)
        else:
            n_ranks = max(1, int(self.n_ranks_fallback))
            self._ndir = (n_ranks, 1, 1)

        self._grid = SpatialDomainGrid(
            box_side_A=float(self.box_side_A),
            n_ranks=n_ranks,
            halo_radius_A=float(self.halo_radius_A),
        )

    def refresh(self) -> None:
        """Re-query ``domdec_common`` and rebuild the grid.

        Call this at the start of each MD step if NDIR may change (unusual).
        """
        self._build()

    # ------------------------------------------------------------------
    # Forwarded SpatialDomainGrid interface
    # ------------------------------------------------------------------

    @property
    def grid(self) -> SpatialDomainGrid:
        """The underlying ``SpatialDomainGrid`` (read-only)."""
        return self._grid

    @property
    def ndir(self) -> tuple[int, int, int]:
        """Active ``(Nx, Ny, Nz)`` as read from DOMDEC (or fallback)."""
        return self._ndir

    @property
    def domdec_active(self) -> bool:
        """``True`` when CHARMM DOMDEC was active at construction / last ``refresh()``."""
        return self._domdec_active

    def owned_monomer_mask(self, coms: np.ndarray, rank: int) -> np.ndarray:
        """Delegate to inner grid."""
        return self._grid.owned_monomer_mask(coms, rank)

    def monomers_in_extended_domain(self, coms: np.ndarray, rank: int) -> np.ndarray:
        """Delegate to inner grid."""
        return self._grid.monomers_in_extended_domain(coms, rank)

    def rank_for_com(self, com: np.ndarray) -> int:
        """Delegate to inner grid."""
        return self._grid.rank_for_com(com)

    # ------------------------------------------------------------------
    # Phase 3 fast-path: ctypes atom index access
    # ------------------------------------------------------------------

    def get_local_atom_indices(self) -> np.ndarray:
        """0-based atom indices owned by the calling MPI rank.

        Uses ``domdec_atoms.get_local_atom_indices()`` when DOMDEC is active
        and the Fortran symbols are present.  Returns an empty array otherwise.
        """
        if not (self._domdec_active and self.use_ctypes_arrays):
            return np.empty(0, dtype=np.int32)
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_indices,
        )

        return get_local_atom_indices()

    def get_ghost_atom_indices(self) -> np.ndarray:
        """0-based atom indices of ghost atoms visible to the calling MPI rank.

        Uses ``domdec_atoms.get_ghost_atom_indices()`` when DOMDEC is active
        and the Fortran symbols are present.  Returns an empty array otherwise.
        """
        if not (self._domdec_active and self.use_ctypes_arrays):
            return np.empty(0, dtype=np.int32)
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ghost_atom_indices,
        )

        return get_ghost_atom_indices()

    def molecules_owned_by_this_rank(
        self,
        local_atom_indices: np.ndarray,
        atoms_per_monomer: Union[int, Sequence[int]],
        n_monomers: int,
    ) -> np.ndarray:
        """Map local atom indices → 0-based monomer indices owned by this rank.

        Atoms are mapped to their monomer using ``atoms_per_monomer``.  A
        monomer is *owned* if **all** its atoms appear in ``local_atom_indices``.

        Parameters
        ----------
        local_atom_indices:
            0-based atom indices as returned by ``get_local_atom_indices()``.
        atoms_per_monomer:
            Uniform int or per-monomer list.
        n_monomers:
            Total number of monomers in the system.

        Returns
        -------
        np.ndarray, dtype int32
            0-based monomer indices fully owned by this rank.
        """
        return _atoms_to_monomers(local_atom_indices, atoms_per_monomer, n_monomers)

    def molecules_in_ghost_halo(
        self,
        ghost_atom_indices: np.ndarray,
        local_atom_indices: np.ndarray,
        atoms_per_monomer: Union[int, Sequence[int]],
        n_monomers: int,
    ) -> np.ndarray:
        """Map ghost atom indices → monomer indices in the halo (not owned).

        Returns monomers that have *at least one* ghost atom and are NOT fully
        owned (i.e. not in ``molecules_owned_by_this_rank``).

        Parameters
        ----------
        ghost_atom_indices:
            0-based ghost atom indices from ``get_ghost_atom_indices()``.
        local_atom_indices:
            0-based local atom indices from ``get_local_atom_indices()``.
        atoms_per_monomer, n_monomers:
            As for ``molecules_owned_by_this_rank``.
        """
        owned = set(
            _atoms_to_monomers(local_atom_indices, atoms_per_monomer, n_monomers).tolist()
        )
        ghost_monomers = _atoms_to_monomer_set(
            ghost_atom_indices, atoms_per_monomer, n_monomers
        )
        halo = sorted(m for m in ghost_monomers if m not in owned)
        return np.asarray(halo, dtype=np.int32)


def _atom_to_monomer_index(
    atom_idx: int,
    atoms_per_monomer: Union[int, Sequence[int]],
) -> int:
    """Return monomer index (0-based) for a 0-based atom index."""
    if isinstance(atoms_per_monomer, int):
        return atom_idx // atoms_per_monomer
    per = list(atoms_per_monomer)
    offset = 0
    for mi, apm in enumerate(per):
        if atom_idx < offset + apm:
            return mi
        offset += apm
    raise IndexError(f"atom_idx {atom_idx} out of range for {len(per)} monomers")


def _atoms_to_monomers(
    atom_indices: np.ndarray,
    atoms_per_monomer: Union[int, Sequence[int]],
    n_monomers: int,
) -> np.ndarray:
    """Return 0-based monomer indices **fully** covered by ``atom_indices``."""
    if atom_indices.size == 0:
        return np.empty(0, dtype=np.int32)

    if isinstance(atoms_per_monomer, int):
        apm = int(atoms_per_monomer)
        # monomer m is owned iff all atoms [m*apm, (m+1)*apm) are present
        present = np.zeros(n_monomers * apm, dtype=bool)
        valid = atom_indices[atom_indices < n_monomers * apm]
        present[valid] = True
        owned = []
        for m in range(n_monomers):
            if np.all(present[m * apm : (m + 1) * apm]):
                owned.append(m)
        return np.asarray(owned, dtype=np.int32)

    per = [int(x) for x in atoms_per_monomer]
    offsets = np.zeros(len(per) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(per)
    present_set = set(int(x) for x in atom_indices)
    owned = [
        m
        for m, (a, b) in enumerate(zip(offsets[:-1], offsets[1:]))
        if all(i in present_set for i in range(int(a), int(b)))
    ]
    return np.asarray(owned, dtype=np.int32)


def _atoms_to_monomer_set(
    atom_indices: np.ndarray,
    atoms_per_monomer: Union[int, Sequence[int]],
    n_monomers: int,
) -> set[int]:
    """Return the set of monomer indices that have at least one atom in ``atom_indices``."""
    if atom_indices.size == 0:
        return set()
    if isinstance(atoms_per_monomer, int):
        apm = int(atoms_per_monomer)
        return {int(a) // apm for a in atom_indices if int(a) < n_monomers * apm}
    per = [int(x) for x in atoms_per_monomer]
    offsets = np.zeros(len(per) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(per)
    result: set[int] = set()
    for a in atom_indices:
        ai = int(a)
        for m, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            if int(lo) <= ai < int(hi):
                result.add(m)
                break
    return result
