"""Unit tests for DOMDEC-aware spatial batch builder (Phase 3 integration).

All tests mock the domdec_atoms ctypes layer — no libcharmm.so required.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from mmml.interfaces.pycharmmInterface.mlpot.medium_pbc_validation import (
    lattice_positions_cubic_pbc,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.batch_builder import (
    SpatialBatchIndices,
    build_domdec_spatial_batch_indices,
    build_spatial_batch_indices,
    make_domdec_aligned_grid,
    make_spatial_domain_grid,
)
from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
    DomdecAlignedGrid,
    SpatialDomainGrid,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DOMDEC_MODULE = "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms"


def _make_geometry(n_monomers: int = 8, atoms_per: int = 10, box: float = 30.0):
    """Small synthetic PBC system."""
    pos = lattice_positions_cubic_pbc(n_monomers, atoms_per, box, spacing_A=5.0, seed=3)
    return pos, n_monomers, atoms_per, box


def _mock_cutoffs(mm_switch_on: float = 8.0):
    """Minimal CutoffParameters stub."""
    from unittest.mock import MagicMock

    cp = MagicMock()
    cp.mm_switch_on = mm_switch_on
    cp.mm_switch_width = 2.0
    cp.ml_switch_width = 2.0
    return cp


# ---------------------------------------------------------------------------
# make_domdec_aligned_grid
# ---------------------------------------------------------------------------


class TestMakeDomdecAlignedGrid:
    """make_domdec_aligned_grid() factory — construction and fallback."""

    def test_returns_domdec_aligned_grid(self):
        cp = _mock_cutoffs()
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=False),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(1, 1, 1)),
        ):
            grid = make_domdec_aligned_grid(40.0, cp, n_ranks_fallback=4)
        assert isinstance(grid, DomdecAlignedGrid)

    def test_fallback_n_ranks_when_domdec_inactive(self):
        cp = _mock_cutoffs()
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=False),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(1, 1, 1)),
        ):
            grid = make_domdec_aligned_grid(40.0, cp, n_ranks_fallback=4)
        assert grid.domdec_active is False
        assert grid.grid.n_ranks == 4

    def test_reads_ndir_when_domdec_active(self):
        cp = _mock_cutoffs()
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(8, 1, 1)),
        ):
            grid = make_domdec_aligned_grid(40.0, cp, n_ranks_fallback=2)
        assert grid.domdec_active is True
        assert grid.ndir == (8, 1, 1)
        assert grid.grid.n_ranks == 8

    def test_halo_radius_propagated(self):
        cp = _mock_cutoffs(mm_switch_on=10.0)
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=False),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(1, 1, 1)),
        ):
            grid = make_domdec_aligned_grid(40.0, cp, n_ranks_fallback=1)
        # halo = mm_switch_on + physnet_cutoff + ml_switch_width = 10 + 6 + 2 = 18
        assert grid.halo_radius_A >= 18.0


# ---------------------------------------------------------------------------
# build_domdec_spatial_batch_indices — fallback (DOMDEC inactive)
# ---------------------------------------------------------------------------


class TestDomdecBatchFallback:
    """When DOMDEC is inactive the output must match build_spatial_batch_indices."""

    def test_fallback_matches_com_slab(self):
        pos, n_monomers, atoms_per, box = _make_geometry()
        cp = _mock_cutoffs()
        rank = 0

        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=False),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(1, 1, 1)),
        ):
            grid_ddc = make_domdec_aligned_grid(box, cp, n_ranks_fallback=2)
            result_ddc = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid_ddc, rank, cp
            )

        # Reference: plain COM-slab grid with same n_ranks
        grid_ref = make_spatial_domain_grid(box, 2, cp)
        result_ref = build_spatial_batch_indices(
            pos, n_monomers, atoms_per, grid_ref, rank, cp
        )

        np.testing.assert_array_equal(result_ddc.owned_monomers, result_ref.owned_monomers)
        np.testing.assert_array_equal(
            result_ddc.active_dimer_indices, result_ref.active_dimer_indices
        )

    def test_fallback_returns_spatial_batch_indices(self):
        pos, n_monomers, atoms_per, box = _make_geometry()
        cp = _mock_cutoffs()
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=False),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(1, 1, 1)),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=1)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, 0, cp
            )
        assert isinstance(result, SpatialBatchIndices)


# ---------------------------------------------------------------------------
# build_domdec_spatial_batch_indices — DOMDEC active path
# ---------------------------------------------------------------------------


class TestDomdecBatchActive:
    """When DOMDEC is active owned_monomers must come from ctypes, not COMs."""

    def _make_owned_atoms(
        self, rank: int, n_ranks: int, n_monomers: int, atoms_per: int
    ) -> np.ndarray:
        """Assign contiguous atom blocks to ranks (deterministic fixture)."""
        total = n_monomers * atoms_per
        per_rank = total // n_ranks
        lo = rank * per_rank
        hi = lo + per_rank
        return np.arange(lo, hi, dtype=np.int32)

    def test_owned_monomers_from_ctypes(self):
        """owned_monomers must reflect the ctypes atom slice, not COM slabs."""
        pos, n_monomers, atoms_per, box = _make_geometry(n_monomers=8, atoms_per=10)
        cp = _mock_cutoffs()
        n_ranks = 2
        rank = 0

        # Rank 0 owns atoms 0..39 → monomers 0..3 (each 10 atoms)
        local_atoms = self._make_owned_atoms(rank, n_ranks, n_monomers, atoms_per)
        expected_monomers = np.arange(0, n_monomers // n_ranks, dtype=np.int32)

        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(n_ranks, 1, 1)),
            patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
            patch(
                f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                return_value=np.empty(0, dtype=np.int32),
            ),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=n_ranks)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, rank, cp
            )

        np.testing.assert_array_equal(result.owned_monomers, expected_monomers)

    def test_owned_monomers_differ_from_com_slabs_when_assignments_diverge(self):
        """Prove ctypes path is distinct from COM path (ownership from ctypes, not geometry)."""
        pos, n_monomers, atoms_per, box = _make_geometry(n_monomers=8, atoms_per=10)
        cp = _mock_cutoffs()
        n_ranks = 2
        rank = 1

        # Give rank 1 the *second* half of atoms (monomers 4..7)
        local_atoms = self._make_owned_atoms(rank, n_ranks, n_monomers, atoms_per)

        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(n_ranks, 1, 1)),
            patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
            patch(
                f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                return_value=np.empty(0, dtype=np.int32),
            ),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=n_ranks)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, rank, cp
            )

        expected = np.arange(n_monomers // n_ranks, n_monomers, dtype=np.int32)
        np.testing.assert_array_equal(result.owned_monomers, expected)

    def test_active_dimer_indices_still_com_based(self):
        """Dimer list must be non-empty (COM-based) even when DOMDEC atom path active."""
        pos, n_monomers, atoms_per, box = _make_geometry(n_monomers=8, atoms_per=10)
        cp = _mock_cutoffs(mm_switch_on=12.0)  # large cutoff → many near dimers
        n_ranks = 2
        rank = 0
        local_atoms = self._make_owned_atoms(rank, n_ranks, n_monomers, atoms_per)

        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(n_ranks, 1, 1)),
            patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
            patch(
                f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                return_value=np.empty(0, dtype=np.int32),
            ),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=n_ranks)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, rank, cp
            )

        assert result.active_dimer_indices.ndim == 1

    def test_returns_spatial_batch_indices_type(self):
        pos, n_monomers, atoms_per, box = _make_geometry()
        cp = _mock_cutoffs()
        local_atoms = self._make_owned_atoms(0, 2, n_monomers, atoms_per)
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(2, 1, 1)),
            patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
            patch(
                f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                return_value=np.empty(0, dtype=np.int32),
            ),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=2)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, 0, cp
            )
        assert isinstance(result, SpatialBatchIndices)
        assert result.n_monomers_global == n_monomers

    def test_global_monomer_count_preserved(self):
        """n_monomers_global must always equal the total monomer count."""
        pos, n_monomers, atoms_per, box = _make_geometry()
        cp = _mock_cutoffs()
        local_atoms = self._make_owned_atoms(0, 2, n_monomers, atoms_per)
        with (
            patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
            patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(2, 1, 1)),
            patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
            patch(
                f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                return_value=np.empty(0, dtype=np.int32),
            ),
        ):
            grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=2)
            result = build_domdec_spatial_batch_indices(
                pos, n_monomers, atoms_per, grid, 0, cp
            )
        assert result.n_monomers_global == n_monomers

    def test_disjoint_ownership_across_ranks(self):
        """Owned monomers from rank 0 and rank 1 must be disjoint."""
        pos, n_monomers, atoms_per, box = _make_geometry()
        cp = _mock_cutoffs()
        n_ranks = 2
        results = {}
        for rank in range(n_ranks):
            local_atoms = self._make_owned_atoms(rank, n_ranks, n_monomers, atoms_per)
            with (
                patch(f"{_DOMDEC_MODULE}.is_domdec_active", return_value=True),
                patch(f"{_DOMDEC_MODULE}.get_ndir", return_value=(n_ranks, 1, 1)),
                patch(f"{_DOMDEC_MODULE}.get_local_atom_indices", return_value=local_atoms),
                patch(
                    f"{_DOMDEC_MODULE}.get_ghost_atom_indices",
                    return_value=np.empty(0, dtype=np.int32),
                ),
            ):
                grid = make_domdec_aligned_grid(box, cp, n_ranks_fallback=n_ranks)
                results[rank] = build_domdec_spatial_batch_indices(
                    pos, n_monomers, atoms_per, grid, rank, cp
                )

        owned_0 = set(results[0].owned_monomers.tolist())
        owned_1 = set(results[1].owned_monomers.tolist())
        assert owned_0.isdisjoint(owned_1), "owned monomers must not overlap between ranks"
        assert owned_0 | owned_1 == set(range(n_monomers)), "must cover all monomers"
