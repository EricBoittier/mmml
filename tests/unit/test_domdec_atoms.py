"""Unit tests for mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms.

Three paths are exercised via mocked ctypes:

1. **No libcharmm** — ``_get_libcharmm`` returns ``None``.
   All public functions return their safe defaults.

2. **DOMDEC compiled out** (``KEY_DOMDEC=0``) — ``q_domdec`` symbol present
   but value is 0 (False); NDIR scalars missing (``OSError``).

3. **DOMDEC active** (``KEY_DOMDEC=1``) — all scalars and array descriptors
   present; ``get_local_atom_indices`` / ``get_ghost_atom_indices`` return
   correct 0-based arrays.

Tests for ``DomdecAlignedGrid`` cover:

- No-libcharmm fallback (uses ``n_ranks_fallback``).
- DOMDEC inactive fallback (same).
- NDIR 8 1 1 → inner SpatialDomainGrid has n_ranks=8.
- NDIR 2 2 2 with ``allow_nd=False`` raises ``ValueError``.
- NDIR 2 2 2 with ``allow_nd=True`` warns but succeeds (n_ranks=2).
- ``molecules_owned_by_this_rank`` / ``molecules_in_ghost_halo`` correctness.
"""

from __future__ import annotations

import ctypes
import struct
import warnings
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake ctypes objects that look like module variables
# ---------------------------------------------------------------------------


def _make_int32_in_dll(value: int):
    """Return a ctypes c_int32 holding *value* (simulates in_dll return)."""
    return ctypes.c_int32(value)


def _pack_gf1d_descriptor(
    base_addr: int,
    n: int,
    lbound: int = 1,
    span: int = 4,
) -> bytes:
    """Pack a fake gfortran 1-D array descriptor as raw bytes (8-byte fields)."""
    # Layout: base_addr, offset, dtype, span, dim0_stride, dim0_lbound, dim0_ubound
    ubound = lbound + n - 1
    offset = lbound  # typical gfortran 1-based offset
    dtype = 0        # don't care in tests
    stride = 1
    return struct.pack("<QQQQQQQ", base_addr, offset, dtype, span, stride, lbound, ubound)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_libcharmm():
    """Patch _get_libcharmm to return None (library not loaded)."""
    with patch(
        "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms._get_libcharmm",
        return_value=None,
    ) as m:
        yield m


def _make_lib_with_scalars(
    q_domdec: int = 0,
    ndomx: int = 1,
    ndomy: int = 1,
    ndomz: int = 1,
    natoml: int = 0,
    natom_foreign: int = 0,
    iiml_present: bool = False,
    iimf_present: bool = False,
    iiml_data: Optional[np.ndarray] = None,
    iimf_data: Optional[np.ndarray] = None,
) -> MagicMock:
    """Build a mock libcharmm handle with configurable symbol responses."""
    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial import domdec_atoms

    scalar_map = {
        domdec_atoms._SYM_Q_DOMDEC: q_domdec,
        domdec_atoms._SYM_NDOMX: ndomx,
        domdec_atoms._SYM_NDOMY: ndomy,
        domdec_atoms._SYM_NDOMZ: ndomz,
        domdec_atoms._SYM_NATOML: natoml,
        domdec_atoms._SYM_NATOM_FOREIGN: natom_foreign,
    }

    lib = MagicMock()

    # Build actual ctypes c_int32 buffers so in_dll-style access works
    # We patch c_int32.in_dll and _GF1DArrayDescriptor.in_dll separately
    return lib, scalar_map, iiml_data, iimf_data


# ---------------------------------------------------------------------------
# 1. No libcharmm — safe defaults
# ---------------------------------------------------------------------------


class TestNoLibcharmm:
    def test_is_domdec_active_false(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            is_domdec_active,
        )

        assert is_domdec_active() is False

    def test_get_ndir_ones(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ndir,
        )

        assert get_ndir() == (1, 1, 1)

    def test_get_local_atom_count_zero(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_count,
        )

        assert get_local_atom_count() == 0

    def test_get_ghost_atom_count_zero(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ghost_atom_count,
        )

        assert get_ghost_atom_count() == 0

    def test_get_local_atom_indices_empty(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_indices,
        )

        arr = get_local_atom_indices()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.int32
        assert arr.size == 0

    def test_get_ghost_atom_indices_empty(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ghost_atom_indices,
        )

        arr = get_ghost_atom_indices()
        assert arr.size == 0

    def test_discover_symbols_no_lib(self, no_libcharmm):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            discover_domdec_symbols,
        )

        result = discover_domdec_symbols()
        assert "_libcharmm_not_found" in result
        assert result["_libcharmm_not_found"] is False


# ---------------------------------------------------------------------------
# 2. DOMDEC compiled out (q_domdec=0, NDIR symbols raise OSError)
# ---------------------------------------------------------------------------


def _mock_in_dll_domdec_off(sym_name: str, scalar_map: dict):
    """Return a patched c_int32.in_dll that serves scalars or raises OSError."""

    def _in_dll(lib_handle, sym):
        if sym in scalar_map:
            return ctypes.c_int32(scalar_map[sym])
        raise OSError(f"symbol not found: {sym}")

    return _in_dll


class TestDomdecCompiledOut:
    @pytest.fixture()
    def lib_domdec_off(self):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial import domdec_atoms

        scalar_map = {
            domdec_atoms._SYM_Q_DOMDEC: 0,  # .FALSE.
        }

        fake_lib = MagicMock()

        def _in_dll_scalar(lib_handle, sym):
            if sym in scalar_map:
                return ctypes.c_int32(scalar_map[sym])
            raise OSError(f"symbol not found: {sym}")

        with (
            patch(
                "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms._get_libcharmm",
                return_value=fake_lib,
            ),
            patch.object(ctypes.c_int32, "in_dll", staticmethod(_in_dll_scalar)),
        ):
            yield

    def test_is_domdec_active_false(self, lib_domdec_off):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            is_domdec_active,
        )

        assert is_domdec_active() is False

    def test_get_ndir_fallback(self, lib_domdec_off):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ndir,
        )

        assert get_ndir() == (1, 1, 1)

    def test_local_indices_empty(self, lib_domdec_off):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_indices,
        )

        assert get_local_atom_indices().size == 0


# ---------------------------------------------------------------------------
# 3. DOMDEC active — scalars + arrays populated
# ---------------------------------------------------------------------------


class TestDomdecActive:
    """Simulate an active DOMDEC run with 8 local atoms and 3 ghost atoms."""

    N_LOCAL = 8
    N_GHOST = 3
    # 1-based atom indices that CHARMM stores
    LOCAL_1BASED = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    GHOST_1BASED = np.array([9, 10, 11], dtype=np.int32)

    @pytest.fixture()
    def lib_domdec_active(self):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial import domdec_atoms

        scalar_map = {
            domdec_atoms._SYM_Q_DOMDEC: 1,
            domdec_atoms._SYM_NDOMX: 8,
            domdec_atoms._SYM_NDOMY: 1,
            domdec_atoms._SYM_NDOMZ: 1,
            domdec_atoms._SYM_NATOML: self.N_LOCAL,
            domdec_atoms._SYM_NATOM_FOREIGN: self.N_GHOST,
        }

        # Build real ctypes buffers for the arrays so from_address works
        local_buf = (ctypes.c_int32 * self.N_LOCAL)(*self.LOCAL_1BASED.tolist())
        ghost_buf = (ctypes.c_int32 * self.N_GHOST)(*self.GHOST_1BASED.tolist())
        local_addr = ctypes.addressof(local_buf)
        ghost_addr = ctypes.addressof(ghost_buf)

        fake_lib = MagicMock()

        def _in_dll_scalar(lib_handle, sym):
            if sym in scalar_map:
                return ctypes.c_int32(scalar_map[sym])
            raise OSError(f"symbol not found: {sym}")

        def _in_dll_desc(lib_handle, sym):
            if sym == domdec_atoms._SYM_IIML:
                desc = domdec_atoms._GF1DArrayDescriptor()
                desc.base_addr = local_addr
                desc.dim0_lbound = 1
                desc.dim0_ubound = self.N_LOCAL
                desc.span = 4
                return desc
            if sym == domdec_atoms._SYM_IIMF:
                desc = domdec_atoms._GF1DArrayDescriptor()
                desc.base_addr = ghost_addr
                desc.dim0_lbound = 1
                desc.dim0_ubound = self.N_GHOST
                desc.span = 4
                return desc
            raise OSError(f"descriptor not found: {sym}")

        with (
            patch(
                "mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms._get_libcharmm",
                return_value=fake_lib,
            ),
            patch.object(ctypes.c_int32, "in_dll", staticmethod(_in_dll_scalar)),
            patch.object(
                domdec_atoms._GF1DArrayDescriptor,
                "in_dll",
                staticmethod(_in_dll_desc),
            ),
        ):
            # Keep buffers alive for the duration of the test
            self._local_buf = local_buf
            self._ghost_buf = ghost_buf
            yield

    def test_is_domdec_active(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            is_domdec_active,
        )

        assert is_domdec_active() is True

    def test_get_ndir(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ndir,
        )

        assert get_ndir() == (8, 1, 1)

    def test_get_local_atom_count(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_count,
        )

        assert get_local_atom_count() == self.N_LOCAL

    def test_get_ghost_atom_count(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ghost_atom_count,
        )

        assert get_ghost_atom_count() == self.N_GHOST

    def test_local_atom_indices_0based(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_local_atom_indices,
        )

        arr = get_local_atom_indices()
        expected = self.LOCAL_1BASED - 1  # 0-based
        np.testing.assert_array_equal(arr, expected)

    def test_ghost_atom_indices_0based(self, lib_domdec_active):
        from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domdec_atoms import (
            get_ghost_atom_indices,
        )

        arr = get_ghost_atom_indices()
        expected = self.GHOST_1BASED - 1  # 0-based
        np.testing.assert_array_equal(arr, expected)


# ---------------------------------------------------------------------------
# DomdecAlignedGrid tests (patching domdec_atoms functions directly)
# ---------------------------------------------------------------------------


def _patch_domdec_atoms(
    active: bool = False,
    ndir: tuple[int, int, int] = (1, 1, 1),
    local_indices: Optional[np.ndarray] = None,
    ghost_indices: Optional[np.ndarray] = None,
):
    """Context manager that patches all domdec_atoms public functions."""
    import contextlib

    from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial import domdec_atoms

    @contextlib.contextmanager
    def _ctx():
        with (
            patch.object(domdec_atoms, "is_domdec_active", return_value=active),
            patch.object(domdec_atoms, "get_ndir", return_value=ndir),
            patch.object(
                domdec_atoms,
                "get_local_atom_indices",
                return_value=(local_indices if local_indices is not None else np.empty(0, dtype=np.int32)),
            ),
            patch.object(
                domdec_atoms,
                "get_ghost_atom_indices",
                return_value=(ghost_indices if ghost_indices is not None else np.empty(0, dtype=np.int32)),
            ),
        ):
            yield

    return _ctx()


class TestDomdecAlignedGridFallback:
    def test_no_domdec_uses_fallback_n_ranks(self):
        with _patch_domdec_atoms(active=False, ndir=(1, 1, 1)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0, n_ranks_fallback=4)
            assert g.grid.n_ranks == 4
            assert g.domdec_active is False
            assert g.ndir == (4, 1, 1)

    def test_domdec_active_ndir_8_1_1(self):
        with _patch_domdec_atoms(active=True, ndir=(8, 1, 1)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            assert g.grid.n_ranks == 8
            assert g.domdec_active is True
            assert g.ndir == (8, 1, 1)

    def test_ndir_3d_raises_without_allow_nd(self):
        with _patch_domdec_atoms(active=True, ndir=(2, 2, 2)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            with pytest.raises(ValueError, match="allow_nd"):
                DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0, allow_nd=False)

    def test_ndir_3d_allow_nd_uses_nx(self):
        with _patch_domdec_atoms(active=True, ndir=(2, 2, 2)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                g = DomdecAlignedGrid(
                    box_side_A=50.0, halo_radius_A=8.0, allow_nd=True
                )
            assert g.grid.n_ranks == 2
            assert any("x-axis only" in str(warning.message) for warning in w)

    def test_get_local_atom_indices_inactive(self):
        with _patch_domdec_atoms(active=False):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            assert g.get_local_atom_indices().size == 0

    def test_get_local_atom_indices_active(self):
        local = np.array([0, 1, 2, 3], dtype=np.int32)
        with _patch_domdec_atoms(
            active=True, ndir=(8, 1, 1), local_indices=local
        ):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            np.testing.assert_array_equal(g.get_local_atom_indices(), local)


# ---------------------------------------------------------------------------
# molecules_owned_by_this_rank / molecules_in_ghost_halo
# ---------------------------------------------------------------------------


class TestMoleculeOwnership:
    """Verify atom-to-monomer mapping helpers."""

    def test_uniform_apm_owned(self):
        with _patch_domdec_atoms(active=True, ndir=(2, 1, 1)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            # 3 monomers × 3 atoms = 9 atoms total
            # Rank owns atoms 0–5 (monomers 0,1 fully covered; monomer 2 not)
            local = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            owned = g.molecules_owned_by_this_rank(local, atoms_per_monomer=3, n_monomers=3)
            np.testing.assert_array_equal(owned, [0, 1])

    def test_ghost_halo(self):
        with _patch_domdec_atoms(active=True, ndir=(2, 1, 1)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            local = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            ghost = np.array([6, 7, 8], dtype=np.int32)  # monomer 2
            halo = g.molecules_in_ghost_halo(
                ghost, local, atoms_per_monomer=3, n_monomers=3
            )
            np.testing.assert_array_equal(halo, [2])

    def test_variable_apm(self):
        with _patch_domdec_atoms(active=True, ndir=(2, 1, 1)):
            from mmml.interfaces.pycharmmInterface.mlpot.mpi_spatial.domain import (
                DomdecAlignedGrid,
            )

            g = DomdecAlignedGrid(box_side_A=50.0, halo_radius_A=8.0)
            # 2 monomers: first has 2 atoms, second has 3 atoms
            atoms_per_monomer = [2, 3]
            local = np.array([0, 1], dtype=np.int32)   # only monomer 0
            owned = g.molecules_owned_by_this_rank(
                local, atoms_per_monomer=atoms_per_monomer, n_monomers=2
            )
            np.testing.assert_array_equal(owned, [0])
