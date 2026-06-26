"""Unified MM neighbor-list backends (Vesin, cell-list, jax-md).

``mm_nl_backend`` selects the builder for static PBC lists and jax-md overflow
fallback.  JAX-MD incremental updates remain in ``jax_md_neighbor_list.py``;
rebuild paths delegate here for consistency.
"""

from __future__ import annotations

import os
from typing import Literal, Protocol, Sequence

import numpy as np

from mmml.interfaces.pycharmmInterface.nl_reference import (
    cell_matrix_3x3,
    have_vesin,
    monomer_id_from_offsets,
    vesin_mic_pairs,
)

MmNlBackendName = Literal["auto", "vesin", "cell_list", "jax_md"]

try:
    from mmml.interfaces.pycharmmInterface.cell_list import (
        PairListTruncationError,
        cell_list_pairs,
        estimate_max_pairs,
    )
except Exception:
    PairListTruncationError = RuntimeError  # type: ignore[misc, assignment]
    cell_list_pairs = None  # type: ignore[assignment]
    estimate_max_pairs = None  # type: ignore[assignment]

try:
    from mmml.interfaces.pycharmmInterface.jax_md_neighbor_list import have_jax_md
except Exception:
    def have_jax_md() -> bool:
        return False


def resolve_mm_nl_backend(name: str | None = None) -> MmNlBackendName:
    """Resolve backend name from argument, ``MMML_MM_NL_BACKEND`` env, or ``auto``."""
    raw = (name or os.environ.get("MMML_MM_NL_BACKEND", "auto")).strip().lower()
    if raw in ("auto", "vesin", "cell_list", "jax_md"):
        return raw  # type: ignore[return-value]
    raise ValueError(
        f"mm_nl_backend must be auto|vesin|cell_list|jax_md; got {name!r}"
    )


def pick_static_rebuild_backend(
    requested: str | None = None,
    *,
    use_jax_md_neighbor_list: bool = True,
) -> MmNlBackendName:
    """Choose backend for static lists, rebuild updates, and jax-md overflow fallback."""
    name = resolve_mm_nl_backend(requested)
    if name == "auto":
        if have_vesin():
            return "vesin"
        if use_jax_md_neighbor_list and have_jax_md():
            return "jax_md"
        return "cell_list"
    if name == "jax_md" and not (use_jax_md_neighbor_list and have_jax_md()):
        if have_vesin():
            return "vesin"
        return "cell_list"
    if name == "vesin" and not have_vesin():
        return "cell_list"
    return name


class NeighborListBackend(Protocol):
    def build_pairs(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        *,
        cutoff: float,
        monomer_offsets: np.ndarray,
        atoms_per_monomer_list: Sequence[int] | None = None,
        mm_r_min: float | None = None,
        max_pairs: int | None = None,
        cell_list_safety_factor: float = 2.5,
        cell_list_density_estimate: float | None = None,
        total_atoms: int | None = None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Return (pair_i, pair_j, mask, n_valid, capacity)."""


def _pad_pairs(
    pair_set: set[tuple[int, int]],
    *,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n_valid = len(pair_set)
    if n_valid > max_pairs:
        raise PairListTruncationError(n_valid, max_pairs)
    pair_i = np.zeros(max_pairs, dtype=np.int32)
    pair_j = np.zeros(max_pairs, dtype=np.int32)
    mask = np.zeros(max_pairs, dtype=bool)
    for k, (ai, aj) in enumerate(sorted(pair_set)):
        pair_i[k] = ai
        pair_j[k] = aj
        mask[k] = True
    return pair_i, pair_j, mask, n_valid


def _resolve_max_pairs(
    *,
    total_atoms: int,
    box: np.ndarray,
    cutoff: float,
    max_pairs: int | None,
    cell_list_safety_factor: float,
    cell_list_density_estimate: float | None,
) -> int:
    if max_pairs is not None:
        return int(max_pairs)
    if estimate_max_pairs is None:
        return max(256, total_atoms * 32)
    from mmml.interfaces.pycharmmInterface.cell_list import cubic_box_side_from_cell_matrix

    side = cubic_box_side_from_cell_matrix(np.asarray(box))
    return int(
        estimate_max_pairs(
            total_atoms,
            cutoff=cutoff,
            safety_factor=cell_list_safety_factor,
            density_estimate=cell_list_density_estimate,
            box_side_A=side,
        )
    )


class VesinBackend:
    """Vesin half-list + MM monomer/COM filters."""

    def build_pairs(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        *,
        cutoff: float,
        monomer_offsets: np.ndarray,
        atoms_per_monomer_list: Sequence[int] | None = None,
        mm_r_min: float | None = None,
        max_pairs: int | None = None,
        cell_list_safety_factor: float = 2.5,
        cell_list_density_estimate: float | None = None,
        total_atoms: int | None = None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if not have_vesin():
            raise RuntimeError("VesinBackend requires vesin (pip install vesin)")
        R = np.asarray(positions, dtype=np.float64)
        n_atoms = int(total_atoms if total_atoms is not None else R.shape[0])
        offsets = np.asarray(monomer_offsets, dtype=np.int32)
        monomer_id = monomer_id_from_offsets(offsets, n_atoms)
        cell_mat = cell_matrix_3x3(box)
        pair_set = vesin_mic_pairs(
            R,
            cell_mat,
            cutoff,
            monomer_id,
            mm_r_min=mm_r_min,
            monomer_offsets=offsets,
        )
        capacity = _resolve_max_pairs(
            total_atoms=n_atoms,
            box=cell_mat,
            cutoff=cutoff,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            cell_list_density_estimate=cell_list_density_estimate,
        )
        pair_i, pair_j, mask, n_valid = _pad_pairs(pair_set, max_pairs=capacity)
        if debug:
            print(f"[nl_backend:vesin] n_valid={n_valid} capacity={capacity}")
        return pair_i, pair_j, mask, n_valid, capacity


class CellListBackend:
    """Wrapper around ``cell_list_pairs`` with retry-friendly capacity."""

    def build_pairs(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        *,
        cutoff: float,
        monomer_offsets: np.ndarray,
        atoms_per_monomer_list: Sequence[int] | None = None,
        mm_r_min: float | None = None,
        max_pairs: int | None = None,
        cell_list_safety_factor: float = 2.5,
        cell_list_density_estimate: float | None = None,
        total_atoms: int | None = None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        if cell_list_pairs is None:
            raise RuntimeError("CellListBackend requires cell_list module")
        R = np.asarray(positions, dtype=np.float64)
        n_atoms = int(total_atoms if total_atoms is not None else R.shape[0])
        offsets = np.asarray(monomer_offsets, dtype=np.int32)
        cell_mat = cell_matrix_3x3(box)
        capacity = _resolve_max_pairs(
            total_atoms=n_atoms,
            box=cell_mat,
            cutoff=cutoff,
            max_pairs=max_pairs,
            cell_list_safety_factor=cell_list_safety_factor,
            cell_list_density_estimate=cell_list_density_estimate,
        )
        last_exc: PairListTruncationError | None = None
        for attempt in range(4):
            try:
                cl_i, cl_j, cl_mask, n_valid = cell_list_pairs(
                    R,
                    cell_mat,
                    cutoff=cutoff,
                    max_pairs=capacity,
                    monomer_offsets=offsets,
                    atoms_per_monomer_list=list(atoms_per_monomer_list or []),
                    exclude_intra_monomer=True,
                    suppress_warning=(attempt < 3),
                )
                mask = np.asarray(cl_mask, dtype=bool)
                if mm_r_min is not None:
                    from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
                        _filter_pairs_by_com_min,
                    )

                    monomer_id = monomer_id_from_offsets(offsets, n_atoms)
                    mask = _filter_pairs_by_com_min(
                        R,
                        np.asarray(cl_i),
                        np.asarray(cl_j),
                        mask,
                        offsets,
                        monomer_id,
                        float(mm_r_min),
                        pbc_cell=cell_mat,
                    )
                if debug and attempt > 0:
                    print(f"[nl_backend:cell_list] autoscale attempt {attempt} capacity={capacity}")
                return (
                    np.asarray(cl_i, dtype=np.int32),
                    np.asarray(cl_j, dtype=np.int32),
                    np.asarray(mask, dtype=bool),
                    int(n_valid),
                    int(capacity),
                )
            except PairListTruncationError as exc:
                last_exc = exc
                capacity = int(exc.suggested_max_pairs)
        assert last_exc is not None
        raise last_exc


def get_neighbor_list_backend(name: MmNlBackendName) -> NeighborListBackend:
    if name == "vesin":
        return VesinBackend()
    if name == "cell_list":
        return CellListBackend()
    if name == "jax_md":
        raise ValueError("jax_md backend is not a rebuild builder; use create_jax_md_neighbor_list")
    raise ValueError(f"unknown backend {name!r}")


def build_mm_pairs_with_backend(
    backend_name: MmNlBackendName,
    positions: np.ndarray,
    box: np.ndarray,
    *,
    cutoff: float,
    monomer_offsets: np.ndarray,
    atoms_per_monomer_list: Sequence[int] | None = None,
    mm_r_min: float | None = None,
    max_pairs: int | None = None,
    cell_list_safety_factor: float = 2.5,
    cell_list_density_estimate: float | None = None,
    total_atoms: int | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    """Build pairs, trying Vesin then cell-list when ``auto``/``vesin`` fails."""
    if backend_name == "vesin":
        order: list[MmNlBackendName] = ["vesin", "cell_list"]
    elif backend_name == "cell_list":
        order = ["cell_list"]
    else:
        order = ["vesin", "cell_list"] if have_vesin() else ["cell_list"]

    last_err: Exception | None = None
    for name in order:
        try:
            backend = get_neighbor_list_backend(name)
            pi, pj, mask, n_valid, capacity = backend.build_pairs(
                positions,
                box,
                cutoff=cutoff,
                monomer_offsets=monomer_offsets,
                atoms_per_monomer_list=atoms_per_monomer_list,
                mm_r_min=mm_r_min,
                max_pairs=max_pairs,
                cell_list_safety_factor=cell_list_safety_factor,
                cell_list_density_estimate=cell_list_density_estimate,
                total_atoms=total_atoms,
                debug=debug,
            )
            return pi, pj, mask, n_valid, capacity, name
        except Exception as exc:
            last_err = exc
            if debug:
                print(f"[nl_backend] {name} failed: {exc}")
            continue
    raise RuntimeError(f"all neighbor-list backends failed: {last_err}")
