"""Neighbor-list factory for the driver's block-boundary refresh.

The :class:`~mmml.md.drivers.JaxmdDriver` calls a ``neighbor_fn(real_pos, box)``
at each block boundary and routes the returned arrays into the hybrid energy
(decision B — the driver owns the rebuild cadence, terms own their capacity).
This module builds that callback for the intermolecular MM pair list that
``mm_nonbonded`` consumes: host-side pair construction (``get_intermolecular_pairs``)
padded to a fixed capacity with an overflow guard.

Host-side and numpy-only at call time (it runs between jitted blocks), so it may
use the non-jittable ``_build_pair_indices`` path.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from mmml.md.energy.capacity import check_capacity, shell_capacity
from mmml.md.system import MolecularSystem

__all__ = ["make_intermolecular_neighbor_fn"]


def make_intermolecular_neighbor_fn(
    system: MolecularSystem,
    cutoff_A: float,
    capacity: int | None = None,
    *,
    peptide_water_ml: bool = False,
    on_overflow: str = "raise",
) -> Callable[[np.ndarray, np.ndarray | None], Mapping[str, Any]]:
    """Build a ``neighbor_fn`` yielding padded ``pair_i`` / ``pair_j`` / ``pair_mask``.

    ``capacity`` is the padded pair-slot count; when ``None`` it is estimated from
    the cutoff shell and atom density with headroom. Intramolecular pairs are
    filtered by ``system.mol_id``; exclusions come from ``FFParams``.
    """
    from mmml.interfaces.jaxmdInterface.hybrid_energy import get_intermolecular_pairs

    mol_id = np.asarray(system.mol_id, dtype=np.int32)
    excluded = frozenset()
    if system.ff_params is not None:
        excluded = frozenset(map(tuple, system.ff_params.exclusions.tolist()))

    if capacity is None:
        n = system.n_atoms
        if system.box is not None:
            volume = float(abs(np.linalg.det(np.asarray(system.box))))
            density = n / volume if volume > 0 else 0.0
        else:
            density = 0.0
        # per-atom shell × atoms, generous headroom for a dense pair list
        per_atom = shell_capacity(cutoff_A, max(density, 1e-6), headroom=2.0, minimum=8)
        capacity = int(max(n * per_atom, 16))

    cap = int(capacity)

    def neighbor_fn(real_pos: np.ndarray, box: np.ndarray | None) -> Mapping[str, Any]:
        cell = np.asarray(system.box if box is None else box, dtype=np.float64)
        pi, pj = get_intermolecular_pairs(
            np.asarray(real_pos, dtype=np.float64), cell, excluded, cutoff_A, mol_id,
            peptide_water_ml=peptide_water_ml,
        )
        n_pairs = int(len(pi))
        check_capacity(n_pairs, cap, "intermolecular pairs", on_overflow=on_overflow)
        n_pairs = min(n_pairs, cap)
        pair_i = np.zeros(cap, dtype=np.int32)
        pair_j = np.zeros(cap, dtype=np.int32)
        pair_mask = np.zeros(cap, dtype=np.int8)
        pair_i[:n_pairs] = np.asarray(pi[:n_pairs], dtype=np.int32)
        pair_j[:n_pairs] = np.asarray(pj[:n_pairs], dtype=np.int32)
        pair_mask[:n_pairs] = 1
        return {"pair_i": pair_i, "pair_j": pair_j, "pair_mask": pair_mask}

    return neighbor_fn
