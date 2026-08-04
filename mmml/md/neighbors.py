"""Neighbor-list factory for the driver's block-boundary refresh.

The :class:`~mmml.md.drivers.JaxmdDriver` calls a ``neighbor_fn(real_pos, box)``
at each block boundary and routes the returned arrays into the hybrid energy
(decision B — the driver owns the rebuild cadence, terms own their capacity).
This module builds that callback for the intermolecular MM pair list that
``mm_nonbonded`` consumes: host-side pair construction (``get_intermolecular_pairs``)
padded to a fixed capacity with an overflow guard.

Host-side and numpy-only at call time (it runs between jitted blocks), so it may
use the non-jittable ``_build_pair_indices`` path (Vesin / vectorized NumPy).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from mmml.md.energy.capacity import check_capacity, pair_capacity
from mmml.md.system import MolecularSystem

__all__ = ["make_intermolecular_neighbor_fn"]


def make_intermolecular_neighbor_fn(
    system: MolecularSystem,
    cutoff_A: float,
    capacity: int | None = None,
    *,
    peptide_water_ml: bool = False,
    on_overflow: str = "raise",
    skin_A: float = 0.0,
) -> Callable[[np.ndarray, np.ndarray | None], Mapping[str, Any]]:
    """Build a ``neighbor_fn`` yielding padded ``pair_i`` / ``pair_j`` / ``pair_mask``.

    ``capacity`` is the padded pair-slot count; when ``None`` it is estimated from
    the cutoff shell and atom density with headroom. Intramolecular pairs are
    filtered by ``system.mol_id``; exclusions come from ``FFParams``.

    ``skin_A > 0`` builds the list at ``cutoff_A + skin_A`` and wraps the result
    in :func:`mmml.md.neighbor_cache.with_verlet_skin`, so blocks that move every
    atom less than ``skin_A / 2`` reuse the list instead of paying a host
    rebuild. The extra pairs inside the skin are inert: ``mm_nonbonded`` zeroes
    every pair beyond ``ctofnb``. Default ``0.0`` keeps the previous
    rebuild-every-call behavior.
    """
    from mmml.interfaces.jaxmdInterface.hybrid_energy import get_intermolecular_pairs

    mol_id = np.asarray(system.mol_id, dtype=np.int32)
    excluded = frozenset()
    if system.ff_params is not None:
        excluded = frozenset(map(tuple, system.ff_params.exclusions.tolist()))

    skin = max(0.0, float(skin_A))
    build_cutoff_A = float(cutoff_A) + skin

    if capacity is None:
        n = system.n_atoms
        if system.box is not None:
            volume = float(abs(np.linalg.det(np.asarray(system.box))))
            density = n / volume if volume > 0 else 0.0
        else:
            density = 0.0
        # Size the shell at the *build* cutoff so the skin pairs fit too.
        # pair_capacity halves the per-atom shell count (an unordered pair list
        # holds each pair once) and bounds the result by the pairs that can
        # exist, so PAIR_HEADROOM is the only safety factor and means what it
        # says. See its docstring for what that factor is sized against.
        capacity = pair_capacity(
            n,
            build_cutoff_A,
            density,
            mol_sizes=np.bincount(np.asarray(system.mol_id, dtype=np.int64)),
        )

    cap = int(capacity)

    def neighbor_fn(real_pos: np.ndarray, box: np.ndarray | None) -> Mapping[str, Any]:
        cell = np.asarray(system.box if box is None else box, dtype=np.float64)
        pi, pj = get_intermolecular_pairs(
            np.asarray(real_pos, dtype=np.float64), cell, excluded, build_cutoff_A, mol_id,
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

    if skin > 0.0:
        from mmml.md.neighbor_cache import with_verlet_skin

        return with_verlet_skin(neighbor_fn, skin_A=skin)

    return neighbor_fn
