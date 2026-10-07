"""Bond-graph helpers for internal-coordinate move masks."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def covalent_bond_graph(
    atoms: Atoms,
    *,
    scale: float = 1.2,
) -> dict[int, set[int]]:
    """Undirected adjacency from covalent-radii distance cutoffs."""

    positions = np.asarray(atoms.get_positions(), dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    n_atoms = len(atoms)
    adj: dict[int, set[int]] = defaultdict(set)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            cutoff = scale * (
                float(covalent_radii[numbers[i]]) + float(covalent_radii[numbers[j]])
            )
            if float(np.linalg.norm(positions[i] - positions[j])) < cutoff:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def atoms_on_side(
    adjacency: dict[int, set[int]],
    *,
    seed: int,
    block: int,
) -> list[int]:
    """Return ``seed`` and every atom reachable from it without crossing ``block``.

    For a scanned bond ``block–seed``, this is the rigid fragment that should move
    when the internal coordinate is changed.
    """

    if seed == block:
        raise ValueError("seed and block must differ")
    seen = {seed}
    queue: deque[int] = deque([seed])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, ()):
            if neighbor == block or neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return sorted(seen)


def circular_delta_deg(actual: float, target: float) -> float:
    """Signed shortest difference ``actual - target`` in degrees on (-180, 180]."""

    return (float(actual) - float(target) + 180.0) % 360.0 - 180.0


def angles_match(actual: float, target: float, *, atol_deg: float = 1.0) -> bool:
    return abs(circular_delta_deg(actual, target)) <= float(atol_deg)
