"""Static-capacity sizing, overflow guards, and the dtype policy for padded terms.

jax energy terms run inside a compile-once graph, so their pair/slot arrays are
padded to a fixed capacity (see ``docs/hybrid-mlmm-decomposition.md`` §5-7). This
module centralizes three things that keep that safe and cheap:

1. **Shell sizing** — pick a capacity from the cutoff-sphere volume × density so
   ``vdw_core`` / ML-dimer costs scale ~linearly with system size, not with the
   box (recommendation 1).
2. **Overflow guards** — never silently drop interactions when the real count
   exceeds capacity; raise (or warn) so the caller grows capacity and accepts one
   recompile (recommendation 2).
3. **Dtype policy** — all *float* compute stays float64 (distances, energies,
   PME-like sums are numerically stiff); only *indices* and *masks* drop
   precision (``int32`` / ``int8``), a host→device bandwidth win with no
   numerical cost.

Dependency-light on purpose (numpy + stdlib only) — no jax import here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = [
    "INDEX_DTYPE",
    "MASK_DTYPE",
    "COMPUTE_DTYPE",
    "DEFAULT_HEADROOM",
    "PAIR_HEADROOM",
    "CapacityOverflow",
    "shell_capacity",
    "pair_capacity",
    "check_capacity",
    "pad_indices",
]

# --- dtype policy -----------------------------------------------------------
# Floats: keep f64 everywhere the energy/force math happens. The repulsive
# (sigma/r)^12 branch and PME-like sums are stiff; f32 risks energy drift.
COMPUTE_DTYPE = np.float64
# Indices and masks are the only low-precision arrays: int32 indices address up
# to ~2.1e9 atoms (far beyond any MD system), and an int8 (0/1) mask promotes to
# the compute dtype on multiply, so it costs nothing numerically while shrinking
# the per-step host->device transfer.
INDEX_DTYPE = np.int32
MASK_DTYPE = np.int8

DEFAULT_HEADROOM = 1.5

#: Safety factor on the *pair* estimate in :func:`pair_capacity`.
#:
#: Chosen from measurement, not inherited. On TIP3P water from 300 to 10 800
#: atoms, the worst live pair count over perturbed and 0.65x-compressed
#: configurations -- a 3.6x density spike, well past anything equilibrium
#: sampling reaches -- needed at most **2.50x** the mean-field estimate, rising
#: with system size (2.00x at 4 800, 2.30x at 7 200, 2.50x at 10 800). At
#: equilibrium the requirement is ~1.0x. 3.0 keeps a fifth in hand above the
#: pathological case; ``scripts/bench_static_vs_neighbor_pairs.py`` and the
#: tests in ``tests/unit/test_md_capacity.py`` are where that is checked.
PAIR_HEADROOM = 3.0


class CapacityOverflow(RuntimeError):
    """Raised when a padded array would need to drop real interactions."""


def shell_capacity(
    cutoff_A: float,
    number_density_per_A3: float,
    *,
    headroom: float = DEFAULT_HEADROOM,
    minimum: int = 8,
) -> int:
    """Padded slot count for molecules within ``cutoff_A`` of a point/group.

    Sizes to the cutoff-sphere volume times the solvent number density, times a
    ``headroom`` safety factor. Use this for ``MAX_ACTIVE_GROUPS`` (and the ML
    dimer capacity) so the dense blocks scale with the cutoff shell, not the box.
    """
    if cutoff_A <= 0 or number_density_per_A3 <= 0:
        raise ValueError("cutoff_A and number_density_per_A3 must be positive")
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0")
    volume = (4.0 / 3.0) * math.pi * cutoff_A**3
    estimate = volume * number_density_per_A3 * headroom
    return max(int(minimum), int(math.ceil(estimate)))


def pair_capacity(
    n_atoms: int,
    cutoff_A: float,
    number_density_per_A3: float,
    *,
    mol_sizes: np.ndarray | None = None,
    headroom: float = PAIR_HEADROOM,
    minimum: int = 16,
) -> int:
    """Padded slot count for an *unordered* intermolecular pair list.

    Two things distinguish this from ``n_atoms * shell_capacity(...)``, which is
    what the neighbour builder used to do.

    **The pair count is half the shell count.** :func:`shell_capacity` counts the
    neighbours of *one* atom; summing that over atoms counts every unordered
    pair twice. The builders emit ``j > i`` only, so the estimate must be
    halved. That factor was never a deliberate margin -- it was a double-count
    that happened to look like one, and it made ``headroom`` mean twice what it
    said.

    **A pair list cannot exceed the pairs that exist.** The shell estimate
    assumes an unbounded medium, so once the cutoff is comparable to the box it
    asks for the impossible: 300 atoms in a 14.4 A box at a 12 A cutoff scored
    434 400 slots for a list that can never hold more than 44 550. Padding is
    not free -- masked slots are still evaluated, because fixed shapes are what
    keeps the kernel jitted -- so the excess is per-step arithmetic thrown away.
    ``mol_sizes`` (atoms per molecule) tightens the bound by the intramolecular
    pairs the builder drops.

    ``headroom`` is now the only safety factor, and :data:`PAIR_HEADROOM`
    records what it is sized against.
    """
    n = int(n_atoms)
    if n < 2:
        return int(minimum)
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0")

    per_atom = shell_capacity(
        cutoff_A, max(float(number_density_per_A3), 1e-6), headroom=1.0, minimum=1
    )
    estimate = math.ceil(n * per_atom / 2.0 * float(headroom))

    max_possible = n * (n - 1) // 2
    if mol_sizes is not None:
        sizes = np.asarray(mol_sizes, dtype=np.int64)
        max_possible -= int((sizes * (sizes - 1) // 2).sum())
    max_possible = max(int(max_possible), 1)

    # The exact bound is applied last: the minimum is there so a tiny system
    # still gets a sane buffer, not a licence to allocate impossible pairs.
    return int(min(max(estimate, int(minimum)), max_possible))


def check_capacity(
    n_required: int,
    capacity: int,
    name: str = "",
    *,
    on_overflow: str = "raise",
) -> None:
    """Guard against silently dropping interactions when ``n_required > capacity``.

    ``on_overflow="raise"`` (default) raises :class:`CapacityOverflow`;
    ``"warn"`` emits a warning; ``"ignore"`` is a no-op (use only when the caller
    has already sized capacity from the same neighbor list).
    """
    if n_required <= capacity:
        return
    msg = (
        f"{name or 'capacity'} overflow: {n_required} required > {capacity} "
        f"allocated; interactions would be dropped. Increase the capacity and "
        f"accept one recompile rather than truncating."
    )
    if on_overflow == "raise":
        raise CapacityOverflow(msg)
    if on_overflow == "warn":
        warnings.warn(msg, stacklevel=2)


def pad_indices(indices, capacity: int, *, fill: int = 0, on_overflow: str = "raise"):
    """Pad a 1-D index array to ``capacity``; return ``(padded_int32, mask_int8)``.

    The mask is 1 for real entries and 0 for padding, in the compact
    :data:`MASK_DTYPE`. Padded index slots are set to ``fill`` (default atom 0);
    terms must clamp distances for masked entries so ``sqrt``/``1/r`` never see a
    true zero (NaN-gradient guard).
    """
    idx = np.asarray(indices, dtype=INDEX_DTYPE).reshape(-1)
    n = int(idx.shape[0])
    check_capacity(n, capacity, "pad_indices", on_overflow=on_overflow)
    n = min(n, capacity)
    padded = np.full((capacity,), fill, dtype=INDEX_DTYPE)
    padded[:n] = idx[:n]
    mask = np.zeros((capacity,), dtype=MASK_DTYPE)
    mask[:n] = 1
    return padded, mask
