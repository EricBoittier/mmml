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
    "CapacityOverflow",
    "shell_capacity",
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
