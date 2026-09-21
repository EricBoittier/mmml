"""Host-side choice of how many PhysNet chunks the sparse-dimer forward evaluates.

The sparse ML dimer batch is padded to a static cap (``max_active_dimers``) and
``jnp.nonzero(..., size=cap)`` packs the active dimers first, so only the first
``n_monomers + n_active`` batch slots carry used output. Skipping the chunks
past that count with a device-side ``lax.cond`` (PR #227) makes XLA:GPU copy
each chunk's predicate to the host and block, once per chunk per step.

Here the host picks the number of evaluated chunks instead. It is a static
argument of the jitted forward, so the chunk loop has a compile-time trip count
and the step needs no device-to-host copy until the forces come back. The
forward also returns the step's active-dimer count, which the host reads with
the forces:

* ``covers(n_active)`` says whether this step's result is exact (every used slot
  was evaluated). If not, the caller re-runs the step with the grown budget.
* ``update(n_active)`` sizes the next step: the chunks that hold the used slots
  plus ``headroom_slots`` spare slots (the active count drifts by tens of dimers
  per hundred steps), shrinking only when that is ``shrink_slack_chunks`` or
  more below the current budget (hysteresis: few distinct values, few compiles).
  The headroom is kept small on purpose: a whole spare chunk costs as much
  PhysNet time as the per-chunk syncs it removes. An upward crossing of a chunk
  boundary costs one re-run of that step, and the hysteresis keeps the grown
  budget, so re-runs stay rare.

The first step evaluates every chunk, so it is always exact and yields the
first count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MlChunkLayout:
    """Static geometry of the chunked sparse-dimer PhysNet batch."""

    n_monomers: int
    max_active_dimers: int
    chunk_size: int
    n_chunks: int


def ml_chunk_budget_enabled() -> bool:
    """``MMML_MLPOT_CHUNK_BUDGET=0`` evaluates every chunk (A/B parity checks)."""
    raw = (os.environ.get("MMML_MLPOT_CHUNK_BUDGET") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


@dataclass
class MlChunkBudget:
    layout: MlChunkLayout
    headroom_slots: int = 32
    shrink_slack_chunks: int = 2
    current: int = field(init=False)
    saturated_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.current = int(self.layout.n_chunks)

    def n_valid(self, n_active: int) -> int:
        """Used batch slots: monomers plus in-range dimers (capped)."""
        lay = self.layout
        return lay.n_monomers + min(max(int(n_active), 0), lay.max_active_dimers)

    def needed(self, n_active: int) -> int:
        """Chunks that hold a used slot when ``n_active`` dimers are in range."""
        return max(1, -(-self.n_valid(n_active) // self.layout.chunk_size))

    def covers(self, n_active: int) -> bool:
        return self.needed(n_active) <= self.current

    def update(self, n_active: int) -> int:
        """Budget for the next step (grows at once, shrinks with hysteresis)."""
        lay = self.layout
        slots = self.n_valid(n_active) + max(int(self.headroom_slots), 0)
        target = min(lay.n_chunks, max(1, -(-slots // lay.chunk_size)))
        if target > self.current or self.current - target >= self.shrink_slack_chunks:
            self.current = target
        return self.current

    def note_saturation(self, n_active: int) -> Optional[str]:
        """Host-side replacement for the in-graph cap-saturation warning."""
        cap = self.layout.max_active_dimers
        if int(n_active) <= cap:
            return None
        self.saturated_steps += 1
        if self.saturated_steps not in (1, 10, 100, 1000):
            return None
        return (
            f"mmml WARNING: sparse active-dimer cap saturated: {int(n_active)} in-range "
            f"dimer pairs > cap={cap} ({self.saturated_steps} step(s) so far). "
            f"{int(n_active) - cap} pairs are silently truncated by jnp.nonzero's "
            "fixed-size selection (first-by-enumeration-order, not nearest); this can "
            "discontinuously toggle pairs on/off and inject spurious forces. Raise "
            "ml_max_active_dimers / MMML_MLPOT_MAX_ACTIVE_DIMERS well above the count."
        )
