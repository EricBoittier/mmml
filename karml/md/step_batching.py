"""Compile N JAX-MD integrator steps into one dispatch.

``jaxmd_runner`` already batches steps with ``lax.fori_loop`` inside a jitted
block (``_bind_sim``); λ-dynamics and :class:`~mmml.md.drivers.JaxmdDriver` both
ran a Python ``for`` loop over single jitted steps instead, paying a dispatch
per step. This module packages that pattern so a caller only supplies the
per-step function and the block size.

The block is where neighbor-list freshness is spent: dynamic pair arrays are
passed in as *arguments* (not closed over) and held fixed for the whole block,
so ``block_steps`` must stay within the Verlet skin budget — see
:func:`mmml.md.nl_cadence.verlet_reuse_displacement_limit_A`.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax import lax

__all__ = ["make_block_stepper"]


def make_block_stepper(
    step_fn: Callable[..., Any],
    block_steps: int,
    *,
    normalize: Callable[[Any], Any] | None = None,
) -> Callable[..., Any]:
    """Return ``block(state, **dynamic_kwargs)`` advancing ``block_steps`` steps.

    ``step_fn(state, **dynamic_kwargs)`` is the single-step integrator update.
    ``dynamic_kwargs`` (neighbor lists, pressure, box) are traced as arguments
    and stay constant across the block, so mutating the caller's neighbor cache
    mid-block has no effect — refresh between blocks instead.

    ``normalize`` runs on the carry after every step; pass
    ``mmml.cli.run.jaxmd_runner.normalize_jaxmd_state`` (or an equivalent) when
    the integrator carry must keep a fixed dtype across the loop, since
    ``fori_loop`` requires the carry structure and dtypes to be invariant.
    """
    n = int(block_steps)
    if n < 1:
        raise ValueError(f"block_steps must be >= 1; got {block_steps}")

    cast = normalize if normalize is not None else (lambda s: s)

    @jax.jit
    def block(state, **dynamic_kwargs):
        def body(_i, s):
            return cast(step_fn(s, **dynamic_kwargs))

        return lax.fori_loop(0, n, body, cast(state))

    return block
