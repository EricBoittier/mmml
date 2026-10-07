"""Enable JAX float64 for CHARMM MM parity (mm_system_energy, diagnose scripts)."""

from __future__ import annotations

import os


def ensure_jax_x64(*, context: str = "") -> None:
    """Turn on ``jax_enable_x64`` for CHARMM/JAX energy cross-checks.

  Set ``JAX_ENABLE_X64=1`` before importing JAX when possible (diagnose CLIs do
  this at startup).  If JAX was already imported, updates ``jax.config`` so later
  ``jnp.float64`` arrays are not silently downcast.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    if not bool(jax.config.read("jax_enable_x64")):
        jax.config.update("jax_enable_x64", True)
