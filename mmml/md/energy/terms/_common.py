"""Shared helpers for energy-term implementations."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ase_contribution_from_jax"]


def ase_contribution_from_jax(
    energy_fn: Callable[..., Any],
) -> Callable[[Any], tuple[float, Any]]:
    """Derive an ASE ``(energy, forces)`` contribution from a jax energy function.

    Forces are ``-dE/dR`` via :func:`jax.grad`, so a pure-jax term gets its ASE
    face for free (positions in Å, energy in eV → forces in eV/Å, ASE units).
    The returned callable evaluates the term at fixed defaults (no per-step
    kwargs such as a moving SMD target).
    """
    import jax
    import numpy as np

    grad_fn = jax.grad(lambda R: energy_fn(R))

    def contribution(atoms):
        import jax.numpy as jnp

        R = jnp.asarray(atoms.get_positions())
        energy = float(energy_fn(R))
        forces = -np.asarray(grad_fn(R), dtype=np.float64)
        return energy, forces

    return contribution
