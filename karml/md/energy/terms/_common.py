"""Shared helpers for energy-term implementations."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ase_contribution_from_jax", "resolve_displacement_fn", "resolve_ml_model"]


def resolve_displacement_fn(system: Any, ctx: Any) -> Callable[..., Any]:
    """Return a jax-md displacement function for ``system`` under PBC or free space.

    Prefers an explicit ``ctx.displacement_fn``; otherwise builds one from
    ``system.box`` (orthorhombic ``space.periodic`` on the box diagonal, or
    ``space.free`` when ``box is None``).
    """
    if getattr(ctx, "displacement_fn", None) is not None:
        return ctx.displacement_fn

    import numpy as np
    from jax_md import space

    if system.box is None:
        displacement_fn, _ = space.free()
        return displacement_fn
    box_diag = np.diag(np.asarray(system.box, dtype=float))
    displacement_fn, _ = space.periodic(box_diag)
    return displacement_fn


def resolve_ml_model(term: Any, ctx: Any) -> tuple[Any, Any]:
    """Resolve ``(model, params)`` from a term override or the energy context.

    ML terms are model-agnostic: the trained model/params come from the run
    context (:class:`~mmml.md.energy.registry.EnergyContext`) unless the term was
    constructed with an explicit override.
    """
    model = term.model if getattr(term, "model", None) is not None else getattr(ctx, "model", None)
    params = term.params if getattr(term, "params", None) is not None else getattr(ctx, "params", None)
    if model is None or params is None:
        raise ValueError(
            f"{term.name!r} needs an ML model+params; pass them to the term or set "
            "EnergyContext(model=..., params=...)."
        )
    return model, params


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
