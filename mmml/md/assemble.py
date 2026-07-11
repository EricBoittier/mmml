"""Assembly glue: lower a :class:`RunConfig` into a runnable pipeline.

This is the seam both front-ends (``md-system --backend jaxmd`` and the
``cg_jaxmd`` example) call instead of their bespoke inline setup: it resolves the
builder, composes the selected energy terms into a :class:`HybridEnergy`, and
hands the result to a :class:`~mmml.md.drivers.JaxmdDriver`. Heavy imports (jax,
CHARMM, jax-md) stay lazy so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from mmml.md.config import RunConfig
from mmml.md.system import MolecularSystem, SystemSpec

__all__ = [
    "get_builder",
    "available_builders",
    "build_system",
    "build_hybrid_energy",
    "assemble_and_run",
]


def _builder_registry() -> dict[str, type]:
    from mmml.md.builders import (
        PackmolSystemBuilder,
        PeptideWaterSystemBuilder,
        PsfSystemBuilder,
        PyxtalSystemBuilder,
    )

    return {
        cls.name: cls
        for cls in (
            PsfSystemBuilder,
            PackmolSystemBuilder,
            PyxtalSystemBuilder,
            PeptideWaterSystemBuilder,
        )
    }


def available_builders() -> tuple[str, ...]:
    """Names of the registered system builders."""
    return tuple(sorted(_builder_registry()))


def get_builder(name: str):
    """Instantiate a registered :class:`SystemBuilder` by name (no-arg construct)."""
    registry = _builder_registry()
    try:
        return registry[name]()
    except KeyError:
        raise KeyError(
            f"Unknown builder {name!r}. Registered: {sorted(registry)}"
        ) from None


def build_system(spec: SystemSpec) -> MolecularSystem:
    """Resolve ``spec.builder`` and build the :class:`MolecularSystem`."""
    return get_builder(spec.builder).build(spec)


def build_hybrid_energy(
    system: MolecularSystem,
    term_names: Sequence[str],
    ctx: Any = None,
    term_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
):
    """Compose the named registered energy terms into a :class:`HybridEnergy`.

    ``term_kwargs`` maps a term name to constructor kwargs (e.g. SMD anchors,
    dihedral restraints, an ML model override).
    """
    import mmml.md.energy.terms  # noqa: F401  (importing registers the built-ins)
    from mmml.md.energy import EnergyContext, HybridEnergy, get_term

    ctx = ctx if ctx is not None else EnergyContext()
    term_kwargs = dict(term_kwargs or {})
    terms = [get_term(name)(**dict(term_kwargs.get(name, {}))) for name in term_names]
    return HybridEnergy(terms, system, ctx)


def assemble_and_run(
    config: RunConfig,
    *,
    system: MolecularSystem | None = None,
    ctx: Any = None,
    term_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    driver: Any = None,
    neighbor_fn: Callable[..., Any] | None = None,
    on_overlap: Callable[..., Any] | None = None,
):
    """Build → compose → run: the one call both front-ends share.

    ``system`` may be supplied pre-built (bypassing the builder); ``driver``
    defaults to a :class:`JaxmdDriver` writing to ``config.output_dir``.
    """
    if config.backend not in ("auto", "jaxmd"):
        raise NotImplementedError(
            f"assemble_and_run currently targets the jaxmd backend; got {config.backend!r}"
        )
    if system is None:
        system = build_system(config.system)

    energy = build_hybrid_energy(system, config.terms, ctx, term_kwargs)

    if driver is None:
        from pathlib import Path

        from mmml.md.drivers import JaxmdDriver

        output_path = None
        if config.output_dir is not None:
            output_path = Path(config.output_dir) / "trajectory.npz"
        driver = JaxmdDriver(neighbor_fn=neighbor_fn, output_path=output_path)

    return driver.run(system, energy, config.ensemble, on_overlap=on_overlap)
