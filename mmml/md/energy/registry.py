"""Composable energy terms and the hybrid ML/MM energy provider.

The energy layer has two faces because there are two irreconcilable engine
contracts (see ``docs/md-cg-unification-design.md``, §2): a pure jittable
``energy_fn(R, neighbor, box, **kw)`` for jax-md, and a stateful ASE
``Calculator``. Each :class:`EnergyTerm` may supply either or both via
:class:`TermFns`; :class:`HybridEnergy` composes the selected terms and exposes
both faces.

Decision B (§10): each term declares its own neighbor/pair capacity via
:class:`NeighborRequest` and owns its padding — peptide-water slots and
intermolecular pairs keep independent capacities rather than sharing one
driver-owned list.

This module defines the protocol, the registry, and the composition skeleton.
The concrete terms (``ml_intra``, ``ml_pep_water``, ``mm_nonbonded``, ``smd``,
``dihedral``, ``vdw_core``) migrate into ``mmml/md/energy/terms/`` from
``examples/cg_jaxmd.py`` and ``mmml.interfaces.jaxmdInterface`` in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Protocol, runtime_checkable

from mmml.md.system import MolecularSystem

__all__ = [
    "NeighborRequest",
    "TermFns",
    "EnergyContext",
    "EnergyTerm",
    "HybridEnergy",
    "register_term",
    "get_term",
    "available_terms",
]


@dataclass(frozen=True)
class NeighborRequest:
    """A term's neighbor-list requirement (decision B).

    ``kind`` distinguishes independent pair families (e.g. ``"intermolecular"``
    vs. ``"peptide_water"``) that need separately-padded lists; ``capacity_hint``
    is the padded slot count the term sizes its own buffers to.
    """

    cutoff_A: float
    kind: str
    capacity_hint: int | None = None


class TermFns(NamedTuple):
    """The engine-specific callables a term provides.

    At least one of ``jax_energy_fn`` / ``ase_contribution`` must be non-None.
    ``jax_energy_fn(R, neighbor=None, box=None, **kw) -> scalar`` is pure and
    jittable; ``ase_contribution(atoms) -> (energy, forces)`` is numpy/stateful.
    """

    jax_energy_fn: Callable[..., Any] | None = None
    ase_contribution: Callable[..., Any] | None = None
    neighbor_request: NeighborRequest | None = None


@dataclass(frozen=True)
class EnergyContext:
    """Runtime context shared across terms when a :class:`HybridEnergy` is built.

    Carries the ML model/params, displacement function, unit conventions, and
    any per-run knobs (biasing schedules, cutoffs) that terms need at build time
    but that are not part of the immutable topology.
    """

    model: Any = None
    params: Any = None
    displacement_fn: Callable[..., Any] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class EnergyTerm(Protocol):
    """A single, composable physics term (ML, MM, bias, or restraint)."""

    name: str

    def neighbor_request(self, system: MolecularSystem) -> NeighborRequest | None:
        """Neighbor capacity this term needs, or None if it needs no pair list."""
        ...

    def make(self, system: MolecularSystem, ctx: EnergyContext) -> TermFns:
        """Build the engine callables for ``system`` under ``ctx``."""
        ...


_TERM_REGISTRY: dict[str, type[EnergyTerm]] = {}


def register_term(name: str) -> Callable[[type[EnergyTerm]], type[EnergyTerm]]:
    """Class decorator registering an :class:`EnergyTerm` under ``name``."""

    def _register(cls: type[EnergyTerm]) -> type[EnergyTerm]:
        if name in _TERM_REGISTRY:
            raise ValueError(f"Energy term {name!r} is already registered.")
        _TERM_REGISTRY[name] = cls
        return cls

    return _register


def get_term(name: str) -> type[EnergyTerm]:
    """Look up a registered term class by name."""
    try:
        return _TERM_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown energy term {name!r}. Registered: {sorted(_TERM_REGISTRY)}"
        ) from None


def available_terms() -> tuple[str, ...]:
    """Names of all registered energy terms."""
    return tuple(sorted(_TERM_REGISTRY))


class HybridEnergy:
    """Composes selected :class:`EnergyTerm` instances into one energy provider.

    Gathers each term's :class:`NeighborRequest`, then exposes both engine faces:
    :meth:`as_jax_energy_fn` sums the terms' jittable energy functions and
    :meth:`as_ase_calculator` sums their ASE contributions.
    """

    def __init__(
        self,
        terms: list[EnergyTerm],
        system: MolecularSystem,
        ctx: EnergyContext | None = None,
    ) -> None:
        self.system = system
        self.ctx = ctx if ctx is not None else EnergyContext()
        self.terms = list(terms)
        self.term_fns: list[TermFns] = [t.make(system, self.ctx) for t in self.terms]
        self.neighbor_requests: list[NeighborRequest] = [
            fns.neighbor_request for fns in self.term_fns if fns.neighbor_request is not None
        ]

    def as_jax_energy_fn(self) -> Callable[..., Any]:
        """Return a single jittable ``energy_fn(R, **kw)`` summing all terms."""
        fns = [f.jax_energy_fn for f in self.term_fns if f.jax_energy_fn is not None]
        if not fns:
            raise ValueError("No terms provide a jax energy function.")

        def energy_fn(R, **kwargs):
            total = fns[0](R, **kwargs)
            for fn in fns[1:]:
                total = total + fn(R, **kwargs)
            return total

        return energy_fn

    def as_ase_calculator(self):
        """Return an ASE ``Calculator`` summing the terms' contributions."""
        contribs = [f.ase_contribution for f in self.term_fns if f.ase_contribution is not None]
        if not contribs:
            raise ValueError("No terms provide an ASE contribution.")

        # Lazy import: keep ``mmml.md`` importable without ASE present.
        import numpy as np
        from ase.calculators.calculator import Calculator, all_changes

        class _HybridCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy", "forces"),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                energy = 0.0
                forces = np.zeros((len(atoms), 3), dtype=np.float64)
                for contribution in contribs:
                    e, f = contribution(atoms)
                    energy += float(e)
                    forces += np.asarray(f, dtype=np.float64)
                self.results = {"energy": float(energy), "forces": forces}

        return _HybridCalculator()
