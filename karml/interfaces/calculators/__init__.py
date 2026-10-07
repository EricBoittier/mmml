"""Standalone ML calculator helpers.

Heavy backends (JAX hybrid, metatomic) are imported lazily so ``import
karml.interfaces.calculators.metatomic`` does not pull ``jax_md``.
"""

from __future__ import annotations

__all__ = [
    "JAXIntermolecularCalculator",
    "MolecularPhysNetCalculator",
    "MonomerSumCalculator",
    "AseFragmentHybridCalculator",
    "have_metatomic",
    "is_metatomic_checkpoint",
    "load_metatomic_calculator",
]

_HYBRID_ATTRS = frozenset(
    {
        "JAXIntermolecularCalculator",
        "MolecularPhysNetCalculator",
        "MonomerSumCalculator",
    }
)
_METATOMIC_ATTRS = frozenset(
    {
        "have_metatomic",
        "load_metatomic_calculator",
        "is_metatomic_checkpoint",
    }
)


def __getattr__(name: str):
    if name in _HYBRID_ATTRS:
        from karml.interfaces.calculators import hybrid as _hybrid

        return getattr(_hybrid, name)
    if name == "AseFragmentHybridCalculator":
        from karml.interfaces.calculators.ase_fragment_hybrid import (
            AseFragmentHybridCalculator,
        )

        return AseFragmentHybridCalculator
    if name in _METATOMIC_ATTRS:
        from karml.interfaces.calculators import metatomic as _metatomic

        return getattr(_metatomic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
