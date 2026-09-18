"""Standalone ML calculator helpers."""

from mmml.interfaces.calculators.hybrid import (
    JAXIntermolecularCalculator,
    MolecularPhysNetCalculator,
    MonomerSumCalculator,
)

__all__ = [
    "JAXIntermolecularCalculator",
    "MolecularPhysNetCalculator",
    "MonomerSumCalculator",
    "AseFragmentHybridCalculator",
    "have_metatomic",
    "is_metatomic_checkpoint",
    "load_metatomic_calculator",
]


def __getattr__(name: str):
    if name == "AseFragmentHybridCalculator":
        from mmml.interfaces.calculators.ase_fragment_hybrid import (
            AseFragmentHybridCalculator,
        )

        return AseFragmentHybridCalculator
    if name in {"have_metatomic", "load_metatomic_calculator", "is_metatomic_checkpoint"}:
        from mmml.interfaces.calculators import metatomic as _metatomic

        return getattr(_metatomic, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
