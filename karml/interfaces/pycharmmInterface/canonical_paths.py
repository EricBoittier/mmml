"""Canonical MM/ML calculator entry points for production MD / MLpot / PBC."""

from __future__ import annotations

from typing import Final

CANONICAL: Final[dict[str, str]] = {
    "hybrid_calculator_factory": (
        "karml.interfaces.pycharmmInterface.karml_calculator.setup_calculator"
    ),
    "mlpot_hybrid": (
        "karml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.build_decomposed_mlpot"
    ),
    "mm_forces": (
        "karml.interfaces.pycharmmInterface.mm_energy_forces.build_mm_energy_forces_fn"
    ),
    "jax_com_helpers": (
        "karml.interfaces.pycharmmInterface.calculator_utils.monomer_coms_segment"
    ),
    "sparse_dimer_policy": (
        "karml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy"
    ),
}
