"""Canonical MM/ML calculator entry points for production MD / MLpot / PBC."""

from __future__ import annotations

from typing import Final

CANONICAL: Final[dict[str, str]] = {
    "hybrid_calculator_factory": (
        "mmml.interfaces.pycharmmInterface.mmml_calculator.setup_calculator"
    ),
    "mlpot_hybrid": (
        "mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot.build_decomposed_mlpot"
    ),
    "mm_forces": (
        "mmml.interfaces.pycharmmInterface.mm_energy_forces.build_mm_energy_forces_fn"
    ),
    "jax_com_helpers": (
        "mmml.interfaces.pycharmmInterface.calculator_utils.monomer_coms_segment"
    ),
    "sparse_dimer_policy": (
        "mmml.interfaces.pycharmmInterface.mlpot.mlpot_sparse_dimer_policy"
    ),
}
