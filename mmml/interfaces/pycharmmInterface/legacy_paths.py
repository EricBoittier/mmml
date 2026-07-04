"""Canonical vs deprecated MM/ML calculator entry points.

Production MD / MLpot / PBC workflows should use the paths in ``CANONICAL``.
Legacy modules remain for old notebooks and training scripts but must not receive
new JAX hot-path logic (especially Python loops that build ``jnp.stack`` /
``jnp.concatenate`` over monomers or dimers inside ``@jit``).
"""

from __future__ import annotations

import warnings
from typing import Final

# ---------------------------------------------------------------------------
# Canonical (maintained)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Deprecated (do not extend)
# ---------------------------------------------------------------------------
DEPRECATED: Final[dict[str, str]] = {
    "mmml.models.physnetjax.physnetjax.calc.mmml_calculator": (
        "mmml.interfaces.pycharmmInterface.mmml_calculator.setup_calculator"
    ),
    "mmml.interfaces.aseInterface.mmml_ase.get_spherical_cutoff_calculator": (
        "mmml.interfaces.pycharmmInterface.mmml_calculator.setup_calculator"
    ),
    "mmml.pycharmmInterface": (
        "mmml.interfaces.pycharmmInterface (package alias only)"
    ),
    "mmml.interfaces.pycharmmInterface.monomer_graph_jax.monomer_COMs": (
        "calculator_utils.monomer_coms_segment + pbc_utils_jax.wrap_groups"
    ),
    "tests.models.mm_ml_model": (
        "removed / commented legacy; use tests.unit.test_jax_mm_spoof patterns"
    ),
    "mmml.utils.hybrid_optimization (ML force path)": (
        "prefer setup_calculator + jax.grad on spherical_fn for new optimizers"
    ),
}


def warn_legacy(name: str, replacement: str, *, stacklevel: int = 3) -> None:
    """Emit a single-style DeprecationWarning for legacy calculator paths."""
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
