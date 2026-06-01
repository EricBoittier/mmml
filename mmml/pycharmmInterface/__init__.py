"""Deprecated alias for :mod:`mmml.interfaces.pycharmmInterface`.

Prefer ``from mmml.interfaces.pycharmmInterface import ...`` in new code.
"""

from __future__ import annotations

import importlib
import sys

_REAL_PREFIX = "mmml.interfaces.pycharmmInterface"

_SUBMODULES = (
    "calculator_utils",
    "cell_list",
    "cutoffs",
    "import_pycharmm",
    "jax_md_neighbor_list",
    "ml_batching",
    "mm_energy_forces",
    "mmml_calculator",
    "monomer_graph_jax",
    "packmol_placement",
    "pbc_prep_factory",
    "pbc_utils_jax",
    "pycharmmCommands",
    "setupBox",
    "setupRes",
    "utils",
)


def _alias_submodules() -> None:
    for name in _SUBMODULES:
        full = f"{__name__}.{name}"
        if full not in sys.modules:
            sys.modules[full] = importlib.import_module(f"{_REAL_PREFIX}.{name}")


_alias_submodules()

from mmml.interfaces import pycharmmInterface as _pkg

__doc__ = _pkg.__doc__
__all__ = list(_SUBMODULES)
