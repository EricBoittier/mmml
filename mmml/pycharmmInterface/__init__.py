"""Deprecated alias for :mod:`mmml.interfaces.pycharmmInterface`.

Prefer ``from mmml.interfaces.pycharmmInterface import ...`` in new code.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

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


class _SubmoduleRedirect(ModuleType):
    """Lazy proxy so legacy ``mmml.pycharmmInterface.<name>`` imports work without eager loads."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        super().__init__(alias_name)
        self._target_name = target_name

    def _load_target(self) -> ModuleType:
        target = importlib.import_module(self._target_name)
        sys.modules[self.__name__] = target
        return target

    def __getattr__(self, name: str):
        return getattr(self._load_target(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load_target())


def _register_submodule_aliases() -> None:
    for name in _SUBMODULES:
        full = f"{__name__}.{name}"
        if full not in sys.modules:
            sys.modules[full] = _SubmoduleRedirect(full, f"{_REAL_PREFIX}.{name}")


_register_submodule_aliases()

from mmml.interfaces import pycharmmInterface as _pkg

__doc__ = _pkg.__doc__
__all__ = list(_SUBMODULES)
