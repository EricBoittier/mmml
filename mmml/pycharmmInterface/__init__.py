"""Deprecated alias for :mod:`mmml.interfaces.pycharmmInterface`.

Prefer ``from mmml.interfaces.pycharmmInterface import ...`` in new code.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
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


class _CompatAliasLoader(importlib.abc.Loader):
    """Load a legacy ``mmml.pycharmmInterface.<sub>`` name as the canonical interface module."""

    def __init__(self, alias_name: str, target_name: str) -> None:
        self._alias_name = alias_name
        self._target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return sys.modules.get(self._target_name)

    def exec_module(self, module: ModuleType) -> None:
        target = importlib.import_module(self._target_name)
        sys.modules[self._alias_name] = target
        sys.modules[self._target_name] = target


class _CompatAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        prefix = f"{__name__}."
        if not fullname.startswith(prefix):
            return None
        sub = fullname[len(prefix) :]
        if sub not in _SUBMODULES:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _CompatAliasLoader(fullname, f"{_REAL_PREFIX}.{sub}"),
        )


def _install_compat_finder() -> None:
    if any(isinstance(finder, _CompatAliasFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _CompatAliasFinder())


def __getattr__(name: str):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_install_compat_finder()

from mmml.interfaces import pycharmmInterface as _pkg

__doc__ = _pkg.__doc__
__all__ = list(_SUBMODULES)
