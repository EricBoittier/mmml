"""Utility functions for MMML.

Exports are resolved lazily so importing a lightweight submodule (e.g.
``mmml.utils.geometry_checks``) does not pull JAX via ``model_checkpoint``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "save_model_checkpoint",
    "load_model_checkpoint",
    "create_model_from_checkpoint",
    "quick_save",
    "quick_load",
    "extract_model_config",
    "to_jsonable",
    "HDF5Reporter",
    "DatasetSpec",
    "make_jaxmd_reporter",
    "load_hdf5_trajectory",
    "summarize_hdf5",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "save_model_checkpoint": ("mmml.utils.model_checkpoint", "save_model_checkpoint"),
    "load_model_checkpoint": ("mmml.utils.model_checkpoint", "load_model_checkpoint"),
    "create_model_from_checkpoint": (
        "mmml.utils.model_checkpoint",
        "create_model_from_checkpoint",
    ),
    "quick_save": ("mmml.utils.model_checkpoint", "quick_save"),
    "quick_load": ("mmml.utils.model_checkpoint", "quick_load"),
    "extract_model_config": ("mmml.utils.model_checkpoint", "extract_model_config"),
    "to_jsonable": ("mmml.utils.model_checkpoint", "to_jsonable"),
    "HDF5Reporter": ("mmml.utils.hdf5_reporter", "HDF5Reporter"),
    "DatasetSpec": ("mmml.utils.hdf5_reporter", "DatasetSpec"),
    "make_jaxmd_reporter": ("mmml.utils.hdf5_reporter", "make_jaxmd_reporter"),
    "load_hdf5_trajectory": ("mmml.utils.hdf5_reporter", "load_hdf5_trajectory"),
    "summarize_hdf5": ("mmml.utils.hdf5_reporter", "summarize_hdf5"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(__all__))
