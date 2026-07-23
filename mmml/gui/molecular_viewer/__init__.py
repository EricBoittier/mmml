"""OpenXR molecular trajectory and structure viewer."""

from .debug import debug, log, set_debug
from .molecule import (
    Atom,
    CPK_COLORS,
    VDW_RADII,
    center_and_scale,
    compute_bonds,
    load_pdb,
    load_structure,
    load_xyz,
    load_xyz_trajectory,
)

__all__ = [
    "run_viewer",
    "debug",
    "log",
    "set_debug",
    "Atom",
    "load_pdb",
    "load_xyz",
    "load_xyz_trajectory",
    "load_structure",
    "center_and_scale",
    "compute_bonds",
    "CPK_COLORS",
    "VDW_RADII",
]


def __getattr__(name: str):
    """Lazily import ``run_viewer`` so this package stays importable (e.g. for
    ``molecule.py``'s pure parsers) without the GL/GLFW/OpenXR stack installed."""
    if name == "run_viewer":
        from .viewer import run_viewer

        return run_viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
