"""Local mode / force diagnostics for monomers and small clusters.

Canonical entry points:

- :func:`run_mode_check` — calculator-agnostic FD, bond stretch, vib, kick
- :func:`force_fd_check` — analytic vs finite-difference forces
- :func:`build_psf_and_attach_hybrid` — vacuum hybrid ML/MM with live CHARMM PSF

This package covers **monomers and dimers (and small n-mers)**. Rigid
interaction-energy COM scans remain under :mod:`mmml.dimer_scan`.
"""

from __future__ import annotations

from .config import (
    DEFAULT_BOND_DELTAS,
    RESULT_SCHEMA_VERSION,
    HybridModeCheckSetup,
    ModeCheckConfig,
    ModeCheckPaths,
)
from .forces import (
    bond_stretch_scan,
    force_fd_check,
    reduced_mass_amu,
    spring_constant_to_wavenumber_cm,
)
from .cutoff_ladder import CutoffStation, cutoff_region_stations
from .cutoff_sweep import run_cutoff_sweep
from .hybrid import build_psf_and_attach_hybrid
from .result import ModeCheckResult
from .run import run_mode_check

__all__ = [
    "DEFAULT_BOND_DELTAS",
    "RESULT_SCHEMA_VERSION",
    "CutoffStation",
    "HybridModeCheckSetup",
    "ModeCheckConfig",
    "ModeCheckPaths",
    "ModeCheckResult",
    "bond_stretch_scan",
    "build_psf_and_attach_hybrid",
    "cutoff_region_stations",
    "force_fd_check",
    "reduced_mass_amu",
    "run_cutoff_sweep",
    "run_mode_check",
    "spring_constant_to_wavenumber_cm",
]
