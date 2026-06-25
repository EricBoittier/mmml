"""Geometry helpers for notebooks, ASE calculators, and evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_charmm_session_ready = False


def ensure_charmm_session_ready(
    *,
    prnlev: int = 5,
    warnlev: int = 5,
    bomlev: int = -2,
    force: bool = False,
) -> None:
    """Initialize CHARMM the same way ``mmml md-system`` does before PSF/minimize work.

    Jupyter kernels often leave ``bomlev`` at 0 (CHARMM default). Any benign warning
    during IC build, BLOCK, or minimization then triggers abnormal termination or
    bond-force segfaults. Call once per kernel before ``build_ase_cluster`` or hybrid
    calculator setup.
    """
    global _charmm_session_ready
    import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci  # noqa: F401
    from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        apply_charmm_verbosity,
        prepare_charmm_vacuum,
    )
    from mmml.interfaces.pycharmmInterface.utils import set_up_directories

    if _charmm_session_ready and not force:
        return

    set_up_directories()
    apply_charmm_verbosity(prnlev=int(prnlev), warnlev=int(warnlev), bomlev=int(bomlev))
    prepare_charmm_vacuum()
    reset_block()
    _charmm_session_ready = True


def prepare_charmm_notebook(**kwargs: Any) -> None:
    """Alias for :func:`ensure_charmm_session_ready` (notebook entry point)."""
    ensure_charmm_session_ready(**kwargs)


def prepare_jax_gpu_notebook(*, required: bool = True) -> bool:
    """Prep JAX GPU JIT toolchain (``ptxas``, cuDNN/cuSPARSE libs) for notebook kernels."""
    from mmml.utils.jax_gpu_warmup import prepare_jax_gpu_notebook as _prepare

    return _prepare(required=required)


def prepare_notebook_kernel(*, jax_required: bool = True) -> None:
    """One-shot notebook bootstrap: JAX GPU env first, then CHARMM session."""
    prepare_jax_gpu_notebook(required=jax_required)
    ensure_charmm_session_ready()


def reference_frame_geometry(
    path: str | Path,
    *,
    frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(atomic_numbers, positions)`` for one frame from an NPZ file.

    Accepts md-system handoff NPZs and QM reference trajectories (``R`` / ``Z`` / ``N``).
    Positions are in Angstrom; atomic numbers follow the NPZ frame order (use
    ``*_psf_order.npz`` when matching CHARMM PSF layout).
    """
    from mmml.cli.run.md_handoff import load_handoff_from_npz

    handoff = load_handoff_from_npz(Path(path).expanduser().resolve(), frame=frame)
    return (
        np.asarray(handoff.atomic_numbers, dtype=np.int32),
        np.asarray(handoff.positions, dtype=np.float64),
    )


def atoms_from_reference_npz(
    path: str | Path,
    *,
    frame: int = 0,
) -> Any:
    """Build an ASE ``Atoms`` object from a reference or handoff NPZ frame."""
    from ase import Atoms

    z, r = reference_frame_geometry(path, frame=frame)
    return Atoms(numbers=z, positions=r)


def prepare_vacuum_nbonds_for_mm() -> None:
    """Apply vacuum ``nbonds`` after cluster PSF build, before the first MM/hybrid energy.

    Call once per notebook kernel after ``build_ase_cluster`` when attaching a hybrid
    MMML calculator. Do **not** call after ``pycharmm.MLpot`` is registered (unsafe
    ``update_bnbnd`` / ``upinb`` on large systems).
    """
    ensure_charmm_session_ready()
    from mmml.interfaces.pycharmmInterface.mlpot.cli_common import setup_charmm_nbonds

    setup_charmm_nbonds()
