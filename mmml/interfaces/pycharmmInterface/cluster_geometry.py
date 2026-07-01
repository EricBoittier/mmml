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


def _monomer_geometry_is_3d(coords: np.ndarray, *, min_axis_span: float = 0.3) -> bool:
    span = np.max(coords, axis=0) - np.min(coords, axis=0)
    return float(span[1]) >= min_axis_span and float(span[2]) >= min_axis_span


def ensure_monomer_3d_coords(
    coords: np.ndarray,
    *,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Break collinear/planar monomer IC coordinates with a deterministic 3D spread."""
    out = np.asarray(coords, dtype=np.float64).copy()
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {out.shape}")
    n = int(out.shape[0])
    if n < 2:
        return out
    com = out.mean(axis=0)
    out -= com
    span = np.ptp(out, axis=0)
    if float(span[1]) < 0.3:
        out[min(1, n - 1), 1] += float(amplitude)
    if float(span[2]) < 0.3:
        out[min(2, n - 1), 2] += float(amplitude)
    out += com
    return out


def build_same_residue_reference_cluster(
    positions: np.ndarray,
    monomer_offsets: np.ndarray,
    residue_labels: list[str],
) -> np.ndarray:
    """Copy internal geometry from the first monomer of each residue name.

    Mirrors CHARMM IC when no external template residue is available: copy the
    first instance of a residue type and adjust each copy to its current COM.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    offsets = np.asarray(monomer_offsets, dtype=int).reshape(-1)
    if len(residue_labels) != int(len(offsets) - 1):
        raise ValueError(
            f"residue_labels length ({len(residue_labels)}) != "
            f"n_monomers ({len(offsets) - 1})"
        )
    internal_templates: dict[str, np.ndarray] = {}
    for i, res in enumerate(residue_labels):
        key = str(res).upper()
        s, e = int(offsets[i]), int(offsets[i + 1])
        block = pos[s:e]
        if block.size == 0 or not np.all(np.isfinite(block)):
            continue
        if key not in internal_templates:
            internal_templates[key] = block - block.mean(axis=0, keepdims=True)
    out = pos.copy()
    for i, res in enumerate(residue_labels):
        key = str(res).upper()
        tmpl = internal_templates.get(key)
        if tmpl is None:
            continue
        s, e = int(offsets[i]), int(offsets[i + 1])
        if tmpl.shape[0] != e - s:
            continue
        com = out[s:e].mean(axis=0, keepdims=True)
        out[s:e] = tmpl + com
    return out


def apply_same_residue_template_to_positions(
    positions: np.ndarray,
    *,
    n_molecules: int,
    atoms_per_monomer: int,
    residue_label: str,
) -> np.ndarray:
    """After ``ic.build()`` on a homogeneous sequence, duplicate monomer 0 coords."""
    pos = np.asarray(positions, dtype=np.float64).copy()
    n_mol = max(1, int(n_molecules))
    apm = max(1, int(atoms_per_monomer))
    if n_mol <= 1 or pos.shape[0] < n_mol * apm:
        return pos
    labels = [str(residue_label).upper()] * n_mol
    offsets = np.arange(n_mol + 1, dtype=int) * apm
    return build_same_residue_reference_cluster(pos, offsets, labels)


def resolve_cluster_residue_labels(mlpot_ctx: Any, n_monomers: int) -> list[str]:
    args = getattr(mlpot_ctx, "workflow_args", None)
    if args is not None:
        labels = getattr(args, "_cluster_residue_labels", None)
        if labels is not None and len(labels) == n_monomers:
            return [str(x).upper() for x in labels]
        residue = getattr(args, "residue", None)
        if residue is not None:
            return [str(residue).upper()] * n_monomers
    return ["UNK"] * n_monomers


def same_residue_cluster_reference_from_ctx(
    mlpot_ctx: Any,
    *,
    n_atoms: int | None = None,
) -> np.ndarray | None:
    """Build cluster reference by copying each residue type from its first monomer."""
    from mmml.interfaces.pycharmmInterface.mlpot.mc_density import (
        monomer_offsets_from_atoms_per,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.setup import get_charmm_positions_array

    pos = get_charmm_positions_array()
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float64)
    if n_atoms is not None and int(arr.shape[0]) != int(n_atoms):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    atoms_per = getattr(mlpot_ctx, "atoms_per_monomer", None)
    if atoms_per is None:
        pyCModel = getattr(mlpot_ctx, "pyCModel", None)
        atoms_per = getattr(pyCModel, "_atoms_per_monomer", None) if pyCModel else None
    if not atoms_per:
        return None
    per = [int(x) for x in atoms_per]
    n_monomers = len(per)
    if n_monomers <= 0 or int(sum(per)) != int(arr.shape[0]):
        return None
    offsets = monomer_offsets_from_atoms_per(per)
    labels = resolve_cluster_residue_labels(mlpot_ctx, n_monomers)
    return build_same_residue_reference_cluster(arr, offsets, labels)


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
