"""Cutoff grid search for ``mmml md-system --optimize-cutoffs``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mmml.interfaces.pycharmmInterface.cutoffs import (
    cutoff_grids_from_args,
    handoff_widths_from_args,
)
from mmml.interfaces.pycharmmInterface.hybrid_reference import (
    ReferenceTrajectory,
    load_reference_trajectory_npz,
    run_cutoff_grid_search,
)


def _atoms_per_from_composition(n_atoms: int, composition: str) -> tuple[list[int], list[str], int]:
    from mmml.cli.run.md_handoff import cluster_layout_from_composition_string

    atoms_per_list, residue_labels, summary = cluster_layout_from_composition_string(
        composition,
        n_atoms=n_atoms,
    )
    n_monomers = sum(int(v) for v in summary.values())
    return atoms_per_list, residue_labels, n_monomers


def _build_ase_factory(args: Any, base_ckpt_dir: Path, atoms_per_list: list[int], n_monomers: int):
    from mmml.interfaces.pycharmmInterface.mmml_calculator import setup_calculator

    ml_w, mm_on, mm_w = handoff_widths_from_args(args)
    max_atoms = max(atoms_per_list) * 2
    return setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_list,
        N_MONOMERS=n_monomers,
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        complementary_handoff=not bool(getattr(args, "no_complementary_handoff", False)),
        doML=True,
        doMM=True,
        doML_dimer=not bool(getattr(args, "skip_ml_dimers", False)),
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max_atoms,
        cell=getattr(args, "box_size", None) or getattr(args, "cell", False),
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        ml_compute_dtype=getattr(args, "ml_compute_dtype", None),
        max_pairs=int(getattr(args, "max_pairs", 20_000) or 20_000),
    )


def run_optimize_cutoffs(args: Any) -> int:
    """Grid-search ML/MM handoff cutoffs against a reference trajectory NPZ."""
    from ase import Atoms

    from mmml.cli.base import resolve_checkpoint_paths
    from mmml.cli.run.md_handoff import (
        cluster_geometry_from_handoff,
        ensure_psf_for_handoff_cluster,
    )
    from mmml.cli.run.md_pbc_suite.ase import _parse_composition

    ref_path = Path(args.reference_npz).expanduser().resolve()
    if not getattr(args, "composition", None):
        raise ValueError("--optimize-cutoffs requires --composition")

    composition = str(args.composition)
    parts = _parse_composition(composition)
    n_monomers_cli = int(getattr(args, "n_molecules", 0) or 0)
    if n_monomers_cli <= 0:
        n_monomers_cli = sum(cnt for _, cnt in parts)

    # Peek at frame count / atom count before full load.
    with np.load(ref_path, allow_pickle=False) as peek:
        R_shape = peek["R"].shape
    n_atoms = int(R_shape[1])
    n_atoms_monomer = n_atoms // n_monomers_cli
    if n_atoms_monomer * n_monomers_cli != n_atoms:
        raise ValueError(
            f"reference NPZ has {n_atoms} atoms, not divisible by n_monomers={n_monomers_cli}"
        )

    atoms_per_list, residue_labels, n_monomers = _atoms_per_from_composition(
        n_atoms, composition
    )

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        base_ckpt_dir, _ = resolve_checkpoint_paths(
            Path(args.checkpoint).expanduser().resolve()
        )

    # Build Z from composition topology (same as handoff continuation).
    from mmml.cli.run.md_handoff import MdHandoffState

    handoff_stub = MdHandoffState(
        positions=np.zeros((n_atoms, 3)),
        atomic_numbers=np.zeros(n_atoms, dtype=np.int32),
    )
    z, _, _, _, _ = cluster_geometry_from_handoff(
        handoff_stub,
        composition=composition,
        n_molecules=n_monomers,
    )

    reference: ReferenceTrajectory = load_reference_trajectory_npz(
        ref_path,
        z_fallback=z,
        n_atoms_monomer=n_atoms_monomer,
        n_monomers=n_monomers,
        max_frames=int(getattr(args, "max_frames", None) or 200),
    )

    ensure_psf_for_handoff_cluster(
        composition=parts,
        atomic_numbers=z,
        atoms_per_list=atoms_per_list,
        residue_labels=residue_labels,
        positions=reference.R[int(reference.frame_indices[0])],
        quiet=bool(getattr(args, "quiet", False)),
    )

    atoms = Atoms(numbers=z, positions=reference.R[int(reference.frame_indices[0])])
    if getattr(args, "box_size", None):
        L = float(args.box_size)
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)

    factory = _build_ase_factory(args, base_ckpt_dir, atoms_per_list, n_monomers)

    def attach_calculator(cutoff):
        result = factory(
            atomic_numbers=z,
            atomic_positions=atoms.get_positions(),
            n_monomers=n_monomers,
            cutoff_params=cutoff,
            doML=True,
            doMM=True,
            doML_dimer=not bool(getattr(args, "skip_ml_dimers", False)),
            backprop=True,
            energy_conversion_factor=1.0,
            force_conversion_factor=1.0,
        )
        if len(result) == 3:
            calc, _, _ = result
        else:
            calc, _ = result
        return calc

    ml_grid, mm_on_grid, mm_w_grid = cutoff_grids_from_args(args)
    if not getattr(args, "quiet", False):
        print(
            f"mmml md-system optimize-cutoffs: {ref_path.name} "
            f"({len(reference.frame_indices)} frames, grids "
            f"ml={len(ml_grid)} mm_on={len(mm_on_grid)} mm_w={len(mm_w_grid)})",
            flush=True,
        )

    t0 = time.time()
    results, best = run_cutoff_grid_search(
        ml_grid=ml_grid,
        mm_on_grid=mm_on_grid,
        mm_w_grid=mm_w_grid,
        atoms=atoms,
        attach_calculator=attach_calculator,
        reference=reference,
        energy_weight=float(getattr(args, "energy_weight", 1.0)),
        force_weight=float(getattr(args, "force_weight", 1.0)),
        complementary_handoff=not bool(getattr(args, "no_complementary_handoff", False)),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    elapsed = time.time() - t0

    out_dir = Path(getattr(args, "output_dir", Path("artifacts/md_optimize_cutoffs"))).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = getattr(args, "optimize_output", None) or out_dir / "optimize_cutoffs.json"

    payload = {
        "reference_npz": str(ref_path),
        "composition": composition,
        "checkpoint": str(base_ckpt_dir),
        "n_eval_frames": int(len(reference.frame_indices)),
        "energy_weight": float(getattr(args, "energy_weight", 1.0)),
        "force_weight": float(getattr(args, "force_weight", 1.0)),
        "initial_cutoffs": {
            "ml_switch_width": handoff_widths_from_args(args)[0],
            "mm_switch_on": handoff_widths_from_args(args)[1],
            "mm_switch_width": handoff_widths_from_args(args)[2],
        },
        "best": best,
        "results": results,
        "elapsed_s": elapsed,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not getattr(args, "quiet", False) and best is not None:
        print(f"Best cutoffs: {best}", flush=True)
        print(f"Wrote {out_path} ({elapsed:.1f}s)", flush=True)
    return 0
