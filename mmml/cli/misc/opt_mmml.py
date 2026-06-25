"""
Finds cut-offs and MM non-bonded parameters to better fit the QM data.

Organised so that ``run(args)`` returns an OptContext holding atoms, calculators,
dataset, and results for inspection in a notebook or REPL.

Public API for notebook use:

    from mmml.cli.misc.opt_mmml import run, parse_args, OptContext

    args = parse_args()  # or build argparse.Namespace with required options
    ctx = run(args)      # returns OptContext, not exit code

    # Inspect calculators and data
    ctx.atoms            # ASE atoms with hybrid calculator attached
    ctx.hybrid_calc      # current hybrid calculator
    ctx.calculator_factory
    ctx.cutoff_params    # CutoffParameters (ml_switch_width / mm_switch_on / mm_switch_width)
    ctx.reference        # ReferenceTrajectory from hybrid_reference
    ctx.dataset          # raw npz (legacy alias: ctx.reference path arrays)
    ctx.R_all            # positions (n_frames, n_atoms, 3)
    ctx.E_all, ctx.F_all # reference energies and forces
    ctx.Z_ds             # atomic numbers (dataset order, possibly mapped)
    ctx.com_distances    # COM distance per frame
    ctx.frame_indices    # indices of frames used in evaluation
    ctx.has_E, ctx.has_F
    ctx.results          # list of grid-search result dicts (canonical + ml_cutoff aliases)
    ctx.best             # best result dict
    ctx.save_dir         # directory used for diagnostics and trajectories

Notebook kernel: Use the project venv (e.g. mmml/.venv/bin/python) as the notebook kernel.
Using a different Python (e.g. ~/.local) can cause JaxRuntimeError / PJRT version mismatch.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mmml.cli.base import (
    resolve_checkpoint_paths,
    setup_ase_imports,
    setup_mmml_imports,
)
from mmml.interfaces.pycharmmInterface.cutoffs import (
    add_handoff_cutoff_args,
    add_handoff_cutoff_grid_args,
    cutoff_grids_from_args,
    cutoff_parameters_from_args,
    handoff_widths_from_args,
)
from mmml.interfaces.pycharmmInterface.hybrid_reference import (
    ReferenceTrajectory,
    load_reference_trajectory_npz,
    run_cutoff_grid_search,
)
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_gpu_policy import (
    resolve_ml_gpu_count as _resolve_ml_gpu_count,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PDB file processing and MD simulation demo"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset file to load for pycharmm",
    )
    parser.add_argument(
        "--pdbfile",
        type=Path,
        required=True,
        help="Path to the PDB file to load for pycharmm [requires correct atom names and types].",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    add_handoff_cutoff_grid_args(parser)
    add_handoff_cutoff_args(parser)

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to save best parameters and scores as JSON.",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=None,
        help="Optional path to save detailed results as NPZ file.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the force field (default: False).",
    )

    # ========================================================================
    # MD SIMULATION ARGUMENTS
    # ========================================================================
    parser.add_argument(
        "--energy-catch",
        type=float,
        default=0.5,
        help="Energy catch factor for the simulation (default: 0.05).",
    )

    parser.add_argument(
        "--cell",
        default=None,
        help="Use cell for the simulation (default: False) as a float for a cubic cell length (Å).",
    )

    parser.add_argument(
        "--n-monomers",
        type=int,
        required=True,
        help="Number of monomers in the system (default: 2).",
    )
    parser.add_argument(
        "--n-atoms-monomer",
        type=int,
        required=True,
        help=(
            "Number of atoms per monomer. Defaults to total_atoms/n_monomers "
            "derived from the dataset."
        ),
    )
    parser.add_argument(
        "--include-mm",
        action="store_true",
        help="Keep MM contributions enabled when evaluating the hybrid calculator.",
    )
    parser.add_argument(
        "--no-complementary-handoff",
        action="store_true",
        help="Use legacy MM switching. When set, mm_r_min defaults to mm_switch_on.",
    )
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="If set, skip the ML dimer correction in the hybrid calculator.",
    )
    parser.add_argument(
        "--mm-r-min",
        type=float,
        default=None,
        metavar="Å",
        help="MM inner cutoff: exclude pairs with dimer COM < this.",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Max systems per ML forward pass. When set, chunk large batches to reduce memory.",
    )
    parser.add_argument(
        "--ml-gpu-count",
        type=int,
        default=None,
        metavar="N",
        help="Parallel PhysNet chunks on N local GPUs (default 1; or MMML_MLPOT_N_GPUS).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output inside the calculator factory.",
    )


    return parser.parse_args()


@dataclass
class OptContext:
    """
    Holds all calculators, data, and results from opt_mmml for inspection.
    Returned by run(); use ctx.atoms, ctx.hybrid_calc, ctx.dataset, etc. in a notebook.
    """

    args: argparse.Namespace = field(default_factory=lambda: argparse.Namespace())
    # ASE / structures
    atoms: Any = None
    pdb_ase_atoms: Any = None
    # Model
    params: Any = None
    model: Any = None
    # Calculators (inspectable)
    calculator_factory: Any = None
    hybrid_calc: Any = None
    cutoff_params: Any = None
    reference: ReferenceTrajectory | None = None
    # Dataset: raw npz and extracted arrays (mirrors reference for notebooks)
    dataset: Any = None
    R_all: np.ndarray | None = None
    E_all: np.ndarray | None = None
    F_all: np.ndarray | None = None
    Z_ds: np.ndarray | None = None
    com_distances: np.ndarray | None = None
    frame_indices: np.ndarray | None = None
    has_E: bool = False
    has_F: bool = False
    n_eval: int = 0
    n_frames: int = 0
    # Grid search
    results: list[dict[str, Any]] = field(default_factory=list)
    best: dict[str, Any] | None = None
    save_dir: Path | None = None


def _ensure_opt_imports() -> dict[str, Any]:
    """Load optional imports required by this script. Returns a dict of names for use in callers."""
    try:
        import ase.io as ase_io
        import matplotlib.pyplot as plt
        import pandas as pd
        from ase.io import Trajectory
        from ase.calculators.singlepoint import SinglePointCalculator
        from mmml.pycharmmInterface.setupBox import setup_box_generic
        from mmml.pycharmmInterface.import_pycharmm import coor
        return {
            "ase_io": ase_io,
            "plt": plt,
            "pd": pd,
            "Trajectory": Trajectory,
            "SinglePointCalculator": SinglePointCalculator,
            "setup_box_generic": setup_box_generic,
            "coor": coor,
        }
    except ModuleNotFoundError as exc:
        sys.exit(f"Required modules not available: {exc}")


def load_pdb_and_box(
    pdbfile: Path,
    setup_box_generic_fn: Any,
    ase_io: Any,
) -> Any:
    """Setup PyCHARMM box and load PDB into ASE atoms."""
    setup_box_generic_fn(str(pdbfile), side_length=1000)
    return ase_io.read(str(pdbfile))


def align_dataset_to_pdb_order(
    R_all: np.ndarray,
    Z_ds: np.ndarray,
    F_all: np.ndarray | None,
    Z_pdb: np.ndarray,
    mapping: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reorder R_all, Z_ds, F_all to match PDB atom order using mapping. Returns (R_all, Z_ds, F_all)."""
    if np.all(Z_pdb == Z_ds):
        return R_all, Z_ds, F_all
    if mapping is None:
        mapping = np.array(
            [3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 13, 11, 12, 10, 14, 15, 16, 17, 18, 19]
        )
    Z_ds = Z_ds[mapping]
    R_all = np.array([R_all[i][mapping] for i in range(len(R_all))])
    if F_all is not None:
        F_all = np.array([F_all[i][mapping] for i in range(len(F_all))])
    return R_all, Z_ds, F_all


def build_calculator_factory(
    args: argparse.Namespace,
    base_ckpt_dir: Path,
    natoms: int,
    setup_calculator: Any,
) -> Any:
    """Build the hybrid calculator factory from args and checkpoint."""
    ml_w, mm_on, mm_w = handoff_widths_from_args(args)
    return setup_calculator(
        ATOMS_PER_MONOMER=args.n_atoms_monomer,
        N_MONOMERS=args.n_monomers,
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        complementary_handoff=not getattr(args, "no_complementary_handoff", False),
        doML=True,
        doMM=args.include_mm,
        doML_dimer=not args.skip_ml_dimers,
        debug=args.debug,
        verbose=True,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=natoms,
        ml_energy_conversion_factor=1,
        ml_force_conversion_factor=1,
        cell=args.cell,
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_gpu_count=_resolve_ml_gpu_count(getattr(args, "ml_gpu_count", None)),
        mm_r_min=getattr(args, "mm_r_min", None),
    )


def build_hybrid_calculator(
    calculator_factory: Any,
    Z: np.ndarray,
    R: np.ndarray,
    args: argparse.Namespace,
    cutoff_params: Any | None = None,
) -> Any:
    """Build hybrid calculator for given Z, R and cutoff params."""
    calc, _ = calculator_factory(
        atomic_numbers=Z,
        atomic_positions=R,
        n_monomers=args.n_monomers,
        cutoff_params=cutoff_params,
        doML=True,
        doMM=args.include_mm,
        doML_dimer=True,  # not args.skip_ml_dimers,
        backprop=True,
        debug=args.debug,
        energy_conversion_factor=1,
        force_conversion_factor=1,
        do_pbc_map=getattr(calculator_factory, "do_pbc_map", args.cell is not None),
        pbc_map=getattr(calculator_factory, "pbc_map", None),
    )
    return calc


def setup_atoms_cell_and_calc(
    atoms: Any,
    cell_param: str | float | None,
    hybrid_calc: Any,
) -> None:
    """Set cell/PBC and attach hybrid calculator to atoms."""
    if cell_param is not None:
        from ase.cell import Cell
        cell = Cell.fromcellpar(
            [float(cell_param), float(cell_param), float(cell_param), 90.0, 90.0, 90.0]
        )
        atoms.set_cell(cell)
        atoms.set_pbc(True)
    atoms.calc = hybrid_calc


def _cutoff_label_triple(args: argparse.Namespace) -> tuple[float, float, float]:
    return handoff_widths_from_args(args)


def run_diagnostics(ctx: OptContext, save_dir: Path) -> None:
    """Plot energy scatter, energy vs COM distance, and write diagnostics CSV."""
    imp = _ensure_opt_imports()
    pd = imp["pd"]
    plt = imp["plt"]
    args = ctx.args
    ml_w, mm_on, mm_w = _cutoff_label_triple(args)
    atoms = ctx.atoms
    frame_indices = ctx.frame_indices
    R_all = ctx.R_all
    E_all = ctx.E_all
    F_all = ctx.F_all
    com_distances = ctx.com_distances
    has_E = ctx.has_E
    has_F = ctx.has_F
    if not has_E or frame_indices is None or R_all is None:
        return
    pred_E_list = []
    ref_E_list = []
    for i in frame_indices:
        atoms.set_positions(R_all[i])
        pred_E_list.append(float(atoms.get_potential_energy()))
        ref_E_list.append(float(E_all[i]))
    pred_E_arr = np.array(pred_E_list)
    ref_E_arr = np.array(ref_E_list)
    r_sel = com_distances[frame_indices]
    rows = []
    for i, pred_e, ref_e in zip(frame_indices, pred_E_arr, ref_E_arr):
        if has_F and F_all is not None:
            atoms.set_positions(R_all[i])
            pred_F_i = np.asarray(atoms.get_forces())
            ref_F_i = np.asarray(F_all[i])
            dF = pred_F_i - ref_F_i
            mse_f_i = float(np.mean(dF**2))
            max_f_err_i = float(np.abs(dF).max())
        else:
            mse_f_i = float("nan")
            max_f_err_i = float("nan")
        rows.append({
            "frame_index": int(i),
            "com_dist": float(com_distances[i]),
            "pred_e": float(pred_e),
            "ref_e": float(ref_e),
            "mse_f": mse_f_i,
            "max_f_err": max_f_err_i,
        })
    df_diag = pd.DataFrame(rows).sort_values(by=["com_dist", "frame_index"]).reset_index(drop=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / f"diagnostics_{ml_w:.2f}_{mm_on:.2f}_{mm_w:.2f}.csv"
    df_diag.to_csv(out_csv, index=False)
    print(f"Saved diagnostics CSV to {out_csv}")
    plt.figure(figsize=(5, 5))
    plt.scatter(ref_E_arr, pred_E_arr, s=12, alpha=0.8)
    lim_min = float(min(ref_E_arr.min(), pred_E_arr.min()))
    lim_max = float(max(ref_E_arr.max(), pred_E_arr.max()))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1)
    plt.xlabel("Reference energy (E)")
    plt.ylabel("Predicted energy (E)")
    plt.title(
        f"Energy: predicted vs reference | ml_w={ml_w:.2f}, mm_on={mm_on:.2f}, mm_w={mm_w:.2f}"
    )
    plt.tight_layout()
    out_scatter = save_dir / f"energy_scatter_{ml_w:.2f}_{mm_on:.2f}_{mm_w:.2f}.png"
    plt.savefig(out_scatter, dpi=150)
    plt.close()
    order = np.argsort(r_sel)
    r_sorted = r_sel[order]
    pred_sorted = pred_E_arr[order]
    ref_sorted = ref_E_arr[order]
    plt.figure(figsize=(6, 4))
    plt.plot(r_sorted, ref_sorted, label="Reference", lw=1.5)
    plt.plot(r_sorted, pred_sorted, label="Predicted", lw=1.5)
    plt.xlabel("COM distance (Å)")
    plt.ylabel("Energy (E)")
    plt.title(
        f"Energy vs COM distance | ml_w={ml_w:.2f}, mm_on={mm_on:.2f}, mm_w={mm_w:.2f}"
    )
    plt.legend()
    plt.tight_layout()
    out_curve = save_dir / f"energy_vs_r_{ml_w:.2f}_{mm_on:.2f}_{mm_w:.2f}.png"
    plt.savefig(out_curve, dpi=150)
    plt.close()


def write_reference_trajectory(ctx: OptContext, save_dir: Path) -> None:
    """Write ASE trajectory with reference energies/forces from the dataset."""
    imp = _ensure_opt_imports()
    Trajectory = imp["Trajectory"]
    SinglePointCalculator = imp["SinglePointCalculator"]
    Atoms = setup_ase_imports()
    args = ctx.args
    atoms = ctx.atoms
    frame_indices = ctx.frame_indices
    R_all = ctx.R_all
    E_all = ctx.E_all
    F_all = ctx.F_all
    Z_ds = ctx.Z_ds
    has_E = ctx.has_E
    has_F = ctx.has_F
    if not has_E or frame_indices is None or R_all is None or Z_ds is None:
        return
    ml_w, mm_on, mm_w = _cutoff_label_triple(args)
    ref_traj_path = save_dir / f"reference_trajectory_{ml_w:.2f}_{mm_on:.2f}_{mm_w:.2f}.traj"
    with Trajectory(ref_traj_path, "w") as traj:
        for i in frame_indices:
            cell = atoms.cell if getattr(atoms, "cell", None) is not None else None
            ref_atoms = Atoms(
                numbers=Z_ds,
                positions=R_all[i],
                cell=cell,
                pbc=getattr(atoms, "pbc", False),
            )
            ref_E = float(E_all[i])
            ref_F = F_all[i] if has_F and F_all is not None else None
            ref_atoms.calc = SinglePointCalculator(ref_atoms, energy=ref_E, forces=ref_F)
            traj.write(ref_atoms)
    print(
        f"Saved reference trajectory ({len(frame_indices)} frames) to {ref_traj_path}"
    )


def run(args: argparse.Namespace) -> OptContext:
    """
    Run cutoff optimization with the given args. Callable from CLI or notebook.
    Returns an OptContext with atoms, calculators, dataset, and results for inspection.
    """
    ctx = OptContext(args=args)
    base_ckpt_dir, epoch_dir = resolve_checkpoint_paths(args.checkpoint)
    imp = _ensure_opt_imports()
    ase_io = imp["ase_io"]
    setup_box_generic_fn = imp["setup_box_generic"]

    CutoffParameters, _, setup_calculator, _ = setup_mmml_imports()

    # Load PDB and box
    pdb_ase_atoms = load_pdb_and_box(args.pdbfile, setup_box_generic_fn, ase_io)
    ctx.pdb_ase_atoms = pdb_ase_atoms
    ctx.atoms = pdb_ase_atoms
    print(f"Loaded PDB file: {pdb_ase_atoms}")
    print(
        "Note: for testing the dimer calculator, the pdb file should contain a dimer, "
        "and the atom types should be consistent with the dimer calculator."
    )

    natoms = len(pdb_ase_atoms)
    assert args.n_atoms_monomer * args.n_monomers == natoms, "n_atoms_monomer * n_monomers != natoms"
    # Calculator loads its own model from checkpoint with natoms=max(monomer,dimer) size

    Z, R = pdb_ase_atoms.get_atomic_numbers(), pdb_ase_atoms.get_positions()

    ctx.calculator_factory = build_calculator_factory(args, base_ckpt_dir, natoms, setup_calculator)
    ctx.cutoff_params = cutoff_parameters_from_args(args)
    print(f"Cutoff parameters: {ctx.cutoff_params}")
    ctx.cutoff_params.plot_cutoff_parameters(Path.cwd())

    ctx.hybrid_calc = build_hybrid_calculator(
        ctx.calculator_factory, Z, R, args, cutoff_params=None
    )
    do_ml_dimer = not getattr(args, "skip_ml_dimers", False)
    print(f"Hybrid calculator created: {ctx.hybrid_calc} (doML_dimer={do_ml_dimer})")

    ctx.save_dir = (
        args.out.parent if args.out is not None
        else (args.out_npz.parent if args.out_npz is not None else Path.cwd())
    )
    try:
        ctx.cutoff_params.plot_cutoff_parameters(ctx.save_dir)
    except Exception as e:
        print(f"Warning: could not render cutoff schematic: {e}")

    setup_atoms_cell_and_calc(ctx.atoms, args.cell, ctx.hybrid_calc)
    hybrid_energy = float(ctx.atoms.get_potential_energy())
    hybrid_forces = np.asarray(ctx.atoms.get_forces())
    print(f"Initial energy: {hybrid_energy:.6f} eV")
    print(f"Initial forces: {hybrid_forces}")


    # Load reference trajectory and select frames
    ctx.reference = load_reference_trajectory_npz(
        args.dataset,
        z_fallback=Z,
        n_atoms_monomer=args.n_atoms_monomer,
        n_monomers=args.n_monomers,
        max_frames=args.max_frames,
    )
    ctx.dataset = np.load(args.dataset)
    ctx.R_all = ctx.reference.R
    ctx.E_all = ctx.reference.E
    ctx.F_all = ctx.reference.F
    ctx.Z_ds = ctx.reference.Z
    ctx.com_distances = ctx.reference.com_distances
    ctx.frame_indices = ctx.reference.frame_indices
    ctx.has_E = ctx.reference.has_E
    ctx.has_F = ctx.reference.has_F
    ctx.n_eval = len(ctx.frame_indices)
    ctx.n_frames = ctx.reference.n_frames
    print(f"Dataset: {args.dataset}")
    print(f"Dataset keys: {list(ctx.dataset.keys())}")
    print(f"R shape: {ctx.R_all.shape}")
    print(
        f"Evaluating {len(ctx.frame_indices)} frames (out of {ctx.n_frames}). "
        f"E available: {ctx.has_E}, F available: {ctx.has_F}"
    )

    # Align dataset atom order to PDB if needed
    R_all, Z_ds, F_all = align_dataset_to_pdb_order(
        ctx.R_all, ctx.Z_ds, ctx.F_all, Z,
    )
    ctx.R_all, ctx.Z_ds, ctx.F_all = R_all, Z_ds, F_all
    ctx.reference = ReferenceTrajectory(
        path=ctx.reference.path,
        R=ctx.R_all,
        Z=ctx.Z_ds,
        E=ctx.E_all,
        F=ctx.F_all,
        frame_indices=ctx.frame_indices,
        com_distances=ctx.com_distances,
        has_E=ctx.has_E,
        has_F=ctx.has_F,
        n_frames=ctx.n_frames,
        metadata=ctx.reference.metadata,
    )
    if not np.all(Z == ctx.Z_ds):
        print("Applied atom-order mapping to match PDB.")



    # Diagnostics: energy scatter, energy vs COM, diagnostics CSV
    try:
        run_diagnostics(ctx, ctx.save_dir)
    except Exception as e:
        print(f"Warning: diagnostics plotting failed: {e}")

    # Reference trajectory (ground-truth energies/forces)
    try:
        write_reference_trajectory(ctx, ctx.save_dir)
        print(
            f"Saved reference trajectory ({len(ctx.frame_indices)} frames). "
            f"Visualize with: ase gui <ref_traj_path>"
        )
    except Exception as e:
        print(f"Warning: reference trajectory writing failed: {e}")

    # Grid search
    ml_grid, mm_on_grid, mm_w_grid = cutoff_grids_from_args(args)
    print(
        f"Grid sizes -> ml_w:{len(ml_grid)} mm_on:{len(mm_on_grid)} mm_w:{len(mm_w_grid)}"
    )

    def attach_calculator(cutoff):
        return build_hybrid_calculator(
            ctx.calculator_factory,
            ctx.Z_ds,
            ctx.R_all[int(ctx.frame_indices[0])],
            args,
            cutoff_params=cutoff,
        )

    start = time.time()
    ctx.results, ctx.best = run_cutoff_grid_search(
        ml_grid=ml_grid,
        mm_on_grid=mm_on_grid,
        mm_w_grid=mm_w_grid,
        atoms=ctx.atoms,
        attach_calculator=attach_calculator,
        reference=ctx.reference,
        energy_weight=float(args.energy_weight),
        force_weight=float(args.force_weight),
        complementary_handoff=not bool(getattr(args, "no_complementary_handoff", False)),
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"Grid search completed in {elapsed:.1f}s over {len(ctx.results)} combos.")
    print(f"Best: {ctx.best}")

    if args.out is not None:
        payload = {
            "best": ctx.best,
            "results": ctx.results,
            "n_eval_frames": int(ctx.n_eval),
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.out}")

    if args.out_npz is not None and ctx.best is not None:
        npz_data = {
            "ml_switch_widths": np.array([r["ml_switch_width"] for r in ctx.results]),
            "mm_switch_ons": np.array([r["mm_switch_on"] for r in ctx.results]),
            "mm_switch_widths": np.array([r["mm_switch_width"] for r in ctx.results]),
            "ml_cutoffs": np.array([r["ml_cutoff"] for r in ctx.results]),
            "mm_cutoffs": np.array([r["mm_cutoff"] for r in ctx.results]),
            "mse_energies": np.array([r["mse_energy"] for r in ctx.results]),
            "mse_forces": np.array([r["mse_forces"] for r in ctx.results]),
            "objectives": np.array([r["objective"] for r in ctx.results]),
            "best_ml_switch_width": ctx.best["ml_switch_width"],
            "best_mm_switch_on": ctx.best["mm_switch_on"],
            "best_mm_switch_width": ctx.best["mm_switch_width"],
            "best_ml_cutoff": ctx.best["ml_cutoff"],
            "best_mm_cutoff": ctx.best["mm_cutoff"],
            "best_mse_energy": ctx.best["mse_energy"],
            "best_mse_forces": ctx.best["mse_forces"],
            "best_objective": ctx.best["objective"],
            "n_eval_frames": ctx.n_eval,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
        }
        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_npz, **npz_data)
        print(f"Saved detailed results to {args.out_npz}")

    return ctx


def main() -> int:
    """Entry point for CLI. Returns 0; use run(args) in a notebook to get OptContext."""
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python -m mmml.cli.opt_mmml \
#   --dataset /path/to/data.npz \
#   --checkpoint /path/to/checkpoint \
#   --n-monomers 2 \
#   --pdbfile /path/to/pdb \
#   --include-mm \
#   --n-atoms-monomer 10 \
#   --ml-cutoff-grid 1.5,2.0,2.5 \
#   --mm-switch-on-grid 4.0,5.0,6.0 \
#   --mm-cutoff-grid 0.5,1.0,1.5 \
#   --energy-weight 1.0 \
#   --force-weight 1.0 \
#   --max-frames 200 \
#   --out /tmp/cutoff_opt.json