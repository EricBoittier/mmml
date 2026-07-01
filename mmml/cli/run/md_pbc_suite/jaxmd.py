#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from ase.optimize.fire import FIRE
from jax import random

from mmml.cli.base import resolve_checkpoint_paths
from mmml.cli.run.jaxmd_runner import set_up_nhc_sim_routine
from mmml.utils.geometry_checks import assert_no_intermonomer_atom_overlap
from mmml.interfaces.pycharmmInterface.cutoffs import (
    DEFAULT_ML_SWITCH_WIDTH,
    DEFAULT_MM_SWITCH_ON,
    DEFAULT_MM_SWITCH_WIDTH,
    handoff_widths_from_args,
)
from mmml.interfaces.pycharmmInterface.mm_energy_forces import (
    DEFAULT_JAX_MD_SKIN_DISTANCE_A,
    format_mm_pair_update_stats_summary,
)
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
from mmml.paths import default_meoh_template_pdb

from .ase import (
    _cubic_box_length,
    _check_or_charmm_overlap_rescue,
    _enforce_min_com_separation,
    _numpy_wrap_monomers_primary_cell,
    _parse_composition,
    _randomize_monomer_com_positions,
    _run_charmm_minimize,
    _validate_psf_charges,
    build_initial_cluster_from_args,
    resolve_cluster_geometry,
    resolve_cluster_packmol_sphere,
)


class _BestMinimizationFrame:
    def __init__(self, atoms: Atoms) -> None:
        self.atoms = atoms
        self.best_force_positions: np.ndarray | None = None
        self.best_force_energy = float("inf")
        self.best_force_fmax = float("inf")
        self.best_force_label = ""
        self.best_energy_positions: np.ndarray | None = None
        self.best_energy = float("inf")
        self.best_energy_fmax = float("inf")
        self.best_energy_label = ""

    def record(self, label: str) -> None:
        try:
            energy = float(self.atoms.get_potential_energy())
            fmax = float(np.abs(self.atoms.get_forces()).max())
        except Exception:
            return
        positions = np.asarray(self.atoms.get_positions(), dtype=float).copy()
        if np.isfinite(fmax) and fmax < self.best_force_fmax:
            self.best_force_positions = positions
            self.best_force_energy = energy
            self.best_force_fmax = fmax
            self.best_force_label = label
        if np.isfinite(energy) and energy < self.best_energy:
            self.best_energy_positions = positions
            self.best_energy = energy
            self.best_energy_fmax = fmax
            self.best_energy_label = label

    def restore_best_force(self) -> float:
        if self.best_force_positions is not None:
            self.atoms.set_positions(self.best_force_positions)
        return float(np.abs(self.atoms.get_forces()).max())

    def write(self, out_dir: Path, prefix: str) -> dict[str, float | str]:
        summary: dict[str, float | str] = {}
        if self.best_force_positions is not None:
            force_atoms = self.atoms.copy()
            force_atoms.set_positions(self.best_force_positions)
            path = out_dir / f"{prefix}_best_force.xyz"
            ase_write(str(path), force_atoms)
            summary["best_force_xyz"] = str(path.relative_to(out_dir))
            summary["best_force_energy_eV"] = float(self.best_force_energy)
            summary["best_force_fmax_eVA"] = float(self.best_force_fmax)
            summary["best_force_label"] = self.best_force_label
        if self.best_energy_positions is not None:
            energy_atoms = self.atoms.copy()
            energy_atoms.set_positions(self.best_energy_positions)
            path = out_dir / f"{prefix}_best_energy.xyz"
            ase_write(str(path), energy_atoms)
            summary["best_energy_xyz"] = str(path.relative_to(out_dir))
            summary["best_energy_eV"] = float(self.best_energy)
            summary["best_energy_fmax_eVA"] = float(self.best_energy_fmax)
            summary["best_energy_label"] = self.best_energy_label
        return summary


def main(argv: list[str] | None = None) -> int:
    from mmml.utils.jax_gpu_warmup import apply_xla_cuda_timer_log_filter

    apply_xla_cuda_timer_log_filter()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite_jaxmd"))
    p.add_argument("--template-pdb", type=Path, default=default_meoh_template_pdb())
    p.add_argument("--n-molecules", type=int, default=10)
    p.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Residue composition as RES:count comma list (e.g. MEOH:5,TIP3:5). Overrides --n-molecules.",
    )
    p.add_argument(
        "--builder",
        choices=("gas", "liquid", "crystal"),
        default=None,
        help=(
            "Starting-coordinate builder: gas=open grid, liquid=cube/sphere grid, "
            "crystal=PyXtal."
        ),
    )
    p.add_argument("--spacing", type=float, default=5.0, help="Target minimum random COM spacing in Angstrom.")
    p.add_argument("--min-com-start-distance", type=float, default=6.0)
    p.add_argument(
        "--free-space",
        action="store_true",
        help=(
            "Open boundary cluster: no ASE unit cell / PBC, hybrid calculator with cell=False, "
            "JAX-MD NVE/NVT in free space. Incompatible with --ensemble npt."
        ),
    )
    p.add_argument("--ps", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--traj-every", type=int, default=1)
    p.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split output trajectory into chunks with at most this many frames each (0 = single file).",
    )
    p.add_argument("--ensemble", type=str, default="npt", choices=["nve", "nvt", "npt"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=1.0, help="atm (for NPT)")
    p.add_argument("--nhc-chain-length", type=int, default=3)
    p.add_argument("--nhc-chain-steps", type=int, default=2)
    p.add_argument("--nhc-sy-steps", type=int, default=3)
    p.add_argument("--nhc-tau", type=float, default=100.0)
    p.add_argument("--nhc-barostat-tau", type=float, default=10000.0)
    p.add_argument(
        "--steps-per-recording",
        type=int,
        default=100,
        help="MD steps between trajectory/HDF5 records (smaller → more frames, heavier I/O).",
    )
    p.add_argument(
        "--traj-export-molecular-wrap",
        action="store_true",
        help=(
            "Apply molecular COM wrap when writing HDF5 and final ASE .traj (slower JAX work per frame; "
            "helps visualization). Default off: coordinates match what was accumulated during dynamics."
        ),
    )
    p.add_argument("--jaxmd-minimize-steps", type=int, default=200)
    p.add_argument("--jaxmd-pbc-minimize-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--packmol",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Explicitly pack --composition with Packmol. "
            "Default uses grid placement plus refinement."
        ),
    )
    p.add_argument(
        "--packmol-placement",
        choices=("cube", "sphere"),
        default=None,
        help="Initial placement constraint: cube (default) or sphere (--packmol-radius).",
    )
    p.add_argument(
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="packmol_sphere",
        help="Legacy alias for --packmol-placement sphere.",
    )
    p.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="Å",
        dest="packmol_radius",
        help="Initial inside-sphere radius in Angstrom (independent of --flat-bottom-radius).",
    )
    p.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Initial placement center in Angstrom (default: 0 0 0).",
    )
    p.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Legacy Packmol distance tolerance in Å for explicit --packmol runs.",
    )
    from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import add_box_sizing_args

    add_box_sizing_args(p)
    p.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        dest="flat_bottom_radius",
        help=(
            "Flat-bottom COM restraint radius (Å), independent of --packmol-radius. "
            "Legacy: if only this is set with --composition, Packmol uses it as sphere radius."
        ),
    )
    p.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        dest="flat_bottom_k",
        help="Flat-bottom k when |COM offset| exceeds radius (default: 1.0).",
    )
    p.add_argument(
        "--flat-bottom-mode",
        choices=["system", "monomer"],
        default="system",
        dest="flat_bottom_mode",
        help="system: cluster COM; monomer: sum over monomer COM restraints (same R, k).",
    )
    p.add_argument(
        "--min-com-restraint-distance",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Pairwise inter-monomer COM lower wall. Adds 0.5*k*(r_min-r)^2 "
            "when COM distance r < r_min (default: disabled)."
        ),
    )
    p.add_argument(
        "--min-com-restraint-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Force constant for --min-com-restraint-distance (default: 1.0).",
    )
    p.add_argument(
        "--ml-switch-width",
        "--ml-cutoff",
        dest="ml_switch_width",
        type=float,
        default=DEFAULT_ML_SWITCH_WIDTH,
    )
    p.add_argument("--mm-switch-on", type=float, default=DEFAULT_MM_SWITCH_ON)
    p.add_argument(
        "--include-mm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include JAX MM LJ/Coulomb pairs; --no-include-mm = ML (PhysNet) only.",
    )
    p.add_argument(
        "--mm-switch-width",
        "--mm-cutoff",
        dest="mm_switch_width",
        type=float,
        default=DEFAULT_MM_SWITCH_WIDTH,
    )
    p.add_argument("--max-pairs", type=int, default=20_000)
    p.add_argument(
        "--ml-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Chunk PhysNet monomer/dimer batches (auto: 256 on GPU / 64 on CPU for n>=40)."
    )
    p.add_argument(
        "--ml-max-active-dimers",
        type=int,
        default=None,
        metavar="N",
        help="Sparse ML dimer slot cap (PBC default max(1000, 6*n_monomers))."
    )
    from mmml.interfaces.pycharmmInterface.ml_dtypes import add_ml_compute_dtype_args
    add_ml_compute_dtype_args(p)
    p.add_argument("--pre-min-fmax", type=float, default=0.1)
    p.add_argument("--pre-min-steps", type=int, default=50)
    p.add_argument("--bfgs-maxstep", type=float, default=0.05)
    p.add_argument("--fire-min-steps", type=int, default=100)
    p.add_argument("--fire-min-maxstep", type=float, default=0.02)
    p.add_argument(
        "--rescue-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If post-BFGS fmax stays above --pre-min-fmax, alternate CHARMM SD/ABNR with "
            "ASE BFGS+FIRE on the CHARMM coordinates until fmax converges or rounds are exhausted."
        ),
    )
    p.add_argument("--rescue-charmm-sd-steps", type=int, default=100, help="Rescue CHARMM SD steps.")
    p.add_argument("--rescue-charmm-abnr-steps", type=int, default=300, help="Rescue CHARMM ABNR steps.")
    p.add_argument("--max-fmax-after-min", type=float, default=2.0)
    p.add_argument("--quiet-bfgs", action="store_true")
    p.add_argument(
        "--skip-jit-warmup",
        action="store_true",
        help=(
            "Skip generic XLA GPU compile and pre-JAX-MD hybrid energy/force warmup "
            "(may log XLA cuda_timer delay-kernel warnings on first GPU compile)."
        ),
    )
    p.add_argument(
        "--calculator-pre-minimize",
        dest="calculator_pre_minimize",
        action="store_true",
        default=True,
        help="Run ASE BFGS with the MMML calculator before JAX-MD minimization (default).",
    )
    p.add_argument(
        "--no-calculator-pre-minimize",
        dest="calculator_pre_minimize",
        action="store_false",
        help="Skip ASE BFGS/FIRE pre-minimization before the JAX-MD runner.",
    )
    p.add_argument("--jax-md-capacity-multiplier", type=float, default=1.75)
    p.add_argument("--jax-md-capacity-growth-factor", type=float, default=1.5)
    p.add_argument("--jax-md-max-overflow-retries", type=int, default=4)
    p.add_argument("--jax-md-update-interval", type=int, default=1)
    p.add_argument(
        "--jax-md-skin-distance",
        type=float,
        default=DEFAULT_JAX_MD_SKIN_DISTANCE_A,
        help=(
            "Reuse cached MM neighbor pairs while max displacement since last update is below "
            f"this (Å). Default {DEFAULT_JAX_MD_SKIN_DISTANCE_A}: safe with jax-md dr_threshold=0.5 Å "
            "(Verlet skin); use 0 only for debugging (rebuild every step, much slower)."
        ),
    )
    p.add_argument(
        "--nvt-allow-stale-neighbors",
        action="store_true",
        help="Deprecated no-op: NVT uses configured neighbor update interval/skin by default.",
    )
    p.add_argument(
        "--npt-allow-stale-neighbors",
        action="store_true",
        help="Deprecated no-op: NPT pre-minimization uses configured neighbor update interval/skin by default.",
    )
    p.add_argument("--jax-md-disable-fallback", action="store_true")
    p.add_argument(
        "--charmm-pre-minimize",
        dest="charmm_pre_minimize",
        action="store_true",
        default=True,
        help="Run CHARMM SD/ABNR before calculator minimization (default).",
    )
    p.add_argument(
        "--no-charmm-pre-minimize",
        dest="charmm_pre_minimize",
        action="store_false",
        help="Skip CHARMM SD/ABNR before calculator minimization.",
    )
    p.add_argument("--charmm-sd-steps", type=int, default=200)
    p.add_argument("--charmm-abnr-steps", type=int, default=1000)
    p.add_argument("--charmm-tolenr", type=float, default=1e-3)
    p.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    p.add_argument(
        "--charmm-nbxmod",
        type=int,
        default=5,
        help="CHARMM NBXMOD for SD/ABNR minimization (default 5, matching CGenFF).",
    )
    p.add_argument(
        "--min-intermonomer-atom-distance",
        type=float,
        default=0.1,
        help="Abort if atoms from different monomers get closer than this distance in Angstrom (<=0 disables).",
    )
    p.add_argument(
        "--dynamics-overlap-action",
        choices=["warn", "rescue", "error", "off"],
        default="warn",
        help=(
            "Inter-monomer overlap during JAX-MD: warn/rescue run CHARMM SD/ABNR "
            "(unless --no-dynamics-overlap-charmm-rescue), rethermalize, continue; "
            "error=abort; off=disable checks."
        ),
    )
    p.add_argument(
        "--no-dynamics-overlap-charmm-rescue",
        action="store_true",
        help=(
            "Disable CHARMM SD/ABNR rescue when an overlap is detected during dynamics "
            "(default: run CHARMM minimization with the same periodic box as the MD step)."
        ),
    )
    p.add_argument(
        "--dynamics-overlap-charmm-sd-steps",
        type=int,
        default=200,
        help="CHARMM SD steps for dynamics overlap rescue (default 200).",
    )
    p.add_argument(
        "--dynamics-overlap-charmm-abnr-steps",
        type=int,
        default=400,
        help="CHARMM ABNR steps for dynamics overlap rescue (default 400).",
    )
    p.add_argument(
        "--handoff-pre-minimize",
        action="store_true",
        help="Run pre-minimization even when continuing from a handoff.",
    )
    p.add_argument(
        "--continue-velocities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use velocities from handoff when present (else re-thermalize).",
    )
    p.add_argument(
        "--handoff-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate initial MMML |F| on handoff and optionally pre-minimize.",
    )
    p.add_argument(
        "--handoff-quality-fmax-eVA",
        type=float,
        default=1.0,
        help="|F| threshold (eV/Å) for --handoff-quality-gate.",
    )
    p.add_argument(
        "--handoff-quality-action",
        choices=("minimize", "warn", "error"),
        default="minimize",
        help="Action when handoff quality gate threshold is exceeded.",
    )
    p.add_argument(
        "--handoff-velocity-remove-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove net P/L from handoff velocities before MD.",
    )
    p.add_argument(
        "--handoff-require-cell",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require periodic cell in handoff for PBC runs.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    args = p.parse_args(argv)
    if args.free_space and args.ensemble == "npt":
        raise ValueError("--free-space cannot be combined with NPT (--ensemble npt)")
    if args.box_size is not None and args.box_size <= 0:
        raise ValueError("--box-size must be positive")

    out_dir = (Path.cwd() / args.output_dir.expanduser()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        base_ckpt_dir, _ = resolve_checkpoint_paths(args.checkpoint.expanduser().resolve())
    print(f"Using MMML checkpoint: {base_ckpt_dir}")

    from mmml.cli.run.md_handoff import (
        apply_handoff_geometry_to_atoms,
        ensure_psf_for_handoff_cluster,
        get_handoff_in,
        handoff_from_atoms,
        handoff_skip_pre_min,
        resolve_jaxmd_minimize_steps_for_handoff,
        print_handoff_policy_panel,
        resolve_handoff_box,
        resolve_handoff_velocity_policy,
        set_handoff_out,
        summarize_handoff_policy,
        write_handoff_policy_json,
    )

    handoff_in = get_handoff_in()
    saved_jaxmd_minimize_steps = int(args.jaxmd_minimize_steps)
    saved_jaxmd_pbc_minimize_steps = int(args.jaxmd_pbc_minimize_steps)
    skip_pre_min = handoff_skip_pre_min(
        handoff_in, handoff_pre_minimize=bool(getattr(args, "handoff_pre_minimize", False))
    )
    quality_gate_triggered = False

    z, r0, atoms_per_list, residue_labels, _composition_summary = resolve_cluster_geometry(
        args,
        handoff_in,
    )
    n_molecules = len(atoms_per_list)
    if args.composition:
        composition = _parse_composition(args.composition)
    else:
        composition = [("MEOH", n_molecules)]
    monomer_offsets = np.zeros(n_molecules + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    if handoff_in is not None:
        ensure_psf_for_handoff_cluster(
            composition=composition,
            atomic_numbers=z,
            atoms_per_list=atoms_per_list,
            residue_labels=residue_labels,
            positions=r0,
            quiet=bool(getattr(args, "quiet", False)),
        )
    psf_charge_summary = _validate_psf_charges(
        monomer_offsets=monomer_offsets,
        residue_labels=residue_labels,
        total_atoms=len(z),
    )
    if not resolve_cluster_packmol_sphere(args) and handoff_in is None:
        r0 = _randomize_monomer_com_positions(
            r0,
            monomer_offsets,
            spacing=args.spacing,
            min_com_distance=max(float(args.spacing), float(args.min_com_start_distance)),
            seed=args.seed,
        )
        r0 = _enforce_min_com_separation(r0, monomer_offsets, args.min_com_start_distance)
    ml_w, mm_on, mm_w = handoff_widths_from_args(args)
    free_space = bool(args.free_space)
    auto_L = None
    if not free_space:
        from mmml.interfaces.pycharmmInterface.mlpot.box_sizing import (
            resolve_suite_auto_box_side,
        )

        auto_L, _auto_src = resolve_suite_auto_box_side(args, r0, ml_cutoff=ml_w)
    L, box_source, box_warnings = resolve_handoff_box(
        handoff_in,
        yaml_box_size=args.box_size,
        free_space=free_space,
        auto_box_from_geometry=auto_L,
        require_cell=bool(getattr(args, "handoff_require_cell", False)),
    )
    for msg in box_warnings:
        print(f"Handoff box: {msg}", flush=True)
    jax_equil_ps = float(getattr(args, "jaxmd_mini_box_equil_ps", 0.0) or 0.0)
    if jax_equil_ps > 0.0 and not free_space:
        print(
            "WARN: --jaxmd-mini-box-equil-ps is reserved for a future NPT prelude; "
            "use --ensemble npt with a short --ps for cell equilibration today.",
            flush=True,
        )
    if free_space:
        if args.box_size is not None:
            print(
                "md_10mer_mmml_pbc_suite_jaxmd: note: ignoring --box-size with --free-space "
                f"({float(args.box_size):g} Å)."
            )
        r = np.asarray(r0, dtype=float) if handoff_in is not None else r0 - r0.mean(axis=0)
    else:
        if handoff_in is not None:
            r = np.asarray(r0, dtype=float)
        else:
            assert L is not None
            r = r0 - r0.mean(axis=0) + 0.5 * L
    geom_tag = "vac" if free_space else "pbc"
    atoms = Atoms(numbers=z, positions=r)
    if free_space:
        atoms.set_pbc(False)
    else:
        assert L is not None
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)
    if handoff_in is not None:
        apply_handoff_geometry_to_atoms(
            atoms, handoff_in, monomer_offsets=monomer_offsets
        )
        if not free_space and L is not None:
            atoms.set_cell([L, L, L])
            atoms.set_pbc(True)
    minimization_summary: dict[str, float | str] = {}
    pre_min_ran = False
    if handoff_in is None or not skip_pre_min:
        _check_or_charmm_overlap_rescue(
            atoms,
            monomer_offsets,
            min_distance=args.min_intermonomer_atom_distance,
            context="initial placement",
            nstep_sd=args.charmm_sd_steps,
            nstep_abnr=args.charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            nbxmod=args.charmm_nbxmod,
            timings=minimization_summary,
        )
    elif not getattr(args, "quiet", False):
        print(
            "Skipping initial overlap check (continuing from equilibrated handoff).",
            flush=True,
        )
    if skip_pre_min and not getattr(args, "quiet", False):
        pbc_note = (
            f"PBC FIRE ({saved_jaxmd_pbc_minimize_steps} steps) still runs before dynamics."
            if not free_space and saved_jaxmd_pbc_minimize_steps > 0
            else "No PBC FIRE (free space or --jaxmd-pbc-minimize-steps 0)."
        )
        print(
            "Handoff: vacuum/COM and ASE/CHARMM pre-min skipped (default). "
            f"{pbc_note} "
            "If initial E_pot is high or |F| > ~1 eV/Å, set "
            "handoff_pre_minimize: true or handoff_quality_gate: true in the campaign YAML.",
            flush=True,
        )
    if args.charmm_pre_minimize and not skip_pre_min:
        print(
            f"CHARMM pre-minimization starting "
            f"(SD={args.charmm_sd_steps}, ABNR={args.charmm_abnr_steps})"
        )
        _run_charmm_minimize(
            atoms,
            nstep_sd=args.charmm_sd_steps,
            nstep_abnr=args.charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            nbxmod=args.charmm_nbxmod,
            timings=minimization_summary,
            cubic_box_side_A=float(L) if not free_space else None,
        )
        _check_or_charmm_overlap_rescue(
            atoms,
            monomer_offsets,
            min_distance=args.min_intermonomer_atom_distance,
            context="after CHARMM pre-minimization",
            nstep_sd=args.charmm_sd_steps,
            nstep_abnr=args.charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            nbxmod=args.charmm_nbxmod,
            timings=minimization_summary,
        )
        print(
            "CHARMM pre-minimization complete "
            f"({minimization_summary.get('charmm_min_wall_s', 0.0):.3f} s)"
        )

    if args.ensemble == "npt":
        effective_update_interval = int(max(1, args.jax_md_update_interval))
        effective_skin = float(max(0.0, args.jax_md_skin_distance))
    elif args.ensemble == "nvt":
        effective_update_interval = int(max(1, args.jax_md_update_interval))
        effective_skin = float(max(0.0, args.jax_md_skin_distance))
    else:
        effective_update_interval = int(max(1, args.jax_md_update_interval))
        effective_skin = float(max(0.0, args.jax_md_skin_distance))

    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per_list,
        N_MONOMERS=n_molecules,
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        doML=True,
        doMM=bool(getattr(args, "include_mm", True)),
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max(atoms_per_list) * 2,
        cell=False if free_space else float(L),
        verbose=False,
        max_pairs=args.max_pairs,
        jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
        jax_md_update_interval=effective_update_interval,
        jax_md_skin_distance=effective_skin,
        flat_bottom_radius=args.flat_bottom_radius,
        flat_bottom_force_const=args.flat_bottom_k,
        flat_bottom_mode=args.flat_bottom_mode,
        min_com_restraint_distance=args.min_com_restraint_distance,
        min_com_restraint_force_const=args.min_com_restraint_k,
        defer_xla_gpu_warmup=bool(args.skip_jit_warmup),
        ml_batch_size=getattr(args, "ml_batch_size", None),
        ml_max_active_dimers=getattr(args, "ml_max_active_dimers", None),
        ml_compute_dtype=getattr(args, "ml_compute_dtype", None),
    )
    cutoff = CutoffParameters(ml_switch_width=ml_w, mm_switch_on=mm_on, mm_switch_width=mm_w)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=atoms.get_positions(),
        n_monomers=n_molecules,
        cutoff_params=cutoff,
        doML=True,
        doMM=bool(getattr(args, "include_mm", True)),
        doML_dimer=True,
        backprop=False,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=False,
    )
    if len(calc_result) == 3:
        calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None
    atoms.calc = calc

    if not args.skip_jit_warmup:
        import jax.numpy as jnp

        from mmml.utils.jax_gpu_warmup import warmup_hybrid_spherical_cutoff

        mm_pair_idx = None
        mm_pair_mask = None
        if get_update_fn is not None and not free_space and L is not None:
            box_nl = np.array([L, L, L], dtype=np.float64)
            pos_np = np.asarray(atoms.get_positions(), dtype=np.float64)
            update_fn = get_update_fn(pos_np, cutoff, box=box_nl)
            if update_fn is not None:
                mm_pair_idx, mm_pair_mask = update_fn(pos_np, box=box_nl)
        else:
            _ = float(atoms.get_potential_energy())
        box_warm = (
            jnp.array([float(L), float(L), float(L)], dtype=jnp.float32)
            if not free_space and L is not None
            else None
        )
        include_mm = bool(getattr(args, "include_mm", True))
        warmup_hybrid_spherical_cutoff(
            spherical_cutoff_calculator,
            atomic_numbers=jnp.asarray(z, dtype=jnp.int32),
            positions=jnp.asarray(atoms.get_positions(), dtype=jnp.float32),
            n_monomers=n_molecules,
            cutoff_params=cutoff,
            doML=True,
            doMM=include_mm,
            doML_dimer=True,
            box=box_warm,
            mm_pair_idx=mm_pair_idx,
            mm_pair_mask=mm_pair_mask,
        )
        print("[jaxmd] hybrid JIT warmup complete (delay-kernel calibration)")

    initial_energy_eV: float | None = None
    initial_fmax_eVA: float | None = None
    try:
        initial_energy_eV = float(atoms.get_potential_energy())
        initial_fmax_eVA = float(np.abs(atoms.get_forces()).max())
    except Exception:
        pass

    if (
        handoff_in is not None
        and not getattr(args, "handoff_pre_minimize", False)
        and getattr(args, "handoff_quality_gate", False)
        and initial_fmax_eVA is not None
    ):
        threshold = float(getattr(args, "handoff_quality_fmax_eVA", 1.0))
        action = str(getattr(args, "handoff_quality_action", "minimize")).lower()
        if initial_fmax_eVA > threshold:
            quality_gate_triggered = True
            msg = (
                f"Handoff quality gate: initial MMML max|F|={initial_fmax_eVA:.4f} eV/Å "
                f"> threshold {threshold:.4f} eV/Å."
            )
            if action == "error":
                raise RuntimeError(msg)
            if action == "warn":
                print(f"WARNING: {msg}", flush=True)
            elif action == "minimize":
                print(f"{msg} Enabling pre-minimization.", flush=True)
                skip_pre_min = False
                args.calculator_pre_minimize = True
                args.jaxmd_minimize_steps = saved_jaxmd_minimize_steps
                args.jaxmd_pbc_minimize_steps = saved_jaxmd_pbc_minimize_steps
                _check_or_charmm_overlap_rescue(
                    atoms,
                    monomer_offsets,
                    min_distance=args.min_intermonomer_atom_distance,
                    context="handoff quality gate",
                    nstep_sd=args.charmm_sd_steps,
                    nstep_abnr=args.charmm_abnr_steps,
                    tolenr=args.charmm_tolenr,
                    tolgrd=args.charmm_tolgrd,
                    nbxmod=args.charmm_nbxmod,
                    timings=minimization_summary,
                )

    if skip_pre_min:
        args.calculator_pre_minimize = False
        args.charmm_pre_minimize = False
        args.jaxmd_minimize_steps, args.jaxmd_pbc_minimize_steps = (
            resolve_jaxmd_minimize_steps_for_handoff(
                skip_pre_min=True,
                free_space=free_space,
                jaxmd_minimize_steps=saved_jaxmd_minimize_steps,
                jaxmd_pbc_minimize_steps=saved_jaxmd_pbc_minimize_steps,
            )
        )
        if (
            handoff_in is not None
            and not free_space
            and initial_fmax_eVA is not None
            and str((handoff_in.metadata or {}).get("backend", "")).strip().lower()
            == "pycharmm"
            and initial_fmax_eVA
            <= float(getattr(args, "handoff_quality_fmax_eVA", 1.0))
        ):
            args.jaxmd_pbc_minimize_steps = 0
            if not getattr(args, "quiet", False):
                print(
                    "Handoff: skipping PBC FIRE (pycharmm continuation, "
                    f"max|F|={initial_fmax_eVA:.4f} eV/Å)",
                    flush=True,
                )

    policy_summary = summarize_handoff_policy(
        handoff_in,
        skip_pre_min=skip_pre_min,
        handoff_pre_minimize=bool(getattr(args, "handoff_pre_minimize", False)),
        continue_velocities=bool(getattr(args, "continue_velocities", True)),
        velocity_policy="pending",
        use_handoff_velocities=False,
        box_side_A=L,
        box_source=box_source,
        box_warnings=box_warnings,
        ml_switch_width=ml_w,
        mm_switch_on=mm_on,
        mm_switch_width=mm_w,
        initial_energy_eV=initial_energy_eV,
        initial_fmax_eVA=initial_fmax_eVA,
        quality_gate_enabled=bool(getattr(args, "handoff_quality_gate", False)),
        quality_gate_triggered=quality_gate_triggered,
    )
    print_handoff_policy_panel(policy_summary, quiet=bool(getattr(args, "quiet", False)))
    write_handoff_policy_json(policy_summary, out_dir / "handoff_policy.json")

    if args.calculator_pre_minimize:
        _ = float(atoms.get_potential_energy())
        pre_bfgs_fmax = float(np.abs(atoms.get_forces()).max())
        best_frame = _BestMinimizationFrame(atoms)
        best_frame.record("initial")
        minimization_summary["pre_bfgs_fmax_eVA"] = pre_bfgs_fmax
        print(
            f"ASE BFGS pre-minimization starting "
            f"(max {args.pre_min_steps} steps, fmax={args.pre_min_fmax})"
        )

        def _check_pre_min_overlap(label: str) -> None:
            _check_or_charmm_overlap_rescue(
                atoms,
                monomer_offsets,
                min_distance=args.min_intermonomer_atom_distance,
                context=label,
                nstep_sd=args.charmm_sd_steps,
                nstep_abnr=args.charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=minimization_summary,
            )

        charmm_cubic_box = float(L) if not free_space else None

        def _run_charmm_rescue(phase: str, *, fmax_key: str) -> float:
            print(
                f"CHARMM {phase} rescue starting "
                f"(SD={args.rescue_charmm_sd_steps}, ABNR={args.rescue_charmm_abnr_steps})"
            )
            _run_charmm_minimize(
                atoms,
                nstep_sd=args.rescue_charmm_sd_steps,
                nstep_abnr=args.rescue_charmm_abnr_steps,
                tolenr=args.charmm_tolenr,
                tolgrd=args.charmm_tolgrd,
                nbxmod=args.charmm_nbxmod,
                timings=minimization_summary,
                cubic_box_side_A=charmm_cubic_box,
            )
            record_label = phase.lower().replace(" ", "_").replace("-", "_")
            best_frame.record(f"charmm_{record_label}")
            _check_pre_min_overlap(f"after CHARMM rescue ({phase})")
            fmax = float(np.abs(atoms.get_forces()).max())
            minimization_summary[fmax_key] = fmax
            print(f"CHARMM {phase} rescue complete, fmax={fmax:.6f} eV/A")
            return fmax

        def _run_ase_bfgs_rescue(phase: str, *, traj_suffix: str, fmax_key: str, iter_key: str) -> float:
            print(
                f"ASE BFGS {phase} starting "
                f"(max {args.pre_min_steps} steps, fmax={args.pre_min_fmax})"
            )
            traj_path = out_dir / f"{geom_tag}_{args.ensemble}_{traj_suffix}_bfgs_min.traj"
            opt = BFGS(
                atoms,
                logfile=None if args.quiet_bfgs else "-",
                trajectory=str(traj_path),
                maxstep=args.bfgs_maxstep,
            )
            record_label = phase.lower().replace(" ", "_").replace("-", "_")
            opt.attach(lambda: best_frame.record(f"bfgs_{record_label}"), interval=1)
            opt.attach(lambda: _check_pre_min_overlap(f"ASE BFGS {phase}"), interval=1)
            opt.run(fmax=args.pre_min_fmax, steps=args.pre_min_steps)
            best_frame.record(f"bfgs_{record_label}_final")
            _check_pre_min_overlap(f"after ASE BFGS {phase}")
            fmax = float(np.abs(atoms.get_forces()).max())
            minimization_summary[iter_key] = float(opt.get_number_of_steps())
            minimization_summary[fmax_key] = fmax
            minimization_summary[f"{traj_suffix}_bfgs_traj"] = str(traj_path.relative_to(out_dir))
            print(f"ASE BFGS {phase} complete, fmax={fmax:.6f} eV/A")
            return fmax

        def _run_ase_fire_rescue(phase: str, *, traj_suffix: str, fmax_key: str) -> float:
            print(
                f"ASE FIRE {phase} starting "
                f"(fmax target {args.pre_min_fmax:.6f})"
            )
            traj_path = out_dir / f"{geom_tag}_{args.ensemble}_{traj_suffix}_fire_min.traj"
            fire = FIRE(
                atoms,
                logfile=None if args.quiet_bfgs else "-",
                trajectory=str(traj_path),
                maxstep=args.fire_min_maxstep,
            )
            record_label = phase.lower().replace(" ", "_").replace("-", "_")
            fire.attach(lambda: best_frame.record(f"fire_{record_label}"), interval=1)
            fire.attach(lambda: _check_pre_min_overlap(f"ASE FIRE {phase}"), interval=1)
            fire.run(fmax=args.pre_min_fmax, steps=args.fire_min_steps)
            best_frame.record(f"fire_{record_label}_final")
            _check_pre_min_overlap(f"after ASE FIRE {phase}")
            fmax = float(np.abs(atoms.get_forces()).max())
            minimization_summary[fmax_key] = fmax
            minimization_summary[f"{traj_suffix}_fire_traj"] = str(traj_path.relative_to(out_dir))
            print(f"ASE FIRE {phase} complete, fmax={fmax:.6f} eV/A")
            return fmax

        def _run_mmml_after_charmm(phase: str, traj_suffix: str) -> float:
            """Relax CHARMM coordinates with the hybrid MMML calculator (BFGS then FIRE)."""
            fmax = _run_ase_bfgs_rescue(
                phase,
                traj_suffix=traj_suffix,
                fmax_key=f"{traj_suffix}_bfgs_fmax_eVA",
                iter_key=f"{traj_suffix}_bfgs_iterations",
            )
            if fmax > args.pre_min_fmax:
                fmax = _run_ase_fire_rescue(
                    phase,
                    traj_suffix=traj_suffix,
                    fmax_key=f"{traj_suffix}_fire_fmax_eVA",
                )
            return fmax

        bfgs_traj_path = out_dir / f"{geom_tag}_{args.ensemble}_bfgs_min.traj"
        opt = BFGS(
            atoms,
            logfile=None if args.quiet_bfgs else "-",
            trajectory=str(bfgs_traj_path),
            maxstep=args.bfgs_maxstep,
        )
        opt.attach(lambda: best_frame.record("bfgs"), interval=1)
        opt.attach(lambda: _check_pre_min_overlap("ASE BFGS pre-minimization"), interval=1)
        opt.run(fmax=args.pre_min_fmax, steps=args.pre_min_steps)
        best_frame.record("bfgs_final")
        _check_pre_min_overlap("after ASE BFGS pre-minimization")
        fmin = float(np.abs(atoms.get_forces()).max())
        minimization_summary["bfgs_iterations"] = float(opt.get_number_of_steps())
        minimization_summary["bfgs_fmax_eVA"] = fmin
        minimization_summary["bfgs_traj"] = str(bfgs_traj_path.relative_to(out_dir))
        print(f"ASE BFGS pre-minimization complete, fmax={fmin:.6f} eV/A")
        if fmin > args.pre_min_fmax and args.rescue_minimize:
            bfgs_best_fmax = best_frame.restore_best_force()
            minimization_summary["bfgs_best_force_fmax_eVA"] = bfgs_best_fmax
            print(
                f"Starting CHARMM/ML rescue from best BFGS frame "
                f"({best_frame.best_force_label}, fmax={bfgs_best_fmax:.6f})"
            )
            fmin = _run_charmm_rescue("pre-FIRE", fmax_key="charmm_rescue_pre_fire_fmax_eVA")
            fmin = _run_mmml_after_charmm("rescue (pre-FIRE)", "rescue")
            minimization_summary["fire_fmax_eVA"] = fmin
            if fmin > args.pre_min_fmax:
                fmin = _run_charmm_rescue("post-FIRE", fmax_key="charmm_rescue_post_fire_fmax_eVA")
                fmin = _run_mmml_after_charmm("rescue (post-CHARMM)", "post_charmm_rescue")
                minimization_summary["post_charmm_rescue_fire_fmax_eVA"] = fmin
        elif fmin > args.pre_min_fmax:
            print(
                f"Skipping CHARMM/FIRE rescue (--no-rescue-minimize); "
                f"BFGS fmax={fmin:.6f} eV/A"
            )
        minimization_summary.update(best_frame.write(out_dir, f"{geom_tag}_{args.ensemble}_pre_md"))
        fmin = best_frame.restore_best_force()
        _check_pre_min_overlap("restored best-force pre-MD structure")
        minimization_summary["restored_best_force_fmax_eVA"] = fmin
        print(
            "Restored best-force pre-MD structure "
            f"({best_frame.best_force_label}, fmax={fmin:.6f} eV/A, "
            f"E={best_frame.best_force_energy:.6f} eV)"
        )
        if fmin > args.max_fmax_after_min:
            raise RuntimeError(
                f"post-calculator minimization fmax={fmin:.6f} eV/A exceeds "
                f"--max-fmax-after-min={args.max_fmax_after_min:.6f}. "
                "Increase minimization steps or inspect the generated residue geometry."
            )
        pre_min_ran = True

    rng = np.random.default_rng(args.seed)
    if not np.all(np.isfinite(atoms.get_positions())):
        raise RuntimeError(
            "Cannot initialize velocities: atomic positions contain NaN/Inf. "
            "The upstream handoff is corrupted (often from a partial JAX-MD run after overlap "
            "or numerical instability). Re-run the predecessor stage or point "
            "--continue-from at an earlier valid frame."
        )
    velocity_policy, use_handoff_velocities = resolve_handoff_velocity_policy(
        handoff_in,
        continue_velocities=bool(getattr(args, "continue_velocities", True)),
        pre_min_ran=pre_min_ran,
    )
    initial_velocities: np.ndarray | None = None
    if use_handoff_velocities and handoff_in is not None and handoff_in.velocities is not None:
        atoms.set_velocities(np.asarray(handoff_in.velocities, dtype=float))
        if getattr(args, "handoff_velocity_remove_drift", True):
            Stationary(atoms)
            ZeroRotation(atoms)
        initial_velocities = np.asarray(atoms.get_velocities(), dtype=float)
        if not getattr(args, "quiet", False):
            print(
                f"Using handoff velocities ({len(initial_velocities)} atoms).",
                flush=True,
            )
    else:
        if (
            handoff_in is not None
            and handoff_in.velocities is not None
            and not getattr(args, "quiet", False)
        ):
            reason = (
                "pre-minimization changed coordinates"
                if pre_min_ran
                else "handoff velocities ignored (--no-continue-velocities)"
            )
            print(
                f"Re-initializing velocities (Maxwell–Boltzmann); {reason}.",
                flush=True,
            )
        from mmml.interfaces.pycharmmInterface.mlpot.charmm_ase_velocities import (
            clamp_velocity_assignment_temp_k,
        )

        mb_temp = clamp_velocity_assignment_temp_k(float(args.temperature))
        MaxwellBoltzmannDistribution(atoms, temperature_K=mb_temp, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)

    policy_summary["velocity_policy"] = velocity_policy
    policy_summary["use_handoff_velocities"] = bool(use_handoff_velocities)
    policy_summary["skip_pre_min"] = bool(skip_pre_min)
    write_handoff_policy_json(policy_summary, out_dir / "handoff_policy.json")

    def _dynamics_overlap_charmm_rescue(pos_np: np.ndarray, cell_np: np.ndarray | None) -> np.ndarray:
        """CHARMM MM minimization using PyCHARMM / CGenFF; box matches passed MD cell."""
        atoms.set_positions(np.asarray(pos_np, dtype=float))
        cubic_L: float | None = None
        if cell_np is not None:
            c = np.asarray(cell_np, dtype=float)
            if c.shape == (3, 3):
                atoms.set_cell(c)
                cubic_L = float(np.mean(np.abs(np.diagonal(c)[:3])))
            elif c.size >= 3:
                ll = np.reshape(c, (-1,))[:3]
                cubic_L = float(np.mean(np.abs(ll)))
                atoms.set_cell(np.diag(ll))
            else:
                cubic_L = float(c.reshape(-1)[0])
                atoms.set_cell([cubic_L, cubic_L, cubic_L])
            # Keeps molecules in one image so CHARMM BYGROUP image lists stay valid.
            w = _numpy_wrap_monomers_primary_cell(
                atoms.get_positions(), monomer_offsets, atoms.cell.array
            )
            atoms.set_positions(w)
        _run_charmm_minimize(
            atoms,
            nstep_sd=args.dynamics_overlap_charmm_sd_steps,
            nstep_abnr=args.dynamics_overlap_charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            nbxmod=args.charmm_nbxmod,
            cubic_box_side_A=cubic_L,
            quiet=True,
        )
        out = np.asarray(atoms.get_positions(), dtype=float)
        min_dist = float(args.min_intermonomer_atom_distance)
        if min_dist > 0.0:
            cell = atoms.cell.array if atoms.pbc.any() else None
            assert_no_intermonomer_atom_overlap(
                out,
                monomer_offsets,
                min_distance=min_dist,
                cell=cell,
                context="after CHARMM overlap rescue",
            )
        return out

    overlap_charmm_rescue_fn = None
    if not args.no_dynamics_overlap_charmm_rescue:
        overlap_charmm_rescue_fn = _dynamics_overlap_charmm_rescue

    nsteps = int(round(args.ps * 1000.0 / args.dt_fs))
    output_prefix = out_dir / f"{geom_tag}_{args.ensemble}_jaxmd"
    jargs = SimpleNamespace(
        temperature=args.temperature,
        timestep=args.dt_fs,
        ensemble=args.ensemble,
        cell=None if free_space else float(L),
        include_mm=bool(getattr(args, "include_mm", True)),
        skip_ml_dimers=False,
        debug=False,
        steps_per_recording=max(1, args.steps_per_recording),
        jax_md_update_interval=effective_update_interval,
        jax_md_skin_distance=effective_skin,
        nhc_chain_length=args.nhc_chain_length,
        nhc_chain_steps=args.nhc_chain_steps,
        nhc_sy_steps=args.nhc_sy_steps,
        nhc_tau=args.nhc_tau,
        nhc_barostat_tau=args.nhc_barostat_tau,
        pressure=args.pressure,
        npt_diagnose=False,
        nbr_monitor=False,
        output_prefix=str(output_prefix),
        nsteps_jaxmd=nsteps,
        jaxmd_minimize_steps=args.jaxmd_minimize_steps,
        jaxmd_pbc_minimize_steps=args.jaxmd_pbc_minimize_steps,
        min_intermonomer_atom_distance=args.min_intermonomer_atom_distance,
        dynamics_overlap_action=args.dynamics_overlap_action,
        traj_export_molecular_wrap=bool(args.traj_export_molecular_wrap),
        flat_bottom_radius=args.flat_bottom_radius,
        flat_bottom_k=args.flat_bottom_k,
        flat_bottom_mode=args.flat_bottom_mode,
    )
    run_sim = set_up_nhc_sim_routine(
        atoms=atoms,
        args=jargs,
        spherical_cutoff_calculator=spherical_cutoff_calculator,
        get_update_fn=get_update_fn,
        CUTOFF_PARAMS=cutoff,
        n_monomers=n_molecules,
        monomer_offsets=monomer_offsets,
        Si_mass=np.asarray(atoms.get_masses(), dtype=np.float32),
        atoms_template=atoms.copy(),
        overlap_charmm_rescue_fn=overlap_charmm_rescue_fn,
        initial_velocities=initial_velocities,
        minimization_skipped=bool(skip_pre_min),
    )
    actual_nbr_interval = int(max(1, getattr(run_sim, "neighbor_update_interval_steps", 1)))
    expected_nbr_updates = int(nsteps // actual_nbr_interval) if not free_space and get_update_fn is not None else 0
    if not free_space and get_update_fn is not None:
        print(
            f"[jaxmd_nbr] update cadence: every {actual_nbr_interval} MD steps "
            f"(expected_updates={expected_nbr_updates}, requested_interval={effective_update_interval}, "
            f"skin_distance={effective_skin:.3f} A)"
        )
    if getattr(args, "mlpot_profile", False):
        import sys

        from mmml.interfaces.pycharmmInterface.mlpot.ml_profile import (
            write_profile_git_metadata,
        )

        write_profile_git_metadata(
            args.output_dir,
            argv=sys.argv[1:],
            extra={
                "jaxmd_neighbor_profile": {
                    "ensemble": args.ensemble,
                    "nsteps": int(nsteps),
                    "steps_per_recording": int(args.steps_per_recording),
                    "requested_update_interval": int(effective_update_interval),
                    "effective_update_interval_steps": int(actual_nbr_interval),
                    "expected_neighbor_updates": int(expected_nbr_updates),
                    "skin_distance_A": float(effective_skin),
                    "free_space": bool(free_space),
                }
            },
        )
    update_fn_live = get_update_fn(np.asarray(atoms.get_positions(), dtype=np.float64), cutoff) if get_update_fn else None
    key = random.PRNGKey(args.seed)
    steps_completed, frames, boxes = run_sim(key, total_steps=nsteps)
    run_status = getattr(run_sim, "last_status", "complete")
    run_error = getattr(run_sim, "last_error", None)
    hdf5_path = Path(getattr(run_sim, "last_hdf5_path", f"{output_prefix}_{args.ensemble}.h5"))

    if len(frames) > 0:
        last_xyz = np.asarray(frames[-1], dtype=np.float64)
        if np.all(np.isfinite(last_xyz)):
            last_box = None
            if boxes is not None and len(boxes):
                last_box = np.asarray(boxes[-1], dtype=np.float64)
            out_atoms = Atoms(numbers=z, positions=last_xyz)
            if last_box is not None:
                out_atoms.set_cell(last_box)
                out_atoms.set_pbc(True)
            elif not free_space and L is not None:
                out_atoms.set_cell([L, L, L])
                out_atoms.set_pbc(True)
            final_vel = getattr(run_sim, "last_velocities", None)
            if final_vel is not None:
                out_atoms.set_velocities(np.asarray(final_vel, dtype=float))
            set_handoff_out(
                handoff_from_atoms(
                    out_atoms,
                    velocities=final_vel,
                    temperature_K=float(args.temperature),
                    pressure_atm=float(args.pressure) if args.ensemble == "npt" else None,
                    metadata={"backend": "jaxmd", "ensemble": args.ensemble},
                )
            )
        else:
            print(
                "mmml jaxmd: skipping handoff write — final frame has non-finite coordinates "
                f"(status={run_status!r}).",
                flush=True,
            )

    traj_chunk_frames = int(max(0, args.traj_chunk_frames))
    traj_paths: list[Path] = []
    if traj_chunk_frames <= 0:
        traj_paths = [out_dir / f"{geom_tag}_{args.ensemble}.traj"]
    else:
        n_parts = max(1, int(np.ceil(len(frames) / traj_chunk_frames)))
        traj_paths = [out_dir / f"{geom_tag}_{args.ensemble}.part{i:04d}.traj" for i in range(n_parts)]

    import h5py
    from ase.calculators.singlepoint import SinglePointCalculator

    velocities_data = None
    potential_energy_data = None
    forces_data = None
    if hdf5_path.exists():
        try:
            with h5py.File(hdf5_path, "r") as h5_f:
                if "velocities" in h5_f:
                    velocities_data = np.asarray(h5_f["velocities"])
                if "potential_energy" in h5_f:
                    potential_energy_data = np.asarray(h5_f["potential_energy"])
                if "forces" in h5_f:
                    forces_data = np.asarray(h5_f["forces"])
        except Exception as e:
            print(f"[warning] Failed to read energies/velocities/forces from HDF5: {e}")

    for part_idx, traj_path in enumerate(traj_paths):
        start = part_idx * traj_chunk_frames if traj_chunk_frames > 0 else 0
        stop = min(len(frames), start + traj_chunk_frames) if traj_chunk_frames > 0 else len(frames)
        traj = Trajectory(str(traj_path), "w")
        for i in range(start, stop):
            xyz = frames[i]
            f = Atoms(numbers=z, positions=np.asarray(xyz))
            if free_space:
                f.set_pbc(False)
            elif boxes is not None and i < len(boxes):
                b = np.asarray(boxes[i], dtype=float)
                f.set_cell(b)
                f.set_pbc(True)
            else:
                assert L is not None
                f.set_cell([L, L, L])
                f.set_pbc(True)

            if velocities_data is not None and i < len(velocities_data):
                f.set_velocities(np.asarray(velocities_data[i]))

            results = {}
            if potential_energy_data is not None and i < len(potential_energy_data):
                results["energy"] = float(potential_energy_data[i])
            if forces_data is not None and i < len(forces_data):
                results["forces"] = np.asarray(forces_data[i])

            if results:
                f.calc = SinglePointCalculator(f, **results)

            traj.write(f)
        traj.close()

    summary = {
        "ensemble": args.ensemble,
        "nsteps_requested": nsteps,
        "nsteps_completed": int(steps_completed),
        "frames": int(len(frames)),
        "status": run_status,
        "error": run_error,
        "h5": str(hdf5_path.relative_to(out_dir) if hdf5_path.is_relative_to(out_dir) else hdf5_path),
        "traj": str(traj_paths[0].relative_to(out_dir)),
        "traj_parts": [str(p.relative_to(out_dir)) for p in traj_paths],
        "traj_part_count": len(traj_paths),
        "box_A": (float(L) if L is not None else None),
        "free_space": bool(free_space),
        "pressure_atm": float(args.pressure),
        "temperature_K": float(args.temperature),
        "composition": {res: int(cnt) for res, cnt in composition},
        "psf_charges": psf_charge_summary,
        "charmm_minimization": {
            "nbxmod": int(args.charmm_nbxmod),
            "sd_steps": int(args.charmm_sd_steps),
            "abnr_steps": int(args.charmm_abnr_steps),
            "rescue_minimize": bool(args.rescue_minimize),
            "rescue_sd_steps": int(args.rescue_charmm_sd_steps),
            "rescue_abnr_steps": int(args.rescue_charmm_abnr_steps),
        },
        "placement": "random_3d",
        "placement_seed": int(args.seed),
        "neighbor_update_interval_steps": (
            int(getattr(run_sim, "neighbor_update_interval_steps", 1))
            if not free_space and get_update_fn is not None
            else (
                int(max(1, args.steps_per_recording)) if args.ensemble == "npt" else None
            )
        ),
        "neighbor_expected_updates": (
            int(nsteps // max(1, int(getattr(run_sim, "neighbor_update_interval_steps", 1))))
            if not free_space and get_update_fn is not None
            else (
                int(nsteps // max(1, args.steps_per_recording))
                if args.ensemble == "npt"
                else None
            )
        ),
        "neighbor_internal_update_interval_calls": effective_update_interval,
        "neighbor_internal_skin_distance_A": effective_skin,
        "pre_md_minimization": minimization_summary,
        "dynamics_overlap_action": args.dynamics_overlap_action,
        "dynamics_overlap_charmm_rescue_enabled": (not args.no_dynamics_overlap_charmm_rescue),
        "dynamics_overlap_charmm_sd_steps": int(args.dynamics_overlap_charmm_sd_steps),
        "dynamics_overlap_charmm_abnr_steps": int(args.dynamics_overlap_charmm_abnr_steps),
        "dynamics_overlap_charmm_rescue_count": int(
            getattr(run_sim, "last_charmm_overlap_rescue_count", 0)
        ),
        "traj_export_molecular_wrap": bool(args.traj_export_molecular_wrap),
        "dynamics_overlap_warning_count": int(getattr(run_sim, "last_overlap_warning_count", 0)),
        "dynamics_min_intermonomer_distance_A": (
            None
            if not np.isfinite(float(getattr(run_sim, "last_overlap_min_distance", float("inf"))))
            else float(getattr(run_sim, "last_overlap_min_distance"))
        ),
    }
    if update_fn_live is not None and hasattr(update_fn_live, "get_stats"):
        try:
            summary["mm_pair_update_stats"] = dict(update_fn_live.get_stats())
        except Exception:
            pass
    stats = summary.get("mm_pair_update_stats")
    if isinstance(stats, dict) and stats:
        summary_line = format_mm_pair_update_stats_summary(stats)
        summary["mm_pair_reuse_fraction"] = float(stats.get("reused", 0)) / max(
            1, int(stats.get("calls", 0))
        )
        print(summary_line, flush=True)
    (out_dir / "suite_summary_jaxmd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for traj_path in traj_paths:
        print(f"Wrote {traj_path}")
    if run_status != "complete":
        print(f"Partial output saved after {run_status}: {run_error}")
    print(f"Wrote {out_dir / 'suite_summary_jaxmd.json'}")
    if run_status == "interrupted":
        return 130
    if run_status == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
