#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from md_10mer_mmml_pbc_suite import (  # noqa: E402
    _cubic_box_length,
    _check_or_charmm_overlap_rescue,
    _enforce_min_com_separation,
    _numpy_wrap_monomers_primary_cell,
    _parse_composition,
    _randomize_monomer_com_positions,
    _run_charmm_minimize,
    _validate_psf_charges,
    build_initial_cluster_from_args,
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite_jaxmd"))
    p.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    p.add_argument("--n-molecules", type=int, default=10)
    p.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Residue composition as RES:count comma list (e.g. MEOH:5,TIP3:5). Overrides --n-molecules.",
    )
    p.add_argument("--spacing", type=float, default=5.0, help="Target minimum random COM spacing in Angstrom.")
    p.add_argument("--min-com-start-distance", type=float, default=6.0)
    p.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Override periodic cubic box side length in Angstrom (default: auto from initial geometry).",
    )
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
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="packmol_sphere",
        help=(
            "Pack --composition with Packmol inside a sphere (--packmol-radius). "
            "Default: on when --composition and --packmol-radius (or legacy: --flat-bottom-radius) are set."
        ),
    )
    p.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="Å",
        dest="packmol_radius",
        help="Packmol inside-sphere radius in Angstrom (independent of --flat-bottom-radius).",
    )
    p.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Packmol sphere center in Angstrom (default: 0 0 0).",
    )
    p.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol tolerance (Å) for --packmol-sphere (default: 2.0).",
    )
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
    p.add_argument("--ml-cutoff", type=float, default=0.1)
    p.add_argument("--mm-switch-on", type=float, default=5.5)
    p.add_argument("--mm-cutoff", type=float, default=2.0)
    p.add_argument("--max-pairs", type=int, default=20_000)
    p.add_argument("--pre-min-fmax", type=float, default=0.1)
    p.add_argument("--pre-min-steps", type=int, default=50)
    p.add_argument("--bfgs-maxstep", type=float, default=0.05)
    p.add_argument("--fire-min-steps", type=int, default=100)
    p.add_argument("--fire-min-maxstep", type=float, default=0.02)
    p.add_argument("--max-fmax-after-min", type=float, default=2.0)
    p.add_argument("--quiet-bfgs", action="store_true")
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
    p.add_argument("--jax-md-capacity-multiplier", type=float, default=1.25)
    p.add_argument("--jax-md-capacity-growth-factor", type=float, default=1.5)
    p.add_argument("--jax-md-max-overflow-retries", type=int, default=4)
    p.add_argument("--jax-md-update-interval", type=int, default=10)
    p.add_argument("--jax-md-skin-distance", type=float, default=0.2)
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
        choices=["warn", "error", "off"],
        default="warn",
        help="How to handle inter-monomer distance violations during production dynamics.",
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
    args = p.parse_args()
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

    z, r0, atoms_per_list, residue_labels, _composition_summary = build_initial_cluster_from_args(
        args
    )
    n_molecules = len(atoms_per_list)
    if args.composition:
        composition = _parse_composition(args.composition)
    else:
        composition = [("MEOH", n_molecules)]
    monomer_offsets = np.zeros(n_molecules + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    psf_charge_summary = _validate_psf_charges(
        monomer_offsets=monomer_offsets,
        residue_labels=residue_labels,
        total_atoms=len(z),
    )
    if not resolve_cluster_packmol_sphere(args):
        r0 = _randomize_monomer_com_positions(
            r0,
            monomer_offsets,
            spacing=args.spacing,
            min_com_distance=max(float(args.spacing), float(args.min_com_start_distance)),
            seed=args.seed,
        )
        r0 = _enforce_min_com_separation(r0, monomer_offsets, args.min_com_start_distance)
    free_space = bool(args.free_space)
    if free_space:
        if args.box_size is not None:
            print(
                "md_10mer_mmml_pbc_suite_jaxmd: note: ignoring --box-size with --free-space "
                f"({float(args.box_size):g} Å)."
            )
        L: float | None = None
        r = r0 - r0.mean(axis=0)
    else:
        L = float(args.box_size) if args.box_size is not None else float(_cubic_box_length(r0, args.ml_cutoff))
        r = r0 - r0.mean(axis=0) + 0.5 * L
    geom_tag = "vac" if free_space else "pbc"
    atoms = Atoms(numbers=z, positions=r)
    if free_space:
        atoms.set_pbc(False)
    else:
        assert L is not None
        atoms.set_cell([L, L, L])
        atoms.set_pbc(True)
    minimization_summary: dict[str, float | str] = {}
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
    if args.charmm_pre_minimize:
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
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=True,
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
    )
    cutoff = CutoffParameters(ml_cutoff=args.ml_cutoff, mm_switch_on=args.mm_switch_on, mm_cutoff=args.mm_cutoff)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=atoms.get_positions(),
        n_monomers=n_molecules,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
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
        if fmin > args.pre_min_fmax:
            bfgs_best_fmax = best_frame.restore_best_force()
            minimization_summary["bfgs_best_force_fmax_eVA"] = bfgs_best_fmax
            print(
                f"ASE FIRE rescue starting "
                f"from best BFGS frame "
                f"({best_frame.best_force_label}, fmax={bfgs_best_fmax:.6f} > {args.pre_min_fmax:.6f})"
            )
            fire_traj_path = out_dir / f"{geom_tag}_{args.ensemble}_fire_min.traj"
            fire = FIRE(
                atoms,
                logfile=None if args.quiet_bfgs else "-",
                trajectory=str(fire_traj_path),
                maxstep=args.fire_min_maxstep,
            )
            fire.attach(lambda: best_frame.record("fire"), interval=1)
            fire.attach(lambda: _check_pre_min_overlap("ASE FIRE rescue"), interval=1)
            fire.run(fmax=args.pre_min_fmax, steps=args.fire_min_steps)
            best_frame.record("fire_final")
            _check_pre_min_overlap("after ASE FIRE rescue")
            fmin = float(np.abs(atoms.get_forces()).max())
            minimization_summary["fire_fmax_eVA"] = fmin
            minimization_summary["fire_traj"] = str(fire_traj_path.relative_to(out_dir))
            print(f"ASE FIRE rescue complete, fmax={fmin:.6f} eV/A")
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

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

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
        return np.asarray(atoms.get_positions(), dtype=float)

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
        include_mm=True,
        skip_ml_dimers=False,
        debug=False,
        steps_per_recording=max(1, args.steps_per_recording),
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
    )
    # For NPT path in jaxmd_runner, neighbor list is refreshed once per recording block.
    if args.ensemble == "npt":
        print(
            f"[jaxmd_nbr] update cadence: every {max(1, args.steps_per_recording)} MD steps "
            f"(records={int(round(args.ps * 1000.0 / args.dt_fs)) // max(1, args.steps_per_recording)})"
        )
        print(
            "[jaxmd_nbr] internal updater settings (configured pre-min reuse): "
            f"update_interval_calls={effective_update_interval}, skin_distance={effective_skin:.3f} A"
        )
    elif args.ensemble == "nvt":
        print(
            "[jaxmd_nbr] internal updater settings (NVT fixed-box reuse): "
            f"update_interval_calls={effective_update_interval}, skin_distance={effective_skin:.3f} A"
        )
    update_fn_live = get_update_fn(np.asarray(atoms.get_positions(), dtype=np.float64), cutoff) if get_update_fn else None
    key = random.PRNGKey(args.seed)
    steps_completed, frames, boxes = run_sim(key, total_steps=nsteps)
    run_status = getattr(run_sim, "last_status", "complete")
    run_error = getattr(run_sim, "last_error", None)
    hdf5_path = Path(getattr(run_sim, "last_hdf5_path", f"{output_prefix}_{args.ensemble}.h5"))

    traj_chunk_frames = int(max(0, args.traj_chunk_frames))
    traj_paths: list[Path] = []
    if traj_chunk_frames <= 0:
        traj_paths = [out_dir / f"{geom_tag}_{args.ensemble}.traj"]
    else:
        n_parts = max(1, int(np.ceil(len(frames) / traj_chunk_frames)))
        traj_paths = [out_dir / f"{geom_tag}_{args.ensemble}.part{i:04d}.traj" for i in range(n_parts)]

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
        },
        "placement": "random_3d",
        "placement_seed": int(args.seed),
        "neighbor_update_interval_steps": int(max(1, args.steps_per_recording)) if args.ensemble == "npt" else None,
        "neighbor_expected_updates": int(nsteps // max(1, args.steps_per_recording)) if args.ensemble == "npt" else None,
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

