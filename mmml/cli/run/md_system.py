#!/usr/bin/env python3
"""CLI wrapper for generalized mixed-composition MD setup scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run predefined MD setups (free-space NVE/NVT, periodic NVE/NVT, periodic NPT, "
            "lambda TI for arbitrary compositions) for arbitrary residue compositions. "
            "Runs mmml.cli.run.md_pbc_suite (ASE, JAX-MD, or CHARMM MLpot) and "
            "mmml.cli.run.lambda_dynamics (lambda_ti). "
            "MBAR: mmml lambda-mbar --run-dir <output-dir>."
        )
    )
    parser.add_argument(
        "--setup",
        choices=[
            "free_nve",
            "free_nvt",
            "pbc_nve",
            "pbc_nvt",
            "pbc_npt",
            "lambda_ti",
            "pycharmm_minimize",
            "all",
        ],
        default="pbc_nve",
        help=(
            "Simulation setup preset. lambda_ti: alchemical TI with CHARMM+MMML minimization "
            "per λ window (--lambda-md-mode, --backend ase|jaxmd); mmml lambda-mbar afterward. "
            "pycharmm_minimize: CHARMM MLpot SD only (--backend pycharmm). "
            "free_nve/free_nvt with --backend pycharmm: MLpot minimize + vacuum MD."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ase", "jaxmd", "pycharmm"],
        default="auto",
        help=(
            "MD engine: ase runs md_pbc_suite.ase; jaxmd runs md_pbc_suite.jaxmd; "
            "pycharmm runs CHARMM MLpot (vacuum, non-PBC: SD + NVE/NVT). "
            "auto uses ASE for vacuum (free_*) and fixed-volume PBC, JAX-MD for NPT. "
            "Use jaxmd with --setup free_nve or free_nvt for open-boundary JAX-MD."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for artifacts.")
    parser.add_argument("--template-pdb", type=Path, default=None, help="Monomer template PDB path.")
    parser.add_argument("--n-molecules", type=int, default=10, help="Number of molecules for single-residue runs.")
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help=(
            "Residue composition: comma-separated RES:N entries, e.g. MEOH:5,TIP3:5. "
            "A bare RES (no ':N') implies a single copy (N=1); when this option is set, "
            "--n-molecules is not passed to the backend (use DCM:10 for ten DCM)."
        ),
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Target minimum random COM spacing in Angstrom.")
    parser.add_argument(
        "--box-size",
        type=float,
        default=None,
        help="Override periodic cubic box side length in Angstrom (default: auto from initial geometry).",
    )
    parser.add_argument("--ps", type=float, default=1.0, help="Simulation length in ps.")
    parser.add_argument("--dt-fs", type=float, default=0.25, help="Timestep in fs.")
    parser.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split trajectory output into multi-file chunks with at most this many frames (0 = single file).",
    )
    parser.add_argument(
        "--traj-export-molecular-wrap",
        action="store_true",
        help="JAX-MD only: molecular COM wrap when writing HDF5/.traj (slower).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Target temperature in K (NVT/NPT).",
    )
    parser.add_argument(
        "--nvt-integrator",
        choices=["auto", "nhc", "langevin"],
        default="auto",
        help="Integrator for NVT in ASE route. auto=nhc for homogeneous, langevin for mixed composition.",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Target pressure in atm (NPT).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed for initial placement and velocities.")
    parser.add_argument(
        "--min-intermonomer-atom-distance",
        type=float,
        default=0.1,
        help="Abort if atoms from different monomers get closer than this distance in Angstrom (<=0 disables).",
    )
    parser.add_argument(
        "--packmol-sphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pack --composition inside a sphere with Packmol (--packmol-radius). "
            "Default: on when --composition and --packmol-radius (or legacy: --flat-bottom-radius) are set."
        ),
    )
    parser.add_argument(
        "--packmol-radius",
        type=float,
        default=None,
        metavar="Å",
        help="Packmol sphere radius in Angstrom (independent of --flat-bottom-radius).",
    )
    parser.add_argument(
        "--packmol-center",
        type=float,
        nargs=3,
        metavar=("CX", "CY", "CZ"),
        default=None,
        help="Packmol sphere center in Angstrom (default: 0 0 0).",
    )
    parser.add_argument(
        "--packmol-tolerance",
        type=float,
        default=2.0,
        help="Packmol distance tolerance (Å) when using spherical packing (default: 2.0).",
    )
    parser.add_argument(
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Harmonic flat-bottom on system COM: V=0 inside radius R, V=k(|d|-R)^2 outside. "
            "Independent of --packmol-radius. Vacuum: center at origin; PBC: MIC to box center."
        ),
    )
    parser.add_argument(
        "--flat-bottom-k",
        type=float,
        default=1.0,
        metavar="eV/Å²",
        help="Flat-bottom force constant when COM is outside --flat-bottom-radius (default: 1.0).",
    )
    parser.add_argument(
        "--flat-bottom-mode",
        choices=["system", "monomer"],
        default="system",
        help=(
            "Flat-bottom anchor: system = one restraint on mass-weighted cluster COM; "
            "monomer = sum of harmonic restraints on each monomer COM (same R and k)."
        ),
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw args forwarded to the underlying script; put this option last.",
    )

    # --- pycharmm / MLpot (--backend pycharmm) --------------------------------
    parser.add_argument(
        "--fix-resids",
        type=str,
        default="1",
        help="pycharmm: monomers held in SD pass 2 (comma-separated 1-based resids)",
    )
    parser.add_argument(
        "--constrain-resids",
        type=str,
        default="",
        help="pycharmm: freeze these resids during MD (comma-separated)",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="pycharmm: skip constrained SD pass 2",
    )
    parser.add_argument(
        "--mini-nstep",
        type=int,
        default=20,
        help="pycharmm: SD steps per minimization pass before dynamics",
    )
    parser.add_argument(
        "--no-pre-minimize",
        action="store_true",
        help="pycharmm: skip SD minimization before dynamics",
    )
    parser.add_argument(
        "--echeck",
        type=float,
        default=100.0,
        help="pycharmm: CHARMM ECHECK tolerance (kcal/mol); use --no-echeck to disable",
    )
    parser.add_argument(
        "--no-echeck",
        action="store_true",
        help="pycharmm: disable CHARMM ECHECK early stop",
    )
    parser.add_argument(
        "--dyn-nprint",
        type=int,
        default=100,
        help="pycharmm: print dynamics energy every N steps",
    )
    parser.add_argument(
        "--dyn-iprfrq",
        type=int,
        default=500,
        help="pycharmm: detailed dynamics status every N steps",
    )
    parser.add_argument(
        "--skip-energy-show",
        action="store_true",
        help="pycharmm: skip CHARMM energy.show() (MPI/cluster segfault guard)",
    )
    parser.add_argument(
        "--show-energy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="pycharmm: print CHARMM energy tables (off by default)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="pycharmm: reduce CHARMM console output",
    )
    parser.add_argument(
        "--dcd-nsavc",
        type=int,
        default=1,
        help="pycharmm: DCD frame every N integration/SD steps",
    )
    parser.add_argument(
        "--no-scale-mini-nstep",
        action="store_true",
        help="pycharmm: do not auto-increase --mini-nstep for large clusters",
    )
    parser.add_argument(
        "--no-scale-echeck",
        action="store_true",
        help="pycharmm: do not auto-loosen --echeck for large clusters",
    )
    parser.add_argument(
        "--allow-high-grms",
        action="store_true",
        help="pycharmm: start dynamics even if post-min GRMS is high",
    )
    parser.add_argument(
        "--max-grms-before-dyn",
        type=float,
        default=50.0,
        help="pycharmm: abort if post-min GRMS exceeds this (kcal/mol/Å)",
    )
    parser.add_argument(
        "--test-first",
        action="store_true",
        help="pycharmm: CHARMM TEST FIRSt after MLpot SD minimization",
    )
    parser.add_argument(
        "--test-first-tol",
        type=float,
        default=0.005,
        help="pycharmm: TEST FIRSt tolerance (default: 0.005)",
    )
    parser.add_argument(
        "--test-first-step",
        type=float,
        default=1.0e-4,
        help="pycharmm: TEST FIRSt finite-difference step in Å (default: 1e-4)",
    )
    parser.add_argument(
        "--test-first-resids",
        type=str,
        default="",
        help="pycharmm: limit derivative tests to these resids (default: all atoms)",
    )
    parser.add_argument(
        "--test-first-charmm",
        action="store_true",
        help="pycharmm: also run CHARMM TEST FIRSt (ANALYTIC omits MLpot USER energy)",
    )
    parser.add_argument(
        "--test-first-update-nbonds",
        action="store_true",
        help="pycharmm: UPDATE nonbond lists before CHARMM TEST FIRSt",
    )

    # --- lambda_ti (--setup lambda_ti) ----------------------------------------
    parser.add_argument(
        "--lambda-md-mode",
        choices=["free_nve", "free_nvt", "pbc_nve", "pbc_nvt"],
        default="free_nve",
        help="lambda_ti: MD ensemble (vacuum/PBC × NVE/NVT); use --backend ase or jaxmd.",
    )
    parser.add_argument(
        "--couple-residues",
        type=str,
        default="1",
        help="lambda_ti: 1-based residue numbers sharing λ (comma-separated, cluster order).",
    )
    parser.add_argument(
        "--lambda-windows",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="lambda_ti: shared λ values for coupled residues.",
    )
    parser.add_argument("--pre-min-steps", type=int, default=50, help="lambda_ti: MMML BFGS steps per window.")
    parser.add_argument("--pre-min-fmax", type=float, default=0.1, help="lambda_ti: MMML BFGS fmax (eV/Å).")
    parser.add_argument("--min-steps", type=int, default=None, help="lambda_ti: alias for --pre-min-steps.")
    parser.add_argument("--min-fmax", type=float, default=None, help="lambda_ti: alias for --pre-min-fmax.")
    parser.add_argument("--bfgs-maxstep", type=float, default=0.05)
    parser.add_argument(
        "--charmm-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: CHARMM SD/ABNR before MMML BFGS (default on).",
    )
    parser.add_argument(
        "--calculator-pre-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: MMML-calculator BFGS after CHARMM (default on).",
    )
    parser.add_argument("--charmm-sd-steps", type=int, default=25)
    parser.add_argument("--charmm-abnr-steps", type=int, default=100)
    parser.add_argument("--charmm-tolenr", type=float, default=1e-3)
    parser.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    parser.add_argument("--charmm-nbxmod", type=int, default=5)
    parser.add_argument(
        "--rescue-minimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="lambda_ti: ASE FIRE if BFGS fmax stays high.",
    )
    parser.add_argument("--max-fmax-after-min", type=float, default=2.0)
    parser.add_argument("--n-equil", type=int, default=500, help="lambda_ti: equilibration steps per window.")
    parser.add_argument(
        "--save-equil-traj",
        action="store_true",
        help="lambda_ti: write …_eq.traj under trajectories/ during equilibration (debug).",
    )
    parser.add_argument(
        "--equil-traj-interval",
        type=int,
        default=None,
        help="lambda_ti: equil trajectory frame interval (default: --interval).",
    )
    parser.add_argument("--n-prod", type=int, default=2000, help="lambda_ti: production steps per window.")
    parser.add_argument(
        "--repeats-per-window",
        type=int,
        default=1,
        help="lambda_ti: independent repeats per λ window.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="lambda_ti: sample dU/dλ every N production steps.",
    )
    parser.add_argument(
        "--min-com-start-distance",
        type=float,
        default=2.0,
        help="lambda_ti: minimum inter-monomer COM distance after placement (Å).",
    )
    parser.add_argument(
        "--no-fix-com",
        action="store_true",
        help="lambda_ti: disable ASE FixCom (COM position can drift during MD).",
    )
    parser.add_argument(
        "--no-stationary",
        action="store_true",
        help="lambda_ti: skip Stationary/ZeroRotation on velocity init (with --no-fix-com, COM can translate).",
    )
    parser.add_argument("--ml-cutoff", type=float, default=1.0, help="lambda_ti: ML cutoff (Å).")
    parser.add_argument("--mm-switch-on", type=float, default=5.0, help="lambda_ti: MM switch-on (Å).")
    parser.add_argument("--mm-cutoff", type=float, default=5.0, help="lambda_ti: MM cutoff (Å).")
    parser.add_argument(
        "--residue",
        type=str,
        default="MEOH",
        help=(
            "Residue name when --composition is not set "
            "(lambda_ti default MEOH; use ACO for acetone with --backend pycharmm)."
        ),
    )
    parser.add_argument("--skip-jit-warmup", action="store_true", help="lambda_ti: skip first MMML energy eval per window.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="lambda_ti: skip completed production trajectories; redo partial prod.traj files.",
    )

    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _validate_packmol_sphere_args(args: argparse.Namespace) -> None:
    from mmml.interfaces.pycharmmInterface.packmol_placement import (
        resolve_packmol_sphere_radius,
        resolve_packmol_sphere_use,
    )

    if not resolve_packmol_sphere_use(
        composition=args.composition,
        packmol_radius=getattr(args, "packmol_radius", None),
        flat_bottom_radius=args.flat_bottom_radius,
        packmol_sphere=getattr(args, "packmol_sphere", None),
    ):
        return
    if not args.composition:
        raise ValueError(
            "Spherical Packmol placement requires --composition (e.g. ACO:30) "
            "and --packmol-radius (Å)."
        )
    resolve_packmol_sphere_radius(
        getattr(args, "packmol_radius", None),
        args.flat_bottom_radius,
    )


def _append_packmol_sphere_args(cmd: list[str], args: argparse.Namespace) -> None:
    from mmml.interfaces.pycharmmInterface.packmol_placement import resolve_packmol_sphere_use

    if not resolve_packmol_sphere_use(
        composition=args.composition,
        packmol_radius=getattr(args, "packmol_radius", None),
        flat_bottom_radius=args.flat_bottom_radius,
        packmol_sphere=getattr(args, "packmol_sphere", None),
    ):
        return
    cmd.append("--packmol-sphere")
    _append_optional(cmd, "--packmol-radius", getattr(args, "packmol_radius", None))
    if args.packmol_center is not None:
        cmd.extend(["--packmol-center", *[str(x) for x in args.packmol_center]])
    cmd.extend(["--packmol-tolerance", str(args.packmol_tolerance)])


def _run_lambda_ti_inline(args: argparse.Namespace) -> int:
    from mmml.cli.run.lambda_dynamics import (
        config_from_namespace,
        print_lambda_summary,
        run_lambda_dynamics,
    )

    if args.interval < 1:
        raise ValueError("--interval must be >= 1")
    if args.repeats_per_window < 1:
        raise ValueError("--repeats-per-window must be >= 1")

    if args.output_dir is None:
        args.output_dir = Path("artifacts/lambda_ti")
    if args.backend == "auto":
        args.backend = "ase"
    args.timestep_fs = args.dt_fs
    args.temperature_K = args.temperature
    args.langevin_friction = getattr(args, "langevin_friction", 0.02)

    cfg = config_from_namespace(args, repo_root=_repo_root())
    print("mmml md-system: starting lambda_ti (in-process)", flush=True)
    summary = run_lambda_dynamics(cfg)
    print_lambda_summary(summary)
    print(f"Wrote {summary['_summary_path']}")
    print(f"Snapshots: {summary['snapshots_npz']}")
    print(f"MBAR: mmml lambda-mbar --run-dir {cfg.output_dir}")
    return 0


def _pycharmm_setups() -> set[str]:
    return {"free_nve", "free_nvt", "pycharmm_minimize"}


def _apply_backend_setup_defaults(args: argparse.Namespace) -> None:
    """Align ``--setup`` with ``--backend`` when the CLI default setup is wrong for pycharmm."""
    if args.setup == "lambda_ti":
        return

    pycharmm_setups = _pycharmm_setups()
    if args.backend == "auto":
        if args.setup in pycharmm_setups:
            args.backend = "pycharmm"
        return

    if args.backend != "pycharmm":
        return

    if args.setup in pycharmm_setups:
        return

    if args.setup in ("all",):
        raise ValueError(
            f"--backend pycharmm does not support --setup {args.setup!r}. "
            f"Use one of: {', '.join(sorted(pycharmm_setups))}."
        )

    # Default md-system setup is pbc_nve; pycharmm is vacuum-only.
    print(
        f"mmml md-system: --backend pycharmm uses vacuum (free_nve); "
        f"replacing --setup {args.setup!r}",
        flush=True,
    )
    args.setup = "free_nve"


def build_pycharmm_command(args: argparse.Namespace) -> list[str]:
    if args.setup == "pycharmm_minimize":
        phase = "minimize"
        ensemble = "nve"
    elif args.setup == "free_nvt":
        phase = "full"
        ensemble = "nvt"
    else:
        phase = "full"
        ensemble = "nve"

    if args.output_dir is None:
        args.output_dir = Path("artifacts/pycharmm_mlpot")

    cmd = [
        "--phase",
        phase,
        "--ensemble",
        ensemble,
        "--spacing",
        str(args.spacing),
        "--ps",
        str(args.ps),
        "--dt-fs",
        str(args.dt_fs),
        "--temperature",
        str(args.temperature),
        "--mini-nstep",
        str(args.mini_nstep),
        "--residue",
        str(args.residue),
        "--fix-resids",
        str(args.fix_resids),
        "--constrain-resids",
        str(args.constrain_resids),
        "--dyn-nprint",
        str(args.dyn_nprint),
        "--dyn-iprfrq",
        str(args.dyn_iprfrq),
        "--dcd-nsavc",
        str(args.dcd_nsavc),
        "--echeck",
        str(args.echeck),
    ]
    if args.composition:
        cmd.extend(["--composition", str(args.composition)])
    else:
        cmd.extend(["--n-molecules", str(args.n_molecules)])
    _append_optional(cmd, "--checkpoint", args.checkpoint)
    _append_optional(cmd, "--output-dir", args.output_dir)
    if args.no_fix:
        cmd.append("--no-fix")
    if args.no_pre_minimize:
        cmd.append("--no-pre-minimize")
    if args.charmm_pre_minimize is False:
        cmd.append("--no-charmm-pre-minimize")
    cmd.extend(["--charmm-sd-steps", str(args.charmm_sd_steps)])
    cmd.extend(["--charmm-abnr-steps", str(args.charmm_abnr_steps)])
    cmd.extend(["--charmm-tolenr", str(args.charmm_tolenr)])
    cmd.extend(["--charmm-tolgrd", str(args.charmm_tolgrd)])
    if args.no_echeck:
        cmd.append("--no-echeck")
    if getattr(args, "skip_energy_show", False):
        cmd.append("--skip-energy-show")
    if getattr(args, "show_energy", None) is True:
        cmd.append("--show-energy")
    elif getattr(args, "show_energy", None) is False:
        cmd.append("--no-show-energy")
    if args.quiet:
        cmd.append("--quiet")
    if getattr(args, "no_scale_mini_nstep", False):
        cmd.append("--no-scale-mini-nstep")
    if getattr(args, "no_scale_echeck", False):
        cmd.append("--no-scale-echeck")
    if getattr(args, "allow_high_grms", False):
        cmd.append("--allow-high-grms")
    cmd.extend(["--max-grms-before-dyn", str(args.max_grms_before_dyn)])
    if getattr(args, "test_first", False):
        cmd.append("--test-first")
        cmd.extend(["--test-first-tol", str(args.test_first_tol)])
        cmd.extend(["--test-first-step", str(args.test_first_step)])
        if getattr(args, "test_first_resids", ""):
            cmd.extend(["--test-first-resids", str(args.test_first_resids)])
        if getattr(args, "test_first_charmm", False):
            cmd.append("--test-first-charmm")
        if getattr(args, "test_first_update_nbonds", False):
            cmd.append("--test-first-update-nbonds")
    if args.flat_bottom_radius is not None:
        cmd.extend(["--fb-rad", str(args.flat_bottom_radius)])
        cmd.extend(["--fb-forc", str(args.flat_bottom_k)])
    cmd.extend(["--seed", str(args.seed)])
    _append_packmol_sphere_args(cmd, args)
    if getattr(args, "flat_bottom_radius", None) is not None:
        cmd.extend(["--flat-bottom-radius", str(args.flat_bottom_radius)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def build_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    backend = args.backend
    if backend == "auto":
        backend = "jaxmd" if args.setup == "pbc_npt" else "ase"

    if backend == "pycharmm":
        if args.setup not in _pycharmm_setups():
            raise ValueError(
                f"--backend pycharmm supports {_pycharmm_setups()}; got {args.setup!r}"
            )
        return "pycharmm", build_pycharmm_command(args)

    skip_box_size_for_cmd = False
    if backend == "jaxmd":
        jaxmd_setups = {"pbc_nve", "pbc_nvt", "pbc_npt", "free_nve", "free_nvt"}
        if args.setup not in jaxmd_setups:
            raise ValueError(
                f"--backend jaxmd supports pbc_nve, pbc_nvt, pbc_npt, free_nve, free_nvt; got {args.setup!r}"
            )
        if args.setup.endswith("_nvt") and args.nvt_integrator == "langevin":
            raise ValueError(
                "--backend jaxmd uses Nose-Hoover chain NVT; use --backend ase for Langevin NVT"
            )
        if args.setup.startswith("free_"):
            ensemble = args.setup[len("free_") :]
            jaxmd_free = True
        else:
            ensemble = args.setup[len("pbc_") :]
            jaxmd_free = False
        cmd = [
            "--ensemble",
            ensemble,
            "--spacing",
            str(args.spacing),
            "--ps",
            str(args.ps),
            "--dt-fs",
            str(args.dt_fs),
            "--temperature",
            str(args.temperature),
            "--pressure",
            str(args.pressure),
            "--traj-chunk-frames",
            str(args.traj_chunk_frames),
        ]
        if jaxmd_free:
            cmd.append("--free-space")
            skip_box_size_for_cmd = True
        if args.traj_export_molecular_wrap:
            cmd.append("--traj-export-molecular-wrap")
    else:
        if args.setup == "pbc_npt":
            raise ValueError("pbc_npt requires --backend jaxmd or --backend auto")
        cmd = [
            "--spacing",
            str(args.spacing),
            "--ps",
            str(args.ps),
            "--dt-fs",
            str(args.dt_fs),
            "--traj-chunk-frames",
            str(args.traj_chunk_frames),
        ]
        if args.setup == "all":
            cmd.append("--all")
        elif args.setup == "free_nve":
            cmd.extend(["--only", "vac_nve"])
        elif args.setup == "free_nvt":
            if args.nvt_integrator == "auto":
                use_langevin = bool(args.composition)
            else:
                use_langevin = args.nvt_integrator == "langevin"
            cmd.extend(["--only", "vac_nvt_langevin" if use_langevin else "vac_nvt_nhc"])
            cmd.extend(["--nvt-temp-K", str(args.temperature)])
        elif args.setup == "pbc_nve":
            cmd.extend(["--only", "pbc_nve"])
        elif args.setup == "pbc_nvt":
            if args.nvt_integrator == "auto":
                use_langevin = bool(args.composition)
            else:
                use_langevin = args.nvt_integrator == "langevin"
            cmd.extend(["--only", "pbc_nvt_langevin" if use_langevin else "pbc_nvt_nhc"])
            cmd.extend(["--nvt-temp-K", str(args.temperature)])
        else:
            raise ValueError(f"Unsupported setup: {args.setup}")

    if args.composition:
        cmd.extend(["--composition", str(args.composition)])
    else:
        cmd.extend(["--n-molecules", str(args.n_molecules)])
    if not skip_box_size_for_cmd:
        _append_optional(cmd, "--box-size", args.box_size)
    _append_optional(cmd, "--checkpoint", args.checkpoint)
    _append_optional(cmd, "--output-dir", args.output_dir)
    _append_optional(cmd, "--template-pdb", args.template_pdb)
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--min-intermonomer-atom-distance", str(args.min_intermonomer_atom_distance)])
    _append_packmol_sphere_args(cmd, args)
    _append_optional(cmd, "--flat-bottom-radius", args.flat_bottom_radius)
    if args.flat_bottom_radius is not None:
        cmd.extend(["--flat-bottom-k", str(args.flat_bottom_k)])
        cmd.extend(["--flat-bottom-mode", str(args.flat_bottom_mode)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return backend, cmd


def main() -> int:
    args = parse_args()
    try:
        _validate_packmol_sphere_args(args)
    except ValueError as exc:
        print(f"mmml md-system: error: {exc}", file=sys.stderr)
        return 2
    if args.setup == "lambda_ti":
        try:
            return _run_lambda_ti_inline(args)
        except ValueError as exc:
            print(f"mmml md-system: error: {exc}", file=sys.stderr)
            return 2
    try:
        _apply_backend_setup_defaults(args)
    except ValueError as exc:
        print(f"mmml md-system: error: {exc}", file=sys.stderr)
        return 2
    try:
        backend, argv = build_command(args)
    except ValueError as exc:
        print(f"mmml md-system: error: {exc}", file=sys.stderr)
        return 2
    print(f"mmml md-system: running {backend} in-process:", " ".join(argv), flush=True)
    if backend == "ase":
        from mmml.cli.run.md_pbc_suite import ase as backend_mod
    elif backend == "pycharmm":
        from mmml.cli.run.md_pbc_suite import pycharmm_mlpot as backend_mod
    else:
        from mmml.cli.run.md_pbc_suite import jaxmd as backend_mod
    return int(backend_mod.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
