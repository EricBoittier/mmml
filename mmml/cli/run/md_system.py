#!/usr/bin/env python3
"""CLI wrapper for generalized mixed-composition MD setup scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run predefined MD setups (free-space NVE/NVT, periodic NVE/NVT, periodic NPT, "
            "lambda TI for arbitrary compositions) for arbitrary residue compositions. "
            "Wraps scripts/md_10mer_mmml_pbc_suite.py (ASE), "
            "scripts/md_10mer_mmml_pbc_suite_jaxmd.py (JAX-MD), and "
            "mmml.cli.run.lambda_dynamics (lambda_ti). "
            "MBAR: mmml lambda-mbar --run-dir <output-dir>."
        )
    )
    parser.add_argument(
        "--setup",
        choices=["free_nve", "free_nvt", "pbc_nve", "pbc_nvt", "pbc_npt", "lambda_ti", "all"],
        default="pbc_nve",
        help=(
            "Simulation setup preset. lambda_ti: alchemical TI with CHARMM+MMML minimization "
            "per λ window (--lambda-md-mode, --backend ase|jaxmd); mmml lambda-mbar afterward."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ase", "jaxmd"],
        default="auto",
        help=(
            "MD engine: ase runs scripts/md_10mer_mmml_pbc_suite.py; jaxmd runs ..._jaxmd.py. "
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
        "--flat-bottom-radius",
        type=float,
        default=None,
        metavar="Å",
        help=(
            "Optional harmonic flat-bottom restraining system COM: V=0 inside radius R, "
            "V=k(|d|-R)^2 outside. With PBC, d is MIC displacement to box center; in vacuum, center is origin."
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
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw args forwarded to the underlying script; put this option last.",
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
        help="lambda_ti: residue name when --composition is not set.",
    )
    parser.add_argument("--skip-jit-warmup", action="store_true", help="lambda_ti: skip first MMML energy eval per window.")

    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


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


def build_command(args: argparse.Namespace) -> list[str]:
    root = _repo_root()
    ase_script = root / "scripts" / "md_10mer_mmml_pbc_suite.py"
    jaxmd_script = root / "scripts" / "md_10mer_mmml_pbc_suite_jaxmd.py"
    backend = args.backend
    if backend == "auto":
        backend = "jaxmd" if args.setup == "pbc_npt" else "ase"

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
            sys.executable,
            str(jaxmd_script),
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
            sys.executable,
            str(ase_script),
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
    _append_optional(cmd, "--flat-bottom-radius", args.flat_bottom_radius)
    if args.flat_bottom_radius is not None:
        cmd.extend(["--flat-bottom-k", str(args.flat_bottom_k)])
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def main() -> int:
    args = parse_args()
    if args.setup == "lambda_ti":
        try:
            return _run_lambda_ti_inline(args)
        except ValueError as exc:
            print(f"mmml md-system: error: {exc}", file=sys.stderr)
            return 2
    try:
        cmd = build_command(args)
    except ValueError as exc:
        print(f"mmml md-system: error: {exc}", file=sys.stderr)
        return 2
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(_repo_root()))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
