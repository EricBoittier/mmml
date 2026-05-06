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
            "Run predefined MD setups (free-space NVE/NVT, periodic NVE/NVT, periodic NPT) "
            "for arbitrary residue compositions. Wraps scripts/md_10mer_mmml_pbc_suite.py "
            "and scripts/md_10mer_mmml_pbc_suite_jaxmd.py."
        )
    )
    parser.add_argument(
        "--setup",
        choices=["free_nve", "free_nvt", "pbc_nve", "pbc_nvt", "pbc_npt", "all"],
        default="pbc_nve",
        help="Simulation setup preset.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ase", "jaxmd"],
        default="auto",
        help="MD backend. auto keeps legacy routing: ASE except pbc_npt, which uses JAX-MD.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for artifacts.")
    parser.add_argument("--template-pdb", type=Path, default=None, help="Monomer template PDB path.")
    parser.add_argument("--n-molecules", type=int, default=10, help="Number of molecules for single-residue runs.")
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Residue composition string, e.g. MEOH:5,TIP3:5 (overrides --n-molecules).",
    )
    parser.add_argument("--spacing", type=float, default=5.0, help="Initial spacing in Angstrom.")
    parser.add_argument("--ps", type=float, default=1.0, help="Simulation length in ps.")
    parser.add_argument("--dt-fs", type=float, default=0.25, help="Timestep in fs.")
    parser.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split trajectory output into multi-file chunks with at most this many frames (0 = single file).",
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
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional raw args forwarded to the underlying script (use after '--').",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _append_optional(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_command(args: argparse.Namespace) -> list[str]:
    root = _repo_root()
    ase_script = root / "scripts" / "md_10mer_mmml_pbc_suite.py"
    jaxmd_script = root / "scripts" / "md_10mer_mmml_pbc_suite_jaxmd.py"
    backend = args.backend
    if backend == "auto":
        backend = "jaxmd" if args.setup == "pbc_npt" else "ase"

    if backend == "jaxmd":
        if args.setup not in {"pbc_nve", "pbc_nvt", "pbc_npt"}:
            raise ValueError("--backend jaxmd only supports periodic setups: pbc_nve, pbc_nvt, pbc_npt")
        if args.setup == "pbc_nvt" and args.nvt_integrator == "langevin":
            raise ValueError("--backend jaxmd uses NHC for pbc_nvt; use --backend ase for Langevin NVT")
        ensemble = args.setup[len("pbc_") :]
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
    _append_optional(cmd, "--checkpoint", args.checkpoint)
    _append_optional(cmd, "--output-dir", args.output_dir)
    _append_optional(cmd, "--template-pdb", args.template_pdb)
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def main() -> int:
    args = parse_args()
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
