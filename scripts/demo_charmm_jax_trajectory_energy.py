#!/usr/bin/env python3
"""Demo: live PyCHARMM vs JAX CGenFF energy errors along a trajectory.

Runs a short CHARMM NVT segment (or reads an existing DCD), then at each saved
frame compares every MM energy component (bond, angle, Urey–Bradley, torsion,
improper, CMAP, VDW, electrostatics, totals) and force RMS.

Examples (CHARMM node)::

    export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
    JAX_PLATFORMS=cpu uv run python scripts/demo_charmm_jax_trajectory_energy.py \\
      -o /tmp/charmm_jax_traj

    # Analyze an existing DCD (PyCHARMM session must still match the PSF/topology)
    JAX_PLATFORMS=cpu uv run python scripts/demo_charmm_jax_trajectory_energy.py \\
      --dcd /path/to/traj.dcd --skip-dynamics -o /tmp/charmm_jax_traj

See ``tests/functionality/charmm/README_charmm_jax_benchmark.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyCHARMM vs JAX CGenFF MM energies over a trajectory",
    )
    parser.add_argument(
        "--case",
        choices=("trialanine_water", "tip3_water_box"),
        default="trialanine_water",
        help="Built-in CGENFF system (default: trialanine_water)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/charmm_jax_trajectory"),
        help="Write trajectory_report.md and trajectory_report.json here",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="CHARMM scratch dir (default: output-dir/charmm_work)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=8,
        help="Trajectory frames to compare (default: 8)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth DCD frame when comparing (default: 1)",
    )
    parser.add_argument(
        "--timestep-ps",
        type=float,
        default=0.0002,
        help="MD timestep in ps (default: 0.0002)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=300.0,
        help="NVT target temperature in K (default: 300)",
    )
    parser.add_argument(
        "--dcd",
        type=Path,
        default=None,
        help="Existing DCD to analyze (default: run short dynamics)",
    )
    parser.add_argument(
        "--skip-dynamics",
        action="store_true",
        help="Do not run MD; require --dcd or use --synthetic",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a deterministic noisy coordinate series instead of MD (no DCD)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Box build / synthetic trajectory seed (default: 11)",
    )
    parser.add_argument(
        "--minimize-sd",
        type=int,
        default=20,
        help="SD minimization steps before dynamics (default: 20; 0 to skip)",
    )
    parser.add_argument(
        "--reregister-cgenff",
        action="store_true",
        help="Re-read full CGENFF via READ PARAM APPEND before comparing (slow; use after MLpot)",
    )
    return parser.parse_args()


def _nbond_settings_from_cutoffs(cuts) -> object:
    from mmml.interfaces.pycharmmInterface.mm_system_energy import CharmmNbondSettings

    return CharmmNbondSettings(
        cutnb=float(cuts.cutnb),
        ctonnb=float(cuts.ctonnb),
        ctofnb=float(cuts.ctofnb),
    )


def _build_case(case: str, *, seed: int, workdir: Path):
    if case == "trialanine_water":
        from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
            build_trialanine_water_box_in_charmm,
            have_trialanine_cgenff,
        )

        if not have_trialanine_cgenff():
            raise RuntimeError("bundled CGENFF TRIA RTF not available")
        box = build_trialanine_water_box_in_charmm(
            n_waters=10,
            box_side_A=28.0,
            seed=seed,
            workdir=workdir,
            skip_reset_block=True,
        )
        return {
            "name": "trialanine_water",
            "description": "CGENFF TRIA peptide + 10× TIP3 (28 Å cube)",
            "positions0": box.positions,
            "psf_path": box.psf_path,
            "prm_path": box.cgenff_prm,
            "cell": box.cell,
            "nb_settings": _nbond_settings_from_cutoffs(box.nbond_cutoffs),
            "extra_prm_files": box.cmap_extra_prm_files,
            "box_side_A": float(box.box_side_A),
            "metadata": {
                "psf": str(box.psf_path),
                "n_waters": int(box.n_waters),
                "box_side_A": float(box.box_side_A),
            },
        }

    from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
        build_tip3_water_box,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM

    psf_path, positions, cell, cuts = build_tip3_water_box(
        n_waters=10,
        box_side_A=28.0,
        seed=seed,
        workdir=workdir,
    )
    return {
        "name": "tip3_water_box",
        "description": "CGENFF 10× TIP3 in 28 Å cube",
        "positions0": positions,
        "psf_path": psf_path,
        "prm_path": CGENFF_PRM,
        "cell": cell,
        "nb_settings": _nbond_settings_from_cutoffs(cuts),
        "extra_prm_files": (),
        "box_side_A": 28.0,
        "metadata": {"psf": str(psf_path), "n_waters": 10, "box_side_A": 28.0},
    }


def main() -> int:
    args = _parse_args()
    from mmml.interfaces.pycharmmInterface.charmm_jax_trajectory_energy import (
        compare_trajectory_mm_energy,
        load_trajectory_mm_context,
        read_trajectory_positions,
        render_trajectory_json,
        render_trajectory_markdown,
        run_short_nvt_dynamics_dcd,
        synthetic_trajectory_from_seed,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded

    ensure_pycharmm_loaded()

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = (args.workdir or out_dir / "charmm_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    case = _build_case(args.case, seed=args.seed, workdir=workdir)
    ctx = load_trajectory_mm_context(
        psf_path=case["psf_path"],
        prm_path=case["prm_path"],
        cell=case["cell"],
        nb_settings=case["nb_settings"],
        positions0=case["positions0"],
        extra_prm_files=case["extra_prm_files"],
    )

    traj_meta: dict[str, object] = {"source": "dynamics"}
    if args.synthetic:
        print("Using synthetic coordinate trajectory (no MD).", flush=True)
        positions = synthetic_trajectory_from_seed(
            case["positions0"],
            n_frames=args.n_frames,
            seed=args.seed + 3,
        )
        traj_meta = {"source": "synthetic", "seed": args.seed + 3}
    elif args.dcd is not None:
        dcd_path = args.dcd.expanduser().resolve()
        if not dcd_path.is_file():
            print(f"DCD not found: {dcd_path}", file=sys.stderr)
            return 2
        print(f"Reading trajectory from {dcd_path}", flush=True)
        positions, header = read_trajectory_positions(
            dcd_path,
            max_frames=args.n_frames,
            frame_stride=args.frame_stride,
        )
        traj_meta = {"source": "dcd", "dcd": str(dcd_path), **header}
    elif args.skip_dynamics:
        print("Provide --dcd or --synthetic when using --skip-dynamics", file=sys.stderr)
        return 2
    else:
        dcd_path = workdir / f"{case['name']}_demo.dcd"
        print(
            f"Running short NVT dynamics ({args.n_frames} frames) → {dcd_path}",
            flush=True,
        )
        run_short_nvt_dynamics_dcd(
            dcd_path=dcd_path,
            n_frames=args.n_frames,
            timestep_ps=args.timestep_ps,
            temp=args.temp,
            box_side_A=case["box_side_A"],
            minimize_sd_steps=args.minimize_sd,
        )
        positions, header = read_trajectory_positions(
            dcd_path,
            max_frames=args.n_frames,
            frame_stride=args.frame_stride,
        )
        traj_meta = {
            "source": "dynamics",
            "dcd": str(dcd_path),
            "timestep_ps": args.timestep_ps,
            "temp_K": args.temp,
            **header,
        }

    print(f"Comparing {positions.shape[0]} frames...", flush=True)
    report = compare_trajectory_mm_energy(
        positions,
        ctx,
        name=case["name"],
        description=case["description"],
        metadata={**case["metadata"], "trajectory": traj_meta},
        frame_stride=1,
        max_frames=None,
        reregister_cgenff=bool(args.reregister_cgenff),
    )

    md_path = out_dir / "trajectory_report.md"
    json_path = out_dir / "trajectory_report.json"
    md_path.write_text(render_trajectory_markdown(report), encoding="utf-8")
    json_path.write_text(render_trajectory_json(report), encoding="utf-8")

    print(render_trajectory_markdown(report))
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
