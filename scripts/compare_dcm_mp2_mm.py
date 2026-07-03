#!/usr/bin/env python3
"""Compare DCM CHARMM/JAX MM and hybrid ML energies/forces to MP2 reference NPZ.

MP2 totals are electronic energies; compare **forces** and **interaction** trends,
not absolute MM vs MP2 total energies.

Examples (CHARMM node)::

    ./scripts/mmml-charmm-mpirun.sh python scripts/compare_dcm_mp2_mm.py \\
      --data new-dcm-round-2-only_MP2_41950.npz \\
      --checkpoint /path/to/checkpoint.json \\
      -o artifacts/dcm_mp2_mm_compare \\
      --reference-energy-unit hartree --reference-force-unit ev_angstrom \\
      --max-frames 200 --stride 10

    # Hybrid-only: no CHARMM / mpirun required (doMM=False)
    uv run python scripts/compare_dcm_mp2_mm.py \\
      --data new-dcm-round-2-only_MP2_41950.npz \\
      --checkpoint /path/to/checkpoint.json \\
      --hybrid-only -o artifacts/dcm_mp2_hybrid_compare --max-frames 200
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _parse_args() -> argparse.Namespace:
    from mmml.interfaces.pycharmmInterface.dcm_mp2_mm_compare import HYBRID_CALCULATOR_CHOICES

    parser = argparse.ArgumentParser(
        description="DCM MP2 geometries vs CHARMM/JAX MM and hybrid ML calculators",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="MP2 NPZ with N, Z, R, E [, F] (dimer frames N=10)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/dcm_mp2_mm_compare"),
        help="Write comparison.json and report.md here",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="CHARMM scratch for vacuum PSF build (default: output-dir/charmm_work)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MMML checkpoint for hybrid / direct ML calculators",
    )
    parser.add_argument(
        "--calculators",
        nargs="+",
        default=["hybrid-ml"],
        choices=list(HYBRID_CALCULATOR_CHOICES),
        help="Hybrid calculators to run when --checkpoint is set (default: hybrid-ml)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="ML cutoff / mm_switch_on for hybrid calculators (default: 10 Å)",
    )
    parser.add_argument(
        "--jax-platform",
        choices=["cpu", "gpu", "tpu"],
        default=None,
        help="Set JAX_PLATFORM_NAME before loading calculators",
    )
    parser.add_argument(
        "--reference-energy-unit",
        default="hartree",
        help="Unit of E in the NPZ (default: hartree; do not rely on auto-infer)",
    )
    parser.add_argument(
        "--reference-force-unit",
        default="ev_angstrom",
        help="Unit of F in the NPZ (default: ev_angstrom)",
    )
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--no-interaction",
        action="store_true",
        help="Skip MM/ML and MP2 interaction-energy comparison",
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Skip CHARMM/JAX MM; compare hybrid ML only (no libcharmm required)",
    )
    parser.add_argument(
        "--monomer-permutation",
        default="0,3,4,1,2",
        help="NPZ→PSF atom reorder per monomer (default: 0,3,4,1,2)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.jax_platform:
        os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    if args.hybrid_only and args.checkpoint is None:
        raise SystemExit("--hybrid-only requires --checkpoint")
    if args.checkpoint is None and args.calculators != ["hybrid-ml"]:
        raise SystemExit("--calculators requires --checkpoint")

    from mmml.interfaces.pycharmmInterface.dcm_mp2_mm_compare import (
        parse_monomer_permutation,
        run_dcm_mp2_mm_comparison,
    )

    perm = parse_monomer_permutation(args.monomer_permutation)
    payload = run_dcm_mp2_mm_comparison(
        args.data,
        args.output_dir,
        workdir=args.workdir,
        reference_energy_unit=args.reference_energy_unit,
        reference_force_unit=args.reference_force_unit,
        max_frames=args.max_frames,
        stride=args.stride,
        seed=args.seed,
        compute_interaction=not args.no_interaction,
        monomer_permutation=perm,
        checkpoint=args.checkpoint,
        hybrid_calculators=tuple(args.calculators),
        hybrid_cutoff=args.cutoff,
        run_mm=not args.hybrid_only,
    )
    summary = payload["summary"]
    print(f"Wrote {args.output_dir / 'comparison.json'} and report.md")
    print(f"Frames: {summary['n_frames']}")
    if payload.get("mm", {}).get("enabled"):
        jax_ch = summary.get("jax_charmm_force_rmse_ev_A", {})
        mp2_jax = summary.get("mp2_jax_force_rmse_ev_A", {})
        print(
            f"  JAX−CHARMM force RMSE mean: {jax_ch.get('mean', float('nan')):.4g} eV/Å | "
            f"MP2−JAX MM force RMSE mean: {mp2_jax.get('mean', float('nan')):.4g} eV/Å"
        )
    if args.checkpoint is not None:
        for calc_name in args.calculators:
            slug = calc_name.replace("-", "_")
            block = summary.get(f"hybrid_{slug}_mp2_force_rmse_ev_A", {})
            print(
                f"  MP2−{calc_name} force RMSE mean: "
                f"{block.get('mean', float('nan')):.4g} eV/Å"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
