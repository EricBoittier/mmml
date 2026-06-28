#!/usr/bin/env python3
"""Compare hybrid GRMS across lr_solver settings on the same certified box."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _common import have_jax_pme_package, print_fail, print_header, print_pass


def _default_configs() -> list[tuple[str, str | None]]:
    configs: list[tuple[str, str | None]] = [("mic", None)]
    if have_jax_pme_package():
        for method in ("ewald", "pme", "p3m"):
            configs.append(("jax_pme", method))
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        help="solver_comparison.tsv from run_dcm_long_range_workflow.sh",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Scan run subdirs under this root when TSV is absent",
    )
    parser.add_argument("--psf", type=Path, help="Certified box PSF (live probe mode)")
    parser.add_argument("--crd", type=Path, help="Certified box CRD (live probe mode)")
    parser.add_argument("--checkpoint", type=Path, help="PhysNet checkpoint (live probe mode)")
    parser.add_argument("--box-size", type=float, default=32.0, help="Cubic box side (Å)")
    parser.add_argument("--composition", default="DCM:60")
    parser.add_argument(
        "--jax-pme-rtol",
        type=float,
        default=0.10,
        help="Relative tolerance for jax_pme method agreement",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from mmml.interfaces.pycharmmInterface.lr_solver_grms_compare import (
        LrSolverGrmsRow,
        parse_solver_comparison_tsv,
        probe_hybrid_grms_at_certified_box,
        read_hybrid_grms_from_output_dir,
        validate_lr_solver_hybrid_grms,
    )

    print_header("Hybrid GRMS across long-range solvers")

    rows: list[LrSolverGrmsRow] = []
    if args.summary_tsv is not None:
        rows = parse_solver_comparison_tsv(args.summary_tsv)
    elif args.run_root is not None:
        root = args.run_root.expanduser().resolve()
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            name = run_dir.name
            lr_solver = "jax_pme" if "jax_pme" in name else "mic"
            jax_pme_method = ""
            for method in ("ewald", "pme", "p3m"):
                if f"_{method}_" in f"_{name}_":
                    jax_pme_method = method
                    break
            rows.append(
                LrSolverGrmsRow(
                    lr_solver=lr_solver,
                    jax_pme_method=jax_pme_method,
                    run_dir=str(run_dir),
                    status="ok" if any(run_dir.iterdir()) else "fail",
                    hybrid_grms_kcalmol_A=read_hybrid_grms_from_output_dir(run_dir),
                )
            )
    elif args.psf is not None and args.crd is not None and args.checkpoint is not None:
        try:
            from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
        except Exception:
            CGENFF_PRM = None
        if CGENFF_PRM is None:
            print_fail("PyCHARMM/CGENFF not available for live probe")
            return 1
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        rows = probe_hybrid_grms_at_certified_box(
            psf=args.psf,
            crd=args.crd,
            checkpoint=args.checkpoint,
            box_size=float(args.box_size),
            composition=str(args.composition),
            lr_configs=_default_configs(),
            verbose=bool(args.verbose),
        )
    else:
        print_fail("Provide --summary-tsv, --run-root, or --psf/--crd/--checkpoint")
        return 1

    result = validate_lr_solver_hybrid_grms(rows, jax_pme_rtol=float(args.jax_pme_rtol))
    for msg in result.messages:
        print(f"  {msg}")
    if result.ok:
        print_pass("hybrid GRMS validation passed")
        return 0
    print_fail("hybrid GRMS validation failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
