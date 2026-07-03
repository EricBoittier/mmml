#!/usr/bin/env python3
"""Diagnose JAX MIC vs PyCHARMM nonbonded mismatch for the TRIA water box.

Writes metrics tables, JSON, and matplotlib plots — not just scalar deltas.

Examples (CHARMM node)::

    ./scripts/mmml-charmm-mpirun.sh python scripts/diagnose_trialanine_nb_mismatch.py \\
      -o artifacts/trialanine_nb_parity

    ./scripts/mmml-charmm-mpirun.sh python scripts/diagnose_trialanine_nb_mismatch.py \\
      -o /tmp/tria_diag --perturb-seed 31 --no-build --workdir /tmp/tria_box
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _perturb(pos: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pos + rng.normal(scale=0.02, size=pos.shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JAX MIC vs PyCHARMM nonbonded parity report (metrics + plots)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/trialanine_nb_parity"),
        help="Write report.md, report.json, and PNG plots here",
    )
    parser.add_argument("--seed", type=int, default=11, help="Box build RNG seed")
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=31,
        help="Gaussian coordinate perturbation seed (matches functionality test)",
    )
    parser.add_argument("--n-waters", type=int, default=10)
    parser.add_argument("--box-side-A", type=float, default=28.0)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="CHARMM scratch for box build (default: output-dir/charmm_work)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Reuse PSF/coordinates from --workdir (must exist)",
    )
    parser.add_argument("--top-n-pairs", type=int, default=20)
    parser.add_argument(
        "--category-block",
        action="store_true",
        help=(
            "Run CHARMM segment BLOCK per-category VDW/force breakdown. "
            "Can hang under mpirun unless MMML_ALLOW_SELECTIVE_BONDED_BLOCK=1."
        ),
    )
    parser.add_argument(
        "--skip-switch-audit",
        action="store_true",
        help="Skip JAX fswitch/vfswitch derivative self-check (faster)",
    )
    parser.add_argument(
        "--switch-audit-top-k",
        type=int,
        default=5,
        help="Top |VDW| pairs for switching derivative audit (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MMML_LR_SOLVER", "mic")

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.trialanine_nb_parity import (
        collect_and_render_trialanine_nb_parity,
        render_markdown_report,
    )
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        build_trialanine_water_box_in_charmm,
        have_trialanine_cgenff,
    )

    if not ensure_pycharmm_loaded():
        print("PyCHARMM not available", file=sys.stderr)
        return 2
    if not have_trialanine_cgenff():
        print("Bundled TRIA RTF missing", file=sys.stderr)
        return 2

    out_dir = args.output_dir.expanduser().resolve()
    workdir = (args.workdir or out_dir / "charmm_work").resolve()

    if args.no_build:
        from mmml.interfaces.pycharmmInterface.cgenff_bonded_reference import (
            charmm_positions_xyz_array,
        )

        psf = workdir / "trialanine-water.psf"
        if not psf.is_file():
            print(f"Missing PSF at {psf}; run without --no-build first", file=sys.stderr)
            return 2
        # Minimal box-like namespace for parity collector
        from mmml.interfaces.pycharmmInterface.import_pycharmm import CGENFF_PRM
        from mmml.interfaces.pycharmmInterface.nbonds_config import pbc_nbond_cutoffs
        from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
            trialanine_cgenff_rtf_path,
            trialanine_cmap_extra_prm_files,
        )

        class _Box:
            pass

        box = _Box()
        box.psf_path = psf
        box.positions = charmm_positions_xyz_array()
        box.cell = np.diag([args.box_side_A] * 3)
        box.nbond_cutoffs = pbc_nbond_cutoffs(args.box_side_A)
        box.cgenff_prm = CGENFF_PRM
        box.cmap_extra_prm_files = trialanine_cmap_extra_prm_files()
        box.peptide_rtf = trialanine_cgenff_rtf_path()
        box.n_waters = args.n_waters
        box.box_side_A = args.box_side_A
        box.seed = args.seed
    else:
        print("Building TRIA water box in CHARMM (may take ~30s)...", flush=True)
        box = build_trialanine_water_box_in_charmm(
            n_waters=args.n_waters,
            box_side_A=args.box_side_A,
            seed=args.seed,
            workdir=workdir,
        )
        print("Box build complete.", flush=True)

    pos = _perturb(box.positions, seed=args.perturb_seed)
    report = collect_and_render_trialanine_nb_parity(
        box,
        pos,
        out_dir,
        perturb_seed=args.perturb_seed,
        top_n_pairs=args.top_n_pairs,
        run_category_block=args.category_block,
        run_switch_audit=not args.skip_switch_audit,
        switch_audit_top_k=args.switch_audit_top_k,
        verbose=True,
    )

    print(render_markdown_report(report))
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'report.json'}")
    print(f"Plots in {out_dir}/ (*.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
