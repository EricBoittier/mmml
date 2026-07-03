#!/usr/bin/env python3
"""Diagnose JAX MIC vs PyCHARMM for a TIP3-only periodic water box (no peptide).

Isolates water_water / inter-monomer VDW without TRIA. Includes O–O pair breakdown.

Examples (CHARMM node)::

    ./scripts/mmml-charmm-mpirun.sh python scripts/diagnose_tip3_nb_mismatch.py \\
      -o artifacts/tip3_nb_parity

    ./scripts/mmml-charmm-mpirun.sh python scripts/diagnose_tip3_nb_mismatch.py \\
      -o artifacts/tip3_nb_parity --n-waters 10 --box-side-A 28 --perturb-seed 31
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

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
        description="TIP3-only box: JAX MIC vs PyCHARMM (inter-monomer / O–O VDW)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/tip3_nb_parity"),
    )
    parser.add_argument("--seed", type=int, default=11, help="Box build RNG seed")
    parser.add_argument("--perturb-seed", type=int, default=31)
    parser.add_argument("--n-waters", type=int, default=10)
    parser.add_argument("--box-side-A", type=float, default=28.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--top-n-pairs", type=int, default=20)
    parser.add_argument("--skip-switch-audit", action="store_true")
    parser.add_argument("--switch-audit-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MMML_LR_SOLVER", "mic")

    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.liquid_nb_parity import (
        collect_and_render_liquid_nb_parity,
        monomer_id_from_offsets,
        render_liquid_markdown_report,
    )
    from mmml.interfaces.pycharmmInterface.tip3_liquid_box import (
        build_tip3_liquid_box_in_charmm,
        reload_tip3_liquid_box_in_charmm,
        tip3_liquid_box_coords_path,
    )

    if not ensure_pycharmm_loaded():
        print("PyCHARMM not available", file=sys.stderr)
        return 2

    out_dir = args.output_dir.expanduser().resolve()
    workdir = (args.workdir or out_dir / "charmm_work").resolve()

    if args.no_build:
        if tip3_liquid_box_coords_path(workdir) is None:
            print(f"Workdir {workdir} missing PSF/coords. Run without --no-build.", file=sys.stderr)
            return 2
        print(f"Reloading TIP3 box from {workdir}...", flush=True)
        box = reload_tip3_liquid_box_in_charmm(
            workdir,
            box_side_A=args.box_side_A,
            n_waters=args.n_waters,
            seed=args.seed,
        )
    else:
        print(
            f"Building {args.n_waters}× TIP3 box ({args.box_side_A:.1f} Å cube)...",
            flush=True,
        )
        box = build_tip3_liquid_box_in_charmm(
            n_waters=args.n_waters,
            box_side_A=args.box_side_A,
            seed=args.seed,
            workdir=workdir,
        )
        print("Box build complete.", flush=True)

    pos = _perturb(np.asarray(box.positions, dtype=np.float64), args.perturb_seed)
    monomer_id = monomer_id_from_offsets(box.monomer_offsets, pos.shape[0])

    report = collect_and_render_liquid_nb_parity(
        box,
        pos,
        monomer_id,
        out_dir,
        perturb_seed=args.perturb_seed,
        top_n_pairs=args.top_n_pairs,
        run_switch_audit=not args.skip_switch_audit,
        switch_audit_top_k=args.switch_audit_top_k,
    )
    print(render_liquid_markdown_report(report))
    diag = report.inter_monomer_vdw
    print(
        f"\nInter-monomer VDW: JAX {diag.jax_inter_vdw_kcal:.4f} vs "
        f"CHARMM implied {diag.charmm_implied_inter_vdw_kcal:.4f} "
        f"(Δ {diag.inter_vdw_delta_kcal:+.4f} kcal/mol)",
        flush=True,
    )
    if report.tip3_oo_inter is not None and report.tip3_oo_inter.n_pairs > 0:
        oo = report.tip3_oo_inter
        print(
            f"O–O inter: VDW {oo.vdw_kcal:.4f} kcal/mol, ⟨r⟩ {oo.mean_r_A:.2f} Å, "
            f"{oo.fraction_of_inter_vdw:.0%} of inter VDW",
            flush=True,
        )
    print(f"Wrote {out_dir}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
