#!/usr/bin/env python3
"""Finite-difference force check for MMML PBC clusters.

Canonical implementation lives in :mod:`mmml.mode_check`. Prefer::

    mmml mode-check --pbc-fd --checkpoint … --output-dir …

This module remains as a thin script-compatible entry point for older
suite launchers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmml.mode_check.forces import force_fd_check
from mmml.mode_check.pbc_fd import run_pbc_cluster_fd, write_fd_result
from mmml.paths import default_meoh_template_pdb

__all__ = ["force_fd_check", "main"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PBC cluster analytic vs finite-difference force check "
        "(prefer: mmml mode-check --pbc-fd)."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Portable .json or Orbax path (default: bundled manifest model with "
            "lowest validation force MAE, or $MMML_CKPT)."
        ),
    )
    p.add_argument("--template-pdb", type=Path, default=default_meoh_template_pdb())
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/md_10mer_mmml_pbc_suite/fd_force_check.json"),
    )
    p.add_argument("--n-molecules", type=int, default=10)
    p.add_argument("--spacing", type=float, default=5.0)
    p.add_argument("--min-com-start-distance", type=float, default=6.0)
    p.add_argument("--ml-cutoff", type=float, default=0.1)
    p.add_argument("--mm-switch-on", type=float, default=7.0)
    p.add_argument("--mm-cutoff", type=float, default=5.0)
    p.add_argument("--fd-check-atoms", type=int, default=3)
    p.add_argument("--fd-check-dx", type=float, default=1e-3)
    p.add_argument("--max-pairs", type=int, default=20_000)
    p.add_argument("--jax-md-capacity-multiplier", type=float, default=1.25)
    p.add_argument("--jax-md-capacity-growth-factor", type=float, default=1.5)
    p.add_argument("--jax-md-max-overflow-retries", type=int, default=4)
    p.add_argument("--jax-md-disable-fallback", action="store_true")
    p.add_argument("--jax-md-update-interval", type=int, default=1)
    p.add_argument("--jax-md-skin-distance", type=float, default=0.2)
    p.add_argument("--charmm-pre-minimize", action="store_true")
    p.add_argument("--charmm-sd-steps", type=int, default=25)
    p.add_argument("--charmm-abnr-steps", type=int, default=100)
    p.add_argument("--charmm-tolenr", type=float, default=1e-3)
    p.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    args = p.parse_args(argv)

    result = run_pbc_cluster_fd(
        checkpoint=args.checkpoint,
        residue="MEOH",
        n_molecules=args.n_molecules,
        spacing=args.spacing,
        min_com_start_distance=args.min_com_start_distance,
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        fd_check_atoms=args.fd_check_atoms,
        fd_check_dx=args.fd_check_dx,
        max_pairs=args.max_pairs,
        template_pdb=args.template_pdb.expanduser().resolve(),
        charmm_pre_minimize=bool(args.charmm_pre_minimize),
        charmm_sd_steps=args.charmm_sd_steps,
        charmm_abnr_steps=args.charmm_abnr_steps,
        charmm_tolenr=args.charmm_tolenr,
        charmm_tolgrd=args.charmm_tolgrd,
        jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
        jax_md_update_interval=args.jax_md_update_interval,
        jax_md_skin_distance=args.jax_md_skin_distance,
    )
    write_fd_result(result, args.output)
    from mmml.utils.rich_report import print_colored_json

    print_colored_json(result)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
