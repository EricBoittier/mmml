#!/usr/bin/env python3
"""Finite-difference force check for MMML PBC clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from mmml.cli.base import resolve_checkpoint_paths
from mmml.cli.run.md_pbc_suite.ase import (
    _cubic_box_length,
    _enforce_min_com_separation,
    _factory_mmml,
    _run_charmm_minimize,
)
from mmml.cli.run.md_pbc_suite.cluster import _build_psf_ordered_cluster
from mmml.paths import default_meoh_template_pdb


def force_fd_check(atoms: Atoms, natoms_check: int, dx: float) -> dict[str, float]:
    x0 = atoms.get_positions().copy()
    f_analytic = np.asarray(atoms.get_forces(), dtype=float)
    n_check = min(int(natoms_check), len(atoms))
    f_numeric = np.zeros((n_check, 3), dtype=float)
    for i in range(n_check):
        for a in range(3):
            xp = x0.copy()
            xm = x0.copy()
            xp[i, a] += dx
            xm[i, a] -= dx
            atoms.set_positions(xp)
            ep = float(atoms.get_potential_energy())
            atoms.set_positions(xm)
            em = float(atoms.get_potential_energy())
            f_numeric[i, a] = -(ep - em) / (2.0 * dx)
    atoms.set_positions(x0)
    _ = atoms.get_potential_energy()
    delta = f_numeric - f_analytic[:n_check, :]
    return {
        "fd_atoms_checked": float(n_check),
        "fd_dx_A": float(dx),
        "fd_force_max_abs_diff_eVA": float(np.max(np.abs(delta))),
        "fd_force_rms_diff_eVA": float(np.sqrt(np.mean(delta**2))),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
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
    p.add_argument("--output", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite/fd_force_check.json"))
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

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        base_ckpt_dir, _ = resolve_checkpoint_paths(args.checkpoint.expanduser().resolve())

    z, r0 = _build_psf_ordered_cluster(
        "MEOH",
        args.n_molecules,
        args.spacing,
        template_pdb=args.template_pdb.expanduser().resolve(),
    )
    atoms_per = len(z) // args.n_molecules
    r0 = _enforce_min_com_separation(r0, args.n_molecules, atoms_per, args.min_com_start_distance)
    L = _cubic_box_length(r0, args.ml_cutoff)
    r_pbc = r0 - r0.mean(axis=0) + 0.5 * L
    atoms = Atoms(numbers=z, positions=r_pbc)
    atoms.set_cell([L, L, L])
    atoms.set_pbc(True)
    if args.charmm_pre_minimize:
        _run_charmm_minimize(
            atoms,
            nstep_sd=args.charmm_sd_steps,
            nstep_abnr=args.charmm_abnr_steps,
            tolenr=args.charmm_tolenr,
            tolgrd=args.charmm_tolgrd,
            timings={},
        )

    calc = _factory_mmml(
        z=z,
        r=atoms.get_positions(),
        n_mol=args.n_molecules,
        atoms_per=atoms_per,
        base_ckpt_dir=base_ckpt_dir,
        ml_cut=args.ml_cutoff,
        mm_sw=args.mm_switch_on,
        mm_cut=args.mm_cutoff,
        cell_scalar=L,
        verbose=False,
        jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
        jax_md_update_interval=args.jax_md_update_interval,
        jax_md_skin_distance=args.jax_md_skin_distance,
        max_pairs=args.max_pairs,
        timings={},
    )
    atoms.calc = calc
    result = force_fd_check(atoms, args.fd_check_atoms, args.fd_check_dx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
