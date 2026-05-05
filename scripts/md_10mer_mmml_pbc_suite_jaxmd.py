#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from jax import random

from mmml.cli.base import resolve_checkpoint_paths
from mmml.cli.run.jaxmd_runner import set_up_nhc_sim_routine
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from md_10mer_mmml_pbc_suite import (  # noqa: E402
    _build_psf_ordered_cluster,
    _cubic_box_length,
    _enforce_min_com_separation,
    _run_charmm_minimize,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite_jaxmd"))
    p.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    p.add_argument("--n-molecules", type=int, default=10)
    p.add_argument("--spacing", type=float, default=5.0)
    p.add_argument("--min-com-start-distance", type=float, default=6.0)
    p.add_argument("--ps", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--traj-every", type=int, default=1)
    p.add_argument("--ensemble", type=str, default="npt", choices=["nve", "nvt", "npt"])
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--pressure", type=float, default=1.0, help="atm (for NPT)")
    p.add_argument("--nhc-chain-length", type=int, default=3)
    p.add_argument("--nhc-chain-steps", type=int, default=2)
    p.add_argument("--nhc-sy-steps", type=int, default=3)
    p.add_argument("--nhc-tau", type=float, default=100.0)
    p.add_argument("--nhc-barostat-tau", type=float, default=10000.0)
    p.add_argument("--steps-per-recording", type=int, default=25)
    p.add_argument("--jaxmd-minimize-steps", type=int, default=200)
    p.add_argument("--jaxmd-pbc-minimize-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ml-cutoff", type=float, default=0.1)
    p.add_argument("--mm-switch-on", type=float, default=5.5)
    p.add_argument("--mm-cutoff", type=float, default=2.0)
    p.add_argument("--max-pairs", type=int, default=20_000)
    p.add_argument("--jax-md-capacity-multiplier", type=float, default=1.25)
    p.add_argument("--jax-md-capacity-growth-factor", type=float, default=1.5)
    p.add_argument("--jax-md-max-overflow-retries", type=int, default=4)
    p.add_argument("--jax-md-update-interval", type=int, default=10)
    p.add_argument("--jax-md-skin-distance", type=float, default=0.2)
    p.add_argument("--jax-md-disable-fallback", action="store_true")
    p.add_argument("--charmm-pre-minimize", action="store_true")
    p.add_argument("--charmm-sd-steps", type=int, default=50)
    p.add_argument("--charmm-abnr-steps", type=int, default=200)
    p.add_argument("--charmm-tolenr", type=float, default=1e-3)
    p.add_argument("--charmm-tolgrd", type=float, default=1e-3)
    args = p.parse_args()

    out_dir = (Path.cwd() / args.output_dir.expanduser()).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    r = r0 - r0.mean(axis=0) + 0.5 * L
    atoms = Atoms(numbers=z, positions=r)
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

    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=args.n_molecules,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=atoms_per * 2,
        cell=float(L),
        verbose=False,
        max_pairs=args.max_pairs,
        jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
        jax_md_update_interval=args.jax_md_update_interval,
        jax_md_skin_distance=args.jax_md_skin_distance,
    )
    cutoff = CutoffParameters(ml_cutoff=args.ml_cutoff, mm_switch_on=args.mm_switch_on, mm_cutoff=args.mm_cutoff)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=atoms.get_positions(),
        n_monomers=args.n_molecules,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=False,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=False,
    )
    if len(calc_result) == 3:
        calc, spherical_cutoff_calculator, get_update_fn = calc_result
    else:
        calc, spherical_cutoff_calculator = calc_result
        get_update_fn = None
    atoms.calc = calc

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    nsteps = int(round(args.ps * 1000.0 / args.dt_fs))
    output_prefix = out_dir / f"pbc_{args.ensemble}_jaxmd"
    jargs = SimpleNamespace(
        temperature=args.temperature,
        timestep=args.dt_fs,
        ensemble=args.ensemble,
        cell=float(L),
        include_mm=True,
        skip_ml_dimers=False,
        debug=False,
        steps_per_recording=max(1, args.steps_per_recording),
        nhc_chain_length=args.nhc_chain_length,
        nhc_chain_steps=args.nhc_chain_steps,
        nhc_sy_steps=args.nhc_sy_steps,
        nhc_tau=args.nhc_tau,
        nhc_barostat_tau=args.nhc_barostat_tau,
        pressure=args.pressure,
        npt_diagnose=False,
        nbr_monitor=False,
        output_prefix=str(output_prefix),
        nsteps_jaxmd=nsteps,
        jaxmd_minimize_steps=args.jaxmd_minimize_steps,
        jaxmd_pbc_minimize_steps=args.jaxmd_pbc_minimize_steps,
    )
    run_sim = set_up_nhc_sim_routine(
        atoms=atoms,
        args=jargs,
        spherical_cutoff_calculator=spherical_cutoff_calculator,
        get_update_fn=get_update_fn,
        CUTOFF_PARAMS=cutoff,
        n_monomers=args.n_molecules,
        monomer_offsets=np.arange(0, len(z) + 1, atoms_per, dtype=int),
        Si_mass=np.asarray(atoms.get_masses(), dtype=np.float32),
        atoms_template=atoms.copy(),
    )
    key = random.PRNGKey(args.seed)
    steps_completed, frames, boxes = run_sim(key, total_steps=nsteps)

    traj_path = out_dir / f"pbc_{args.ensemble}.traj"
    traj = Trajectory(str(traj_path), "w")
    for i, xyz in enumerate(frames):
        f = Atoms(numbers=z, positions=np.asarray(xyz))
        if boxes is not None and i < len(boxes):
            b = np.asarray(boxes[i], dtype=float)
            f.set_cell(b)
            f.set_pbc(True)
        else:
            f.set_cell([L, L, L])
            f.set_pbc(True)
        traj.write(f)
    traj.close()

    summary = {
        "ensemble": args.ensemble,
        "nsteps_requested": nsteps,
        "nsteps_completed": int(steps_completed),
        "frames": int(len(frames)),
        "traj": str(traj_path.relative_to(out_dir)),
        "box_A": float(L),
        "pressure_atm": float(args.pressure),
        "temperature_K": float(args.temperature),
    }
    (out_dir / "suite_summary_jaxmd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {traj_path}")
    print(f"Wrote {out_dir / 'suite_summary_jaxmd.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

