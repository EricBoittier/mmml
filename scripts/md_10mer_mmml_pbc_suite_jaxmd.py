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
    _build_cluster_from_composition,
    _parse_composition,
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
    p.add_argument(
        "--composition",
        type=str,
        default=None,
        help="Residue composition as RES:count comma list (e.g. MEOH:5,TIP3:5). Overrides --n-molecules.",
    )
    p.add_argument("--spacing", type=float, default=5.0)
    p.add_argument("--min-com-start-distance", type=float, default=6.0)
    p.add_argument("--ps", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.25)
    p.add_argument("--traj-every", type=int, default=1)
    p.add_argument(
        "--traj-chunk-frames",
        type=int,
        default=0,
        help="Split output trajectory into chunks with at most this many frames each (0 = single file).",
    )
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
    p.add_argument(
        "--npt-allow-stale-neighbors",
        action="store_true",
        help="Allow NPT to use configured neighbor update interval/skin (faster, potentially less stable).",
    )
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

    if args.composition:
        composition = _parse_composition(args.composition)
        z, r0, atoms_per_list, _ = _build_cluster_from_composition(
            composition=composition,
            spacing=args.spacing,
        )
        n_molecules = len(atoms_per_list)
    else:
        composition = [("MEOH", int(args.n_molecules))]
        z, r0, atoms_per_list, _ = _build_cluster_from_composition(
            composition=composition,
            spacing=args.spacing,
        )
        n_molecules = int(args.n_molecules)
    monomer_offsets = np.zeros(n_molecules + 1, dtype=int)
    monomer_offsets[1:] = np.cumsum(np.asarray(atoms_per_list, dtype=int))
    r0 = _enforce_min_com_separation(r0, monomer_offsets, args.min_com_start_distance)
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
        ATOMS_PER_MONOMER=atoms_per_list,
        N_MONOMERS=n_molecules,
        ml_cutoff_distance=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=max(atoms_per_list) * 2,
        cell=float(L),
        verbose=False,
        max_pairs=args.max_pairs,
        jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
        jax_md_update_interval=(
            args.jax_md_update_interval
            if (args.ensemble != "npt" or args.npt_allow_stale_neighbors)
            else 1
        ),
        jax_md_skin_distance=(
            args.jax_md_skin_distance
            if (args.ensemble != "npt" or args.npt_allow_stale_neighbors)
            else 0.0
        ),
    )
    cutoff = CutoffParameters(ml_cutoff=args.ml_cutoff, mm_switch_on=args.mm_switch_on, mm_cutoff=args.mm_cutoff)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=atoms.get_positions(),
        n_monomers=n_molecules,
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
        n_monomers=n_molecules,
        monomer_offsets=monomer_offsets,
        Si_mass=np.asarray(atoms.get_masses(), dtype=np.float32),
        atoms_template=atoms.copy(),
    )
    # For NPT path in jaxmd_runner, neighbor list is refreshed once per recording block.
    if args.ensemble == "npt":
        if args.npt_allow_stale_neighbors:
            effective_update_interval = int(max(1, args.jax_md_update_interval))
            effective_skin = float(max(0.0, args.jax_md_skin_distance))
        else:
            effective_update_interval = 1
            effective_skin = 0.0
        print(
            f"[jaxmd_nbr] update cadence: every {max(1, args.steps_per_recording)} MD steps "
            f"(records={int(round(args.ps * 1000.0 / args.dt_fs)) // max(1, args.steps_per_recording)})"
        )
        mode = "benchmark/unsafe override" if args.npt_allow_stale_neighbors else "safe default"
        print(
            f"[jaxmd_nbr] internal updater settings ({mode}): "
            f"update_interval_calls={effective_update_interval}, skin_distance={effective_skin:.3f} A"
        )
    else:
        effective_update_interval = int(max(1, args.jax_md_update_interval))
        effective_skin = float(max(0.0, args.jax_md_skin_distance))
    update_fn_live = get_update_fn(np.asarray(atoms.get_positions(), dtype=np.float64), cutoff) if get_update_fn else None
    key = random.PRNGKey(args.seed)
    steps_completed, frames, boxes = run_sim(key, total_steps=nsteps)

    traj_chunk_frames = int(max(0, args.traj_chunk_frames))
    traj_paths: list[Path] = []
    if traj_chunk_frames <= 0:
        traj_paths = [out_dir / f"pbc_{args.ensemble}.traj"]
    else:
        n_parts = max(1, int(np.ceil(len(frames) / traj_chunk_frames)))
        traj_paths = [out_dir / f"pbc_{args.ensemble}.part{i:04d}.traj" for i in range(n_parts)]

    for part_idx, traj_path in enumerate(traj_paths):
        start = part_idx * traj_chunk_frames if traj_chunk_frames > 0 else 0
        stop = min(len(frames), start + traj_chunk_frames) if traj_chunk_frames > 0 else len(frames)
        traj = Trajectory(str(traj_path), "w")
        for i in range(start, stop):
            xyz = frames[i]
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
        "traj": str(traj_paths[0].relative_to(out_dir)),
        "traj_parts": [str(p.relative_to(out_dir)) for p in traj_paths],
        "traj_part_count": len(traj_paths),
        "box_A": float(L),
        "pressure_atm": float(args.pressure),
        "temperature_K": float(args.temperature),
        "composition": {res: int(cnt) for res, cnt in composition},
        "neighbor_update_interval_steps": int(max(1, args.steps_per_recording)) if args.ensemble == "npt" else None,
        "neighbor_expected_updates": int(nsteps // max(1, args.steps_per_recording)) if args.ensemble == "npt" else None,
        "neighbor_internal_update_interval_calls": effective_update_interval,
        "neighbor_internal_skin_distance_A": effective_skin,
    }
    if update_fn_live is not None and hasattr(update_fn_live, "get_stats"):
        try:
            summary["mm_pair_update_stats"] = dict(update_fn_live.get_stats())
        except Exception:
            pass
    (out_dir / "suite_summary_jaxmd.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for traj_path in traj_paths:
        print(f"Wrote {traj_path}")
    print(f"Wrote {out_dir / 'suite_summary_jaxmd.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

