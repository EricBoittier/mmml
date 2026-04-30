#!/usr/bin/env python3
"""
10-mer MEOH cluster: MMML-only MD comparisons (PBC vs vacuum).

Runs (when --all):
  - NVE VelocityVerlet
  - NVT Nose–Hoover chain (symplectic NVT)
  - NVT Langevin

For each mode: periodic cubic box (MIC-safe) and non-periodic (no cell).

Performance / JIT
-----------------
The hybrid calculator JIT-compiles a large XLA program on the **first** energy/force
evaluation for a given (approx.) static configuration: atom count, PBC vs vacuum, and
cutoffs. That compile often dominates wall time for minutes; **BFGS and MD reuse the
same compiled code**, so later steps are much faster per evaluation.

Improving perceived performance:
  - Run one short ``--only`` job first to warm caches, then the full ``--all`` suite
    (separate processes still recompile unless you reuse one Python session).
  - Prefer a **single long run** over many restarts if the goal is total throughput.
  - Tighter ``--pre-min-steps`` / looser ``--pre-min-fmax`` cuts BFGS cost if acceptable.
  - ``ml_batch_size`` in ``setup_calculator`` (if exposed) trades memory vs compile size.
  - GPU: ensure persistent daemon (avoid re-JIT on every driver respawn); ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
    can help memory fragmentation at the cost of some allocator overhead.

Stage timings (seconds) are recorded per run under ``timings_s`` and in ``suite_timing.json``.
Use ``PYTHONUNBUFFERED=1`` so timing lines appear promptly in ``nohup.log``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS

import mmml.interfaces.pycharmmInterface.import_pycharmm as pyci
from mmml.cli.base import resolve_checkpoint_paths
from mmml.interfaces.pycharmmInterface.import_pycharmm import reset_block, reset_block_no_internal
from mmml.interfaces.pycharmmInterface.mmml_calculator import CutoffParameters, setup_calculator
import pycharmm.param as param
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.settings as settings

pyci.read = read
pyci.settings = settings
pyci.psf = psf

reset_block()
reset_block_no_internal()

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
import test_orbax_checkpoint_cluster as _toc  # noqa: E402

_build_psf_ordered_cluster = _toc._build_psf_ordered_cluster


def _tmark() -> float:
    return time.perf_counter()


def _tlog(msg: str, log_lines: list[str] | None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_lines is not None:
        log_lines.append(line)


def _cubic_box_length(positions: np.ndarray, ml_cutoff: float, pad: float = 10.0) -> float:
    """Side length (Å) so extent + 2*(cutoff+pad) fits; MIC-safe vs model cutoff."""
    r_cut = float(ml_cutoff) + 2.0
    d = np.ptp(positions, axis=0)
    span = float(np.max(d))
    return max(span + 2.0 * (r_cut + pad), 50.0)


def _factory_mmml(
    *,
    z: np.ndarray,
    r: np.ndarray,
    n_mol: int,
    atoms_per: int,
    base_ckpt_dir: Path,
    ml_cut: float,
    mm_sw: float,
    mm_cut: float,
    cell_scalar: float | None,
    verbose: bool,
    jax_md_capacity_multiplier: float,
    jax_md_capacity_growth_factor: float,
    jax_md_max_overflow_retries: int,
    jax_md_overflow_fallback_to_cell_list: bool,
    timings: dict[str, float] | None = None,
):
    at_codes = np.asarray(psf.get_iac(), dtype=int) - 1
    n_types = len(param.get_atc())
    cell_arg = float(cell_scalar) if cell_scalar is not None else False
    t0 = _tmark()
    factory = setup_calculator(
        ATOMS_PER_MONOMER=atoms_per,
        N_MONOMERS=n_mol,
        ml_cutoff_distance=ml_cut,
        mm_switch_on=mm_sw,
        mm_cutoff=mm_cut,
        doML=True,
        doMM=True,
        doML_dimer=True,
        debug=False,
        model_restart_path=base_ckpt_dir,
        MAX_ATOMS_PER_SYSTEM=atoms_per * 2,
        cell=cell_arg,
        ep_scale=np.ones(n_types),
        sig_scale=np.ones(n_types),
        at_codes_override=at_codes,
        verbose=verbose,
        max_pairs=500_000,
        jax_md_capacity_multiplier=jax_md_capacity_multiplier,
        jax_md_capacity_growth_factor=jax_md_capacity_growth_factor,
        jax_md_max_overflow_retries=jax_md_max_overflow_retries,
        jax_md_overflow_fallback_to_cell_list=jax_md_overflow_fallback_to_cell_list,
    )
    t1 = _tmark()
    cutoff = CutoffParameters(ml_cutoff=ml_cut, mm_switch_on=mm_sw, mm_cutoff=mm_cut)
    calc_result = factory(
        atomic_numbers=z,
        atomic_positions=r,
        n_monomers=n_mol,
        cutoff_params=cutoff,
        doML=True,
        doMM=True,
        doML_dimer=True,
        backprop=True,
        debug=False,
        energy_conversion_factor=1.0,
        force_conversion_factor=1.0,
        verbose=verbose,
    )
    t2 = _tmark()
    if timings is not None:
        timings["mmml_setup_calculator_s"] = t1 - t0
        timings["mmml_factory_call_s"] = t2 - t1
    if len(calc_result) == 3:
        mmml_calc, _, _ = calc_result
    else:
        mmml_calc, _ = calc_result
    return mmml_calc


def run_md(
    *,
    name: str,
    atoms: Atoms,
    mode: str,
    dt_fs: float,
    nsteps: int,
    log_every: int,
    traj_every: int,
    out_dir: Path,
    nvt_temp_K: float,
    nve_temp_K: float,
    langevin_friction: float,
    seed: int,
    timings: dict[str, float] | None = None,
) -> dict:
    dt = dt_fs * units.fs
    traj_path = out_dir / f"{name}.traj"
    log_path = out_dir / f"{name}.log"
    summary_path = out_dir / f"{name}_run.json"

    traj = Trajectory(str(traj_path), "w", atoms)
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    t_md_entry = _tmark()

    if mode == "nve":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nve_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = VelocityVerlet(atoms, timestep=dt)
    elif mode == "nvt_nhc":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nvt_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        tdamp = 100.0 * dt
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=dt,
            temperature_K=nvt_temp_K,
            tdamp=tdamp,
            tchain=3,
            tloop=1,
        )
    elif mode == "nvt_langevin":
        MaxwellBoltzmannDistribution(atoms, temperature_K=nvt_temp_K, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
        dyn = Langevin(
            atoms,
            timestep=dt,
            temperature_K=nvt_temp_K,
            friction=langevin_friction,
            fixcm=False,
            rng=rng,
        )
    else:
        raise ValueError(mode)

    t_before_first_snapshot = _tmark()

    def snapshot(step: int) -> None:
        ep = float(atoms.get_potential_energy())
        ek = float(atoms.get_kinetic_energy())
        et = ep + ek
        temp = float(ek / (1.5 * len(atoms) * units.kB))
        fmax = float(np.abs(atoms.get_forces()).max())
        row = {
            "step": step,
            "time_ps": step * dt_fs * 1e-3,
            "Etot_eV": et,
            "Epot_eV": ep,
            "Ekin_eV": ek,
            "T_K": temp,
            "Fmax_eVA": fmax,
        }
        if mode == "nvt_nhc" and hasattr(dyn, "get_conserved_energy"):
            row["H_eV"] = float(dyn.get_conserved_energy())
        rows.append(row)
        if step % traj_every == 0:
            traj.write(atoms)

    snapshot(0)
    t_after_first_snapshot = _tmark()
    dyn.attach(lambda: snapshot(dyn.get_number_of_steps()), interval=log_every)
    t_run0 = _tmark()
    dyn.run(nsteps)
    t_run1 = _tmark()
    traj.close()
    if timings is not None:
        timings["md_entry_to_integrator_ready_s"] = t_before_first_snapshot - t_md_entry
        timings["md_first_observable_snapshot_s"] = t_after_first_snapshot - t_before_first_snapshot
        timings["md_attach_overhead_s"] = max(0.0, t_run0 - t_after_first_snapshot)
        timings["md_integrator_loop_s"] = t_run1 - t_run0
        timings["md_per_step_mean_ms"] = (t_run1 - t_run0) / max(nsteps, 1) * 1000.0

    with log_path.open("w", encoding="utf-8") as f:
        keys = ["time_ps", "Etot_eV", "Epot_eV", "Ekin_eV", "T_K", "Fmax_eVA"]
        if mode == "nvt_nhc":
            keys.append("H_eV")
        f.write(" ".join(keys) + "\n")
        for r in rows:
            f.write(" ".join(str(r[k]) for k in keys if k in r) + "\n")

    et = np.array([r["Etot_eV"] for r in rows])
    tk = np.array([r["T_K"] for r in rows])
    out = {
        "traj": str(traj_path),
        "log": str(log_path),
        "mode": mode,
        "frames_traj": 1 + nsteps // traj_every,
        "log_samples": len(rows),
        "etot_start_eV": float(et[0]),
        "etot_end_eV": float(et[-1]),
        "etot_drift_eV": float(et[-1] - et[0]),
        "etot_span_eV": float(et.max() - et.min()),
        "temp_start_K": float(tk[0]),
        "temp_end_K": float(tk[-1]),
        "temp_mean_K": float(tk.mean()),
    }
    if mode == "nvt_nhc" and "H_eV" in rows[0]:
        h = np.array([r["H_eV"] for r in rows])
        out["H_drift_eV"] = float(h[-1] - h[0])
        out["H_span_eV"] = float(h.max() - h.min())
    if timings is not None:
        out["timings_s"] = {k: float(v) for k, v in timings.items()}
    summary_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Portable .json or Orbax path (default: bundled MEOH or $MMML_CKPT).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/md_10mer_mmml_pbc_suite"))
    parser.add_argument("--template-pdb", type=Path, default=Path("mmml/generate/sample/pdb/meoh.pdb"))
    parser.add_argument("--n-molecules", type=int, default=10)
    parser.add_argument("--spacing", type=float, default=4.0)
    parser.add_argument("--ps", type=float, default=4.0, help="Simulation length (ps)")
    parser.add_argument("--dt-fs", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--traj-every", type=int, default=5000)
    parser.add_argument("--ml-cutoff", type=float, default=1.0)
    parser.add_argument("--mm-switch-on", type=float, default=5.5)
    parser.add_argument("--mm-cutoff", type=float, default=5.0)
    parser.add_argument("--pre-min-fmax", type=float, default=0.001)
    parser.add_argument("--pre-min-steps", type=int, default=2000)
    parser.add_argument("--nvt-temp-K", type=float, default=300.0)
    parser.add_argument("--nve-temp-K", type=float, default=10.0)
    parser.add_argument("--langevin-friction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--verbose-calc", action="store_true")
    parser.add_argument(
        "--jax-md-capacity-multiplier",
        type=float,
        default=1.25,
        help="Initial jax-md neighbor-list capacity multiplier.",
    )
    parser.add_argument(
        "--jax-md-capacity-growth-factor",
        type=float,
        default=1.5,
        help="Capacity multiplier growth factor on overflow retries.",
    )
    parser.add_argument(
        "--jax-md-max-overflow-retries",
        type=int,
        default=4,
        help="Max overflow-triggered jax-md neighbor-list reallocations.",
    )
    parser.add_argument(
        "--jax-md-disable-fallback",
        action="store_true",
        help="Disable fallback to cell-list pair generation after persistent jax-md overflow.",
    )
    parser.add_argument("--all", action="store_true", help="Run all 6 combinations")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Single run key, e.g. pbc_nve, vac_langevin",
    )
    parser.add_argument(
        "--skip-jit-warmup",
        action="store_true",
        help="Do not run an extra energy eval before BFGS (mixes compile time into first BFGS steps).",
    )
    parser.add_argument(
        "--quiet-bfgs",
        action="store_true",
        help="Hide ASE BFGS per-step output (default: print steps to stdout; large 10-mers can spend hours here before MD).",
    )
    args = parser.parse_args()

    t_suite0 = _tmark()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint is None:
        base_ckpt_dir, _ = resolve_checkpoint_paths(None)
    else:
        ckpt = args.checkpoint.expanduser().resolve()
        base_ckpt_dir, _ = resolve_checkpoint_paths(ckpt)

    timing_log: list[str] = []
    t_c0 = _tmark()
    z, r0 = _build_psf_ordered_cluster(
        "MEOH",
        args.n_molecules,
        args.spacing,
        template_pdb=args.template_pdb.expanduser().resolve(),
    )
    cluster_build_s = _tmark() - t_c0
    _tlog(f"cluster_build: {cluster_build_s:.3f} s", timing_log)
    n_atoms = len(z)
    atoms_per = n_atoms // args.n_molecules

    L = _cubic_box_length(r0, args.ml_cutoff)
    r_pbc = r0 - r0.mean(axis=0) + 0.5 * L

    nsteps = int(round(args.ps * 1000.0 / args.dt_fs))
    if nsteps < 1:
        raise ValueError("nsteps < 1")

    suite_summary: dict = {
        "system": {
            "residue": "MEOH",
            "n_molecules": args.n_molecules,
            "n_atoms": n_atoms,
            "spacing_A": args.spacing,
        },
        "box_A": L,
        "md": {
            "ps": args.ps,
            "dt_fs": args.dt_fs,
            "nsteps": nsteps,
            "log_every": args.log_every,
            "traj_every": args.traj_every,
        },
        "cutoffs_A": {
            "ml_cutoff": args.ml_cutoff,
            "mm_switch_on": args.mm_switch_on,
            "mm_cutoff": args.mm_cutoff,
        },
        "runs": {},
        "timing": {
            "cluster_build_s": cluster_build_s,
            "skip_jit_warmup": bool(args.skip_jit_warmup),
            "runs": {},
        },
    }

    def do_one(key: str, use_pbc: bool, mode: str) -> None:
        r = r_pbc.copy() if use_pbc else r0.copy()
        atoms = Atoms(numbers=z, positions=r)
        if use_pbc:
            atoms.set_cell([L, L, L])
            atoms.set_pbc(True)
        else:
            atoms.set_cell(None)
            atoms.set_pbc(False)

        run_timings: dict[str, float] = {}
        calc = _factory_mmml(
            z=z,
            r=atoms.get_positions(),
            n_mol=args.n_molecules,
            atoms_per=atoms_per,
            base_ckpt_dir=base_ckpt_dir,
            ml_cut=args.ml_cutoff,
            mm_sw=args.mm_switch_on,
            mm_cut=args.mm_cutoff,
            cell_scalar=L if use_pbc else None,
            verbose=args.verbose_calc,
            jax_md_capacity_multiplier=args.jax_md_capacity_multiplier,
            jax_md_capacity_growth_factor=args.jax_md_capacity_growth_factor,
            jax_md_max_overflow_retries=args.jax_md_max_overflow_retries,
            jax_md_overflow_fallback_to_cell_list=not args.jax_md_disable_fallback,
            timings=run_timings,
        )
        atoms.calc = calc
        _tlog(
            f"{key}: mmml setup_calculator {run_timings.get('mmml_setup_calculator_s', 0):.3f} s, "
            f"factory_call {run_timings.get('mmml_factory_call_s', 0):.3f} s",
            timing_log,
        )

        if not args.skip_jit_warmup:
            t_w = _tmark()
            _ = float(atoms.get_potential_energy())
            run_timings["jit_warmup_first_potential_s"] = _tmark() - t_w
            _tlog(
                f"{key}: JIT warmup (first potential energy) {run_timings['jit_warmup_first_potential_s']:.3f} s",
                timing_log,
            )
        else:
            run_timings["jit_warmup_first_potential_s"] = 0.0

        t_b = _tmark()
        _tlog(
            f"{key}: BFGS starting (max {args.pre_min_steps} steps, fmax={args.pre_min_fmax}; "
            "this is pre-MD minimization, not dynamics yet)",
            timing_log,
        )
        bfgs_log = None if args.quiet_bfgs else "-"
        opt = BFGS(atoms, logfile=bfgs_log)
        opt.run(fmax=args.pre_min_fmax, steps=args.pre_min_steps)
        run_timings["bfgs_wall_s"] = _tmark() - t_b
        fmin = float(np.abs(atoms.get_forces()).max())
        n_bfgs = int(opt.get_number_of_steps())
        run_timings["bfgs_iterations"] = float(n_bfgs)
        _tlog(
            f"{key}: BFGS {run_timings['bfgs_wall_s']:.3f} s ({n_bfgs} iters)",
            timing_log,
        )

        res = run_md(
            name=key,
            atoms=atoms,
            mode=mode,
            dt_fs=args.dt_fs,
            nsteps=nsteps,
            log_every=args.log_every,
            traj_every=args.traj_every,
            out_dir=out_dir,
            nvt_temp_K=args.nvt_temp_K,
            nve_temp_K=args.nve_temp_K,
            langevin_friction=args.langevin_friction,
            seed=args.seed,
            timings=run_timings,
        )
        res["pbc"] = use_pbc
        res["box_A"] = L if use_pbc else None
        res["fmax_after_min_eVA"] = fmin
        suite_summary["runs"][key] = res
        suite_summary["timing"]["runs"][key] = dict(run_timings)
        _tlog(
            f"{key}: MD integrator {run_timings.get('md_integrator_loop_s', 0):.3f} s "
            f"({run_timings.get('md_per_step_mean_ms', 0):.4f} ms/step mean)",
            timing_log,
        )

    if args.all:
        do_one("pbc_nve", True, "nve")
        do_one("pbc_nvt_nhc", True, "nvt_nhc")
        do_one("pbc_nvt_langevin", True, "nvt_langevin")
        do_one("vac_nve", False, "nve")
        do_one("vac_nvt_nhc", False, "nvt_nhc")
        do_one("vac_nvt_langevin", False, "nvt_langevin")
    elif args.only:
        mapping = {
            "pbc_nve": (True, "nve"),
            "pbc_nvt_nhc": (True, "nvt_nhc"),
            "pbc_nvt_langevin": (True, "nvt_langevin"),
            "vac_nve": (False, "nve"),
            "vac_nvt_nhc": (False, "nvt_nhc"),
            "vac_nvt_langevin": (False, "nvt_langevin"),
        }
        if args.only not in mapping:
            raise SystemExit(f"--only must be one of {sorted(mapping)}")
        use_pbc, mode = mapping[args.only]
        do_one(args.only, use_pbc, mode)
    else:
        raise SystemExit("Pass --all or --only <key>")

    suite_summary["timing"]["suite_total_wall_s"] = _tmark() - t_suite0
    timing_payload = {
        "timing_log": timing_log,
        "timing": suite_summary["timing"],
    }
    (out_dir / "suite_timing.json").write_text(json.dumps(timing_payload, indent=2))
    (out_dir / "suite_summary.json").write_text(json.dumps(suite_summary, indent=2))
    (out_dir / "timing_log.txt").write_text("\n".join(timing_log) + "\n", encoding="utf-8")
    print(json.dumps(suite_summary["runs"], indent=2))
    print(f"Wrote {out_dir / 'suite_summary.json'}")
    print(f"Wrote {out_dir / 'suite_timing.json'} and {out_dir / 'timing_log.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
