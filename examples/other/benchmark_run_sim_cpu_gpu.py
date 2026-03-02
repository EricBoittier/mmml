#!/usr/bin/env python3
"""Benchmark MMML simulation speed on CPU and GPU.

This script runs MMML workflows on different JAX backends and reports
wall-clock timings.

Modes:
    - full   : `make_res (optional) -> make_box -> run_sim`
    - kernel : `run_sim` only (assumes required inputs already exist)

Examples:
    # Compare both backends (runs child processes for clean backend selection)
    python examples/other/benchmark_run_sim_cpu_gpu.py \
      --backend both \
      --checkpoint /pchem-data/meuwly/boittier/home/mmml/mmml/physnetjax/ckpts/DESdimers/ \
      --res DCM --n 10 --L 22.0 --n-atoms-monomer 5 --mode full

    # Single backend run
    python examples/other/benchmark_run_sim_cpu_gpu.py \
      --backend gpu \
      --checkpoint /path/to/checkpoint \
      --res DCM --n 10 --L 22.0 --n-atoms-monomer 5 --mode kernel
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _resolve_jax_platform(backend: str, env: Dict[str, str] | None = None) -> str:
    """Map user backend label to a concrete JAX platform string."""
    if backend == "cpu":
        return "cpu"
    if backend != "gpu":
        return backend
    source_env = env if env is not None else os.environ
    # Allows explicit override on clusters where ROCm is the GPU backend.
    override = source_env.get("MMML_JAX_GPU_PLATFORM")
    if override:
        return override
    # Default to CUDA because current JAX wheels typically expose 'cuda'.
    return "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark run_sim on CPU/GPU.")
    parser.add_argument(
        "--mode",
        choices=["full", "kernel"],
        default="full",
        help="Benchmark mode: full pipeline or run_sim kernel only.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Backend to benchmark. 'both' runs CPU and GPU in subprocesses.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory passed to mmml.cli.run_sim.run.",
    )
    parser.add_argument("--res", type=str, default="DCM", help="Residue name.")
    parser.add_argument("--n", type=int, default=10, help="Number of monomers.")
    parser.add_argument("--L", type=float, default=22.0, help="Cubic box side length (A).")
    parser.add_argument(
        "--n-atoms-monomer",
        type=int,
        required=True,
        help="Number of atoms per monomer for run_sim.",
    )
    parser.add_argument("--temperature", type=float, default=0.70 * 298.0, help="Temperature (K).")
    parser.add_argument("--timestep", type=float, default=0.25, help="Timestep (fs).")
    parser.add_argument("--nsteps-jaxmd", type=int, default=10_000, help="JAX-MD steps.")
    parser.add_argument("--nsteps-ase", type=int, default=1_000, help="ASE steps.")
    parser.add_argument("--ensemble", type=str, default="nve", help="Ensemble, e.g., nve/nvt.")
    parser.add_argument(
        "--pdbfile",
        type=Path,
        default=Path("pdb/init-packmol.pdb"),
        help="PDB file input for run_sim.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="bench_run",
        help="Output prefix base (backend-specific suffix is added).",
    )
    parser.add_argument(
        "--write-interval",
        type=int,
        default=1,
        help="Trajectory/report write interval.",
    )
    parser.add_argument(
        "--heating-interval",
        type=int,
        default=100,
        help="Heating interval for ASE branch.",
    )
    parser.add_argument(
        "--run-make-res",
        action="store_true",
        help="Run make_res before make_box (recommended in fresh directories).",
    )
    parser.add_argument(
        "--skip-energy-show",
        action="store_true",
        help="Pass --skip-energy-show to make_res to avoid known cluster segfaults.",
    )
    parser.add_argument("--include-mm", action="store_true", default=True, help="Include MM terms.")
    parser.add_argument(
        "--skip-ml-dimers",
        action="store_true",
        help="Skip ML dimer correction in hybrid calculator.",
    )
    parser.add_argument("--ml-cutoff", type=float, default=0.01)
    parser.add_argument("--mm-switch-on", type=float, default=6.0)
    parser.add_argument("--mm-cutoff", type=float, default=3.0)
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _child_command(args: argparse.Namespace, backend: str) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        args.mode,
        "--backend",
        backend,
        "--checkpoint",
        str(args.checkpoint),
        "--res",
        args.res,
        "--n",
        str(args.n),
        "--L",
        str(args.L),
        "--n-atoms-monomer",
        str(args.n_atoms_monomer),
        "--temperature",
        str(args.temperature),
        "--timestep",
        str(args.timestep),
        "--nsteps-jaxmd",
        str(args.nsteps_jaxmd),
        "--nsteps-ase",
        str(args.nsteps_ase),
        "--ensemble",
        args.ensemble,
        "--pdbfile",
        str(args.pdbfile),
        "--output-prefix",
        args.output_prefix,
        "--write-interval",
        str(args.write_interval),
        "--heating-interval",
        str(args.heating_interval),
        "--ml-cutoff",
        str(args.ml_cutoff),
        "--mm-switch-on",
        str(args.mm_switch_on),
        "--mm-cutoff",
        str(args.mm_cutoff),
    ]
    if args.run_make_res:
        cmd.append("--run-make-res")
    if args.skip_energy_show:
        cmd.append("--skip-energy-show")
    if args.include_mm:
        cmd.append("--include-mm")
    if args.skip_ml_dimers:
        cmd.append("--skip-ml-dimers")
    return cmd


def _extract_json_line(stdout: str) -> Dict[str, Any] | None:
    marker = "BENCHMARK_RESULT_JSON="
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            payload = line[len(marker):].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def run_both(args: argparse.Namespace) -> int:
    combined: Dict[str, Any] = {"mode": "both", "results": {}, "errors": {}}
    for backend in ("cpu", "gpu"):
        print(f"\n=== Running backend: {backend} ===")
        cmd = _child_command(args, backend)
        env = os.environ.copy()
        platform = _resolve_jax_platform(backend, env=env)
        env["JAX_PLATFORMS"] = platform
        env["JAX_PLATFORM_NAME"] = platform

        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
        )

        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        result = _extract_json_line(proc.stdout or "")
        if proc.returncode == 0 and result is not None:
            combined["results"][backend] = result
        else:
            combined["errors"][backend] = {
                "returncode": proc.returncode,
                "json_found": result is not None,
            }
            print(
                f"[WARN] Backend '{backend}' failed "
                f"(returncode={proc.returncode}).",
                file=sys.stderr,
            )

    cpu_res = combined["results"].get("cpu")
    gpu_res = combined["results"].get("gpu")
    if cpu_res and gpu_res:
        cpu_t = cpu_res["timings_s"]["total"]
        gpu_t = gpu_res["timings_s"]["total"]
        speedup = (cpu_t / gpu_t) if gpu_t > 0 else None
        combined["gpu_speedup_vs_cpu"] = speedup
        print("\n=== Comparison ===")
        print(f"CPU total: {cpu_t:.3f} s")
        print(f"GPU total: {gpu_t:.3f} s")
        if speedup is not None:
            print(f"GPU speedup vs CPU: {speedup:.3f}x")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {args.json_out}")

    print("BENCHMARK_RESULT_JSON=" + json.dumps(combined))
    return 0 if not combined["errors"] else 1


def run_single_backend(args: argparse.Namespace) -> int:
    # Configure backend before importing JAX-dependent modules.
    platform = _resolve_jax_platform(args.backend)
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORM_NAME"] = platform

    import jax
    from mmml.cli import make_box
    from mmml.cli import make_res
    from mmml.cli.run_sim import run

    print(f"Requested backend: {args.backend}")
    print(f"Resolved JAX platform: {platform}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"JAX devices: {[str(d) for d in jax.devices()]}")

    t0 = time.perf_counter()

    t_res = None
    t_box = None
    if args.mode == "full":
        if args.run_make_res:
            res_args = argparse.Namespace(
                res=args.res,
                skip_energy_show=args.skip_energy_show,
            )
            t_res_start = time.perf_counter()
            _ = make_res.main_loop(res_args)
            t_res = time.perf_counter() - t_res_start
            print(f"make_res done in {t_res:.3f} s")

        box_args = argparse.Namespace(
            res=args.res,
            n=args.n,
            side_length=args.L,
            pdb=None,
            solvent=None,
            density=None,
        )
        t_box_start = time.perf_counter()
        _ = make_box.main_loop(box_args)
        t_box = time.perf_counter() - t_box_start
        print(f"make_box done in {t_box:.3f} s")
    else:
        print("Kernel mode: skipping make_res/make_box and timing run_sim only.")

    run_args = argparse.Namespace(
        pdbfile=Path(args.pdbfile),
        checkpoint=Path(args.checkpoint),
        n_monomers=args.n,
        n_atoms_monomer=args.n_atoms_monomer,
        cell=args.L,
        temperature=args.temperature,
        timestep=args.timestep,
        charmm_heat=False,
        charmm_equilibration=False,
        charmm_production=False,
        nsteps_jaxmd=args.nsteps_jaxmd,
        flat_bottom_radius=0,
        flat_bottom_k=0,
        nsteps_ase=args.nsteps_ase,
        ensemble=args.ensemble,
        optimize_monomers=False,
        output_prefix=f"{args.output_prefix}_{args.backend}",
        validate=False,
        include_mm=args.include_mm,
        skip_ml_dimers=args.skip_ml_dimers,
        debug=False,
        ml_cutoff=args.ml_cutoff,
        mm_switch_on=args.mm_switch_on,
        mm_cutoff=args.mm_cutoff,
        write_interval=args.write_interval,
        heating_interval=args.heating_interval,
    )

    t_run_start = time.perf_counter()
    _ = run(run_args)
    t_run = time.perf_counter() - t_run_start
    t_total = time.perf_counter() - t0

    result: Dict[str, Any] = {
        "backend_requested": args.backend,
        "backend_actual": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "mode": args.mode,
        "config": {
            "res": args.res,
            "n_monomers": args.n,
            "n_atoms_monomer": args.n_atoms_monomer,
            "cell_A": args.L,
            "temperature_K": args.temperature,
            "timestep_fs": args.timestep,
            "nsteps_jaxmd": args.nsteps_jaxmd,
            "nsteps_ase": args.nsteps_ase,
            "ensemble": args.ensemble,
            "checkpoint": str(args.checkpoint),
        },
        "timings_s": {
            "make_res": t_res,
            "make_box": t_box,
            "run_sim": t_run,
            "total": t_total,
        },
    }

    print("\n=== Timing Summary (s) ===")
    if t_res is not None:
        print(f"make_res : {t_res:.3f}")
    if t_box is not None:
        print(f"make_box : {t_box:.3f}")
    print(f"run_sim  : {t_run:.3f}")
    print(f"total    : {t_total:.3f}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {args.json_out}")

    print("BENCHMARK_RESULT_JSON=" + json.dumps(result))
    return 0


def main() -> int:
    args = parse_args()
    if args.backend == "both":
        return run_both(args)
    return run_single_backend(args)


if __name__ == "__main__":
    raise SystemExit(main())
