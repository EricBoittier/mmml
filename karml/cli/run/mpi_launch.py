"""``mmml mpi-launch`` — compose MPI topology and JAX execution policy."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence

JAX_MODES = ("cpu-threaded", "gpu-single", "gpu-per-rank", "rank0", "spatial")
PRESETS = ("single", "cpu", "spatial")


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def shell_command(self) -> str:
        assignments = " ".join(
            f"{key}={shlex.quote(self.env[key])}"
            for key in sorted(self.env)
            if key.startswith("MMML_")
            or key in {"JAX_PLATFORMS", "OMP_NUM_THREADS", "XLA_FLAGS"}
        )
        command = shlex.join(self.argv)
        return f"{assignments} {command}".strip()


def _available_cpus(env: Mapping[str, str]) -> int:
    for key in ("SLURM_CPUS_PER_TASK", "MMML_ALLOCATED_CPUS"):
        raw = env.get(key, "")
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return os.cpu_count() or 1


def build_launch_plan(
    command: Sequence[str],
    *,
    mpi_ranks: int = 1,
    jax_mode: str = "gpu-single",
    jax_cpu_threads: int | None = None,
    charmm_omp_threads: int = 1,
    preset: str | None = None,
    strict_resources: bool = False,
    environ: Mapping[str, str] | None = None,
    python: str | None = None,
    wrapper: Path | None = None,
) -> LaunchPlan:
    """Build, but do not execute, the wrapper launch command."""
    if not command:
        raise ValueError("a command is required after mpi-launch options")
    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be positive")
    if charmm_omp_threads < 1:
        raise ValueError("charmm_omp_threads must be positive")
    if preset not in (None, *PRESETS):
        raise ValueError(f"unknown preset {preset!r}")

    if preset == "single":
        mpi_ranks, jax_mode = 1, "gpu-single"
    elif preset == "cpu":
        mpi_ranks, jax_mode = 1, "cpu-threaded"
    elif preset == "spatial":
        jax_mode = "spatial"
        if mpi_ranks == 1:
            raise ValueError("the spatial preset requires --mpi-ranks greater than 1")

    if jax_mode not in JAX_MODES:
        raise ValueError(f"jax_mode must be one of {JAX_MODES}")
    if jax_mode == "gpu-single" and mpi_ranks != 1:
        raise ValueError("gpu-single requires exactly one MPI rank")
    if jax_mode == "spatial" and mpi_ranks < 2:
        raise ValueError("spatial mode requires at least two MPI ranks")

    base_env = dict(os.environ if environ is None else environ)
    available = _available_cpus(base_env)
    if jax_cpu_threads is None:
        jax_cpu_threads = max(1, available // mpi_ranks) if jax_mode == "cpu-threaded" else 1
    if jax_cpu_threads < 1:
        raise ValueError("jax_cpu_threads must be positive")

    requested_per_rank = max(charmm_omp_threads, jax_cpu_threads)
    requested_total = mpi_ranks * requested_per_rank
    warnings: list[str] = []
    if requested_total > available:
        message = (
            f"requested up to {requested_total} CPU threads "
            f"({mpi_ranks} ranks x {requested_per_rank}) but only {available} are allocated"
        )
        if strict_resources:
            raise ValueError(message)
        warnings.append(message)

    launch_env = dict(base_env)
    launch_env.update(
        {
            "MMML_MPI_NP": str(mpi_ranks),
            "MMML_JAX_MODE": jax_mode,
            "MMML_JAX_CPU_THREADS": str(jax_cpu_threads),
            "MMML_CHARMM_OMP_THREADS": str(charmm_omp_threads),
            "OMP_NUM_THREADS": str(charmm_omp_threads),
            # Preserve the interpreter path selected by ``uv run``. Resolving
            # symlinks (notably /tmp -> /private/tmp on macOS) is unnecessary
            # and makes the launch plan differ from the active environment.
            "MMML_PYTHON": str(Path(python or sys.executable).expanduser().absolute()),
        }
    )
    if jax_mode == "cpu-threaded":
        launch_env["JAX_PLATFORMS"] = "cpu"
        launch_env["MMML_NO_JAX_COMPILE_THREADS"] = "1"
        launch_env["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={jax_cpu_threads}"
        )
    elif jax_mode == "rank0":
        launch_env["MMML_MLPOT_RANK0_BRIDGE"] = "1"
        launch_env["MMML_MPI_PIN_GPU_PER_RANK"] = "0"
    elif jax_mode in {"gpu-per-rank", "spatial"}:
        launch_env["MMML_MPI_PIN_GPU_PER_RANK"] = "1"
    if jax_mode == "spatial":
        launch_env["MMML_MLPOT_SPATIAL_MPI"] = "1"

    repo_root = Path(__file__).resolve().parents[3]
    wrapper_path = wrapper or repo_root / "scripts" / "mmml-charmm-mpirun.sh"
    return LaunchPlan(
        argv=(str(wrapper_path), *command),
        env=launch_env,
        warnings=tuple(warnings),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml mpi-launch",
        description=(
            "Launch CHARMM/OpenMPI with an independent JAX device/thread policy. "
            "When invoked through 'uv run', the active uv interpreter is used on every rank."
        ),
    )
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--jax-mode", choices=JAX_MODES, default="gpu-single")
    parser.add_argument("--jax-cpu-threads", type=int)
    parser.add_argument("--charmm-omp-threads", type=int, default=1)
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--strict-resources", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        plan = build_launch_plan(
            command,
            mpi_ranks=args.mpi_ranks,
            jax_mode=args.jax_mode,
            jax_cpu_threads=args.jax_cpu_threads,
            charmm_omp_threads=args.charmm_omp_threads,
            preset=args.preset,
            strict_resources=args.strict_resources,
        )
    except ValueError as exc:
        build_parser().error(str(exc))

    for warning in plan.warnings:
        print(f"mpi-launch: warning: {warning}", file=sys.stderr)
    if args.dry_run:
        print(plan.shell_command())
        return 0
    return subprocess.run(plan.argv, env=dict(plan.env), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
