#!/usr/bin/env python3
"""Tier-2 spatial MPI CPU profile sweep (DCM:100, up to 8 threads).

Exercises ``md-system --ml-spatial-mpi`` on CPU with:

- ``--mlpot-profile`` / ``MMML_MLPOT_PROFILE`` (ML vs CHARMM callback timers)
- ``MMML_JAX_COMPILE_TIMERS`` (JAX/XLA compile wall time)
- optional ``cProfile`` per rank-0 Python process

**Dry-run (CI / laptop):**

```bash
python tests/functionality/mlpot/10_spatial_mpi_cpu_profile.py --dry-run
```

**Single np run (cluster CPU node):**

```bash
export MMML_CKPT=$PWD/examples/ckpts_json/DESdimers_params.json
export MMML_MLPOT_DEVICE=cpu JAX_PLATFORMS=cpu
python tests/functionality/mlpot/10_spatial_mpi_cpu_profile.py --run \\
  --np 4 --omp 2 --checkpoint "$MMML_CKPT"
```

**Full sweep (1/2/4/8 ranks on 8 CPUs):**

```bash
export MMML_CKPT=$PWD/examples/ckpts_json/DESdimers_params.json
python tests/functionality/mlpot/10_spatial_mpi_cpu_profile.py --sweep \\
  --checkpoint "$MMML_CKPT" --cpus 8 --output-dir artifacts/spatial_mpi_cpu_profile
```

Or use the shell wrapper (writes CSV + cProfile summaries):

```bash
bash workflows/pbc_liquid_density_dyn/scripts/mpi_spatial_cpu_np_sweep.sh
```

**Pass criteria:**

1. Exit 0 for each requested ``np``; ``stage_summary.json`` under ``--output-dir``
2. At ``np>1``: logs show spatial ML / per-rank callbacks (not rank-0-only bridge)
3. ``MLpot profile:`` line present when ``--mlpot-profile`` enabled
4. Optional: ``np=8`` wall time < ``np=1`` on the ML-heavy segment (profile JSON)
"""

from __future__ import annotations

import argparse
import json
import os
import pstats
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


_ENV_KEYS = (
    "MMML_MLPOT_DEVICE",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "MMML_MPI_NP",
    "MMML_MLPOT_SPATIAL_MPI",
    "MMML_JAX_COMPILE_THREADS",
    "MMML_FORCE_JAX_COMPILE_THREADS",
    "MMML_MLPOT_PROFILE",
    "MMML_JAX_COMPILE_TIMERS",
    "CHARMM_LIB_DIR",
)

_PROFILE_RE = re.compile(
    r"MLpot profile: (?P<calls>\d+) ML callbacks, "
    r"ML=(?P<ml>[0-9.]+)s \((?P<ml_pct>[0-9.]+)%\), "
    r"CHARMM\+overhead=(?P<charmm>[0-9.]+)s"
)
_JAX_TIMER_RE = re.compile(
    r"mmml: JAX compile timers — estimated compile=(?P<compile>[0-9.]+)s, "
    r"run=(?P<run>[0-9.]+)s"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_template() -> Path:
    return (
        _repo_root()
        / "workflows/pbc_liquid_density_dyn/benchmarks/dcm100_spatial_mpi_cpu.yaml.tpl"
    )


@dataclass
class SweepOutcome:
    np: int
    omp: int
    spatial_mpi: bool
    exit_code: int
    elapsed_s: float
    output_dir: str
    stdout_log: str
    cprofile_path: str | None = None
    mlpot_profile: dict[str, float | int] | None = None
    jax_compile: dict[str, float] | None = None
    signal_name: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        default=_default_template(),
        help="YAML template with REPLACE_* tokens",
    )
    parser.add_argument("--checkpoint", default=None, help="PhysNet checkpoint (or MMML_CKPT)")
    parser.add_argument("--output-dir", default=None, help="Run output directory")
    parser.add_argument("--box-size", type=float, default=25.0, help="PBC cube side (Å)")
    parser.add_argument("--mini-nstep", type=int, default=20, help="Mini dynamics steps")
    parser.add_argument("--ml-batch-size", type=int, default=128, help="PhysNet chunk size")
    parser.add_argument("--cpus", type=int, default=8, help="Total CPUs for sweep")
    parser.add_argument(
        "--np-list",
        default="1,2,4,8",
        help="Comma-separated MPI ranks to try (sweep mode)",
    )
    parser.add_argument("--np", type=int, default=None, help="Single-run MPI size")
    parser.add_argument("--omp", type=int, default=None, help="Single-run OMP threads")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only",
    )
    parser.add_argument("--run", action="store_true", help="Run a single np/omp configuration")
    parser.add_argument("--sweep", action="store_true", help="Run np sweep on --cpus")
    parser.add_argument(
        "--cprofile",
        action="store_true",
        default=True,
        help="Wrap md-system with cProfile (default: on)",
    )
    parser.add_argument(
        "--no-cprofile",
        action="store_false",
        dest="cprofile",
        help="Disable cProfile wrapper",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write JSON report (default: <output-dir>/spatial_mpi_cpu_profile.json)",
    )
    return parser.parse_args()


def _resolve_checkpoint(raw: str | None, *, required: bool = True) -> str:
    value = raw or os.environ.get("MMML_CKPT") or os.environ.get("MMML_CHECKPOINT")
    if not value:
        if required:
            raise SystemExit("10_spatial_mpi_cpu_profile: set --checkpoint or MMML_CKPT")
        return "<MMML_CKPT>"
    path = Path(value).expanduser()
    if not path.exists():
        if required:
            raise SystemExit(f"10_spatial_mpi_cpu_profile: checkpoint not found: {path}")
        return str(path)
    return str(path.resolve())


def _environment_snapshot() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in _ENV_KEYS}


def _render_config(
    template: Path,
    *,
    out_dir: Path,
    checkpoint: str,
    box_size: float,
    mini_nstep: int,
    ml_batch_size: int,
) -> Path:
    text = template.read_text(encoding="utf-8")
    rendered = (
        text.replace("REPLACE_OUT", str(out_dir))
        .replace("REPLACE_CKPT", checkpoint)
        .replace("REPLACE_BOX", str(box_size))
        .replace("REPLACE_MINI_NSTEP", str(mini_nstep))
        .replace("REPLACE_ML_BATCH", str(ml_batch_size))
    )
    cfg = out_dir / "md_system.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_text(rendered, encoding="utf-8")
    return cfg


def _parse_log_metrics(log_path: Path) -> tuple[dict | None, dict | None]:
    if not log_path.is_file():
        return None, None
    mlpot = None
    jax_t = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _PROFILE_RE.search(line)
        if m:
            mlpot = {
                "ml_callbacks": int(m.group("calls")),
                "ml_seconds": float(m.group("ml")),
                "ml_pct": float(m.group("ml_pct")),
                "charmm_seconds": float(m.group("charmm")),
            }
        j = _JAX_TIMER_RE.search(line)
        if j:
            jax_t = {
                "compile_seconds": float(j.group("compile")),
                "run_seconds": float(j.group("run")),
            }
    return mlpot, jax_t


def _write_cprofile_summary(prof_path: Path, out_path: Path) -> None:
    if not prof_path.is_file():
        return
    stats = pstats.Stats(str(prof_path))
    stats.sort_stats("cumulative")
    with out_path.open("w", encoding="utf-8") as fh:
        stats.stream = fh
        stats.print_stats(40)


def _cpu_env(*, omp: int, np_: int, jax_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MMML_MLPOT_DEVICE"] = "cpu"
    env["JAX_PLATFORMS"] = "cpu"
    env["OMP_NUM_THREADS"] = str(omp)
    env["MMML_MPI_NP"] = str(np_)
    env["MMML_NO_MPI_RERUN"] = "1"
    env["MMML_MLPOT_PROFILE"] = "1"
    env["MMML_JAX_COMPILE_TIMERS"] = "1"
    env["MMML_NO_JAX_COMPILE_THREADS"] = "0"
    env["MMML_JAX_COMPILE_THREADS"] = str(jax_threads)
    env["MMML_FORCE_JAX_COMPILE_THREADS"] = "1"
    if np_ > 1:
        env["MMML_MLPOT_SPATIAL_MPI"] = "1"
    else:
        env.pop("MMML_MLPOT_SPATIAL_MPI", None)
    return env


def _run_one(
    *,
    template: Path,
    checkpoint: str,
    out_dir: Path,
    box_size: float,
    mini_nstep: int,
    ml_batch_size: int,
    np_: int,
    omp: int,
    cpus: int,
    cprofile: bool,
) -> SweepOutcome:
    tag = f"np{np_}_omp{omp}"
    run_dir = out_dir / tag
    cfg = _render_config(
        template,
        out_dir=run_dir,
        checkpoint=checkpoint,
        box_size=box_size,
        mini_nstep=mini_nstep,
        ml_batch_size=ml_batch_size,
    )
    root = _repo_root()
    mpi_sh = root / "scripts/mmml-charmm-mpirun.sh"
    py = sys.executable
    log_path = run_dir / "stdout.log"
    prof_path = run_dir / "md_system.prof"
    md_tail = [
        "md-system",
        "--config",
        str(cfg),
        "--mlpot-profile",
        "--ml-spatial-mpi",
        "--reuse-packmol-cache",
    ]
    if cprofile:
        cmd = [
            str(mpi_sh),
            "python",
            "-m",
            "cProfile",
            "-o",
            str(prof_path),
            "-m",
            "mmml.cli.__main__",
            *md_tail,
        ]
    else:
        cmd = [str(mpi_sh), *md_tail]

    env = _cpu_env(omp=omp, np_=np_, jax_threads=cpus)
    print(f"\n=== {tag} (spatial={np_ > 1}) ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(root), stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    sig = None
    if proc.returncode < 0:
        sig = signal.Signals(-proc.returncode).name

    if cprofile and prof_path.is_file():
        _write_cprofile_summary(prof_path, run_dir / "cprofile_top.txt")

    mlpot, jax_t = _parse_log_metrics(log_path)
    outcome = SweepOutcome(
        np=np_,
        omp=omp,
        spatial_mpi=np_ > 1,
        exit_code=int(proc.returncode),
        elapsed_s=round(elapsed, 3),
        output_dir=str(run_dir),
        stdout_log=str(log_path),
        cprofile_path=str(prof_path) if prof_path.is_file() else None,
        mlpot_profile=mlpot,
        jax_compile=jax_t,
        signal_name=sig,
    )
    status = "OK" if outcome.ok else "FAIL"
    extra = ""
    if mlpot:
        extra = f" ML={mlpot['ml_seconds']:.1f}s ({mlpot['ml_pct']:.0f}%)"
    print(f"-> {status} exit={outcome.exit_code} elapsed={outcome.elapsed_s:.1f}s{extra}", flush=True)
    return outcome


def _dry_run(args: argparse.Namespace, checkpoint: str) -> int:
    root = _repo_root()
    sweep_sh = root / "workflows/pbc_liquid_density_dyn/scripts/mpi_spatial_cpu_np_sweep.sh"
    print("Template:", args.template)
    print("Checkpoint:", checkpoint)
    print("Box:", args.box_size, "Å  DCM:100  mini_nstep:", args.mini_nstep)
    print("CPUs:", args.cpus, "  np-list:", args.np_list)
    print("\nSingle run example:")
    print(
        f"  MMML_CKPT={checkpoint} MMML_MLPOT_DEVICE=cpu JAX_PLATFORMS=cpu \\\n"
        f"  python {Path(__file__).name} --run --np 4 --omp 2 --checkpoint {checkpoint}"
    )
    print("\nSweep wrapper:")
    print(f"  MMML_CKPT={checkpoint} bash {sweep_sh}")
    return 0


def main() -> int:
    args = _parse_args()
    if not args.template.is_file():
        print(f"FAIL: template not found: {args.template}", file=sys.stderr)
        return 1

    run_any = args.run or args.sweep
    if args.dry_run or not run_any:
        checkpoint = _resolve_checkpoint(args.checkpoint, required=False)
        return _dry_run(args, checkpoint)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    out_root = Path(args.output_dir or "artifacts/spatial_mpi_cpu_profile").expanduser()
    outcomes: list[SweepOutcome] = []

    if args.run:
        np_ = int(args.np or 1)
        omp = int(args.omp or max(1, args.cpus // max(np_, 1)))
        outcomes.append(
            _run_one(
                template=args.template,
                checkpoint=checkpoint,
                out_dir=out_root,
                box_size=args.box_size,
                mini_nstep=args.mini_nstep,
                ml_batch_size=args.ml_batch_size,
                np_=np_,
                omp=omp,
                cpus=args.cpus,
                cprofile=args.cprofile,
            )
        )
    elif args.sweep:
        for raw in args.np_list.split(","):
            np_ = int(raw.strip())
            if np_ > args.cpus:
                print(f"skip np={np_} (> --cpus {args.cpus})", flush=True)
                continue
            omp = max(1, args.cpus // np_)
            outcomes.append(
                _run_one(
                    template=args.template,
                    checkpoint=checkpoint,
                    out_dir=out_root,
                    box_size=args.box_size,
                    mini_nstep=args.mini_nstep,
                    ml_batch_size=args.ml_batch_size,
                    np_=np_,
                    omp=omp,
                    cpus=args.cpus,
                    cprofile=args.cprofile,
                )
            )

    report_path = args.report_json or (out_root / "spatial_mpi_cpu_profile.json")
    payload = {
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "composition": "DCM:100",
        "box_size_A": args.box_size,
        "mini_nstep": args.mini_nstep,
        "ml_batch_size": args.ml_batch_size,
        "cpus_per_task": args.cpus,
        "checkpoint": checkpoint,
        "environment": _environment_snapshot(),
        "runs": [o.to_dict() for o in outcomes],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}", flush=True)

    csv_path = out_root / "results.csv"
    if outcomes:
        lines = ["np,omp,wall_s,exit_code,spatial_mpi,ml_s,ml_pct,charmm_s"]
        for o in outcomes:
            ml = o.mlpot_profile or {}
            lines.append(
                f"{o.np},{o.omp},{o.elapsed_s},{o.exit_code},{int(o.spatial_mpi)},"
                f"{ml.get('ml_seconds', '')},{ml.get('ml_pct', '')},{ml.get('charmm_seconds', '')}"
            )
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {csv_path}", flush=True)

    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
