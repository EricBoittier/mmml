#!/usr/bin/env python3
"""Compare serial ``python md-system`` vs ``mpirun`` for MPI-linked CHARMM.

Historical docs and DCM-cluster reports say serial ``md-system`` with MPI-linked
``libcharmm.so`` **may** segfault in Fortran ``upinb`` during MLpot registration.
This script measures both paths on **your** cluster so the claim can be verified or
retired. **gpu09 (June 2026):** both paths passed with matching outputs.

**Dry-run (CI / anywhere):**

```bash
python tests/functionality/mlpot/08_serial_vs_mpirun_md_system.py --dry-run
```

**Cluster A/B test (GPU node with PyCHARMM + checkpoint):**

```bash
export MMML_CKPT=/path/to/checkpoint.json
python tests/functionality/mlpot/08_serial_vs_mpirun_md_system.py --run-both \\
  --checkpoint "$MMML_CKPT" \\
  --output-dir artifacts/serial_vs_mpirun_$(date +%Y%m%d_%H%M%S)
```

**Interpretation:**

- Both exit 0 → serial path is OK on this node; mpirun still recommended for production.
- Serial SIGSEGV / exit 139, mpirun OK → use ``mmml-charmm-mpirun.sh`` on this stack.
- Both fail → unrelated setup issue (checkpoint, GPU, etc.).

Re-run after OpenMPI / module / ``libcharmm.so`` changes or on a new node.

Serial run sets ``MMML_NO_MPI_RERUN=1`` so ``md-system`` does **not** auto re-exec under mpirun.
The JSON report records hostname, timestamp, and env snapshot (``MMML_MLPOT_DEVICE``,
``OMP_NUM_THREADS``, ``JAX_PLATFORMS``, ``CUDA_VISIBLE_DEVICES``) plus per-run elapsed time.
"""

from __future__ import annotations

import argparse
import json
import os
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
    "MMML_CHARMM_OMP_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "MMML_MPI_NP",
    "MMML_NO_MPI_RERUN",
    "CHARMM_LIB_DIR",
)


def _environment_snapshot(*, overrides: dict[str, str] | None = None) -> dict[str, str | None]:
    """Capture env vars relevant to serial vs mpirun MLpot comparisons."""
    snap: dict[str, str | None] = {}
    for key in _ENV_KEYS:
        if overrides and key in overrides:
            snap[key] = overrides[key]
        else:
            val = os.environ.get(key)
            snap[key] = val if val is not None else None
    return snap


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config() -> Path:
    return _repo_root() / "mmml/cli/run/md_system.serial_mpi_probe.yaml"


@dataclass
class RunOutcome:
    label: str
    command: list[str]
    exit_code: int
    elapsed_s: float
    signal_name: str | None = None
    environment: dict[str, str | None] | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(),
        help="Probe YAML (default: md_system.serial_mpi_probe.yaml)",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint (or MMML_CKPT)")
    parser.add_argument("--output-dir", default=None, help="Override output_dir in YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only (default when neither --run-* given)",
    )
    parser.add_argument(
        "--run-serial",
        action="store_true",
        help="Run serial python md-system (MMML_NO_MPI_RERUN=1)",
    )
    parser.add_argument(
        "--run-mpirun",
        action="store_true",
        help="Run MMML_MPI_NP=1 mmml-charmm-mpirun.sh md-system",
    )
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Run serial then mpirun and write comparison JSON",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write outcomes JSON (default: <output-dir>/serial_vs_mpirun.json)",
    )
    return parser.parse_args()


def _resolve_checkpoint(raw: str | None, *, required: bool = True) -> str:
    value = raw or os.environ.get("MMML_CKPT") or os.environ.get("MMML_CHECKPOINT")
    if not value:
        if required:
            raise SystemExit("08_serial_vs_mpirun: set --checkpoint or MMML_CKPT")
        return "<MMML_CKPT>"
    path = Path(value).expanduser()
    if not path.exists():
        if required:
            raise SystemExit(f"08_serial_vs_mpirun: checkpoint not found: {path}")
        return str(path)
    return str(path.resolve())


def _md_system_argv(config: Path, checkpoint: str, output_dir: str | None) -> list[str]:
    argv = ["md-system", "--config", str(config), "--checkpoint", checkpoint]
    if output_dir:
        argv.extend(["--output-dir", output_dir])
    return argv


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run_subprocess(
    label: str,
    cmd: list[str],
    env: dict[str, str],
    *,
    env_snapshot: dict[str, str | None],
) -> RunOutcome:
    print(f"\n=== {label} ===", flush=True)
    print(_format_cmd(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(_repo_root()))
    elapsed = time.perf_counter() - t0
    sig = None
    if proc.returncode < 0:
        sig = signal.Signals(-proc.returncode).name
    outcome = RunOutcome(
        label=label,
        command=cmd,
        exit_code=int(proc.returncode),
        elapsed_s=round(elapsed, 3),
        signal_name=sig,
        environment=env_snapshot,
    )
    print(
        f"-> exit={outcome.exit_code}"
        + (f" ({outcome.signal_name})" if outcome.signal_name else "")
        + f" elapsed={outcome.elapsed_s:.1f}s",
        flush=True,
    )
    return outcome


def _dry_run(config: Path, checkpoint: str, output_dir: str | None) -> int:
    root = _repo_root()
    py = sys.executable
    mpi_sh = root / "scripts/mmml-charmm-mpirun.sh"
    md_argv = _md_system_argv(config, checkpoint, output_dir)

    serial_cmd = [py, "-m", "mmml.cli.__main__", *md_argv]
    mpirun_cmd = [str(mpi_sh), "md-system", *md_argv[1:]]

    print("Serial (MMML_NO_MPI_RERUN=1, OMP_NUM_THREADS=1):")
    print("  " + _format_cmd(serial_cmd))
    print("\nMPI np=1 (mmml-charmm-mpirun.sh):")
    print("  MMML_MPI_NP=1 " + _format_cmd(mpirun_cmd))
    print("\nPass: compare exit codes with --run-both on a GPU CHARMM node.")
    return 0


def main() -> int:
    args = _parse_args()
    if not args.config.is_file():
        print(f"FAIL: config not found: {args.config}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    run_any = args.run_serial or args.run_mpirun or args.run_both
    if args.dry_run or not run_any:
        checkpoint = _resolve_checkpoint(args.checkpoint, required=False)
        return _dry_run(args.config, checkpoint, output_dir)

    checkpoint = _resolve_checkpoint(args.checkpoint)

    root = _repo_root()
    py = sys.executable
    mpi_sh = root / "scripts/mmml-charmm-mpirun.sh"
    if not mpi_sh.is_file():
        print(f"FAIL: missing {mpi_sh}", file=sys.stderr)
        return 1

    md_argv = _md_system_argv(args.config, checkpoint, output_dir)
    outcomes: list[RunOutcome] = []

    if args.run_serial or args.run_both:
        env = os.environ.copy()
        env["MMML_NO_MPI_RERUN"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        serial_overrides = {
            "MMML_NO_MPI_RERUN": "1",
            "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
            "MMML_MPI_NP": None,
        }
        outcomes.append(
            _run_subprocess(
                "serial_python",
                [py, "-m", "mmml.cli.__main__", *md_argv],
                env,
                env_snapshot=_environment_snapshot(overrides=serial_overrides),
            )
        )

    if args.run_mpirun or args.run_both:
        env = os.environ.copy()
        env["MMML_MPI_NP"] = "1"
        mpi_overrides = {
            "MMML_MPI_NP": "1",
            "MMML_NO_MPI_RERUN": env.get("MMML_NO_MPI_RERUN"),
        }
        outcomes.append(
            _run_subprocess(
                "mpirun_np1",
                [str(mpi_sh), "md-system", *md_argv[1:]],
                env,
                env_snapshot=_environment_snapshot(overrides=mpi_overrides),
            )
        )

    report_path = args.report_json
    if report_path is None and output_dir:
        report_path = Path(output_dir) / "serial_vs_mpirun.json"
    elif report_path is None:
        report_path = Path("serial_vs_mpirun.json")

    payload = {
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "checkpoint": checkpoint,
        "output_dir": output_dir,
        "environment": _environment_snapshot(),
        "runs": [o.to_dict() for o in outcomes],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {report_path}", flush=True)

    serial = next((o for o in outcomes if o.label == "serial_python"), None)
    mpi = next((o for o in outcomes if o.label == "mpirun_np1"), None)
    if serial and mpi:
        if serial.ok and mpi.ok:
            print("RESULT: both paths succeeded — serial md-system OK on this node.", flush=True)
        elif not serial.ok and mpi.ok:
            print(
                "RESULT: serial failed, mpirun OK — supports using mmml-charmm-mpirun.sh.",
                flush=True,
            )
        elif serial.ok and not mpi.ok:
            print("RESULT: serial OK, mpirun failed — investigate MPI launcher.", flush=True)
        else:
            print("RESULT: both failed — check checkpoint/GPU/CHARMM setup.", flush=True)

    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
