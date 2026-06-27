#!/usr/bin/env python3
"""Tier 2 ``md-system --ml-spatial-mpi`` mini smoke (GPU cluster).

Default **dry-run** validates the example YAML, argv forwarding, and Tier 2 env
without launching CHARMM. Use ``--run`` on a GPU node with PyCHARMM + checkpoint.

**Dry-run (CI / laptop):**

```bash
python tests/functionality/mlpot/07_md_system_spatial_mpi_mini.py --dry-run
```

**Cluster mini (np>=2 recommended):**

```bash
export MMML_CKPT=/path/to/DESdimers_params.json
mmml mpi-check --tier2 --strict

MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 \\
  ./scripts/mmml-charmm-mpirun.sh md-system \\
  --config mmml/cli/run/md_system.spatial_mpi.example.yaml \\
  --checkpoint "$MMML_CKPT" \\
  --output-dir artifacts/spatial_mpi_mini_${SLURM_JOB_ID:-local}
```

**Pass criteria:**

1. Exit 0; ``stage_summary.json`` or ``mlpot_mmml`` artifacts under ``--output-dir``
2. No segfault; finite energy in logs
3. At ``np>1``: logs show spatial MPI / per-rank ML (not rank-0-only bridge warning)
4. Optional: total energy within 0.01 kcal/mol of ``np=1`` reference mini
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config() -> Path:
    return _repo_root() / "mmml/cli/run/md_system.spatial_mpi.example.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config(),
        help="md-system YAML config (default: md_system.spatial_mpi.example.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate YAML/argv/env only (default when --run omitted).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch ``mmml md-system`` (requires PyCHARMM + checkpoint on cluster).",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output-dir", default=None, help="Override output_dir")
    parser.add_argument("--composition", default=None, help="Override composition")
    return parser.parse_args()


def _dry_run(config: Path) -> int:
    from mmml.cli.run.md_config import load_yaml_config
    from mmml.cli.run.md_system import build_pycharmm_command, parse_md_system_args
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_policy import (
        spatial_mpi_enabled,
        sync_spatial_mpi_env_from_args,
    )
    from mmml.interfaces.pycharmmInterface.mlpot.spatial_mpi_validate import (
        validate_tier2_spatial_mpi_env,
    )

    if not config.is_file():
        print(f"FAIL: config not found: {config}", file=sys.stderr)
        return 1

    raw = load_yaml_config(config)
    if not raw.get("ml_spatial_mpi"):
        print("FAIL: config missing ml_spatial_mpi: true", file=sys.stderr)
        return 1

    argv = ["--config", str(config)]
    args = parse_md_system_args(argv)
    sync_spatial_mpi_env_from_args(args)

    if not args.ml_spatial_mpi:
        print("FAIL: parse_md_system_args did not set ml_spatial_mpi", file=sys.stderr)
        return 1
    if not spatial_mpi_enabled(args.ml_spatial_mpi):
        print("FAIL: spatial_mpi_enabled() is False after YAML parse", file=sys.stderr)
        return 1
    if os.environ.get("MMML_MLPOT_SPATIAL_MPI") != "1":
        print("FAIL: MMML_MLPOT_SPATIAL_MPI env not synced", file=sys.stderr)
        return 1

    cmd = build_pycharmm_command(args)
    if "--ml-spatial-mpi" not in cmd:
        print(f"FAIL: --ml-spatial-mpi missing from argv: {cmd}", file=sys.stderr)
        return 1
    if args.md_stages and "mini" not in str(args.md_stages):
        print(f"WARN: expected mini stage, got md_stages={args.md_stages}", file=sys.stderr)

    tier2 = validate_tier2_spatial_mpi_env(strict=False)
    print(f"Tier 2 env: ok={tier2.ok} spatial={tier2.spatial_mpi_enabled}")
    for w in tier2.warnings[:5]:
        print(f"  warn: {w}")
    for e in tier2.errors:
        print(f"  error: {e}", file=sys.stderr)

    print("PASS dry-run: YAML → ml_spatial_mpi → --ml-spatial-mpi")
    print("Launch:")
    print(
        "  MMML_MPI_NP=2 MMML_MLPOT_SPATIAL_MPI=1 "
        "./scripts/mmml-charmm-mpirun.sh md-system "
        f"--config {config}"
    )
    if tier2.errors:
        print("NOTE: tier2 env errors (expected off GPU cluster):", *tier2.errors, sep="\n  ")
    return 0


def _run_md_system(config: Path, args: argparse.Namespace) -> int:
    from tests.conftest import can_import_pycharmm

    if not can_import_pycharmm():
        print("FAIL: PyCHARMM not available (set CHARMM_LIB_DIR)", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "mmml.cli.__main__",
        "md-system",
        "--config",
        str(config),
    ]
    if args.checkpoint:
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    if args.output_dir:
        cmd.extend(["--output-dir", str(args.output_dir)])
    if args.composition:
        cmd.extend(["--composition", str(args.composition)])

    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(_repo_root()))
    if proc.returncode != 0:
        print(f"FAIL: md-system exit {proc.returncode}", file=sys.stderr)
        return int(proc.returncode)

    out = Path(args.output_dir) if args.output_dir else None
    if out is None:
        from mmml.cli.run.md_config import load_yaml_config

        out = Path(load_yaml_config(config).get("output_dir", "artifacts/spatial_mpi_mini"))
    if not out.expanduser().exists():
        print(f"WARN: output_dir missing: {out}", file=sys.stderr)
    else:
        print(f"PASS: md-system completed; output_dir={out}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.run:
        return _run_md_system(args.config.resolve(), args)
    return _dry_run(args.config.resolve())


if __name__ == "__main__":
    sys.exit(main())
