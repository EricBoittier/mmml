#!/usr/bin/env python3
"""Run the mmml asv suite on a GPU: correctness first, then timings + HTML.

Examples::

    uv run python benchmarks/gpu_bench.py
    uv run python benchmarks/gpu_bench.py --bench bench_ml_physnet
    uv run python benchmarks/gpu_bench.py --checks-only
    sbatch benchmarks/slurm_bench_gpu.sh

The job refuses a CPU JAX backend, probes PhysNet / MM / SHAKE on tiny
inputs, and only then calls ``asv run`` + ``asv publish``. Open
``benchmarks/html/gpu-report.html`` in a browser (or ``uv run asv preview``
for the full asv graphs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.gpu_bench_lib import run_gpu_benchmark  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU asv runner with a correctness gate and HTML report",
    )
    parser.add_argument(
        "--bench",
        default=None,
        help="asv --bench regex (module, class, or method)",
    )
    parser.add_argument(
        "--append-samples",
        action="store_true",
        help="Merge new asv samples into this commit's existing result JSON",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Run correctness probes and write the report; do not time",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip the pre-timing probes (not recommended)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Time even if a correctness probe failed",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not refuse a CPU JAX backend (for local smoke only)",
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Skip `asv publish` after the run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write gpu-report.html here (default: benchmarks/html)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_gpu_benchmark(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
