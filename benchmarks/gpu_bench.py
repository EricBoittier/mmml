#!/usr/bin/env python3
"""Run the mmml asv suite on a GPU: correctness first, then timings + HTML.

Examples::

    uv run python benchmarks/gpu_bench.py
    uv run python benchmarks/gpu_bench.py --group md
    uv run python benchmarks/gpu_bench.py --group throughput
    uv run python benchmarks/gpu_bench.py --bench bench_ml_physnet
    uv run python benchmarks/gpu_bench.py --check neighbors --checks-only
    uv run python benchmarks/gpu_bench.py --list-groups
    uv run python benchmarks/gpu_bench.py --list-checks
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

from benchmarks.gpu_bench_lib import (  # noqa: E402
    BENCH_GROUPS,
    correctness_check_catalog,
    resolve_bench_regex,
    run_gpu_benchmark,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU asv runner with a correctness gate and HTML report",
    )
    parser.add_argument(
        "--bench",
        default=None,
        help="asv --bench regex; also selects matching correctness probes",
    )
    parser.add_argument(
        "--group",
        dest="groups",
        action="append",
        default=None,
        metavar="NAME",
        help="Named module subset (repeatable or comma-separated). See --list-groups.",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="Print named bench groups and their asv regexes, then exit",
    )
    parser.add_argument(
        "--check",
        dest="checks",
        action="append",
        default=None,
        metavar="NAME",
        help="Run only this probe (repeatable). See --list-checks.",
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="Print probe names and which --bench fragments select them, then exit",
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
    args = _parse_args(argv)
    if getattr(args, "list_groups", False):
        width = max(len(name) for name in BENCH_GROUPS)
        for name, regex in BENCH_GROUPS.items():
            print(f"{name:<{width}}  {regex}")
        return 0
    if getattr(args, "list_checks", False):
        print("jax_gpu  (always, unless --check omits it)")
        for spec in correctness_check_catalog():
            tags = ", ".join(sorted(spec.tags))
            print(f"{spec.name}  [{tags}]")
        return 0
    try:
        args.bench = resolve_bench_regex(
            groups=getattr(args, "groups", None),
            bench=getattr(args, "bench", None),
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return run_gpu_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
