#!/usr/bin/env python3
"""Benchmark PyCHARMM vs JAX-MM energies for supported CGENFF systems.

Examples (CHARMM node)::

    export CHARMM_HOME=... CHARMM_LIB_DIR=... LD_LIBRARY_PATH=...
    JAX_PLATFORMS=cpu uv run python scripts/benchmark_charmm_jax_energy.py

    JAX_PLATFORMS=cpu uv run python scripts/benchmark_charmm_jax_energy.py \\
      --cases tip3_monomer trialanine_water -o /tmp/charmm_jax_bench

See ``tests/functionality/charmm/README_charmm_jax_benchmark.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CHARMM vs JAX-MM energy benchmark")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("tip3_monomer", "tip3_water_box", "trialanine_water", "all"),
        default=["all"],
        help="Systems to benchmark (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("artifacts/charmm_jax_benchmark"),
        help="Write benchmark.json and benchmark.md here",
    )
    parser.add_argument("--seed", type=int, default=11, help="Position perturbation seed")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="CHARMM scratch dir for box builds (default: output-dir/charmm_work)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    from mmml.interfaces.pycharmmInterface.charmm_jax_energy_benchmark import (
        all_layers_passed,
        render_json_report,
        render_markdown_report,
        run_tip3_monomer_benchmark,
        run_tip3_water_box_benchmark,
        run_trialanine_water_benchmark,
    )
    from mmml.interfaces.pycharmmInterface.import_pycharmm import ensure_pycharmm_loaded
    from mmml.interfaces.pycharmmInterface.trialanine_water_box import (
        build_trialanine_water_box_in_charmm,
        have_trialanine_cgenff,
    )

    ensure_pycharmm_loaded()

    selected = set(args.cases)
    if "all" in selected:
        selected = {"tip3_monomer", "tip3_water_box", "trialanine_water"}

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = (args.workdir or out_dir / "charmm_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    if "tip3_monomer" in selected:
        print("Benchmarking tip3_monomer (bonded)...", flush=True)
        results.append(run_tip3_monomer_benchmark(seed=args.seed))

    if "tip3_water_box" in selected:
        print("Benchmarking tip3_water_box (bonded + nonbonded + total)...", flush=True)
        results.append(
            run_tip3_water_box_benchmark(
                seed=args.seed,
                workdir=workdir / "tip3_water_box",
            )
        )

    if "trialanine_water" in selected:
        if not have_trialanine_cgenff():
            print("Skipping trialanine_water: bundled TRIA RTF missing", file=sys.stderr)
        else:
            print("Benchmarking trialanine_water...", flush=True)
            box = build_trialanine_water_box_in_charmm(
                n_waters=10,
                box_side_A=28.0,
                seed=args.seed,
                workdir=workdir / "trialanine_water",
                skip_reset_block=True,
            )
            results.append(run_trialanine_water_benchmark(box, seed=args.seed + 12))

    if not results:
        print("No benchmarks ran.", file=sys.stderr)
        return 2

    cases = tuple(results)
    md_path = out_dir / "benchmark.md"
    json_path = out_dir / "benchmark.json"
    md_path.write_text(render_markdown_report(cases), encoding="utf-8")
    json_path.write_text(render_json_report(cases), encoding="utf-8")

    print(render_markdown_report(cases))
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")

    return 0 if all_layers_passed(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
