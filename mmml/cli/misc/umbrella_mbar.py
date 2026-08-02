#!/usr/bin/env python3
"""CLI for MBAR post-processing of umbrella sampling runs.

Usage:
    mmml umbrella-mbar --run-dir out/umbrella [--checkpoint PATH] [--temperature-K 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmml.umbrella.config import UmbrellaMbarConfig
from mmml.umbrella.io import SUMMARY_JSON
from mmml.umbrella.mbar import run_umbrella_mbar


def _format_pmf_report(result: dict[str, Any]) -> list[str]:
    """Human-readable MBAR PMF lines (1D list and optional 2D matrix)."""
    lines: list[str] = ["Umbrella MBAR done:"]
    xi0 = list(result.get("xi0") or [])
    yi0 = result.get("yi0")
    pmf = list(result.get("pmf_rel_kcal_mol") or [])
    d_pmf = list(result.get("d_pmf_rel_kcal_mol") or [])
    ndim = int(result.get("ndim") or (2 if yi0 else 1))

    if ndim >= 2 and yi0 is not None and len(yi0) == len(pmf):
        for i, (x, y, f) in enumerate(zip(xi0, yi0, pmf)):
            err = f"  ±{d_pmf[i]:.4f}" if i < len(d_pmf) else ""
            lines.append(f"  ξ₀={x:.4f} Å  η₀={y:.4f} Å   PMF={f:.4f}{err} kcal/mol")
        grid = result.get("pmf_rel_kcal_mol_2d")
        shape = result.get("grid_shape")
        if grid is not None and shape is not None and len(shape) == 2:
            nx, ny = int(shape[0]), int(shape[1])
            # Unique axis ticks in ravel order
            xs = [xi0[i * ny] for i in range(nx)] if nx * ny == len(xi0) else []
            ys = list(yi0[:ny]) if len(yi0) >= ny else []
            lines.append(f"  PMF grid (kcal/mol) shape={nx}×{ny}  rows=ξ₀ cols=η₀:")
            if ys:
                hdr = "           " + " ".join(f"{y:8.3f}" for y in ys)
                lines.append(hdr)
            for ix, row in enumerate(grid):
                label = f"{xs[ix]:8.3f}" if ix < len(xs) else f"{ix:8d}"
                body = " ".join(f"{float(v):8.3f}" for v in row)
                lines.append(f"  {label}  {body}")
    else:
        for i, (x, f) in enumerate(zip(xi0, pmf)):
            err = f"  ±{d_pmf[i]:.4f}" if i < len(d_pmf) else ""
            lines.append(f"  ξ₀={x:.4f} Å   PMF={f:.4f}{err} kcal/mol")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml umbrella-mbar",
        description=(
            "MBAR analysis for a completed umbrella-sample run. "
            "Reads umbrella_snapshots.npz from --run-dir and updates umbrella_summary.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Output directory from mmml umbrella-sample",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint for U_ML re-evaluation (default: from snapshots/summary).",
    )
    parser.add_argument(
        "--temperature-K",
        type=float,
        default=None,
        help="kT for reduced potentials (default: from snapshots/summary).",
    )
    parser.add_argument(
        "--mbar-verbose",
        action="store_true",
        help="Verbose pymbar output",
    )
    parser.add_argument(
        "--ml-batch-size",
        type=int,
        default=32,
        help="Reserved for batched U_ML re-eval (default: 32)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = UmbrellaMbarConfig(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        temperature_K=args.temperature_K,
        mbar_verbose=args.mbar_verbose,
        ml_batch_size=args.ml_batch_size,
    )
    result = run_umbrella_mbar(cfg)
    if "error" in result:
        print(f"MBAR error: {result['error']}")
        return 1
    for line in _format_pmf_report(result):
        print(line)
    summary_path = Path(args.run_dir).expanduser().resolve() / SUMMARY_JSON
    print(f"  summary: {summary_path}")
    # Ensure JSON-serializable echo for scripting
    n_eff = result.get("N_k_effective") or []
    echo: dict[str, Any] = {"N_k_effective": n_eff}
    if n_eff:
        echo["N_k_effective_min"] = int(min(n_eff))
        echo["N_k_effective_median"] = float(sorted(n_eff)[len(n_eff) // 2])
    print(json.dumps(echo, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
