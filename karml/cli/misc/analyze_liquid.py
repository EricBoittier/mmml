"""Analyze neat-liquid jaxmd / campaign trajectories (density, RDF, MSD, plots).

Examples:
  mmml analyze-liquid --campaign-dir artifacts/lj_scales/liquid_dcm -o analysis/
  mmml analyze-liquid --h5 path/to/pbc_nvt_jaxmd_nvt.h5 --box-size 30 --solvent DCM -o out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmml.analysis.liquid_md import (
    analyze_campaign_dir,
    analyze_h5,
    write_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmml analyze-liquid",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--campaign-dir",
        type=Path,
        help="Campaign output root (searches for jaxmd *.h5, prefers jaxmd_nvt)",
    )
    src.add_argument(
        "--h5",
        type=Path,
        help="Single jaxmd HDF5 trajectory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics.json and PNG plots",
    )
    parser.add_argument("--box-size", type=float, default=None, help="Cubic box side (Å)")
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help="Neat solvent residue (DCM, ACO, …) for MW / atoms-per-monomer",
    )
    parser.add_argument(
        "--prefer-run",
        type=str,
        default="jaxmd_npt",
        help="Campaign run id substring preferred when choosing an HDF5",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=400,
        help="Analyze at most this many frames (tail of trajectory)",
    )
    parser.add_argument("--r-max", type=float, default=12.0, help="RDF cutoff (Å)")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write metrics.json only (skip matplotlib PNGs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.campaign_dir is not None:
        report = analyze_campaign_dir(
            args.campaign_dir,
            box_side_A=args.box_size,
            solvent=args.solvent,
            prefer_run=args.prefer_run,
            stride=args.stride,
            max_frames=args.max_frames,
            r_max=args.r_max,
        )
    else:
        if not args.h5.is_file():
            print(f"ERROR: HDF5 not found: {args.h5}", file=sys.stderr)
            return 2
        report = analyze_h5(
            args.h5,
            box_side_A=args.box_size,
            solvent=args.solvent,
            stride=args.stride,
            max_frames=args.max_frames,
            r_max=args.r_max,
        )

    if report.get("error"):
        print(f"ERROR: {report['error']}", file=sys.stderr)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    if args.no_plots:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Compact summary without full histograms.
        summary = dict(report)
        rdf = dict(summary.get("rdf") or {})
        pairs = rdf.get("pairs") or {}
        rdf["pairs"] = {
            label: {
                "peak_r_A": rec.get("peak_r_A"),
                "peak_g": rec.get("peak_g"),
            }
            for label, rec in pairs.items()
        }
        summary["rdf"] = rdf
        if isinstance(summary.get("msd"), dict):
            summary["msd"] = {
                k: v
                for k, v in summary["msd"].items()
                if k not in {"time_ps", "msd_A2"}
            }
        out = args.output_dir / "metrics.json"
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        artifacts = {"metrics": str(out)}
    else:
        artifacts = write_analysis_outputs(report, args.output_dir)

    dens = report.get("density") or {}
    peaks = ((report.get("rdf") or {}).get("top_peaks")) or []
    print(f"analyze-liquid: wrote {artifacts.get('metrics')}")
    if dens.get("density_g_cm3") is not None:
        ref = dens.get("reference_g_cm3")
        rel = dens.get("relative_error")
        msg = f"  ρ = {dens['density_g_cm3']:.4f} g/cm³"
        if ref is not None and rel is not None:
            msg += f"  (ref {ref:.3f}, Δ={100.0 * rel:+.1f}%)"
        print(msg)
        if dens.get("note"):
            print(f"  note: {dens['note']}")
    if peaks:
        top = peaks[0]
        print(
            f"  top RDF peak: {top['pair']}  r={top['peak_r_A']:.2f} Å  "
            f"g={top['peak_g']:.2f}"
        )
    for key in ("timeseries_png", "rdf_png", "msd_png"):
        if key in artifacts:
            print(f"  {key}: {artifacts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
