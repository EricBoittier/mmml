#!/usr/bin/env python3
"""Summarize pbc_solvent_burst job health, DYNA traces, and restart progress."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import cell_from_tag, iter_matrix_cells, load_config  # noqa: E402
from monitor_lib import (  # noqa: E402
    inspect_run,
    iter_monitors,
    parse_dyna_lines,
    plot_dyna_png,
    print_run_detail,
    print_summary_table,
    read_text_tail,
)


def _csv_fieldnames(sample: dict[str, str]) -> list[str]:
    preferred = [
        "run_tag",
        "status",
        "health",
        "solvent",
        "n_monomers",
        "temperature_target_K",
        "box_size_A",
        "active_leg",
        "legs_done",
        "legs_total",
        "progress_note",
        "log_stage",
        "log_size_B",
        "log_mtime",
        "dyna_n_frames",
        "temperature_last_K",
        "temperature_mean_K",
        "energy_drift_kcal",
        "campaign_failed_leg",
        "notes",
        "errors",
        "last_dyna_lines",
        "out_dir",
    ]
    return [k for k in preferred if k in sample] + [
        k for k in sample if k not in preferred
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bash scripts/status.sh
  bash scripts/status.sh --verbose
  bash scripts/status.sh --tag dcm_154_t150_l32
  bash scripts/status.sh --plot-dir results/plots
  bash scripts/status.sh --failed --verbose
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "status.csv",
    )
    parser.add_argument(
        "--tag",
        help="Show detailed monitor report for one run tag (e.g. dcm_154_t150_l32)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Include last DYNA> lines in the summary table",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help="Only show failed / BAD health runs",
    )
    parser.add_argument(
        "--running",
        action="store_true",
        help="Only show running / partial runs",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Write per-run DYNA energy/temperature PNGs here",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write full monitor JSON for all (or filtered) runs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.tag:
        cell = cell_from_tag(cfg, args.tag)
        monitor = inspect_run(cfg, cell)
        print_run_detail(monitor, cfg=cfg)
        if args.plot_dir:
            log_text = read_text_tail(Path(monitor.out_dir) / "stdout.log")
            rows = parse_dyna_lines(log_text)
            out_png = args.plot_dir / f"{monitor.run_tag}_dyna.png"
            if plot_dyna_png(
                rows,
                out_png,
                title=f"{monitor.run_tag} (stdout DYNA>)",
                target_temp_K=monitor.temperature_target_K,
            ):
                print(f"\nWrote plot: {out_png}")
            else:
                print("\nNo plot (need ≥2 DYNA> lines and matplotlib)", flush=True)
        return 0

    monitors = list(iter_monitors(cfg))
    if args.failed:
        monitors = [m for m in monitors if m.status == "failed" or m.health == "BAD"]
    elif args.running:
        monitors = [m for m in monitors if m.status in {"running", "partial", "started"}]

    print_summary_table(monitors, cfg, verbose=args.verbose)

    if args.plot_dir:
        n_plots = 0
        for m in monitors:
            log_text = read_text_tail(Path(m.out_dir) / "stdout.log")
            rows = parse_dyna_lines(log_text)
            if plot_dyna_png(
                rows,
                args.plot_dir / f"{m.run_tag}_dyna.png",
                title=f"{m.run_tag}",
                target_temp_K=m.temperature_target_K,
            ):
                n_plots += 1
        print(f"\nWrote {n_plots} DYNA plot(s) under {args.plot_dir.resolve()}")

    if args.json:
        payload = [
            {
                **m.__dict__,
                "dyna": m.dyna,
                "restarts": m.restarts,
            }
            for m in monitors
        ]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON: {args.json.resolve()}")

    if monitors:
        rows = [m.to_csv_row() for m in monitors]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = _csv_fieldnames(rows[0])
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv.resolve()}")

    all_monitors = list(iter_monitors(cfg))
    n_done = sum(1 for m in all_monitors if m.status == "done")
    n_bad = sum(1 for m in all_monitors if m.health == "BAD")
    shown = len(monitors)
    if args.failed or args.running:
        print(
            f"\nShowing {shown} run(s)  |  overall: {n_done}/{len(all_monitors)} complete, "
            f"{n_bad} BAD health",
            flush=True,
        )
    else:
        print(f"\n{n_done}/{len(all_monitors)} complete  |  {n_bad} BAD health", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
