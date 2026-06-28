#!/usr/bin/env python3
"""Gather liquid-density campaign diagnostics: DYNA traces, RDF, VACF, plots.

Examples:
  # Matrix health + CSV (same as status.sh)
  python scripts/collect_diagnostics.py matrix --config config.yaml

  # Full diagnostic bundle for all cells with logs
  python scripts/collect_diagnostics.py collect \\
    --config config.gpu08.local.yaml \\
    --output-dir results/diagnostics_gpu08

  # One cell deep dive
  python scripts/collect_diagnostics.py cell dcm_277_t300_l32 --config config.yaml -v

  # Trajectory RDF/VACF only
  python scripts/collect_diagnostics.py trajectory dcm_277_t300_l32 --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from campaign_lib import cell_from_tag, iter_matrix_cells, load_config, paths_for_run  # noqa: E402
from monitor_lib import (  # noqa: E402
    inspect_run,
    iter_monitors,
    parse_dyna_lines,
    plot_dyna_png,
    print_run_detail,
    print_summary_table,
    read_text_tail,
)
from trajectory_diag import (  # noqa: E402
    analyze_cell_trajectories,
    plot_rdf_png,
    write_trajectory_json,
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
        "dyna_n_frames",
        "temperature_last_K",
        "energy_drift_kcal",
        "campaign_failed_leg",
        "errors",
        "out_dir",
    ]
    return [k for k in preferred if k in sample] + [
        k for k in sample if k not in preferred
    ]


def cmd_matrix(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.tag:
        cell = cell_from_tag(cfg, args.tag)
        monitor = inspect_run(cfg, cell)
        print_run_detail(monitor, cfg=cfg)
        if args.plot_dir:
            _write_dyna_plot(monitor, args.plot_dir)
        return 0

    monitors = list(iter_monitors(cfg))
    monitors = _filter_monitors(monitors, args)
    print_summary_table(monitors, cfg, verbose=args.verbose)

    if args.plot_dir:
        n = 0
        for m in monitors:
            if _write_dyna_plot(m, args.plot_dir):
                n += 1
        print(f"\nWrote {n} DYNA plot(s) under {args.plot_dir.resolve()}")

    if args.json:
        _write_monitor_json(monitors, args.json)

    if monitors:
        rows = [m.to_csv_row() for m in monitors]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = _csv_fieldnames(rows[0])
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv.resolve()}")

    all_m = list(iter_monitors(cfg))
    n_done = sum(1 for m in all_m if m.status == "done")
    n_bad = sum(1 for m in all_m if m.health == "BAD")
    print(f"\n{n_done}/{len(all_m)} complete  |  {n_bad} BAD health")
    return 0


def cmd_cell(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cell = cell_from_tag(cfg, args.tag)
    monitor = inspect_run(cfg, cell)
    print_run_detail(monitor, cfg=cfg)
    traj = analyze_cell_trajectories(Path(monitor.out_dir))
    print("\nTrajectory / handoff:")
    print(json.dumps(traj, indent=2))
    if args.output_dir:
        out = args.output_dir / args.tag
        out.mkdir(parents=True, exist_ok=True)
        write_trajectory_json(traj, out / "trajectory.json")
        _write_dyna_plot(monitor, out)
        rdf = traj.get("trajectory", {}).get("rdf", {})
        if plot_rdf_png(rdf, out / "rdf.png", title=f"{args.tag} g(r)"):
            print(f"Wrote {out / 'rdf.png'}")
        print(f"Wrote bundle under {out}")
    return 0


def cmd_trajectory(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    paths = paths_for_run(cfg, cell_from_tag(cfg, args.tag))
    payload = analyze_cell_trajectories(
        paths["out_dir"],
        stride=args.stride,
        max_frames=args.max_frames,
    )
    print(json.dumps(payload, indent=2))
    if args.output:
        write_trajectory_json(payload, args.output)
        print(f"Wrote {args.output}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    monitors = list(iter_monitors(cfg))
    monitors = _filter_monitors(monitors, args)

    manifest: list[dict] = []
    n_dyna = n_rdf = 0
    for m in monitors:
        if m.log_size_B <= 0 and m.legs_done == 0:
            continue
        cell_dir = out_root / m.run_tag
        cell_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "run_tag": m.run_tag,
            "status": m.status,
            "health": m.health,
            "dyna": m.dyna,
            "active_leg": m.active_leg,
            "progress_note": m.progress_note,
            "errors": m.errors[:5],
        }

        if _write_dyna_plot(m, cell_dir):
            n_dyna += 1
            entry["dyna_plot"] = str(cell_dir / f"{m.run_tag}_dyna.png")

        traj = analyze_cell_trajectories(Path(m.out_dir), stride=args.stride, max_frames=args.max_frames)
        write_trajectory_json(traj, cell_dir / "trajectory.json")
        entry["trajectory"] = traj.get("trajectory", {})
        rdf = entry["trajectory"].get("rdf", {})
        if plot_rdf_png(rdf, cell_dir / "rdf.png", title=f"{m.run_tag} g(r)"):
            n_rdf += 1
            entry["rdf_plot"] = str(cell_dir / "rdf.png")

        manifest.append(entry)

    summary = {
        "config": str(args.config.resolve()),
        "cells_collected": len(manifest),
        "dyna_plots": n_dyna,
        "rdf_plots": n_rdf,
        "cells": manifest,
    }
    (out_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Collected {len(manifest)} cell(s) → {out_root}")
    print(f"  DYNA plots: {n_dyna}  RDF plots: {n_rdf}")
    print(f"  Manifest: {out_root / 'manifest.json'}")
    return 0


def _filter_monitors(monitors, args) -> list:
    if getattr(args, "failed", False):
        return [m for m in monitors if m.status == "failed" or m.health == "BAD"]
    if getattr(args, "running", False):
        return [m for m in monitors if m.status in {"running", "partial", "started"}]
    return monitors


def _write_dyna_plot(monitor, plot_dir: Path) -> bool:
    log_text = read_text_tail(Path(monitor.out_dir) / "stdout.log")
    rows = parse_dyna_lines(log_text)
    return plot_dyna_png(
        rows,
        plot_dir / f"{monitor.run_tag}_dyna.png",
        title=f"{monitor.run_tag}",
        target_temp_K=monitor.temperature_target_K,
    )


def _write_monitor_json(monitors, path: Path) -> None:
    payload = [{**m.__dict__, "dyna": m.dyna, "restarts": m.restarts} for m in monitors]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=_SCRIPTS.parent / "config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("matrix", help="Matrix health table (DYNA from stdout.log)")
    m.add_argument("--tag", help="Single-cell detail mode")
    m.add_argument("-v", "--verbose", action="store_true")
    m.add_argument("--failed", action="store_true")
    m.add_argument("--running", action="store_true")
    m.add_argument("--csv", type=Path, default=_SCRIPTS.parent / "results" / "status.csv")
    m.add_argument("--json", type=Path)
    m.add_argument("--plot-dir", type=Path)
    m.set_defaults(func=cmd_matrix)

    c = sub.add_parser("cell", help="Deep dive one cell")
    c.add_argument("tag")
    c.add_argument("-v", "--verbose", action="store_true")
    c.add_argument("--output-dir", type=Path)
    c.set_defaults(func=cmd_cell)

    t = sub.add_parser("trajectory", help="RDF/VACF from DCD + handoff NPZ")
    t.add_argument("tag")
    t.add_argument("--stride", type=int, default=5)
    t.add_argument("--max-frames", type=int, default=100)
    t.add_argument("--output", type=Path)
    t.set_defaults(func=cmd_trajectory)

    col = sub.add_parser("collect", help="Bundle diagnostics for all active cells")
    col.add_argument("--output-dir", type=Path, default=_SCRIPTS.parent / "results" / "diagnostics")
    col.add_argument("--failed", action="store_true")
    col.add_argument("--running", action="store_true")
    col.add_argument("--stride", type=int, default=5)
    col.add_argument("--max-frames", type=int, default=100)
    col.set_defaults(func=cmd_collect)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
