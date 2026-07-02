#!/usr/bin/env python3
"""Plot velocity distributions from CHARMM dynamics restart files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mmml.interfaces.pycharmmInterface.mlpot.restart_velocity_analysis import (
    RestartVelocityReport,
    analyze_restart_velocities,
    collect_numbered_restart_paths,
)
from mmml.utils.rich_report import emit_dashboard, emit_table


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mmml plot-restart-velocities",
        description=(
            "Plot |v| distributions from CHARMM restart files (heat.NNNN.res) "
            "and flag velocity outliers."
        ),
    )
    p.add_argument(
        "directory",
        type=Path,
        help="Directory containing numbered restarts (e.g. artifacts/md_run2)",
    )
    p.add_argument(
        "--stem",
        default="heat",
        help="Restart stem for numbered files (default: heat → heat.0000.res)",
    )
    p.add_argument(
        "--z-threshold",
        type=float,
        default=4.0,
        help="MAD z-score cutoff for per-atom speed outliers (default: 4)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write PNG dashboard (default: <directory>/<stem>_velocity_dashboard.png)",
    )
    p.add_argument(
        "--dt-ps",
        type=float,
        default=0.00025,
        help="Timestep in ps for coord-delta velocity inference (default: 0.00025 = 0.25 fs)",
    )
    p.add_argument(
        "--no-infer-velocities",
        action="store_true",
        help="Do not infer velocities from consecutive restart coordinates",
    )
    p.add_argument(
        "--max-outliers",
        type=int,
        default=20,
        help="Max outlier rows in Rich summary (default: 20)",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _plot_dashboard(
    reports: list[RestartVelocityReport],
    out_path: Path,
    *,
    z_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    valid = [r for r in reports if r.has_velocities]
    if not valid:
        raise RuntimeError("no restart files with readable velocities")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_hist, ax_t, ax_max, ax_scatter = axes.ravel()

    all_speeds: list[float] = []
    for rep in valid:
        vel = rep.vel_akma
        if vel is None:
            continue
        speeds = np.linalg.norm(vel, axis=1)
        all_speeds.extend(speeds.tolist())
        label = rep.path.stem
        ax_hist.hist(
            speeds,
            bins=40,
            alpha=0.35,
            density=True,
            label=label if len(valid) <= 12 else None,
        )

    if all_speeds:
        ax_hist.set_xlabel("|v| (AKMA)")
        ax_hist.set_ylabel("density")
        ax_hist.set_title("Speed distribution (all restarts)")
        if len(valid) <= 12:
            ax_hist.legend(fontsize=7, loc="upper right")

    steps = [r.global_step for r in valid if r.global_step is not None]
    temps = [r.temperature_K for r in valid if r.temperature_K is not None]
    names = [r.path.stem for r in valid if r.global_step is not None]
    if steps and temps:
        ax_t.plot(steps, temps, "o-", markersize=3)
        ax_t.set_xlabel("global step (JHSTRT)")
        ax_t.set_ylabel("T (K, unit mass)")
        ax_t.set_title("Restart kinetic temperature")

    max_speeds = [r.speed_max for r in valid]
    x_idx = np.arange(len(valid))
    ax_max.bar(x_idx, max_speeds, color="steelblue", alpha=0.8)
    ax_max.set_xticks(x_idx[:: max(1, len(valid) // 20)])
    ax_max.set_xticklabels(
        [valid[i].path.stem for i in x_idx[:: max(1, len(valid) // 20)]],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    ax_max.set_ylabel("max |v| (AKMA)")
    ax_max.set_title("Per-restart max speed")

    outlier_counts = [len(r.outliers) for r in valid]
    ax_scatter.scatter(
        [r.speed_mean for r in valid],
        outlier_counts,
        c=[r.speed_p99 for r in valid],
        cmap="viridis",
        s=28,
    )
    ax_scatter.set_xlabel("mean |v| (AKMA)")
    ax_scatter.set_ylabel(f"outliers (z ≥ {z_threshold})")
    ax_scatter.set_title("Outlier count vs mean speed")

    fig.suptitle(f"Restart velocities — {valid[0].path.parent.name}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _emit_summary(
    reports: list[RestartVelocityReport],
    *,
    z_threshold: float,
    max_outliers: int,
    quiet: bool,
) -> int:
    n_missing = sum(1 for r in reports if not r.has_velocities)
    n_inferred = sum(1 for r in reports if r.inferred_from_coords)
    n_coord_bug = sum(1 for r in reports if r.coords_as_velocities)
    all_outliers = [o for r in reports for o in r.outliers]
    all_outliers.sort(key=lambda o: o.z_score, reverse=True)

    overview = {
        "restarts": len(reports),
        "with velocities": sum(1 for r in reports if r.has_velocities),
        "inferred (Δcoords)": n_inferred,
        "missing velocities": n_missing,
        "coords-as-vel bug": n_coord_bug,
        "outliers (z≥" + str(z_threshold) + ")": len(all_outliers),
    }
    if reports:
        last = reports[-1]
        overview["last file"] = last.path.name
        if last.temperature_K is not None:
            overview["last T (K)"] = f"{last.temperature_K:.1f}"
        overview["last max |v|"] = f"{last.speed_max:.2f} AKMA"

    emit_dashboard(
        "Restart velocity audit",
        [("Overview", overview)],
        quiet=quiet,
    )

    if all_outliers:
        emit_table(
            f"Top velocity outliers (z ≥ {z_threshold})",
            [
                (
                    f"{o.restart} atom {o.atom_index}",
                    (
                        f"|v|={o.speed_akma:.2f} AKMA, z={o.z_score:.1f}, "
                        f"v=({o.vx:.2f},{o.vy:.2f},{o.vz:.2f})"
                    ),
                )
                for o in all_outliers[:max_outliers]
            ],
            quiet=quiet,
        )
        if not quiet:
            print(
                "Hint: large |v| outliers often precede geometry blow-ups; "
                "run per-monomer JAX bonded recovery before continuing heat.",
                flush=True,
            )

    return 1 if (all_outliers or n_coord_bug) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = Path(args.directory).expanduser().resolve()
    paths = collect_numbered_restart_paths(directory, stem=str(args.stem))
    if not paths:
        print(f"No numbered {args.stem}.*.res files under {directory}", file=sys.stderr)
        return 1

    reports: list[RestartVelocityReport] = []
    for i, pth in enumerate(paths):
        prev = paths[i - 1] if i > 0 else None
        reports.append(
            analyze_restart_velocities(
                pth,
                z_threshold=float(args.z_threshold),
                prev_path=prev,
                dt_ps=float(args.dt_ps),
                allow_inferred=not bool(args.no_infer_velocities),
            )
        )

    out = args.output
    if out is None:
        out = directory / f"{args.stem}_velocity_dashboard.png"
    try:
        _plot_dashboard(reports, Path(out), z_threshold=float(args.z_threshold))
        if not args.quiet:
            print(f"Wrote {out}", flush=True)
    except Exception as exc:
        print(f"WARN: plot failed ({exc}); continuing with text summary", file=sys.stderr)

    return _emit_summary(
        reports,
        z_threshold=float(args.z_threshold),
        max_outliers=int(args.max_outliers),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    raise SystemExit(main())
