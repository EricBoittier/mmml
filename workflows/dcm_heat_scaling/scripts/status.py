#!/usr/bin/env python3
"""Summarize dcm_heat_scaling run progress (no Snakemake required)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from heat_lib import composition_tag, dt_fs_slug, load_config, workflow_root  # noqa: E402


def _classify(out_dir: Path, heat_dcd: Path, done: Path, log: Path) -> str:
    if done.is_file():
        return "done"
    if log.is_file() and log.stat().st_size > 0:
        text = log.read_text(encoding="utf-8", errors="replace").lower()
        if any(
            x in text
            for x in (
                "failed with exit code",
                "traceback (most recent call last)",
                "could not find openmpi",
                "workflowerror",
            )
        ):
            if heat_dcd.is_file() and heat_dcd.stat().st_size > 0:
                return "partial"
            return "failed"
        if heat_dcd.is_file() and heat_dcd.stat().st_size > 0:
            return "running"
        if "running:" in text or "mlpot sd minimize" in text or "heat segment" in text:
            return "running"
        return "started"
    if out_dir.is_dir():
        return "empty"
    return "pending"


def _format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1000:
        return f"{n / 1e3:.1f} kB"
    return f"{n} B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=workflow_root() / "config.yaml",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write summary CSV (default: workflows/dcm_heat_scaling/results/status.csv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = workflow_root().parents[1]
    wf = workflow_root()
    out_root = repo / str(cfg.get("output_root", "artifacts/pycharmm_mlpot"))
    stale_root = wf / str(cfg.get("output_root", "artifacts/pycharmm_mlpot"))

    sizes = [int(n) for n in cfg["cluster_sizes"]]
    repeats = [int(r) for r in cfg.get("repeats", [1])]
    dt_values = [float(x) for x in cfg.get("dt_fs_values", [0.25, 0.125])]
    prefix = str(cfg.get("composition_prefix", "DCM"))

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    for n in sizes:
        tag = composition_tag(n, prefix=prefix)
        for rep in repeats:
            for dt in dt_values:
                slug = dt_fs_slug(dt)
                rel = f"dcm{n}_npt_x64_{rep}/{slug}"
                out_dir = out_root / rel
                heat_dcd = out_dir / f"heat_{tag}.dcd"
                done = out_dir / "done.txt"
                log = out_dir / "stdout.log"

                # Snakemake log path (relative to workflow dir)
                sm_log = wf / ".." / cfg.get("output_root", "artifacts/pycharmm_mlpot") / rel / "stdout.log"
                sm_log = sm_log.resolve()
                if not log.is_file() and sm_log.is_file():
                    log = sm_log

                status = _classify(out_dir, heat_dcd, done, log)
                counts[status] = counts.get(status, 0) + 1

                dcd_size = ""
                if heat_dcd.is_file():
                    dcd_size = _format_bytes(heat_dcd.stat().st_size)

                stale = ""
                stale_dir = stale_root / rel
                if stale_dir.is_dir() and stale_dir != out_dir:
                    stale = str(stale_dir)

                rows.append(
                    {
                        "n": str(n),
                        "repeat": str(rep),
                        "dt_slug": slug,
                        "status": status,
                        "heat_dcd": str(heat_dcd) if heat_dcd.is_file() else "",
                        "dcd_size": dcd_size,
                        "stdout_log": str(log) if log.is_file() else "",
                        "stale_dir": stale,
                    }
                )

    total = len(rows)
    print(f"Artifact root: {out_root}")
    if stale_root.is_dir() and stale_root != out_root:
        print(f"Stale root (old runs): {stale_root}")
    print(f"Jobs: {total}  |  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    print(f"{'N':>3} {'rep':>3} {'dt':<8} {'status':<8} {'dcd_size':>10}  log / notes")
    print("-" * 72)
    for row in rows:
        note = ""
        if row["stale_dir"]:
            note = " [stale dir under workflow/]"
        elif row["stdout_log"]:
            note = row["stdout_log"]
        print(
            f"{row['n']:>3} {row['repeat']:>3} {row['dt_slug']:<8} {row['status']:<8} "
            f"{row['dcd_size']:>10}  {note}"
        )

    csv_path = args.csv or (workflow_root() / "results" / "status.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
