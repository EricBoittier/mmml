#!/usr/bin/env python3
"""Plot cross-backend comparison bars for the unified backend sweep.

Reads ``results/summary.csv`` (from ``collect_results.py``) and produces
``results/figures/summary_bars.png``: per (backend, seed) bars for energy
drift (endpoint delta -- this sweep only records 2-3 frames, see
docs/md-cg-unification-design.md §11, so it's not a full conservation
trace like ``mixed_calculator_sweep``'s) and wall-clock elapsed time.
Failed settings (e.g. ``jaxmd_npt``'s documented deterministic cluster
failure) are marked with a red X instead of a bar, since they have no
numeric drift to plot.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

_COLORS = {
    "jaxmd_min": "#4e79a7",
    "jaxmd_nve": "#f28e2b",
    "jaxmd_nvt": "#59a14f",
    "jaxmd_npt": "#e15759",
    "rigid_mc": "#b07aa1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_summary(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_summary_bars(rows: list[dict[str, str]], out: Path) -> Path:
    rows = sorted(rows, key=lambda r: (r["backend"], int(r["seed"])))
    labels = [f"{row['backend']}\n(seed {row['seed']})" for row in rows]
    colors = [_COLORS.get(row["backend"], "#999999") for row in rows]

    fig, (ax_drift, ax_time) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, row in enumerate(rows):
        if row["completed"] == "True":
            ax_drift.bar(i, float(row["energy_drift_ev"]), color=colors[i])
        else:
            ax_drift.text(i, 0, "FAILED", rotation=90, ha="center", va="bottom",
                           color="crimson", fontsize=8, fontweight="bold")
    ax_drift.axhline(0, color="black", linewidth=0.8)
    ax_drift.set_ylabel("energy drift (eV)\n[E(last frame) - E(first frame)]")
    ax_drift.set_title("Unified backend sweep — endpoint drift and cost per backend/seed")
    ax_drift.grid(alpha=0.3, axis="y")

    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    ax_time.bar(range(len(rows)), elapsed, color=colors)
    ax_time.set_ylabel("elapsed (s)")
    ax_time.set_xticks(range(len(rows)))
    ax_time.set_xticklabels(labels, rotation=75)
    ax_time.grid(alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=b) for b, c in _COLORS.items()]
    ax_drift.legend(handles=handles, loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(args.results_dir)
    bars_path = plot_summary_bars(rows, out_dir / "summary_bars.png")
    print(f"wrote {bars_path}")


if __name__ == "__main__":
    main()
