#!/usr/bin/env python3
"""Plot energy traces and cross-setting summaries for the mixed sweep.

Reads ``results/summary.csv`` (from ``collect_results.py``) and each
setting's ``trajectory.npz`` (written by ``JaxmdDriver`` since
``run_setting.py`` passes ``output_dir``) to produce:

- ``results/figures/energy_traces.png`` — energy vs. recorded frame, one line
  per (setting, seed), so a conservation problem (e.g. the `mixed_core_vdw`
  initial blow-up documented in README.md) is visible at a glance.
- ``results/figures/summary_bars.png`` — per-setting bars for energy std
  (conservation quality) and wall-clock elapsed time (cost), split by
  system (water_box vs peptide_water) since they're on very different
  scales.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#2ca02c", "#d62728",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_summary(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_energy_traces(rows: list[dict[str, str]], results_dir: Path, out: Path) -> Path:
    settings = sorted({row["setting"] for row in rows})
    color_of = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(settings)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted_any = False
    for row in rows:
        if row["completed"] != "True":
            continue
        traj_path = results_dir / row["setting"] / f"seed_{row['seed']}" / "trajectory.npz"
        if not traj_path.is_file():
            continue
        data = np.load(traj_path)
        energies = np.asarray(data["energies"], dtype=float)
        frames = np.arange(len(energies))
        label = f"{row['setting']} (seed {row['seed']})"
        ax.plot(
            frames, energies, marker="o", markersize=3, linewidth=1.2,
            color=color_of[row["setting"]], alpha=0.85, label=label,
        )
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "no trajectory.npz found", ha="center", va="center",
                 transform=ax.transAxes)

    ax.set_xlabel("recorded frame (every record_every steps)")
    ax.set_ylabel("energy (eV)")
    ax.set_title("Mixed-system / calculator sweep — energy traces")
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary_bars(rows: list[dict[str, str]], out: Path) -> Path:
    completed = [row for row in rows if row["completed"] == "True"]
    completed.sort(key=lambda r: (r["system"], r["setting"], r["seed"]))
    labels = [f"{row['setting']}\n(seed {row['seed']})" for row in completed]
    stds = [float(row["energy_std_ev"]) for row in completed]
    elapsed = [float(row["elapsed_seconds"]) for row in completed]
    colors = ["#4e79a7" if row["system"] == "water_box" else "#e15759" for row in completed]

    fig, (ax_std, ax_time) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_std.bar(labels, stds, color=colors)
    ax_std.set_ylabel("energy std (eV)")
    ax_std.set_title("Conservation quality (lower = tighter) and cost per setting")
    ax_std.set_yscale("log")
    ax_std.grid(alpha=0.3, axis="y")

    ax_time.bar(labels, elapsed, color=colors)
    ax_time.set_ylabel("elapsed (s)")
    ax_time.tick_params(axis="x", rotation=75)
    ax_time.grid(alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    handles = [
        Patch(color="#4e79a7", label="water_box"),
        Patch(color="#e15759", label="peptide_water (mixed)"),
    ]
    ax_std.legend(handles=handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(args.results_dir)
    traces_path = plot_energy_traces(rows, args.results_dir, out_dir / "energy_traces.png")
    bars_path = plot_summary_bars(rows, out_dir / "summary_bars.png")
    print(f"wrote {traces_path}")
    print(f"wrote {bars_path}")


if __name__ == "__main__":
    main()
