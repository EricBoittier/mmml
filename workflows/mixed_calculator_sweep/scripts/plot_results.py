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


def _outlier_settings(rows: list[dict[str, str]], results_dir: Path, threshold_ev: float = 1000.0) -> set[str]:
    """Settings whose |E - E[0]| ever exceeds threshold -- plotted on their own axis."""
    outliers = set()
    for row in rows:
        if row["completed"] != "True":
            continue
        traj_path = results_dir / row["setting"] / f"seed_{row['seed']}" / "trajectory.npz"
        if not traj_path.is_file():
            continue
        energies = np.asarray(np.load(traj_path)["energies"], dtype=float)
        if energies.size and np.max(np.abs(energies - energies[0])) > threshold_ev:
            outliers.add(row["setting"])
    return outliers


def plot_energy_traces(rows: list[dict[str, str]], results_dir: Path, out: Path) -> Path:
    """ΔE(frame) = E(frame) - E(frame=0) per (setting, seed).

    Plotting relative to each trace's own starting energy puts water_box
    (~-75 eV) and peptide_water (~-500 to -560 eV, or the mixed_core_vdw
    outlier's ~1.7M eV -- see README.md/docs §11) on one comparable axis.
    Settings whose energy swings by >1000 eV (i.e. mixed_core_vdw's
    documented initial-configuration blow-up) get their own panel so they
    don't squash the rest of the (well-behaved) traces to a flat line.
    """
    settings = sorted({row["setting"] for row in rows})
    color_of = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(settings)}
    outliers = _outlier_settings(rows, results_dir)

    fig, (ax_main, ax_outlier) = plt.subplots(
        2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    plotted_main = plotted_outlier = False
    for row in rows:
        if row["completed"] != "True":
            continue
        traj_path = results_dir / row["setting"] / f"seed_{row['seed']}" / "trajectory.npz"
        if not traj_path.is_file():
            continue
        energies = np.asarray(np.load(traj_path)["energies"], dtype=float)
        frames = np.arange(len(energies))
        delta = energies - energies[0]
        label = f"{row['setting']} (seed {row['seed']})"
        ax = ax_outlier if row["setting"] in outliers else ax_main
        ax.plot(
            frames, delta, marker="o", markersize=3, linewidth=1.2,
            color=color_of[row["setting"]], alpha=0.85, label=label,
        )
        if ax is ax_main:
            plotted_main = True
        else:
            plotted_outlier = True

    if not (plotted_main or plotted_outlier):
        ax_main.text(0.5, 0.5, "no trajectory.npz found", ha="center", va="center",
                      transform=ax_main.transAxes)

    ax_main.set_ylabel(r"$E(t) - E(0)$ (eV)")
    ax_main.set_title("Energy traces relative to each run's initial frame")
    ax_main.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
    ax_main.grid(alpha=0.3)

    ax_outlier.set_xlabel("recorded frame (every record_every steps)")
    ax_outlier.set_ylabel(r"$E(t) - E(0)$ (eV)")
    if outliers:
        ax_outlier.set_title(
            f"Large-swing settings (own axis): {', '.join(sorted(outliers))}"
        )
        ax_outlier.legend(fontsize=7, loc="best", framealpha=0.9)
    else:
        ax_outlier.set_title("No settings exceeded the 1000 eV outlier threshold")
    ax_outlier.grid(alpha=0.3)

    fig.tight_layout()
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
