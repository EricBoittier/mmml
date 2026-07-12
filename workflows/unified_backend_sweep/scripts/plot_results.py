#!/usr/bin/env python3
"""Plot cross-backend comparison bars for the unified backend sweep.

Reads ``results/summary.csv`` (from ``collect_results.py``) and produces
``results/figures/summary_bars.png``: per (backend, seed) bars for energy
fluctuation ($\\sigma$) and tendency (linear trend) -- see
``docs/plotting-style-guide.md`` for why these replace a bare endpoint delta
-- plus wall-clock elapsed time. This sweep only records 2-3 frames per
setting (see README's "What backend means here"), so these are a much
coarser sanity check than ``mixed_calculator_sweep``'s 100-sample traces.
Failed settings (e.g. ``jaxmd_npt``'s documented deterministic cluster
failure) are marked "FAILED" instead of a bar.

**Color is semantic, not palette-index**: each backend is colored by *what
kind of physics it represents*, not by arbitrary series order --
``jaxmd_min`` (deterministic minimization) is neutral gray, ``jaxmd_nve``
(energy-conserving reference ensemble) is deep blue, ``jaxmd_nvt``
(thermostatted -- deliberately exchanges heat) is forest green, ``jaxmd_npt``
(documented deterministic failure on this cluster) is brick red, and
``rigid_mc`` (stochastic, non-dynamical sampler) is muted purple. See
``docs/plotting-style-guide.md`` "Semantic color, not palette index".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from mmml.utils.plotting.styles import apply_plot_style, legend_outside

_STYLE_NAME = "icml"  # see docs/plot-style-gallery.md

_BACKEND_COLORS = {
    "jaxmd_min": "#5D6D7E",   # neutral gray -- deterministic minimization
    "jaxmd_nve": "#1A5276",   # deep blue -- energy-conserving reference ensemble
    "jaxmd_nvt": "#1E8449",   # forest green -- thermostatted (deliberately exchanges heat)
    "jaxmd_npt": "#943126",   # brick red -- documented deterministic failure on this cluster
    "rigid_mc": "#6C3483",    # muted purple -- stochastic, non-dynamical sampler
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_summary(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fluctuation_and_trend(row: dict[str, str]) -> tuple[float, float]:
    """Prefer the precomputed columns; fall back to the old endpoint delta
    for summary.csv files predating energy_drift_metrics in run_setting.py."""
    if row.get("energy_fluctuation_std_ev"):
        return float(row["energy_fluctuation_std_ev"]), abs(float(row["energy_trend_ev_per_frame"]))
    drift = abs(float(row["energy_drift_ev"]))
    return drift, drift


def plot_summary_bars(rows: list[dict[str, str]], out: Path) -> Path:
    rows = sorted(rows, key=lambda r: (r["backend"], int(r["seed"])))
    labels = [f"{row['backend']}\n(seed {row['seed']})" for row in rows]
    colors = [_BACKEND_COLORS.get(row["backend"], "#999999") for row in rows]

    fig, (ax_fluct, ax_trend, ax_time) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i, row in enumerate(rows):
        if row["completed"] == "True":
            fluctuation, trend = _fluctuation_and_trend(row)
            ax_fluct.bar(i, fluctuation, color=colors[i], edgecolor="#222222", linewidth=0.8)
            ax_trend.bar(i, trend, color=colors[i], edgecolor="#222222", linewidth=0.8)
        else:
            for axis in (ax_fluct, ax_trend):
                axis.text(i, 0, "FAILED", rotation=90, ha="center", va="bottom",
                           color="#943126", fontsize=11, fontweight="bold")

    ax_fluct.set_ylabel(r"fluctuation: $\sigma$ (eV)")
    ax_fluct.set_title("Unified backend sweep — fluctuation, tendency, and cost per backend/seed")

    ax_trend.set_ylabel(r"tendency: $|{\rm d}E/{\rm d}n|$ (eV/frame)")

    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    ax_time.bar(range(len(rows)), elapsed, color=colors, edgecolor="#222222", linewidth=0.8)
    ax_time.set_ylabel("elapsed (s)")
    ax_time.set_xticks(range(len(rows)))
    ax_time.set_xticklabels(labels, rotation=75)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#222222", label=b) for b, c in _BACKEND_COLORS.items()]
    legend_outside(ax_fluct, handles=handles, fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style(_STYLE_NAME)
    rows = load_summary(args.results_dir)
    bars_path = plot_summary_bars(rows, out_dir / "summary_bars.png")
    print(f"wrote {bars_path}")


if __name__ == "__main__":
    main()
