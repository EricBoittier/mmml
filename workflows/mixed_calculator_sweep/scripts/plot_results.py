#!/usr/bin/env python3
"""Plot energy traces and cross-setting summaries for the mixed sweep.

Reads ``results/summary.csv`` (from ``collect_results.py``) and each
setting's ``trajectory.npz`` (written by ``JaxmdDriver``/``RigidBodySampler``
since ``run_setting.py`` passes ``output_dir``) to produce:

- ``results/figures/energy_traces.png`` — energy vs. recorded frame (relative
  to each run's initial frame) with a shaded ±1σ fluctuation band and a
  fitted linear trend line overlaid, one line per (setting, seed), so a
  conservation problem (e.g. the ``mixed_core_vdw`` initial blow-up
  documented in README.md) is visible at a glance. Settings whose energy
  swings by >1000 eV get their own panel so they don't flatten the
  well-behaved traces to a line.
- ``results/figures/summary_bars.png`` — per-setting bars for energy
  fluctuation ($\\sigma$, the noise floor) and trend (linear slope, the
  systematic tendency) — see ``docs/plotting-style-guide.md`` for why this
  replaces a bare endpoint delta — plus wall-clock elapsed time.

**Color is semantic, not palette-index**: every setting is colored by its
*system* (``water_box`` = deep blue "MM only", ``peptide_water`` = brick red
"mixed ML/MM") using the house "editorial" style family's `train`/`valid` roles, not an
arbitrary per-series hue — see docs/plotting-style-guide.md "Semantic color,
not palette index". Individual settings within a system share that color but
get a distinct marker shape, so identity is still readable without inventing
a new hue per line.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.md.results import energy_drift_metrics
from mmml.utils.plotting.styles import apply_plot_style

_STYLE_NAME = "editorial_dejavu_serif"  # swap freely -- see docs/plot-style-gallery.md

# Semantic system -> color (see module docstring). Not palette-index colors.
_SYSTEM_COLORS = {"water_box": "#1A5276", "peptide_water": "#943126"}
_SYSTEM_LABELS = {"water_box": "water_box (MM only)", "peptide_water": "peptide_water (mixed ML/MM)"}
# Symbols distinguish individual settings within a system's shared color.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]


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
    r"""$\Delta E(t) = E(t) - E(0)$ per (setting, seed), with a $\pm\sigma$ band and trend fit.

    Plotting relative to each trace's own starting energy puts water_box
    (~-75 eV) and peptide_water (~-500 to -560 eV, or the mixed_core_vdw
    outlier's ~1.7M eV -- see README.md/docs §11) on one comparable axis. The
    dashed trend line is the same linear fit `mmml.md.results.energy_drift_metrics`
    reports as `energy_trend_ev_per_frame` -- the systematic tendency, as
    distinct from the shaded $\pm\sigma$ fluctuation band around it.
    """
    settings = sorted({row["setting"] for row in rows})
    marker_of = dict(zip(settings, _MARKERS))
    outliers = _outlier_settings(rows, results_dir)

    fig, (ax_main, ax_outlier) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]}
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
        color = _SYSTEM_COLORS[row["system"]]
        marker = marker_of[row["setting"]]

        metrics = energy_drift_metrics(energies)
        sigma = metrics["energy_fluctuation_std_ev"]
        trend = metrics["energy_trend_ev_per_frame"] * frames
        ax.fill_between(frames, trend - sigma, trend + sigma, color=color, alpha=0.12, linewidth=0)
        ax.plot(frames, delta, marker=marker, markersize=6, markevery=max(1, len(frames) // 20),
                linewidth=2.4, color=color, alpha=0.9, label=label)
        ax.plot(frames, trend, linestyle="--", linewidth=1.8, color=color, alpha=0.6)
        if ax is ax_main:
            plotted_main = True
        else:
            plotted_outlier = True

    if not (plotted_main or plotted_outlier):
        ax_main.text(0.5, 0.5, "no trajectory.npz found", ha="center", va="center",
                      transform=ax_main.transAxes)

    ax_main.set_ylabel(r"$E(t) - E(0)$  (eV)")
    ax_main.set_title(r"Energy traces relative to $E(0)$  (dashed = trend fit, band = $\pm\sigma$)")
    ax_main.legend(fontsize=10, ncol=2, loc="best", framealpha=0.85, edgecolor="#CCCCCC")

    ax_outlier.set_xlabel("recorded frame (every record_every steps)")
    ax_outlier.set_ylabel(r"$E(t) - E(0)$  (eV)")
    if outliers:
        ax_outlier.set_title(
            f"Large-swing settings (own axis): {', '.join(sorted(outliers))}"
        )
        ax_outlier.legend(fontsize=10, loc="best", framealpha=0.85, edgecolor="#CCCCCC")
    else:
        ax_outlier.set_title("No settings exceeded the 1000 eV outlier threshold")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _drift_metrics_for_row(row: dict[str, str], results_dir: Path) -> dict[str, float]:
    """Prefer summary.csv's precomputed fields; fall back to trajectory.npz.

    Older summary.csv files (from before energy_drift_metrics existed in
    run_setting.py) won't have these columns -- recompute from the raw
    trajectory instead of skipping the setting.
    """
    if row.get("energy_fluctuation_std_ev"):
        return {
            "energy_fluctuation_std_ev": float(row["energy_fluctuation_std_ev"]),
            "energy_trend_ev_per_frame": float(row["energy_trend_ev_per_frame"]),
        }
    traj_path = results_dir / row["setting"] / f"seed_{row['seed']}" / "trajectory.npz"
    if traj_path.is_file():
        energies = np.asarray(np.load(traj_path)["energies"], dtype=float)
        return energy_drift_metrics(energies)
    return {"energy_fluctuation_std_ev": float("nan"), "energy_trend_ev_per_frame": float("nan")}


def plot_summary_bars(rows: list[dict[str, str]], results_dir: Path, out: Path) -> Path:
    completed = [row for row in rows if row["completed"] == "True"]
    completed.sort(key=lambda r: (r["system"], r["setting"], r["seed"]))
    labels = [f"{row['setting']}\n(seed {row['seed']})" for row in completed]
    metrics = [_drift_metrics_for_row(row, results_dir) for row in completed]
    fluctuation = [m["energy_fluctuation_std_ev"] for m in metrics]
    trend = [abs(m["energy_trend_ev_per_frame"]) for m in metrics]
    elapsed = [float(row["elapsed_seconds"]) for row in completed]
    colors = [_SYSTEM_COLORS[row["system"]] for row in completed]

    fig, (ax_fluct, ax_trend, ax_time) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax_fluct.bar(labels, fluctuation, color=colors, edgecolor="#222222", linewidth=0.8)
    ax_fluct.set_ylabel(r"fluctuation: $\sigma$ (eV)")
    ax_fluct.set_title("Conservation quality (fluctuation + tendency) and cost per setting")
    ax_fluct.set_yscale("log")

    ax_trend.bar(labels, trend, color=colors, edgecolor="#222222", linewidth=0.8)
    ax_trend.set_ylabel(r"tendency: $|{\rm d}E/{\rm d}n|$ (eV/frame)")
    ax_trend.set_yscale("log")

    ax_time.bar(labels, elapsed, color=colors, edgecolor="#222222", linewidth=0.8)
    ax_time.set_ylabel("elapsed (s)")
    ax_time.tick_params(axis="x", rotation=75)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#222222", label=_SYSTEM_LABELS[s])
               for s, c in _SYSTEM_COLORS.items()]
    ax_fluct.legend(handles=handles, loc="upper left", fontsize=11, framealpha=0.85, edgecolor="#CCCCCC")

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
    traces_path = plot_energy_traces(rows, args.results_dir, out_dir / "energy_traces.png")
    bars_path = plot_summary_bars(rows, args.results_dir, out_dir / "summary_bars.png")
    print(f"wrote {traces_path}")
    print(f"wrote {bars_path}")


if __name__ == "__main__":
    main()
