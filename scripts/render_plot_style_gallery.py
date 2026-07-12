#!/usr/bin/env python3
"""Render the same example figure under every registered style variant.

Produces docs/plot-style-gallery-assets/<style>.png for each style so
docs/plot-style-gallery.md can show them side by side -- lets a human pick a
look by eye rather than from a description. Also demonstrates two other
house rules from docs/plotting-style-guide.md: "Semantic color, not palette
index" (bonds/angles/dihedrals each get their own color, not one blue reused
three times) and "Legends live outside the plot" (via legend_outside()).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style, get_plot_style, legend_outside

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"

# Same coordinate-type semantic colors used in the plot_internal fix
# (scripts/plot_trajectory_structure.py) -- bonds/angles/dihedrals each get
# their own meaning, not one color reused three times.
_COORD_COLORS = {"bonds": "#1A5276", "angles": "#B9770E", "dihedrals": "#1E8449"}


def _example_figure(style_name: str) -> plt.Figure:
    rng = np.random.default_rng(0)
    fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(17, 5.5))
    fig.subplots_adjust(wspace=0.45)

    style = get_plot_style(style_name)
    frames = np.arange(100)
    water = -75 + 0.1 * np.sin(frames / 5) + rng.normal(0, 0.03, 100)
    mixed = -520 + 0.5 * np.sin(frames / 8 + 1) + rng.normal(0, 0.15, 100)
    ax_trace.plot(frames, water, color=style.colors["train"], linewidth=2.8, label="water_box (MM only)")
    ax_trace.plot(frames, mixed / 6.9, color=style.colors["valid"], linewidth=2.8,
                  label="peptide_water (mixed, rescaled)")
    ax_trace.set_xlabel("recorded frame")
    ax_trace.set_ylabel(r"$E(t) - E(0)$  (eV)")
    ax_trace.set_title(r"Energy trace: $\Delta E = E(t) - E(0)$")
    # Left column of a 2-column figure -> legend attaches further LEFT (not
    # "right", which would land between the two panels and crowd ax_hist).
    legend_outside(ax_trace, side="left", fontsize=10)

    # Each coordinate TYPE gets its own semantic color -- not one blue reused
    # for bonds/angles/dihedrals (the mistake this gallery/fix addresses).
    bonds = rng.normal(0.96, 0.02, 400)
    angles = rng.normal(104.5, 2.5, 300)
    dihedrals = rng.uniform(0, 360, 500)
    ax_hist.hist(bonds, bins=25, density=True, color=_COORD_COLORS["bonds"], alpha=0.75, label="bonds (Å)")
    ax_hist.set_xlabel(r"bond length (Å)  /  angle, dihedral ($^\circ$, rescaled)")
    ax_hist.set_ylabel("probability density")
    ax_hist.set_title("Each coordinate type: its own color")
    ax2 = ax_hist.twinx()
    ax2.hist(angles, bins=25, density=True, color=_COORD_COLORS["angles"], alpha=0.6, label="angles (°)")
    ax2.hist(dihedrals / 3.4, bins=25, density=True, color=_COORD_COLORS["dihedrals"], alpha=0.5,
             label="dihedrals (°, rescaled)")
    ax2.set_yticks([])
    lines1, labels1 = ax_hist.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Right column -> legend attaches further RIGHT, mirroring the left column.
    legend_outside(ax2, handles=lines1 + lines2, labels=labels1 + labels2, side="right", fontsize=9)

    fig.suptitle(f'Style: "{style_name}"', y=1.03)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [
        "editorial_dejavu_sans",
        "editorial_dejavu_serif",
        "editorial_stix",
        "editorial_cm",
        "icml",
    ]
    for name in variants:
        plt.rcdefaults()
        apply_plot_style(name)
        fig = _example_figure(name)
        out = OUT_DIR / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
