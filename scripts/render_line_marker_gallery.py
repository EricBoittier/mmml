#!/usr/bin/env python3
"""Line style, scatter marker, and symbol conventions -- a second axis of
encoding alongside color, for when color alone can't (or shouldn't) carry
every distinction: grayscale printing, colorblind-safety redundancy, or a
figure that already spends color on something else (e.g. a colormap) and
needs line/marker identity for a categorical split on top of it.

See docs/plotting-style-guide.md "Line styles, markers, and symbols".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import (
    LINE_STYLE_CYCLE,
    MARKER_CYCLE,
    apply_plot_style,
    comparison_colors,
    legend_outside,
)

STYLE_NAME = "icml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(7)


def line_style_by_role(out: Path) -> None:
    """Line style encodes a SECOND categorical axis on top of color: here,
    color = force field (a genuinely different series identity), line style
    = replicate run (the same underlying thing, repeated) -- so replicates
    of the same force field visually group by color while still being
    individually traceable by dash pattern, and the distinction survives
    grayscale printing or a colorblind viewer even without the color axis.
    """
    force_fields = ["MM/ff14SB", "ML potential"]
    colors = comparison_colors(STYLE_NAME, n=len(force_fields))
    n_replicates = len(LINE_STYLE_CYCLE)
    t = np.linspace(0, 10, 200)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ff, color in zip(force_fields, colors):
        base = -74.0 if ff.startswith("MM") else -73.6
        for rep in range(n_replicates):
            trace = base + 0.4 * np.sin(t / 2 + rep) + RNG.normal(0, 0.05, t.size)
            ax.plot(t, trace, color=color, linestyle=LINE_STYLE_CYCLE[rep],
                    linewidth=2.0, label=f"{ff}, replicate {rep + 1}" if rep < 2 else None)
    # Two-part legend: color (force field) and line style (replicate), each its own mini-legend.
    color_handles = [plt.Line2D([0], [0], color=c, linewidth=2.5, label=ff)
                      for ff, c in zip(force_fields, colors)]
    style_handles = [plt.Line2D([0], [0], color="#333333", linestyle=ls, linewidth=2.0,
                                  label=f"replicate {i + 1}")
                      for i, ls in enumerate(LINE_STYLE_CYCLE)]
    leg1 = ax.legend(handles=color_handles, title="force field", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="replicate", loc="lower left",
              bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Line style = replicate, color = force field (two independent axes)")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def marker_style_by_role(out: Path) -> None:
    """Same idea for scatter: color = system, marker shape = sampling
    method. A reader can filter on either axis by eye ("show me all circles"
    vs. "show me all blue points") without a third color needed.
    """
    systems = ["water_box", "peptide_water", "vacuum"]
    methods = ["MD snapshot", "umbrella sampling", "metadynamics"]
    colors = comparison_colors(STYLE_NAME, n=len(systems))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (system, color) in enumerate(zip(systems, colors)):
        for j, method in enumerate(methods):
            n = 25
            x = RNG.normal(i * 2.2, 0.3, n)
            y = RNG.normal(-70 - i * 2 + j * 0.6, 0.4, n)
            ax.scatter(x, y, marker=MARKER_CYCLE[j], color=color, s=60,
                       edgecolor="#222222", linewidth=0.6, alpha=0.85)

    color_handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color=c,
                                  markersize=9, label=s) for s, c in zip(systems, colors)]
    marker_handles = [plt.Line2D([0], [0], marker=MARKER_CYCLE[j], linestyle="none",
                                   color="#333333", markersize=9, label=m)
                       for j, m in enumerate(methods)]
    leg1 = ax.legend(handles=color_handles, title="system", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="sampling method", loc="lower left",
              bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    ax.set_xlabel("collective variable")
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Marker shape = method, color = system (two independent axes)")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def linewidth_and_alpha_hierarchy(out: Path) -> None:
    """A third, non-categorical encoding: linewidth/alpha as a visual
    hierarchy -- the "headline" series drawn bold and opaque, supporting
    context drawn thin and faint, so the eye finds the point without a
    legend lookup at all.
    """
    t = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(8):
        trace = -74 + RNG.normal(0, 0.3) + 0.2 * np.sin(t / 3 + i)
        ax.plot(t, trace, color="#999999", linewidth=1.0, alpha=0.35)
    headline = -74 + 0.5 * np.sin(t / 3)
    ax.plot(t, headline, color="#C44E52", linewidth=3.2, alpha=1.0, label="this run")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Linewidth/alpha hierarchy: one bold headline, seven faint context traces")
    legend_outside(ax)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    renders = {
        "chart_line_style_roles": line_style_by_role,
        "chart_marker_style_roles": marker_style_by_role,
        "chart_linewidth_hierarchy": linewidth_and_alpha_hierarchy,
    }
    for name, fn in renders.items():
        out = OUT_DIR / f"{name}.png"
        fn(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
