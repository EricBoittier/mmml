#!/usr/bin/env python3
"""Render "defaults at a glance" swatches: every named house style's
categorical color cycle (comparison_palette), the shared colormap defaults,
the line-style/marker cycles, and the semantic status palette. Meant to sit
at the TOP of docs/plot-style-gallery.md so a reader can see what each
default actually looks like -- across the graph types that would plausibly
use it -- before scrolling through individual chart examples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import (
    DEFAULT_CYCLIC_CMAP,
    DEFAULT_DIVERGING_CMAP,
    DEFAULT_SEQUENTIAL_CMAP,
    LINE_STYLE_CYCLE,
    MARKER_CYCLE,
    PLOT_STYLES,
    STATUS_COLORS,
    STATUS_HATCHES,
    STYLE_DISPLAY_ORDER,
    apply_plot_style,
    default_cmap,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(0)


def style_palette_swatches(out: Path) -> None:
    """One row per style (in STYLE_DISPLAY_ORDER, default first), its name
    plus a strip of comparison_palette swatches."""
    names = list(STYLE_DISPLAY_ORDER)
    max_colors = max(len(PLOT_STYLES[n].comparison_palette or ()) for n in names)

    fig, ax = plt.subplots(figsize=(2 + 1.1 * max_colors, 0.62 * len(names) + 0.6))
    for row, name in enumerate(names):
        y = len(names) - row - 1
        palette = PLOT_STYLES[name].comparison_palette or ()
        label = f"{name}  (default)" if name == "icml" else name
        ax.text(-0.15, y + 0.5, label, ha="right", va="center", fontsize=11,
                 fontweight="bold" if name == "icml" else "normal")
        for i, color in enumerate(palette):
            ax.add_patch(plt.Rectangle((i, y), 0.92, 0.92, facecolor=color, edgecolor="#222222", linewidth=0.8))

    ax.set_xlim(-4.6, max_colors + 0.2)
    ax.set_ylim(0, len(names))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Categorical color cycle per style (comparison_palette)\nin STYLE_DISPLAY_ORDER, default (\"icml\") first",
                 fontsize=13, pad=14)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_colormap_swatches(out: Path) -> None:
    """The three shared house colormap defaults (sequential/diverging/cyclic),
    rendered as gradient bars -- these are global (not per-style), chosen
    once via the `cmap` library, see docs/plotting-style-guide.md "Colormaps".
    """
    kinds = [
        ("sequential", DEFAULT_SEQUENTIAL_CMAP),
        ("diverging", DEFAULT_DIVERGING_CMAP),
        ("cyclic", DEFAULT_CYCLIC_CMAP),
    ]
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    fig, axes = plt.subplots(len(kinds), 1, figsize=(7, 0.9 * len(kinds) + 0.4))
    for ax, (kind, name) in zip(axes, kinds):
        ax.imshow(gradient, aspect="auto", cmap=default_cmap(kind))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f"{kind}\n{name}", rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle("Default colormaps (default_cmap(kind), shared across all styles)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def line_and_marker_cycle_swatches(out: Path) -> None:
    """LINE_STYLE_CYCLE and MARKER_CYCLE, each demonstrated on the graph
    type they actually apply to (lines on a line plot, markers on a
    scatter) rather than as an abstract legend key -- the point is to show
    what each looks like *in use*, at the same size/weight the house
    presets actually draw them."""
    apply_plot_style("icml")
    fig, (ax_lines, ax_markers) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.linspace(0, 10, 200)
    for i, ls in enumerate(LINE_STYLE_CYCLE):
        ax_lines.plot(x, np.sin(x) + i * 0.6, linestyle=ls, color="#222222",
                       linewidth=2.2, label=repr(ls))
    ax_lines.set_title("LINE_STYLE_CYCLE\n(line plots)", fontsize=12)
    ax_lines.set_xticks([])
    ax_lines.set_yticks([])
    ax_lines.legend(loc="upper right", fontsize=9)

    xs = RNG.uniform(0, 10, 12)
    for i, marker in enumerate(MARKER_CYCLE):
        ax_markers.scatter(xs, np.full_like(xs, i), marker=marker, s=90,
                            color="#4C72B0", edgecolor="#222222", linewidth=0.6)
        ax_markers.text(-0.6, i, repr(marker), ha="right", va="center", fontsize=10)
    ax_markers.set_title("MARKER_CYCLE\n(scatter plots)", fontsize=12)
    ax_markers.set_xlim(-2.2, 10.5)
    ax_markers.set_ylim(-0.8, len(MARKER_CYCLE) - 0.2)
    ax_markers.set_xticks([])
    ax_markers.set_yticks([])

    fig.suptitle("Non-color categorical cycles -- see \"Line styles, markers, and symbols\"", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def status_palette_swatches(out: Path) -> None:
    """STATUS_COLORS + STATUS_HATCHES demonstrated on the graph types that
    actually use semantic status: a bar chart (pass/fail-style summary) and
    a matshow-style cell grid (per-item status), both with the paired
    hatch as texture redundancy, not color alone."""
    apply_plot_style("icml")
    levels = list(STATUS_COLORS)
    fig, (ax_bars, ax_grid) = plt.subplots(1, 2, figsize=(12, 4.5))

    values = [0.98, 0.82, 0.55, 0.12, 0.5]
    ax_bars.bar(levels, values, color=[STATUS_COLORS[l] for l in levels],
                hatch=[STATUS_HATCHES[l] for l in levels],
                edgecolor="#222222", linewidth=1.0)
    ax_bars.set_ylim(0, 1.15)
    ax_bars.set_ylabel("e.g. fraction passing")
    ax_bars.set_title("STATUS_COLORS + STATUS_HATCHES\n(bar chart)", fontsize=12)
    for i, l in enumerate(levels):
        ax_bars.text(i, values[i] + 0.04, l, ha="center", fontsize=9)

    n = len(levels)
    for i, l in enumerate(levels):
        ax_grid.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, facecolor=STATUS_COLORS[l],
                                          hatch=STATUS_HATCHES[l], edgecolor="#222222", linewidth=1.0))
        ax_grid.text(i + 0.45, -0.35, l, ha="center", fontsize=9)
    ax_grid.set_xlim(-0.2, n)
    ax_grid.set_ylim(-0.7, 1.1)
    ax_grid.set_aspect("equal")
    ax_grid.axis("off")
    ax_grid.set_title("Same palette on filled cells\n(matshow-style grid)", fontsize=12)

    fig.suptitle('Semantic status palette -- reserved for STATE, not series identity', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style_palette_swatches(OUT_DIR / "chart_style_palettes.png")
    print(f"wrote {OUT_DIR / 'chart_style_palettes.png'}")
    default_colormap_swatches(OUT_DIR / "chart_default_colormaps.png")
    print(f"wrote {OUT_DIR / 'chart_default_colormaps.png'}")
    line_and_marker_cycle_swatches(OUT_DIR / "chart_line_marker_cycles.png")
    print(f"wrote {OUT_DIR / 'chart_line_marker_cycles.png'}")
    status_palette_swatches(OUT_DIR / "chart_status_palette.png")
    print(f"wrote {OUT_DIR / 'chart_status_palette.png'}")


if __name__ == "__main__":
    main()
