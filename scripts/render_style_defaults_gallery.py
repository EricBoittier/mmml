#!/usr/bin/env python3
"""Render "defaults at a glance" swatches for every named house style: its
categorical color cycle (comparison_palette) and, once, the shared
sequential/diverging/cyclic colormap defaults (these are global, not
per-style). Meant to sit at the TOP of docs/plot-style-gallery.md so a
reader can see what each preset actually looks like before scrolling
through individual chart examples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import (
    DEFAULT_CYCLIC_CMAP,
    DEFAULT_DIVERGING_CMAP,
    DEFAULT_SEQUENTIAL_CMAP,
    PLOT_STYLES,
    default_cmap,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"


def style_palette_swatches(out: Path) -> None:
    """One row per style: its name plus a strip of comparison_palette swatches."""
    names = list(PLOT_STYLES)
    max_colors = max(len(PLOT_STYLES[n].comparison_palette or ()) for n in names)

    fig, ax = plt.subplots(figsize=(2 + 1.1 * max_colors, 0.62 * len(names) + 0.6))
    for row, name in enumerate(names):
        y = len(names) - row - 1
        palette = PLOT_STYLES[name].comparison_palette or ()
        ax.text(-0.15, y + 0.5, name, ha="right", va="center", fontsize=11, fontweight="bold")
        for i, color in enumerate(palette):
            ax.add_patch(plt.Rectangle((i, y), 0.92, 0.92, facecolor=color, edgecolor="#222222", linewidth=0.8))

    ax.set_xlim(-4.6, max_colors + 0.2)
    ax.set_ylim(0, len(names))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Categorical color cycle per style (comparison_palette)", fontsize=13, pad=10)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style_palette_swatches(OUT_DIR / "chart_style_palettes.png")
    print(f"wrote {OUT_DIR / 'chart_style_palettes.png'}")
    default_colormap_swatches(OUT_DIR / "chart_default_colormaps.png")
    print(f"wrote {OUT_DIR / 'chart_default_colormaps.png'}")


if __name__ == "__main__":
    main()
