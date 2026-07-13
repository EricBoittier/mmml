#!/usr/bin/env python3
"""Multipole-coefficient visualization: a Pascal's-triangle-style grid (one
box per (l, m) spherical-harmonic coefficient, colored by value, annotated
with the number) -- plus a "standardized complex figure" combining it with
a per-atom bar chart and a booktabs-style table under ONE shared legend and
shared axis labels, per docs/plotting-style-guide.md "Complex figure
layout".

No saved real multipole-coefficient array exists in this checkout (real
data comes from mmml.models.multipoles at runtime, cached to Orbax
checkpoints not present here) -- values below are physically-plausible
synthetic ones (charge O(1) e, dipole O(1) e*bohr, decaying magnitude for
quadrupole/octupole), matching mmml/models/multipoles/electrostatics.py's
documented units and mmml/models/multipoles/representations.py's packed
irrep convention (max_ell=3 -> l=0,1,2,3 blocks of width 2l+1).
"""

from __future__ import annotations

from pathlib import Path

import cmap as cmap_lib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from mmml.utils.plotting.styles import (
    apply_plot_style,
    booktabs_table,
    default_cmap,
    legend_outside,
    shared_axis_labels,
)

STYLE_NAME = "icml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(3)

_L_LABELS = {0: "monopole (l=0)", 1: "dipole (l=1)", 2: "quadrupole (l=2)", 3: "octupole (l=3)"}

# Red/blue-leaning diverging candidates, shortlisted for this use case
# specifically (a classic "red = positive, blue = negative" multipole/charge
# convention), distinct from the house general-purpose diverging default
# (contrib:pampa, which is muted pink/teal, not red/blue).
_RED_BLUE_CANDIDATES = [
    "colorbrewer:RdBu_11",
    "crameri:vik",
    "crameri:roma",
    "cmocean:balance",
    "matplotlib:seismic",
    "matplotlib:coolwarm",
]


def _synthetic_multipole(max_l: int = 3) -> dict[int, np.ndarray]:
    """Physically-plausible synthetic (l, m) coefficients, magnitude
    decaying with l as real multipole expansions typically do."""
    scales = {0: 0.6, 1: 0.9, 2: 0.35, 3: 0.15}
    return {
        l: RNG.normal(0, scales[l], size=2 * l + 1)
        for l in range(max_l + 1)
    }


def _text_color_for(rgba) -> str:
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.6 else "white"


def plot_multipole_triangle(ax, coeffs_by_l: dict[int, np.ndarray], mpl_cmap, vmax: float,
                             fontsize: float = 12, fmt: str = "{:.2f}") -> None:
    """Draw one (l, m) coefficient per colored, labeled box, in a
    Pascal's-triangle layout (row l has 2l+1 boxes, centered)."""
    max_l = max(coeffs_by_l)
    n_cols = 2 * max_l + 1
    norm = Normalize(vmin=-vmax, vmax=vmax)

    for l, values in coeffs_by_l.items():
        offset = max_l - l
        row_bottom = -(l + 1)
        for i, value in enumerate(values):
            col_x = offset + i
            color = mpl_cmap(norm(value))
            ax.add_patch(Rectangle((col_x, row_bottom), 1, 1, facecolor=color,
                                    edgecolor="#222222", linewidth=1.1))
            ax.text(col_x + 0.5, row_bottom + 0.5, fmt.format(value), ha="center", va="center",
                    fontsize=fontsize, color=_text_color_for(color), fontweight="medium")
        m_labels = list(range(-l, l + 1))
        ax.text(offset - 0.15, row_bottom + 0.5, _L_LABELS[l], ha="right", va="center", fontsize=fontsize - 1)

    ax.set_xlim(-4.5, n_cols + 0.2)
    ax.set_ylim(-(max_l + 1.2), 0.2)
    ax.set_aspect("equal")
    ax.axis("off")


def colormap_comparison(out: Path) -> None:
    """Same coefficients, six red/blue-leaning diverging candidates -- pick
    one by eye, the same process used for the general house defaults."""
    coeffs = _synthetic_multipole()
    vmax = max(np.abs(v).max() for v in coeffs.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, _RED_BLUE_CANDIDATES):
        mpl_cmap = cmap_lib.Colormap(name).to_mpl()
        plot_multipole_triangle(ax, coeffs, mpl_cmap, vmax, fontsize=9)
        ax.set_title(name, fontsize=12)
    fig.suptitle("Multipole triangle: red/blue diverging candidates")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def complex_figure(out: Path, cmap_name: str = "crameri:vik") -> None:
    """A standardized "complex figure": multipole triangle + per-atom charge
    bars + a booktabs-style summary table, ONE shared legend, shared axis
    labels where panels repeat a quantity -- see
    docs/plotting-style-guide.md "Complex figure layout"."""
    coeffs = _synthetic_multipole()
    vmax = max(np.abs(v).max() for v in coeffs.values())
    mpl_cmap = cmap_lib.Colormap(cmap_name).to_mpl()
    norm = Normalize(vmin=-vmax, vmax=vmax)

    n_atoms = 8
    charges = RNG.normal(0, 0.4, n_atoms)
    atom_labels = [f"atom {i}" for i in range(n_atoms)]

    # constrained_layout (not tight_layout) -- tight_layout doesn't know how
    # to reserve room for a colorbar shared across two axes from a GridSpec
    # and silently overlaps it with the second axes; constrained_layout does.
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1.3, 1])
    ax_triangle = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    plot_multipole_triangle(ax_triangle, coeffs, mpl_cmap, vmax)
    ax_triangle.set_title("Molecular multipole moments")

    colors = [mpl_cmap(norm(q)) for q in charges]
    ax_bars.barh(atom_labels, charges, color=colors, edgecolor="#222222", linewidth=0.8)
    ax_bars.axvline(0, color="#222222", linewidth=0.8)
    ax_bars.set_title("Per-atom partial charge")
    ax_bars.set_xlabel("charge (e)")

    # One shared colorbar for BOTH panels (same colormap+norm) instead of two.
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=mpl_cmap)
    fig.colorbar(mappable, ax=[ax_triangle, ax_bars], shrink=0.7, pad=0.02,
                 label="signed magnitude (shared scale, e / e·bohr$^l$)")

    col_labels = ["quantity", "value", "units"]
    cell_text = [
        ["monopole", f"{coeffs[0][0]:+.3f}", "e"],
        ["|dipole|", f"{np.linalg.norm(coeffs[1]):.3f}", "e·bohr"],
        ["|quadrupole|", f"{np.linalg.norm(coeffs[2]):.3f}", "e·bohr²"],
        ["|octupole|", f"{np.linalg.norm(coeffs[3]):.3f}", "e·bohr³"],
        ["total atoms", str(n_atoms), "--"],
    ]
    booktabs_table(ax_table, cell_text, col_labels=col_labels, fontsize=12, header_fontsize=12,
                    col_widths=[0.22, 0.16, 0.16])
    ax_table.set_title("Summary (booktabs-style table)", fontsize=13, pad=14)

    fig.suptitle(f'Standardized complex figure ("{cmap_name}", shared legend/labels)')
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    colormap_comparison(OUT_DIR / "chart_multipole_colormaps.png")
    print(f"wrote {OUT_DIR / 'chart_multipole_colormaps.png'}")
    complex_figure(OUT_DIR / "chart_multipole_complex.png")
    print(f"wrote {OUT_DIR / 'chart_multipole_complex.png'}")


if __name__ == "__main__":
    main()
