#!/usr/bin/env python3
"""Render the user's shortlisted `cmap`-library colormaps, each on synthetic
data suited to its category (sequential / diverging / cyclic / misc), so a
default per category can be picked by eye -- see
docs/plotting-style-guide.md "Colormaps" and docs/plot-style-gallery.md
"Colormap picks: choosing defaults".

Categories are read from `cmap.Colormap(name).category`, not guessed --
see docs/plotting-style-guide.md for why a diverging map should never be
used on strictly-positive data and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import cmap
import matplotlib.pyplot as plt
import numpy as np

from mmml.utils.plotting.styles import apply_plot_style

STYLE_NAME = "icml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(0)

# name -> (kind, what it's rendered on)
_PICKS = {
    "cmocean:thermal": "sequential",
    "matplotlib:terrain": "sequential_terrain",
    "crameri:lipari": "sequential",
    "cmasher:ghostlight": "sequential",
    "cmasher:horizon": "sequential",
    "contrib:pampa": "diverging",
    "cmasher:watermelon": "diverging",
    "cmocean:delta": "diverging",
    "cmocean:diff": "diverging",
    "cmasher:guppy_r": "diverging",
    "yorick:stern": "misc_hdr",
    "cmocean:phase": "cyclic",
}


def _sequential_field():
    """A smooth, strictly-positive field -- e.g. local density/RDF-like decay."""
    x = np.linspace(-3, 3, 120)
    y = np.linspace(-3, 3, 120)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, np.exp(-(xx**2 + yy**2) / 3.0) + 0.4 * np.exp(-((xx - 1.5) ** 2 + (yy + 1.0) ** 2) / 0.8)


def _terrain_field():
    """Elevation-like field with a "sea level" -- what `terrain` is actually for."""
    x = np.linspace(0, 10, 120)
    y = np.linspace(0, 10, 120)
    xx, yy = np.meshgrid(x, y)
    z = 2.0 * np.sin(xx / 2) * np.cos(yy / 2) + 0.5 * xx - 3.0
    return xx, yy, z


def _diverging_field():
    """Zero-centered residual/difference field -- e.g. two calculators' energy delta."""
    xx, yy, base = _sequential_field()
    return xx, yy, (base - base.mean()) * 6.0


def _hdr_field():
    """High-dynamic-range field (one sharp spike over a broad low background) --
    what "stern special"-style maps were originally designed to show (astronomical
    images with a huge brightness range)."""
    xx, yy, base = _sequential_field()
    spike = 25.0 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.2) ** 2) / 0.02)
    return xx, yy, base + spike


def _cyclic_field():
    """A phase/angle field -- exactly what a cyclic colormap is for."""
    x = np.linspace(-3, 3, 120)
    y = np.linspace(-3, 3, 120)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, np.arctan2(yy, xx)  # in (-pi, pi], wraps around


_FIELD_FNS = {
    "sequential": _sequential_field,
    "sequential_terrain": _terrain_field,
    "diverging": _diverging_field,
    "misc_hdr": _hdr_field,
    "cyclic": _cyclic_field,
}

_FIELD_LABELS = {
    "sequential": "sequential: smooth density-like field",
    "sequential_terrain": "sequential: elevation field (sea level at 0)",
    "diverging": "diverging: zero-centered residual field",
    "misc_hdr": "misc: high-dynamic-range field (one sharp spike)",
    "cyclic": "cyclic: phase/angle field (wraps at ±π)",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)

    names = list(_PICKS)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, name in zip(axes.flat, names):
        kind = _PICKS[name]
        xx, yy, zz = _FIELD_FNS[kind]()
        mpl_cmap = cmap.Colormap(name).to_mpl()
        vmax = np.abs(zz).max()
        vmin = -vmax if kind in ("diverging", "cyclic") else zz.min()
        im = ax.pcolormesh(xx, yy, zz, cmap=mpl_cmap, shading="gouraud", vmin=vmin, vmax=vmax)
        category = cmap.Colormap(name).category
        ax.set_title(f"{name}\n({category})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    # blank the unused 12th slot's twin (none here: exactly 12 picks, 3x4 grid)
    fig.suptitle("Shortlisted colormaps, each on data suited to its category", y=1.0)
    fig.tight_layout()
    out = OUT_DIR / "colormap_picks.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # A second figure: same field family, grouped by category, for an
    # easier apples-to-apples comparison within sequential/diverging.
    for kind in ("sequential", "diverging"):
        picks = [n for n, k in _PICKS.items() if k == kind or (kind == "sequential" and k == "sequential_terrain")]
        n = len(picks)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
        for ax, name in zip(np.atleast_1d(axes), picks):
            field_kind = _PICKS[name]
            xx, yy, zz = _FIELD_FNS[field_kind]()
            mpl_cmap = cmap.Colormap(name).to_mpl()
            vmax = np.abs(zz).max()
            vmin = -vmax if kind == "diverging" else zz.min()
            im = ax.pcolormesh(xx, yy, zz, cmap=mpl_cmap, shading="gouraud", vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(f"{kind.capitalize()} picks, same field, side by side")
        fig.tight_layout()
        out = OUT_DIR / f"colormap_picks_{kind}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
