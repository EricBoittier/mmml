"""Matplotlib plot style presets for training curves and scientific figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PlotStyle",
    "PLOT_STYLES",
    "DEFAULT_PLOT_STYLE",
    "apply_plot_style",
    "get_plot_style",
    "list_plot_styles",
    "comparison_colors",
    "legend_outside",
    "seed_symbol",
    "SEED_DICE",
    "default_cmap",
    "DEFAULT_SEQUENTIAL_CMAP",
    "DEFAULT_DIVERGING_CMAP",
    "DEFAULT_CYCLIC_CMAP",
]


@dataclass(frozen=True)
class PlotStyle:
    """Named matplotlib styling bundle for MMML training plots."""

    name: str
    description: str
    colors: Mapping[str, str]
    rc_params: Mapping[str, Any] = field(default_factory=dict)
    train_linewidth: float = 2.0
    valid_linewidth: float = 2.4
    best_marker_edge: str = "#222222"
    best_marker_size: float = 120.0
    comparison_palette: Sequence[str] = ()
    text_box: Mapping[str, Any] = field(
        default_factory=lambda: {
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.95,
        }
    )
    summary_font_family: str = "monospace"
    suptitle_color: str | None = None


def _style(
    name: str,
    description: str,
    *,
    colors: Mapping[str, str],
    rc_params: Mapping[str, Any],
    comparison_palette: Sequence[str],
    **kwargs: Any,
) -> PlotStyle:
    return PlotStyle(
        name=name,
        description=description,
        colors=colors,
        rc_params=rc_params,
        comparison_palette=comparison_palette,
        **kwargs,
    )


# Nature / Science: compact sans-serif, light grid, restrained palette.
_NATURE_COLORS = {
    "train": "#3C5488",
    "valid": "#E64B35",
    "best": "#F39B7F",
    "accent": "#00A087",
    "lr": "#7E6148",
    "muted": "#8491B4",
}
_NATURE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.color": "#B0B0B0",
    "legend.framealpha": 1.0,
    "legend.fontsize": 8,
    "legend.edgecolor": "#CCCCCC",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
}

# XMGrace: black axes, sparse dotted grid, high-contrast classic scientific look.
_XMGRACE_COLORS = {
    "train": "#000000",
    "valid": "#CC0000",
    "best": "#006600",
    "accent": "#0000CC",
    "lr": "#660066",
    "muted": "#666666",
}
_XMGRACE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.grid": True,
    "grid.alpha": 0.55,
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "grid.color": "#888888",
    "legend.framealpha": 1.0,
    "legend.fontsize": 9,
    "legend.edgecolor": "black",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "lines.linewidth": 1.8,
}

# Google / Material: soft surfaces, rounded feel, brand-like primaries.
_GOOGLE_COLORS = {
    "train": "#4285F4",
    "valid": "#EA4335",
    "best": "#FBBC04",
    "accent": "#34A853",
    "lr": "#9334E6",
    "muted": "#9AA0A6",
}
_GOOGLE_RC = {
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#DADCE0",
    "axes.linewidth": 0.9,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "grid.color": "#E8EAED",
    "legend.framealpha": 0.98,
    "legend.fontsize": 9,
    "legend.edgecolor": "#E8EAED",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Google Sans", "Arial", "DejaVu Sans"],
    "lines.linewidth": 2.2,
    "lines.solid_capstyle": "round",
}

# TRON / dark mode: neon lines on deep background.
_TRON_COLORS = {
    "train": "#00E5FF",
    "valid": "#FF2BD6",
    "best": "#FFE600",
    "accent": "#39FF14",
    "lr": "#B388FF",
    "muted": "#6E7A8A",
}
_TRON_RC = {
    "figure.facecolor": "#0A0E17",
    "axes.facecolor": "#101820",
    "axes.edgecolor": "#00E5FF",
    "axes.labelcolor": "#C8E6FF",
    "axes.titlecolor": "#E8F4FF",
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#1E3A5F",
    "legend.framealpha": 0.85,
    "legend.facecolor": "#101820",
    "legend.edgecolor": "#00E5FF",
    "legend.fontsize": 9,
    "legend.labelcolor": "#C8E6FF",
    "text.color": "#C8E6FF",
    "xtick.color": "#8ECAE6",
    "ytick.color": "#8ECAE6",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "DejaVu Sans"],
    "lines.linewidth": 2.0,
}

# Tufte: large type, thick lines, LaTeX-style math (mathtext, no TeX install
# "Editorial" family: large type, thick lines, LaTeX-style math (mathtext,
# no TeX install needed), minimal chart junk (no top/right spines, sparse
# ticks, faint grid). NOTE: this axis/spacing treatment draws on Tufte's
# data-ink-ratio *principles* -- it is not itself "the Tufte style" (Tufte
# never specified one font or palette; the principles are about ink,
# redundant encoding, and small multiples, not a fixed look). Several font
# variants share this same axis treatment so the font choice can be judged
# on its own -- see docs/plot-style-gallery.md for a rendered comparison.
# Colors are semantic pairs, not a generic cycling palette -- see
# docs/plotting-style-guide.md "Semantic color, not palette index" for how
# callers are expected to map domain categories (e.g. MM vs ML, pass vs
# fail) onto fixed, meaningful colors rather than an arbitrary series order.
_EDITORIAL_COLORS = {
    "train": "#1A5276",   # deep slate blue
    "valid": "#943126",   # brick red
    "best": "#B9770E",    # ochre
    "accent": "#1E8449",  # forest green
    "lr": "#6C3483",      # muted purple
    "muted": "#5D6D7E",   # slate gray
}


def _editorial_rc(*, family: str, serif: list[str] | None = None,
                   sans: list[str] | None = None, mathtext_fontset: str) -> dict[str, Any]:
    """Shared axis/spacing treatment for the editorial_* presets; only the
    typeface (family/serif-or-sans/mathtext_fontset) differs between them."""
    rc: dict[str, Any] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.color": "#888888",
        "legend.framealpha": 0.92,
        "legend.fontsize": 12,
        "legend.edgecolor": "#CCCCCC",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": family,
        "font.size": 14,
        "mathtext.fontset": mathtext_fontset,  # no TeX install needed
        "lines.linewidth": 2.8,
        "lines.solid_capstyle": "round",
        "lines.markersize": 8,
        "patch.linewidth": 1.2,
        "savefig.dpi": 200,
    }
    if serif is not None:
        rc["font.serif"] = serif
    if sans is not None:
        rc["font.sans-serif"] = sans
    return rc


# Five font variants, same axes/spacing -- pick by eye from the gallery.
_EDITORIAL_RC_DEJAVU_SANS = _editorial_rc(
    family="sans-serif", sans=["DejaVu Sans"], mathtext_fontset="dejavusans")
_EDITORIAL_RC_DEJAVU_SERIF = _editorial_rc(
    family="serif", serif=["DejaVu Serif"], mathtext_fontset="dejavuserif")
_EDITORIAL_RC_STIX = _editorial_rc(
    family="serif", serif=["STIXGeneral", "DejaVu Serif"], mathtext_fontset="stix")
_EDITORIAL_RC_CM = _editorial_rc(
    family="serif", serif=["DejaVu Serif"], mathtext_fontset="cm")


# ICML/NeurIPS-paper-figure vibe: clean sans-serif, moderate (not oversized)
# type, muted "seaborn deep"-style categorical colors, legend meant to live
# OUTSIDE the axes (see legend_outside() below) rather than overlapping data.
# This is the closest preset to a modern ML-conference plot: less soft/round
# than "google", less serif/journal than the editorial_* family.
_ICML_COLORS = {
    "train": "#4C72B0",   # muted blue
    "valid": "#DD8452",   # muted orange
    "best": "#55A868",    # muted green
    "accent": "#C44E52",  # muted red
    "lr": "#8172B2",      # muted purple
    "muted": "#937860",   # muted brown
}
_ICML_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#DDDDDD",
    "axes.axisbelow": True,
    "legend.framealpha": 0.95,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "legend.edgecolor": "#CCCCCC",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 14,
    "mathtext.fontset": "dejavusans",
    "lines.linewidth": 2.4,
    "lines.solid_capstyle": "round",
    "lines.markersize": 7,
    "patch.linewidth": 1.0,
    "savefig.dpi": 200,
}


# Filled-circle counts instead of "(seed N)" text -- tried Unicode die faces
# (U+2680-2685) first, but they render as generic missing-glyph boxes on
# common sans-serif fonts (Helvetica Neue, and likely fonts on the cluster);
# "●" (filled circle, U+25CF) is part of essentially every font
# (including matplotlib's bundled DejaVu Sans) and renders reliably.
SEED_DICE = {n: "●" * n for n in range(1, 7)}


def seed_symbol(seed: int) -> str:
    """seed 1 -> "●", seed 2 -> "●●", etc. (falls back to the
    plain number above 6, where repeated dots stop being readable at a glance)."""
    seed = int(seed)
    return SEED_DICE[seed] if seed in SEED_DICE else str(seed)


def legend_outside(target, *, side: str = "auto", ncol: int | None = None, **kwargs: Any):
    """Place a legend outside the plotted data, on whichever side matches the
    figure's *longest* dimension -- never squeezed against the short side.

    See docs/plotting-style-guide.md "Legends live outside the plot":

    - ``target`` an ``Axes``: legend is anchored to that axes specifically
      (right or left of it) -- use this for a multi-column figure, where the
      left column's legend goes further left and the right column's goes
      further right, rather than stacking both on one side.
    - ``target`` a ``Figure``: legend is anchored to the whole figure (right
      or below it) -- use this for a single-column, multi-row (stacked)
      figure, where "outside" more naturally means below the tallest side.
    - ``side="auto"`` (default): compares the *figure's* width to height
      (``fig.get_size_inches()``) -- wider-than-tall figures get a
      right-hand legend, taller-than-wide figures get a below legend. Pass
      ``side="left"``/``"right"``/``"bottom"`` to override.

    A legend placed this way is also free to grow large (many entries, long
    labels) without crowding the data, so it can double as a compact table
    (e.g. one row per setting with its color/marker/seed-die as the row key).
    ``ncol`` defaults to 1 for a side legend (reads top-to-bottom like a
    table) and wraps to multiple columns for a bottom legend (so a long
    legend doesn't stretch further than the figure itself).
    """
    import matplotlib.figure

    fig = target if isinstance(target, matplotlib.figure.Figure) else target.figure

    if side == "auto":
        width_in, height_in = fig.get_size_inches()
        side = "right" if width_in >= height_in else "bottom"

    if side == "right":
        return target.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                              borderaxespad=0.0, ncol=ncol or 1, **kwargs)
    if side == "left":
        return target.legend(loc="upper right", bbox_to_anchor=(-0.02, 1.0),
                              borderaxespad=0.0, ncol=ncol or 1, **kwargs)
    if side == "bottom":
        n_entries = len(kwargs.get("labels") or kwargs.get("handles") or [])
        if not n_entries and hasattr(target, "get_legend_handles_labels"):
            n_entries = len(target.get_legend_handles_labels()[1])
        default_ncol = max(1, min(4, n_entries)) if n_entries else 3
        return target.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                              borderaxespad=0.0, ncol=ncol or default_ncol, **kwargs)
    raise ValueError(f"side must be 'auto', 'left', 'right', or 'bottom'; got {side!r}")


# Classic matplotlib defaults (pre-seaborn era feel).
_MPL_CLASSIC_COLORS = {
    "train": "#1f77b4",
    "valid": "#d62728",
    "best": "#ff7f0e",
    "accent": "#2ca02c",
    "lr": "#9467bd",
    "muted": "#7f7f7f",
}
_MPL_CLASSIC_RC = {
    "figure.facecolor": "0.75",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.grid": False,
    "legend.framealpha": 1.0,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "lines.linewidth": 1.5,
}


PLOT_STYLES: dict[str, PlotStyle] = {
    "nature": _style(
        "nature",
        "Nature/Science journal: compact sans-serif, restrained palette.",
        colors=_NATURE_COLORS,
        rc_params=_NATURE_RC,
        comparison_palette=("#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F", "#8491B4"),
        train_linewidth=1.6,
        valid_linewidth=1.9,
        best_marker_edge="#333333",
    ),
    "science": _style(
        "science",
        "Alias of nature (Science-style publication figures).",
        colors=_NATURE_COLORS,
        rc_params=_NATURE_RC,
        comparison_palette=("#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F", "#8491B4"),
        train_linewidth=1.6,
        valid_linewidth=1.9,
        best_marker_edge="#333333",
    ),
    "xmgrace": _style(
        "xmgrace",
        "XMGrace-like: black axes, dotted grid, serif labels.",
        colors=_XMGRACE_COLORS,
        rc_params=_XMGRACE_RC,
        comparison_palette=("#000000", "#CC0000", "#0000CC", "#006600", "#660066", "#CC6600"),
        train_linewidth=1.8,
        valid_linewidth=2.0,
        best_marker_edge="black",
    ),
    "google": _style(
        "google",
        "Google/Material charts: soft surfaces and brand primaries.",
        colors=_GOOGLE_COLORS,
        rc_params=_GOOGLE_RC,
        comparison_palette=("#4285F4", "#EA4335", "#34A853", "#FBBC04", "#9334E6", "#00ACC1"),
        train_linewidth=2.2,
        valid_linewidth=2.5,
        best_marker_edge="#FFFFFF",
    ),
    "tron": _style(
        "tron",
        "Dark mode / TRON: neon curves on a deep background.",
        colors=_TRON_COLORS,
        rc_params=_TRON_RC,
        comparison_palette=("#00E5FF", "#FF2BD6", "#39FF14", "#FFE600", "#B388FF", "#FF6B35"),
        train_linewidth=2.0,
        valid_linewidth=2.3,
        best_marker_edge="#0A0E17",
        text_box={
            "boxstyle": "round",
            "facecolor": "#101820",
            "edgecolor": "#00E5FF",
            "alpha": 0.92,
        },
        summary_font_family="monospace",
        suptitle_color="#E8F4FF",
    ),
    "dark": _style(
        "dark",
        "Alias of tron (dark-mode neon aesthetic).",
        colors=_TRON_COLORS,
        rc_params=_TRON_RC,
        comparison_palette=("#00E5FF", "#FF2BD6", "#39FF14", "#FFE600", "#B388FF", "#FF6B35"),
        train_linewidth=2.0,
        valid_linewidth=2.3,
        best_marker_edge="#0A0E17",
        text_box={
            "boxstyle": "round",
            "facecolor": "#101820",
            "edgecolor": "#00E5FF",
            "alpha": 0.92,
        },
        summary_font_family="monospace",
        suptitle_color="#E8F4FF",
    ),
    # "editorial_*": same large-type/thick-line/no-top-right-spine axis
    # treatment (data-ink-ratio principles, not "the Tufte style" -- see the
    # comment above _EDITORIAL_COLORS); each entry below differs ONLY in
    # typeface, so they can be compared on font choice alone. Rendered
    # side-by-side in docs/plot-style-gallery.md.
    "editorial_dejavu_sans": _style(
        "editorial_dejavu_sans",
        "Editorial axes (large type, thick lines, no top/right spine) in DejaVu Sans.",
        colors=_EDITORIAL_COLORS, rc_params=_EDITORIAL_RC_DEJAVU_SANS,
        comparison_palette=("#1A5276", "#943126", "#B9770E", "#1E8449", "#6C3483", "#5D6D7E"),
        train_linewidth=2.8, valid_linewidth=3.2, best_marker_edge="#222222", best_marker_size=160.0,
        text_box={"boxstyle": "round", "facecolor": "#FBFBF8", "edgecolor": "#999999", "alpha": 0.95},
    ),
    "editorial_dejavu_serif": _style(
        "editorial_dejavu_serif",
        "Editorial axes in DejaVu Serif (matplotlib's bundled serif -- always renders identically).",
        colors=_EDITORIAL_COLORS, rc_params=_EDITORIAL_RC_DEJAVU_SERIF,
        comparison_palette=("#1A5276", "#943126", "#B9770E", "#1E8449", "#6C3483", "#5D6D7E"),
        train_linewidth=2.8, valid_linewidth=3.2, best_marker_edge="#222222", best_marker_size=160.0,
        text_box={"boxstyle": "round", "facecolor": "#FBFBF8", "edgecolor": "#999999", "alpha": 0.95},
    ),
    "editorial_stix": _style(
        "editorial_stix",
        "Editorial axes in STIX serif (journal-typeset feel; was previously called 'tufte').",
        colors=_EDITORIAL_COLORS, rc_params=_EDITORIAL_RC_STIX,
        comparison_palette=("#1A5276", "#943126", "#B9770E", "#1E8449", "#6C3483", "#5D6D7E"),
        train_linewidth=2.8, valid_linewidth=3.2, best_marker_edge="#222222", best_marker_size=160.0,
        text_box={"boxstyle": "round", "facecolor": "#FBFBF8", "edgecolor": "#999999", "alpha": 0.95},
    ),
    "editorial_cm": _style(
        "editorial_cm",
        "Editorial axes with Computer-Modern-style math (classic LaTeX-paper look).",
        colors=_EDITORIAL_COLORS, rc_params=_EDITORIAL_RC_CM,
        comparison_palette=("#1A5276", "#943126", "#B9770E", "#1E8449", "#6C3483", "#5D6D7E"),
        train_linewidth=2.8, valid_linewidth=3.2, best_marker_edge="#222222", best_marker_size=160.0,
        text_box={"boxstyle": "round", "facecolor": "#FBFBF8", "edgecolor": "#999999", "alpha": 0.95},
    ),
    "icml": _style(
        "icml",
        "ICML/NeurIPS-paper vibe: clean sans-serif, muted categorical colors, "
        "legend meant to live outside the axes (see legend_outside()).",
        colors=_ICML_COLORS, rc_params=_ICML_RC,
        comparison_palette=("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"),
        train_linewidth=2.4, valid_linewidth=2.6, best_marker_edge="#333333", best_marker_size=130.0,
        text_box={"boxstyle": "round", "facecolor": "#FAFAFA", "edgecolor": "#CCCCCC", "alpha": 0.95},
    ),
    "mpl_classic": _style(
        "mpl_classic",
        "Classic matplotlib defaults (blue/red, no grid).",
        colors=_MPL_CLASSIC_COLORS,
        rc_params=_MPL_CLASSIC_RC,
        comparison_palette=("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"),
        train_linewidth=1.5,
        valid_linewidth=1.5,
        best_marker_edge="black",
        text_box={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 1.0,
        },
    ),
}

DEFAULT_PLOT_STYLE = "google"

# House colormap defaults -- chosen from a rendered shortlist, see
# docs/plot-style-gallery.md "Colormap picks: choosing defaults". Perceptually
# uniform/colorblind-safe where possible; muted rather than high-saturation
# per the "quiet" Tufte-aligned house look.
DEFAULT_SEQUENTIAL_CMAP = "crameri:lipari"
DEFAULT_DIVERGING_CMAP = "contrib:pampa"
DEFAULT_CYCLIC_CMAP = "cmocean:phase"


def default_cmap(kind: str):
    """Resolve one of the house colormap defaults to a matplotlib Colormap.

    ``kind`` is one of ``"sequential"``, ``"diverging"``, ``"cyclic"`` --
    see docs/plotting-style-guide.md "Colormaps" for when to use which, and
    never use a diverging map on strictly-positive data or vice versa.
    Requires the optional ``cmap`` library (``pip install cmap`` / the
    ``plotting`` extra); raises with an explicit message if missing rather
    than silently falling back to an unrelated matplotlib colormap.
    """
    names = {
        "sequential": DEFAULT_SEQUENTIAL_CMAP,
        "diverging": DEFAULT_DIVERGING_CMAP,
        "cyclic": DEFAULT_CYCLIC_CMAP,
    }
    if kind not in names:
        raise ValueError(f"kind must be one of {sorted(names)}; got {kind!r}")
    try:
        import cmap as cmap_lib
    except ImportError as exc:
        raise ImportError(
            "default_cmap() requires the 'cmap' library -- install the "
            "'plotting' extra (uv sync --extra plotting) or `pip install cmap`."
        ) from exc
    return cmap_lib.Colormap(names[kind]).to_mpl()

_STYLE_ALIASES = {
    "default": DEFAULT_PLOT_STYLE,
    "pub": "nature",
    "publication": "nature",
    "grace": "xmgrace",
    "material": "google",
    "classic": "mpl_classic",
    "matplotlib": "mpl_classic",
    "editorial": "editorial_dejavu_serif",
    "tufte": "editorial_stix",  # renamed: Tufte is a set of principles, not one font/preset
}


def list_plot_styles(*, include_aliases: bool = False) -> list[str]:
    """Return registered style names (canonical keys only unless aliases requested)."""
    names = sorted(PLOT_STYLES.keys())
    if include_aliases:
        names = sorted(set(names) | set(_STYLE_ALIASES.keys()))
    return names


def get_plot_style(name: str | PlotStyle | None = None) -> PlotStyle:
    """Resolve a style name (or pass-through an existing PlotStyle)."""
    if isinstance(name, PlotStyle):
        return name
    key = (name or DEFAULT_PLOT_STYLE).strip().lower()
    key = _STYLE_ALIASES.get(key, key)
    if key not in PLOT_STYLES:
        valid = ", ".join(sorted(PLOT_STYLES))
        raise ValueError(f"Unknown plot style {name!r}. Choose from: {valid}")
    return PLOT_STYLES[key]


def apply_plot_style(name: str | PlotStyle | None = None) -> PlotStyle:
    """Apply matplotlib rcParams for the requested style and return it."""
    style = get_plot_style(name)
    plt.rcParams.update(style.rc_params)
    return style


def comparison_colors(style: str | PlotStyle | None, n: int) -> list[str]:
    """Return *n* distinct line colors for multi-run overlays."""
    resolved = get_plot_style(style)
    if resolved.comparison_palette:
        palette = list(resolved.comparison_palette)
        if n <= len(palette):
            return palette[:n]
        reps = int(np.ceil(n / len(palette)))
        return (palette * reps)[:n]
    return [plt.cm.tab10(i % 10) for i in range(n)]
