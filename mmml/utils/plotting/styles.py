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

_STYLE_ALIASES = {
    "default": DEFAULT_PLOT_STYLE,
    "pub": "nature",
    "publication": "nature",
    "grace": "xmgrace",
    "material": "google",
    "classic": "mpl_classic",
    "matplotlib": "mpl_classic",
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
