"""Matplotlib plot style presets for training curves and scientific figures."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
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
    "OKABE_ITO_PALETTE",
    "MULTI_CMAP_SHORTLIST",
    "LINE_STYLE_CYCLE",
    "MARKER_CYCLE",
    "shared_axis_labels",
    "booktabs_table",
    "latex_available",
    "render_latex_table",
    "latex_table_image",
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
    # Wire comparison_palette into matplotlib's actual color cycle
    # (axes.prop_cycle), not just comparison_colors()'s manual lookup -- a
    # bare `ax.plot(x, y)`/`ax.scatter(...)` with no explicit `color=` should
    # already draw from the house palette instead of silently falling back
    # to matplotlib's built-in tab10 cycle.
    resolved_rc = dict(rc_params)
    resolved_rc.setdefault("axes.prop_cycle", plt.cycler(color=list(comparison_palette)))
    return PlotStyle(
        name=name,
        description=description,
        colors=colors,
        rc_params=resolved_rc,
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
    "axes.titlepad": 10.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.color": "#B0B0B0",
    "legend.frameon": False,
    "legend.fontsize": 8,
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
    "axes.titlepad": 10.0,
    "axes.grid": True,
    "grid.alpha": 0.55,
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "grid.color": "#888888",
    "legend.frameon": False,
    "legend.fontsize": 9,
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
    "axes.titlepad": 12.0,
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "grid.color": "#E8EAED",
    "legend.frameon": False,
    "legend.fontsize": 9,
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
    "axes.titlepad": 12.0,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#1E3A5F",
    "legend.frameon": False,
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
        "axes.titlepad": 16.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.color": "#888888",
        "legend.frameon": False,
        "legend.fontsize": 12,
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
    "axes.titlepad": 16.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "#DDDDDD",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
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
      (``fig.get_size_inches()``) -- a wide figure already has little spare
      width, so a right-hand legend would squeeze the data further; put it
      **below** instead, where the surplus width becomes room for a
      multi-column legend. A tall (portrait/stacked) figure has little
      spare height for a bottom legend to live in without stretching the
      figure further, so put it on the **side** instead, where the surplus
      height is already there to absorb it. Pass
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
        side = "bottom" if width_in >= height_in else "right"

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
        # -0.18, not -0.08: an Axes' own x-axis label already occupies roughly
        # the -0.08..-0.14 band below the axes box (tick labels then the
        # label itself) -- -0.08 sits the legend right on top of it. This
        # matters more now that "auto" picks "bottom" for wide figures (the
        # common case), not just occasionally.
        return target.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                              borderaxespad=0.0, ncol=ncol or default_ncol, **kwargs)
    raise ValueError(f"side must be 'auto', 'left', 'right', or 'bottom'; got {side!r}")


def shared_axis_labels(fig, *, xlabel: str | None = None, ylabel: str | None = None,
                        clear_panel_labels: bool = True, **kwargs: Any) -> None:
    """One x/y-axis label for the whole figure instead of repeating it on
    every panel -- see docs/plotting-style-guide.md "Complex figure layout":
    a multi-panel figure sharing units/quantity on an axis should say so
    once (`fig.supxlabel`/`fig.supylabel`), not N times.

    ``clear_panel_labels=True`` (default) blanks each axes' own x/y label so
    the shared one isn't duplicated underneath it.
    """
    if xlabel is not None:
        fig.supxlabel(xlabel, **kwargs)
    if ylabel is not None:
        fig.supylabel(ylabel, **kwargs)
    if clear_panel_labels:
        for ax in fig.axes:
            if xlabel is not None:
                ax.set_xlabel("")
            if ylabel is not None:
                ax.set_ylabel("")


def booktabs_table(ax, cell_text, *, col_labels=None, row_labels=None,
                    fontsize: float = 12, header_fontsize: float | None = None,
                    col_widths=None, row_height: float = 2.0,
                    numeric_cols: Sequence[int] | None = None):
    """A LaTeX-`booktabs`-style table drawn with matplotlib: a rule above the
    header, a rule below the header, a rule at the bottom -- and nothing
    else. No vertical rules, no per-cell grid, no zebra-striping -- the
    classic "table ink" minimalism `booktabs` popularized in LaTeX, done in
    matplotlib so it renders in the same figure/style pipeline as everything
    else (same font, exportable as one PNG/PDF with the rest of a panel).

    matplotlib's raw `ax.table()` defaults to cramped, vertically-centered,
    center-aligned cells that read as "a grid dumped onto a figure" rather
    than a typeset table -- ``row_height`` opens up breathing room (via
    `Table.scale`), and text is nudged left-aligned within its cell padding
    for row/data labels and right-aligned for ``numeric_cols`` (columns that
    are actual numbers read better ones-place-aligned; auto-detected from
    the first data row when not given).

    ``ax`` should have no other content -- give it its own subplot/axes.
    Returns the underlying `matplotlib.table.Table` for further tweaks.
    """
    ax.set_axis_off()
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, rowLabels=row_labels,
                    cellLoc="center", loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, row_height)

    has_header = col_labels is not None
    n_rows = len(cell_text) + (1 if has_header else 0)
    n_cols = len(cell_text[0]) if cell_text else 0
    col_range = range(-1 if row_labels is not None else 0, n_cols)

    if numeric_cols is None and cell_text:
        numeric_cols = [
            c for c in range(n_cols)
            if all(_looks_numeric(row[c]) for row in cell_text)
        ]
    numeric_cols = set(numeric_cols or ())

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0)  # start from no rules; only 2-3 exact rules added back below
        cell.set_edgecolor("#222222")
        cell.PAD = 0.06
        is_data_row = row > 0 if has_header else row >= 0
        if col in numeric_cols and is_data_row:
            cell.get_text().set_ha("right")
            cell.get_text().set_position((1.0 - cell.PAD, 0.5))
        elif col >= 0:
            cell.get_text().set_ha("left")
            cell.get_text().set_position((cell.PAD, 0.5))
        if has_header and row == 0:
            cell.set_text_props(fontweight="bold")
            if header_fontsize:
                cell.set_fontsize(header_fontsize)
            if col in numeric_cols:
                cell.get_text().set_ha("right")
                cell.get_text().set_position((1.0 - cell.PAD, 0.5))

    def _set_edge(row: int, col: int, edge: str) -> None:
        try:
            cell = tbl[row, col]
        except KeyError:
            return
        existing = cell.visible_edges if cell.visible_edges != "closed" else ""
        cell.visible_edges = "".join(sorted(set(existing) | {edge}))
        cell.set_linewidth(RULE_WIDTH)

    RULE_WIDTH = 1.4
    # Top rule: above the header (or above the first data row if headerless).
    for col in col_range:
        _set_edge(0, col, "T")
    # Rule below the header, separating it from the data -- only if there IS one.
    if has_header:
        for col in col_range:
            _set_edge(0, col, "B")
    # Bottom rule: below the last row.
    last_row = n_rows - 1
    for col in col_range:
        _set_edge(last_row, col, "B")
    return tbl


def _looks_numeric(value: object) -> bool:
    text = str(value).strip()
    text = text.lstrip("+-").replace(",", "")
    for suffix in ("%", "°"):
        text = text.removesuffix(suffix)
    try:
        float(text)
        return True
    except ValueError:
        return False


_LATEX_SPECIAL = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "|": r"\textbar{}"}


def _latex_escape(value: object) -> str:
    """Escape the handful of LaTeX-special characters that show up in plain
    labels/numbers (underscore, percent, ampersand, hash, pipe -- a bare `|`
    in LaTeX text mode does not render as a literal vertical bar with every
    font). Deliberately does NOT touch `$...$`/`\\` -- a cell that already
    contains real LaTeX math (e.g. ``r"$|\\mathbf{q}|$"``) is passed through
    untouched."""
    text = str(value)
    if "$" in text or "\\" in text:
        return text
    for char, escaped in _LATEX_SPECIAL.items():
        text = text.replace(char, escaped)
    return text


def latex_available() -> bool:
    """Whether a LaTeX toolchain (pdflatex + pdftocairo) is on PATH -- both
    `render_latex_table` and `latex_table_image` need these to actually
    typeset with real LaTeX/booktabs rather than matplotlib's own `Axes.table`
    approximation of it."""
    return shutil.which("pdflatex") is not None and shutil.which("pdftocairo") is not None


def render_latex_table(cell_text, *, col_labels=None, row_labels=None,
                        out_path: str | Path | None = None, fontsize_pt: float = 11,
                        column_format: str | None = None, escape: bool = True,
                        dpi: int = 400) -> Path:
    """Typeset a real LaTeX `booktabs` table (compiled with `pdflatex`,
    rasterized with `pdftocairo`) and return the path to a transparent PNG.

    This exists because matplotlib's own `Axes.table` (see `booktabs_table`
    below) can only approximate a typeset table -- it can't do real kerning,
    proper decimal-point alignment, or LaTeX math in cells. When a real LaTeX
    toolchain is available (`latex_available()`), prefer this for any table
    that will appear in a final figure; `booktabs_table` remains as a
    dependency-free fallback.

    ``out_path`` defaults to a content-hashed path under a repo-local cache
    directory (`.cache/latex_tables/`) so repeated calls with the same table
    reuse the same compiled PNG instead of re-invoking LaTeX every time.
    """
    if not latex_available():
        raise RuntimeError(
            "render_latex_table() requires `pdflatex` and `pdftocairo` on PATH "
            "-- install a LaTeX distribution (e.g. MacTeX/TeX Live) providing "
            "both, plus the `booktabs` and `standalone` packages, or use "
            "booktabs_table() instead (matplotlib-only, no LaTeX dependency)."
        )

    has_header = col_labels is not None
    n_cols = len(cell_text[0]) if cell_text else (len(col_labels) if col_labels else 0)
    if column_format is None:
        col_letter = "l" if row_labels is not None else ""
        column_format = col_letter + "l" * n_cols

    def _row(cells) -> str:
        rendered = [_latex_escape(c) if escape else str(c) for c in cells]
        return " & ".join(rendered) + r" \\"

    lines = [
        r"\documentclass[preview,border=3pt,varwidth]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\begin{document}",
        rf"\fontsize{{{fontsize_pt}pt}}{{{fontsize_pt * 1.2}pt}}\selectfont",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
    ]
    if has_header:
        header_cells = ([""] if row_labels is not None else []) + list(col_labels)
        lines.append(_row(header_cells))
        lines.append(r"\midrule")
    for i, row in enumerate(cell_text):
        row_cells = ([row_labels[i]] if row_labels is not None else []) + list(row)
        lines.append(_row(row_cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{document}"]
    tex_source = "\n".join(lines)

    if out_path is None:
        digest = hashlib.sha256(tex_source.encode() + str(dpi).encode()).hexdigest()[:16]
        cache_dir = Path(__file__).resolve().parents[3] / ".cache" / "latex_tables"
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"table_{digest}.png"
    out_path = Path(out_path)
    if out_path.exists():
        return out_path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tex_file = tmp_path / "table.tex"
        tex_file.write_text(tex_source)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf_file = tmp_path / "table.pdf"
        if result.returncode != 0 or not pdf_file.exists():
            raise RuntimeError(f"pdflatex failed to compile the table:\n{result.stdout[-4000:]}")
        subprocess.run(
            ["pdftocairo", "-png", "-r", str(dpi), "-transp", "-singlefile",
             pdf_file.name, "table"],
            cwd=tmp, check=True, capture_output=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "table.png", out_path)
    return out_path


def latex_table_image(ax, cell_text, *, col_labels=None, row_labels=None,
                       fontsize_pt: float = 11, column_format: str | None = None,
                       escape: bool = True, dpi: int = 400):
    """Compile a real LaTeX `booktabs` table (`render_latex_table`) and draw
    it into ``ax`` via `imshow`, so it drops into a matplotlib figure/subplot
    exactly like `booktabs_table` does, but typeset by an actual LaTeX
    engine rather than approximated by `Axes.table`.

    ``ax`` should have no other content -- give it its own subplot/axes.
    """
    png_path = render_latex_table(
        cell_text, col_labels=col_labels, row_labels=row_labels,
        fontsize_pt=fontsize_pt, column_format=column_format, escape=escape, dpi=dpi,
    )
    image = plt.imread(png_path)
    ax.imshow(image)
    ax.set_axis_off()
    return image


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
    "axes.titlepad": 10.0,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "lines.linewidth": 1.5,
}


# Okabe & Ito (2008) colorblind-safe categorical palette -- eight colors
# designed by two Japanese vision scientists specifically to remain
# distinguishable under the common forms of color vision deficiency
# (protanopia, deuteranopia, tritanopia), while still reading well for
# non-colorblind viewers. Order matters: this is the sequence the authors
# recommend introducing colors in as series count grows.
OKABE_ITO_PALETTE = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# A figure that needs SEVERAL colormaps at once (e.g. one sequential panel
# per quantity in a small-multiples grid, or several diverging quantities
# shown side by side) should not just reuse the single house default for
# every panel -- that erases the fact that they're different quantities.
# Instead, pick a distinct map per panel from within its category, in this
# fixed order, so repeated use across a project stays consistent (panel 1
# always gets the same map as last time, not a random pick).
MULTI_CMAP_SHORTLIST: dict[str, tuple[str, ...]] = {
    "sequential": (
        "crameri:lipari",   # house default -- reserve for the "primary" quantity
        "crameri:batlow",
        "cmocean:thermal",
        "crameri:acton",
        "cmocean:matter",
    ),
    "diverging": (
        "contrib:pampa",    # house default -- reserve for the "primary" quantity
        "crameri:vik",      # red/blue -- charge- or sign-like quantities
        "cmocean:balance",
        "crameri:broc",
        "colorbrewer:PuOr_11",
    ),
    "cyclic": (
        "cmocean:phase",    # house default
        "crameri:romaO",
        "crameri:vikO",
        "matplotlib:twilight",
    ),
}

# Line style and marker cycles -- assigned by the *role* a series plays
# (e.g. "this is the same run, alternate replicate" vs. "this is a
# different quantity entirely"), never by index alone. See "Line styles,
# markers, and symbols" in docs/plotting-style-guide.md.
LINE_STYLE_CYCLE: tuple[str, ...] = ("-", "--", "-.", ":")
MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


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
        "ICML/NeurIPS-paper vibe: clean sans-serif, legend meant to live "
        "outside the axes (see legend_outside()). Categorical color cycle is "
        "the Okabe-Ito colorblind-safe palette (OKABE_ITO_PALETTE) -- the "
        "house default, not an opt-in variant.",
        colors=_ICML_COLORS, rc_params=_ICML_RC,
        comparison_palette=OKABE_ITO_PALETTE,
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
    "icml_okabe_ito": "icml",  # Okabe-Ito is now icml's own default categorical cycle
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
