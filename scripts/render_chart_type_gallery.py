#!/usr/bin/env python3
"""Render a range of chart *types* (not just fonts) under the house style,
so a form can be picked by eye alongside the font choice in
docs/plot-style-gallery.md. Each one calls out the specific Tufte principle
it's chosen to demonstrate -- see docs/plotting-style-guide.md.

All render under "icml" (the current default) by construction; re-running
with STYLE_NAME changed at the top will re-render the whole set under a
different preset for comparison.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from mmml.utils.plotting.styles import apply_plot_style, legend_outside, seed_symbol

STYLE_NAME = "icml"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(0)

_SYSTEM_COLORS = {"water_box": "#1A5276", "peptide_water": "#943126"}


def _load_plot_utils():
    """Import scripts/plot_utils.py::render_dimer_atoms by path (not a package)."""
    spec = importlib.util.spec_from_file_location("plot_utils", REPO_ROOT / "scripts" / "plot_utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _water_dimer():
    """A small hand-placed water dimer (2 x H2O) as an ASE Atoms for rendering demos."""
    from ase import Atoms

    numbers = [8, 1, 1, 8, 1, 1]
    positions = [
        [0.00, 0.00, 0.00], [0.76, 0.59, 0.00], [-0.76, 0.59, 0.00],
        [0.00, 2.90, 0.10], [0.60, 3.50, -0.30], [-0.60, 3.30, 0.55],
    ]
    fragments = (np.array([0, 1, 2]), np.array([3, 4, 5]))
    return Atoms(numbers=numbers, positions=positions), fragments


def _mpl_cmap(name: str):
    """Resolve a colormap name through the `cmap` library when available
    (cmocean/crameri/colorbrewer/etc.), falling back to matplotlib's own
    registry for plain names -- see docs/plotting-style-guide.md "Colormaps"."""
    try:
        import cmap as cmap_lib

        return cmap_lib.Colormap(name).to_mpl()
    except ImportError:
        return name


def radial_plot(out: Path) -> None:
    """Circular histogram of a periodic quantity (dihedral angle).

    Tufte principle: use the geometry the *data* has (an angle is inherently
    circular) rather than forcing it into a linear 0-360 axis, which hides
    the wraparound at the boundary.
    """
    dihedrals_deg = np.concatenate([
        RNG.normal(60, 15, 200), RNG.normal(180, 20, 300), RNG.normal(300, 15, 150),
    ]) % 360
    bins = np.linspace(0, 2 * np.pi, 37)
    counts, _ = np.histogram(np.deg2rad(dihedrals_deg), bins=bins)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.bar(bins[:-1], counts, width=(2 * np.pi / 36), color="#1E8449", alpha=0.85,
           edgecolor="white", linewidth=0.5, align="edge")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Dihedral angle distribution (circular histogram)", pad=20)
    ax.set_rlabel_position(135)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def surface_3d(out: Path) -> None:
    """3D PES-style surface: energy as a function of two geometric coordinates.

    Tufte principle used loosely here -- 3D is data-ink-*expensive* (occlusion,
    projection distortion) and Tufte generally preferred 2D + color/contour
    for exactly this reason. Included because it was asked for; the
    `range_frame` and `matshow` panels are the better-Tufte alternative for
    the same 2-variable-vs-response data (see below).
    """
    x = np.linspace(2.5, 5.5, 40)
    y = np.linspace(-2.0, 2.0, 40)
    xx, yy = np.meshgrid(x, y)
    zz = 4.0 * ((2.5 / xx) ** 12 - (2.5 / xx) ** 6) + 0.3 * yy**2

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("lateral offset (Å)")
    ax.set_zlabel("energy (kcal/mol)")
    ax.set_title("Dimer PES surface (3D)")
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="energy (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_with_ci(out: Path) -> None:
    """XY scatter with a fitted trend and shaded confidence band.

    Tufte principle: the band *is* the uncertainty -- no error-bar clutter
    on every point, and the fit line carries the "so what" (the trend) while
    the raw points stay visible underneath it, not replaced by it.
    """
    checkpoint_epoch = np.array([3, 4, 5, 10])
    fluct = np.array([0.31, 0.28, 3.9, 0.32]) + RNG.normal(0, 0.05, 4)
    # a few repeats per epoch for a believable scatter + CI
    x = np.repeat(checkpoint_epoch, 6) + RNG.normal(0, 0.08, 24)
    y = np.repeat(fluct, 6) + RNG.normal(0, 0.15, 24)

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    coeffs = np.polyfit(x_sorted, y_sorted, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    residual_std = np.std(y_sorted - np.polyval(coeffs, x_sorted))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.fill_between(x_fit, y_fit - 1.96 * residual_std, y_fit + 1.96 * residual_std,
                     color="#4C72B0", alpha=0.15, linewidth=0, label="95% CI")
    ax.plot(x_fit, y_fit, color="#4C72B0", linewidth=2.6, label="linear fit")
    ax.scatter(x, y, color="#DD8452", s=50, alpha=0.85, edgecolor="#222222", linewidth=0.6, label="checkpoints")
    ax.set_xlabel("checkpoint epoch")
    ax.set_ylabel(r"fluctuation $\sigma$ (eV)")
    ax.set_title("Checkpoint epoch vs. energy fluctuation")
    legend_outside(ax, side="right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def timeseries_with_marginals(out: Path) -> None:
    """Time series flanked by two histograms sharing its y-axis.

    Not seaborn's jointplot (marginals top+right sharing the *x*-axis with
    the scatter) -- here both extra panels sit above and below the middle
    panel and share its y-axis instead, showing how the value distribution
    itself shifts between the first and second half of the run. A form of
    Tufte's small-multiples: two related distributions placed for direct
    visual comparison rather than described in text.
    """
    frames = np.arange(200)
    energy = -75.0 + 0.15 * np.sin(frames / 12) + RNG.normal(0, 0.03, 200)
    energy += np.linspace(0, -0.05, 200)  # slight drift, first half vs second half differ a bit
    first_half, second_half = energy[:100], energy[100:]

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(8, 8), sharey=False,
        gridspec_kw={"height_ratios": [1, 3, 1]},
    )
    bins = np.linspace(energy.min(), energy.max(), 20)

    ax_top.hist(first_half, bins=bins, orientation="horizontal", color="#1A5276", alpha=0.8)
    ax_top.invert_yaxis()  # bars point "up", away from the middle panel
    ax_top.set_xlabel("count (first half)")
    ax_top.set_ylim(energy.min(), energy.max())
    ax_top.set_title("Energy trace with value-distribution margins (first half above, second half below)")

    ax_mid.plot(frames, energy, color="#333333", linewidth=2.0)
    ax_mid.axvline(100, color="#943126", linestyle=":", linewidth=1.5, alpha=0.7)
    ax_mid.set_ylabel("energy (eV)")
    ax_mid.set_xlabel("recorded frame")
    ax_mid.set_ylim(energy.min(), energy.max())

    ax_bot.hist(second_half, bins=bins, orientation="horizontal", color="#943126", alpha=0.8)
    ax_bot.set_xlabel("count (second half)")
    ax_bot.set_ylim(energy.min(), energy.max())

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def lollipop(out: Path) -> None:
    """Lollipop chart: a thin stem + a dot, instead of a filled bar.

    Tufte principle directly: the stem is nearly all the ink a bar has minus
    the fill, so this is a strictly higher data-ink ratio for the same
    information (position along the stem = value).
    """
    labels = ["mixed_baseline", "mixed_older_epoch", "mixed_damping_sigma",
              "water_baseline", "water_tight_cutoffs", "water_loose_cutoffs"]
    values = np.array([0.31, 0.62, 0.35, 0.10, 0.14, 0.16])
    colors = [_SYSTEM_COLORS["peptide_water"]] * 3 + [_SYSTEM_COLORS["water_box"]] * 3
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hlines(y, 0, values, color=colors, linewidth=2.5, alpha=0.8)
    ax.scatter(values, y, color=colors, s=140, zorder=3, edgecolor="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"fluctuation $\sigma$ (eV)")
    ax.set_title("Lollipop chart (higher data-ink ratio than filled bars)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def matshow_heatmap(out: Path) -> None:
    """Matrix heatmap: pairwise atom-atom distance matrix for a small system.

    Tufte principle: color encodes magnitude directly on the natural (i, j)
    grid the data already has -- no need to invent x/y positions the way a
    scatter would. Distance is strictly positive (no natural zero to
    diverge around), so this uses the house **sequential** default
    (`crameri:lipari`), not a diverging map -- see
    docs/plotting-style-guide.md "Colormaps" for why the earlier version of
    this exact panel (diverging `RdBu_r` on positive-only distances) was a
    real mismatch, not just a style choice.
    """
    from mmml.utils.plotting.styles import default_cmap

    n = 24
    rng_positions = RNG.uniform(0, 10, size=(n, 3))
    diff = rng_positions[:, None, :] - rng_positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.matshow(dist, cmap=default_cmap("sequential"), vmin=0, vmax=dist.max())
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("atom index")
    ax.set_ylabel("atom index")
    ax.set_title("Pairwise atom-distance matrix")
    fig.colorbar(im, ax=ax, label="distance (Å)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def small_multiples(out: Path) -> None:
    """Grid of small, axis-stripped sparklines -- Tufte's signature form.

    Tufte principle by name: many small, identically-scaled panels let the
    eye compare shape across settings directly, with almost all chart
    furniture (ticks, boxes, labels) removed since the *shape* is the point,
    not reading exact values off any one panel.
    """
    settings = ["water_baseline", "water_tight_cutoffs", "water_loose_cutoffs",
                "mixed_baseline", "mixed_older_epoch", "mixed_damping_sigma"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.5))
    for ax, name in zip(axes.flat, settings):
        frames = np.arange(100)
        trend = RNG.normal(0, 0.002) * frames
        noise = RNG.normal(0, 0.1 if "mixed" in name else 0.03, 100)
        trace = trend + noise
        color = _SYSTEM_COLORS["peptide_water"] if "mixed" in name else _SYSTEM_COLORS["water_box"]
        ax.plot(frames, trace, color=color, linewidth=1.8)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Small multiples: energy-trace shape, compared directly")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def range_frame_scatter(out: Path) -> None:
    """Tufte's own "range-frame": axis lines span only the data's actual
    range (not an arbitrary origin-anchored box), with ticks only at the
    min/median/max. This is Tufte's most literal, named contribution to
    plot design -- worth including explicitly since it's not a generic
    "minimalist" choice but a specific technique from *The Visual Display of
    Quantitative Information*.
    """
    x = RNG.normal(10.0, 2.0, 60)
    y = 2.0 * x + RNG.normal(0, 3.0, 60)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, color="#4C72B0", s=45, alpha=0.8, edgecolor="#222222", linewidth=0.5)

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    # Range-frame: bottom/left spines only span [min, max] of the data, not
    # the full axes box.
    ax.spines["bottom"].set_bounds(x.min(), x.max())
    ax.spines["left"].set_bounds(y.min(), y.max())
    ax.set_xticks([x.min(), np.median(x), x.max()])
    ax.set_yticks([y.min(), np.median(y), y.max()])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title('Tufte "range-frame": axes span only the data, ticks at min/median/max')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def energy_term_diagram(out: Path) -> None:
    """A simple schematic diagram (boxes + arrows), not a data plot -- for
    showing composition/flow (e.g. how HybridEnergy sums its terms).

    Tufte principle: even a diagram should minimize non-data ink -- thin
    connecting lines, no drop shadows/gradients/3D bevels, labels placed
    directly on the elements they describe rather than in a legend.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    boxes = [
        ("ml_intra", 0.05, "#1A5276"),
        ("ml_pep_water", 0.05, "#1A5276"),
        ("vdw_core", 0.05, "#5D6D7E"),
        ("mm_nonbonded", 0.05, "#943126"),
    ]
    y_positions = np.linspace(0.85, 0.15, len(boxes))
    for (label, _, color), y in zip(boxes, y_positions):
        ax.add_patch(plt.Rectangle((0.05, y - 0.06), 0.35, 0.12, facecolor=color,
                                    alpha=0.85, edgecolor="#222222", linewidth=1.0))
        ax.text(0.225, y, label, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.annotate("", xy=(0.55, 0.5), xytext=(0.40, y),
                    arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 1.2})
    ax.add_patch(plt.Rectangle((0.55, 0.38), 0.4, 0.24, facecolor="white",
                                edgecolor="#222222", linewidth=1.5))
    ax.text(0.75, 0.5, "HybridEnergy\n(sum)", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Energy term composition (schematic diagram)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ase_atoms_overlay(out: Path) -> None:
    """ASE Atoms structure rendered as an inset *on top of* a data plot.

    Reuses `scripts/plot_utils.py::render_dimer_atoms` (the good precedent
    from the SpookyNet dimer-scan figures) rather than re-deriving atom
    rendering -- see docs/plotting-style-guide.md "Rendering ASE Atoms".
    Tufte principle: put the structure where the eye already is (right next
    to the energy value it corresponds to) instead of a separate figure the
    reader has to cross-reference by hand.
    """
    plot_utils = _load_plot_utils()
    atoms, fragments = _water_dimer()

    r = np.linspace(2.0, 6.0, 60)
    energy = 4.0 * ((2.9 / r) ** 12 - (2.9 / r) ** 6)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r, energy, color="#1A5276", linewidth=2.8)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("O···O distance (Å)")
    ax.set_ylabel("interaction energy (kcal/mol)")
    ax.set_title("Water-dimer scan with structure overlay")

    # Inset axes hold the ASE-atoms render directly over the minimum, where
    # the reader's eye already is -- not a separate side-by-side figure.
    r_min = r[np.argmin(energy)]
    e_min = energy.min()
    inset = ax.inset_axes([0.58, 0.55, 0.38, 0.38])
    plot_utils.render_dimer_atoms(inset, atoms, fragments, rotation="15x,10y,0z")
    ax.annotate("", xy=(r_min, e_min), xytext=(r_min + 0.9, e_min + 2.5),
                arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 1.2})

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def colormap_gallery(out: Path) -> None:
    """The same 2D field rendered under several colormaps from the `cmap`
    library (cmocean/crameri/colorbrewer/...), for direct comparison.

    Tufte principle: color should encode magnitude *faithfully* -- several
    of these (crameri's `batlow`, cmocean's `haline`) are specifically
    designed to be perceptually uniform and colorblind-safe, unlike a
    rainbow map that implies false discontinuities in the data.
    """
    x = np.linspace(2.5, 6.0, 80)
    y = np.linspace(-2.5, 2.5, 80)
    xx, yy = np.meshgrid(x, y)
    zz = 4.0 * ((2.9 / xx) ** 12 - (2.9 / xx) ** 6) + 0.2 * yy**2

    names = [
        ("viridis", "viridis (matplotlib default, perceptually uniform)"),
        ("cmocean:haline", "cmocean:haline (oceanographic sequential)"),
        ("cmocean:balance", "cmocean:balance (diverging, zero-centered)"),
        ("crameri:batlow", "crameri:batlow (colorblind-safe sequential)"),
        ("crameri:vik", "crameri:vik (colorblind-safe diverging)"),
        ("colorbrewer:RdYlBu_11", "colorbrewer:RdYlBu (classic diverging)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, label) in zip(axes.flat, names):
        im = ax.pcolormesh(xx, yy, zz, cmap=_mpl_cmap(name), shading="gouraud")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Same field, different colormaps (via the cmap library)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def hist2d(out: Path) -> None:
    """2D histogram of two correlated variables (bond length vs. angle).

    Tufte principle: for many overlapping points, binned density (not a
    scatter of thousands of translucent dots) is the honest representation
    -- a scatter would just be a saturated blob past a few hundred points.
    """
    n = 20000
    bond = RNG.normal(0.96, 0.02, n) + 0.01 * RNG.standard_normal(n)
    angle = 104.5 + 15 * (bond - 0.96) / 0.02 + RNG.normal(0, 3.5, n)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    h = ax.hist2d(bond, angle, bins=60, cmap=_mpl_cmap("cmocean:dense"))
    fig.colorbar(h[3], ax=ax, label="count")
    ax.set_xlabel("O–H bond length (Å)")
    ax.set_ylabel("H–O–H angle (°)")
    ax.set_title("2D histogram: bond length vs. angle")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def histogram_timeseries(out: Path) -> None:
    """A time-resolved distribution: x = frame, y = value bin, color =
    local density (a "kymograph" of the trace's own histogram over time).

    Tufte principle: this is the small-multiples idea taken to its limit --
    instead of N separate histograms side by side (one per time window),
    stack them as columns of one image, so a drift in the *distribution*
    (not just the mean) is visible as a single continuous shape rather than
    something you'd have to notice across many separate panels.
    """
    n_frames, n_windows = 2000, 80
    frames = np.arange(n_frames)
    energy = -75.0 + 0.3 * np.sin(frames / 80) + RNG.normal(0, 0.05, n_frames)
    energy += np.linspace(0, 0.15, n_frames)  # slow drift, on top of the oscillation

    bins = np.linspace(energy.min(), energy.max(), 50)
    window_edges = np.linspace(0, n_frames, n_windows + 1).astype(int)
    density = np.zeros((len(bins) - 1, n_windows))
    for i in range(n_windows):
        chunk = energy[window_edges[i]:window_edges[i + 1]]
        density[:, i], _ = np.histogram(chunk, bins=bins, density=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.pcolormesh(window_edges[:-1], bins[:-1], density, cmap=_mpl_cmap("cmocean:thermal"),
                        shading="nearest")
    fig.colorbar(im, ax=ax, label="local probability density")
    ax.set_xlabel("recorded frame")
    ax.set_ylabel("energy (eV)")
    ax.set_title("Histogram time series: how the value distribution drifts")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def jitter_strip(out: Path) -> None:
    """A jitter/strip plot: every individual observation as one point, x
    position randomly jittered within its category so overlapping points
    stay visible instead of stacking into a solid blob.

    Tufte/data-ink principle: no summary statistic is drawn on top of
    anything -- unlike a box or bar chart, nothing here is a derived
    quantity. When N is small-to-moderate, showing the raw data itself is
    more honest and often more informative than any summary of it (e.g. a
    bimodal or skewed distribution is invisible in a mean+errorbar but
    obvious in a strip plot).
    """
    groups = ["water_box", "peptide_water", "vacuum"]
    group_colors = [_SYSTEM_COLORS.get(g, "#6C3483") for g in groups]
    data = {
        "water_box": RNG.normal(-74.8, 0.35, 60),
        "peptide_water": RNG.normal(-71.2, 0.55, 60),
        "vacuum": np.concatenate([RNG.normal(-69.0, 0.2, 30), RNG.normal(-67.5, 0.25, 30)]),
    }

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for i, (group, color) in enumerate(zip(groups, group_colors)):
        values = data[group]
        x = i + RNG.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(x, values, s=22, color=color, alpha=0.65, edgecolor="none")
        ax.hlines(np.median(values), i - 0.28, i + 0.28, color="#222222", linewidth=2.2, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Jitter/strip plot: raw per-run values, median as a short bar")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def violin_comparison(out: Path) -> None:
    """A violin plot: the same three groups as the jitter plot, but showing
    the estimated full distribution shape (a mirrored, smoothed histogram)
    rather than individual points -- complementary to jitter/strip, better
    when N is large enough that individual points would overplot.
    """
    groups = ["water_box", "peptide_water", "vacuum"]
    group_colors = [_SYSTEM_COLORS.get(g, "#6C3483") for g in groups]
    data = [
        RNG.normal(-74.8, 0.35, 400),
        RNG.normal(-71.2, 0.55, 400),
        np.concatenate([RNG.normal(-69.0, 0.2, 200), RNG.normal(-67.5, 0.25, 200)]),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], group_colors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#222222")
        parts[key].set_linewidth(1.2)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_title("Violin plot: full distribution shape (bimodal vacuum group visible)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)
    renders = {
        "chart_radial": radial_plot,
        "chart_3d_surface": surface_3d,
        "chart_scatter_ci": scatter_with_ci,
        "chart_timeseries_marginals": timeseries_with_marginals,
        "chart_lollipop": lollipop,
        "chart_matshow": matshow_heatmap,
        "chart_small_multiples": small_multiples,
        "chart_range_frame": range_frame_scatter,
        "chart_diagram": energy_term_diagram,
        "chart_ase_overlay": ase_atoms_overlay,
        "chart_colormaps": colormap_gallery,
        "chart_hist2d": hist2d,
        "chart_histogram_timeseries": histogram_timeseries,
        "chart_jitter_strip": jitter_strip,
        "chart_violin": violin_comparison,
    }
    for name, fn in renders.items():
        out = OUT_DIR / f"{name}.png"
        fn(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
