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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from mmml.utils.plotting.styles import apply_plot_style, legend_outside, seed_symbol

STYLE_NAME = "icml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
RNG = np.random.default_rng(0)

_SYSTEM_COLORS = {"water_box": "#1A5276", "peptide_water": "#943126"}


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
    scatter would.
    """
    n = 24
    rng_positions = RNG.uniform(0, 10, size=(n, 3))
    diff = rng_positions[:, None, :] - rng_positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.matshow(dist, cmap="RdBu_r", vmin=0, vmax=dist.max())
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
    }
    for name, fn in renders.items():
        out = OUT_DIR / f"{name}.png"
        fn(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
