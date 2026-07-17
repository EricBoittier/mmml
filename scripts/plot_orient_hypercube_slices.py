#!/usr/bin/env python3
"""Surfaces / 3D views / hypercube-slice figures for dimer orientation scans.

The rigid-dimer configuration space is 6D: separation ``r``, approach direction
on ``S^2`` (2), and monomer-B orientation on ``SO(3)`` (3).
``scripts/scan_dimer_orientations.py`` samples that product with low-discrepancy
sets (Fibonacci sphere × super-Fibonacci quaternions) and collapses each ray
to ``(e_min, r_at_min, n_min_ml)``.

This script turns those ray tables (and optional xTB validate CSVs) into:

1. Well-depth heatmaps with spurious-minimum markers
2. 3D surfaces of well depth over the (direction × orientation) face
3. Spurious-fraction vs energy-threshold curves
4. Several *intersecting hypercube-slice* representations of the discrete
   product space (and of the low-discrepancy parameter cube)
5. ML vs xTB ray overlays / depth scatter when ``--validate`` is given

House style: ``docs/plotting-style-guide.md`` /
``mmml.utils.plotting.styles`` (``icml`` + ``default_cmap`` + ``STATUS_COLORS``).

Example::

    uv run python scripts/plot_orient_hypercube_slices.py \\
        --orient-6A /Volumes/PortableSSD/DATA/acodcm/orient_6A \\
        --orient-8A /Volumes/PortableSSD/DATA/acodcm/orient_8A \\
        --validate /Volumes/PortableSSD/DATA/acodcm/validate_ACO/rays_ACO.csv \\
        --out /Volumes/PortableSSD/DATA/acodcm/orient_plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mmml.utils.plotting.styles import (
    STATUS_COLORS,
    apply_plot_style,
    comparison_colors,
    default_cmap,
    legend_outside,
    status_color,
)

EV_TO_KCAL = 23.0605


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform directions on S^2 (matches scan_dimer_orientations)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def super_fibonacci(n: int) -> np.ndarray:
    """``n`` near-uniform unit quaternions on SO(3) (Alexa, CVPR 2022)."""
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i = np.arange(n) + 0.5
    s = i / n
    t = s * n / phi
    d = 2.0 * np.pi * (t - np.floor(t))
    r = np.sqrt(s)
    R = np.sqrt(1.0 - s)
    t2 = i / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))
    return np.stack([r * np.sin(d), r * np.cos(d), R * np.sin(a), R * np.cos(a)], axis=1)


def _load_rays(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"empty rays table: {path}")
    keys = (
        "ray",
        "direction",
        "orientation",
        "n_min",
        "n_min_ml",
        "n_min_wall",
        "e_min_kcal",
        "r_at_min",
    )
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.asarray([float(r[k]) for r in rows])
    for k in ("ray", "direction", "orientation", "n_min", "n_min_ml", "n_min_wall"):
        out[k] = out[k].astype(int)
    return out


def _load_validate(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    keys = ("ray", "direction", "orientation", "r_com", "E_ml", "E_charmm", "E_xtb")
    out: dict[str, np.ndarray] = {}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k, "")
            vals.append(np.nan if v in ("", "nan", "NaN", None) else float(v))
        out[k] = np.asarray(vals)
    for k in ("ray", "direction", "orientation"):
        out[k] = out[k].astype(int)
    return out


def _grid(
    rays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dirs = np.sort(np.unique(rays["direction"]))
    oris = np.sort(np.unique(rays["orientation"]))
    e = np.full((len(dirs), len(oris)), np.nan)
    n_ml = np.full((len(dirs), len(oris)), 0, dtype=int)
    rmin = np.full((len(dirs), len(oris)), np.nan)
    d_to_i = {int(d): i for i, d in enumerate(dirs)}
    o_to_i = {int(o): i for i, o in enumerate(oris)}
    for d, o, ee, nn, rr in zip(
        rays["direction"],
        rays["orientation"],
        rays["e_min_kcal"],
        rays["n_min_ml"],
        rays["r_at_min"],
        strict=True,
    ):
        i, j = d_to_i[int(d)], o_to_i[int(o)]
        e[i, j] = ee
        n_ml[i, j] = nn
        rmin[i, j] = rr
    return dirs, oris, e, n_ml, rmin


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _energy_norm(e: np.ndarray) -> mcolors.TwoSlopeNorm | mcolors.Normalize:
    finite = e[np.isfinite(e)]
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin < 0.0 < vmax:
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Core panels
# ---------------------------------------------------------------------------


def plot_heatmap(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
) -> None:
    dirs, oris, e, n_ml, _ = _grid(rays)
    spur = n_ml > 1
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    im = ax.imshow(
        e,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=_energy_norm(e),
        interpolation="nearest",
        extent=(-0.5, len(oris) - 0.5, -0.5, len(dirs) - 0.5),
    )
    yy, xx = np.where(spur)
    ax.scatter(
        xx,
        yy,
        s=22,
        c=status_color("critical"),
        marker="o",
        linewidths=0.4,
        edgecolors="white",
        zorder=3,
        label="spurious (>1 ML min)",
    )
    ax.set_xlabel("orientation index (super-Fibonacci SO(3))")
    ax.set_ylabel("direction index (Fibonacci S²)")
    ax.set_title(title)
    ax.set_xticks(range(0, len(oris), 2))
    ax.set_yticks(list(dirs))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="well depth (kcal/mol)")
    legend_outside(ax, side="right")
    _save(fig, out)


def plot_surface_3d(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
) -> None:
    dirs, oris, e, n_ml, _ = _grid(rays)
    oo, dd = np.meshgrid(oris, dirs)
    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_subplot(projection="3d")
    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(e)
    surf = ax.plot_surface(
        oo,
        dd,
        e,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    yy, xx = np.where(n_ml > 1)
    if len(xx):
        ax.scatter(
            oris[xx],
            dirs[yy],
            e[yy, xx],
            c=status_color("critical"),
            s=18,
            depthshade=False,
            label="spurious",
        )
    ax.set_xlabel("orientation index")
    ax.set_ylabel("direction index")
    ax.set_zlabel("well depth (kcal/mol)")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="kcal/mol")
    _save(fig, out)


def plot_spurious_vs_threshold(
    series: list[tuple[str, dict[str, np.ndarray]]],
    out: Path,
) -> None:
    """Fraction of rays with e_min < −τ (and separately n_min_ml>1 among those)."""
    style = apply_plot_style("icml")  # already applied in main; refresh palette
    colors = comparison_colors(style, n=len(series))
    thresholds = np.linspace(0.0, 8.0, 81)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for (label, rays), color in zip(series, colors, strict=True):
        e = rays["e_min_kcal"]
        spur = rays["n_min_ml"] > 1
        frac_spur = np.array([(spur & (e < -t)).mean() for t in thresholds])
        frac_deep = np.array([(e < -t).mean() for t in thresholds])
        axes[0].plot(thresholds, frac_spur, color=color, label=label)
        axes[1].plot(thresholds, frac_deep, color=color, label=label)

    axes[0].set_ylabel("fraction of rays")
    axes[0].set_title("spurious AND deeper than −τ")
    axes[1].set_title("any well deeper than −τ")
    for ax in axes:
        ax.set_xlabel("depth threshold τ (kcal/mol)")
        ax.set_ylim(0.0, 1.05)
        ax.axvline(0.6, color=STATUS_COLORS["neutral"], ls=":", lw=1.0)  # ~kT
    legend_outside(axes[1], side="right")
    fig.suptitle("Sensitivity of the orientation-scan verdict to depth threshold", y=1.02)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Hypercube-slice representations
# ---------------------------------------------------------------------------


def plot_intersecting_index_faces(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
    dir_slice: int,
    ori_slice: int,
) -> None:
    """Rep A — intersecting faces of the discrete (dir × ori × E) prism.

    The (direction × orientation) face is drawn as a translucent surface.
    One fixed-direction row and one fixed-orientation column are drawn as
    thick polylines that meet at a shared cell — the 1D slices of the
    hypercube that intersect there.
    """
    dirs, oris, e, n_ml, _ = _grid(rays)
    if dir_slice not in set(dirs) or ori_slice not in set(oris):
        raise SystemExit(f"slice ({dir_slice},{ori_slice}) outside grid")
    oo, dd = np.meshgrid(oris.astype(float), dirs.astype(float))
    fig = plt.figure(figsize=(7.8, 6.0))
    ax = fig.add_subplot(projection="3d")
    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    ax.plot_surface(
        oo,
        dd,
        e,
        cmap=cmap,
        norm=_energy_norm(e),
        linewidth=0,
        antialiased=True,
        alpha=0.35,
    )

    di = int(np.where(dirs == dir_slice)[0][0])
    oj = int(np.where(oris == ori_slice)[0][0])
    row_c = comparison_colors(apply_plot_style("icml"), n=2)[0]
    col_c = comparison_colors(apply_plot_style("icml"), n=2)[1]

    ax.plot(
        oris,
        np.full_like(oris, dir_slice, dtype=float),
        e[di, :],
        color=row_c,
        lw=2.4,
        label=f"dir={dir_slice} row",
    )
    ax.plot(
        np.full_like(dirs, ori_slice, dtype=float),
        dirs,
        e[:, oj],
        color=col_c,
        lw=2.4,
        label=f"ori={ori_slice} column",
    )
    ax.scatter(
        [ori_slice],
        [dir_slice],
        [e[di, oj]],
        c=status_color("critical" if n_ml[di, oj] > 1 else "good"),
        s=60,
        depthshade=False,
        zorder=5,
    )
    ax.set_xlabel("orientation index")
    ax.set_ylabel("direction index")
    ax.set_zlabel("well depth (kcal/mol)")
    ax.set_title(title)
    legend_outside(ax, side="right")
    _save(fig, out)


def plot_parallel_direction_slices(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
    directions: list[int] | None = None,
) -> None:
    """Rep B — parallel 1D slices (fixed direction) extruded as ribbons in 3D.

    Each ribbon is E_min(orientation) living on a plane of constant direction
    index — a loaf of parallel hypercube cuts through the (dir × ori) face.
    """
    dirs, oris, e, n_ml, _ = _grid(rays)
    if directions is None:
        # deepest mean wells first, then pad with shallow/clean rows
        order = np.argsort(np.nanmean(e, axis=1))
        directions = [int(dirs[i]) for i in order[:6]]

    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=len(directions))
    fig = plt.figure(figsize=(8.0, 5.8))
    ax = fig.add_subplot(projection="3d")

    for d, color in zip(directions, colors, strict=True):
        di = int(np.where(dirs == d)[0][0])
        z = e[di, :]
        # ribbon: a thin strip in the direction-index axis
        y0, y1 = d - 0.35, d + 0.35
        verts = []
        for j in range(len(oris) - 1):
            verts.append(
                [
                    (oris[j], y0, z[j]),
                    (oris[j + 1], y0, z[j + 1]),
                    (oris[j + 1], y1, z[j + 1]),
                    (oris[j], y1, z[j]),
                ]
            )
        poly = Poly3DCollection(
            verts,
            facecolors=color,
            edgecolors=color,
            linewidths=0.2,
            alpha=0.55,
        )
        ax.add_collection3d(poly)
        spur = n_ml[di] > 1
        if spur.any():
            ax.scatter(
                oris[spur],
                np.full(spur.sum(), d),
                z[spur],
                c=status_color("critical"),
                s=14,
                depthshade=False,
            )
        ax.plot([], [], [], color=color, lw=3, label=f"dir {d}")

    ax.set_xlabel("orientation index")
    ax.set_ylabel("direction index")
    ax.set_zlabel("well depth (kcal/mol)")
    ax.set_xlim(oris.min() - 0.5, oris.max() + 0.5)
    ax.set_ylim(min(directions) - 1, max(directions) + 1)
    zmin, zmax = float(np.nanmin(e)), float(np.nanmax(e))
    ax.set_zlim(zmin - 0.3, zmax + 0.3)
    ax.set_title(title)
    legend_outside(ax, side="right")
    _save(fig, out)


def plot_pairwise_cube_faces(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
) -> None:
    """Rep C — three pairwise faces of the (dir, ori, r_at_min) index cube.

    Colour = well depth; red edge markers = spurious. Shared axes make the
    intersections of these faces readable as a cube-net.
    """
    dirs, oris, e, n_ml, rmin = _grid(rays)
    spur = n_ml > 1
    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(e)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    # face 1: dir × ori (already the main heatmap, kept for the net)
    im0 = axes[0].imshow(
        e,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    yy, xx = np.where(spur)
    axes[0].scatter(xx, yy, s=10, c=status_color("critical"), linewidths=0)
    axes[0].set_xlabel("orientation")
    axes[0].set_ylabel("direction")
    axes[0].set_title("face: dir × ori")

    # face 2: ori × r_at_min (scatter / hex-like via binned image proxy)
    axes[1].scatter(
        rays["orientation"],
        rays["r_at_min"],
        c=rays["e_min_kcal"],
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="none",
        alpha=0.9,
    )
    bad = rays["n_min_ml"] > 1
    axes[1].scatter(
        rays["orientation"][bad],
        rays["r_at_min"][bad],
        facecolors="none",
        edgecolors=status_color("critical"),
        s=36,
        linewidths=0.9,
    )
    axes[1].set_xlabel("orientation")
    axes[1].set_ylabel("r at minimum (Å)")
    axes[1].set_title("face: ori × r_min")

    axes[2].scatter(
        rays["direction"],
        rays["r_at_min"],
        c=rays["e_min_kcal"],
        cmap=cmap,
        norm=norm,
        s=28,
        edgecolors="none",
        alpha=0.9,
    )
    axes[2].scatter(
        rays["direction"][bad],
        rays["r_at_min"][bad],
        facecolors="none",
        edgecolors=status_color("critical"),
        s=36,
        linewidths=0.9,
    )
    axes[2].set_xlabel("direction")
    axes[2].set_ylabel("r at minimum (Å)")
    axes[2].set_title("face: dir × r_min")

    fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="kcal/mol")
    fig.suptitle(title, y=1.03)
    _save(fig, out)


def plot_parameter_hypercube_embedding(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
    n_directions: int = 10,
    n_orientations: int = 24,
) -> None:
    """Rep D — embed samples in the low-discrepancy parameter cube.

    Directions → (θ, φ) from the Fibonacci construction; orientations →
    (s, α) from super-Fibonacci. Plot three intersecting 3D projections of
    that 4D parameter box, coloured by well depth. This is the continuous
    hypercube the discrete indices live inside — not the index grid itself.
    """
    dirs_xyz = fibonacci_sphere(n_directions)
    quats = super_fibonacci(n_orientations)
    # Fibonacci polar angles (same construction as fibonacci_sphere)
    i_d = np.arange(n_directions) + 0.5
    phi = np.arccos(1.0 - 2.0 * i_d / n_directions)
    theta = np.pi * (1.0 + 5.0**0.5) * i_d
    # super-Fibonacci (s, a)
    phi_sf = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    i_o = np.arange(n_orientations) + 0.5
    s = i_o / n_orientations
    t2 = i_o / psi
    a = 2.0 * np.pi * (t2 - np.floor(t2))

    d_idx = rays["direction"]
    o_idx = rays["orientation"]
    e = rays["e_min_kcal"]
    spur = rays["n_min_ml"] > 1

    x_theta = theta[d_idx]
    x_phi = phi[d_idx]
    x_s = s[o_idx]
    x_a = a[o_idx]

    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(e)

    fig = plt.figure(figsize=(12.5, 4.2))
    panels = [
        (x_theta, x_phi, x_s, "θ (rad)", "φ (rad)", "s (SO(3))"),
        (x_theta, x_a, e, "θ (rad)", "α (rad)", "well depth (kcal/mol)"),
        (x_s, x_a, e, "s (SO(3))", "α (rad)", "well depth (kcal/mol)"),
    ]
    # First panel uses s as z (geometry embedding); others use energy as z.
    for k, (xs, ys, zs, xl, yl, zl) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        sc = ax.scatter(
            xs,
            ys,
            zs if k == 0 else e,
            c=e,
            cmap=cmap,
            norm=norm,
            s=22,
            depthshade=False,
            alpha=0.9,
        )
        if spur.any():
            ax.scatter(
                xs[spur],
                ys[spur],
                (zs if k == 0 else e)[spur],
                facecolors="none",
                edgecolors=status_color("critical"),
                s=40,
                linewidths=0.8,
                depthshade=False,
            )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_zlabel(zl if k == 0 else "well depth (kcal/mol)")
    fig.colorbar(sc, ax=fig.axes, fraction=0.015, pad=0.04, label="kcal/mol")
    fig.suptitle(title, y=1.02)
    # silence unused
    _ = (dirs_xyz, quats, x_phi, phi_sf)
    _save(fig, out)


def plot_minima_cloud_for_direction(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    title: str,
    direction: int,
) -> None:
    """Proxy E(r, orientation) surface for one direction row.

    Full E(r) grids were not persisted — only (r_at_min, e_min). Plot those
    minima as a 3D cloud / stem plot so neighbouring orientations' double-well
    behaviour is visible as jumps in r_at_min and depth.
    """
    sel = rays["direction"] == direction
    ori = rays["orientation"][sel]
    r = rays["r_at_min"][sel]
    e = rays["e_min_kcal"][sel]
    spur = rays["n_min_ml"][sel] > 1
    order = np.argsort(ori)
    ori, r, e, spur = ori[order], r[order], e[order], spur[order]

    fig = plt.figure(figsize=(8.0, 5.6))
    ax = fig.add_subplot(projection="3d")
    cmap = default_cmap("diverging") if (np.nanmin(e) < 0 < np.nanmax(e)) else default_cmap(
        "sequential"
    )
    ax.plot(ori, r, e, color=STATUS_COLORS["neutral"], lw=1.0, alpha=0.7)
    ax.scatter(
        ori[~spur],
        r[~spur],
        e[~spur],
        c=e[~spur],
        cmap=cmap,
        norm=_energy_norm(e),
        s=36,
        depthshade=False,
        label="single minimum",
    )
    if spur.any():
        ax.scatter(
            ori[spur],
            r[spur],
            e[spur],
            c=status_color("critical"),
            s=50,
            depthshade=False,
            label="spurious",
        )
    # stems down to a floor so depth is readable
    z0 = float(min(0.0, np.nanmin(e) - 0.5))
    for o, rr, ee in zip(ori, r, e, strict=True):
        ax.plot([o, o], [rr, rr], [z0, ee], color=STATUS_COLORS["neutral"], lw=0.5, alpha=0.4)

    ax.set_xlabel("orientation index")
    ax.set_ylabel("r at minimum (Å)")
    ax.set_zlabel("well depth (kcal/mol)")
    ax.set_title(title)
    legend_outside(ax, side="right")
    _save(fig, out)


# ---------------------------------------------------------------------------
# Validate (xTB)
# ---------------------------------------------------------------------------


def plot_validate_curves(val: dict[str, np.ndarray], out: Path) -> None:
    rays = sorted(set(int(r) for r in val["ray"]))
    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=2)
    n = len(rays)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ray in zip(axes, rays, strict=True):
        m = val["ray"] == ray
        r = val["r_com"][m]
        order = np.argsort(r)
        r = r[order]
        e_ml = val["E_ml"][m][order] * EV_TO_KCAL
        e_xtb = val["E_xtb"][m][order] * EV_TO_KCAL
        # shift both to asymptotic zero at largest r
        e_ml = e_ml - e_ml[-1]
        e_xtb = e_xtb - e_xtb[-1]
        ax.plot(r, e_ml, color=colors[0], label="ML hybrid")
        ax.plot(r, e_xtb, color=colors[1], label="GFN2-xTB")
        d, o = int(val["direction"][m][0]), int(val["orientation"][m][0])
        ax.set_title(f"ray {ray} (dir {d}, ori {o})")
        ax.set_xlabel("r_COM (Å)")
        ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.8)
    axes[0].set_ylabel("binding energy (kcal/mol)")
    legend_outside(axes[-1], side="right")
    fig.suptitle("ML vs xTB along validated rays (shifted to E(r_max)=0)", y=1.03)
    _save(fig, out)


def plot_validate_depth_scatter(val: dict[str, np.ndarray], out: Path) -> None:
    """Per-ray well depths: ML vs xTB (anecdotes → correlation panel)."""
    rows = []
    for ray in sorted(set(int(r) for r in val["ray"])):
        m = val["ray"] == ray
        r = val["r_com"][m]
        order = np.argsort(r)
        e_ml = val["E_ml"][m][order] * EV_TO_KCAL
        e_xtb = val["E_xtb"][m][order] * EV_TO_KCAL
        e_ml = e_ml - e_ml[-1]
        e_xtb = e_xtb - e_xtb[-1]
        rows.append(
            (
                ray,
                int(val["direction"][m][0]),
                int(val["orientation"][m][0]),
                float(np.min(e_ml)),
                float(np.min(e_xtb)),
            )
        )
    ml = np.array([r[3] for r in rows])
    xtb = np.array([r[4] for r in rows])
    labels = [f"{r[0]} (d{r[1]},o{r[2]})" for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(xtb, ml, c=status_color("serious"), s=55, zorder=3)
    for x, y, lab in zip(xtb, ml, labels, strict=True):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    lo = float(min(ml.min(), xtb.min()) - 0.5)
    hi = float(max(ml.max(), xtb.max()) + 0.5)
    ax.plot([lo, hi], [lo, hi], color=STATUS_COLORS["neutral"], ls="--", lw=1.0, label="y = x")
    ax.set_xlabel("xTB well depth (kcal/mol)")
    ax.set_ylabel("ML hybrid well depth (kcal/mol)")
    ax.set_title("ML vs xTB depths on validated rays")
    ax.set_aspect("equal", adjustable="box")
    legend_outside(ax, side="right")
    _save(fig, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orient-6A", type=Path, required=True, help="dir with rays.csv (6.0 model)")
    p.add_argument("--orient-8A", type=Path, default=None, help="dir with rays.csv (8.0 model)")
    p.add_argument("--validate", type=Path, default=None, help="validate_ACO rays CSV")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dir-slice", type=int, default=2, help="direction for intersecting slice")
    p.add_argument("--ori-slice", type=int, default=11, help="orientation for intersecting slice")
    args = p.parse_args()

    apply_plot_style("icml")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    series: list[tuple[str, dict[str, np.ndarray]]] = []
    r6 = _load_rays(args.orient_6A / "rays.csv")
    series.append(("mm_switch_on=6.0", r6))
    r8 = None
    if args.orient_8A is not None:
        r8 = _load_rays(args.orient_8A / "rays.csv")
        series.append(("mm_switch_on=8.0", r8))

    print("heatmaps / surfaces")
    plot_heatmap(r6, out / "heatmap_6A.png", title="ACO well depth — mm_switch_on=6.0")
    plot_surface_3d(r6, out / "surface3d_6A.png", title="ACO well-depth surface — 6.0")
    if r8 is not None:
        plot_heatmap(r8, out / "heatmap_8A.png", title="ACO well depth — mm_switch_on=8.0")
        plot_surface_3d(r8, out / "surface3d_8A.png", title="ACO well-depth surface — 8.0")

    print("threshold sensitivity")
    plot_spurious_vs_threshold(series, out / "spurious_vs_threshold.png")

    print("hypercube representations")
    plot_intersecting_index_faces(
        r6,
        out / "hypercube_intersecting_faces_6A.png",
        title=f"Intersecting index faces — dir={args.dir_slice} x ori={args.ori_slice} (6.0)",
        dir_slice=args.dir_slice,
        ori_slice=args.ori_slice,
    )
    plot_parallel_direction_slices(
        r6,
        out / "hypercube_parallel_slices_6A.png",
        title="Parallel direction-row slices through the (dir × ori) face (6.0)",
    )
    plot_pairwise_cube_faces(
        r6,
        out / "hypercube_pairwise_faces_6A.png",
        title="Pairwise faces of the (dir, ori, r_min) cube (6.0)",
    )
    plot_parameter_hypercube_embedding(
        r6,
        out / "hypercube_parameter_embedding_6A.png",
        title="Low-discrepancy parameter-cube embeddings (6.0)",
    )
    for d in (2, 5, 6, 8):
        plot_minima_cloud_for_direction(
            r6,
            out / f"minima_cloud_dir{d}_6A.png",
            title=f"Minima cloud — direction {d} (6.0); proxy for E(r, ori)",
            direction=d,
        )

    if args.validate is not None:
        print("xTB validate panels")
        val = _load_validate(args.validate)
        plot_validate_curves(val, out / "validate_ml_vs_xtb_curves.png")
        plot_validate_depth_scatter(val, out / "validate_ml_vs_xtb_depths.png")

    # small index for humans
    index = out / "README.md"
    index.write_text(
        "\n".join(
            [
                "# Orientation-scan figures",
                "",
                "Generated by `scripts/plot_orient_hypercube_slices.py`.",
                "",
                "## Geometry of the potential",
                "- `heatmap_{6,8}A.png` — direction × orientation well depths; red = spurious",
                "- `surface3d_{6,8}A.png` — same face as a 3D surface",
                "- `spurious_vs_threshold.png` — why the 6.0/8.0 ranking flips with τ",
                "",
                "## Hypercube-slice representations",
                "- `hypercube_intersecting_faces_6A.png` — translucent (dir×ori) face + one row/column intersection",
                "- `hypercube_parallel_slices_6A.png` — parallel direction-row ribbons",
                "- `hypercube_pairwise_faces_6A.png` — cube-net of (dir, ori, r_min) faces",
                "- `hypercube_parameter_embedding_6A.png` — Fibonacci / super-Fibonacci parameter cube",
                "- `minima_cloud_dir*_6A.png` — (ori, r_min, e_min) clouds for damaged approach dirs",
                "",
                "## xTB validation",
                "- `validate_ml_vs_xtb_curves.png`",
                "- `validate_ml_vs_xtb_depths.png`",
                "",
            ]
        )
        + "\n"
    )
    print(f"  wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
