#!/usr/bin/env python3
"""Orientation energy as parametric surfaces on 6 hemispheres.

A rigid dimer's relative orientation is an element of SO(3).  Write the rotation
as ``R = [e0 e1 e2]`` (body axes of monomer B in the lab frame).  Each column
``ek`` is a point on the unit sphere ``S²``.  That is a *projection* of SO(3):
spin about ``ek`` is invisible on that sphere.

Layout — three spheres × north/south = **six hemispheres**:

    e0 north | e1 north | e2 north
    e0 south | e1 south | e2 south

Each hemisphere is a parametric mesh ``(θ,φ) ↦ (x,y,z)`` on ``S²``, coloured by
an RBF interpolant of well depth aggregated over approach directions
(``min`` or ``mean`` of ``e_min_kcal``).

This is deliberately *not* the approach-direction sphere (that is a seventh
``S²`` for the translational ray).  With only 24 super-Fibonacci orientations
the surfaces are smooth but sparse — treat them as a reading aid, not a PES.

    uv run python scripts/plot_orient_hemisphere_surfaces.py \\
        --rays /Volumes/PortableSSD/DATA/acodcm/orient_6A/rays.csv \\
        --out /Volumes/PortableSSD/DATA/acodcm/orient_plots/hemispheres
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RBFInterpolator

from mmml.utils.plotting.styles import (
    STATUS_COLORS,
    apply_plot_style,
    default_cmap,
    status_color,
)

EV_TO_KCAL = 23.0605  # unused; rays already in kcal


def super_fibonacci(n: int) -> np.ndarray:
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


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _load_rays(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    out = {
        k: np.asarray([float(r[k]) for r in rows])
        for k in ("direction", "orientation", "e_min_kcal", "n_min_ml", "r_at_min")
    }
    for k in ("direction", "orientation", "n_min_ml"):
        out[k] = out[k].astype(int)
    return out


def _aggregate_by_orientation(
    rays: dict[str, np.ndarray], how: str
) -> tuple[np.ndarray, np.ndarray]:
    oris = np.sort(np.unique(rays["orientation"]))
    e = np.zeros(len(oris))
    spur_frac = np.zeros(len(oris))
    for i, o in enumerate(oris):
        m = rays["orientation"] == o
        vals = rays["e_min_kcal"][m]
        e[i] = float(np.min(vals) if how == "min" else np.mean(vals))
        spur_frac[i] = float(np.mean(rays["n_min_ml"][m] > 1))
    return e, spur_frac


def _aggregate_by_direction(
    rays: dict[str, np.ndarray], how: str
) -> np.ndarray:
    dirs = np.sort(np.unique(rays["direction"]))
    e = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        m = rays["direction"] == d
        vals = rays["e_min_kcal"][m]
        e[i] = float(np.min(vals) if how == "min" else np.mean(vals))
    return e


def _sphere_mesh(n_lat: int = 48, n_lon: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    x = np.sin(lat_g) * np.cos(lon_g)
    y = np.sin(lat_g) * np.sin(lon_g)
    z = np.cos(lat_g)
    return x, y, z


def _rbf_on_sphere(points: np.ndarray, values: np.ndarray, mesh_xyz: np.ndarray) -> np.ndarray:
    """Interpolate scalar values given on unit-sphere samples onto a mesh."""
    # mild smoothing so 24 points don't make a spiky surface
    rbf = RBFInterpolator(points, values, kernel="thin_plate_spline", smoothing=0.15)
    return rbf(mesh_xyz)


def _energy_norm(
    vals: np.ndarray,
    *,
    lo: float = 5.0,
    hi: float = 95.0,
    n_mad: float = 1.5,
    min_span: float | None = None,
) -> mcolors.TwoSlopeNorm | mcolors.Normalize:
    """Colourscale matched to data variance (median ± ``n_mad``·MAD).

    Falls back to ``lo``/``hi`` percentiles if MAD is degenerate.  Pivot is 0
    when the window straddles zero, otherwise the data median, so deeper- and
    shallower-than-typical wells each get a full half of a diverging ramp.
    """
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return mcolors.Normalize(vmin=-1.0, vmax=1.0)

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = 1.4826 * mad  # MAD → approx. σ for normal data
    if sigma > 1e-9:
        vmin = median - n_mad * sigma
        vmax = median + n_mad * sigma
    else:
        vmin = float(np.percentile(finite, lo))
        vmax = float(np.percentile(finite, hi))

    if abs(vmax - vmin) < 1e-9:
        vmin, vmax = float(finite.min()), float(finite.max())
    span = vmax - vmin
    if min_span is None:
        min_span = max(0.5, 2.0 * float(np.std(finite)) if finite.size > 1 else 0.5)
    if span < min_span:
        vmin, vmax = median - 0.5 * min_span, median + 0.5 * min_span
        span = min_span
    pad = 0.03 * span
    vmin, vmax = vmin - pad, vmax + pad

    if vmin < 0.0 < vmax:
        vcenter = 0.0
    else:
        eps = 1e-6 * max(span, 1e-6)
        vcenter = float(np.clip(median, vmin + eps, vmax - eps))
    return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)


def _draw_hemisphere(
    ax,
    *,
    axis_points: np.ndarray,
    energy: np.ndarray,
    hemisphere: str,
    title: str,
    norm,
    cmap,
    show_samples: bool = True,
) -> None:
    x, y, z = _sphere_mesh()
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    e_mesh = _rbf_on_sphere(axis_points, energy, pts).reshape(x.shape)

    if hemisphere == "north":
        mask = z < -1e-9
        elev, azim = 22, -60
    else:
        mask = z > 1e-9
        elev, azim = 22, -60
        # view from below: flip elev
        elev, azim = -22, -60

    # hide the other hemisphere (transparent facecolors)
    facecolors = cmap(norm(np.where(mask, np.nanmedian(e_mesh), e_mesh)))
    facecolors[mask] = (1, 1, 1, 0)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    # equatorial rim
    th = np.linspace(0, 2 * np.pi, 120)
    ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), color=STATUS_COLORS["neutral"], lw=0.7)

    if show_samples:
        keep = axis_points[:, 2] >= -1e-9 if hemisphere == "north" else axis_points[:, 2] <= 1e-9
        # near-equator samples shown on both
        near = np.abs(axis_points[:, 2]) < 0.15
        keep = keep | near
        p = axis_points[keep]
        if len(p):
            ax.scatter(
                p[:, 0],
                p[:, 1],
                p[:, 2],
                c=energy[keep],
                cmap=cmap,
                norm=norm,
                s=28,
                edgecolors="white",
                linewidths=0.4,
                depthshade=False,
            )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_six_hemispheres(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    how: str,
    n_orientations: int = 24,
) -> None:
    apply_plot_style("icml")
    energy, _spur = _aggregate_by_orientation(rays, how)
    quats = super_fibonacci(n_orientations)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)  # (n, 3, 3)
    axis_tips = [Rs[:, :, k] for k in range(3)]  # each (n, 3)

    all_e = energy
    cmap = default_cmap("diverging") if (all_e.min() < 0 < all_e.max()) else default_cmap(
        "sequential"
    )
    # fixed norm across all 6 panels
    norm = _energy_norm(all_e)

    fig = plt.figure(figsize=(11.5, 7.2))
    axis_names = ("e₀ (R col 0)", "e₁ (R col 1)", "e₂ (R col 2)")
    for col, (tips, name) in enumerate(zip(axis_tips, axis_names, strict=True)):
        for row, hemi in enumerate(("north", "south")):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            _draw_hemisphere(
                ax,
                axis_points=tips,
                energy=energy,
                hemisphere=hemi,
                title=f"{name} — {hemi}\ncolour = {how} well depth",
                norm=norm,
                cmap=cmap,
            )

    # shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"{how} e_min over approach dirs (kcal/mol)")
    fig.suptitle(
        "SO(3) -> 3 body-axis spheres x N/S = 6 hemispheres\n"
        "(parametric S2 surfaces; spin about each axis is lost in that panel)",
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_approach_hemispheres(
    rays: dict[str, np.ndarray],
    out: Path,
    *,
    how: str,
    n_directions: int = 10,
) -> None:
    """Companion: approach direction û ∈ S² as a N/S hemisphere pair."""
    apply_plot_style("icml")
    energy = _aggregate_by_direction(rays, how)
    tips = fibonacci_sphere(n_directions)
    cmap = default_cmap("diverging") if (energy.min() < 0 < energy.max()) else default_cmap(
        "sequential"
    )
    norm = _energy_norm(energy)

    fig = plt.figure(figsize=(8.0, 4.0))
    for col, hemi in enumerate(("north", "south")):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _draw_hemisphere(
            ax,
            axis_points=tips,
            energy=energy,
            hemisphere=hemi,
            title=f"approach û — {hemi}\ncolour = {how} well depth",
            norm=norm,
            cmap=cmap,
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.04, pad=0.04)
    cbar.set_label(f"{how} e_min over orientations (kcal/mol)")
    fig.suptitle("Approach direction on S² (not part of the 6-hemisphere SO(3) layout)", y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_xtb_on_hemispheres(
    validate: Path,
    rays_for_layout: Path,
    out: Path,
) -> None:
    """Same 6-hemisphere layout, but only the 4 xTB-validated orientations marked.

    Background surface still from the ML orientation table (context); xTB well
    depths drawn as large markers on the matching axis tips.
    """
    apply_plot_style("icml")
    rays = _load_rays(rays_for_layout)
    energy_ml, _ = _aggregate_by_orientation(rays, "min")

    rows = list(csv.DictReader(validate.open()))
    # per validated ray: orientation + xTB well (asymptote-shifted)
    xtb_by_ori: dict[int, float] = {}
    for ray in sorted({int(r["ray"]) for r in rows}):
        sub = sorted(
            [r for r in rows if int(r["ray"]) == ray],
            key=lambda r: float(r["r_com"]),
        )
        ori = int(sub[0]["orientation"])
        e = np.asarray([float(x["E_xtb"]) for x in sub]) * EV_TO_KCAL
        e = e - e[-1]
        xtb_by_ori[ori] = float(np.min(e))

    n_orientations = 24
    quats = super_fibonacci(n_orientations)
    Rs = np.stack([quat_to_matrix(q) for q in quats], axis=0)
    axis_tips = [Rs[:, :, k] for k in range(3)]

    cmap = default_cmap("diverging")
    norm = _energy_norm(energy_ml)

    fig = plt.figure(figsize=(11.5, 7.2))
    axis_names = ("e₀", "e₁", "e₂")
    for col, (tips, name) in enumerate(zip(axis_tips, axis_names, strict=True)):
        for row, hemi in enumerate(("north", "south")):
            ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
            _draw_hemisphere(
                ax,
                axis_points=tips,
                energy=energy_ml,
                hemisphere=hemi,
                title=f"{name} — {hemi}",
                norm=norm,
                cmap=cmap,
                show_samples=False,
            )
            # xTB markers
            for ori, e_xtb in xtb_by_ori.items():
                p = tips[ori]
                on_hemi = (p[2] >= -0.05) if hemi == "north" else (p[2] <= 0.05)
                if not on_hemi and abs(p[2]) >= 0.05:
                    continue
                ax.scatter(
                    [p[0]],
                    [p[1]],
                    [p[2]],
                    s=90,
                    c=[e_xtb],
                    cmap=cmap,
                    norm=mcolors.Normalize(vmin=-2.5, vmax=0.0),
                    edgecolors=status_color("critical"),
                    linewidths=1.2,
                    depthshade=False,
                    zorder=6,
                )
                ax.text(p[0] * 1.15, p[1] * 1.15, p[2] * 1.15, f"o{ori}\n{e_xtb:.1f}", fontsize=7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label("ML min well depth (kcal/mol); rings = xTB oris")
    fig.suptitle(
        "Six hemispheres with xTB-validated orientations highlighted (red edge)",
        y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rays", type=Path, required=True, help="orient_*/rays.csv")
    p.add_argument("--validate", type=Path, default=None, help="optional xTB validate CSV")
    p.add_argument("--how", choices=("min", "mean"), default="min")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rays = _load_rays(args.rays)
    plot_six_hemispheres(rays, args.out / f"six_hemispheres_{args.how}.png", how=args.how)
    plot_approach_hemispheres(
        rays, args.out / f"approach_hemispheres_{args.how}.png", how=args.how
    )
    if args.validate is not None:
        plot_xtb_on_hemispheres(
            args.validate,
            args.rays,
            args.out / "six_hemispheres_xtb_markers.png",
        )
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# Six-hemisphere orientation surfaces",
                "",
                "SO(3) rotation `R=[e0 e1 e2]` → three body-axis tips on `S²`.",
                "Each sphere is drawn as north + south hemispheres → **6 panels**.",
                "Colour = well depth aggregated over Fibonacci approach directions.",
                "",
                "Projection caveat: rotation about an axis does not move that",
                "axis tip, so each panel loses one twist DOF.",
                "",
                "Companion: `approach_hemispheres_*.png` is the translational",
                "ray direction on `S²` (not part of the 6).",
                "",
            ]
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
