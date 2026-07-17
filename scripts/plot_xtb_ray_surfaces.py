#!/usr/bin/env python3
"""Clear, smooth GFN2-xTB projections of validated dimer rays.

The orientation-scan hypercube figures are hard to read because the sample is
discrete and the interesting physics lives on E(r) along each ray.  This script
takes the existing ``validate_*/rays_*.csv`` xTB columns, cubic-spline densifies
each ray, and draws:

1. Overlaid smooth binding curves (well region + full range)
2. A smooth 3D curtain ``E(ray, r)`` via bivariate spline on the densified grid
3. Small-multiples of each ray with the attractive well filled

Cross-ray interpolation in the curtain is a *visual* projection only — the four
rays are unrelated orientations; the surface just makes the family of 1D curves
readable as one object.

    uv run python scripts/plot_xtb_ray_surfaces.py \\
        --validate /Volumes/PortableSSD/DATA/acodcm/validate_ACO/rays_ACO.csv \\
        --out /Volumes/PortableSSD/DATA/acodcm/orient_plots/xtb_smooth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import CubicSpline

from mmml.utils.plotting.styles import (
    STATUS_COLORS,
    apply_plot_style,
    comparison_colors,
    default_cmap,
    legend_outside,
)

EV_TO_KCAL = 23.0605


def _load(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"empty validate CSV: {path}")
    return rows


def _ray_tables(rows: list[dict]) -> list[dict]:
    """One dict per ray with sorted r / E_xtb / E_ml (kcal, asymptote-shifted)."""
    rays = sorted({int(r["ray"]) for r in rows})
    out = []
    for ray in rays:
        sub = sorted(
            [r for r in rows if int(r["ray"]) == ray],
            key=lambda r: float(r["r_com"]),
        )
        r = np.asarray([float(x["r_com"]) for x in sub], dtype=float)
        e_xtb = np.asarray([float(x["E_xtb"]) for x in sub], dtype=float) * EV_TO_KCAL
        e_ml = np.asarray([float(x["E_ml"]) for x in sub], dtype=float) * EV_TO_KCAL
        e_xtb = e_xtb - e_xtb[-1]
        e_ml = e_ml - e_ml[-1]
        out.append(
            {
                "ray": ray,
                "direction": int(sub[0]["direction"]),
                "orientation": int(sub[0]["orientation"]),
                "r": r,
                "e_xtb": e_xtb,
                "e_ml": e_ml,
                "label": f"ray {ray} (dir {sub[0]['direction']}, ori {sub[0]['orientation']})",
            }
        )
    return out


def _smooth_1d(r: np.ndarray, e: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    cs = CubicSpline(r, e, bc_type="natural")
    rr = np.linspace(float(r.min()), float(r.max()), n)
    return rr, cs(rr)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_binding_curves(tables: list[dict], out: Path) -> None:
    """Two panels: well zoom + full range, xTB only, spline + raw points."""
    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=len(tables))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for t, color in zip(tables, colors, strict=True):
        rr, ee = _smooth_1d(t["r"], t["e_xtb"], n=500)
        for ax in axes:
            ax.plot(rr, ee, color=color, lw=2.0, label=t["label"])
            ax.scatter(
                t["r"],
                t["e_xtb"],
                s=14,
                color=color,
                alpha=0.35,
                zorder=3,
                linewidths=0,
            )
            i = int(np.argmin(ee))
            ax.scatter(
                [rr[i]],
                [ee[i]],
                s=36,
                facecolors="white",
                edgecolors=color,
                linewidths=1.4,
                zorder=4,
            )

    axes[0].set_ylim(-3.0, 4.0)
    axes[0].set_title("binding well (zoomed)")
    axes[1].set_ylim(-3.0, 25.0)
    axes[1].set_title("full range (repulsive wall)")
    for ax in axes:
        ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.8)
        ax.set_xlabel("r_COM (Å)")
        ax.set_xlim(tables[0]["r"].min(), tables[0]["r"].max())
    axes[0].set_ylabel("GFN2-xTB binding energy (kcal/mol)")
    legend_outside(axes[1], side="right")
    fig.suptitle("Validated ACO rays — GFN2-xTB (cubic spline; E(r_max)=0)", y=1.02)
    _save(fig, out)


def _linear_curtain(
    tables: list[dict],
    r_min: float,
    r_max: float,
    *,
    n_r: int = 200,
    n_between: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a curtain by cubic-smoothing each ray in r, then *linear* blend in ray.

    Bicubic across only 4 rays Runge-overshoots into ±hundreds of kcal/mol; linear
    in the ray axis stays faithful to the measured curves.
    """
    n_ray = len(tables)
    r_hi = np.linspace(r_min, r_max, n_r)
    e_at = np.stack(
        [CubicSpline(t["r"], t["e_xtb"], bc_type="natural")(r_hi) for t in tables]
    )
    y_s = np.linspace(0.0, n_ray - 1, (n_ray - 1) * n_between + 1)
    e_s = np.zeros((len(y_s), n_r))
    for k, y in enumerate(y_s):
        i0 = int(np.floor(y))
        i1 = min(i0 + 1, n_ray - 1)
        w = y - i0
        e_s[k] = (1.0 - w) * e_at[i0] + w * e_at[i1]
    yy, rr = np.meshgrid(y_s, r_hi, indexing="ij")
    return rr, yy, e_s


def plot_smooth_curtain(tables: list[dict], out: Path) -> None:
    """Smooth 3D curtain E(ray_param, r) with true ray curves on top."""
    apply_plot_style("icml")
    n_ray = len(tables)
    r_min = float(tables[0]["r"].min())
    r_max = float(tables[0]["r"].max())
    rr, yy, zz = _linear_curtain(tables, r_min, r_max)

    vmax = 8.0
    vmin = -3.0
    z_show = np.clip(zz, vmin, vmax)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    cmap = default_cmap("diverging")

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        rr,
        yy,
        z_show,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.88,
        rstride=1,
        cstride=1,
    )

    colors = comparison_colors(apply_plot_style("icml"), n=n_ray)
    for i, (t, color) in enumerate(zip(tables, colors, strict=True)):
        rr1, ee1 = _smooth_1d(t["r"], t["e_xtb"], n=400)
        ax.plot(rr1, np.full_like(rr1, float(i)), np.clip(ee1, vmin, vmax), color=color, lw=2.4)
        ax.scatter(
            t["r"],
            np.full_like(t["r"], float(i)),
            np.clip(t["e_xtb"], vmin, vmax),
            color=color,
            s=12,
            alpha=0.5,
            depthshade=False,
        )

    ax.set_xlabel("r_COM (Å)")
    ax.set_ylabel("ray panel")
    ax.set_zlabel("E_xTB (kcal/mol)")
    ax.set_yticks(list(range(n_ray)))
    ax.set_yticklabels([f"{i}: {t['ray']}" for i, t in enumerate(tables)])
    ax.set_zlim(vmin, vmax)
    ax.set_ylim(-0.2, n_ray - 0.8)
    ax.view_init(elev=28, azim=-55)
    ax.set_title("xTB curtain — cubic in r, linear between rays (E clipped to [−3, 8])")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="kcal/mol")
    _save(fig, out)


def plot_smooth_curtain_well(tables: list[dict], out: Path) -> None:
    """Binding-region curtain so the wells dominate the view."""
    apply_plot_style("icml")
    n_ray = len(tables)
    r_min, r_max = 4.2, 9.0
    rr, yy, zz = _linear_curtain(tables, r_min, r_max, n_r=180, n_between=32)

    vmin = -2.5
    vmax = 3.0
    z_show = np.clip(zz, vmin, vmax)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    cmap = default_cmap("diverging")

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        rr,
        yy,
        z_show,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rstride=1,
        cstride=1,
    )
    colors = comparison_colors(apply_plot_style("icml"), n=n_ray)
    for i, (t, color) in enumerate(zip(tables, colors, strict=True)):
        mask = (t["r"] >= r_min) & (t["r"] <= r_max)
        rr1 = np.linspace(r_min, r_max, 300)
        ee1 = CubicSpline(t["r"], t["e_xtb"], bc_type="natural")(rr1)
        ax.plot(rr1, np.full_like(rr1, float(i)), np.clip(ee1, vmin, vmax), color=color, lw=2.6)
        ax.scatter(
            t["r"][mask],
            np.full(mask.sum(), float(i)),
            np.clip(t["e_xtb"][mask], vmin, vmax),
            color=color,
            s=20,
            depthshade=False,
            alpha=0.75,
        )
        j = int(np.argmin(ee1))
        ax.scatter(
            [rr1[j]],
            [float(i)],
            [float(np.clip(ee1[j], vmin, vmax))],
            s=50,
            facecolors="white",
            edgecolors=color,
            linewidths=1.6,
            depthshade=False,
            zorder=5,
        )

    ax.set_xlabel("r_COM (Å)")
    ax.set_ylabel("ray panel")
    ax.set_zlabel("E_xTB (kcal/mol)")
    ax.set_yticks(list(range(n_ray)))
    ax.set_yticklabels([f"{i}: {t['ray']}" for i, t in enumerate(tables)])
    ax.set_zlim(vmin, vmax)
    ax.set_ylim(-0.15, n_ray - 0.85)
    ax.view_init(elev=30, azim=-58)
    ax.set_title("xTB binding wells — smooth projection (4.2–9 Å)")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label="kcal/mol")
    _save(fig, out)


def plot_small_multiples(tables: list[dict], out: Path) -> None:
    style = apply_plot_style("icml")
    colors = comparison_colors(style, n=len(tables))
    n = len(tables)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, t, color in zip(axes, tables, colors, strict=True):
        rr, ee = _smooth_1d(t["r"], t["e_xtb"], n=500)
        ax.plot(rr, ee, color=color, lw=2.0)
        ax.fill_between(rr, ee, 0.0, where=(ee < 0), color=color, alpha=0.18, linewidth=0)
        ax.scatter(t["r"], t["e_xtb"], s=12, color=color, alpha=0.4, linewidths=0)
        ax.axhline(0.0, color=STATUS_COLORS["neutral"], lw=0.8)
        ax.set_ylim(-3.0, 5.0)
        ax.set_xlabel("r_COM (Å)")
        ax.set_title(t["label"], fontsize=10)
    axes[0].set_ylabel("E_xTB (kcal/mol)")
    fig.suptitle("GFN2-xTB per ray — smooth spline, attractive well shaded", y=1.03)
    _save(fig, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    apply_plot_style("icml")
    tables = _ray_tables(_load(args.validate))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(tables)} xTB rays")
    plot_binding_curves(tables, out / "xtb_binding_curves.png")
    plot_small_multiples(tables, out / "xtb_small_multiples.png")
    plot_smooth_curtain(tables, out / "xtb_smooth_curtain.png")
    plot_smooth_curtain_well(tables, out / "xtb_smooth_curtain_wells.png")

    (out / "README.md").write_text(
        "\n".join(
            [
                "# Smooth GFN2-xTB ray projections",
                "",
                "From `validate_ACO/rays_ACO.csv` — four rays, cubic-spline densified.",
                "",
                "- `xtb_binding_curves.png` — overlay (well zoom + full wall)",
                "- `xtb_small_multiples.png` — one panel per ray",
                "- `xtb_smooth_curtain.png` — 3D E(ray, r) bivariate spline (wall clipped)",
                "- `xtb_smooth_curtain_wells.png` — same, 4–9 Å so wells dominate",
                "",
                "The curtain's cross-ray axis is ordinal (visual only).",
                "",
            ]
        )
        + "\n"
    )
    print(f"  wrote {out / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
