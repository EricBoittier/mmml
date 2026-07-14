#!/usr/bin/env python3
"""Ramachandran (phi/psi) energy-landscape plots, merging the exploratory
ideas from a working notebook (Untitled.ipynb: phi/psi scatter, contour
overlay, torus/"donut" surface for periodic 2D data) with the house style
(`icml` + `default_cmap` + `legend_outside` + big fonts, no chart junk).

Uses the real 64x64 phi/psi PES scan
(`artifacts/trialanine_phi_psi_mm_then_ml_64x64/phi_psi_pes.csv`) rather
than synthetic data -- this is the same dataset the source notebook used.

Each plot picks its colormap by what the *data* actually is (see
docs/plotting-style-guide.md "Colormaps"), not by taste:
- MM or ML energy (relative to its own minimum) is strictly positive ->
  sequential (`default_cmap("sequential")`).
- ML - MM energy difference has a meaningful zero -> diverging
  (`default_cmap("diverging")`).
- phi/psi themselves are periodic -> the torus plot's angular coordinate
  uses the cyclic default (`default_cmap("cyclic")`) where color encodes
  angle; the energy-colored torus surface still uses sequential (energy is
  the encoded magnitude, not an angle).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import gaussian_filter

from mmml.utils.plotting.styles import apply_plot_style, default_cmap

STYLE_NAME = "icml"
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plot-style-gallery-assets"
CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts" / "trialanine_phi_psi_mm_then_ml_64x64" / "phi_psi_pes.csv"
)


def _wrap180(deg: pd.Series) -> pd.Series:
    """Wrap degrees into (-180, 180] -- the CSV's actual_phi/psi_deg
    columns are post-minimization values reported in [0, 360), not the
    nominal -180..180 grid, so plotting them raw leaves half the
    Ramachandran plot empty."""
    return ((deg + 180) % 360) - 180


def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["actual_phi_deg"] = _wrap180(df["actual_phi_deg"])
    df["actual_psi_deg"] = _wrap180(df["actual_psi_deg"])
    df["E_mm"] = df["charmm_mm_min_energy_kcal_mol"] - df["charmm_mm_min_energy_kcal_mol"].min()
    df["E_ml"] = 23.06 * (df["ml_energy_eV"] - df["ml_energy_eV"].min())  # eV -> kcal/mol
    return df


def ramachandran_scatter(df: pd.DataFrame, out: Path) -> None:
    """Phi/psi scatter colored by (positive, baseline-subtracted) MM energy
    -- sequential data, sequential colormap."""
    vmax = 60.0
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    s = ax.scatter(df["actual_phi_deg"], df["actual_psi_deg"], s=16,
                    c=df["E_mm"], cmap=default_cmap("sequential"), vmin=0, vmax=vmax)
    fig.colorbar(s, ax=ax, label="MM energy above minimum (kcal/mol)")
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_title("Ramachandran scatter: MM energy landscape")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ramachandran_contour(df: pd.DataFrame, out: Path) -> None:
    """Filled contour + iso-lines + raw-sample scatter overlay -- reads both
    the smoothed landscape and exactly where it was actually sampled."""
    x = df["actual_phi_deg"].to_numpy()
    y = df["actual_psi_deg"].to_numpy()
    z = df["E_mm"].to_numpy()

    xi = np.linspace(-180, 180, 300)
    yi = np.linspace(-180, 180, 300)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata((x, y), z, (xx, yy), method="cubic")
    zz = gaussian_filter(zz, sigma=2)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = default_cmap("sequential")
    cs = ax.contourf(xx, yy, zz, levels=np.linspace(0, 60, 31), cmap=cmap, extend="max")
    ax.contour(xx, yy, zz, levels=10, colors="#222222", linewidths=0.3, alpha=0.4)
    ax.scatter(x, y, s=4, c="white", alpha=0.15)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title("Ramachandran contour + raw samples")
    fig.colorbar(cs, ax=ax, label="MM energy above minimum (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ramachandran_diff(df: pd.DataFrame, out: Path) -> None:
    """ML - MM energy difference: this data genuinely has a meaningful zero
    (agreement) -- diverging colormap, not sequential."""
    diff = (df["E_ml"] - df["E_mm"]).to_numpy()
    vmax = np.abs(diff).std() * 2.5

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    s = ax.scatter(df["actual_phi_deg"], df["actual_psi_deg"], s=16, c=diff,
                    cmap=default_cmap("diverging"), vmin=-vmax, vmax=vmax)
    fig.colorbar(s, ax=ax, label="ML - MM energy (kcal/mol)")
    ax.set_xlabel(r"$\phi$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal")
    ax.set_title("Where ML and MM disagree (diverging: has a real zero)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def periodic_torus(df: pd.DataFrame, out: Path) -> None:
    """The energy landscape wrapped onto an actual torus, so phi=-180 and
    phi=+180 are physically adjacent (as they are) instead of a fake seam
    at the edge of a flat plot. Energy magnitude -> sequential colormap.
    """
    phi = np.deg2rad(df["actual_phi_deg"].to_numpy())
    psi = np.deg2rad(df["actual_psi_deg"].to_numpy())
    e = df["E_mm"].to_numpy()

    x_features = np.column_stack([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)])
    rbf = RBFInterpolator(x_features, e, kernel="thin_plate_spline", smoothing=0.1)

    n = 160
    phi_g = np.linspace(-np.pi, np.pi, n)
    psi_g = np.linspace(-np.pi, np.pi, n)
    Phi, Psi = np.meshgrid(phi_g, psi_g)
    grid_features = np.column_stack(
        [np.cos(Phi.ravel()), np.sin(Phi.ravel()), np.cos(Psi.ravel()), np.sin(Psi.ravel())]
    )
    Z = rbf(grid_features).reshape(Phi.shape)
    Z = gaussian_filter(Z, sigma=1.5)

    vmax = 60.0
    zn = np.clip(Z / vmax, 0, 1)
    cmap = default_cmap("sequential")
    facecolors = cmap(zn)

    major_r, minor_r = 1.0, 1.0
    xt = (major_r + minor_r * np.cos(Psi)) * np.cos(Phi)
    yt = (major_r + minor_r * np.cos(Psi)) * np.sin(Phi)
    zt = minor_r * np.sin(Psi)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(xt, yt, zt, facecolors=facecolors, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=25, azim=45)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 0.55])
    ax.set_title(r"Periodic $\phi,\psi$ landscape on its natural topology (a torus)")

    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize

    mappable = mcm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, label="MM energy above minimum (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mm_vs_ml_small_multiples(df: pd.DataFrame, out: Path) -> None:
    """MM and ML landscapes side by side, same sequential colormap and
    scale -- small multiples again, this time for a direct MM-vs-ML
    comparison rather than font/style comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = default_cmap("sequential")
    for ax, (col, label) in zip(axes, [("E_mm", "MM"), ("E_ml", "ML")]):
        s = ax.scatter(df["actual_phi_deg"], df["actual_psi_deg"], s=14,
                        c=df[col], cmap=cmap, vmin=0, vmax=60)
        ax.set_title(f"{label} energy landscape")
        ax.set_xlabel(r"$\phi$ (deg)")
        if ax is axes[0]:
            ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_aspect("equal")
        fig.colorbar(s, ax=ax, shrink=0.85)
    fig.suptitle("MM vs. ML: same colormap, same scale, direct comparison")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style(STYLE_NAME)
    df = _load()

    renders = {
        "chart_ramachandran_scatter": ramachandran_scatter,
        "chart_ramachandran_contour": ramachandran_contour,
        "chart_ramachandran_diff": ramachandran_diff,
        "chart_periodic_torus": periodic_torus,
        "chart_mm_vs_ml_multiples": mm_vs_ml_small_multiples,
    }
    for name, fn in renders.items():
        out = OUT_DIR / f"{name}.png"
        fn(df, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
