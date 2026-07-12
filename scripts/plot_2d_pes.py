#!/usr/bin/env python3
"""Plot 2D potential energy surfaces as heatmaps/contour plots.

For each (pair, backend) combination, interpolates the discrete (distance, offset)
grid onto a smooth surface and renders contour + scatter plots.

Input CSV must have columns:
    molecule_a, molecule_b, distance_angstrom, offset_angstrom,
    energy_kcal_mol, backend

Usage
-----
    python scripts/plot_2d_pes.py --csv results/dimer_scan_campaign/scan_results_new.csv
    python scripts/plot_2d_pes.py --csv foo.csv --backends xtb_gfn2 charmm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.dpi": 150,
        "text.usetex": False,
    }
)

BACKEND_LABELS = {
    "learned_multipole": "Multipoles",
    "learned_mbd": "MBD",
    "xtb_gfn2": "GFN2-xTB",
    "charmm": "CGenFF",
}
BACKEND_CMAPS = {
    "learned_multipole": "RdBu_r",
    "learned_mbd":       "PuOr_r",
    "xtb_gfn2":          "RdYlGn_r",
    "charmm":            "seismic",
}


def _reference_energy(series: pd.Series, ref_dist: float | None = None) -> pd.Series:
    """Subtract the energy at the largest distance (or explicit *ref_dist*)."""
    if ref_dist is not None:
        idx = (series.index - ref_dist).abs().argmin()
        ref = series.iloc[idx]
    else:
        ref = series.iloc[-1]
    return series - ref


def plot_2d_pes_for_pair(
    df_pair: pd.DataFrame,
    label_a: str,
    label_b: str,
    backends: list[str],
    out_dir: Path,
    n_grid: int = 80,
    energy_clip_kcal: float = 5.0,
) -> None:
    """Render a 2D PES heatmap for one (pair, backend) set."""
    n_be = len(backends)
    if n_be == 0:
        return

    fig, axes = plt.subplots(
        1, n_be,
        figsize=(5.5 * n_be, 5.5),
        constrained_layout=True,
    )
    if n_be == 1:
        axes = [axes]

    pair_tag = f"{label_a}_{label_b}"
    fig.suptitle(
        f"2D PES: {label_a} + {label_b}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, backend in zip(axes, backends):
        df_be = df_pair[df_pair["backend"] == backend].copy()
        if df_be.empty:
            ax.set_visible(False)
            continue

        # Reference energy at largest distance, offset=0 (on-axis)
        df_on = df_be[df_be["offset_angstrom"] == df_be["offset_angstrom"].min()]
        if not df_on.empty:
            ref = df_on.sort_values("distance_angstrom")["energy_kcal_mol"].iloc[-1]
        else:
            ref = df_be["energy_kcal_mol"].max()

        df_be["E_int"] = df_be["energy_kcal_mol"] - ref

        dist_vals = np.sort(df_be["distance_angstrom"].unique())
        off_vals  = np.sort(df_be["offset_angstrom"].unique())

        if len(dist_vals) < 2 or len(off_vals) < 2:
            # Fall back to scatter only
            sc = ax.scatter(
                df_be["distance_angstrom"],
                df_be["offset_angstrom"],
                c=df_be["E_int"].clip(-energy_clip_kcal, energy_clip_kcal),
                cmap=BACKEND_CMAPS.get(backend, "RdBu_r"),
                s=80,
                edgecolors="k",
                linewidths=0.4,
            )
            plt.colorbar(sc, ax=ax, label="$E_{int}$ / kcal mol$^{-1}$")
        else:
            # Pivot to grid and interpolate
            pivot = df_be.pivot_table(
                index="offset_angstrom",
                columns="distance_angstrom",
                values="E_int",
                aggfunc="mean",
            )
            Z = pivot.values  # shape (n_off, n_dist)
            D = pivot.columns.values
            O = pivot.index.values

            # Clip extremes before interpolation
            Z_clipped = np.clip(Z, -energy_clip_kcal, energy_clip_kcal)

            try:
                spline = RectBivariateSpline(O, D, Z_clipped, kx=min(3, len(O)-1), ky=min(3, len(D)-1))
                D_fine = np.linspace(D.min(), D.max(), n_grid)
                O_fine = np.linspace(O.min(), O.max(), n_grid)
                Z_fine = spline(O_fine, D_fine)
                Z_fine = np.clip(Z_fine, -energy_clip_kcal, energy_clip_kcal)
            except Exception:
                D_fine, O_fine = D, O
                Z_fine = Z_clipped

            vmax = min(energy_clip_kcal, np.percentile(np.abs(Z_fine), 95))
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

            cmap = BACKEND_CMAPS.get(backend, "RdBu_r")
            im = ax.contourf(
                D_fine, O_fine, Z_fine,
                levels=30,
                cmap=cmap,
                norm=norm,
            )
            ax.contour(
                D_fine, O_fine, Z_fine,
                levels=[-2, -1, -0.5, -0.2, 0],
                colors="k",
                linewidths=0.6,
                linestyles="--",
                alpha=0.6,
            )
            cb = plt.colorbar(im, ax=ax, shrink=0.85)
            cb.set_label("$E_{int}$ / kcal mol$^{-1}$", fontsize=9)

            # Mark minimum
            min_idx = np.unravel_index(np.argmin(Z_fine), Z_fine.shape)
            ax.plot(
                D_fine[min_idx[1]], O_fine[min_idx[0]],
                "*", color="gold", markersize=14, markeredgecolor="k", markeredgewidth=0.5,
                label=f"min {Z_fine[min_idx]:.2f} kcal/mol",
                zorder=5,
            )
            ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

            # Data point dots
            ax.scatter(
                df_be["distance_angstrom"],
                df_be["offset_angstrom"],
                c="w", s=15, edgecolors="k", linewidths=0.3, zorder=4,
            )

        ax.set_xlabel("Centre distance / Å")
        ax.set_ylabel("Lateral offset / Å")
        ax.set_title(BACKEND_LABELS.get(backend, backend))

    out_path = out_dir / f"{pair_tag}_2d_pes.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input scan CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign/pes_2d"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to plot (default: all found)",
    )
    parser.add_argument(
        "--energy-clip", type=float, default=5.0,
        help="Clip energies to ±N kcal/mol for colour scale (default 5.0)",
    )
    parser.add_argument(
        "--n-grid", type=int, default=80,
        help="Interpolation grid resolution (default 80)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "offset_angstrom" not in df.columns:
        print("No 'offset_angstrom' column found — adding 0.0 (treating as 1D scan).")
        df["offset_angstrom"] = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_backends = df["backend"].unique().tolist()
    backends = args.backends if args.backends else all_backends
    backends = [b for b in backends if b in all_backends]

    pairs = df[["molecule_a", "molecule_b"]].drop_duplicates().values
    print(f"Plotting 2D PES for {len(pairs)} pairs × {len(backends)} backends...")

    for label_a, label_b in pairs:
        df_pair = df[(df["molecule_a"] == label_a) & (df["molecule_b"] == label_b)]
        plot_2d_pes_for_pair(
            df_pair, label_a, label_b, backends, args.output_dir,
            n_grid=args.n_grid, energy_clip_kcal=args.energy_clip,
        )

    print(f"\nAll 2D PES plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
