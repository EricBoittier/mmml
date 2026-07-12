#!/usr/bin/env python3
"""Plot families of 1D energy curves, one curve per lateral offset value.

For each pair and each backend, renders a panel where each line is the
energy-vs-distance profile at a fixed lateral offset.  Curves are:
  • coloured from blue (offset=0, on-axis) to red (maximum offset)
  • scaled per-curve via z-score on the well region (≥3.8 Å, E≤0)
  • annotated with ASE atoms overlay showing the minimum-energy geometry

Usage
-----
    python scripts/plot_1d_slices_by_offset.py --csv path/to/results.csv
    python scripts/plot_1d_slices_by_offset.py --csv foo.csv --backends xtb_gfn2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.visualize.plot import plot_atoms
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from mmml.analysis.dimer_molecules import PAIR_SCAN_CONFIG, ORIENTED_MONOMERS, MOLECULES
from mmml.analysis.dimer_scans import build_rigid_dimer_2d
from plot_utils import (
    BACKEND_COLORS,
    BACKEND_LABELS,
    load_and_enrich,
    ordered_backends,
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.dpi": 150,
        "text.usetex": False,
    }
)


def _scale_series(y: np.ndarray, x: np.ndarray, min_dist: float = 3.8) -> np.ndarray:
    """Z-score normalise using only the attractive well region."""
    mask = (x >= min_dist) & (y <= 0.0)
    y_well = y[mask]
    if len(y_well) < 2:
        y_well_fb = y[x >= min_dist]
        if len(y_well_fb) >= 2:
            y_well = y_well_fb
        else:
            y_well = y
    mu = np.mean(y_well)
    sigma = np.std(y_well)
    if sigma < 1e-10:
        sigma = 1.0
    return (y - mu) / sigma


def _atoms_snapshot(label_a: str, label_b: str, distance: float, offset: float) -> object:
    """Build an ASE Atoms object for a given geometry."""
    pair = (label_a, label_b)
    if pair not in ORIENTED_MONOMERS:
        pair = (label_b, label_a)
        if pair not in ORIENTED_MONOMERS:
            return None
    monomers = ORIENTED_MONOMERS[pair]
    cfg = PAIR_SCAN_CONFIG[pair]
    atoms, _ = build_rigid_dimer_2d(
        monomers["a"],
        monomers["b"],
        distance_angstrom=distance,
        offset_angstrom=offset,
        axis=(0, 0, 1),
        transverse_axis=cfg["transverse_axis"],
        center="none",
    )
    return atoms


def plot_slices_for_pair(
    df_pair: pd.DataFrame,
    label_a: str,
    label_b: str,
    backends: list[str],
    out_dir: Path,
    show_atoms: bool = True,
) -> None:
    """One figure per pair: rows = backends, columns = [atoms | 1D slices | 1D slices (raw)]."""
    avail_backends = [b for b in backends if b in df_pair["backend"].unique()]
    if not avail_backends:
        return

    offsets = sorted(df_pair["offset_angstrom"].unique())
    n_off = len(offsets)
    cmap = mpl.colormaps.get_cmap("coolwarm")
    norm = Normalize(vmin=0, vmax=max(offsets) if max(offsets) > 0 else 1)

    pair = (label_a, label_b)
    cfg = PAIR_SCAN_CONFIG.get(pair) or PAIR_SCAN_CONFIG.get((label_b, label_a))
    description = cfg["description"] if cfg else f"{label_a}+{label_b}"

    n_be = len(avail_backends)
    n_cols = 3  # atoms | scaled | raw
    fig, axes = plt.subplots(
        n_be, n_cols,
        figsize=(14, 3.8 * n_be),
        gridspec_kw={"width_ratios": [1.2, 2.2, 2.2]},
        constrained_layout=True,
    )
    if n_be == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"{label_a} + {label_b}  —  {description}",
        fontsize=12,
        fontweight="bold",
    )

    # Add offset colourbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:, 2], shrink=0.6, pad=0.02)
    cb.set_label("Lateral offset / Å", fontsize=9)

    for row_idx, backend in enumerate(avail_backends):
        df_be = df_pair[df_pair["backend"] == backend].copy()
        ax_atoms, ax_scaled, ax_raw = axes[row_idx]

        # ── Panel 1: filmstrip of ASE atoms along the reaction coordinate (offset=0) ──
        ax_atoms.set_axis_off()
        if show_atoms and cfg is not None:
            dist_vals_row = sorted(df_be["distance_angstrom"].unique())
            n_snap = min(4, len(dist_vals_row)) or 1
            snap_idx = np.linspace(0, len(dist_vals_row) - 1, n_snap).round().astype(int)
            snap_distances = [dist_vals_row[i] for i in sorted(set(snap_idx))]
            n_snap = len(snap_distances)
            for j, snap_dist in enumerate(snap_distances):
                inset = ax_atoms.inset_axes([j / n_snap, 0.15, 1 / n_snap, 0.7])
                inset.set_axis_off()
                try:
                    atoms_snap = _atoms_snapshot(label_a, label_b, snap_dist, 0.0)
                    if atoms_snap is not None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            plot_atoms(atoms_snap, inset, rotation="0x,0y,0z", radii=0.4)
                except Exception:
                    pass
                inset.set_title(f"{snap_dist:.1f} Å", fontsize=6, pad=1)
        ax_atoms.set_title(f"{backend}\nreaction coordinate (offset=0)", fontsize=8)

        # ── Panels 2 & 3: scaled and raw energy curves ──
        ax_scaled.axhline(0, color="gray", lw=0.6, ls=":")
        ax_raw.axhline(0, color="gray", lw=0.6, ls=":")

        # Well-region y-limits accumulator for scaled panel
        well_scaled_vals = []

        for off in offsets:
            df_off = df_be[df_be["offset_angstrom"].round(6) == round(off, 6)].sort_values("distance_angstrom")
            if df_off.empty:
                continue

            x = df_off["distance_angstrom"].values
            y = df_off["energy_kcal_mol"].values

            # Reference: subtract energy at largest distance
            y = y - y[-1]
            color = cmap(norm(off))

            # Scaled panel
            y_scaled = _scale_series(y, x)
            lbl = f"Δ={off:.1f} Å"
            ax_scaled.plot(x, y_scaled, color=color, lw=1.6, label=lbl, marker="o", ms=3)

            # Collect well region for axis limits
            mask_well = (x >= 3.8)
            if mask_well.any():
                ws = y_scaled[mask_well]
                ws = ws[ws < 5.0]  # exclude big repulsions
                well_scaled_vals.extend(ws.tolist())

            # Raw panel
            ax_raw.plot(x, y, color=color, lw=1.6, label=lbl, marker="o", ms=3)

        # Set well-zoomed y-limits on scaled panel
        if well_scaled_vals:
            arr = np.array(well_scaled_vals)
            lo = np.percentile(arr, 2)
            hi = np.percentile(arr, 98)
            pad = max(0.3, (hi - lo) * 0.2)
            ax_scaled.set_ylim(lo - pad, hi + pad)

        # Raw y-limits: well region only (distance > 3.5, E < 0 or E < 1)
        df_well = df_be[df_be["distance_angstrom"] >= 3.5].copy()
        df_well["E_int"] = df_well["energy_kcal_mol"] - df_be.groupby("offset_angstrom")["energy_kcal_mol"].transform("last")
        if not df_well.empty:
            e_min = df_well["E_int"].min()
            e_max = df_well["E_int"].quantile(0.9)
            pad_raw = max(0.1, (e_max - e_min) * 0.15)
            ax_raw.set_ylim(min(e_min - pad_raw, -0.05), e_max + pad_raw)

        ax_scaled.set_xlabel("Centre distance / Å")
        ax_scaled.set_ylabel("Scaled energy (a.u.)")
        ax_scaled.set_title(f"{BACKEND_LABELS.get(backend, backend)} — scaled")
        ax_scaled.legend(fontsize=7, framealpha=0.6, ncol=2, loc="upper right")

        ax_raw.set_xlabel("Centre distance / Å")
        ax_raw.set_ylabel("$E_{int}$ / kcal mol$^{-1}$")
        ax_raw.set_title(f"{BACKEND_LABELS.get(backend, backend)} — raw")
        ax_raw.legend(fontsize=7, framealpha=0.6, ncol=2, loc="upper right")

    pair_tag = f"{label_a}_{label_b}"
    out_path = out_dir / f"{pair_tag}_1d_slices.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input scan CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign/slices_by_offset"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to include (default: all found)",
    )
    parser.add_argument(
        "--no-atoms",
        action="store_true",
        help="Skip ASE atoms structure panels",
    )
    args = parser.parse_args()

    df = load_and_enrich(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backends = ordered_backends(df, args.backends)

    pairs = df[["molecule_a", "molecule_b"]].drop_duplicates().values
    print(f"Plotting 1D slice families for {len(pairs)} pairs, {len(backends)} backends...")

    for label_a, label_b in pairs:
        df_pair = df[(df["molecule_a"] == label_a) & (df["molecule_b"] == label_b)]
        plot_slices_for_pair(
            df_pair, label_a, label_b, backends, args.output_dir,
            show_atoms=not args.no_atoms,
        )

    print(f"\nAll 1D slice plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
