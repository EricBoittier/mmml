#!/usr/bin/env python3
"""Plot multipole component decomposition vs lateral offset at the energy minimum.

For each pair, finds the equilibrium distance at each lateral offset and plots
how each spherical-harmonic component contributes to the total interaction energy
as a function of the offset coordinate.

Provides three views per pair:
  1. Component bar chart at each offset value
  2. Component curves vs distance (one subplot per offset)
  3. Stacked area chart showing how component fractions change with offset

Usage
-----
    python scripts/plot_components_vs_offset.py --csv path/to/results.csv
    python scripts/plot_components_vs_offset.py --csv foo.csv --pair TIP3 TIP3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

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

# LaTeX labels for multipole components
LATEX_LABELS: dict[str, str] = {
    "comp_0-0": r"$E_{00}\ (q$-$q)$",
    "comp_0-1": r"$E_{01}\ (q$-$\mu)$",
    "comp_1-1": r"$E_{11}\ (\mu$-$\mu)$",
    "comp_0-2": r"$E_{02}\ (q$-$\Theta)$",
    "comp_1-2": r"$E_{12}\ (\mu$-$\Theta)$",
    "comp_2-2": r"$E_{22}\ (\Theta$-$\Theta)$",
    "comp_0-3": r"$E_{03}\ (q$-$\Omega)$",
    "comp_1-3": r"$E_{13}\ (\mu$-$\Omega)$",
    "comp_2-3": r"$E_{23}\ (\Theta$-$\Omega)$",
    "comp_3-3": r"$E_{33}\ (\Omega$-$\Omega)$",
}

COMP_COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
    "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _component_cols(df: pd.DataFrame) -> list[str]:
    """Return all comp_*_kcal_mol columns that are non-trivially non-zero."""
    cols = [c for c in df.columns if c.startswith("comp_") and c.endswith("_kcal_mol") and "total" not in c]
    # Keep only cols where max|value| > threshold
    significant = [c for c in cols if df[c].abs().max() > 1e-4]
    return significant


def _equilibrium_row(df_off: pd.DataFrame) -> pd.Series | None:
    """Find the row closest to the energy minimum in the well region."""
    mask = df_off["distance_angstrom"] >= 3.5
    df_well = df_off[mask]
    if df_well.empty:
        return None
    idx_min = df_well["energy_kcal_mol"].idxmin()
    return df_off.loc[idx_min]


def plot_components_vs_offset(
    df_pair: pd.DataFrame,
    label_a: str,
    label_b: str,
    out_dir: Path,
) -> None:
    """Render component decomposition plots for one pair."""
    df_mp = df_pair[df_pair["backend"] == "learned_multipole"].copy()
    if df_mp.empty:
        print(f"  {label_a}+{label_b}: no multipole data, skipping.")
        return

    comp_cols = _component_cols(df_mp)
    if not comp_cols:
        print(f"  {label_a}+{label_b}: no component columns, skipping.")
        return

    offsets = sorted(df_mp["offset_angstrom"].unique())
    n_off = len(offsets)
    n_comp = len(comp_cols)

    # ── Figure 1: component bar chart at equilibrium per offset ──────────────
    fig1, axes1 = plt.subplots(
        1, max(n_off, 1),
        figsize=(3.5 * max(n_off, 1), 5),
        constrained_layout=True,
        sharey=True,
    )
    if n_off == 1:
        axes1 = [axes1]

    fig1.suptitle(
        f"{label_a} + {label_b}: Multipole Components at Energy Minimum",
        fontsize=11,
        fontweight="bold",
    )

    all_comp_vals = []
    eq_rows = []
    for off in offsets:
        df_off = df_mp[df_mp["offset_angstrom"].round(6) == round(off, 6)].sort_values("distance_angstrom")
        row = _equilibrium_row(df_off)
        eq_rows.append(row)
        if row is not None:
            for col in comp_cols:
                if col in row.index:
                    all_comp_vals.append(float(row[col]))

    for ax, off, row in zip(axes1, offsets, eq_rows):
        if row is None:
            ax.set_title(f"Δ={off:.1f} Å\n(no data)")
            continue

        vals = []
        lbls = []
        cols_used = []
        for col in comp_cols:
            base = col.replace("_kcal_mol", "")
            v = float(row.get(col, 0.0))
            vals.append(v)
            lbls.append(LATEX_LABELS.get(base, base))
            cols_used.append(col)

        colors = COMP_COLORS[: len(vals)]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.4, width=0.7)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(lbls, rotation=45, ha="right", fontsize=7)
        d_eq = float(row["distance_angstrom"])
        ax.set_title(f"Δ={off:.1f} Å\n(d={d_eq:.2f} Å)", fontsize=8)
        ax.set_ylabel("$E$ / kcal mol$^{-1}$")

    out1 = out_dir / f"{label_a}_{label_b}_comp_bar.png"
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {out1.name}")

    # ── Figure 2: component curves vs distance, one panel per offset ─────────
    n_rows = (n_off + 2) // 3
    n_cols_fig = min(3, n_off)
    fig2, axes2 = plt.subplots(
        n_rows, n_cols_fig,
        figsize=(5 * n_cols_fig, 4 * n_rows),
        constrained_layout=True,
        sharex=True,
    )
    axes2_flat = np.array(axes2).flatten() if n_off > 1 else [axes2]

    fig2.suptitle(
        f"{label_a} + {label_b}: Multipole Components vs Distance by Offset",
        fontsize=11,
        fontweight="bold",
    )

    for ax, off in zip(axes2_flat, offsets):
        df_off = df_mp[df_mp["offset_angstrom"].round(6) == round(off, 6)].sort_values("distance_angstrom")
        x = df_off["distance_angstrom"].values

        for col, color, lbl in zip(comp_cols, COMP_COLORS, [LATEX_LABELS.get(c.replace("_kcal_mol", ""), c) for c in comp_cols]):
            if col not in df_off.columns:
                continue
            y = df_off[col].values
            ax.plot(x, y, color=color, lw=1.4, label=lbl)

        # Also plot total multipole energy
        ax.plot(x, df_off["energy_kcal_mol"].values, color="k", lw=1.8, ls="--", label="Total")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"Offset Δ = {off:.1f} Å", fontsize=9)
        ax.set_xlabel("Distance / Å")
        ax.set_ylabel("$E$ / kcal mol$^{-1}$")

        # Zoom y-limits to well region
        df_well = df_off[df_off["distance_angstrom"] >= 3.5]
        if not df_well.empty:
            e_min = df_well["energy_kcal_mol"].min()
            e_max = max(df_well["energy_kcal_mol"].max(), 0.5)
            pad = max(0.1, (e_max - e_min) * 0.15)
            ax.set_ylim(e_min - pad, e_max + pad)

    # Hide unused axes
    for ax in axes2_flat[n_off:]:
        ax.set_visible(False)

    # Single legend
    handles, labels_leg = axes2_flat[0].get_legend_handles_labels()
    fig2.legend(
        handles, labels_leg,
        loc="lower center",
        ncol=min(6, len(handles)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
        framealpha=0.7,
    )

    out2 = out_dir / f"{label_a}_{label_b}_comp_curves.png"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out2.name}")

    # ── Figure 3: stacked area — component fraction vs offset at equilibrium ─
    if len(eq_rows) >= 2:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        fig3.suptitle(
            f"{label_a} + {label_b}: Component Fractions vs Lateral Offset",
            fontsize=11,
            fontweight="bold",
        )

        # Only show components at equilibrium
        comp_matrix_pos = np.zeros((len(offsets), n_comp))
        comp_matrix_neg = np.zeros((len(offsets), n_comp))
        d_eq_list = []

        for i, (off, row) in enumerate(zip(offsets, eq_rows)):
            if row is None:
                d_eq_list.append(np.nan)
                continue
            d_eq_list.append(float(row["distance_angstrom"]))
            for j, col in enumerate(comp_cols):
                v = float(row.get(col, 0.0))
                if v >= 0:
                    comp_matrix_pos[i, j] = v
                else:
                    comp_matrix_neg[i, j] = v

        offsets_arr = np.array(offsets)
        d_eq_arr = np.array(d_eq_list)

        # Stacked bars for positive components
        bottom_pos = np.zeros(len(offsets))
        bottom_neg = np.zeros(len(offsets))
        lbls_used = [LATEX_LABELS.get(c.replace("_kcal_mol", ""), c) for c in comp_cols]

        for j, (col, color, lbl) in enumerate(zip(comp_cols, COMP_COLORS, lbls_used)):
            ax3a.bar(
                offsets_arr, comp_matrix_pos[:, j],
                bottom=bottom_pos,
                color=color, edgecolor="k", linewidth=0.3, width=0.15,
                label=lbl,
            )
            ax3a.bar(
                offsets_arr, comp_matrix_neg[:, j],
                bottom=bottom_neg,
                color=color, edgecolor="k", linewidth=0.3, width=0.15,
            )
            bottom_pos += comp_matrix_pos[:, j]
            bottom_neg += comp_matrix_neg[:, j]

        ax3a.axhline(0, color="k", lw=0.7)
        ax3a.set_xlabel("Lateral offset / Å")
        ax3a.set_ylabel("Component energy at $E_{min}$ / kcal mol$^{-1}$")
        ax3a.set_title("Component energies at equilibrium geometry")
        ax3a.legend(
            fontsize=7, ncol=2, framealpha=0.6,
            loc="upper right" if bottom_pos.mean() > abs(bottom_neg.mean()) else "lower right",
        )

        # Equilibrium distance vs offset
        ax3b.plot(offsets_arr, d_eq_arr, "o-", color="#4e79a7", lw=2, ms=7)
        ax3b.set_xlabel("Lateral offset / Å")
        ax3b.set_ylabel("Equilibrium distance / Å")
        ax3b.set_title("Equilibrium distance vs lateral offset")
        ax3b.grid(True, alpha=0.3)

        out3 = out_dir / f"{label_a}_{label_b}_comp_fraction.png"
        fig3.savefig(out3, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved: {out3.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Input scan CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign/comp_vs_offset"),
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="Only plot a single pair (e.g. --pair TIP3 TIP3)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "offset_angstrom" not in df.columns:
        print("No 'offset_angstrom' column — adding 0.0 for backward-compat.")
        df["offset_angstrom"] = 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair:
        pairs = [tuple(args.pair)]
    else:
        pairs = [tuple(row) for row in df[["molecule_a", "molecule_b"]].drop_duplicates().values]

    print(f"Plotting component decompositions for {len(pairs)} pairs...")
    for label_a, label_b in pairs:
        df_pair = df[(df["molecule_a"] == label_a) & (df["molecule_b"] == label_b)]
        plot_components_vs_offset(df_pair, str(label_a), str(label_b), args.output_dir)

    print(f"\nAll component plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
