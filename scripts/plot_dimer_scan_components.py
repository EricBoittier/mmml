#!/usr/bin/env python3
"""Plot component-wise energy contributions for each dimer pair from scan results."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/dimer_scan_campaign/scan_results.csv"),
        help="Path to scan_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign/components_plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    print(f"Reading scan results from {args.csv}...")
    df = pd.read_csv(args.csv)

    # Filter to learned_multipole since it has components
    df_mp = df[df["backend"] == "learned_multipole"]
    if df_mp.empty:
        print(
            "Error: No 'learned_multipole' entries found in CSV, or they do not have components columns."
        )
        print("Please rerun the scan campaign with the updated code to collect component energies.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Decomposed component terms to plot
    components = [
        ("0-0", "Monopole-Monopole"),
        ("0-1", "Monopole-Dipole"),
        ("1-1", "Dipole-Dipole"),
        ("0-2", "Monopole-Quadrupole"),
        ("1-2", "Dipole-Quadrupole"),
        ("2-2", "Quadrupole-Quadrupole"),
        ("0-3", "Monopole-Octupole"),
        ("1-3", "Dipole-Octupole"),
        ("2-3", "Quadrupole-Octupole"),
        ("3-3", "Octupole-Octupole"),
    ]

    pairs = df_mp.groupby(["molecule_a", "molecule_b"])
    print(f"Generating component-wise plots for {len(pairs)} pairs...")

    for (label_a, label_b), group in pairs:
        group = group.sort_values("distance_angstrom")

        # Create a 4x3 grid of subplots (10 components + total energy)
        fig, axes = plt.subplots(4, 3, figsize=(14, 15), sharex=True)
        axes = axes.flatten()

        for idx, (comp_key, comp_name) in enumerate(components):
            ax = axes[idx]
            col_name = f"comp_{comp_key}_kcal_mol"
            if col_name in group.columns:
                ax.plot(
                    group["distance_angstrom"],
                    group[col_name],
                    marker="o",
                    color="steelblue",
                    linewidth=1.5,
                )
                ax.set_title(f"{comp_name} ({comp_key})", fontsize=10)
                ax.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=8)
                ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                ax.tick_params(axis="both", which="major", labelsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{comp_name} ({comp_key})", fontsize=10)

        # Plot total energy in subplot 10
        ax_tot = axes[10]
        ax_tot.plot(
            group["distance_angstrom"],
            group["energy_kcal_mol"],
            marker="o",
            color="crimson",
            linewidth=2.0,
        )
        ax_tot.set_title("Total Electrostatics Energy", fontsize=11, fontweight="bold")
        ax_tot.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
        ax_tot.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax_tot.tick_params(axis="both", which="major", labelsize=8)

        # Remove the unused 12th subplot (index 11)
        fig.delaxes(axes[11])

        # Add shared X labels to bottom row subplots
        for i in [9, 10]:
            axes[i].set_xlabel("Center distance / Å", fontsize=10)

        plt.suptitle(
            f"Multipole Energy Components: {label_a} + {label_b}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = args.output_dir / f"{label_a}_{label_b}_components.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"All component plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
