#!/usr/bin/env python3
"""Plot component-wise and total energy contributions with dimer structure illustrations."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from ase.visualize.plot import plot_atoms

from mmml.analysis.dimer_scans import build_rigid_dimer

# Monomer registry for structure rendering
MOLECULES = {
    "DCM": {
        "atoms": Atoms(
            "CCl2H2",
            positions=[
                [0.000, 0.000, 0.000],
                [1.760, 0.000, 0.000],
                [-1.760, 0.000, 0.000],
                [0.000, 0.950, 0.720],
                [0.000, -0.950, 0.720],
            ],
        ),
    },
    "ACE": {
        "atoms": Atoms(
            "C3OH6",
            positions=[
                [0.000, 0.000, 0.000],
                [1.520, 0.000, 0.000],
                [-1.520, 0.000, 0.000],
                [0.000, 1.220, 0.000],
                [2.050, 0.900, 0.000],
                [2.050, -0.450, 0.780],
                [2.050, -0.450, -0.780],
                [-2.050, 0.900, 0.000],
                [-2.050, -0.450, 0.780],
                [-2.050, -0.450, -0.780],
            ],
        ),
    },
    "BENZ": {
        "atoms": Atoms(
            "C6H6",
            positions=[
                [1.397, 0.000, 0.000],
                [0.699, 1.210, 0.000],
                [-0.699, 1.210, 0.000],
                [-1.397, 0.000, 0.000],
                [-0.699, -1.210, 0.000],
                [0.699, -1.210, 0.000],
                [2.480, 0.000, 0.000],
                [1.240, 2.148, 0.000],
                [-1.240, 2.148, 0.000],
                [-2.480, 0.000, 0.000],
                [-1.240, -2.148, 0.000],
                [1.240, -2.148, 0.000],
            ],
        ),
    },
    "TIP3": {
        "atoms": Atoms(
            "OH2",
            positions=[
                [0.000000, 0.000000, 0.000000],
                [0.957200, 0.000000, 0.000000],
                [-0.239987, 0.926627, 0.000000],
            ],
        ),
    },
    "MEOH": {
        "atoms": Atoms(
            "COH4",
            positions=[
                [0.000, 0.000, 0.000],
                [1.430, 0.000, 0.000],
                [1.770, 0.910, 0.000],
                [-0.540, 0.900, 0.000],
                [-0.540, -0.450, 0.780],
                [-0.540, -0.450, -0.780],
            ],
        ),
    },
}


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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by molecular pairs
    pairs = df.groupby(["molecule_a", "molecule_b"])
    print(f"Generating multi-panel plots for {len(pairs)} pairs...")

    for (label_a, label_b), group in pairs:
        # Reconstruct dataframes per backend
        df_mp = group[group["backend"] == "learned_multipole"].sort_values("distance_angstrom")
        df_mbd = group[group["backend"] == "learned_mbd"].sort_values("distance_angstrom")
        df_xtb = group[group["backend"] == "xtb_gfn2"].sort_values("distance_angstrom")

        # Create a 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Panel 1: Dimer Geometry Structure (Top-Left)
        ax_geom = axes[0, 0]
        if label_a in MOLECULES and label_b in MOLECULES:
            try:
                mon_a = MOLECULES[label_a]["atoms"]
                mon_b = MOLECULES[label_b]["atoms"]
                dimer, _ = build_rigid_dimer(mon_a, mon_b, distance_angstrom=5.0)
                plot_atoms(dimer, ax_geom, rotation="15x,15y,0z")
                ax_geom.set_title(f"Dimer Structure: {label_a} + {label_b} (5.0 Å)", fontsize=10)
            except Exception as e:
                ax_geom.text(0.5, 0.5, f"Render Error: {e}", ha="center", va="center")
        else:
            ax_geom.text(0.5, 0.5, "Structure N/A", ha="center", va="center")
        ax_geom.axis("off")

        # Check if learned_multipole is available
        has_mp = not df_mp.empty
        has_mbd = not df_mbd.empty
        has_xtb = not df_xtb.empty

        # Panel 2: Low-order Electrostatic components (Top-Middle)
        ax_low = axes[0, 1]
        if has_mp:
            plotted_any = False
            for k in ["0-0", "0-1", "1-1"]:
                col = f"comp_{k}_kcal_mol"
                if col in df_mp.columns:
                    ax_low.plot(
                        df_mp["distance_angstrom"], df_mp[col], marker="o", label=f"comp {k}"
                    )
                    plotted_any = True
            if plotted_any:
                ax_low.set_title("Low-order Components", fontsize=10)
                ax_low.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
                ax_low.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                ax_low.legend(frameon=False, fontsize=8)
            else:
                ax_low.text(0.5, 0.5, "Components N/A", ha="center", va="center")
        else:
            ax_low.text(0.5, 0.5, "Multipoles N/A", ha="center", va="center")

        # Panel 3: Higher-order Electrostatic components (Top-Right)
        ax_high = axes[0, 2]
        if has_mp:
            plotted_any = False
            for k in ["0-2", "1-2", "2-2", "0-3", "1-3", "2-3", "3-3"]:
                col = f"comp_{k}_kcal_mol"
                if col in df_mp.columns:
                    ax_high.plot(
                        df_mp["distance_angstrom"], df_mp[col], marker="o", label=f"comp {k}"
                    )
                    plotted_any = True
            if plotted_any:
                ax_high.set_title("Higher-order Components", fontsize=10)
                ax_high.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
                ax_high.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                ax_high.legend(frameon=False, fontsize=8)
            else:
                ax_high.text(0.5, 0.5, "Components N/A", ha="center", va="center")
        else:
            ax_high.text(0.5, 0.5, "Multipoles N/A", ha="center", va="center")

        # Panel 4: Total Electrostatics comparison (Bottom-Left)
        ax_elec = axes[1, 0]
        if has_mp:
            ax_elec.plot(
                df_mp["distance_angstrom"],
                df_mp["energy_kcal_mol"],
                marker="o",
                color="blue",
                label="Learned Multipoles",
            )
            # Check for CHARMM components in group columns
            if "charmm_ELEC_kcal" in group.columns:
                df_ch = group[group["backend"] == "charmm"].sort_values("distance_angstrom")
                if not df_ch.empty:
                    ax_elec.plot(
                        df_ch["distance_angstrom"],
                        df_ch["charmm_ELEC_kcal"],
                        marker="s",
                        color="cyan",
                        label="CGenFF ELEC",
                    )
            ax_elec.set_title("Total Electrostatics", fontsize=10)
            ax_elec.set_xlabel("Center distance / Å", fontsize=9)
            ax_elec.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
            ax_elec.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax_elec.legend(frameon=False, fontsize=8)
        else:
            ax_elec.text(0.5, 0.5, "Multipoles N/A", ha="center", va="center")

        # Panel 5: Total Dispersion comparison (Bottom-Middle)
        ax_disp = axes[1, 1]
        if has_mbd:
            ax_disp.plot(
                df_mbd["distance_angstrom"],
                df_mbd["energy_kcal_mol"],
                marker="o",
                color="green",
                label="Learned MBD",
            )
            if "charmm_VDW_kcal" in group.columns:
                df_ch = group[group["backend"] == "charmm"].sort_values("distance_angstrom")
                if not df_ch.empty:
                    ax_disp.plot(
                        df_ch["distance_angstrom"],
                        df_ch["charmm_VDW_kcal"],
                        marker="s",
                        color="lightgreen",
                        label="CGenFF VDW",
                    )
            ax_disp.set_title("Total Dispersion", fontsize=10)
            ax_disp.set_xlabel("Center distance / Å", fontsize=9)
            ax_disp.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
            ax_disp.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax_disp.legend(frameon=False, fontsize=8)
        else:
            ax_disp.text(0.5, 0.5, "MBD N/A", ha="center", va="center")

        # Panel 6: Total Interaction comparison (Bottom-Right)
        ax_tot = axes[1, 2]
        # Calculate JAX ML total: multipole + mbd
        if has_mp and has_mbd:
            # Align multipole and mbd on distance
            merged_ml = pd.merge(
                df_mp, df_mbd, on="distance_angstrom", suffixes=("_mp", "_mbd")
            )
            if not merged_ml.empty:
                merged_ml["total_ml_kcal"] = (
                    merged_ml["energy_kcal_mol_mp"] + merged_ml["energy_kcal_mol_mbd"]
                )
                ax_tot.plot(
                    merged_ml["distance_angstrom"],
                    merged_ml["total_ml_kcal"],
                    marker="o",
                    color="crimson",
                    label="ML Total (Multipoles+MBD)",
                )

        if has_xtb:
            ax_tot.plot(
                df_xtb["distance_angstrom"],
                df_xtb["energy_kcal_mol"],
                marker="^",
                color="orange",
                label="xTB GFN2",
            )

        # Check for CHARMM total or ENER
        df_ch = group[group["backend"] == "charmm"].sort_values("distance_angstrom")
        if not df_ch.empty:
            if "charmm_ENER_kcal" in df_ch.columns:
                ax_tot.plot(
                    df_ch["distance_angstrom"],
                    df_ch["charmm_ENER_kcal"],
                    marker="s",
                    color="black",
                    label="CGenFF Total ENER",
                )
            elif "charmm_VDW_kcal" in df_ch.columns and "charmm_ELEC_kcal" in df_ch.columns:
                df_ch["total_ch_kcal"] = (
                    df_ch["charmm_VDW_kcal"] + df_ch["charmm_ELEC_kcal"]
                )
                ax_tot.plot(
                    df_ch["distance_angstrom"],
                    df_ch["total_ch_kcal"],
                    marker="s",
                    color="black",
                    label="CGenFF VDW+ELEC",
                )

        ax_tot.set_title("Total Interaction Energy", fontsize=10, fontweight="bold")
        ax_tot.set_xlabel("Center distance / Å", fontsize=9)
        ax_tot.set_ylabel("Energy / kcal mol$^{-1}$", fontsize=9)
        ax_tot.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax_tot.legend(frameon=False, fontsize=8)

        # Set shared/consistent axis styling where sensible
        for r in range(2):
            for c in range(3):
                ax = axes[r, c]
                if (r, c) != (0, 0):  # skip geometry panel
                    ax.tick_params(axis="both", which="major", labelsize=8)

        plt.suptitle(
            f"Dimer Scan Campaign: {label_a} + {label_b}", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()

        plot_path = args.output_dir / f"{label_a}_{label_b}_multi_panel.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"All multi-panel plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
