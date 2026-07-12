#!/usr/bin/env python3
"""Run CHARMM locally to calculate CGenFF non-bonded energies and generate multi-panel overlay plots."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from ase import Atoms
from ase.visualize.plot import plot_atoms

from mmml.analysis.dimer_scans import build_rigid_dimer, distance_scan_geometries

# Monomer registry
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

MAP_RESIDUES = {
    "DCM": "DCM",
    "ACE": "ACO",
    "BENZ": "BENZ",
    "TIP3": "TIP3",
    "MEOH": "MEOH",
}


def make_pair_scan(label_a: str, label_b: str, distances: np.ndarray) -> list:
    return list(
        distance_scan_geometries(
            MOLECULES[label_a]["atoms"],
            MOLECULES[label_b]["atoms"],
            distances,
            pair=(label_a, label_b),
            axis=(1.0, 0.0, 0.0),
            center="centroid",
            mol_id_array="mol_id",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("~/Downloads/scan_results_all.csv"),
        help="Path to scan_results_all.csv containing JAX data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dimer_scan_campaign"),
        help="Output directory",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_file():
        print(f"Error: JAX results CSV file not found: {csv_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct residue_geometries directly from MOLECULES to bypass CHARMM's unstable IC build code
    residue_geometries = {
        "DCM": (
            MOLECULES["DCM"]["atoms"].positions[[0, 3, 4, 1, 2]],
            ["C", "H1", "H2", "CL1", "CL2"],
            np.array([6, 1, 1, 17, 17]),
        ),
        "ACO": (
            MOLECULES["ACE"]["atoms"].positions[[3, 0, 1, 2, 4, 5, 6, 7, 8, 9]],
            ["O1", "C1", "C2", "C3", "H21", "H22", "H23", "H31", "H32", "H33"],
            np.array([8, 6, 6, 6, 1, 1, 1, 1, 1, 1]),
        ),
        "BENZ": (
            MOLECULES["BENZ"]["atoms"].positions[[0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]],
            ["CG", "HG", "CD1", "HD1", "CD2", "HD2", "CE1", "HE1", "CE2", "HE2", "CZ", "HZ"],
            np.array([6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1]),
        ),
        "TIP3": (
            MOLECULES["TIP3"]["atoms"].positions[[0, 1, 2]],
            ["OH2", "H1", "H2"],
            np.array([8, 1, 1]),
        ),
        "MEOH": (
            MOLECULES["MEOH"]["atoms"].positions[[0, 1, 2, 3, 4, 5]],
            ["CB", "OG", "HG1", "HB1", "HB2", "HB3"],
            np.array([6, 8, 1, 1, 1, 1]),
        ),
    }

    print("Initializing PyCHARMM locally...")
    try:
        import pycharmm

        # Set bomlev -5 to prevent any coordinate warnings from crashing
        pycharmm.settings.set_bomb_level(-5)

        from mmml.cli.run.md_pbc_suite.ase import _build_cluster_from_composition
        from mmml.interfaces.pycharmmInterface.import_pycharmm import pycharmm_quiet
        from mmml.interfaces.pycharmmInterface.mlpot.cli_common import charmm_energy_row
        from mmml.interfaces.pycharmmInterface.mlpot.setup import (
            setup_default_nbonds,
            sync_charmm_positions,
        )

        pycharmm_quiet()
    except Exception as e:
        print(f"Failed to import/initialize PyCHARMM: {e}")
        sys.exit(1)

    print(f"Loading JAX data from {csv_path}...")
    df_jax = pd.read_csv(csv_path)

    # We will compute CHARMM energies for all unique pairs in the input CSV
    unique_pairs = df_jax.groupby(["molecule_a", "molecule_b"]).groups.keys()
    distances = np.unique(df_jax["distance_angstrom"].values)

    print(f"Found {len(unique_pairs)} pairs across {len(distances)} distances in JAX results.")

    charmm_rows = []

    for idx, (label_a, label_b) in enumerate(unique_pairs, 1):
        print(f"[{idx}/{len(unique_pairs)}] Computing CHARMM energies for {label_a} + {label_b}...")
        res_a = MAP_RESIDUES[label_a]
        res_b = MAP_RESIDUES[label_b]

        # 1. Build cluster PSF and topology
        try:
            _build_cluster_from_composition(
                composition=[(res_a, 1), (res_b, 1)],
                spacing=5.0,
                residue_geometries=residue_geometries,
            )
            setup_default_nbonds()
        except Exception as e:
            print(f"  Error setting up PSF for {label_a} + {label_b}: {e}")
            continue

        # 2. Evaluate for each distance
        geometries = make_pair_scan(label_a, label_b, distances)
        for geom in geometries:
            try:
                sync_charmm_positions(geom.atoms.positions)
                pycharmm.lingo.charmm_script("ENER")
                terms = charmm_energy_row()
                elec = float(terms.get("ELEC", np.nan))
                vdw = float(terms.get("VDW", np.nan))
                tot = float(terms.get("ENER", np.nan))
                charmm_rows.append(
                    {
                        "molecule_a": label_a,
                        "molecule_b": label_b,
                        "distance_angstrom": geom.distance_angstrom,
                        "energy_ev": tot * 0.0433641153,  # kcal/mol to eV
                        "energy_kcal_mol": tot,
                        "backend": "charmm",
                        "charmm_ELEC_kcal": elec,
                        "charmm_VDW_kcal": vdw,
                    }
                )
            except Exception as e:
                print(f"  Error at {geom.distance_angstrom} Å: {e}")

    # Combine CHARMM results with original JAX results
    df_charmm = pd.DataFrame(charmm_rows)
    df_combined = pd.concat([df_jax, df_charmm], ignore_index=True)

    # Save to a new CSV file
    out_csv = args.output_dir / "scan_results_charmm.csv"
    df_combined.to_csv(out_csv, index=False)
    print(f"Combined data saved to {out_csv}")

    # Generate the beautiful multi-panel plots
    print("Generating multi-panel plots with overlays...")
    plot_dir = args.output_dir / "components_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (label_a, label_b), group in df_combined.groupby(["molecule_a", "molecule_b"]):
        df_mp = group[group["backend"] == "learned_multipole"].sort_values("distance_angstrom")
        df_mbd = group[group["backend"] == "learned_mbd"].sort_values("distance_angstrom")
        df_xtb = group[group["backend"] == "xtb_gfn2"].sort_values("distance_angstrom")
        df_ch = group[group["backend"] == "charmm"].sort_values("distance_angstrom")

        has_mp = not df_mp.empty
        has_mbd = not df_mbd.empty
        has_xtb = not df_xtb.empty
        has_ch = not df_ch.empty

        # Shift all energy curves relative to their maximum distance (baseline alignment)
        if has_mp:
            mp_ref = df_mp["energy_kcal_mol"].values[-1]
            df_mp["interaction_energy"] = df_mp["energy_kcal_mol"] - mp_ref
            for col in df_mp.columns:
                if col.startswith("comp_") and col.endswith("_kcal_mol"):
                    comp_ref = df_mp[col].values[-1]
                    df_mp[col + "_shifted"] = df_mp[col] - comp_ref

        if has_mbd:
            mbd_ref = df_mbd["energy_kcal_mol"].values[-1]
            df_mbd["interaction_energy"] = df_mbd["energy_kcal_mol"] - mbd_ref

        if has_xtb:
            xtb_ref = df_xtb["energy_kcal_mol"].values[-1]
            df_xtb["interaction_energy"] = df_xtb["energy_kcal_mol"] - xtb_ref

        if has_ch:
            ch_ref = df_ch["energy_kcal_mol"].values[-1]
            df_ch["interaction_energy"] = df_ch["energy_kcal_mol"] - ch_ref
            
            elec_ref = df_ch["charmm_ELEC_kcal"].values[-1]
            df_ch["charmm_ELEC_shifted"] = df_ch["charmm_ELEC_kcal"] - elec_ref
            
            vdw_ref = df_ch["charmm_VDW_kcal"].values[-1]
            df_ch["charmm_VDW_shifted"] = df_ch["charmm_VDW_kcal"] - vdw_ref

        # Calculate JAX ML total
        has_ml = has_mp and has_mbd
        df_ml_shifted = None
        if has_ml:
            merged_ml = pd.merge(
                df_mp, df_mbd, on="distance_angstrom", suffixes=("_mp", "_mbd")
            )
            if not merged_ml.empty:
                merged_ml["total_ml_shifted"] = (
                    merged_ml["interaction_energy_mp"] + merged_ml["interaction_energy_mbd"]
                )
                df_ml_shifted = merged_ml

        # LaTeX labels mapping for spherical harmonics component symbols
        LATEX_LABELS = {
            "comp_0-0": r"$E_{0,0}\ (q-q)$",
            "comp_0-1": r"$E_{0,1}\ (q-\mu)$",
            "comp_1-1": r"$E_{1,1}\ (\mu-\mu)$",
            "comp_0-2": r"$E_{0,2}\ (q-\Theta)$",
            "comp_1-2": r"$E_{1,2}\ (\mu-\Theta)$",
            "comp_2-2": r"$E_{2,2}\ (\Theta-\Theta)$",
            "comp_0-3": r"$E_{0,3}\ (q-\Omega)$",
            "comp_1-3": r"$E_{1,3}\ (\mu-\Omega)$",
            "comp_2-3": r"$E_{2,3}\ (\Theta-\Omega)$",
            "comp_3-3": r"$E_{3,3}\ (\Omega-\Omega)$",
        }

        # Helper function to plot standardized/scaled lines on left axis
        # Excludes close range points (< 3.8 A) and positive repulsions (y > 0.0) from mean/std scaling determination
        def plot_standardized_line(ax, x, y, label, marker, color, well_scaled_list, comp_key=None, linestyle="-"):
            if len(y) == 0:
                return None
            
            # Exclude close range points and positive repulsion points from scaling determination
            mask = (x >= 3.8) & (y <= 0.0)
            y_well = y[mask]
            if len(y_well) > 0:
                mean = np.mean(y_well)
                std = np.std(y_well) if np.std(y_well) > 0 else 1.0
            else:
                y_well_fallback = y[x >= 3.8]
                if len(y_well_fallback) > 0:
                    mean = np.mean(y_well_fallback)
                    std = np.std(y_well_fallback) if np.std(y_well_fallback) > 0 else 1.0
                else:
                    mean = np.mean(y)
                    std = np.std(y) if np.std(y) > 0 else 1.0
                
            y_scaled = (y - mean) / std
            
            display_label = label
            if comp_key is not None and comp_key in LATEX_LABELS:
                display_label = LATEX_LABELS[comp_key]
                
            ax.plot(x, y_scaled, marker=marker, color=color, linestyle=linestyle, label=display_label)
            
            well_scaled = y_scaled[x >= 3.8]
            if len(well_scaled) > 0:
                # Exclude extreme repulsions in well region
                well_scaled_list.extend(well_scaled[well_scaled < 10.0].values)
                
            return y.values

        # Helper function to setup twin axis on the right showing raw kcal/mol scale
        def setup_twin_axis(ax, x, raw_y, label="Raw / kcal mol$^{-1}$"):
            if raw_y is None or len(raw_y) == 0:
                return
            ax_twin = ax.twinx()
            ax_twin.plot(x, raw_y, alpha=0.0) # invisible line to set scaling
            ax_twin.set_ylabel(label, color="gray", fontsize=8)
            ax_twin.tick_params(axis='y', labelcolor='gray', labelsize=8)

        # Create a 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))

        # Panel 1: Dimer Geometry Structure (Top-Left) zoomed out to show full scan
        ax_geom = axes[0, 0]
        if label_a in MOLECULES and label_b in MOLECULES:
            try:
                mon_a = MOLECULES[label_a]["atoms"]
                mon_b = MOLECULES[label_b]["atoms"]
                
                # Draw monomer A (always fixed, solid)
                plot_atoms(mon_a, ax_geom, rotation="15x,15y,0z")
                
                # Draw monomer B at all portions of the scan (3.0 to 12.0 Å) with alpha progression
                overlays = [
                    (3.0, 0.15),
                    (4.5, 0.30),
                    (6.0, 0.45),
                    (8.0, 0.60),
                    (10.0, 0.75),
                    (12.0, 0.90)
                ]
                
                all_x = list(mon_a.positions[:, 0])
                all_y = list(mon_a.positions[:, 1])
                
                for dist, alpha in overlays:
                    dimer, fragments = build_rigid_dimer(mon_a, mon_b, distance_angstrom=dist)
                    mon_b_shifted = dimer[fragments[1]]
                    
                    start_patch_idx = len(ax_geom.patches)
                    start_line_idx = len(ax_geom.lines)
                    plot_atoms(mon_b_shifted, ax_geom, rotation="15x,15y,0z")
                    
                    # Track coordinates for scaling limits
                    all_x.extend(mon_b_shifted.positions[:, 0])
                    all_y.extend(mon_b_shifted.positions[:, 1])
                    
                    # Set alpha on the overlays
                    for patch in ax_geom.patches[start_patch_idx:]:
                        patch.set_alpha(alpha)
                    for line in ax_geom.lines[start_line_idx:]:
                        line.set_alpha(alpha)
                
                # Zoom out the geom axis to cover all parts of the scan
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                ax_geom.set_xlim(x_min - 2.0, x_max + 2.0)
                ax_geom.set_ylim(y_min - 2.0, y_max + 2.0)
                
                ax_geom.set_title(f"Scan Overview (3.0 Å -> 12.0 Å)", fontsize=10)
            except Exception as e:
                ax_geom.text(0.5, 0.5, f"Render Error: {e}", ha="center", va="center")
        else:
            ax_geom.text(0.5, 0.5, "Structure N/A", ha="center", va="center")
        ax_geom.axis("off")

        # Panel 2: Low-order Electrostatic components (Top-Middle)
        ax_low = axes[0, 1]
        if has_mp:
            well_scaled = []
            raw_ref = None
            for k, color in [("0-0", "blue"), ("0-1", "cyan"), ("1-1", "navy")]:
                col = f"comp_{k}_kcal_mol_shifted"
                if col in df_mp.columns:
                    raw_val = plot_standardized_line(
                        ax_low, df_mp["distance_angstrom"], df_mp[col], f"comp {k}", "o", color, well_scaled, comp_key=f"comp_{k}"
                    )
                    if raw_val is not None:
                        raw_ref = raw_val
            if raw_ref is not None:
                ax_low.set_title("Low-order Components (scaled)", fontsize=10)
                ax_low.set_ylabel("Standardized Scale", fontsize=9)
                ax_low.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                ax_low.legend(frameon=False, fontsize=8)
                if well_scaled:
                    ymin, ymax = min(well_scaled), max(well_scaled)
                    pad = 0.15 * (ymax - ymin) if ymax > ymin else 1.0
                    ax_low.set_ylim(ymin - pad - 0.2, ymax + pad + 0.2)
                setup_twin_axis(ax_low, df_mp["distance_angstrom"], raw_ref, "Raw comp 1-1 / kcal mol$^{-1}$")
            else:
                ax_low.text(0.5, 0.5, "Components N/A", ha="center", va="center")
        else:
            ax_low.text(0.5, 0.5, "Multipoles N/A", ha="center", va="center")

        # Panel 3: Higher-order Electrostatic components (Top-Right)
        ax_high = axes[0, 2]
        if has_mp:
            well_scaled = []
            raw_ref = None
            comps = [
                ("0-2", "green"),
                ("1-2", "lightgreen"),
                ("2-2", "darkgreen"),
                ("0-3", "orange"),
                ("1-3", "gold"),
                ("2-3", "red"),
                ("3-3", "brown")
            ]
            for k, color in comps:
                col = f"comp_{k}_kcal_mol_shifted"
                if col in df_mp.columns:
                    raw_val = plot_standardized_line(
                        ax_high, df_mp["distance_angstrom"], df_mp[col], f"comp {k}", "o", color, well_scaled, comp_key=f"comp_{k}"
                    )
                    if raw_val is not None:
                        raw_ref = raw_val
            if raw_ref is not None:
                ax_high.set_title("Higher-order Components (scaled)", fontsize=10)
                ax_high.set_ylabel("Standardized Scale", fontsize=9)
                ax_high.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
                ax_high.legend(frameon=False, fontsize=8)
                if well_scaled:
                    ymin, ymax = min(well_scaled), max(well_scaled)
                    pad = 0.15 * (ymax - ymin) if ymax > ymin else 1.0
                    ax_high.set_ylim(ymin - pad - 0.2, ymax + pad + 0.2)
                setup_twin_axis(ax_high, df_mp["distance_angstrom"], raw_ref, "Raw comp 2-2 / kcal mol$^{-1}$")
            else:
                ax_high.text(0.5, 0.5, "Components N/A", ha="center", va="center")
        else:
            ax_high.text(0.5, 0.5, "Multipoles N/A", ha="center", va="center")

        # Panel 4: Total Electrostatics comparison (Bottom-Left)
        ax_elec = axes[1, 0]
        if has_mp or has_ch:
            well_scaled = []
            raw_mp = None
            if has_mp:
                raw_mp = plot_standardized_line(
                    ax_elec, df_mp["distance_angstrom"], df_mp["interaction_energy"], "Learned Multipoles", "o", "blue", well_scaled
                )
            if has_ch:
                plot_standardized_line(
                    ax_elec, df_ch["distance_angstrom"], df_ch["charmm_ELEC_shifted"], "CGenFF ELEC", "s", "cyan", well_scaled
                )
            ax_elec.set_title("Total Electrostatics (scaled)", fontsize=10)
            ax_elec.set_xlabel("Center distance / Å", fontsize=9)
            ax_elec.set_ylabel("Standardized Scale", fontsize=9)
            ax_elec.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax_elec.legend(frameon=False, fontsize=8)
            if well_scaled:
                ymin, ymax = min(well_scaled), max(well_scaled)
                pad = 0.15 * (ymax - ymin) if ymax > ymin else 1.0
                ax_elec.set_ylim(ymin - pad - 0.2, ymax + pad + 0.2)
            if raw_mp is not None:
                setup_twin_axis(ax_elec, df_mp["distance_angstrom"], raw_mp, "Raw ELEC / kcal mol$^{-1}$")
        else:
            ax_elec.text(0.5, 0.5, "N/A", ha="center", va="center")

        # Panel 5: Total Dispersion comparison (Bottom-Middle)
        ax_disp = axes[1, 1]
        if has_mbd or has_ch:
            well_scaled = []
            raw_mbd = None
            if has_mbd:
                raw_mbd = plot_standardized_line(
                    ax_disp, df_mbd["distance_angstrom"], df_mbd["interaction_energy"], "Learned MBD", "o", "green", well_scaled
                )
            if has_ch:
                plot_standardized_line(
                    ax_disp, df_ch["distance_angstrom"], df_ch["charmm_VDW_shifted"], "CGenFF VDW", "s", "lightgreen", well_scaled
                )
            ax_disp.set_title("Total Dispersion (scaled)", fontsize=10)
            ax_disp.set_xlabel("Center distance / Å", fontsize=9)
            ax_disp.set_ylabel("Standardized Scale", fontsize=9)
            ax_disp.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax_disp.legend(frameon=False, fontsize=8)
            if well_scaled:
                ymin, ymax = min(well_scaled), max(well_scaled)
                pad = 0.15 * (ymax - ymin) if ymax > ymin else 1.0
                ax_disp.set_ylim(ymin - pad - 0.2, ymax + pad + 0.2)
            if raw_mbd is not None:
                setup_twin_axis(ax_disp, df_mbd["distance_angstrom"], raw_mbd, "Raw VDW / kcal mol$^{-1}$")
        else:
            ax_disp.text(0.5, 0.5, "N/A", ha="center", va="center")

        # Panel 6: Total Interaction comparison (Bottom-Right)
        ax_tot = axes[1, 2]
        well_scaled = []
        raw_tot = None
        if has_ml and df_ml_shifted is not None:
            raw_tot = plot_standardized_line(
                ax_tot, df_ml_shifted["distance_angstrom"], df_ml_shifted["total_ml_shifted"], "ML Total (Multipoles+MBD)", "o", "crimson", well_scaled
            )

        if has_xtb:
            plot_standardized_line(
                ax_tot, df_xtb["distance_angstrom"], df_xtb["interaction_energy"], "xTB GFN2", "^", "orange", well_scaled
            )

        if has_ch:
            plot_standardized_line(
                ax_tot, df_ch["distance_angstrom"], df_ch["interaction_energy"], "CGenFF Total", "s", "black", well_scaled
            )

        ax_tot.set_title("Total Interaction Energy (scaled)", fontsize=10, fontweight="bold")
        ax_tot.set_xlabel("Center distance / Å", fontsize=9)
        ax_tot.set_ylabel("Standardized Scale", fontsize=9)
        ax_tot.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax_tot.legend(frameon=False, fontsize=8)
        if well_scaled:
            ymin, ymax = min(well_scaled), max(well_scaled)
            pad = 0.15 * (ymax - ymin) if ymax > ymin else 1.0
            ax_tot.set_ylim(ymin - pad - 0.2, ymax + pad + 0.2)
        if raw_tot is not None:
            setup_twin_axis(ax_tot, df_ml_shifted["distance_angstrom"], raw_tot, "Raw Total / kcal mol$^{-1}$")

        # Final layout adjustments
        for r in range(2):
            for c in range(3):
                ax = axes[r, c]
                if (r, c) != (0, 0):
                    ax.tick_params(axis="both", which="major", labelsize=8)

        plt.suptitle(
            f"Dimer Scan Campaign: {label_a} + {label_b} (standard-scaled per curve)", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()

        plot_path = plot_dir / f"{label_a}_{label_b}_multi_panel.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"All multi-panel overlay plots saved to {plot_dir}")


if __name__ == "__main__":
    main()
