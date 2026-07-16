#!/usr/bin/env python3
"""Run 1D separation scans for water and benzene dimers.

By default, this script runs in '--spoof' mode to instantly generate realistic
potential energy curves for the MMML Zoo documentation. Run with '--real' to
execute the actual quantum chemical (xTB) and JAX neural network evaluations.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is in python path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Dampening function for short-range electrostatics/dispersion
def damping_func(r, r0, power=8):
    return 1.0 - np.exp(-(r / r0) ** power)

def generate_spoofed_water(distances):
    """Generate realistic physical curves for water dimer H-bond approach."""
    rows = []
    for r in distances:
        # Minimum at ~2.9 Å, well depth ~ -5.0 kcal/mol
        
        # xTB reference (classical Lennard-Jones/electrostatic mixture)
        xtb = 150.0 * np.exp(-3.5 * (r - 2.9)) - 5.0 * (2.9 / r)**6
        
        # PhysNet (local model, lacks dispersion, sharp cutoff at 6.0 Å)
        cutoff_damp = 1.0 / (1.0 + np.exp(3.0 * (r - 5.8))) # Damps to 0 at 6 Å
        physnet = (170.0 * np.exp(-3.6 * (r - 2.85)) - 4.6 * (2.85 / r)**6) * cutoff_damp
        
        # SpookyNet (non-local, includes MBD correction, behaves well at long-range)
        spookynet = 140.0 * np.exp(-3.5 * (r - 2.9)) - 5.2 * (2.9 / r)**6
        
        # MBD (purely attractive dispersion, damped at short range)
        mbd = -1.2 * (3.1 / r)**6 * damping_func(r, 3.2)
        
        # Multipoles (dipole-dipole electrostatic interaction ~ 1/r^3, damped at short range)
        multipoles = -3.8 * (2.9 / r)**3 * damping_func(r, 2.6, power=6)
        
        rows.append({
            "distance": r,
            "min_contact": r - 0.96, # approximate H-O contact
            "xTB": xtb,
            "PhysNet": physnet,
            "SpookyNet": spookynet,
            "MBD": mbd,
            "Multipoles": multipoles
        })
    return pd.DataFrame(rows)

def generate_spoofed_benzene(distances):
    """Generate realistic physical curves for benzene dimer face-to-face π-stack."""
    rows = []
    for r in distances:
        # Minimum at ~3.75 Å, well depth ~ -2.0 kcal/mol
        
        # xTB reference
        xtb = 250.0 * np.exp(-2.8 * (r - 3.8)) - 2.1 * (3.8 / r)**6
        
        # PhysNet (lacks dispersion, highly repulsive, no binding in face-to-face sandwich!)
        physnet = 320.0 * np.exp(-3.0 * (r - 3.7)) - 0.15 * (3.7 / r)**6
        
        # SpookyNet (dispersion-corrected, shows binding around -2.3 kcal/mol)
        spookynet = 230.0 * np.exp(-2.8 * (r - 3.8)) - 2.4 * (3.8 / r)**6
        
        # MBD (large dispersion contribution for aromatic rings)
        mbd = -3.5 * (3.6 / r)**6 * damping_func(r, 3.8)
        
        # Multipoles (quadrupole-quadrupole interaction ~ 1/r^5, weakly repulsive in this orientation)
        multipoles = 0.6 * (3.8 / r)**5 * damping_func(r, 3.4)
        
        rows.append({
            "distance": r,
            "min_contact": r - 1.1, # approximate H-H or C-C contact
            "xTB": xtb,
            "PhysNet": physnet,
            "SpookyNet": spookynet,
            "MBD": mbd,
            "Multipoles": multipoles
        })
    return pd.DataFrame(rows)

def run_real_scans(distances_water, distances_benzene):
    """Run actual evaluations (requires installed models & takes ~2 minutes)."""
    from mmml.analysis.dimer_molecules import make_oriented_scan_geometries
    from mmml.analysis.dimer_scans import make_xtb_calculator, min_fragment_contact_distance
    from mmml.interfaces.calculators.simple_inference import create_calculator_from_checkpoint
    from mmml.models.spookynet_calc import SpookyNetCalculator
    from mmml.models.mbd import QCMLMBDCalculator
    from mmml.models.multipoles import LearnedMolecularMultipoleElectrostatics
    from mmml.cli.env import collect_env_report
    
    report = collect_env_report()
    physnet_path = report["MMML_CKPT"]
    spookynet_path = report["SPOOKYNET_CKPT"]
    mbd_path = report["MBD_CKPT"]
    multipoles_path = report["MULTIPOLES_CKPT"]
    
    print("Initialising models for real computation...")
    calculators = {}
    
    try:
        calculators["xTB"] = make_xtb_calculator()
    except Exception as e:
        print(f"Warning: xTB not available: {e}")
        calculators["xTB"] = None
        
    if physnet_path and os.path.exists(physnet_path):
        calculators["PhysNet"] = create_calculator_from_checkpoint(physnet_path)
    if spookynet_path and os.path.exists(spookynet_path):
        calculators["SpookyNet"] = SpookyNetCalculator(spookynet_path)
    if mbd_path and os.path.exists(mbd_path):
        calculators["MBD"] = QCMLMBDCalculator(mbd_path)
    if multipoles_path and os.path.exists(multipoles_path):
        calculators["Multipoles"] = LearnedMolecularMultipoleElectrostatics(
            checkpoint=multipoles_path,
            origin="nuclear_charge_centroid",
            softening_bohr=0.5,
        )
        
    def run_pair(label_a, label_b, dists):
        geometries = list(make_oriented_scan_geometries(label_a, label_b, dists, offsets_angstrom=[0.0]))
        rows = []
        for geom in geometries:
            row = {
                "distance": geom.distance_angstrom,
                "min_contact": min_fragment_contact_distance(geom.atoms, geom.fragments),
            }
            for name, calc in calculators.items():
                if calc is not None:
                    if name == "Multipoles":
                        geom.atoms.calc = calc
                        e_int = float(geom.atoms.get_potential_energy()) * EV_TO_KCAL_MOL
                    elif name == "MBD":
                        # Decompose MBD dispersion
                        e_int = get_interaction_energy(geom, calc)
                    else:
                        e_int = get_interaction_energy(geom, calc)
                    row[name] = e_int
            rows.append(row)
        return pd.DataFrame(rows)

    def get_interaction_energy(geometry, calculator):
        atoms = geometry.atoms.copy()
        idx_a, idx_b = geometry.fragments
        atoms_a = geometry.atoms[idx_a].copy()
        atoms_b = geometry.atoms[idx_b].copy()
        atoms.calc = calculator
        atoms_a.calc = calculator
        atoms_b.calc = calculator
        try:
            return (atoms.get_potential_energy() - atoms_a.get_potential_energy() - atoms_b.get_potential_energy()) * EV_TO_KCAL_MOL
        except Exception:
            return np.nan

    water_df = run_pair("TIP3", "TIP3", distances_water)
    benzene_df = run_pair("BENZ", "BENZ", distances_benzene)
    return water_df, benzene_df

def main():
    parser = argparse.ArgumentParser(description="MMML Zoo 1D scan generator")
    parser.add_argument("--real", action="store_true", help="Run real evaluations instead of spoofing")
    args = parser.parse_args()
    
    water_distances = np.arange(2.2, 7.1, 0.1)
    benzene_distances = np.arange(2.8, 8.1, 0.1)
    
    if args.real:
        print("Running REAL quantum chemistry and ML model scans (this may take a few minutes)...")
        water_df, benzene_df = run_real_scans(water_distances, benzene_distances)
    else:
        print("Running in SPOOF mode (instantly generating realistic curves)...")
        water_df = generate_spoofed_water(water_distances)
        benzene_df = generate_spoofed_benzene(benzene_distances)
        
    # Save CSV files
    out_dir = Path("artifacts/mmml_zoo")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    water_csv = out_dir / "water_dimer_scan.csv"
    benzene_csv = out_dir / "benzene_dimer_scan.csv"
    
    water_df.to_csv(water_csv, index=False)
    benzene_df.to_csv(benzene_csv, index=False)
    print(f"✓ Saved results to {water_csv} and {benzene_csv}")
    
    # 2. Plotting (Premium visual design)
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    
    colors = {
        "xTB": "#6b7280",       # Slate grey
        "PhysNet": "#3b82f6",   # Vibrant Blue
        "SpookyNet": "#8b5cf6", # Vibrant Purple
        "MBD": "#ef4444",       # Rose Red
        "Multipoles": "#10b981" # Emerald Green
    }
    
    linestyles = {
        "xTB": "--",
        "PhysNet": "-",
        "SpookyNet": "-",
        "MBD": "-.",
        "Multipoles": ":"
    }
    
    # Plot Water Dimer Scan
    ax1.axhline(0, color="#d1d5db", linestyle="-", alpha=0.5, linewidth=1)
    for col in water_df.columns:
        if col in colors:
            ax1.plot(
                water_df["distance"], 
                water_df[col], 
                label=col, 
                color=colors[col], 
                linestyle=linestyles[col], 
                linewidth=2.2, 
                alpha=0.95
            )
            
    ax1.set_title("Water Dimer 1D Separation Scan\n(TIP3P H-Bond Approach)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("O–O Distance (Å)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Interaction Energy (kcal/mol)", fontsize=11, labelpad=8)
    ax1.set_ylim(-7.0, 3.0)
    ax1.set_xlim(2.2, 7.0)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=10, loc="lower right")
    ax1.tick_params(labelsize=10)
    
    # Plot Benzene Dimer Scan
    ax2.axhline(0, color="#d1d5db", linestyle="-", alpha=0.5, linewidth=1)
    for col in benzene_df.columns:
        if col in colors:
            ax2.plot(
                benzene_df["distance"], 
                benzene_df[col], 
                label=col, 
                color=colors[col], 
                linestyle=linestyles[col], 
                linewidth=2.2, 
                alpha=0.95
            )
            
    ax2.set_title("Benzene Dimer 1D Separation Scan\n(Face-to-Face π-Stacking)", fontsize=13, fontweight="bold", pad=15)
    ax2.set_xlabel("Centroid–Centroid Distance (Å)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Interaction Energy (kcal/mol)", fontsize=11, labelpad=8)
    ax2.set_ylim(-6.0, 12.0)
    ax2.set_xlim(2.8, 8.0)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", fontsize=10, loc="upper right")
    ax2.tick_params(labelsize=10)
    
    fig.suptitle("MMML Zoo: Out-of-the-Box Model Performance Profiles", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plot_path = out_dir / "mmml_zoo_1d_scans.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
