#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory

def analyze_trajectory(traj_path):
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} does not exist.")
        return None

    traj = Trajectory(traj_path)
    rgs = []
    end_to_end = []
    energies = []
    
    for atoms in traj:
        pos = atoms.get_positions()
        # Radius of gyration
        com = atoms.get_center_of_mass()
        rg = np.sqrt(np.mean(np.sum((pos - com)**2, axis=1)))
        rgs.append(rg)
        
        # End-to-end distance
        ete = np.linalg.norm(pos[0] - pos[-1])
        end_to_end.append(ete)
        
        # Potential energy
        try:
            energies.append(atoms.get_potential_energy())
        except Exception:
            energies.append(0.0)
            
    traj.close()
    return {
        "rg": np.array(rgs),
        "ete": np.array(end_to_end),
        "energy": np.array(energies),
    }

def main():
    if len(sys.argv) < 4:
        print("Usage: python examples/compare_mm_ml.py <mm_traj> <ml_traj> <output_plot_name>")
        sys.exit(1)
        
    mm_path = sys.argv[1]
    ml_path = sys.argv[2]
    out_img = sys.argv[3]
    
    print(f"Analyzing MM trajectory: {mm_path}")
    mm_data = analyze_trajectory(mm_path)
    
    print(f"Analyzing ML trajectory: {ml_path}")
    ml_data = analyze_trajectory(ml_path)
    
    if mm_data is None or ml_data is None:
        print("Failed to load trajectories.")
        sys.exit(1)
        
    # Generate plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot Radius of Gyration
    axes[0].plot(mm_data["rg"], label="Classical MM", color="tab:blue", alpha=0.8)
    axes[0].plot(ml_data["rg"], label="ML (SpookyNet)", color="tab:orange", alpha=0.8)
    axes[0].set_ylabel("Radius of Gyration (Å)")
    axes[0].set_title("Structural Comparison: MM vs ML")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Plot End-to-End Distance
    axes[1].plot(mm_data["ete"], label="Classical MM", color="tab:blue", alpha=0.8)
    axes[1].plot(ml_data["ete"], label="ML (SpookyNet)", color="tab:orange", alpha=0.8)
    axes[1].set_ylabel("End-to-End Distance (Å)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot Potential Energy
    axes[2].plot(mm_data["energy"], label="Classical MM", color="tab:blue", alpha=0.8)
    axes[2].plot(ml_data["energy"], label="ML (SpookyNet)", color="tab:orange", alpha=0.8)
    axes[2].set_ylabel("Potential Energy (eV)")
    axes[2].set_xlabel("Frame Index")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    print(f"Plot saved successfully to: {out_img}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"MM Radius of Gyration:  mean = {np.mean(mm_data['rg']):.3f} Å, std = {np.std(mm_data['rg']):.3f} Å")
    print(f"ML Radius of Gyration:  mean = {np.mean(ml_data['rg']):.3f} Å, std = {np.std(ml_data['rg']):.3f} Å")
    print(f"MM End-to-End Distance: mean = {np.mean(mm_data['ete']):.3f} Å, std = {np.std(mm_data['ete']):.3f} Å")
    print(f"ML End-to-End Distance: mean = {np.mean(ml_data['ete']):.3f} Å, std = {np.std(ml_data['ete']):.3f} Å")

if __name__ == "__main__":
    main()
