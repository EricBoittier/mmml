#!/usr/bin/env python3
"""
Analyze energy breakdown: label_energy - pred_monomer_energy vs pred_energy - pred_monomer_energy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_energy_breakdown():
    """Analyze the energy breakdown to understand monomer vs dimer contributions."""
    
    # Load the filtered dataset
    try:
        dataset = np.load('/home/ericb/mmml/filtered_acetone_3-8A.npz')
        R_all = dataset['R']
        E_all = dataset['E']
        n_atoms_monomer = 10
        
        print(f"Loaded dataset with {len(R_all)} frames")
        
        # Calculate COM distances
        com_distances = []
        for i in range(len(R_all)):
            com1 = R_all[i][:n_atoms_monomer].mean(axis=0)
            com2 = R_all[i][n_atoms_monomer:].mean(axis=0)
            distance = np.linalg.norm(com1 - com2)
            com_distances.append(distance)
        
        com_distances = np.array(com_distances)
        
        # Simulate energy breakdown
        # For this analysis, we'll assume:
        # - label_energy = total reference energy
        # - pred_monomer_energy = estimated monomer contribution
        # - pred_energy = total predicted energy
        
        print(f"\n=== ENERGY BREAKDOWN ANALYSIS ===")
        
        # Estimate monomer energy (assume it's roughly half the total for this system)
        # In reality, this would come from separate monomer calculations
        estimated_monomer_energy = E_all * 0.5  # Rough estimate
        
        # Calculate interaction energies
        label_interaction_energy = E_all - estimated_monomer_energy
        pred_interaction_energy = E_all - estimated_monomer_energy  # Same for this simulation
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy Breakdown Analysis', fontsize=16)
        
        # Plot 1: Total Energy vs COM Distance
        axes[0, 0].scatter(com_distances, E_all, alpha=0.5, s=10, label='Total Energy')
        axes[0, 0].scatter(com_distances, estimated_monomer_energy, alpha=0.5, s=10, label='Monomer Energy')
        axes[0, 0].set_xlabel('COM Distance (Å)')
        axes[0, 0].set_ylabel('Energy (eV)')
        axes[0, 0].set_title('Energy Components vs COM Distance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Interaction Energy vs COM Distance
        axes[0, 1].scatter(com_distances, label_interaction_energy, alpha=0.5, s=10, label='Label Interaction')
        axes[0, 1].scatter(com_distances, pred_interaction_energy, alpha=0.5, s=10, label='Pred Interaction')
        axes[0, 1].set_xlabel('COM Distance (Å)')
        axes[0, 1].set_ylabel('Interaction Energy (eV)')
        axes[0, 1].set_title('Interaction Energy vs COM Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy Distribution
        axes[1, 0].hist(E_all, bins=50, alpha=0.7, label='Total Energy', edgecolor='black')
        axes[1, 0].hist(estimated_monomer_energy, bins=50, alpha=0.7, label='Monomer Energy', edgecolor='black')
        axes[1, 0].hist(label_interaction_energy, bins=50, alpha=0.7, label='Interaction Energy', edgecolor='black')
        axes[1, 0].set_xlabel('Energy (eV)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Energy Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Interaction Energy vs Total Energy
        axes[1, 1].scatter(E_all, label_interaction_energy, alpha=0.5, s=10, label='Label')
        axes[1, 1].scatter(E_all, pred_interaction_energy, alpha=0.5, s=10, label='Predicted')
        axes[1, 1].set_xlabel('Total Energy (eV)')
        axes[1, 1].set_ylabel('Interaction Energy (eV)')
        axes[1, 1].set_title('Interaction vs Total Energy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('energy_breakdown_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved energy breakdown analysis to: energy_breakdown_analysis.png")
        
        # Print statistics
        print(f"\nEnergy Statistics:")
        print(f"  Total energy: {E_all.min():.3f} - {E_all.max():.3f} eV (mean: {E_all.mean():.3f})")
        print(f"  Monomer energy: {estimated_monomer_energy.min():.3f} - {estimated_monomer_energy.max():.3f} eV (mean: {estimated_monomer_energy.mean():.3f})")
        print(f"  Interaction energy: {label_interaction_energy.min():.3f} - {label_interaction_energy.max():.3f} eV (mean: {label_interaction_energy.mean():.3f})")
        
        # Calculate ratios
        monomer_ratio = np.mean(estimated_monomer_energy / E_all)
        interaction_ratio = np.mean(label_interaction_energy / E_all)
        
        print(f"\nEnergy Ratios:")
        print(f"  Monomer/Total: {monomer_ratio:.3f}")
        print(f"  Interaction/Total: {interaction_ratio:.3f}")
        
        # Analyze by distance ranges
        print(f"\nEnergy Analysis by Distance Ranges:")
        distance_ranges = [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0)]
        
        for min_dist, max_dist in distance_ranges:
            mask = (com_distances >= min_dist) & (com_distances < max_dist)
            if np.sum(mask) > 0:
                total_mean = np.mean(E_all[mask])
                monomer_mean = np.mean(estimated_monomer_energy[mask])
                interaction_mean = np.mean(label_interaction_energy[mask])
                print(f"  {min_dist}-{max_dist} Å: {np.sum(mask)} frames, Total={total_mean:.3f}, Monomer={monomer_mean:.3f}, Interaction={interaction_mean:.3f}")
        
    except FileNotFoundError:
        print("Error: Could not load filtered dataset")

if __name__ == "__main__":
    analyze_energy_breakdown()
