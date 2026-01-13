#!/usr/bin/env python3
"""
Analyze NPZ results from cutoff optimization to debug identical metrics issue.
"""
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

def analyze_npz_results():
    """Load and analyze all NPZ files from cutoff optimization."""
    
    # Find all NPZ files
    npz_files = glob.glob("cutoff_opt_*.npz")
    if not npz_files:
        print("No NPZ files found. Run the optimization first.")
        return
    
    print(f"Found {len(npz_files)} NPZ files")
    
    # Load all data
    all_data = {}
    for file in sorted(npz_files):
        data = np.load(file)
        all_data[file] = {
            'ml_cutoff': data['ml_cutoffs'][0],  # Should be single value
            'mm_switch_on': data['mm_switch_ons'][0],
            'mm_cutoff': data['mm_cutoffs'][0],
            'mse_energy': data['mse_energies'][0],
            'mse_forces': data['mse_forces'][0],
            'objective': data['objectives'][0],
        }
    
    # Create arrays for analysis
    ml_cutoffs = np.array([d['ml_cutoff'] for d in all_data.values()])
    mm_switch_ons = np.array([d['mm_switch_on'] for d in all_data.values()])
    mm_cutoffs = np.array([d['mm_cutoff'] for d in all_data.values()])
    mse_energies = np.array([d['mse_energy'] for d in all_data.values()])
    mse_forces = np.array([d['mse_forces'] for d in all_data.values()])
    objectives = np.array([d['objective'] for d in all_data.values()])
    
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Number of unique ML cutoffs: {len(np.unique(ml_cutoffs))}")
    print(f"Number of unique MM switch-ons: {len(np.unique(mm_switch_ons))}")
    print(f"Number of unique MM cutoffs: {len(np.unique(mm_cutoffs))}")
    print(f"Number of unique objectives: {len(np.unique(objectives))}")
    
    print(f"\nObjective statistics:")
    print(f"  Min: {objectives.min():.6f}")
    print(f"  Max: {objectives.max():.6f}")
    print(f"  Mean: {objectives.mean():.6f}")
    print(f"  Std: {objectives.std():.6f}")
    
    # Check for identical results
    unique_objectives, counts = np.unique(objectives, return_counts=True)
    duplicates = unique_objectives[counts > 1]
    
    if len(duplicates) > 0:
        print(f"\nWARNING: Found {len(duplicates)} duplicate objective values!")
        for dup_obj in duplicates:
            dup_indices = np.where(objectives == dup_obj)[0]
            print(f"  Objective {dup_obj:.6f} appears {len(dup_indices)} times")
            for idx in dup_indices:
                file_name = list(all_data.keys())[idx]
                data = all_data[file_name]
                print(f"    {file_name}: ml={data['ml_cutoff']}, mm_on={data['mm_switch_on']}, mm_cut={data['mm_cutoff']}")
    else:
        print("\n✓ All objectives are unique - cutoff optimization is working!")
    
    # Show best results
    best_idx = np.argmin(objectives)
    best_file = list(all_data.keys())[best_idx]
    best_data = all_data[best_file]
    
    print(f"\nBest result:")
    print(f"  File: {best_file}")
    print(f"  ML cutoff: {best_data['ml_cutoff']}")
    print(f"  MM switch-on: {best_data['mm_switch_on']}")
    print(f"  MM cutoff: {best_data['mm_cutoff']}")
    print(f"  Objective: {best_data['objective']:.6f}")
    print(f"  MSE Energy: {best_data['mse_energy']:.6f}")
    print(f"  MSE Forces: {best_data['mse_forces']:.6f}")
    
    # Create summary table
    print(f"\n=== SUMMARY TABLE ===")
    print(f"{'File':<30} {'ML':<6} {'MM_on':<6} {'MM_cut':<6} {'Objective':<12} {'MSE_E':<12} {'MSE_F':<12}")
    print("-" * 90)
    
    # Sort by objective
    sorted_indices = np.argsort(objectives)
    for idx in sorted_indices:
        file_name = list(all_data.keys())[idx]
        data = all_data[file_name]
        print(f"{file_name:<30} {data['ml_cutoff']:<6.1f} {data['mm_switch_on']:<6.1f} {data['mm_cutoff']:<6.1f} "
              f"{data['objective']:<12.6f} {data['mse_energy']:<12.6f} {data['mse_forces']:<12.6f}")
    
    # Create plots
    create_analysis_plots(all_data)

def create_analysis_plots(all_data):
    """Create analysis plots for the optimization results."""
    
    # Convert to DataFrame for easier plotting
    df_data = []
    for file_name, data in all_data.items():
        df_data.append({
            'file': file_name,
            'ml_cutoff': data['ml_cutoff'],
            'mm_switch_on': data['mm_switch_on'],
            'mm_cutoff': data['mm_cutoff'],
            'mse_energy': data['mse_energy'],
            'mse_forces': data['mse_forces'],
            'objective': data['objective']
        })
    
    df = pd.DataFrame(df_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cutoff Optimization Analysis', fontsize=16)
    
    # Plot 1: Error vs ML Cutoff
    axes[0, 0].scatter(df['ml_cutoff'], df['mse_energy'], alpha=0.7, s=50)
    axes[0, 0].set_xlabel('ML Cutoff (Å)')
    axes[0, 0].set_ylabel('MSE Energy')
    axes[0, 0].set_title('Energy Error vs ML Cutoff')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error vs MM Switch-on
    axes[0, 1].scatter(df['mm_switch_on'], df['mse_energy'], alpha=0.7, s=50)
    axes[0, 1].set_xlabel('MM Switch-on (Å)')
    axes[0, 1].set_ylabel('MSE Energy')
    axes[0, 1].set_title('Energy Error vs MM Switch-on')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error vs MM Cutoff
    axes[1, 0].scatter(df['mm_cutoff'], df['mse_energy'], alpha=0.7, s=50)
    axes[1, 0].set_xlabel('MM Cutoff (Å)')
    axes[1, 0].set_ylabel('MSE Energy')
    axes[1, 0].set_title('Energy Error vs MM Cutoff')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Objective vs ML Cutoff
    axes[1, 1].scatter(df['ml_cutoff'], df['objective'], alpha=0.7, s=50)
    axes[1, 1].set_xlabel('ML Cutoff (Å)')
    axes[1, 1].set_ylabel('Objective')
    axes[1, 1].set_title('Objective vs ML Cutoff')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cutoff_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved analysis plots to: cutoff_optimization_analysis.png")
    
    # Create detailed energy analysis
    create_energy_analysis_plots(all_data)

def create_energy_analysis_plots(all_data):
    """Create detailed energy analysis plots."""
    
    # Load the filtered dataset to get reference energies
    try:
        dataset = np.load('/home/ericb/mmml/filtered_acetone_3-8A.npz')
        R_all = dataset['R']
        E_all = dataset['E']
        n_atoms_monomer = 10
        
        print(f"\n=== DETAILED ENERGY ANALYSIS ===")
        print(f"Loaded dataset with {len(R_all)} frames")
        
        # Calculate COM distances
        com_distances = []
        for i in range(len(R_all)):
            com1 = R_all[i][:n_atoms_monomer].mean(axis=0)
            com2 = R_all[i][n_atoms_monomer:].mean(axis=0)
            distance = np.linalg.norm(com1 - com2)
            com_distances.append(distance)
        
        com_distances = np.array(com_distances)
        
        # Create energy analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Energy Analysis', fontsize=16)
        
        # Plot 1: COM Distance Distribution
        axes[0, 0].hist(com_distances, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('COM Distance (Å)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('COM Distance Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Reference Energy Distribution
        axes[0, 1].hist(E_all, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Reference Energy (eV)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reference Energy Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy vs COM Distance
        axes[1, 0].scatter(com_distances, E_all, alpha=0.5, s=10)
        axes[1, 0].set_xlabel('COM Distance (Å)')
        axes[1, 0].set_ylabel('Reference Energy (eV)')
        axes[1, 0].set_title('Energy vs COM Distance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: MSE Energy vs ML Cutoff (colored by MM switch-on)
        df_data = []
        for file_name, data in all_data.items():
            df_data.append({
                'ml_cutoff': data['ml_cutoff'],
                'mm_switch_on': data['mm_switch_on'],
                'mse_energy': data['mse_energy']
            })
        
        df = pd.DataFrame(df_data)
        unique_mm_switch_ons = df['mm_switch_on'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_mm_switch_ons)))
        
        for i, mm_switch_on in enumerate(unique_mm_switch_ons):
            mask = df['mm_switch_on'] == mm_switch_on
            axes[1, 1].scatter(df[mask]['ml_cutoff'], df[mask]['mse_energy'], 
                             label=f'MM switch-on={mm_switch_on}', 
                             color=colors[i], alpha=0.7, s=50)
        
        axes[1, 1].set_xlabel('ML Cutoff (Å)')
        axes[1, 1].set_ylabel('MSE Energy')
        axes[1, 1].set_title('MSE Energy vs ML Cutoff (by MM switch-on)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_energy_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved detailed energy analysis to: detailed_energy_analysis.png")
        
        # Print statistics
        print(f"\nDataset Statistics:")
        print(f"  COM distances: {com_distances.min():.3f} - {com_distances.max():.3f} Å (mean: {com_distances.mean():.3f})")
        print(f"  Reference energies: {E_all.min():.3f} - {E_all.max():.3f} eV (mean: {E_all.mean():.3f})")
        
    except FileNotFoundError:
        print("Warning: Could not load filtered dataset for detailed analysis")

if __name__ == "__main__":
    analyze_npz_results()
