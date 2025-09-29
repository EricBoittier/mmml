#!/usr/bin/env python3
"""
Analyze NPZ results from cutoff optimization to debug identical metrics issue.
"""
import numpy as np
import glob
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    analyze_npz_results()
