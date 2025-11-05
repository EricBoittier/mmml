#!/usr/bin/env python3
"""
Quick test of glycol.npz data loading and basic statistics.
"""

import numpy as np
from pathlib import Path

def main():
    data_path = Path(__file__).parent / "glycol.npz"
    print(f"Loading {data_path}...")
    
    data = np.load(data_path)
    
    print("\nDataset Keys:")
    for key in ['Z', 'R', 'F', 'E', 'N', 'D']:
        if key in data:
            print(f"  {key}: {data[key].shape}")
    
    # Extract main arrays
    Z = data['Z']  # atomic numbers
    R = data['R']  # positions
    F = data['F']  # forces
    E = data['E']  # energies
    N = data['N']  # number of atoms
    
    print(f"\nDataset Statistics:")
    print(f"  Total molecules: {len(E)}")
    print(f"  Max atoms per molecule: {Z.shape[1]}")
    print(f"  Avg atoms per molecule: {N.mean():.1f}")
    print(f"\nEnergy Statistics:")
    print(f"  Mean: {E.mean():.4f} eV")
    print(f"  Std:  {E.std():.4f} eV")
    print(f"  Min:  {E.min():.4f} eV")
    print(f"  Max:  {E.max():.4f} eV")
    print(f"\nForces Statistics:")
    print(f"  Mean magnitude: {np.linalg.norm(F, axis=-1).mean():.4f} eV/Å")
    print(f"  Max magnitude:  {np.linalg.norm(F, axis=-1).max():.4f} eV/Å")
    
    print("\n✅ Data loaded successfully!")

if __name__ == "__main__":
    main()

