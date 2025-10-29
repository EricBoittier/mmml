#!/usr/bin/env python3
"""
Filter dataset to only include frames with COM distances in the 3-8 Å range
where the switching function is active.
"""
import numpy as np
import argparse
from pathlib import Path

def filter_dataset(input_file: str, output_file: str, min_distance: float = 3.0, max_distance: float = 8.0, n_atoms_monomer: int = 10):
    """Filter dataset to only include frames within specified COM distance range."""
    
    print(f"Loading dataset: {input_file}")
    dataset = np.load(input_file)
    
    # Get data - copy all available keys
    R_all = dataset['R']
    E_all = dataset.get('E', None)
    F_all = dataset.get('F', None)
    Z_all = dataset.get('Z', None)
    N_all = dataset.get('N', None)
    Q_all = dataset.get('Q', None)
    D_all = dataset.get('D', None)
    
    print(f"Original dataset: {len(R_all)} frames")
    
    # Calculate COM distances
    com_distances = []
    for i in range(len(R_all)):
        com1 = R_all[i][:n_atoms_monomer].mean(axis=0)
        com2 = R_all[i][n_atoms_monomer:].mean(axis=0)
        distance = np.linalg.norm(com1 - com2)
        com_distances.append(distance)
    
    com_distances = np.array(com_distances)
    
    # Filter frames
    mask = (com_distances >= min_distance) & (com_distances <= max_distance)
    filtered_indices = np.where(mask)[0]
    
    print(f"Distance range: {min_distance}-{max_distance} Å")
    print(f"Frames in range: {len(filtered_indices)} out of {len(R_all)} ({100*len(filtered_indices)/len(R_all):.1f}%)")
    
    if len(filtered_indices) == 0:
        print("ERROR: No frames found in the specified distance range!")
        return
    
    # Filter data
    R_filtered = R_all[filtered_indices]
    E_filtered = E_all[filtered_indices] if E_all is not None else None
    F_filtered = F_all[filtered_indices] if F_all is not None else None
    Z_filtered = Z_all[filtered_indices] if Z_all is not None else None
    N_filtered = N_all[filtered_indices] if N_all is not None else None
    Q_filtered = Q_all[filtered_indices] if Q_all is not None else None
    D_filtered = D_all[filtered_indices] if D_all is not None else None
    
    # Save filtered dataset
    print(f"Saving filtered dataset: {output_file}")
    save_data = {'R': R_filtered}
    if E_filtered is not None:
        save_data['E'] = E_filtered
    if F_filtered is not None:
        save_data['F'] = F_filtered
    if Z_filtered is not None:
        save_data['Z'] = Z_filtered
    if N_filtered is not None:
        save_data['N'] = N_filtered
    if Q_filtered is not None:
        save_data['Q'] = Q_filtered
    if D_filtered is not None:
        save_data['D'] = D_filtered
    
    np.savez(output_file, **save_data)
    
    # Print statistics
    filtered_distances = com_distances[filtered_indices]
    print(f"Filtered dataset statistics:")
    print(f"  Frames: {len(R_filtered)}")
    print(f"  Distance range: {filtered_distances.min():.3f} - {filtered_distances.max():.3f} Å")
    print(f"  Mean distance: {filtered_distances.mean():.3f} Å")
    print(f"  Std distance: {filtered_distances.std():.3f} Å")
    
    # Test switching function effectiveness
    print(f"\nTesting switching function effectiveness:")
    try:
        from e3x.nn import smooth_switch, smooth_cutoff
        import jax.numpy as jnp
        
        mm_switch_on = 6.0
        ml_cutoff_values = [0.1, 0.2, 0.3, 1.0]
        
        for ml_cutoff in ml_cutoff_values:
            ml_cutoff_region = mm_switch_on - ml_cutoff
            switching_values = []
            for r in filtered_distances:
                ml_cutoff_fn = 1 - smooth_cutoff(jnp.array(r), cutoff=ml_cutoff_region)
                switch_off_ml = 1 - smooth_switch(jnp.array(r), x0=mm_switch_on-0.01, x1=mm_switch_on)
                ml_scale = ml_cutoff_fn * switch_off_ml
                switching_values.append(float(ml_scale))
            
            switching_values = np.array(switching_values)
            non_zero = np.sum(switching_values > 0.001)
            mean_scale = switching_values.mean()
            print(f"  ml_cutoff={ml_cutoff}: {non_zero}/{len(switching_values)} frames active, mean scale={mean_scale:.3f}")
    except ImportError:
        print("  e3x not available - skipping switching function test")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset by COM distance")
    parser.add_argument("--input", required=True, help="Input dataset file")
    parser.add_argument("--output", required=True, help="Output dataset file")
    parser.add_argument("--min-distance", type=float, default=3.0, help="Minimum COM distance (Å)")
    parser.add_argument("--max-distance", type=float, default=8.0, help="Maximum COM distance (Å)")
    parser.add_argument("--n-atoms-monomer", type=int, default=10, help="Number of atoms per monomer")
    
    args = parser.parse_args()
    
    filter_dataset(
        args.input,
        args.output,
        args.min_distance,
        args.max_distance,
        args.n_atoms_monomer
    )

if __name__ == "__main__":
    main()
