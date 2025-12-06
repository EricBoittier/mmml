#!/usr/bin/env python3
"""
Example: How to debug hybrid calculator forces after initialization from batch.
"""

import numpy as np
from mmml.utils.simulation_utils import initialize_simulation_from_batch
from debug_hybrid_forces_from_batch import debug_hybrid_calculator_forces, compare_ml_mm_contributions

# Example usage (adapt to your actual code):
"""
# Your existing code:
from mmml.utils.simulation_utils import (
    reorder_atoms_to_match_pycharmm,
    initialize_simulation_from_batch,
    initialize_multiple_simulations,
)

# Initialize first simulation from batch 0
atoms, hybrid_calc = initialize_simulation_from_batch(
    train_batches_copy[0], 
    calculator_factory_lj_optimized, 
    None, 
    args
)

# ADD THIS: Debug the forces
# Option 1: Basic debugging (no reference forces)
diagnostics = debug_hybrid_calculator_forces(atoms, hybrid_calc, verbose=True)

# Option 2: Compare with reference forces (if you have them)
ref_forces = ...  # Your reference forces array (n_atoms, 3)
diagnostics = debug_hybrid_calculator_forces(
    atoms, 
    hybrid_calc, 
    ref_forces=ref_forces,
    verbose=True
)

# Option 3: Compare ML vs MM contributions
ml_mm_breakdown = compare_ml_mm_contributions(atoms, hybrid_calc, verbose=True)

# Check specific issues:
print("\nKey diagnostics:")
print(f"  Zero force atoms: {diagnostics['zero_force_indices']}")
print(f"  Has NaN: {diagnostics.get('has_nan', False)}")
if 'problematic_atoms' in diagnostics:
    print(f"  Problematic atoms (zero computed, non-zero ref): {diagnostics['problematic_atoms']}")
"""

# Example with your actual force arrays:
if __name__ == "__main__":
    # Your computed forces
    computed_forces = np.array([
        [-0.77742296,  0.6827544 , -0.1022442 ],
        [ 0.6947665 , -0.94786   , -0.8985358 ],
        [ 0.        ,  0.        ,  0.        ],  # Zero!
        [ 0.        ,  0.        ,  0.        ],  # Zero!
        [-0.07586901, -0.05301681, -0.29063153],
        [ 0.        ,  0.        ,  0.        ],  # Zero!
        [ 0.09218006,  0.03094792,  0.4335372 ],
        [ 0.        ,  0.        ,  0.        ],  # Zero!
        [-0.15781285,  0.14848545, -0.6789729 ],
        [-0.27007565, -0.46399748,  1.2189962 ],
        [ 0.78182304, -0.6865988 , -0.01179719],
        [-0.08742553,  0.62157786, -1.3105353 ],
        [-0.09730127,  0.99524426, -2.090323  ],
        [-2.1488578 , -0.72964394,  2.9129624 ],
        [-0.68232507, -1.0814397 ,  0.3325614 ],
        [-1.2308909 , -0.74783444,  1.2137635 ],
        [ 1.8443935 ,  0.571167  ,  0.31756997],
        [ 1.8548793 ,  1.5693676 , -0.8140422 ],
        [ 0.19132665, -0.64021677, -0.2618367 ],
        [-0.42562184,  0.12837708, -0.28832293]
    ])
    
    # Your reference forces
    ref_forces = np.array([
        [-0.69319672,  0.66001612,  0.08380666],
        [ 0.23720794, -0.70627746, -1.28438697],
        [ 1.18310246,  0.52286832,  0.61898388],  # Non-zero!
        [ 1.43786435,  1.05678056, -0.24633793],  # Non-zero!
        [ 0.03767335, -0.39615869, -0.0782829 ],
        [-1.96260649, -0.22414673,  0.28378348],  # Non-zero!
        [ 0.12514537,  0.16268142,  0.44868068],
        [-0.37618739, -0.79095058, -0.04534604],  # Non-zero!
        [ 0.08578024,  0.0413022 , -0.52004576],
        [-0.07424935, -0.32612543,  0.73943801],
        [ 0.61073989, -0.64974096, -0.00687153],
        [ 0.54194488,  0.17154607, -1.1680214 ],
        [-0.77818711,  1.07111754, -1.32825616],
        [-2.61410043, -0.80831067,  2.17735379],
        [-0.23790625, -1.05380013, -0.2960317 ],
        [-1.04374506, -0.65092109,  1.04927396],
        [ 1.66540289,  0.54548015,  0.20401551],
        [ 1.98444545,  1.60756489, -0.76753092],
        [-0.01695283, -0.25740293, -0.11635117],
        [-0.11217518,  0.02447742,  0.25212651]
    ])
    
    print("Example force comparison:")
    print(f"Computed forces with zero: {np.sum(np.all(np.abs(computed_forces) < 1e-10, axis=1))} atoms")
    print(f"Reference forces with zero: {np.sum(np.all(np.abs(ref_forces) < 1e-10, axis=1))} atoms")
    
    # Find problematic atoms
    computed_zero = np.all(np.abs(computed_forces) < 1e-10, axis=1)
    ref_nonzero = np.any(np.abs(ref_forces) > 1e-10, axis=1)
    problematic = np.where(computed_zero & ref_nonzero)[0]
    
    print(f"\nProblematic atoms (zero computed, non-zero ref): {problematic}")
    print("\nThis suggests:")
    print("  1. Dimer forces may not be computed (see CRITICAL_ISSUES_SUMMARY.md)")
    print("  2. MM forces may not be computed for all atoms")
    print("  3. Atom index mapping may be incorrect")
    
    print("\nTo debug with your actual calculator:")
    print("  diagnostics = debug_hybrid_calculator_forces(atoms, hybrid_calc, ref_forces=ref_forces)")

