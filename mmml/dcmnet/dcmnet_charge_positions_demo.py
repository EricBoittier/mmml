#!/usr/bin/env python3
"""
Updated DCMNET MCTS Example - With Charge Positions

This script demonstrates the corrected approach where we select individual
charges from different DCMNET models, including their 3D positions.
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add the dcmnet directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcmnet_mcts import optimize_dcmnet_combination, DCMNETSelectionEnv

def create_example_data_with_positions():
    """Create example data with charge positions for demonstration."""
    
    # Example molecular data (CH4)
    molecular_data = {
        'atomic_numbers': jnp.array([6, 1, 1, 1, 1]),  # C, H, H, H, H
        'positions': jnp.array([[0.0, 0.0, 0.0],      # C at origin
                               [1.0, 1.0, 1.0],        # H1
                               [-1.0, -1.0, 1.0],      # H2
                               [-1.0, 1.0, -1.0],      # H3
                               [1.0, -1.0, -1.0]]),   # H4
        'dst_idx': jnp.array([0, 0, 0, 0]),            # All H atoms connected to C
        'src_idx': jnp.array([1, 2, 3, 4])             # H atom indices
    }
    
    # Example ESP target and VdW surface
    esp_target = jnp.random.normal(0, 1, (100,))
    vdw_surface = jnp.random.normal(0, 2, (100, 3))
    
    n_atoms = len(molecular_data['atomic_numbers'])
    
    # Example model charges - each model predicts different numbers of charges per atom
    model_charges = {
        0: jnp.array([[0.5], [0.1], [0.1], [0.1], [0.1]]),  # DCM1: 1 charge per atom
        1: jnp.array([[0.3, 0.2], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05]]),  # DCM2: 2 charges per atom
        2: jnp.array([[0.2, 0.15, 0.15], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04], [0.03, 0.03, 0.04]]),  # DCM3: 3 charges per atom
        3: jnp.array([[0.15, 0.1, 0.1, 0.15], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025]]),  # DCM4: 4 charges per atom
    }
    
    # Example model positions - each charge has a 3D position
    model_positions = {
        0: jnp.array([[[0.0, 0.0, 0.0]],                    # DCM1: 1 charge per atom at atom center
                     [[1.0, 1.0, 1.0]], 
                     [[-1.0, -1.0, 1.0]], 
                     [[-1.0, 1.0, -1.0]], 
                     [[1.0, -1.0, -1.0]]]),
        
        1: jnp.array([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],  # DCM2: 2 charges per atom, slightly offset
                     [[1.1, 1.0, 1.0], [0.9, 1.0, 1.0]], 
                     [[-0.9, -1.0, 1.0], [-1.1, -1.0, 1.0]], 
                     [[-0.9, 1.0, -1.0], [-1.1, 1.0, -1.0]], 
                     [[1.1, -1.0, -1.0], [0.9, -1.0, -1.0]]]),
        
        2: jnp.array([[[0.1, 0.1, 0.0], [-0.1, 0.1, 0.0], [0.0, -0.1, 0.0]],  # DCM3: 3 charges per atom
                     [[1.1, 1.1, 1.0], [0.9, 1.1, 1.0], [1.0, 0.9, 1.0]], 
                     [[-0.9, -0.9, 1.0], [-1.1, -0.9, 1.0], [-1.0, -1.1, 1.0]], 
                     [[-0.9, 1.1, -1.0], [-1.1, 1.1, -1.0], [-1.0, 0.9, -1.0]], 
                     [[1.1, -0.9, -1.0], [0.9, -0.9, -1.0], [1.0, -1.1, -1.0]]]),
        
        3: jnp.array([[[0.1, 0.1, 0.1], [-0.1, 0.1, 0.1], [0.1, -0.1, 0.1], [-0.1, -0.1, 0.1]],  # DCM4: 4 charges per atom
                     [[1.1, 1.1, 1.1], [0.9, 1.1, 1.1], [1.1, 0.9, 1.1], [0.9, 0.9, 1.1]], 
                     [[-0.9, -0.9, 1.1], [-1.1, -0.9, 1.1], [-0.9, -1.1, 1.1], [-1.1, -1.1, 1.1]], 
                     [[-0.9, 1.1, -0.9], [-1.1, 1.1, -0.9], [-0.9, 0.9, -0.9], [-1.1, 0.9, -0.9]], 
                     [[1.1, -0.9, -0.9], [0.9, -0.9, -0.9], [1.1, -1.1, -0.9], [0.9, -1.1, -0.9]]]),
    }
    
    return molecular_data, esp_target, vdw_surface, model_charges, model_positions

def demonstrate_charge_selection_with_positions():
    """Demonstrate the charge-level selection approach with positions."""
    
    print("=== DCMNET Charge-Level Selection with Positions Demo ===\n")
    
    # Create example data
    molecular_data, esp_target, vdw_surface, model_charges, model_positions = create_example_data_with_positions()
    
    print("1. Molecular System:")
    print(f"   - {len(molecular_data['atomic_numbers'])} atoms")
    print(f"   - Atomic numbers: {molecular_data['atomic_numbers']}")
    print(f"   - Atom positions:")
    for i, pos in enumerate(molecular_data['positions']):
        print(f"     Atom {i}: {pos}")
    
    print("\n2. Available Models and Charges:")
    total_charges = 0
    for model_id, charges in model_charges.items():
        n_charges = charges.shape[1]
        total_charges += n_charges
        print(f"   - DCM{model_id+1}: {n_charges} charges per atom")
        print(f"     Example charges for atom 0: {charges[0].tolist()}")
        print(f"     Example positions for atom 0:")
        for j, pos in enumerate(model_positions[model_id][0]):
            print(f"       Charge {j}: {pos}")
    
    print(f"\n   Total charges per atom across all models: {total_charges}")
    print(f"   Total possible charge combinations: {total_charges ** len(molecular_data['atomic_numbers'])}")
    
    print("\n3. Creating DCMNET Selection Environment...")
    env = DCMNETSelectionEnv(molecular_data, esp_target, vdw_surface, model_charges, model_positions)
    
    print(f"   - Environment state shape: {env.selected_charges.shape}")
    print(f"   - Charge mapping: {env.charge_mapping}")
    print(f"   - Legal actions: {len(env.legal_actions())} possible (atom_idx, charge_idx) pairs")
    
    print("\n4. Running MCTS Optimization...")
    best_selection, best_loss = optimize_dcmnet_combination(
        molecular_data=molecular_data,
        esp_target=esp_target,
        vdw_surface=vdw_surface,
        model_charges=model_charges,
        model_positions=model_positions,
        n_simulations=100,  # Small number for demo
        temperature=1.0
    )
    
    print(f"\n5. Results:")
    print(f"   - Best ESP loss: {best_loss:.6f}")
    print(f"   - Selection matrix shape: {best_selection.shape}")
    
    print(f"\n6. Selected Charges per Atom:")
    for atom_idx in range(len(molecular_data['atomic_numbers'])):
        selected_charges = np.where(best_selection[atom_idx])[0]
        print(f"   Atom {atom_idx} (atomic number {molecular_data['atomic_numbers'][atom_idx]}):")
        print(f"     Selected charge indices: {selected_charges.tolist()}")
        
        # Show which models these charges come from
        for charge_idx in selected_charges:
            model_id, charge_within_model = env.charge_mapping[charge_idx]
            charge_value = model_charges[model_id][atom_idx, charge_within_model]
            charge_position = model_positions[model_id][atom_idx, charge_within_model]
            print(f"       Charge {charge_idx} -> DCM{model_id+1}, charge {charge_within_model}")
            print(f"         Value: {charge_value:.3f}")
            print(f"         Position: [{charge_position[0]:.3f}, {charge_position[1]:.3f}, {charge_position[2]:.3f}]")
    
    print(f"\n7. Summary:")
    total_selected = np.sum(best_selection)
    print(f"   - Total charges selected: {total_selected}")
    print(f"   - Average charges per atom: {total_selected / len(molecular_data['atomic_numbers']):.1f}")
    
    # Show which models are being used
    used_models = set()
    for atom_idx in range(len(molecular_data['atomic_numbers'])):
        for charge_idx in np.where(best_selection[atom_idx])[0]:
            model_id, _ = env.charge_mapping[charge_idx]
            used_models.add(model_id)
    
    print(f"   - Models being used: DCM{[m+1 for m in sorted(used_models)]}")

def explain_distributed_charges():
    """Explain the concept of distributed charges with positions."""
    
    print("\n=== Understanding Distributed Charges with Positions ===\n")
    
    print("Key Concept:")
    print("- Each DCMNET model predicts not just charge VALUES, but also charge POSITIONS")
    print("- Charges are distributed in 3D space around each atom")
    print("- Different models predict different numbers of charges per atom")
    print("- Each charge has both a magnitude (value) and a 3D position")
    
    print("\nExample:")
    print("DCM1: 1 charge per atom at the atom center")
    print("DCM2: 2 charges per atom, slightly offset from center")
    print("DCM3: 3 charges per atom, forming a triangle around the atom")
    print("DCM4: 4 charges per atom, forming a tetrahedron around the atom")
    
    print("\nESP Calculation:")
    print("- ESP at each surface point = sum(q_i / r_i) for all selected charges")
    print("- q_i = charge value, r_i = distance from charge to surface point")
    print("- This is why charge positions matter - they affect the ESP calculation")
    
    print("\nMCTS Optimization:")
    print("- MCTS explores different combinations of individual charges")
    print("- Each charge is selected based on both its value AND position")
    print("- Goal: find the combination that best reproduces the target ESP")

def main():
    """Main demonstration function."""
    
    try:
        demonstrate_charge_selection_with_positions()
        explain_distributed_charges()
        
        print("\n=== Key Points ===")
        print("✓ Charges have both VALUES and POSITIONS")
        print("✓ ESP calculation uses: sum(q_i / r_i) where r_i depends on charge positions")
        print("✓ MCTS optimizes over individual charges, not entire models")
        print("✓ Each charge contributes to ESP based on its 3D position")
        print("✓ This allows for sophisticated charge distributions")
        
    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
